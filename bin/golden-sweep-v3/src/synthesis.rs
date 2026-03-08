//! Cross-decision synthesis — implemented with context-engine primitives.
//!
//! Re-exports structural types from `neuron-op-sweep-v2`. Implements
//! [`synthesize`] as a free function using [`Context`] directly instead of
//! the v2 `SynthesisOperator` pattern.
//!
//! # Two-Pass Architecture
//!
//! - **Pass 1** — broad cross-decision scan: all verdicts → flagged patterns.
//!   Uses [`PASS_1_SYSTEM_PROMPT`] with `{min_flag_confidence}` substituted.
//!   Items below [`SynthesisConfig::min_flag_confidence`] are discarded.
//!
//! - **Pass 2** — deep dive per flagged item → [`StructuralChange`] or
//!   [`CandidateDecision`]. Uses [`PASS_2_SYSTEM_PROMPT`] with
//!   `{flag.summary}`, `{flag.decision_ids}`, and `{full_cards_and_deltas}`
//!   substituted from the verdicts themselves.
//!
//! Unlike the v2 `SynthesisOperator`, this function does not access a state
//! store. `{full_cards_and_deltas}` is assembled from the
//! [`SweepVerdict`] payload (narrative + evidence) already available.
//!
//! Returns `Err` when Pass 1 fails (provider or parse error). Pass 2
//! per-item failures are non-fatal. Callers that want non-fatal synthesis
//! should call `.ok()` on the result.

use layer0::content::Content;
use layer0::context::{Message, Role};
use neuron_context_engine::{CompileConfig, Context, EngineError};
use neuron_turn::provider::Provider;
use serde::Deserialize;
use tracing::{debug, info};

use crate::types::SweepVerdict;

// Re-export synthesis types so callers only need to import from this crate.
pub use neuron_op_sweep_v2::synthesis::{
    advance_candidate, merge_candidate_observation, should_stage_candidate, CandidateDecision,
    CandidateStage, RelationshipAction, RelationshipKind, RelationshipUpdate, StructuralChange,
    StructuralChangeType, SynthesisConfig, SynthesisReport, PASS_1_SYSTEM_PROMPT,
    PASS_2_SYSTEM_PROMPT,
};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A flagged cross-decision pattern returned by the Pass 1 broad scan.
///
/// Both structural items and candidate-decision items arrive in the same
/// Pass 1 JSON array. All fields absent from a given item deserialise to
/// their `Default`. Callers inspect `confidence` and `decision_ids` to
/// route items into Pass 2.
#[allow(dead_code)] // fields populated by serde; not all are used in logic
#[derive(Debug, Deserialize)]
struct FlaggedItem {
    /// Pattern type tag for structural items (e.g. `"convergence"`).
    #[serde(rename = "type", default)]
    item_type: Option<String>,
    /// Decision IDs involved in this pattern.
    #[serde(default)]
    decision_ids: Vec<String>,
    /// One-line description of the flagged pattern.
    #[serde(default)]
    summary: String,
    /// Confidence score 0.0–1.0.
    #[serde(default)]
    confidence: f64,
    /// Proposed candidate decision title (candidates only).
    #[serde(default)]
    candidate_title: Option<String>,
    /// Related existing decision IDs (candidates only).
    #[serde(default)]
    related_decisions: Option<Vec<String>>,
    /// Evidence summary for the candidate (candidates only).
    #[serde(default)]
    evidence_summary: Option<String>,
}

/// Result returned by a single Pass 2 deep-dive LLM call.
///
/// The LLM outputs either a [`StructuralChange`] or a [`CandidateDecision`].
/// Serde's untagged enum tries `Structural` first (via its `change_type`
/// field) then falls back to `Candidate`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Pass2Result {
    /// A detected structural change between two or more decisions.
    Structural(StructuralChange),
    /// A candidate decision surfaced by cross-decision evidence.
    Candidate(CandidateDecision),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip optional `` ```json `` / `` ``` `` fence and return inner JSON.
fn extract_json_block(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(inner) = trimmed.strip_prefix("```json") {
        inner.trim_start().trim_end_matches("```").trim()
    } else if let Some(inner) = trimmed.strip_prefix("```") {
        inner.trim_end_matches("```").trim()
    } else {
        trimmed
    }
}

/// Build `{full_cards_and_deltas}` for Pass 2 from available verdicts.
///
/// Since `synthesize` has no state store, it assembles context from each
/// relevant [`SweepVerdict`]'s `narrative` and `evidence` fields.
fn build_cards_and_deltas(decision_ids: &[String], verdicts: &[SweepVerdict]) -> String {
    let mut buf = String::new();
    for id in decision_ids {
        match verdicts.iter().find(|v| v.decision_id == *id) {
            Some(v) => {
                let evidence_lines: Vec<String> = v
                    .evidence
                    .iter()
                    .map(|e| {
                        let stance = serde_json::to_value(&e.stance)
                            .ok()
                            .and_then(|val| val.as_str().map(str::to_string))
                            .unwrap_or_default();
                        format!("  - [{stance}] {} — {}", e.title, e.summary)
                    })
                    .collect();

                let status = serde_json::to_value(&v.status)
                    .ok()
                    .and_then(|val| val.as_str().map(str::to_string))
                    .unwrap_or_else(|| format!("{:?}", v.status));

                buf.push_str(&format!(
                    "=== {id} ===\n\
                     Verdict: {status} (confidence: {:.2})\n\
                     Narrative: {}\n\
                     Evidence:\n{}\n\n",
                    v.confidence,
                    v.narrative,
                    if evidence_lines.is_empty() {
                        "  (none)".to_string()
                    } else {
                        evidence_lines.join("\n")
                    }
                ));
            }
            None => {
                buf.push_str(&format!("=== {id} ===\n[no verdict available]\n\n"));
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// synthesize
// ---------------------------------------------------------------------------

/// Run two-pass cross-decision synthesis over a completed sweep cycle.
///
/// Uses [`Context`](neuron_context_engine::Context) and
/// `compile()→infer()` directly — no `SynthesisOperator`, no
/// `dispatch_typed`, no `Orchestrator`.
///
/// # Two passes
///
/// **Pass 1** sends all verdicts (as JSON) to the LLM with
/// [`PASS_1_SYSTEM_PROMPT`] as the system prompt and receives a JSON array
/// of flagged cross-decision patterns. Items below
/// [`SynthesisConfig::min_flag_confidence`] are discarded.
///
/// **Pass 2** runs for each flagged item (up to
/// [`SynthesisConfig::max_deep_dives`]). The LLM classifies each item as a
/// [`StructuralChange`] or a [`CandidateDecision`]. Per-item Pass 2
/// failures are non-fatal: the item is skipped.
///
/// # Errors
///
/// Returns [`EngineError`] when the provider call fails at Pass 1, or when
/// the Pass 1 response cannot be parsed as a JSON array of flagged items.
pub async fn synthesize<P: Provider>(
    provider: &P,
    verdicts: &[SweepVerdict],
    config: &SynthesisConfig,
) -> Result<SynthesisReport, EngineError> {
    let mut total_cost_usd: f64 = 0.0;

    info!(count = verdicts.len(), "synthesis: starting pass 1");

    // ── Pass 1: broad scan ────────────────────────────────────────────────────

    let pass1_system = PASS_1_SYSTEM_PROMPT
        .replace("{min_flag_confidence}", &config.min_flag_confidence.to_string());

    let verdicts_json = serde_json::to_string(verdicts).unwrap_or_else(|_| "[]".to_string());

    let mut ctx1 = Context::new();
    ctx1.inject_message(Message::new(Role::User, Content::text(&verdicts_json)))
        .await?;

    let pass1_config = CompileConfig {
        system: Some(pass1_system),
        model: Some(config.model.clone()),
        max_tokens: Some(4096),
        ..Default::default()
    };

    let pass1_result = ctx1.compile(&pass1_config).infer(provider).await?;

    total_cost_usd += pass1_result
        .response
        .cost
        .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);

    let pass1_raw = pass1_result.text().unwrap_or_default();

    let all_flags: Vec<FlaggedItem> =
        serde_json::from_str(extract_json_block(pass1_raw)).map_err(|e| {
            EngineError::Custom(
                format!("synthesis: pass 1 parse failed: {e}\nraw:\n{pass1_raw}").into(),
            )
        })?;

    let flags: Vec<FlaggedItem> = all_flags
        .into_iter()
        .filter(|f| f.confidence >= config.min_flag_confidence)
        .take(config.max_deep_dives)
        .collect();

    info!(flagged = flags.len(), "synthesis: pass 1 complete");

    // ── Pass 2: deep dive per flagged item ───────────────────────────────────

    let mut structural_changes: Vec<StructuralChange> = Vec::new();
    let mut candidates: Vec<CandidateDecision> = Vec::new();

    for item in &flags {
        let cards_and_deltas = build_cards_and_deltas(&item.decision_ids, verdicts);
        let decision_ids_str = item.decision_ids.join(", ");

        let pass2_system = PASS_2_SYSTEM_PROMPT
            .replace("{flag.summary}", &item.summary)
            .replace("{flag.decision_ids}", &decision_ids_str)
            .replace("{full_cards_and_deltas}", &cards_and_deltas);

        let mut ctx2 = Context::new();
        ctx2.inject_message(Message::new(
            Role::User,
            Content::text("Produce the deep analysis."),
        ))
        .await?;

        let pass2_config = CompileConfig {
            system: Some(pass2_system),
            model: Some(config.model.clone()),
            max_tokens: Some(2048),
            ..Default::default()
        };

        let pass2_result = match ctx2.compile(&pass2_config).infer(provider).await {
            Ok(r) => r,
            Err(e) => {
                debug!(
                    error = %e,
                    flag_summary = %item.summary,
                    "synthesis: pass 2 infer failed, skipping item"
                );
                continue;
            }
        };

        total_cost_usd += pass2_result
            .response
            .cost
            .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);

        let pass2_raw = pass2_result.text().unwrap_or_default();

        match serde_json::from_str::<Pass2Result>(extract_json_block(pass2_raw)) {
            Ok(Pass2Result::Structural(sc)) => structural_changes.push(sc),
            Ok(Pass2Result::Candidate(cd)) => candidates.push(cd),
            Err(e) => {
                debug!(
                    error = %e,
                    flag_summary = %item.summary,
                    "synthesis: pass 2 parse failed, skipping item"
                );
            }
        }
    }

    let n_structural = structural_changes.len();
    let n_candidates = candidates.len();

    info!(
        structural_changes = n_structural,
        candidates = n_candidates,
        "synthesis: complete"
    );

    Ok(SynthesisReport {
        structural_changes,
        candidates,
        relationship_updates: vec![],
        health_summary: format!(
            "Synthesis complete: {n_structural} structural change(s), \
             {n_candidates} candidate decision(s)."
        ),
        cost_usd: total_cost_usd,
        cycle_id: chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string(),
        completed_at: chrono::Utc::now().to_rfc3339(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use neuron_turn::infer::InferResponse;
    use neuron_turn::test_utils::TestProvider;
    use neuron_turn::types::{StopReason, TokenUsage};

    use crate::types::{EvidenceItem, EvidenceStance, ProcessorTier, VerdictStatus};

    fn dummy_verdict(id: &str) -> SweepVerdict {
        SweepVerdict {
            decision_id: id.to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.9,
            num_supporting: 3,
            num_contradicting: 0,
            cost_usd: 0.05,
            processor: ProcessorTier::Base,
            duration_secs: 1.5,
            swept_at: "2026-03-06T00:00:00Z".to_string(),
            evidence: vec![EvidenceItem {
                source_url: "https://example.com".into(),
                title: "Example".into(),
                summary: "Supports decision".into(),
                stance: EvidenceStance::Supporting,
            }],
            narrative: "Decision confirmed.".to_string(),
            proposed_diff: None,
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),
        }
    }

    fn text_response(text: &str) -> InferResponse {
        InferResponse {
            content: Content::text(text),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
            cost: None,
            truncated: None,
        }
    }

    #[test]
    fn extract_json_block_strips_fence() {
        let fenced = "```json\n[]\n```";
        assert_eq!(extract_json_block(fenced), "[]");
    }

    #[test]
    fn extract_json_block_passthrough() {
        let raw = r#"{"key":"value"}"#;
        assert_eq!(extract_json_block(raw), raw);
    }

    #[test]
    fn build_cards_includes_narrative() {
        let verdicts = vec![dummy_verdict("D1")];
        let result = build_cards_and_deltas(&["D1".to_string()], &verdicts);
        assert!(result.contains("=== D1 ==="));
        assert!(result.contains("Decision confirmed."));
    }

    #[test]
    fn build_cards_missing_id() {
        let result = build_cards_and_deltas(&["D99".to_string()], &[]);
        assert!(result.contains("[no verdict available]"));
    }

    #[tokio::test]
    async fn synthesize_returns_empty_report_when_no_flags() {
        // Pass 1 returns empty array → no Pass 2 calls → empty report
        let provider = TestProvider::with_responses(vec![text_response("[]")]);
        let verdicts: Vec<SweepVerdict> = (1..=3).map(|i| dummy_verdict(&format!("D{i}"))).collect();
        let config = SynthesisConfig::default();

        let report = synthesize(&provider, &verdicts, &config).await.unwrap();
        assert!(report.structural_changes.is_empty());
        assert!(report.candidates.is_empty());
    }

    #[tokio::test]
    async fn synthesize_returns_err_on_pass1_parse_failure() {
        // Pass 1 returns non-JSON → parse error → Err
        let provider = TestProvider::with_responses(vec![text_response("not json at all")]);
        let verdicts: Vec<SweepVerdict> = (1..=3).map(|i| dummy_verdict(&format!("D{i}"))).collect();
        let config = SynthesisConfig::default();

        let result = synthesize(&provider, &verdicts, &config).await;
        assert!(result.is_err());
    }
}
