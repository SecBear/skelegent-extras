//! Synthesis operator — two-pass cross-decision structural analysis.
//!
//! [`SynthesisOperator`] wraps the two-pass synthesis logic as a dispatchable
//! [`Operator`]. Pass 1 performs a broad cross-decision scan using all sweep
//! verdicts; Pass 2 deep-dives each flagged item for precise classification.
//!
//! # Two-Pass Architecture
//!
//! - **Pass 1 (broad scan):** All sweep verdicts → array of flagged items.
//!   Uses [`PASS_1_SYSTEM_PROMPT`] with `{min_flag_confidence}` substituted.
//!   Items below [`SynthesisConfig::min_flag_confidence`] are discarded.
//!
//! - **Pass 2 (deep dive):** Each flagged item (up to
//!   [`SynthesisConfig::max_deep_dives`]) → [`StructuralChange`] or
//!   [`CandidateDecision`]. Uses [`PASS_2_SYSTEM_PROMPT`] with `{flag.summary}`,
//!   `{flag.decision_ids}`, and `{full_cards_and_deltas}` substituted.
//!
//! [`PASS_1_SYSTEM_PROMPT`]: crate::synthesis::PASS_1_SYSTEM_PROMPT
//! [`PASS_2_SYSTEM_PROMPT`]: crate::synthesis::PASS_2_SYSTEM_PROMPT
//! [`StructuralChange`]: crate::synthesis::StructuralChange
//! [`CandidateDecision`]: crate::synthesis::CandidateDecision
//! [`SynthesisConfig`]: crate::synthesis::SynthesisConfig

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};

use layer0::duration::DurationMs;
use layer0::operator::OperatorMetadata;
use layer0::content::Content;
use layer0::context::{Message, Role};
use layer0::{ExitReason, Operator, OperatorError, OperatorInput, OperatorOutput};
use neuron_orch_kit::ScopedState;
use neuron_turn::infer::InferRequest;
use neuron_turn::provider::{Provider, ProviderError};
use tracing::{debug, info};
use crate::synthesis::{
    CandidateDecision, StructuralChange, SynthesisConfig, SynthesisReport, PASS_1_SYSTEM_PROMPT,
    PASS_2_SYSTEM_PROMPT,
};
use crate::types::SweepVerdict;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Strip an optional `` ```json `` / `` ``` `` fence and return the inner JSON slice.
///
/// Many LLMs wrap JSON responses in a fenced code block. If no fence is found,
/// the input is returned unchanged.
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

/// Map a [`ProviderError`] to an [`OperatorError`].
///
/// Mapping rules match `CompareOperator`:
/// - `RateLimited` → `Retryable`
/// - `TransientError` → `Retryable` (passes the message through)
/// - `AuthFailed` → `NonRetryable`
/// - everything else → `Model`
fn map_provider_err(e: ProviderError) -> OperatorError {
    match e {
        ProviderError::RateLimited => {
            OperatorError::Retryable("rate limited by LLM provider".into())
        }
        ProviderError::TransientError { message, .. } => OperatorError::Retryable(message),
        ProviderError::AuthFailed(msg) => OperatorError::NonRetryable(format!("auth: {msg}")),
        other => OperatorError::Model(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Internal Pass 1 output type
// ---------------------------------------------------------------------------

/// A single cross-decision pattern or candidate decision flagged by Pass 1.
///
/// Both structural items and candidate-decision items arrive in the same Pass 1
/// JSON array. All fields not present in a given item deserialize to their
/// `Default` — callers should inspect `item_type` vs `candidate_title` to
/// distinguish the two kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlaggedItem {
    /// Pattern type tag for structural items (e.g. `"convergence"`).
    ///
    /// `None` for candidate-decision items.
    #[serde(rename = "type", default)]
    item_type: Option<String>,

    /// Decision IDs involved in this pattern.
    #[serde(default)]
    decision_ids: Vec<String>,

    /// One-line description of the flagged pattern.
    ///
    /// Empty string when absent (candidate items may omit this field).
    #[serde(default)]
    summary: String,

    /// Confidence score 0.0–1.0.
    ///
    /// Items whose confidence falls below [`SynthesisConfig::min_flag_confidence`]
    /// are discarded before Pass 2.
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

// ---------------------------------------------------------------------------
// Internal Pass 2 output type
// ---------------------------------------------------------------------------

/// Result returned by a single Pass 2 deep-dive LLM call.
///
/// The LLM outputs either a [`StructuralChange`] or a [`CandidateDecision`];
/// serde's untagged enum tries `Structural` first (distinguished by its
/// `change_type` field) then falls back to `Candidate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum Pass2Result {
    /// A detected structural change between two or more decisions.
    Structural(StructuralChange),
    /// A candidate decision surfaced by cross-decision evidence.
    Candidate(CandidateDecision),
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Typed input for [`SynthesisOperator`].
///
/// Serialized as JSON and placed in [`OperatorInput::message`] for dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisInput {
    /// Verdicts from the sweep cycle to analyze for cross-decision patterns.
    pub verdicts: Vec<SweepVerdict>,
}

/// Operator that performs two-pass cross-decision structural synthesis.
///
/// **Pass 1** sends all sweep verdicts to the LLM and asks it to flag any
/// cross-decision patterns whose confidence exceeds
/// [`SynthesisConfig::min_flag_confidence`].
///
/// **Pass 2** performs a deep dive on each flagged item (up to
/// [`SynthesisConfig::max_deep_dives`]), reading the relevant decision cards
/// from the scoped state store and sending them with the pattern description
/// for precise classification.
///
/// Own-scope writes (`synthesis:latest`) go directly through [`ScopedState`].
/// No cross-scope [`Effect::WriteMemory`] effects are emitted.
///
/// [`Effect::WriteMemory`]: layer0::Effect::WriteMemory
pub struct SynthesisOperator<P: Provider> {
    llm: P,
    state: Arc<dyn ScopedState>,
    /// Configuration for both synthesis passes.
    pub config: SynthesisConfig,
}

impl<P: Provider> SynthesisOperator<P> {
    /// Create a new synthesis operator.
    ///
    /// - `llm` — LLM provider used for both passes.
    /// - `state` — Scoped state for reading decision cards and writing the
    ///   assembled [`SynthesisReport`] to `synthesis:latest`.
    /// - `config` — Synthesis configuration (confidence threshold, deep-dive
    ///   cap, model name, etc.).
    pub fn new(llm: P, state: Arc<dyn ScopedState>, config: SynthesisConfig) -> Self {
        Self { llm, state, config }
    }
}

#[async_trait]
impl<P: Provider + 'static> Operator for SynthesisOperator<P> {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        let start = Instant::now();
        let mut total_tokens_in: u64 = 0;
        let mut total_tokens_out: u64 = 0;
        let mut total_cost = Decimal::ZERO;

        // --- Parse typed input ---
        let msg_text = input.message.as_text().ok_or_else(|| {
            OperatorError::NonRetryable(
                "SynthesisOperator: input.message must be text containing SynthesisInput JSON"
                    .into(),
            )
        })?;
        let synthesis_in: SynthesisInput = serde_json::from_str(msg_text).map_err(|e| {
            OperatorError::NonRetryable(format!(
                "SynthesisOperator: failed to parse SynthesisInput: {e}"
            ))
        })?;

        let verdicts_json = serde_json::to_string(&synthesis_in.verdicts)
            .unwrap_or_else(|_| "[]".to_string());

        // --- Pass 1: broad cross-decision scan ---
        let pass1_system = PASS_1_SYSTEM_PROMPT.replace(
            "{min_flag_confidence}",
            &self.config.min_flag_confidence.to_string(),
        );
        let context = [Message::new(Role::User, Content::text(&verdicts_json))];

        let pass1_request = InferRequest {
            model: Some(self.config.model.clone()),
            messages: context.to_vec(),
            tools: vec![],
            max_tokens: None,
            temperature: None,
            system: Some(pass1_system),
            extra: serde_json::Value::Null,
        };
        let pass1_response = self.llm.infer(pass1_request).await.map_err(map_provider_err)?;

        let mut total_cost_usd: f64 = pass1_response.cost.and_then(|d| d.to_f64()).unwrap_or(0.0);
        total_tokens_in += pass1_response.usage.input_tokens;
        total_tokens_out += pass1_response.usage.output_tokens;
        total_cost += pass1_response.cost.unwrap_or(Decimal::ZERO);

        info!(
            pass = 1,
            tokens_in = pass1_response.usage.input_tokens,
            tokens_out = pass1_response.usage.output_tokens,
            cost = %pass1_response.cost.unwrap_or(Decimal::ZERO),
            duration_ms = start.elapsed().as_millis() as u64,
            "synthesis: pass 1 complete"
        );

        let pass1_raw = pass1_response.content.as_text().unwrap_or("");

        let all_flags: Vec<FlaggedItem> = {
            let json_str = extract_json_block(pass1_raw);
            serde_json::from_str(json_str).map_err(|e| {
                OperatorError::Model(format!(
                    "SynthesisOperator: failed to parse Pass 1 flags: {e}\nRaw: {pass1_raw}"
                ))
            })?
        };

        // Filter by confidence threshold before committing to deep dives.
        let flags: Vec<FlaggedItem> = all_flags
            .into_iter()
            .filter(|f| f.confidence >= self.config.min_flag_confidence)
            .collect();

        // --- Pass 2: deep dive on each flagged item ---
        let mut structural_changes: Vec<StructuralChange> = Vec::new();
        let mut candidates: Vec<CandidateDecision> = Vec::new();

        for flag in flags.iter().take(self.config.max_deep_dives) {
            // Read decision cards from own-scope state for all involved decisions.
            let mut cards_and_deltas = String::new();
            for decision_id in &flag.decision_ids {
                match self.state.read(&format!("card:{decision_id}")).await {
                    Ok(Some(val)) => {
                        let text = val
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| val.to_string());
                        cards_and_deltas
                            .push_str(&format!("=== {decision_id} ===\n{text}\n\n"));
                    }
                    Ok(None) => {
                        // Card not in state — log warning in narrative, continue.
                        cards_and_deltas.push_str(&format!(
                            "=== {decision_id} ===\n[card not available in state]\n\n"
                        ));
                    }
                    Err(e) => {
                        // State read failure — log warning, continue with available data.
                        cards_and_deltas.push_str(&format!(
                            "=== {decision_id} ===\n[state read error: {e}]\n\n"
                        ));
                    }
                }
            }

            let decision_ids_str = flag.decision_ids.join(", ");
            let pass2_system = PASS_2_SYSTEM_PROMPT
                .replace("{flag.summary}", &flag.summary)
                .replace("{flag.decision_ids}", &decision_ids_str)
                .replace("{full_cards_and_deltas}", &cards_and_deltas);

            let context = [Message::new(Role::User, Content::text("Produce the deep analysis."))];

            let pass2_request = InferRequest {
                model: Some(self.config.model.clone()),
                messages: context.to_vec(),
                tools: vec![],
                max_tokens: None,
                temperature: None,
                system: Some(pass2_system),
                extra: serde_json::Value::Null,
            };

            let pass2_response =
                self.llm.infer(pass2_request).await.map_err(map_provider_err)?;
            total_cost_usd += pass2_response.cost.and_then(|d| d.to_f64()).unwrap_or(0.0);
            total_tokens_in += pass2_response.usage.input_tokens;
            total_tokens_out += pass2_response.usage.output_tokens;
            total_cost += pass2_response.cost.unwrap_or(Decimal::ZERO);

            debug!(
                pass = 2,
                flag_summary = %flag.summary,
                tokens_in = pass2_response.usage.input_tokens,
                tokens_out = pass2_response.usage.output_tokens,
                "synthesis: pass 2 deep dive"
            );

            let pass2_raw = pass2_response.content.as_text().unwrap_or("");

            let json_str = extract_json_block(pass2_raw);
            match serde_json::from_str::<Pass2Result>(json_str) {
                Ok(Pass2Result::Structural(sc)) => structural_changes.push(sc),
                Ok(Pass2Result::Candidate(cd)) => candidates.push(cd),
                Err(e) => {
                    return Err(OperatorError::Model(format!(
                        "SynthesisOperator: failed to parse Pass 2 result: {e}\nRaw: {pass2_raw}"
                    )));
                }
            }
        }

        // --- Assemble report ---
        let n_structural = structural_changes.len();
        let n_candidates = candidates.len();
        let report = SynthesisReport {
            structural_changes,
            candidates,
            relationship_updates: vec![],
            health_summary: format!(
                "Synthesis complete: {n_structural} structural change(s), \
                 {n_candidates} candidate decision(s)."
            ),
            cost_usd: total_cost_usd,
            cycle_id: chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string(),
            completed_at: chrono::Utc::now().to_rfc3339(),
        };

        // Write to own scope — own-scope writes via ScopedState are within the effects boundary.
        if let Ok(report_value) = serde_json::to_value(&report) {
            // Non-fatal if this write fails; the caller still receives the report.
            let _ = self.state.write("synthesis:latest", report_value).await;
        }

        let report_json = serde_json::to_string(&report).unwrap_or_else(|_| "{}".to_string());
        let duration = DurationMs::from(start.elapsed());
        let mut metadata = OperatorMetadata::default();
        metadata.tokens_in = total_tokens_in;
        metadata.tokens_out = total_tokens_out;
        metadata.cost = total_cost;
        metadata.turns_used = 1 + flags.len().min(self.config.max_deep_dives) as u32;
        metadata.duration = duration;

        info!(
            structural = n_structural,
            candidates = n_candidates,
            total_tokens_in,
            total_tokens_out,
            total_cost = %total_cost,
            duration_ms = duration.as_millis(),
            "synthesis: complete"
        );

        let mut output = OperatorOutput::new(Content::text(report_json), ExitReason::Complete);
        output.metadata = metadata;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::StructuralChangeType;
    use crate::types::{EvidenceItem, EvidenceStance, ProcessorTier, VerdictStatus};
    use layer0::operator::TriggerType;
    use layer0::{SearchResult, StateError};
    use neuron_turn::infer::InferResponse;
    use neuron_turn::test_utils::FunctionProvider;
    use neuron_turn::types::{StopReason, TokenUsage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -----------------------------------------------------------------------
    // Mock state
    // -----------------------------------------------------------------------

    /// Minimal no-op [`ScopedState`] — reads always return `None`, writes succeed.
    struct MockState;

    #[async_trait]
    impl ScopedState for MockState {
        async fn read(&self, _key: &str) -> Result<Option<serde_json::Value>, StateError> {
            Ok(None)
        }

        async fn write(
            &self,
            _key: &str,
            _value: serde_json::Value,
        ) -> Result<(), StateError> {
            Ok(())
        }

        async fn delete(&self, _key: &str) -> Result<(), StateError> {
            Ok(())
        }

        async fn list(&self, _prefix: &str) -> Result<Vec<String>, StateError> {
            Ok(vec![])
        }

        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SearchResult>, StateError> {
            Ok(vec![])
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build an `InferResponse` with text content and a small cost.
    fn text_response(text: &str) -> InferResponse {
        InferResponse {
            content: Content::text(text),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
            cost: Some(rust_decimal::Decimal::new(1, 2)),
            truncated: None,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

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
                title: String::new(),
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

    fn canned_structural_change() -> StructuralChange {
        StructuralChange {
            change_type: StructuralChangeType::Convergence,
            decision_ids: vec!["D1".to_string(), "D2".to_string()],
            summary: "D1 and D2 are converging on the same architecture pattern.".to_string(),
            evidence: "Both decisions reference overlapping sources.".to_string(),
            observation_count: 1,
            confidence: 0.85,
            source_urls: vec![],    
        }
    }

    /// Serialize a [`SynthesisInput`] containing `n` dummy verdicts.
    fn synthesis_input_json(n: usize) -> String {
        let verdicts: Vec<SweepVerdict> =
            (0..n).map(|i| dummy_verdict(&format!("D{i}"))).collect();
        serde_json::to_string(&SynthesisInput { verdicts }).expect("serializable")
    }

    /// Build a Pass 1 JSON array with `n` flags, all above the default threshold.
    fn high_confidence_flags_json(n: usize) -> String {
        let flags: Vec<serde_json::Value> = (0..n)
            .map(|i| {
                serde_json::json!({
                    "type": "convergence",
                    "decision_ids": [format!("D{i}"), format!("D{}", i + 1)],
                    "summary": format!("Pattern {i}"),
                    "confidence": 0.9
                })
            })
            .collect();
        serde_json::to_string(&flags).expect("serializable")
    }

    /// Build a Pass 1 JSON array with `n` flags, all *below* the default threshold.
    fn low_confidence_flags_json(n: usize) -> String {
        let flags: Vec<serde_json::Value> = (0..n)
            .map(|i| {
                serde_json::json!({
                    "type": "divergence",
                    "decision_ids": [format!("D{i}")],
                    "summary": format!("Weak pattern {i}"),
                    "confidence": 0.1
                })
            })
            .collect();
        serde_json::to_string(&flags).expect("serializable")
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// Happy path: one high-confidence flag in Pass 1, one StructuralChange from Pass 2.
    /// The assembled report must deserialize cleanly and the call count must be 2.
    #[tokio::test]
    async fn synthesis_operator_returns_report_json() {
        let pass1_json = high_confidence_flags_json(1);
        let pass2_json =
            serde_json::to_string(&canned_structural_change()).expect("serializable");
        let call_count = Arc::new(AtomicUsize::new(0));

        let cc = call_count.clone();
        let p1 = pass1_json.clone();
        let p2 = pass2_json.clone();
        let provider = FunctionProvider::new(move |_req| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            let json = if n == 0 { p1.clone() } else { p2.clone() };
            Ok(text_response(&json))
        });
        let op = SynthesisOperator::new(provider, Arc::new(MockState), SynthesisConfig::default());

        let input = OperatorInput::new(Content::text(synthesis_input_json(2)), TriggerType::Task);
        let output = op.execute(input).await.expect("execute should succeed");

        let text = output.message.as_text().expect("output should be text");
        let report: SynthesisReport =
            serde_json::from_str(text).expect("output should be valid SynthesisReport JSON");

        // 1 Pass 1 call + 1 Pass 2 call = 2 total.
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "expected 2 LLM calls");
        assert_eq!(report.structural_changes.len(), 1);
        assert_eq!(report.structural_changes[0].change_type, StructuralChangeType::Convergence);
    }

    /// When Pass 1 flags all have confidence below the threshold, no Pass 2
    /// calls must be made and the report must be empty.
    #[tokio::test]
    async fn synthesis_operator_skips_low_confidence_flags() {
        // All 3 flags are below the default threshold of 0.6.
        let pass1_json = low_confidence_flags_json(3);
        let call_count = Arc::new(AtomicUsize::new(0));

        let cc = call_count.clone();
        let p1 = pass1_json.clone();
        let provider = FunctionProvider::new(move |_req| {
            cc.fetch_add(1, Ordering::SeqCst);
            Ok(text_response(&p1))
        });
        let op = SynthesisOperator::new(provider, Arc::new(MockState), SynthesisConfig::default());

        let input = OperatorInput::new(Content::text(synthesis_input_json(2)), TriggerType::Task);
        let output = op.execute(input).await.expect("execute should succeed");

        let text = output.message.as_text().expect("output should be text");
        let report: SynthesisReport =
            serde_json::from_str(text).expect("output should be valid SynthesisReport JSON");

        // Only Pass 1 was called.
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "Pass 2 must not be called when all flags are below threshold"
        );
        assert!(report.structural_changes.is_empty());
        assert!(report.candidates.is_empty());
    }

    /// When Pass 1 returns more flags than `max_deep_dives`, only
    /// `max_deep_dives` Pass 2 calls must be made (total = 1 + max_deep_dives).
    #[tokio::test]
    async fn synthesis_operator_respects_max_deep_dives() {
        // 20 high-confidence flags, all above threshold.
        let pass1_json = high_confidence_flags_json(20);
        let pass2_json =
            serde_json::to_string(&canned_structural_change()).expect("serializable");
        let call_count = Arc::new(AtomicUsize::new(0));

        // Default config: max_deep_dives = 10.
        let config = SynthesisConfig::default();
        let max_dives = config.max_deep_dives;

        let cc = call_count.clone();
        let p1 = pass1_json.clone();
        let p2 = pass2_json.clone();
        let provider = FunctionProvider::new(move |_req| {
            let n = cc.fetch_add(1, Ordering::SeqCst);
            let json = if n == 0 { p1.clone() } else { p2.clone() };
            Ok(text_response(&json))
        });
        let op = SynthesisOperator::new(provider, Arc::new(MockState), config);

        let input = OperatorInput::new(Content::text(synthesis_input_json(2)), TriggerType::Task);
        let output = op.execute(input).await.expect("execute should succeed");

        let text = output.message.as_text().expect("output should be text");
        let report: SynthesisReport =
            serde_json::from_str(text).expect("output should be valid SynthesisReport JSON");

        let expected_calls = 1 + max_dives;
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            expected_calls,
            "expected exactly {expected_calls} LLM calls (1 pass1 + {max_dives} pass2)"
        );
        assert_eq!(
            report.structural_changes.len(),
            max_dives,
            "report should contain exactly max_deep_dives structural changes"
        );
    }

}
