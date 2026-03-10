//! Sweep cycle orchestration — context-engine implementation.
//!
//! [`sweep_cycle`] sequences a batch of [`SweepDecision`]s through the
//! comparison pipeline and triggers a cross-decision synthesis pass when
//! enough verdicts accumulate. It uses [`compare_decision`] and
//! [`synthesize`] directly as functions — no `Operator` trait, no
//! `Orchestrator`, no `dispatch_typed`, no `CompositionTrace`.
//!
//! # Pipeline (per decision)
//!
//! 1. **Dedup check** — read-only; skip if swept too recently (< 20 h).
//! 2. **Budget check** — skip when accumulated cost ≥ cap.
//! 3. **Processor selection** — tier from remaining-budget ratio and prior verdict.
//! 4. **Empty-results short-circuit** — 0 results → `Confirmed(0.3)`, no compare.
//! 5. **Read decision card** — `card:{id}` from scoped state.
//! 6. **Read prior findings** — `delta:{id}:*` keys from scoped state.
//! 7. **Compare** — call [`compare_decision`] directly.
//! 8. **Write sweep timestamp** — `meta:{id}:last_sweep` (non-fatal).
//! 9. **Collect** — push verdict, accumulate cost.
//!
//! # Synthesis
//!
//! When ≥ 3 verdicts are produced, [`synthesize`] is called. Synthesis failure
//! is non-fatal: `CycleReport::synthesis` is left as `None`.
//!
//! # Error handling
//!
//! - State errors during dedup → skip the decision with reason.
//! - Provider errors during compare → skip the decision with reason.
//! - Synthesis failure → `report.synthesis = None`.
//! - State read/write errors for card, prior findings, or timestamp → logged
//!   inline, processing continues with whatever data is available.

use neuron_orch_kit::ScopedState;
use neuron_turn::provider::Provider;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::compare::{compare_decision, CompareConfig};
use crate::synthesis::{synthesize, SynthesisConfig, SynthesisReport};
use crate::types::{
    select_processor, SweepDecision, SweepMeta, SweepVerdict, VerdictStatus,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum hours that must elapse between two sweeps of the same decision.
const MIN_SWEEP_INTERVAL_HOURS: i64 = 20;

/// Minimum number of verdicts required to trigger the synthesis pass.
const SYNTHESIS_THRESHOLD: usize = 3;

// ---------------------------------------------------------------------------
// CycleReport
// ---------------------------------------------------------------------------

/// Output of a complete sweep cycle produced by [`sweep_cycle`].
///
/// Verdicts are in decision order (skipped decisions appear in `skipped`,
/// not in `verdicts`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleReport {
    /// Verdicts from completed comparisons.
    pub verdicts: Vec<SweepVerdict>,

    /// Decisions that were skipped and the reason for each skip.
    ///
    /// Each entry is `(decision_id, reason)`. Reasons include:
    /// "swept too recently", budget exhaustion, compare failures, and
    /// state-read errors during dedup.
    pub skipped: Vec<(String, String)>,

    /// Cross-decision synthesis report, present when ≥ 3 verdicts were
    /// produced and synthesis succeeded.
    pub synthesis: Option<SynthesisReport>,

    /// Accumulated USD cost for this cycle (comparisons + synthesis).
    pub total_cost: f64,
}

// ---------------------------------------------------------------------------
// sweep_cycle
// ---------------------------------------------------------------------------

/// Sequence the sweep cycle pipeline for a batch of decisions.
///
/// Calls [`compare_decision`] for each eligible decision, then optionally
/// calls [`synthesize`] when ≥ [`SYNTHESIS_THRESHOLD`] verdicts are produced.
///
/// # Parameters
///
/// - `provider` — LLM provider used for both comparison and synthesis.
/// - `decisions` — batch of decisions to process.
/// - `state` — scoped state for dedup reads, decision card reads, and sweep
///   timestamp writes.
/// - `budget_cap` — total USD cap for this cycle; decisions are skipped once
///   the accumulated cost meets or exceeds the cap.
/// - `compare_config` — configuration forwarded to [`compare_decision`].
/// - `synthesis_config` — configuration forwarded to [`synthesize`].
///
/// # Errors
///
/// Returns `Err` only for unrecoverable internal errors. Per-decision failures
/// (dedup, budget, provider) produce `skipped` entries rather than errors.
pub async fn sweep_cycle<P: Provider>(
    provider: &P,
    decisions: Vec<SweepDecision>,
    state: &dyn ScopedState,
    budget_cap: f64,
    compare_config: &CompareConfig,
    synthesis_config: &SynthesisConfig,
) -> Result<CycleReport, anyhow::Error> {
    let mut report = CycleReport::default();
    let mut accumulated_cost: f64 = 0.0;

    for d in decisions {
        // ------------------------------------------------------------------
        // Step 1: DEDUP CHECK
        // Read last_sweep timestamp; skip if too recent.
        // State errors here are non-fatal: skip the decision with reason.
        // ------------------------------------------------------------------
        let meta_key = format!("meta:{}:last_sweep", d.id);
        match state.read(&meta_key).await {
            Err(e) => {
                debug!(
                    decision_id = %d.id,
                    error = %e,
                    "cycle: dedup state read failed, skipping"
                );
                report
                    .skipped
                    .push((d.id.clone(), format!("dedup state error: {e}")));
                continue;
            }
            Ok(Some(meta_val)) => {
                // Accept either a full SweepMeta JSON object or a plain
                // RFC 3339 timestamp string (written by v3's compare_decision).
                let swept_at = serde_json::from_value::<SweepMeta>(meta_val.clone())
                    .ok()
                    .map(|m| m.swept_at)
                    .or_else(|| meta_val.as_str().map(str::to_string));

                if let Some(ts) = swept_at
                    && let Ok(swept_at_dt) = chrono::DateTime::parse_from_rfc3339(&ts)
                {
                    let age = chrono::Utc::now()
                        .signed_duration_since(swept_at_dt.with_timezone(&chrono::Utc));
                    if age < chrono::Duration::hours(MIN_SWEEP_INTERVAL_HOURS) {
                        debug!(decision_id = %d.id, "cycle: swept too recently, skipping");
                        report
                            .skipped
                            .push((d.id.clone(), "swept too recently".to_string()));
                        continue;
                    }
                }
            }
            Ok(None) => {} // Never swept before — proceed.
        }

        // ------------------------------------------------------------------
        // Step 2: BUDGET CHECK
        // ------------------------------------------------------------------
        if accumulated_cost >= budget_cap {
            debug!(
                decision_id = %d.id,
                accumulated_cost,
                budget_cap,
                "cycle: budget exhausted, skipping"
            );
            report.skipped.push((
                d.id.clone(),
                format!("budget exhausted (cap: {budget_cap:.4} USD)"),
            ));
            continue;
        }

        // ------------------------------------------------------------------
        // Step 3: PROCESSOR SELECTION
        // ------------------------------------------------------------------
        let budget_remaining = (budget_cap - accumulated_cost).max(0.0);
        let processor =
            select_processor(budget_remaining, budget_cap, d.previous_verdict.as_ref());

        // ------------------------------------------------------------------
        // Step 4: EMPTY RESULTS SHORT-CIRCUIT
        // ------------------------------------------------------------------
        if d.research_results.is_empty() {
            debug!(decision_id = %d.id, "cycle: no research results, short-circuiting");
            report.verdicts.push(SweepVerdict {
                decision_id: d.id.clone(),
                status: VerdictStatus::Confirmed,
                confidence: 0.3,
                num_supporting: 0,
                num_contradicting: 0,
                cost_usd: 0.0,
                processor,
                duration_secs: 0.0,
                swept_at: chrono::Utc::now().to_rfc3339(),
                evidence: vec![],
                narrative: "No new evidence found".to_string(),
                proposed_diff: None,
                research_inputs: vec![],
                query: String::new(),
                query_angle: String::new(),
            });
            continue;
        }

        // ------------------------------------------------------------------
        // Step 5: READ DECISION CARD
        // ------------------------------------------------------------------
        let decision_text = match state.read(&format!("card:{}", d.id)).await {
            Ok(Some(serde_json::Value::String(s))) => s,
            Ok(Some(other)) => other.to_string(),
            Ok(None) => {
                debug!(decision_id = %d.id, "cycle: decision card not found");
                "[Decision text not available]".to_string()
            }
            Err(e) => {
                warn!(decision_id = %d.id, error = %e, "cycle: card read failed");
                "[Decision text not available]".to_string()
            }
        };

        // ------------------------------------------------------------------
        // Step 6: READ PRIOR FINDINGS
        // ------------------------------------------------------------------
        let delta_keys = state
            .list(&format!("delta:{}:", d.id))
            .await
            .unwrap_or_default();

        let mut prior_findings: Vec<String> = Vec::with_capacity(delta_keys.len());
        for key in &delta_keys {
            match state.read(key).await {
                Ok(Some(serde_json::Value::String(s))) => prior_findings.push(s),
                Ok(Some(other)) => prior_findings.push(other.to_string()),
                Ok(None) => {}
                Err(e) => {
                    debug!(key = %key, error = %e, "cycle: delta read failed");
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 7: COMPARE
        // Errors are non-fatal per decision — skip with reason.
        // ------------------------------------------------------------------
        info!(decision_id = %d.id, "cycle: running compare");
        let verdict = match compare_decision(
            provider,
            &d.id,
            &decision_text,
            &prior_findings,
            &d.research_results,
            compare_config,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                warn!(decision_id = %d.id, error = %e, "cycle: compare failed, skipping");
                report
                    .skipped
                    .push((d.id.clone(), format!("compare failed: {e}")));
                continue;
            }
        };

        // ------------------------------------------------------------------
        // Step 8: WRITE SWEEP TIMESTAMP (non-fatal)
        // ------------------------------------------------------------------
        let sweep_meta = SweepMeta {
            swept_at: verdict.swept_at.clone(),
            verdict: verdict.status.clone(),
            cost_usd: verdict.cost_usd,
            query: verdict.query.clone(),
            query_angle: verdict.query_angle.clone(),
            processor,
        };
        if let Ok(meta_val) = serde_json::to_value(&sweep_meta) {
            let _ = state.write(&meta_key, meta_val).await;
        }

        // ------------------------------------------------------------------
        // Step 9: COLLECT
        // ------------------------------------------------------------------
        accumulated_cost += verdict.cost_usd;
        report.total_cost += verdict.cost_usd;
        report.verdicts.push(verdict);
    }

    // -----------------------------------------------------------------------
    // SYNTHESIS PASS
    // Dispatched when enough verdicts were produced. Failure is non-fatal.
    // -----------------------------------------------------------------------
    if report.verdicts.len() >= SYNTHESIS_THRESHOLD {
        info!(
            count = report.verdicts.len(),
            "cycle: triggering synthesis"
        );
        report.synthesis = synthesize(provider, &report.verdicts, synthesis_config)
            .await
            .ok();
        if let Some(ref synthesis) = report.synthesis {
            report.total_cost += synthesis.cost_usd;
        }
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::error::StateError;
    use layer0::SearchResult;
    use neuron_turn::infer::{InferRequest, InferResponse};
    use neuron_turn::provider::ProviderError;
    use neuron_turn::test_utils::TestProvider;
    use neuron_turn::types::{StopReason, TokenUsage};

    use crate::types::{ResearchResult, VerdictStatus};

    use async_trait::async_trait;
    use layer0::content::Content;

    // -----------------------------------------------------------------------
    // Mock state
    // -----------------------------------------------------------------------

    /// No-op [`ScopedState`] — reads return `None`, writes succeed.
    struct NullState;

    #[async_trait]
    impl ScopedState for NullState {
        async fn read(&self, _key: &str) -> Result<Option<serde_json::Value>, StateError> {
            Ok(None)
        }
        async fn write(&self, _key: &str, _val: serde_json::Value) -> Result<(), StateError> {
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

    fn verdict_response(verdict_json: &str) -> InferResponse {
        InferResponse {
            content: Content::text(format!("```json\n{verdict_json}\n```")),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            model: "mock".into(),
            cost: None,
            truncated: None,
        }
    }

    fn confirmed_verdict_json(id: &str) -> String {
        format!(
            r#"{{
                "decision_id": "{id}",
                "status": "confirmed",
                "confidence": 0.9,
                "num_supporting": 3,
                "num_contradicting": 0,
                "cost_usd": 0.0,
                "processor": "ultra",
                "duration_secs": 0.0,
                "swept_at": "2026-01-01T00:00:00Z",
                "evidence": [],
                "narrative": "All good.",
                "proposed_diff": null,
                "research_inputs": [],
                "query": "",
                "query_angle": ""
            }}"#
        )
    }

    fn research_result() -> ResearchResult {
        ResearchResult {
            url: "https://example.com".into(),
            title: "Example".into(),
            snippet: "Some snippet".into(),
            retrieved_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn empty_results_short_circuits_to_confirmed() {
        let provider = TestProvider::with_responses(vec![]);
        let decisions = vec![SweepDecision {
            id: "D1".into(),
            research_results: vec![],
            previous_verdict: None,
        }];

        let report = sweep_cycle(
            &provider,
            decisions,
            &NullState,
            100.0,
            &CompareConfig::default(),
            &SynthesisConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(report.verdicts.len(), 1);
        assert_eq!(report.verdicts[0].status, VerdictStatus::Confirmed);
        assert_eq!(report.verdicts[0].confidence, 0.3);
        assert!(report.skipped.is_empty());
    }

    #[tokio::test]
    async fn compare_failure_produces_skip_not_error() {
        // Use a provider that always returns Err (not panic) to test non-fatal handling.
        struct AlwaysErrProvider;
        impl neuron_turn::provider::Provider for AlwaysErrProvider {
            fn infer(
                &self,
                _req: InferRequest,
            ) -> impl std::future::Future<Output = Result<InferResponse, ProviderError>> + Send {
                async {
                    Err(ProviderError::TransientError {
                        message: "injected test error".into(),
                        status: None,
                    })
                }
            }
        }

        let decisions = vec![SweepDecision {
            id: "D1".into(),
            research_results: vec![research_result()],
            previous_verdict: None,
        }];

        let report = sweep_cycle(
            &AlwaysErrProvider,
            decisions,
            &NullState,
            100.0,
            &CompareConfig::default(),
            &SynthesisConfig::default(),
        )
        .await
        .unwrap();

        // The error is non-fatal: cycle succeeds but D1 is in skipped.
        assert!(report.verdicts.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].0, "D1");
    }

    #[tokio::test]
    async fn budget_exhausted_skips_decision() {
        let provider = TestProvider::with_responses(vec![]);
        let decisions = vec![SweepDecision {
            id: "D1".into(),
            research_results: vec![research_result()],
            previous_verdict: None,
        }];

        // budget_cap of 0.0 → immediately exhausted.
        let report = sweep_cycle(
            &provider,
            decisions,
            &NullState,
            0.0,
            &CompareConfig::default(),
            &SynthesisConfig::default(),
        )
        .await
        .unwrap();

        assert!(report.verdicts.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].1.contains("budget exhausted"));
    }

    #[tokio::test]
    async fn single_decision_produces_verdict() {
        let provider =
            TestProvider::with_responses(vec![verdict_response(&confirmed_verdict_json("D1"))]);

        let decisions = vec![SweepDecision {
            id: "D1".into(),
            research_results: vec![research_result()],
            previous_verdict: None,
        }];

        let report = sweep_cycle(
            &provider,
            decisions,
            &NullState,
            100.0,
            &CompareConfig::default(),
            &SynthesisConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(report.verdicts.len(), 1);
        assert_eq!(report.verdicts[0].decision_id, "D1");
        assert_eq!(report.verdicts[0].status, VerdictStatus::Confirmed);
        assert!(report.skipped.is_empty());
        // < 3 verdicts → no synthesis
        assert!(report.synthesis.is_none());
    }
}
