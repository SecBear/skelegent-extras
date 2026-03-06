//! Sweep workflow function — sequences dedup, budget, and comparison.
//!
//! [`run_sweep`] orchestrates the sweep pipeline end-to-end by dispatching
//! [`CompareOperator`] through an [`Orchestrator`]. Research results are
//! supplied by the caller (orchestrator-owned research). Context assembly
//! is **turn-owned**: [`CompareOperator`] reads decision cards and prior
//! deltas from the state store during `execute()`.
//!
//! # Pipeline
//!
//! 1. **Dedup check** \u2014 read-only; skip if swept too recently.
//! 2. **Budget guard** \u2014 return [`VerdictStatus::Skipped`] immediately if no budget.
//! 3. **Processor selection** \u2014 choose tier from budget ratio and previous verdict.
//! 4. **Empty results check** \u2014 0 results \u2192 Confirmed with low confidence.
//! 5. **Compare dispatch** \u2014 [`CompareOperator`] via [`Orchestrator::dispatch`].
//! 6. **Emit** \u2014 parse and return the final [`SweepVerdict`].

use std::time::Instant;

use layer0::operator::TriggerType;
use layer0::{AgentId, Orchestrator, Scope, StateStore};
use neuron_orch_kit::{dispatch_typed, DispatchError};

use crate::operator::SWEEP_SCOPE;
use crate::provider::{CompareInput, ResearchResult, SweepError};
use crate::types::{ProcessorTier, SweepMeta, SweepVerdict, VerdictStatus};

/// Sequence the sweep pipeline for one decision via an [`Orchestrator`].
///
/// Dispatches [`crate::operator::CompareOperator`] through `orch`.
/// Context assembly (decision card, prior deltas) is **turn-owned** \u2014
/// the operator reads from `store` during `execute()`. This function
/// only performs read-only state access for the dedup check.
///
/// # Parameters
///
/// - `orch` \u2014 orchestrator used to dispatch the compare operator.
/// - `compare_agent` \u2014 [`AgentId`] of the registered [`crate::operator::CompareOperator`].
/// - `decision_id` \u2014 identifier for the decision under review.
/// - `research_results` \u2014 research results to compare against the decision.
/// - `previous_verdict` \u2014 outcome of the last sweep; drives processor tier selection.
/// - `budget_remaining_usd` \u2014 remaining budget; zero or negative returns Skipped immediately.
/// - `budget_total_usd` \u2014 total budget cap; used to compute budget ratio for tier selection.
/// - `store` \u2014 state store used **read-only** for the dedup check.
///
/// # Returns
///
/// A [`SweepVerdict`] in all non-error cases. Status may be
/// [`VerdictStatus::Skipped`] for dedup/budget short-circuits, or
/// [`VerdictStatus::Confirmed`] with low confidence when research returns no results.
///
/// # Errors
///
/// Returns [`SweepError`] for unrecoverable operator or dispatch failures.
#[allow(clippy::too_many_arguments)]
pub async fn run_sweep(
    orch: &dyn Orchestrator,
    compare_agent: &AgentId,
    decision_id: &str,
    research_results: &[ResearchResult],
    previous_verdict: Option<VerdictStatus>,
    budget_remaining_usd: f64,
    budget_total_usd: f64,
    store: &dyn StateStore,
) -> Result<SweepVerdict, SweepError> {
    let start = Instant::now();
    let scope = Scope::Custom(SWEEP_SCOPE.to_string());

    // ------------------------------------------------------------------
    // Step 1: DEDUP CHECK
    // Read-only: skip if the same decision was swept too recently.
    // The 20-hour default matches SweepOperatorConfig::default().min_sweep_interval.
    // ------------------------------------------------------------------
    let meta_key = format!("meta:{}:last_sweep", decision_id);
    if let Ok(Some(meta_val)) = store.read(&scope, &meta_key).await
        && let Ok(meta) = serde_json::from_value::<SweepMeta>(meta_val)
        && let Ok(swept_at) = chrono::DateTime::parse_from_rfc3339(&meta.swept_at)
    {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(swept_at.with_timezone(&chrono::Utc));
        let min_interval = chrono::Duration::hours(20);
        if age < min_interval {
            return Ok(SweepVerdict {
                decision_id: decision_id.to_string(),
                status: VerdictStatus::Skipped,
                confidence: 0.0,
                num_supporting: 0,
                num_contradicting: 0,
                cost_usd: 0.0,
                processor: ProcessorTier::Base,
                duration_secs: start.elapsed().as_secs_f64(),
                swept_at: chrono::Utc::now().to_rfc3339(),
                evidence: vec![],
                narrative: "Too soon since last sweep".to_string(),
                proposed_diff: None,
            });
        }
    }

    // ------------------------------------------------------------------
    // Step 2: BUDGET GUARD
    // ------------------------------------------------------------------
    if budget_remaining_usd <= 0.0 {
        return Ok(SweepVerdict {
            decision_id: decision_id.to_string(),
            status: VerdictStatus::Skipped,
            confidence: 0.0,
            num_supporting: 0,
            num_contradicting: 0,
            cost_usd: 0.0,
            processor: ProcessorTier::Base,
            duration_secs: start.elapsed().as_secs_f64(),
            swept_at: chrono::Utc::now().to_rfc3339(),
            evidence: vec![],
            narrative: "Budget insufficient for any processor".to_string(),
            proposed_diff: None,
        });
    }

    // ------------------------------------------------------------------
    // Step 3: PROCESSOR SELECTION
    // ------------------------------------------------------------------
    let processor = crate::operator::select_processor(
        budget_remaining_usd,
        budget_total_usd,
        previous_verdict.as_ref(),
    );

    // ------------------------------------------------------------------
    // Step 4: EMPTY RESULTS CHECK
    // 0 results \u2192 Confirmed with low confidence (no compare needed).
    // ------------------------------------------------------------------
    if research_results.is_empty() {
        return Ok(SweepVerdict {
            decision_id: decision_id.to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.3,
            num_supporting: 0,
            num_contradicting: 0,
            cost_usd: 0.0,
            processor,
            duration_secs: start.elapsed().as_secs_f64(),
            swept_at: chrono::Utc::now().to_rfc3339(),
            evidence: vec![],
            narrative: "No new evidence found".to_string(),
            proposed_diff: None,
        });
    }

    // ------------------------------------------------------------------
    // Step 5: DISPATCH COMPARE OPERATOR
    // Context assembly happens inside CompareOperator::execute() (turn-owned).
    // ------------------------------------------------------------------
    let compare_in = CompareInput {
        research_results: research_results.to_vec(),
        decision_id: decision_id.to_string(),
    };

    let verdict = match dispatch_typed::<CompareInput, SweepVerdict>(
        orch,
        compare_agent,
        compare_in,
        TriggerType::Task,
    )
    .await
    {
        Ok((verdict, _output)) => verdict,
        Err(DispatchError::DeserializeOutput(msg)) => {
            // The operator returned a non-JSON response. This covers the
            // hypothetical DECISION_NOT_FOUND prefix and any other malformed
            // output. Return Skipped rather than propagating a parse error.
            return Ok(SweepVerdict {
                decision_id: decision_id.to_string(),
                status: VerdictStatus::Skipped,
                confidence: 0.0,
                num_supporting: 0,
                num_contradicting: 0,
                cost_usd: 0.0,
                processor,
                duration_secs: start.elapsed().as_secs_f64(),
                swept_at: chrono::Utc::now().to_rfc3339(),
                evidence: vec![],
                narrative: format!("Comparison step returned non-verdict response: {msg}"),
                proposed_diff: None,
            });
        }
        Err(e) => return Err(SweepError::Permanent(e.to_string())),
    };

    // ------------------------------------------------------------------
    // Step 6: EMIT
    // Override identity fields so callers don't need to trust the operator.
    // ------------------------------------------------------------------
    Ok(SweepVerdict {
        decision_id: decision_id.to_string(),
        processor,
        duration_secs: start.elapsed().as_secs_f64(),
        swept_at: chrono::Utc::now().to_rfc3339(),
        ..verdict
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::{CompareOperator, SweepOperatorConfig};
    use crate::types::{EvidenceItem, EvidenceStance, VerdictStatus};
    use layer0::error::StateError;
    use layer0::state::{SearchResult, StateStore};
    use layer0::test_utils::LocalOrchestrator;
    use neuron_turn::provider::{Provider, ProviderError};
    use neuron_turn::types::{
        ContentPart, ProviderRequest, ProviderResponse, Role, StopReason, TokenUsage,
    };
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    struct EmptyStore;

    #[async_trait::async_trait]
    impl StateStore for EmptyStore {
        async fn read(
            &self,
            _s: &Scope,
            _k: &str,
        ) -> Result<Option<serde_json::Value>, StateError> {
            Ok(None)
        }
        async fn write(
            &self,
            _s: &Scope,
            _k: &str,
            _v: serde_json::Value,
        ) -> Result<(), StateError> {
            Ok(())
        }
        async fn delete(&self, _s: &Scope, _k: &str) -> Result<(), StateError> {
            Ok(())
        }
        async fn list(&self, _s: &Scope, _p: &str) -> Result<Vec<String>, StateError> {
            Ok(vec![])
        }
        async fn search(
            &self,
            _s: &Scope,
            _q: &str,
            _l: usize,
        ) -> Result<Vec<SearchResult>, StateError> {
            Ok(vec![])
        }
    }

    fn dummy_result() -> crate::provider::ResearchResult {
        crate::provider::ResearchResult {
            url: "https://example.com/paper".to_string(),
            title: "Agent Architecture 2026".to_string(),
            snippet: "Key findings about agent systems.".to_string(),
            retrieved_at: "2026-03-04T17:00:00Z".to_string(),
        }
    }

    fn dummy_verdict(decision_id: &str) -> SweepVerdict {
        SweepVerdict {
            decision_id: decision_id.to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.9,
            num_supporting: 3,
            num_contradicting: 0,
            cost_usd: 0.10,
            processor: ProcessorTier::Base,
            duration_secs: 2.0,
            swept_at: "2026-03-04T18:00:00Z".to_string(),
            evidence: vec![EvidenceItem {
                source_url: "https://example.com".into(),
                summary: "Supports decision".into(),
                stance: EvidenceStance::Supporting,
                retrieved_at: "2026-03-04T00:00:00Z".into(),
            }],
            narrative: "Confirmed by research".to_string(),
            proposed_diff: None,
        }
    }

    /// Build a `LocalOrchestrator` with a `CompareOperator` using a mock LLM
    /// that returns `compare_verdict` as JSON.
    fn build_orch(compare_verdict: SweepVerdict) -> (LocalOrchestrator, AgentId) {
        struct MockLlm {
            verdict: SweepVerdict,
        }
        impl Provider for MockLlm {
            fn complete(
                &self,
                _req: ProviderRequest,
            ) -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>> + Send
            {
                let json = serde_json::to_string(&self.verdict).expect("serialize");
                async move {
                    Ok(ProviderResponse {
                        content: vec![ContentPart::Text { text: json }],
                        stop_reason: StopReason::EndTurn,
                        usage: TokenUsage::default(),
                        model: "mock".into(),
                        cost: None,
                        truncated: None,
                    })
                }
            }
        }
        let compare_id = AgentId::new("compare");
        let mut orch = LocalOrchestrator::new();
        orch.register(
            compare_id.clone(),
            Arc::new(CompareOperator::new(
                MockLlm { verdict: compare_verdict },
                Arc::new(EmptyStore),
                SweepOperatorConfig::default(),
            )),
        );
        (orch, compare_id)
    }

    // -----------------------------------------------------------------------
    // run_sweep — short-circuit paths
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn run_sweep_returns_skipped_when_budget_is_zero() {
        let (orch, compare_id) = build_orch(dummy_verdict("topic-3b"));

        let verdict = run_sweep(
            &orch,
            &compare_id,
            "topic-3b",
            &[dummy_result()],
            None,
            0.0,
            10.0,
            &EmptyStore,
        )
        .await
        .expect("run_sweep should not error for budget skip");

        assert_eq!(verdict.status, VerdictStatus::Skipped);
        assert!(verdict.narrative.contains("Budget"));
    }

    #[tokio::test]
    async fn run_sweep_returns_confirmed_low_confidence_on_no_results() {
        // Empty research_results → workflow returns Confirmed(0.3).
        let (orch, compare_id) = build_orch(dummy_verdict("topic-3b"));

        let verdict = run_sweep(
            &orch,
            &compare_id,
            "topic-3b",
            &[],
            None,
            8.0,
            10.0,
            &EmptyStore,
        )
        .await
        .expect("run_sweep should succeed");

        assert_eq!(verdict.status, VerdictStatus::Confirmed);
        assert!(
            (verdict.confidence - 0.3).abs() < f64::EPSILON,
            "confidence should be 0.3, got {}",
            verdict.confidence
        );
        assert!(verdict.narrative.contains("No new evidence found"));
    }

    #[tokio::test]
    async fn run_sweep_returns_verdict_from_compare_operator() {
        // Full happy-path: research results passed in, compare produces a verdict.
        let expected = dummy_verdict("topic-3b");
        let (orch, compare_id) = build_orch(expected.clone());

        let verdict = run_sweep(
            &orch,
            &compare_id,
            "topic-3b",
            &[dummy_result()],
            None,
            8.0,
            10.0,
            &EmptyStore,
        )
        .await
        .expect("run_sweep should succeed");

        assert_eq!(verdict.decision_id, "topic-3b");
        assert_eq!(verdict.status, VerdictStatus::Confirmed);
        // Workflow overrides processor and swept_at; provider's values are replaced.
        assert!(!verdict.swept_at.is_empty());
    }

    #[tokio::test]
    async fn run_sweep_passes_decision_text_to_compare() {
        // Verify that decision_text flows through to the compare operator's LLM request.
        let verdict = dummy_verdict("topic-3b");
        let verdict_json = serde_json::to_string(&verdict).expect("serialize");

        struct CaptureLlm {
            captured: Arc<std::sync::Mutex<String>>,
            verdict_json: String,
        }
        impl Provider for CaptureLlm {
            fn complete(
                &self,
                req: ProviderRequest,
            ) -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>> + Send
            {
                let captured = self.captured.clone();
                let json = self.verdict_json.clone();
                let user_text = req
                    .messages
                    .into_iter()
                    .find(|m| matches!(m.role, Role::User))
                    .and_then(|m| {
                        m.content.into_iter().find_map(|c| {
                            if let ContentPart::Text { text } = c {
                                Some(text)
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or_default();
                async move {
                    *captured.lock().unwrap() = user_text;
                    Ok(ProviderResponse {
                        content: vec![ContentPart::Text { text: json }],
                        stop_reason: StopReason::EndTurn,
                        usage: TokenUsage::default(),
                        model: "mock".into(),
                        cost: None,
                        truncated: None,
                    })
                }
            }
        }

        let captured_text = Arc::new(std::sync::Mutex::new(String::new()));
        let compare_id = AgentId::new("compare");
        let mut orch = LocalOrchestrator::new();
        orch.register(
            compare_id.clone(),
            Arc::new(CompareOperator::new(
                CaptureLlm {
                    captured: captured_text.clone(),
                    verdict_json,
                },
                Arc::new(EmptyStore),
                SweepOperatorConfig::default(),
            )),
        );

        let _verdict = run_sweep(
            &orch,
            &compare_id,
            "topic-3b",
            &[dummy_result()],
            None,
            8.0,
            10.0,
            &EmptyStore,
        )
        .await
        .expect("run_sweep should succeed");

        let text = captured_text.lock().unwrap().clone();
        assert!(
            text.contains("<decision>"),
            "user message should contain <decision> section, got: {text}"
        );
        // decision text comes from state store — EmptyStore returns None,
        // so the fallback text is used.
        assert!(
            text.contains("[Decision text not available]"),
            "user message should contain fallback decision text, got: {text}"
        );
        assert!(
            text.contains("<prior_findings>"),
            "user message should contain <prior_findings> section, got: {text}"
        );
    }
}
