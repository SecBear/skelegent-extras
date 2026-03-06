//! Sweep workflow function — sequences research and comparison operators.
//!
//! [`run_sweep`] orchestrates the sweep pipeline end-to-end by dispatching
//! [`ResearchOperator`] and [`CompareOperator`] through an [`Orchestrator`].
//! Neither the workflow function nor the operators write state directly;
//! all state mutations are declared as [`layer0::Effect::WriteMemory`] in
//! operator outputs and executed by the orchestrator's effect interpreter.
//!
//! # Pipeline
//!
//! 1. **Dedup check** — read-only; skip if swept too recently.
//! 2. **Budget guard** — return [`VerdictStatus::Skipped`] immediately if no budget.
//! 3. **Processor selection** — choose tier from budget ratio and previous verdict.
//! 4. **Query building** — keyword query from the decision card (v1; plan-first in v2).
//! 5. **Research dispatch** — [`ResearchOperator`] via [`Orchestrator::dispatch`].
//! 6. **Compare dispatch** — [`CompareOperator`] via [`Orchestrator::dispatch`].
//! 7. **Emit** — parse and return the final [`SweepVerdict`].

use std::time::Instant;

use layer0::operator::TriggerType;
use layer0::{AgentId, Content, OperatorInput, Orchestrator, Scope, StateStore};

use crate::operator::{StoreAsReader, COMPARISON_SYSTEM_PROMPT};
use crate::provider::{ResearchResult, SweepError};
use crate::types::{ProcessorTier, SweepMeta, SweepVerdict, VerdictStatus};

/// Sequence the sweep pipeline for one decision via an [`Orchestrator`].
///
/// This function dispatches [`crate::operator::ResearchOperator`] and
/// [`crate::operator::CompareOperator`] through `orch`, passing data between
/// them via [`OperatorInput`] messages. All state writes are delegated to the
/// effects declared by each operator — this function performs only read-only
/// state access (dedup check and card text lookup).
///
/// # Parameters
///
/// - `orch` — orchestrator used to dispatch the research and compare operators.
/// - `research_agent` — [`AgentId`] of the registered [`crate::operator::ResearchOperator`].
/// - `compare_agent` — [`AgentId`] of the registered [`crate::operator::CompareOperator`].
/// - `decision_id` — identifier for the decision under review.
/// - `previous_verdict` — outcome of the last sweep; drives processor tier selection.
/// - `budget_remaining_usd` — remaining budget; zero or negative returns Skipped immediately.
/// - `budget_total_usd` — total budget cap; used to compute budget ratio for tier selection.
/// - `store` — state store used **read-only** for the dedup check and card text lookup.
///
/// # Returns
///
/// A [`SweepVerdict`] in all non-error cases. Status may be
/// [`VerdictStatus::Skipped`] for dedup/budget/not-found short-circuits, or
/// [`VerdictStatus::Confirmed`] with low confidence when research returns no results.
///
/// # Errors
///
/// Returns [`SweepError`] for unrecoverable operator or dispatch failures.
#[allow(clippy::too_many_arguments)]
pub async fn run_sweep(
    orch: &dyn Orchestrator,
    research_agent: &AgentId,
    compare_agent: &AgentId,
    decision_id: &str,
    previous_verdict: Option<VerdictStatus>,
    budget_remaining_usd: f64,
    budget_total_usd: f64,
    store: &dyn StateStore,
) -> Result<SweepVerdict, SweepError> {
    let start = Instant::now();
    let scope = Scope::Custom("sweep".to_string());

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
    // Step 3: PROCESSOR SELECTION + CONTEXT ASSEMBLY
    // ------------------------------------------------------------------
    let processor =
        crate::operator::select_processor(budget_remaining_usd, budget_total_usd, previous_verdict.as_ref());

    // Read the decision card for keyword query generation (read-only).
    let card_key = format!("card:{decision_id}");
    let card_text = match store.read(&scope, &card_key).await.ok().flatten() {
        Some(serde_json::Value::String(s)) => s,
        Some(other) => serde_json::to_string(&other).unwrap_or_default(),
        None => decision_id.to_string(),
    };

    // Context assembly (v1: result discarded; included for spec compliance).
    let assembler =
        neuron_context::ContextAssembler::new(neuron_context::ContextAssemblyConfig::default());
    let reader = StoreAsReader(store);
    let _context_messages = assembler
        .assemble(&reader, &scope, decision_id, Some(COMPARISON_SYSTEM_PROMPT))
        .await
        .unwrap_or_default();

    // ------------------------------------------------------------------
    // Step 4: BUILD QUERY (keyword in v1; plan-first in v2)
    // ------------------------------------------------------------------
    let query = crate::operator::keyword_query(&card_text);

    // Encode processor tier as JSON string for metadata.
    let processor_json = serde_json::to_value(&processor).unwrap_or(serde_json::json!("base"));

    // ------------------------------------------------------------------
    // Step 5: DISPATCH RESEARCH OPERATOR
    // ------------------------------------------------------------------
    let mut research_input = OperatorInput::new(Content::text(query), TriggerType::Task);
    research_input.metadata = serde_json::json!({
        "processor": processor_json,
        "decision_id": decision_id,
    });

    let research_output = orch
        .dispatch(research_agent, research_input)
        .await
        .map_err(|e| SweepError::Permanent(e.to_string()))?;

    // Parse research output.
    let research_text = research_output.message.as_text().unwrap_or("[]");

    if research_text.starts_with("DECISION_NOT_FOUND:") {
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
            narrative: format!("Decision {decision_id} not found"),
            proposed_diff: None,
        });
    }

    let results: Vec<ResearchResult> = serde_json::from_str(research_text)
        .map_err(|e| SweepError::Permanent(format!("research output parse failed: {e}")))?;

    // Edge case: 0 results → Confirmed with low confidence (no compare needed).
    if results.is_empty() {
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
    // Step 6: DISPATCH COMPARE OPERATOR
    // ------------------------------------------------------------------
    let research_json =
        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());

    let mut compare_input = OperatorInput::new(Content::text(research_json), TriggerType::Task);
    compare_input.metadata = serde_json::json!({
        "decision_id": decision_id,
    });

    let compare_output = orch
        .dispatch(compare_agent, compare_input)
        .await
        .map_err(|e| SweepError::Permanent(e.to_string()))?;

    // Parse compare output.
    let verdict_text = compare_output.message.as_text().unwrap_or("");

    if verdict_text.starts_with("DECISION_NOT_FOUND:") {
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
            narrative: format!("Decision {decision_id} not found during comparison"),
            proposed_diff: None,
        });
    }

    let verdict: SweepVerdict = serde_json::from_str(verdict_text)
        .map_err(|e| SweepError::Permanent(format!("compare output parse failed: {e}")))?;

    // ------------------------------------------------------------------
    // Step 7: EMIT
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
    use crate::operator::{CompareOperator, ResearchOperator, SweepOperatorConfig};
    use crate::provider::MockProvider;
    use crate::types::{EvidenceItem, EvidenceStance, VerdictStatus};
    use layer0::error::StateError;
    use layer0::state::{SearchResult, StateStore};
    use layer0::test_utils::LocalOrchestrator;
    use neuron_turn::provider::{Provider, ProviderError};
    use neuron_turn::types::{ContentPart, ProviderRequest, ProviderResponse, StopReason, TokenUsage};
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

    /// Build a `LocalOrchestrator` with a `ResearchOperator` and a `CompareOperator`.
    /// The compare operator uses a mock LLM that returns `compare_verdict` as JSON.
    fn build_orch(
        research_mock: MockProvider,
        compare_verdict: SweepVerdict,
    ) -> (LocalOrchestrator, AgentId, AgentId) {
        struct MockLlm { verdict: SweepVerdict }
        impl Provider for MockLlm {
            fn complete(&self, _req: ProviderRequest)
                -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>> + Send {
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
        let research_id = AgentId::new("research");
        let compare_id = AgentId::new("compare");
        let mut orch = LocalOrchestrator::new();
        orch.register(
            research_id.clone(),
            Arc::new(ResearchOperator::new(Box::new(research_mock), SweepOperatorConfig::default())),
        );
        orch.register(
            compare_id.clone(),
            Arc::new(CompareOperator::new(MockLlm { verdict: compare_verdict }, SweepOperatorConfig::default())),
        );
        (orch, research_id, compare_id)
    }

    // -----------------------------------------------------------------------
    // run_sweep — short-circuit paths
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn run_sweep_returns_skipped_when_budget_is_zero() {
        let (orch, research_id, compare_id) = build_orch(
            MockProvider::new(vec![dummy_result()]),
            dummy_verdict("topic-3b"),
        );

        let verdict = run_sweep(
            &orch,
            &research_id,
            &compare_id,
            "topic-3b",
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
        // Research operator returns empty results → workflow returns Confirmed(0.3).
        let (orch, research_id, compare_id) = build_orch(
            MockProvider::new(vec![]),
            dummy_verdict("topic-3b"),
        );

        let verdict = run_sweep(
            &orch,
            &research_id,
            &compare_id,
            "topic-3b",
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
    async fn run_sweep_returns_verdict_from_operators() {
        // Full happy-path: research finds results, compare produces a verdict.
        let expected = dummy_verdict("topic-3b");
        let (orch, research_id, compare_id) = build_orch(
            MockProvider::new(vec![dummy_result()]),
            expected.clone(),
        );

        let verdict = run_sweep(
            &orch,
            &research_id,
            &compare_id,
            "topic-3b",
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
    async fn run_sweep_sequences_research_before_compare() {
        // Verifies that compare is NOT called when research returns empty results.
        // CompareOperator uses a provider that panics if called.
        struct PanicLlm;
        impl Provider for PanicLlm {
            fn complete(&self, _req: ProviderRequest)
                -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>> + Send {
                async move { panic!("compare LLM should not be called when research returns empty") }
            }
        }
        let research_id = AgentId::new("research");
        let compare_id = AgentId::new("compare");
        let mut orch = LocalOrchestrator::new();
        orch.register(
            research_id.clone(),
            Arc::new(ResearchOperator::new(Box::new(MockProvider::new(vec![])), SweepOperatorConfig::default())),
        );
        orch.register(
            compare_id.clone(),
            Arc::new(CompareOperator::new(PanicLlm, SweepOperatorConfig::default())),
        );

        let verdict = run_sweep(
            &orch,
            &research_id,
            &compare_id,
            "topic-3b",
            None,
            8.0,
            10.0,
            &EmptyStore,
        )
        .await
        .expect("run_sweep should short-circuit before compare");

        assert_eq!(verdict.status, VerdictStatus::Confirmed);
        assert!((verdict.confidence - 0.3).abs() < f64::EPSILON);
    }
}
