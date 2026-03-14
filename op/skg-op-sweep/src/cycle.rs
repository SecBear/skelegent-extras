//! Sweep cycle composition — multi-decision orchestration and synthesis.
//!
//! [`run_sweep_cycle`] sequences a batch of [`SweepDecision`]s through the
//! comparison pipeline, respecting budget and trace guards, and triggers a
//! cross-decision synthesis pass when enough verdicts accumulate.
//!
//! # Pipeline (per decision)
//!
//! 1. **Dedup check** — read-only; skip if swept too recently (< 20 h).
//! 2. **Budget check** — skip on `Deny`; continue on `Degraded` or `Allow`.
//! 3. **Processor selection** — tier from remaining-budget ratio and prior verdict.
//! 4. **Empty-results short-circuit** — 0 results → `Confirmed(0.3)`, no dispatch.
//! 5. **Trace enter** — cycle/depth guard; skip on error.
//! 6. **Compare dispatch** — `dispatch_typed::<CompareInput, SweepVerdict>`.
//! 7. **Trace exit** — always paired with enter.
//! 8. **Budget record** — accumulate verdict cost.
//! 9. **Collect** — push verdict into report, accumulate cost tracker.
//!
//! # Synthesis
//!
//! When the cycle produces ≥ 3 verdicts, a synthesis agent is dispatched via
//! `dispatch_typed::<SynthesisInput, SynthesisReport>`. A deserialization
//! failure is non-fatal (synthesis absent from report); other errors propagate
//! as [`SweepError::Permanent`].
//!
//! # Fractal promotion
//!
//! [`SweepCycleOperator`] wraps `run_sweep_cycle` behind the [`Operator`] trait,
//! allowing the entire cycle to be dispatched through a [`Dispatcher`] as a
//! single agent — composing cycles within larger workflows.

use std::sync::Arc;

use async_trait::async_trait;
use layer0::operator::TriggerType;
use layer0::dispatch::Dispatcher;
use layer0::{
    OperatorId, Content, ExitReason, Operator, OperatorError, OperatorInput, OperatorOutput,
};
use layer0::dispatch::EffectEmitter;
use skg_orch_compose::{
    dispatch_typed, BudgetDecision, BudgetPolicy, BudgetTracker, CompositionTrace, DispatchError,
    ScopedState,
};
use serde::{Deserialize, Serialize};

use crate::cost::SweepCostTracker;
use crate::compare::select_processor;
use crate::provider::{CompareInput, ResearchResult, SweepError};
use crate::synthesis::SynthesisReport;
use crate::synthesis_operator::SynthesisInput;
use crate::types::{SweepMeta, SweepVerdict, VerdictStatus};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum hours that must elapse between two sweeps of the same decision.
const MIN_SWEEP_INTERVAL_HOURS: i64 = 20;

/// Minimum number of verdicts required to trigger the synthesis pass.
const SYNTHESIS_THRESHOLD: usize = 3;

// ---------------------------------------------------------------------------
// SweepDecision
// ---------------------------------------------------------------------------

/// Per-decision input for a sweep cycle.
///
/// The caller assembles one [`SweepDecision`] per decision under review and
/// passes the full slice to [`run_sweep_cycle`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepDecision {
    /// Decision identifier.
    pub id: String,
    /// Research results for this decision (caller-supplied).
    ///
    /// When empty, the pipeline short-circuits to `Confirmed(0.3)` without
    /// dispatching the compare operator.
    pub research_results: Vec<ResearchResult>,
    /// Previous verdict, if any.
    ///
    /// Drives processor tier selection: a prior `Challenged` verdict with
    /// sufficient remaining budget selects `Ultra`; `Refined` selects `Core`.
    pub previous_verdict: Option<VerdictStatus>,
}

// ---------------------------------------------------------------------------
// CycleReport
// ---------------------------------------------------------------------------

/// Output of a complete sweep cycle.
///
/// Produced by [`run_sweep_cycle`]. Verdicts are indexed in decision order
/// (skipped decisions are absent from `verdicts` and present in `skipped`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleReport {
    /// Verdicts from completed comparisons.
    pub verdicts: Vec<SweepVerdict>,
    /// Decisions that were skipped and the reason for each skip.
    ///
    /// Each entry is `(decision_id, reason)`. Reasons include:
    /// "swept too recently", budget denial messages, trace errors, and
    /// output parse failures.
    pub skipped: Vec<(String, String)>,
    /// Cross-decision synthesis report, present when ≥ 3 verdicts were produced.
    pub synthesis: Option<SynthesisReport>,
    /// Accumulated cost tracker for this cycle.
    pub cost: SweepCostTracker,
}

impl CycleReport {
    /// Create a new, empty report.
    pub fn new() -> Self {
        Self::default()
    }
}


// ---------------------------------------------------------------------------
// run_sweep_cycle
// ---------------------------------------------------------------------------

/// Sequence the sweep cycle pipeline for a batch of decisions.
///
/// Dispatches the compare operator for each decision through `orch`, guarded by
/// `budget` and `trace`. When ≥ 3 verdicts are collected, dispatches the
/// synthesis operator to produce cross-decision analysis.
///
/// Context assembly (decision card, prior deltas) is **turn-owned**: the
/// compare operator reads from state during its `execute()`. This function
/// performs read-only state access for dedup checks and a single write at
/// the end to persist the cycle report.
///
/// # Parameters
///
/// - `orch` — dispatcher used to dispatch compare and synthesis operators.
/// - `state` — scoped state for dedup reads and cycle report persistence.
/// - `budget` — budget tracker; consulted before each decision and updated
///   after each verdict.
/// - `trace` — composition trace for cycle and depth detection.
/// - `compare_operator` — [`OperatorId`] of the registered compare operator.
/// - `synthesis_operator` — [`OperatorId`] of the registered synthesis operator.
/// - `budget_total_usd` — full budget cap in USD; used to compute the
///   remaining-budget ratio for processor tier selection.
/// - `decisions` — batch of decisions to sweep in this cycle.
///
/// # Errors
///
/// Returns [`SweepError::Permanent`] for unrecoverable dispatch or
/// serialization failures. Transient, per-decision errors (dedup, budget
/// denial, trace depth, output parse) produce `skipped` entries rather than
/// hard errors.
#[allow(clippy::too_many_arguments)]
pub async fn run_sweep_cycle<B: BudgetPolicy>(
    orch: &dyn Dispatcher,
    state: &dyn ScopedState,
    budget: &BudgetTracker<B>,
    trace: &CompositionTrace,
    compare_operator: &OperatorId,
    synthesis_operator: &OperatorId,
    budget_total_usd: f64,
    decisions: Vec<SweepDecision>,
) -> Result<CycleReport, SweepError> {
    let mut report = CycleReport::new();

    for d in decisions {
        // ------------------------------------------------------------------
        // Step 1: DEDUP CHECK
        // Read-only: skip if the same decision was swept too recently.
        // ------------------------------------------------------------------
        let meta_key = format!("meta:{}:last_sweep", d.id);
        if let Ok(Some(meta_val)) = state.read(&meta_key).await
            && let Ok(meta) = serde_json::from_value::<SweepMeta>(meta_val)
            && let Ok(swept_at) = chrono::DateTime::parse_from_rfc3339(&meta.swept_at)
        {
            let age = chrono::Utc::now()
                .signed_duration_since(swept_at.with_timezone(&chrono::Utc));
            if age < chrono::Duration::hours(MIN_SWEEP_INTERVAL_HOURS) {
                report
                    .skipped
                    .push((d.id.clone(), "swept too recently".to_string()));
                continue;
            }
        }

        // ------------------------------------------------------------------
        // Step 2: BUDGET CHECK
        // ------------------------------------------------------------------
        match budget.check() {
            BudgetDecision::Deny { reason } => {
                report.skipped.push((d.id.clone(), reason));
                continue;
            }
            BudgetDecision::Degraded { .. } | BudgetDecision::Allow => {}
        }

        // ------------------------------------------------------------------
        // Step 3: PROCESSOR SELECTION
        // Derive remaining budget from tracker state and configured cap.
        // ------------------------------------------------------------------
        let budget_remaining = (budget_total_usd - budget.total_usd()).max(0.0);
        let processor =
            select_processor(budget_remaining, budget_total_usd, d.previous_verdict.as_ref());

        // ------------------------------------------------------------------
        // Step 4: EMPTY RESULTS SHORT-CIRCUIT
        // 0 results → Confirmed with low confidence; no dispatch needed.
        // ------------------------------------------------------------------
        if d.research_results.is_empty() {
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
        // Step 5: TRACE ENTER
        // Guard against dispatch cycles and excessive depth.
        // ------------------------------------------------------------------
        let span = match trace.enter(compare_operator) {
            Ok(span) => span,
            Err(e) => {
                report.skipped.push((d.id.clone(), format!("trace enter failed: {e}")));
                continue;
            }
        };

        // ------------------------------------------------------------------
        // Step 6: DISPATCH COMPARE OPERATOR
        // Context assembly happens inside CompareOperator::execute() (turn-owned).
        // ------------------------------------------------------------------
        let compare_in = CompareInput {
            research_results: d.research_results,
            decision_id: d.id.clone(),
            query: None,
            query_angle: None,
        };
        let dispatch_result = dispatch_typed::<CompareInput, SweepVerdict>(
            orch,
            compare_operator,
            compare_in,
            TriggerType::Task,
        )
        .await;

        // ------------------------------------------------------------------
        // Step 7: TRACE EXIT — always paired with enter.
        // ------------------------------------------------------------------
        trace.exit(&span);

        // Handle dispatch result.
        let verdict = match dispatch_result {
            Ok((verdict, _output)) => verdict,
            Err(DispatchError::DeserializeOutput(msg)) => {
                // Non-fatal: operator returned a non-JSON response.
                report.skipped.push((
                    d.id.clone(),
                    format!("compare output parse failed: {msg}"),
                ));
                continue;
            }
            Err(e) => return Err(SweepError::Permanent(e.to_string())),
        };

        // ------------------------------------------------------------------
        // Step 8: BUDGET RECORD
        // ------------------------------------------------------------------
        budget.record(verdict.cost_usd);

        // ------------------------------------------------------------------
        // Step 9: COLLECT
        // ------------------------------------------------------------------
        report.cost.comparison_cost_usd += verdict.cost_usd;
        report.verdicts.push(verdict);
    }

    // -----------------------------------------------------------------------
    // SYNTHESIS PASS
    // Dispatched when enough verdicts were produced in this cycle.
    // -----------------------------------------------------------------------
    if report.verdicts.len() >= SYNTHESIS_THRESHOLD {
        let synthesis_in = SynthesisInput {
            verdicts: report.verdicts.clone(),
        };
        match dispatch_typed::<SynthesisInput, SynthesisReport>(
            orch,
            synthesis_operator,
            synthesis_in,
            TriggerType::Task,
        )
        .await
        {
            Ok((synthesis_report, _)) => report.synthesis = Some(synthesis_report),
            Err(DispatchError::DeserializeOutput(_)) => {
                // Non-fatal: synthesis absent from report.
            }
            Err(e) => return Err(SweepError::Permanent(e.to_string())),
        }
    }

    // -----------------------------------------------------------------------
    // PERSIST CYCLE REPORT
    // -----------------------------------------------------------------------
    let report_value = serde_json::to_value(&report)
        .map_err(|e| SweepError::Permanent(format!("failed to serialize cycle report: {e}")))?;
    state
        .write("cycle:latest", report_value)
        .await
        .map_err(|e| SweepError::Permanent(format!("failed to persist cycle report: {e}")))?;

    Ok(report)
}

// ---------------------------------------------------------------------------
// SweepCycleOperator
// ---------------------------------------------------------------------------

/// Fractal-promotion wrapper: exposes [`run_sweep_cycle`] as an [`Operator`].
///
/// Allows the full sweep cycle — budget-guarded, trace-aware, multi-decision
/// orchestration plus synthesis — to itself be dispatched through an
/// [`Dispatcher`] as a single composable agent.
///
/// # Input
///
/// `input.message` must be JSON-serialized `Vec<SweepDecision>`.
///
/// # Output
///
/// `output.message` is JSON-serialized [`CycleReport`].
pub struct SweepCycleOperator<B: BudgetPolicy> {
    orch: Arc<dyn Dispatcher>,
    state: Arc<dyn ScopedState>,
    budget: Arc<BudgetTracker<B>>,
    trace: Arc<CompositionTrace>,
    compare_operator: OperatorId,
    synthesis_operator: OperatorId,
    budget_total_usd: f64,
}

impl<B: BudgetPolicy> SweepCycleOperator<B> {
    /// Create a new [`SweepCycleOperator`].
    ///
    /// # Parameters
    ///
    /// - `orch` — dispatcher used to dispatch compare and synthesis operators.
    /// - `state` — scoped state for dedup reads and cycle report persistence.
    /// - `budget` — shared budget tracker; shared across dispatches.
    /// - `trace` — composition trace for cycle/depth detection.
    /// - `compare_operator` — [`OperatorId`] of the registered compare operator.
    /// - `synthesis_operator` — [`OperatorId`] of the registered synthesis operator.
    /// - `budget_total_usd` — full budget cap in USD.
    pub fn new(
        orch: Arc<dyn Dispatcher>,
        state: Arc<dyn ScopedState>,
        budget: Arc<BudgetTracker<B>>,
        trace: Arc<CompositionTrace>,
        compare_operator: OperatorId,
        synthesis_operator: OperatorId,
        budget_total_usd: f64,
    ) -> Self {
        Self {
            orch,
            state,
            budget,
            trace,
            compare_operator,
            synthesis_operator,
            budget_total_usd,
        }
    }
}

#[async_trait]
impl<B: BudgetPolicy + 'static> Operator for SweepCycleOperator<B> {
    async fn execute(&self, input: OperatorInput, _emitter: &EffectEmitter) -> Result<OperatorOutput, OperatorError> {
        let text = input.message.as_text().ok_or_else(|| {
            OperatorError::NonRetryable(
                "SweepCycleOperator: input.message must be JSON text".into(),
            )
        })?;

        let decisions: Vec<SweepDecision> = serde_json::from_str(text).map_err(|e| {
            OperatorError::NonRetryable(format!(
                "SweepCycleOperator: failed to parse Vec<SweepDecision>: {e}"
            ))
        })?;

        let report = run_sweep_cycle(
            self.orch.as_ref(),
            self.state.as_ref(),
            self.budget.as_ref(),
            self.trace.as_ref(),
            &self.compare_operator,
            &self.synthesis_operator,
            self.budget_total_usd,
            decisions,
        )
        .await
        .map_err(|e| OperatorError::NonRetryable(e.to_string()))?;

        let report_json = serde_json::to_string(&report).map_err(|e| {
            OperatorError::NonRetryable(format!(
                "SweepCycleOperator: failed to serialize CycleReport: {e}"
            ))
        })?;

        Ok(OperatorOutput::new(
            Content::text(report_json),
            ExitReason::Complete,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use layer0::dispatch::{DispatchEvent, DispatchHandle};
    use layer0::DispatchId;

    use layer0::dispatch::Dispatcher;
    use async_trait::async_trait;
    use layer0::{
        OperatorId, Content, ExitReason, OperatorInput, OperatorOutput, OrchError,
        Scope,
    };
    use layer0::test_utils::InMemoryStore;
    use skg_orch_compose::budget::{CapPolicy, NoLimitPolicy};
    use skg_orch_compose::{BudgetTracker, CompositionTrace, ScopedStateView};

    use super::*;
    use crate::provider::ResearchResult;
    use crate::synthesis::SynthesisReport;
    use crate::types::{EvidenceStance, EvidenceItem, ProcessorTier, VerdictStatus};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_state() -> ScopedStateView {
        ScopedStateView::new(
            Arc::new(InMemoryStore::new()),
            Scope::Custom("sweep".into()),
        )
    }

    fn dummy_result() -> ResearchResult {
        ResearchResult {
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
                title: String::new(),
                summary: "Supports decision".into(),
                stance: EvidenceStance::Supporting,
                retrieved_at: "2026-03-04T00:00:00Z".into(),
            }],
            narrative: "Confirmed by research".to_string(),
            proposed_diff: None,
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),    
        }
    }

    fn dummy_synthesis_report() -> SynthesisReport {
        SynthesisReport {
            structural_changes: vec![],
            candidates: vec![],
            relationship_updates: vec![],
            health_summary: "Framework stable".to_string(),
            cost_usd: 0.05,
            cycle_id: "test-cycle".to_string(),
            completed_at: "2026-03-04T18:00:00Z".to_string(),
        }
    }

    /// Mock dispatcher: returns canned verdicts for compare dispatches and
    /// a canned synthesis report for synthesis dispatches. Tracks whether
    /// synthesis was dispatched via an atomic flag.
    struct MockDispatcher {
        #[allow(dead_code)] // used only to construct; dispatch routes by synthesis_operator
        compare_operator: OperatorId,
        synthesis_operator: OperatorId,
        compare_response: String,
        synthesis_response: String,
        synthesis_dispatched: Arc<AtomicBool>,
    }

    impl MockDispatcher {
        fn new(
            compare_operator: OperatorId,
            synthesis_operator: OperatorId,
            verdict: &SweepVerdict,
            synthesis: &SynthesisReport,
        ) -> (Self, Arc<AtomicBool>) {
            let flag = Arc::new(AtomicBool::new(false));
            let orch = Self {
                compare_operator: compare_operator.clone(),
                synthesis_operator: synthesis_operator.clone(),
                compare_response: serde_json::to_string(verdict).expect("serialize verdict"),
                synthesis_response: serde_json::to_string(synthesis)
                    .expect("serialize synthesis"),
                synthesis_dispatched: Arc::clone(&flag),
            };
            (orch, flag)
        }
    }

    #[async_trait]
    impl Dispatcher for MockDispatcher {
        async fn dispatch(
            &self,
            operator: &OperatorId,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let json = if *operator == self.synthesis_operator {
                self.synthesis_dispatched.store(true, Ordering::SeqCst);
                self.synthesis_response.clone()
            } else {
                self.compare_response.clone()
            };
            let output = OperatorOutput::new(
                Content::text(json),
                ExitReason::Complete,
            );
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("test"));
            tokio::spawn(async move {
                let _ = sender.send(DispatchEvent::Completed { output }).await;
            });
            Ok(handle)
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// CapPolicy(0, 0) denies at 0 USD spent, so every decision must be
    /// collected in `skipped` and no verdicts should be produced.
    #[tokio::test]
    async fn run_sweep_cycle_skips_when_budget_denied() {
        let compare_operator = OperatorId::new("compare");
        let synthesis_operator = OperatorId::new("synthesis");
        let verdict = dummy_verdict("d1");
        let synthesis = dummy_synthesis_report();

        let (orch, _flag) = MockDispatcher::new(
            compare_operator.clone(),
            synthesis_operator.clone(),
            &verdict,
            &synthesis,
        );

        let state = make_state();
        // CapPolicy(0, 0): total_usd (0.0) >= hard_cap (0.0) → Deny.
        let budget = BudgetTracker::new(CapPolicy {
            soft_cap: 0.0,
            hard_cap: 0.0,
        });
        let trace = CompositionTrace::new(10);

        let decisions = vec![
            SweepDecision {
                id: "d1".to_string(),
                research_results: vec![dummy_result()],
                previous_verdict: None,
            },
            SweepDecision {
                id: "d2".to_string(),
                research_results: vec![dummy_result()],
                previous_verdict: None,
            },
        ];

        let report = run_sweep_cycle(
            &orch,
            &state,
            &budget,
            &trace,
            &compare_operator,
            &synthesis_operator,
            10.0,
            decisions,
        )
        .await
        .expect("cycle must not error on budget denial");

        assert!(
            report.verdicts.is_empty(),
            "no verdicts should be produced when budget is denied"
        );
        assert_eq!(
            report.skipped.len(),
            2,
            "both decisions should be skipped"
        );
        assert!(report.synthesis.is_none());
    }

    /// A decision with empty research_results should produce a
    /// `Confirmed(0.3)` verdict without dispatching the compare operator.
    #[tokio::test]
    async fn run_sweep_cycle_skips_empty_research() {
        let compare_operator = OperatorId::new("compare");
        let synthesis_operator = OperatorId::new("synthesis");
        let verdict = dummy_verdict("d1");
        let synthesis = dummy_synthesis_report();

        let (orch, _flag) = MockDispatcher::new(
            compare_operator.clone(),
            synthesis_operator.clone(),
            &verdict,
            &synthesis,
        );

        let state = make_state();
        let budget = BudgetTracker::new(NoLimitPolicy);
        let trace = CompositionTrace::new(10);

        let decisions = vec![SweepDecision {
            id: "d1".to_string(),
            research_results: vec![],
            previous_verdict: None,
        }];

        let report = run_sweep_cycle(
            &orch,
            &state,
            &budget,
            &trace,
            &compare_operator,
            &synthesis_operator,
            100.0,
            decisions,
        )
        .await
        .expect("cycle must succeed");

        assert_eq!(report.verdicts.len(), 1, "one verdict should be produced");
        assert_eq!(report.verdicts[0].status, VerdictStatus::Confirmed);
        assert!(
            (report.verdicts[0].confidence - 0.3).abs() < f64::EPSILON,
            "confidence should be 0.3, got {}",
            report.verdicts[0].confidence
        );
        assert!(
            report.skipped.is_empty(),
            "nothing should be in skipped for empty-research short-circuit"
        );
        assert!(report.synthesis.is_none(), "synthesis needs >= 3 verdicts");
    }

    /// When research results are present and budget allows, the compare agent
    /// is dispatched and the returned verdict collected in the report.
    #[tokio::test]
    async fn run_sweep_cycle_dispatches_compare() {
        let compare_operator = OperatorId::new("compare");
        let synthesis_operator = OperatorId::new("synthesis");
        let expected_verdict = dummy_verdict("d1");
        let synthesis = dummy_synthesis_report();

        let (orch, _flag) = MockDispatcher::new(
            compare_operator.clone(),
            synthesis_operator.clone(),
            &expected_verdict,
            &synthesis,
        );

        let state = make_state();
        let budget = BudgetTracker::new(NoLimitPolicy);
        let trace = CompositionTrace::new(10);

        let decisions = vec![SweepDecision {
            id: "d1".to_string(),
            research_results: vec![dummy_result()],
            previous_verdict: None,
        }];

        let report = run_sweep_cycle(
            &orch,
            &state,
            &budget,
            &trace,
            &compare_operator,
            &synthesis_operator,
            100.0,
            decisions,
        )
        .await
        .expect("cycle must succeed");

        assert_eq!(report.verdicts.len(), 1);
        assert_eq!(report.verdicts[0].decision_id, "d1");
        assert_eq!(report.verdicts[0].status, VerdictStatus::Confirmed);
        assert!(report.skipped.is_empty());
        // Synthesis should not trigger for a single verdict.
        assert!(report.synthesis.is_none());
    }

    /// When ≥ 3 verdicts are collected, the synthesis agent must be dispatched
    /// and the synthesis report attached to the cycle report.
    #[tokio::test]
    async fn run_sweep_cycle_triggers_synthesis_at_threshold() {
        let compare_operator = OperatorId::new("compare");
        let synthesis_operator = OperatorId::new("synthesis");
        let verdict = dummy_verdict("dx");
        let synthesis = dummy_synthesis_report();

        let (orch, synthesis_dispatched) = MockDispatcher::new(
            compare_operator.clone(),
            synthesis_operator.clone(),
            &verdict,
            &synthesis,
        );

        let state = make_state();
        let budget = BudgetTracker::new(NoLimitPolicy);
        let trace = CompositionTrace::new(10);

        let decisions = vec![
            SweepDecision {
                id: "d1".to_string(),
                research_results: vec![dummy_result()],
                previous_verdict: None,
            },
            SweepDecision {
                id: "d2".to_string(),
                research_results: vec![dummy_result()],
                previous_verdict: None,
            },
            SweepDecision {
                id: "d3".to_string(),
                research_results: vec![dummy_result()],
                previous_verdict: None,
            },
        ];

        let report = run_sweep_cycle(
            &orch,
            &state,
            &budget,
            &trace,
            &compare_operator,
            &synthesis_operator,
            100.0,
            decisions,
        )
        .await
        .expect("cycle must succeed");

        assert_eq!(report.verdicts.len(), 3, "all three decisions should produce verdicts");
        assert!(
            synthesis_dispatched.load(Ordering::SeqCst),
            "synthesis agent must be dispatched when >= 3 verdicts are collected"
        );
        assert!(
            report.synthesis.is_some(),
            "synthesis report must be present in the cycle report"
        );
    }
}
