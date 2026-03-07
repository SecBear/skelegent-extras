//! Sweep cycle orchestration.
//!
//! The central function is [`run_cycle`], which pops decisions from a priority
//! queue, enforces the budget, bounds concurrency, and collects verdicts into a
//! [`CycleReport`].
//!
//! The operator is intentionally injected as a generic closure so the cycle
//! logic can be tested independently of any concrete [`neuron_op_sweep`]
//! provider or HTTP client.

use std::collections::BinaryHeap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

use neuron_op_sweep::{CompareConfig, SweepVerdict, VerdictStatus};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use crate::budget::{BudgetConfig, BudgetState, DegradationLevel};
use crate::priority::QueuedDecision;

/// Top-level orchestrator configuration.
///
/// This struct is not serializable because [`CompareConfig`] and
/// [`Duration`] do not implement [`serde::Serialize`]. Persist it as source
/// code or a custom config file.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// How often to trigger a sweep cycle. Default: 4 hours.
    pub cycle_interval: Duration,
    /// Path to the sweep-config file in the golden repository.
    pub sweep_config_path: PathBuf,
    /// Path to the `neuron-state-sqlite` database used for budget persistence.
    pub state_db_path: PathBuf,
    /// Budget enforcement configuration.
    pub budget: BudgetConfig,
    /// Sweep operator configuration forwarded to each operator invocation.
    pub operator: CompareConfig,
}

/// Summary produced at the end of each sweep cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleReport {
    /// All verdicts collected during this cycle, in completion order.
    pub verdicts: Vec<SweepVerdict>,
    /// Budget state at the moment the cycle completed.
    pub budget: BudgetState,
    /// `true` when every queued decision was either swept or skipped due to
    /// degraded-mode staleness filtering — i.e., no decision was left unvisited
    /// because of a hard stop or budget exhaustion.
    pub all_swept: bool,
    /// RFC 3339 timestamp of when the cycle started.
    pub started_at: String,
    /// RFC 3339 timestamp of when the cycle completed.
    pub completed_at: String,
    /// Sum of all verdict costs accumulated in this cycle (USD).
    pub total_cost_usd: f64,
}

/// Rate limits for external API calls.
///
/// These values are advisory — the orchestrator reads them to throttle work
/// submission. Actual enforcement is delegated to the operator and its retry
/// logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiter {
    /// Maximum requests per minute to the Parallel.ai research API. Default: 30.
    pub parallel_rpm: usize,
    /// Maximum requests per minute to the GitHub REST API. Default: 60.
    pub github_rpm: usize,
    /// Maximum content-generating requests per minute to the GitHub API.
    ///
    /// GitHub's secondary rate limit is 80 RPM; this default of 20 provides
    /// a generous safety margin.
    pub github_content_rpm: usize,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self {
            parallel_rpm: 30,
            github_rpm: 60,
            github_content_rpm: 20,
        }
    }
}

/// Execute one sweep cycle against the provided decision queue.
///
/// # Algorithm (spec §5.4)
///
/// 1. Pop the highest-priority decision from a [`BinaryHeap`].
/// 2. Check the budget degradation level:
///    - [`DegradationLevel::HardStop`] → break, no more dispatches.
///    - [`DegradationLevel::Degraded`] → skip `Confirmed` decisions with
///      staleness below [`BudgetConfig::degraded_min_staleness_days`].
/// 3. Pre-flight cost check: if the estimated cost would breach the daily cap,
///    break immediately.
/// 4. If `max_parallel` tasks are already running, drain one result before
///    dispatching the next.
/// 5. Call `operator(decision_id, previous_verdict)` and spawn the returned
///    future.
/// 6. After the queue is exhausted, drain all remaining active tasks.
/// 7. Return a [`CycleReport`] with verdicts, final budget state, and totals.
///
/// # Panics from the operator
///
/// If a spawned operator future panics, [`JoinSet::join_next`] returns an
/// error. The panic is caught and that decision's verdict is dropped; the
/// cycle continues with remaining tasks (spec §5.8).
///
/// # Type parameter
///
/// `F` must produce a `'static + Send` future so it can be handed to
/// [`tokio::spawn`]. Capture all needed state in the closure via [`Arc`] or
/// owned values.
///
/// [`Arc`]: std::sync::Arc
pub async fn run_cycle<F>(
    config: &OrchestratorConfig,
    mut budget: BudgetState,
    decisions: Vec<QueuedDecision>,
    operator: F,
) -> CycleReport
where
    F: Fn(
        String,
        Option<VerdictStatus>,
    ) -> Pin<Box<dyn Future<Output = SweepVerdict> + Send + 'static>>,
{
    let started_at = chrono::Utc::now().to_rfc3339();
    let total = decisions.len();

    // Build max-heap: highest priority pops first.
    let mut heap: BinaryHeap<QueuedDecision> = decisions.into_iter().collect();

    let mut verdicts: Vec<SweepVerdict> = Vec::new();
    let mut active: JoinSet<SweepVerdict> = JoinSet::new();
    let mut skipped_degraded: usize = 0;

    'outer: while let Some(decision) = heap.pop() {
        // --- 1. Budget degradation check ---
        match budget.degradation_mode(&config.budget) {
            DegradationLevel::HardStop => break 'outer,
            DegradationLevel::Degraded => {
                if decision.previous_verdict == Some(VerdictStatus::Confirmed)
                    && decision.staleness_days < config.budget.degraded_min_staleness_days
                {
                    skipped_degraded += 1;
                    continue 'outer;
                }
            }
            DegradationLevel::Normal => {}
        }

        // --- 2. Pre-flight cost check ---
        let estimated = budget.estimate_cost(&decision.decision_id);
        if budget.daily_total_usd + estimated > config.budget.daily_cap_usd {
            break 'outer;
        }

        // --- 3. Concurrency limit: drain one task when at capacity ---
        while active.len() >= config.budget.max_parallel {
            if let Some(result) = active.join_next().await {
                match result {
                    Ok(verdict) => {
                        budget.daily_total_usd += verdict.cost_usd;
                        budget
                            .last_costs
                            .insert(verdict.decision_id.clone(), verdict.cost_usd);
                        verdicts.push(verdict);
                    }
                    Err(_join_err) => {
                        // Operator panicked — log would go here; continue per spec §5.8.
                    }
                }
            }
        }

        // --- 4. Dispatch ---
        let future = operator(
            decision.decision_id.clone(),
            decision.previous_verdict.clone(),
        );
        active.spawn(future);
    }

    // --- 5. Drain all remaining active tasks ---
    while let Some(result) = active.join_next().await {
        match result {
            Ok(verdict) => {
                budget.daily_total_usd += verdict.cost_usd;
                budget
                    .last_costs
                    .insert(verdict.decision_id.clone(), verdict.cost_usd);
                verdicts.push(verdict);
            }
            Err(_join_err) => {
                // Operator panicked — continue per spec §5.8.
            }
        }
    }

    let total_cost_usd: f64 = verdicts.iter().map(|v| v.cost_usd).sum();
    // all_swept is true when every decision was either swept or skipped via
    // degraded-mode filtering.  A hard stop or budget break leaves decisions
    // unvisited, setting this to false.
    let all_swept = verdicts.len() + skipped_degraded == total;
    let completed_at = chrono::Utc::now().to_rfc3339();

    CycleReport {
        verdicts,
        budget,
        all_swept,
        started_at,
        completed_at,
        total_cost_usd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuron_op_sweep::{EvidenceStance, ProcessorTier};
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_config() -> OrchestratorConfig {
        OrchestratorConfig {
            cycle_interval: Duration::from_secs(4 * 3600),
            sweep_config_path: PathBuf::from("/tmp/sweep.md"),
            state_db_path: PathBuf::from("/tmp/state.db"),
            budget: BudgetConfig::default(),
            operator: CompareConfig::default(),
        }
    }

    fn fresh_budget() -> BudgetState {
        BudgetState {
            daily_total_usd: 0.0,
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            last_costs: HashMap::new(),
        }
    }

    fn make_decision(id: &str, priority: f64) -> QueuedDecision {
        QueuedDecision {
            decision_id: id.to_string(),
            priority,
            staleness_days: 30.0,
            previous_verdict: None,
            estimated_cost_usd: 0.10,
        }
    }

    fn confirmed_verdict(id: &str, cost: f64) -> SweepVerdict {
        SweepVerdict {
            decision_id: id.to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.9,
            num_supporting: 3,
            num_contradicting: 0,
            cost_usd: cost,
            processor: ProcessorTier::Base,
            duration_secs: 1.0,
            swept_at: chrono::Utc::now().to_rfc3339(),
            evidence: vec![],
            narrative: "Confirmed".to_string(),
            proposed_diff: None,
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),
        }
    }

    /// Simple mock operator that returns a fixed Confirmed verdict.
    fn mock_operator(
        cost_per_sweep: f64,
    ) -> impl Fn(
        String,
        Option<VerdictStatus>,
    ) -> Pin<Box<dyn Future<Output = SweepVerdict> + Send + 'static>> {
        move |id: String, _prev: Option<VerdictStatus>| {
            let verdict = confirmed_verdict(&id, cost_per_sweep);
            Box::pin(async move { verdict })
        }
    }

    // -----------------------------------------------------------------------
    // CycleReport serde round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn cycle_report_serde_round_trip() {
        let report = CycleReport {
            verdicts: vec![confirmed_verdict("topic-1", 0.10)],
            budget: fresh_budget(),
            all_swept: true,
            started_at: "2026-03-04T18:00:00Z".to_string(),
            completed_at: "2026-03-04T18:05:00Z".to_string(),
            total_cost_usd: 0.10,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let back: CycleReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.verdicts.len(), 1);
        assert_eq!(back.verdicts[0].decision_id, "topic-1");
        assert!(back.all_swept);
        assert!((back.total_cost_usd - 0.10).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // RateLimiter
    // -----------------------------------------------------------------------

    #[test]
    fn rate_limiter_defaults() {
        let rl = RateLimiter::default();
        assert_eq!(rl.parallel_rpm, 30);
        assert_eq!(rl.github_rpm, 60);
        assert_eq!(rl.github_content_rpm, 20);
    }

    #[test]
    fn rate_limiter_serde_round_trip() {
        let rl = RateLimiter::default();
        let json = serde_json::to_string(&rl).expect("serialize");
        let back: RateLimiter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.parallel_rpm, rl.parallel_rpm);
        assert_eq!(back.github_content_rpm, rl.github_content_rpm);
    }

    // -----------------------------------------------------------------------
    // run_cycle — integration tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn integration_five_decisions_all_swept() {
        let config = default_config();
        let budget = fresh_budget();

        let decisions = vec![
            make_decision("topic-1", 0.9),
            make_decision("topic-2", 0.7),
            make_decision("topic-3", 0.5),
            make_decision("topic-4", 0.4),
            make_decision("topic-5", 0.2),
        ];

        let report = run_cycle(&config, budget, decisions, mock_operator(0.10)).await;

        assert_eq!(
            report.verdicts.len(),
            5,
            "all 5 decisions must produce verdicts"
        );
        assert!(
            report.all_swept,
            "all_swept must be true when nothing was halted early"
        );
        let expected_total = 5.0 * 0.10;
        assert!(
            (report.total_cost_usd - expected_total).abs() < 1e-9,
            "total cost must equal sum of verdict costs: {:.4} vs {:.4}",
            report.total_cost_usd,
            expected_total
        );
        // Budget state must reflect accumulated spend
        assert!(
            (report.budget.daily_total_usd - expected_total).abs() < 1e-9,
            "budget daily_total_usd must be updated"
        );
    }

    #[tokio::test]
    async fn integration_empty_decision_queue() {
        let config = default_config();
        let budget = fresh_budget();
        let report = run_cycle(&config, budget, vec![], mock_operator(0.10)).await;
        assert_eq!(report.verdicts.len(), 0);
        assert!(report.all_swept, "vacuously all swept when queue is empty");
        assert_eq!(report.total_cost_usd, 0.0);
    }

    #[tokio::test]
    async fn hard_stop_halts_remaining_decisions() {
        // With $9.60 spent and a 95% hard-stop at $9.50, the first degradation
        // check should trigger HardStop and dispatch nothing.
        let config = OrchestratorConfig {
            budget: BudgetConfig {
                daily_cap_usd: 10.0,
                hard_stop_threshold: 0.95,
                ..BudgetConfig::default()
            },
            ..default_config()
        };
        let budget = BudgetState {
            daily_total_usd: 9.60, // 96% — hard stop territory
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            last_costs: HashMap::new(),
        };

        let decisions = vec![make_decision("topic-1", 0.9), make_decision("topic-2", 0.5)];
        let report = run_cycle(&config, budget, decisions, mock_operator(0.10)).await;

        assert_eq!(report.verdicts.len(), 0, "HardStop must dispatch nothing");
        assert!(!report.all_swept, "all_swept must be false after hard stop");
    }

    #[tokio::test]
    async fn degraded_mode_skips_confirmed_decisions_below_staleness_threshold() {
        // Spend 8.5 / 10.0 = 85% → Degraded.
        let config = OrchestratorConfig {
            budget: BudgetConfig {
                daily_cap_usd: 10.0,
                degradation_threshold: 0.80,
                hard_stop_threshold: 0.95,
                max_parallel: 3,
                degraded_min_staleness_days: 14.0,
            },
            ..default_config()
        };
        let budget = BudgetState {
            daily_total_usd: 8.5,
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            last_costs: HashMap::new(),
        };

        // One Confirmed with staleness 10 days → skipped; one with staleness 20 → swept.
        let decisions = vec![
            QueuedDecision {
                decision_id: "SKIP".to_string(),
                priority: 0.9,
                staleness_days: 10.0, // < 14 threshold
                previous_verdict: Some(VerdictStatus::Confirmed),
                estimated_cost_usd: 0.10,
            },
            QueuedDecision {
                decision_id: "SWEEP".to_string(),
                priority: 0.5,
                staleness_days: 20.0, // >= 14 threshold
                previous_verdict: Some(VerdictStatus::Confirmed),
                estimated_cost_usd: 0.10,
            },
        ];

        let report = run_cycle(&config, budget, decisions, mock_operator(0.10)).await;

        // SKIP was filtered; SWEEP was dispatched.
        assert_eq!(report.verdicts.len(), 1);
        assert_eq!(report.verdicts[0].decision_id, "SWEEP");
        // 1 swept + 1 skipped_degraded == 2 total → all_swept
        assert!(report.all_swept);
    }

    #[tokio::test]
    async fn budget_preflight_halts_when_next_cost_would_breach_cap() {
        // Only $0.05 remaining — cannot afford a $0.30-default sweep.
        let config = OrchestratorConfig {
            budget: BudgetConfig {
                daily_cap_usd: 10.0,
                ..BudgetConfig::default()
            },
            ..default_config()
        };
        let budget = BudgetState {
            daily_total_usd: 9.95,
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            last_costs: HashMap::new(), // no history → estimate_cost defaults to 0.30
        };

        let decisions = vec![make_decision("topic-1", 0.9)];
        let report = run_cycle(&config, budget, decisions, mock_operator(0.10)).await;

        assert_eq!(
            report.verdicts.len(),
            0,
            "pre-flight check must block dispatch"
        );
        assert!(!report.all_swept);
    }

    #[tokio::test]
    async fn operator_panic_is_caught_cycle_continues() {
        // Two decisions: first operator panics, second succeeds.
        // We need max_parallel=1 to guarantee ordering so the panic is encountered.
        let config = OrchestratorConfig {
            budget: BudgetConfig {
                max_parallel: 1,
                ..BudgetConfig::default()
            },
            ..default_config()
        };
        let budget = fresh_budget();

        let decisions = vec![make_decision("PANIC", 0.9), make_decision("OK", 0.5)];

        let op = |id: String,
                  _: Option<VerdictStatus>|
         -> Pin<Box<dyn Future<Output = SweepVerdict> + Send + 'static>> {
            Box::pin(async move {
                if id == "PANIC" {
                    panic!("simulated operator panic");
                }
                confirmed_verdict(&id, 0.10)
            })
        };

        let report = run_cycle(&config, budget, decisions, op).await;

        // The panic is swallowed; the "OK" decision should have a verdict.
        assert_eq!(
            report.verdicts.len(),
            1,
            "only the non-panicking decision should produce a verdict"
        );
        assert_eq!(report.verdicts[0].decision_id, "OK");
        // all_swept is false because the panic dropped one verdict (only 1 of 2 recorded)
        assert!(
            !report.all_swept,
            "a panicked operator means not all decisions were covered"
        );
    }

    #[tokio::test]
    async fn priority_queue_processes_highest_priority_first_when_sequential() {
        // Use max_parallel=1 to guarantee strict ordering.
        let config = OrchestratorConfig {
            budget: BudgetConfig {
                max_parallel: 1,
                ..BudgetConfig::default()
            },
            ..default_config()
        };
        let budget = fresh_budget();

        // Add decisions in reverse priority order to verify the heap re-sorts.
        let decisions = vec![
            make_decision("LOW", 0.1),
            make_decision("HIGH", 0.9),
            make_decision("MID", 0.5),
        ];

        use std::sync::{Arc, Mutex};
        let order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let order_clone = Arc::clone(&order);

        let op = move |id: String,
                       _: Option<VerdictStatus>|
              -> Pin<Box<dyn Future<Output = SweepVerdict> + Send + 'static>> {
            let tracker = Arc::clone(&order_clone);
            Box::pin(async move {
                tracker.lock().unwrap().push(id.clone());
                confirmed_verdict(&id, 0.10)
            })
        };

        run_cycle(&config, budget, decisions, op).await;

        let seen = order.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec!["HIGH", "MID", "LOW"],
            "decisions must be processed highest-priority first"
        );
    }

    // -----------------------------------------------------------------------
    // EvidenceStance import smoke-test (ensures neuron-op-sweep re-export works)
    // -----------------------------------------------------------------------

    #[test]
    fn evidence_stance_accessible_from_cycle_tests() {
        let _s = EvidenceStance::Supporting;
        let _c = EvidenceStance::Contradicting;
        let _n = EvidenceStance::Neutral;
    }
}
