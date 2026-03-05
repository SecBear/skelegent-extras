//! End-to-end smoke test: orchestrator → operator → mock provider → effects.
//!
//! Validates that the full sweep pipeline completes with a realistic
//! decision queue, producing correct verdicts and PR routing decisions.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use layer0::effect::Scope;
use layer0::error::StateError;
use layer0::state::{SearchResult, StateStore, StoreOptions};
use neuron_effects_git::{PrAction, route_verdict};
use neuron_op_sweep::{
    EvidenceItem, EvidenceStance, ProcessorTier, ResearchProvider, ResearchResult, SweepError,
    SweepOperator, SweepOperatorConfig, SweepVerdict, VerdictStatus,
};
use neuron_orch_sweep::{
    BudgetConfig, BudgetState, CycleReport, OrchestratorConfig, QueuedDecision, run_cycle,
};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Recording StateStore — captures writes for assertion
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RecordingStore {
    writes: Mutex<Vec<(String, serde_json::Value)>>,
}

#[async_trait]
impl StateStore for RecordingStore {
    async fn read(&self, _s: &Scope, _k: &str) -> Result<Option<serde_json::Value>, StateError> {
        Ok(None)
    }
    async fn write(&self, _s: &Scope, k: &str, v: serde_json::Value) -> Result<(), StateError> {
        self.writes.lock().await.push((k.to_string(), v));
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
    async fn write_hinted(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
        _options: &StoreOptions,
    ) -> Result<(), StateError> {
        self.write(scope, key, value).await
    }
}

// ---------------------------------------------------------------------------
// Test provider — returns realistic verdicts based on decision ID
// ---------------------------------------------------------------------------

struct TestProvider;

#[async_trait]
impl ResearchProvider for TestProvider {
    async fn search(
        &self,
        _query: &str,
        _processor: ProcessorTier,
    ) -> Result<Vec<ResearchResult>, SweepError> {
        Ok(vec![ResearchResult {
            url: "https://example.com/paper".to_string(),
            title: "Agent Architecture 2026".to_string(),
            snippet: "Key findings about agent systems.".to_string(),
            retrieved_at: chrono::Utc::now().to_rfc3339(),
        }])
    }

    async fn compare(
        &self,
        _system: &str,
        _context: &str,
        _research: &str,
        current_decision: &str,
    ) -> Result<SweepVerdict, SweepError> {
        // Return different verdicts based on decision ID to test routing
        let (status, confidence) = if current_decision.contains("STALE") {
            (VerdictStatus::Challenged, 0.7)
        } else if current_decision.contains("UPDATE") {
            (VerdictStatus::Refined, 0.8)
        } else {
            (VerdictStatus::Confirmed, 0.95)
        };
        let num_contradicting = if status == VerdictStatus::Challenged {
            2
        } else {
            0
        };

        Ok(SweepVerdict {
            decision_id: current_decision.to_string(),
            status,
            confidence,
            num_supporting: 3,
            num_contradicting,
            cost_usd: 0.05,
            processor: ProcessorTier::Base,
            duration_secs: 0.5,
            swept_at: chrono::Utc::now().to_rfc3339(),
            evidence: vec![EvidenceItem {
                source_url: "https://example.com".into(),
                summary: "Test evidence".into(),
                stance: EvidenceStance::Supporting,
                retrieved_at: chrono::Utc::now().to_rfc3339(),
            }],
            narrative: "Test narrative".into(),
            proposed_diff: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

fn default_orch_config() -> OrchestratorConfig {
    OrchestratorConfig {
        cycle_interval: Duration::from_secs(4 * 3600),
        sweep_config_path: "/tmp/sweep.md".into(),
        state_db_path: "/tmp/state.db".into(),
        budget: BudgetConfig::default(),
        operator: SweepOperatorConfig {
            min_sweep_interval: Duration::from_secs(0), // disable dedup for test
            ..SweepOperatorConfig::default()
        },
    }
}

#[tokio::test]
async fn smoke_full_cycle_with_three_decisions() {
    let store = Arc::new(RecordingStore::default());
    let config = default_orch_config();
    let budget = BudgetState {
        daily_total_usd: 0.0,
        date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
        last_costs: HashMap::new(),
    };

    // Three decisions at different priorities
    let decisions = vec![
        QueuedDecision {
            decision_id: "topic-1-confirm".to_string(),
            priority: 0.9,
            staleness_days: 30.0,
            previous_verdict: None,
            estimated_cost_usd: 0.10,
        },
        QueuedDecision {
            decision_id: "topic-2-update".to_string(),
            priority: 0.7,
            staleness_days: 20.0,
            previous_verdict: Some(VerdictStatus::Refined),
            estimated_cost_usd: 0.10,
        },
        QueuedDecision {
            decision_id: "topic-3-stale".to_string(),
            priority: 0.5,
            staleness_days: 45.0,
            previous_verdict: Some(VerdictStatus::Challenged),
            estimated_cost_usd: 0.10,
        },
    ];

    // Build the operator closure that uses real SweepOperator + TestProvider + RecordingStore
    let store_clone = Arc::clone(&store);
    let operator =
        move |id: String,
              prev: Option<VerdictStatus>|
              -> Pin<Box<dyn std::future::Future<Output = SweepVerdict> + Send + 'static>> {
            let s = Arc::clone(&store_clone);
            Box::pin(async move {
                let op = SweepOperator::new(
                    SweepOperatorConfig {
                        min_sweep_interval: Duration::from_secs(0),
                        ..SweepOperatorConfig::default()
                    },
                    Box::new(TestProvider),
                );
                op.run(&id, prev, 8.0, 10.0, s.as_ref())
                    .await
                    .unwrap_or_else(|e| panic!("operator failed for {id}: {e}"))
            })
        };

    let report: CycleReport = run_cycle(&config, budget, decisions, operator).await;

    // --- Assertions ---

    // All 3 decisions were swept
    assert_eq!(
        report.verdicts.len(),
        3,
        "all decisions should produce verdicts"
    );
    assert!(report.all_swept, "all_swept should be true");

    // Verdicts have correct decision IDs (order is completion order, not priority)
    let ids: Vec<&str> = report
        .verdicts
        .iter()
        .map(|v| v.decision_id.as_str())
        .collect();
    assert!(ids.contains(&"topic-1-confirm"), "topic-1 should be in verdicts");
    assert!(ids.contains(&"topic-2-update"), "topic-2 should be in verdicts");
    assert!(ids.contains(&"topic-3-stale"), "topic-3 should be in verdicts");

    // Total cost should be positive
    assert!(report.total_cost_usd > 0.0, "total cost should be positive");

    // Budget should be updated
    assert!(
        report.budget.daily_total_usd > 0.0,
        "budget should reflect spending"
    );

    // Verify PR routing for each verdict
    for verdict in &report.verdicts {
        let action = route_verdict(&verdict.status);
        match verdict.status {
            VerdictStatus::Confirmed => assert_eq!(action, PrAction::AutoMerged),
            VerdictStatus::Refined => assert_eq!(action, PrAction::Created),
            VerdictStatus::Challenged => assert_eq!(action, PrAction::Created),
            _ => {}
        }
    }

    // Verify the RecordingStore captured writes from steps 5 and 7
    let writes = store.writes.lock().await;
    assert!(
        !writes.is_empty(),
        "StateStore should have received writes from steps 5 and 7"
    );

    // Check for artifact writes (step 5)
    let artifact_writes: Vec<_> = writes
        .iter()
        .filter(|(k, _)| k.starts_with("artifact:"))
        .collect();
    assert!(
        !artifact_writes.is_empty(),
        "step 5 should have stored research artifacts"
    );

    // Check for metadata writes (step 7)
    let meta_writes: Vec<_> = writes
        .iter()
        .filter(|(k, _)| k.starts_with("meta:"))
        .collect();
    assert!(
        !meta_writes.is_empty(),
        "step 7 should have stored sweep metadata"
    );

    // Check for delta writes (step 7)
    let delta_writes: Vec<_> = writes
        .iter()
        .filter(|(k, _)| k.starts_with("delta:"))
        .collect();
    assert!(
        !delta_writes.is_empty(),
        "step 7 should have stored delta summaries"
    );

    // 3 decisions × (1 artifact + 1 meta + 1 delta) = at least 9 writes
    assert!(
        writes.len() >= 9,
        "expected at least 9 writes (3 decisions × 3 write types), got {}",
        writes.len()
    );
}

#[tokio::test]
async fn smoke_budget_exhaustion_stops_cycle() {
    let store = Arc::new(RecordingStore::default());
    let config = OrchestratorConfig {
        budget: BudgetConfig {
            daily_cap_usd: 0.10,
            hard_stop_threshold: 0.95,
            ..BudgetConfig::default()
        },
        ..default_orch_config()
    };

    // Pre-exhaust 96% of budget
    let budget = BudgetState {
        daily_total_usd: 0.096,
        date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
        last_costs: HashMap::new(),
    };

    let decisions = vec![QueuedDecision {
        decision_id: "WONT-RUN".to_string(),
        priority: 0.9,
        staleness_days: 30.0,
        previous_verdict: None,
        estimated_cost_usd: 0.10,
    }];

    let store_clone = Arc::clone(&store);
    let operator =
        move |id: String,
              prev: Option<VerdictStatus>|
              -> Pin<Box<dyn std::future::Future<Output = SweepVerdict> + Send + 'static>> {
            let s = Arc::clone(&store_clone);
            Box::pin(async move {
                let op = SweepOperator::new(SweepOperatorConfig::default(), Box::new(TestProvider));
                op.run(&id, prev, 0.004, 0.10, s.as_ref())
                    .await
                    .unwrap_or_else(|e| panic!("operator failed: {e}"))
            })
        };

    let report = run_cycle(&config, budget, decisions, operator).await;

    assert_eq!(
        report.verdicts.len(),
        0,
        "hard stop should prevent any dispatches"
    );
    assert!(
        !report.all_swept,
        "all_swept should be false after hard stop"
    );
}
