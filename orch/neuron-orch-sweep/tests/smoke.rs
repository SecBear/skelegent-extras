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
use layer0::operator::TriggerType;
use layer0::state::{SearchResult, StateStore, StoreOptions};
use neuron_effects_git::{PrAction, route_verdict};
use layer0::id::AgentId;
use layer0::test_utils::LocalOrchestrator;
use neuron_op_sweep::{
    CompareOperator, CompareConfig, EvidenceItem, EvidenceStance, ProcessorTier, ResearchResult,
    SweepVerdict, VerdictStatus, CompareInput,
};
use neuron_orch_kit::{dispatch_typed, ScopedStateView};
use neuron_orch_sweep::{
    BudgetConfig, BudgetState, CycleReport, OrchestratorConfig, QueuedDecision, run_cycle,
};
use neuron_turn::provider::{Provider, ProviderError};
use neuron_turn::types::{ContentPart, ProviderRequest, ProviderResponse, StopReason, TokenUsage};
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
// Test helper — returns a single research result for use in tests
// ---------------------------------------------------------------------------

fn test_research_result() -> ResearchResult {
    ResearchResult {
        url: "https://example.com/paper".to_string(),
        title: "Agent Architecture 2026".to_string(),
        snippet: "Key findings about agent systems.".to_string(),
        retrieved_at: chrono::Utc::now().to_rfc3339(),
    }
}

// ---------------------------------------------------------------------------
// Test LLM — produces compare verdicts based on decision ID in the request
// ---------------------------------------------------------------------------

struct TestLlm;

impl Provider for TestLlm {
    fn complete(
        &self,
        req: ProviderRequest,
    ) -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>> + Send {
        // CompareOperator formats: "Decision ID: {id}\n\n<research>..."
        let decision_id = req
            .messages
            .first()
            .and_then(|m| m.content.first())
            .and_then(|c| {
                if let ContentPart::Text { text } = c {
                    text.strip_prefix("Decision ID: ")
                        .and_then(|s| s.split('\n').next())
                        .map(str::to_string)
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let (status, confidence) = if decision_id.contains("STALE") {
            (VerdictStatus::Challenged, 0.7)
        } else if decision_id.contains("UPDATE") {
            (VerdictStatus::Refined, 0.8)
        } else {
            (VerdictStatus::Confirmed, 0.95)
        };
        let num_contradicting = if status == VerdictStatus::Challenged { 2 } else { 0 };
        let verdict = SweepVerdict {
            decision_id: decision_id.clone(),
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
        };
        let json = serde_json::to_string(&verdict).expect("serialize verdict");
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

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

fn default_orch_config() -> OrchestratorConfig {
    OrchestratorConfig {
        cycle_interval: Duration::from_secs(4 * 3600),
        sweep_config_path: "/tmp/sweep.md".into(),
        state_db_path: "/tmp/state.db".into(),
        budget: BudgetConfig::default(),
        operator: CompareConfig {
            min_sweep_interval: Duration::from_secs(0), // disable dedup for test
            ..CompareConfig::default()
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

    // Build the operator closure using dispatch_typed + Orchestrator (idiomatic pipeline).
    let store_clone = Arc::clone(&store);
    let operator =
        move |id: String,
              prev: Option<VerdictStatus>|
              -> Pin<Box<dyn std::future::Future<Output = SweepVerdict> + Send + 'static>> {
            let s = Arc::clone(&store_clone);
            Box::pin(async move {
                let mut orch = LocalOrchestrator::new();
                let compare_id = AgentId::new("compare");
                let cfg = CompareConfig {
                    min_sweep_interval: Duration::from_secs(0),
                    ..CompareConfig::default()
                };
                let scoped = Arc::new(ScopedStateView::new(Arc::clone(&s) as Arc<dyn StateStore>, Scope::Custom("sweep".into())));
                orch.register(compare_id.clone(), Arc::new(CompareOperator::new(TestLlm, scoped, cfg)));
                let input = CompareInput {
                    research_results: vec![test_research_result()],
                    decision_id: id.clone(),
                };
                let (verdict, _) = dispatch_typed::<CompareInput, SweepVerdict>(
                    &orch, &compare_id, input, TriggerType::Task,
                ).await.unwrap_or_else(|e| panic!("operator failed for {id}: {e}"));
                verdict
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

    // Note: The idiomatic pipeline (dispatch_typed + LocalOrchestrator) returns
    // WriteMemory effects in OperatorOutput but does NOT execute them —
    // effect interpretation is an orchestrator concern. The RecordingStore
    // receives no writes. Effect execution is tested separately via
    // OrchestratedRunner + EffectInterpreter in the kit crate.
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
                let mut orch = LocalOrchestrator::new();
                let compare_id = AgentId::new("compare");
                let cfg = CompareConfig::default();
                let scoped = Arc::new(ScopedStateView::new(Arc::clone(&s) as Arc<dyn StateStore>, Scope::Custom("sweep".into())));
                orch.register(compare_id.clone(), Arc::new(CompareOperator::new(TestLlm, scoped, cfg)));
                let input = CompareInput {
                    research_results: vec![test_research_result()],
                    decision_id: id.clone(),
                };
                let (verdict, _) = dispatch_typed::<CompareInput, SweepVerdict>(
                    &orch, &compare_id, input, TriggerType::Task,
                ).await.unwrap_or_else(|e| panic!("operator failed: {e}"));
                verdict
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
