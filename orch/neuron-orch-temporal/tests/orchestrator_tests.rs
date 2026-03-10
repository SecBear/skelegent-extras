//! Integration tests for `neuron-orch-temporal`.
//!
//! All tests run with default features only (no native deps, no cmake).

use layer0::content::Content;
use layer0::effect::SignalPayload;
use layer0::error::{OperatorError, OrchError};
use layer0::id::{OperatorId, WorkflowId};
use layer0::operator::{ExitReason, Operator, OperatorInput, OperatorOutput, TriggerType};
use layer0::orchestrator::{Orchestrator, QueryPayload};
use layer0::test_utils::EchoOperator;
use neuron_orch_temporal::{RetryPolicy, TemporalConfig, TemporalOrch};
use std::sync::Arc;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn simple_input(msg: &str) -> OperatorInput {
    OperatorInput::new(Content::text(msg), TriggerType::User)
}

// ── Config tests ─────────────────────────────────────────────────────────────

#[test]
fn temporal_config_defaults() {
    let cfg = TemporalConfig::default();
    assert_eq!(cfg.server_url, "localhost:7233");
    assert_eq!(cfg.namespace, "default");
    assert_eq!(cfg.task_queue, "neuron-worker");
    assert_eq!(cfg.identity, "");
}

#[test]
fn retry_policy_defaults() {
    let rp = RetryPolicy::default();
    assert_eq!(rp.initial_interval_ms, 1000);
    assert_eq!(rp.max_interval_ms, 60_000);
    assert_eq!(rp.max_attempts, 3);
    assert!((rp.backoff_coefficient - 2.0).abs() < f64::EPSILON);
}

#[test]
fn temporal_config_serde_round_trip() {
    let original = TemporalConfig {
        server_url: "my-server:7233".to_string(),
        namespace: "prod".to_string(),
        task_queue: "fast-queue".to_string(),
        identity: "worker-1".to_string(),
    };
    let json = serde_json::to_string(&original).expect("serialize TemporalConfig");
    let recovered: TemporalConfig = serde_json::from_str(&json).expect("deserialize TemporalConfig");
    assert_eq!(recovered.server_url, original.server_url);
    assert_eq!(recovered.namespace, original.namespace);
    assert_eq!(recovered.task_queue, original.task_queue);
    assert_eq!(recovered.identity, original.identity);
}

#[test]
fn retry_policy_serde_round_trip() {
    let original = RetryPolicy {
        initial_interval_ms: 500,
        max_interval_ms: 30_000,
        max_attempts: 5,
        backoff_coefficient: 1.5,
    };
    let json = serde_json::to_string(&original).expect("serialize RetryPolicy");
    let recovered: RetryPolicy = serde_json::from_str(&json).expect("deserialize RetryPolicy");
    assert_eq!(recovered.initial_interval_ms, original.initial_interval_ms);
    assert_eq!(recovered.max_interval_ms, original.max_interval_ms);
    assert_eq!(recovered.max_attempts, original.max_attempts);
    assert!((recovered.backoff_coefficient - original.backoff_coefficient).abs() < f64::EPSILON);
}

// ── Dispatch tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn dispatch_single_operator() {
    let mut orch = TemporalOrch::new(TemporalConfig::default());
    orch.register(OperatorId::new("echo"), Arc::new(EchoOperator));

    let output = orch
        .dispatch(&OperatorId::new("echo"), simple_input("hello"))
        .await
        .expect("dispatch should succeed");

    assert_eq!(output.message, Content::text("hello"));
    assert_eq!(output.exit_reason, ExitReason::Complete);
}

#[tokio::test]
async fn dispatch_unknown_agent_returns_error() {
    let orch = TemporalOrch::new(TemporalConfig::default());

    let err = orch
        .dispatch(&OperatorId::new("ghost"), simple_input("x"))
        .await
        .expect_err("unregistered agent must return an error");

    assert!(
        err.to_string().contains("operator not found"),
        "expected 'operator not found' in error, got: {err}"
    );
    assert!(matches!(err, OrchError::OperatorNotFound(_)));
}

#[tokio::test]
async fn dispatch_many_all_succeed() {
    let mut orch = TemporalOrch::new(TemporalConfig::default());
    orch.register(OperatorId::new("a"), Arc::new(EchoOperator));
    orch.register(OperatorId::new("b"), Arc::new(EchoOperator));

    let tasks = vec![
        (OperatorId::new("a"), simple_input("msg-a")),
        (OperatorId::new("b"), simple_input("msg-b")),
    ];
    let results = orch.dispatch_many(tasks).await;

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].as_ref().unwrap().message,
        Content::text("msg-a")
    );
    assert_eq!(
        results[1].as_ref().unwrap().message,
        Content::text("msg-b")
    );
}

#[tokio::test]
async fn dispatch_many_partial_failure() {
    let mut orch = TemporalOrch::new(TemporalConfig::default());
    orch.register(OperatorId::new("ok"), Arc::new(EchoOperator));
    // "bad" is intentionally not registered.

    let tasks = vec![
        (OperatorId::new("ok"), simple_input("fine")),
        (OperatorId::new("bad"), simple_input("boom")),
    ];
    let results = orch.dispatch_many(tasks).await;

    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok(), "known agent should succeed");
    assert!(results[1].is_err(), "unknown agent should fail");
    assert!(matches!(
        results[1].as_ref().unwrap_err(),
        OrchError::OperatorNotFound(_)
    ));
}

// ── Dispatch with a failing operator ─────────────────────────────────────────

struct AlwaysFailOperator;

#[async_trait::async_trait]
impl Operator for AlwaysFailOperator {
    async fn execute(&self, _input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        Err(OperatorError::NonRetryable("intentional failure".into()))
    }
}

#[tokio::test]
async fn dispatch_propagates_operator_failure() {
    let mut orch = TemporalOrch::new(TemporalConfig::default());
    orch.register(OperatorId::new("fail"), Arc::new(AlwaysFailOperator));

    let err = orch
        .dispatch(&OperatorId::new("fail"), simple_input("trigger"))
        .await
        .expect_err("failing operator must propagate an error");

    // The error comes through as DispatchFailed (operator error serialised
    // through the mock client pipeline).
    assert!(
        err.to_string().contains("intentional failure"),
        "expected operator message in error, got: {err}"
    );
}

// ── Signal / query tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn signal_recorded_in_journal() {
    let orch = TemporalOrch::new(TemporalConfig::default());
    let wf = WorkflowId::new("wf-signal");

    orch.signal(
        &wf,
        SignalPayload::new("test-signal", serde_json::json!({ "key": "val" })),
    )
    .await
    .expect("signal must succeed");

    // Verify the signal is reflected by query.
    let result = orch
        .query(&wf, QueryPayload::new("any", serde_json::json!({})))
        .await
        .expect("query must succeed");

    assert_eq!(result["signals"], serde_json::json!(1));
}

#[tokio::test]
async fn query_returns_signal_count() {
    let orch = TemporalOrch::new(TemporalConfig::default());
    let wf = WorkflowId::new("wf-count");

    // Zero signals initially.
    let initial = orch
        .query(&wf, QueryPayload::new("count", serde_json::json!({})))
        .await
        .unwrap();
    assert_eq!(initial, serde_json::json!({ "signals": 0 }));

    // Send three signals.
    for i in 0u32..3 {
        orch.signal(
            &wf,
            SignalPayload::new("ping", serde_json::json!({ "i": i })),
        )
        .await
        .unwrap();
    }

    let after = orch
        .query(&wf, QueryPayload::new("count", serde_json::json!({})))
        .await
        .unwrap();
    assert_eq!(after, serde_json::json!({ "signals": 3 }));
}

#[tokio::test]
async fn signal_to_new_workflow_creates_journal_entry() {
    // Signal to a workflow that has never been referenced — must succeed and
    // create a journal entry (no "workflow must be pre-registered" semantics).
    let orch = TemporalOrch::new(TemporalConfig::default());
    let wf = WorkflowId::new("brand-new-wf");

    orch.signal(&wf, SignalPayload::new("init", serde_json::json!(null)))
        .await
        .expect("signal to brand-new workflow must succeed");

    let result = orch
        .query(&wf, QueryPayload::new("any", serde_json::json!({})))
        .await
        .unwrap();
    assert_eq!(result["signals"], serde_json::json!(1));
}

// ── Send + Sync compile-time check ───────────────────────────────────────────

#[test]
fn temporal_orch_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TemporalOrch>();
}

// ── Config accessor ───────────────────────────────────────────────────────────

#[test]
fn config_accessor_returns_configured_values() {
    let cfg = TemporalConfig {
        server_url: "remote:7233".to_string(),
        namespace: "staging".to_string(),
        task_queue: "staging-queue".to_string(),
        identity: "test-worker".to_string(),
    };
    let orch = TemporalOrch::new(cfg.clone());
    assert_eq!(orch.config().server_url, "remote:7233");
    assert_eq!(orch.config().namespace, "staging");
}
