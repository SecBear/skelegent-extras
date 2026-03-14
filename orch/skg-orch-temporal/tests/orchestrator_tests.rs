//! Integration tests for `skg-orch-temporal`.
//!
//! Default-feature tests stay mock-backed and dependency-light. Feature-gated
//! checks for `temporal-sdk` are compile-oriented unless a real Temporal server
//! is provided separately.

use layer0::content::Content;
use layer0::dispatch::Dispatcher;
use layer0::effect::SignalPayload;
use layer0::error::{OperatorError, OrchError};
use layer0::id::{OperatorId, WorkflowId};
use layer0::operator::{ExitReason, Operator, OperatorInput, OperatorOutput, TriggerType};
use layer0::test_utils::EchoOperator;
use layer0::dispatch::EffectEmitter;
use skg_effects_core::{QueryPayload, Queryable, Signalable};
use skg_orch_temporal::{RetryPolicy, TemporalConfig, TemporalOrch};
use skg_run_core::{ResumeInput, RunController, RunStarter, RunStatus, RunView, WaitReason};
use std::sync::Arc;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn simple_input(msg: &str) -> OperatorInput {
    OperatorInput::new(Content::text(msg), TriggerType::User)
}

fn mock_run_start_input(ticket: u64) -> serde_json::Value {
    serde_json::json!({
        "operator_id": "approval",
        "input": { "ticket": ticket }
    })
}

// ── Config tests ─────────────────────────────────────────────────────────────

#[test]
fn temporal_config_defaults() {
    let cfg = TemporalConfig::default();
    assert_eq!(cfg.server_url, "localhost:7233");
    assert_eq!(cfg.namespace, "default");
    assert_eq!(cfg.task_queue, "skg-worker");
    assert_eq!(cfg.identity, "");
    assert_eq!(cfg.workflow_type, "skg.generic.durable-run");
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
        workflow_type: "company.generic.durable-run".to_string(),
    };
    let json = serde_json::to_string(&original).expect("serialize TemporalConfig");
    let recovered: TemporalConfig = serde_json::from_str(&json).expect("deserialize TemporalConfig");
    assert_eq!(recovered.server_url, original.server_url);
    assert_eq!(recovered.namespace, original.namespace);
    assert_eq!(recovered.task_queue, original.task_queue);
    assert_eq!(recovered.identity, original.identity);
    assert_eq!(recovered.workflow_type, original.workflow_type);
}

#[test]
fn temporal_config_deserializes_when_workflow_type_is_missing() {
    let legacy = serde_json::json!({
        "server_url": "legacy:7233",
        "namespace": "prod",
        "task_queue": "legacy-queue",
        "identity": "worker-legacy"
    });

    let recovered: TemporalConfig =
        serde_json::from_value(legacy).expect("deserialize legacy TemporalConfig");
    assert_eq!(recovered.server_url, "legacy:7233");
    assert_eq!(recovered.namespace, "prod");
    assert_eq!(recovered.task_queue, "legacy-queue");
    assert_eq!(recovered.identity, "worker-legacy");
    assert_eq!(recovered.workflow_type, "skg.generic.durable-run");
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
        .expect("dispatch should succeed")
        .collect()
        .await
        .expect("collect should succeed");

    assert_eq!(output.message, Content::text("hello"));
    assert_eq!(output.exit_reason, ExitReason::Complete);
}

#[tokio::test]
async fn dispatch_unknown_agent_returns_error() {
    let orch = TemporalOrch::new(TemporalConfig::default());

    let result = orch
        .dispatch(&OperatorId::new("ghost"), simple_input("x"))
        .await;
    let err = match result {
        Err(e) => e,
        Ok(handle) => handle.collect().await.expect_err("unregistered agent must return an error"),
    };

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

    let result_a = orch
        .dispatch(&OperatorId::new("a"), simple_input("msg-a"))
        .await
        .expect("dispatch a should succeed")
        .collect()
        .await
        .expect("collect a should succeed");
    let result_b = orch
        .dispatch(&OperatorId::new("b"), simple_input("msg-b"))
        .await
        .expect("dispatch b should succeed")
        .collect()
        .await
        .expect("collect b should succeed");

    assert_eq!(result_a.message, Content::text("msg-a"));
    assert_eq!(result_b.message, Content::text("msg-b"));
}

#[tokio::test]
async fn dispatch_many_partial_failure() {
    let mut orch = TemporalOrch::new(TemporalConfig::default());
    orch.register(OperatorId::new("ok"), Arc::new(EchoOperator));
    // "bad" is intentionally not registered.

    let ok_output = orch
        .dispatch(&OperatorId::new("ok"), simple_input("fine"))
        .await
        .and_then(|h| Ok(h));
    let bad_result = orch
        .dispatch(&OperatorId::new("bad"), simple_input("boom"))
        .await;

    assert!(ok_output.is_ok(), "known agent should succeed");
    let bad_err = match bad_result {
        Err(e) => Some(e),
        Ok(handle) => handle.collect().await.err(),
    };
    assert!(bad_err.is_some(), "unknown agent should fail");
    assert!(matches!(
        bad_err.unwrap(),
        OrchError::OperatorNotFound(_)
    ));
}

// ── Dispatch with a failing operator ─────────────────────────────────────────

struct AlwaysFailOperator;

#[async_trait::async_trait]
impl Operator for AlwaysFailOperator {
    async fn execute(&self, _input: OperatorInput, _emitter: &EffectEmitter) -> Result<OperatorOutput, OperatorError> {
        Err(OperatorError::NonRetryable("intentional failure".into()))
    }
}

#[tokio::test]
async fn dispatch_propagates_operator_failure() {
    let mut orch = TemporalOrch::new(TemporalConfig::default());
    orch.register(OperatorId::new("fail"), Arc::new(AlwaysFailOperator));

    let result = orch
        .dispatch(&OperatorId::new("fail"), simple_input("trigger"))
        .await;
    let err = match result {
        Err(e) => e,
        Ok(handle) => handle.collect().await.expect_err("failing operator must propagate an error"),
    };

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
        workflow_type: "staging.generic.durable-run".to_string(),
    };
    let orch = TemporalOrch::new(cfg.clone());
    assert_eq!(orch.config().server_url, "remote:7233");
    assert_eq!(orch.config().namespace, "staging");
    assert_eq!(orch.config().workflow_type, "staging.generic.durable-run");
}

// ── Durable run control tests ─────────────────────────────────────────────────

#[tokio::test]
async fn start_run_get_run_and_query_current_state() {
    let orch = TemporalOrch::new(TemporalConfig::default());
    let run_id = RunStarter::start_run(&orch, mock_run_start_input(7))
        .await
        .expect("start durable run");

    let run = RunController::get_run(&orch, &run_id)
        .await
        .expect("fetch waiting run");
    match &run {
        RunView::Waiting { wait_reason, .. } => {
            assert_eq!(wait_reason, &WaitReason::ExternalInput);
        }
        other => panic!("expected waiting run, got {other:?}"),
    }
    assert_eq!(run.status(), RunStatus::Waiting);

    let queried = orch
        .query(
            &WorkflowId::new(run_id.as_str()),
            QueryPayload::new("skg.run.view", serde_json::json!({})),
        )
        .await
        .expect("query current run view");
    assert_eq!(queried["status"], serde_json::json!("waiting"));
    assert_eq!(queried["run_id"], serde_json::json!(run_id.as_str()));
    // The portable RunId remains the external identifier on the mock path too;
    // callers never see a separate Temporal server-assigned run ID.
    assert_eq!(WorkflowId::new(run_id.as_str()).as_str(), run_id.as_str());
}

#[tokio::test]
async fn signal_run_is_distinct_from_resume_run() {
    let orch = TemporalOrch::new(TemporalConfig::default());
    let run_id = RunStarter::start_run(&orch, mock_run_start_input(8))
        .await
        .expect("start durable run");

    let waiting = orch.get_run(&run_id).await.expect("fetch waiting run");
    let wait_point = match waiting {
        RunView::Waiting {
            wait_point,
            wait_reason,
            ..
        } => {
            assert_eq!(wait_reason, WaitReason::ExternalInput);
            wait_point
        }
        other => panic!("expected waiting run before signal, got {other:?}"),
    };

    orch.signal_run(&run_id, serde_json::json!({ "kind": "poke" }))
        .await
        .expect("send durable signal");

    let after_signal = orch.get_run(&run_id).await.expect("fetch signalled run");
    match after_signal {
        RunView::Waiting {
            wait_point: active_wait,
            wait_reason,
            ..
        } => {
            assert_eq!(active_wait, wait_point);
            assert_eq!(wait_reason, WaitReason::ExternalInput);
        }
        other => panic!("signal must not resume waiting run, got {other:?}"),
    }
}

#[tokio::test]
async fn resume_waiting_run_completes_with_resume_payload() {
    let orch = TemporalOrch::new(TemporalConfig::default());
    let run_id = RunStarter::start_run(&orch, mock_run_start_input(9))
        .await
        .expect("start durable run");

    let waiting = orch.get_run(&run_id).await.expect("fetch waiting run");
    let wait_point = match waiting {
        RunView::Waiting { wait_point, .. } => wait_point,
        other => panic!("expected waiting run before resume, got {other:?}"),
    };

    orch.resume_run(
        &run_id,
        &wait_point,
        ResumeInput::new(serde_json::json!({ "approved": true })),
    )
    .await
    .expect("resume waiting run");

    let completed = orch.get_run(&run_id).await.expect("fetch completed run");
    match completed {
        RunView::Completed { result, .. } => {
            assert_eq!(result["source"], serde_json::json!("resume"));
            assert_eq!(result["resume"], serde_json::json!({ "approved": true }));
            assert_eq!(result["start"]["input"]["ticket"], serde_json::json!(9));
        }
        other => panic!("expected completed run after resume, got {other:?}"),
    }
}

#[cfg(feature = "temporal-sdk")]
#[test]
fn temporal_sdk_config_can_describe_generic_durable_run_workflow() {
    let cfg = TemporalConfig::default();
    assert!(!cfg.workflow_type.is_empty());
}
