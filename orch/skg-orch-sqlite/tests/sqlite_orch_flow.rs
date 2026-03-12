use async_trait::async_trait;
use layer0::{
    Content, Effect, ExitReason, Operator, OperatorError, OperatorId, OperatorInput,
    OperatorOutput, Scope,
};
use serde_json::json;
use skg_orch_sqlite::SqliteDurableOrchestrator;
use skg_run_core::{
    PortableWakeDeadline, ResumeInput, RunControlError, RunController, RunStarter, RunStatus,
    RunView, TimerStore, WaitReason, WaitStore,
};
use skg_run_sqlite::SqliteRunStore;
use std::sync::Arc;

fn input_payload(input: &OperatorInput) -> serde_json::Value {
    input
        .message
        .as_text()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .unwrap_or_else(|| json!({}))
}

fn wait_output(wait_point: &str) -> OperatorOutput {
    OperatorOutput::new(
        Content::text(
            json!({
                "kind": "wait",
                "wait_point": wait_point,
                "reason": "external_input"
            })
            .to_string(),
        ),
        ExitReason::Complete,
    )
}

struct ApprovalOperator;

#[async_trait]
impl Operator for ApprovalOperator {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        let payload = input_payload(&input);

        if payload.get("approved") == Some(&json!(true)) {
            return Ok(OperatorOutput::new(
                Content::text(
                    json!({
                        "kind": "complete",
                        "result": { "approved": true, "source": "resume" }
                    })
                    .to_string(),
                ),
                ExitReason::Complete,
            ));
        }

        Ok(wait_output("approval-token"))
    }
}

struct UnsupportedEffectOnResumeOperator;

#[async_trait]
impl Operator for UnsupportedEffectOnResumeOperator {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        let payload = input_payload(&input);

        if payload.get("approved") == Some(&json!(true)) {
            let mut output =
                OperatorOutput::new(Content::text("resume should fail"), ExitReason::Complete);
            output.effects = vec![Effect::WriteMemory {
                scope: Scope::Global,
                key: "approval:result".into(),
                value: json!("approved"),
                tier: None,
                lifetime: None,
                content_kind: None,
                salience: None,
                ttl: None,
            }];
            return Ok(output);
        }

        Ok(wait_output("failure-token"))
    }
}

struct TimedWaitOnResumeOperator;

#[async_trait]
impl Operator for TimedWaitOnResumeOperator {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        let payload = input_payload(&input);

        if payload.get("approved") == Some(&json!(true)) {
            return Ok(OperatorOutput::new(
                Content::text(
                    json!({
                        "kind": "wait",
                        "wait_point": "timed-token",
                        "reason": "timer",
                        "wake_at": "2026-03-12T08:15:30Z"
                    })
                    .to_string(),
                ),
                ExitReason::Complete,
            ));
        }

        Ok(wait_output("timed-entry"))
    }
}

async fn orch() -> (SqliteDurableOrchestrator, Arc<SqliteRunStore>) {
    let run_store = Arc::new(SqliteRunStore::open_in_memory().expect("open run store"));
    let mut orch = SqliteDurableOrchestrator::new(run_store.clone());
    orch.register(OperatorId::new("approval"), Arc::new(ApprovalOperator));
    (orch, run_store)
}

#[tokio::test]
async fn start_wait_resume_query_and_complete() {
    let (orch, _run_store) = orch().await;

    let run_id = orch
        .start_operator_run(OperatorId::new("approval"), json!({ "ticket": 7 }))
        .await
        .expect("start durable run");

    let run = RunController::get_run(&orch, &run_id)
        .await
        .expect("query waiting run");
    let wait_point = match &run {
        RunView::Waiting {
            wait_point,
            wait_reason,
            ..
        } => {
            assert_eq!(wait_reason, &WaitReason::ExternalInput);
            wait_point.clone()
        }
        other => panic!("expected waiting run, got {other:?}"),
    };

    assert_eq!(run.status(), RunStatus::Waiting);

    orch.resume_run(
        &run_id,
        &wait_point,
        ResumeInput::new(json!({ "approved": true })),
    )
    .await
    .expect("resume waiting run");

    let completed = orch.get_run(&run_id).await.expect("query completed run");
    match completed {
        RunView::Completed { result, .. } => {
            assert_eq!(result, json!({ "approved": true, "source": "resume" }));
        }
        other => panic!("expected completed run, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_waiting_run_marks_terminal_state() {
    let (orch, _run_store) = orch().await;
    let run_id = orch
        .start_operator_run(OperatorId::new("approval"), json!({ "ticket": 8 }))
        .await
        .expect("start durable run");

    orch.cancel_run(&run_id).await.expect("cancel waiting run");

    let run = orch.get_run(&run_id).await.expect("query cancelled run");
    assert!(matches!(run, RunView::Cancelled { .. }));
}

#[tokio::test]
async fn generic_start_contract_accepts_operator_envelope() {
    let (orch, _run_store) = orch().await;
    let run_id = RunStarter::start_run(
        &orch,
        json!({
            "operator_id": "approval",
            "input": { "ticket": 9 }
        }),
    )
    .await
    .expect("start generic durable run");

    let run = orch.get_run(&run_id).await.expect("query generic run");
    assert!(matches!(run, RunView::Waiting { .. }));
}

#[tokio::test]
async fn signal_run_rejects_unsupported_delivery_without_changing_wait_state() {
    let (orch, run_store) = orch().await;
    let run_id = orch
        .start_operator_run(OperatorId::new("approval"), json!({ "ticket": 10 }))
        .await
        .expect("start durable run");

    let err = orch
        .signal_run(&run_id, json!({ "kind": "poke" }))
        .await
        .expect_err("reject unsupported signal delivery");
    assert!(matches!(
        &err,
        RunControlError::Backend(message)
            if message.contains("signal delivery is unsupported")
    ));

    let waiting = orch
        .get_run(&run_id)
        .await
        .expect("query rejected-signal run");
    let wait_point = match waiting {
        RunView::Waiting {
            wait_point,
            wait_reason,
            ..
        } => {
            assert_eq!(wait_reason, WaitReason::ExternalInput);
            wait_point
        }
        other => panic!("signal rejection must keep waitpoint unchanged, got {other:?}"),
    };
    assert!(
        run_store
            .drain_signals(&run_id)
            .await
            .expect("drain queued signals after rejection")
            .is_empty()
    );

    orch.resume_run(
        &run_id,
        &wait_point,
        ResumeInput::new(json!({ "approved": true })),
    )
    .await
    .expect("resume after rejected signal");

    let completed = orch.get_run(&run_id).await.expect("query resumed run");
    assert!(matches!(completed, RunView::Completed { .. }));
}

#[tokio::test]
async fn operator_effects_are_rejected_and_persist_failed_terminal_state() {
    let (mut orch, _run_store) = orch().await;
    orch.register(
        OperatorId::new("unsupported_effect_on_resume"),
        Arc::new(UnsupportedEffectOnResumeOperator),
    );

    let run_id = orch
        .start_operator_run(
            OperatorId::new("unsupported_effect_on_resume"),
            json!({ "ticket": 11 }),
        )
        .await
        .expect("start waiting failure run");
    let waiting = orch
        .get_run(&run_id)
        .await
        .expect("query waiting failure run");
    let wait_point = match waiting {
        RunView::Waiting { wait_point, .. } => wait_point,
        other => panic!("expected waiting run before failing resume, got {other:?}"),
    };

    let err = orch
        .resume_run(
            &run_id,
            &wait_point,
            ResumeInput::new(json!({ "approved": true })),
        )
        .await
        .expect_err("surface unsupported effect failure");
    assert!(matches!(
        &err,
        RunControlError::InvalidInput(message)
            if message.contains("operator effects are unsupported")
                && message.contains("write_memory")
    ));

    let failed = orch
        .get_run(&run_id)
        .await
        .expect("query failed run after driver error");
    match failed {
        RunView::Failed { error, .. } => {
            assert!(error.contains("operator effects are unsupported"));
            assert!(error.contains("write_memory"));
        }
        other => panic!("expected failed run after driver error, got {other:?}"),
    }
}

#[tokio::test]
async fn timed_waits_are_rejected_without_scheduling_timer_and_persist_failed_terminal_state() {
    let (mut orch, run_store) = orch().await;
    orch.register(
        OperatorId::new("timed_wait_on_resume"),
        Arc::new(TimedWaitOnResumeOperator),
    );

    let run_id = orch
        .start_operator_run(
            OperatorId::new("timed_wait_on_resume"),
            json!({ "ticket": 12 }),
        )
        .await
        .expect("start timed wait run");
    let waiting = orch
        .get_run(&run_id)
        .await
        .expect("query timed wait run before failing resume");
    let wait_point = match waiting {
        RunView::Waiting { wait_point, .. } => wait_point,
        other => panic!("expected waiting run before timed wait rejection, got {other:?}"),
    };

    let err = orch
        .resume_run(
            &run_id,
            &wait_point,
            ResumeInput::new(json!({ "approved": true })),
        )
        .await
        .expect_err("surface unsupported timed wait failure");
    assert!(matches!(
        &err,
        RunControlError::InvalidInput(message)
            if message.contains("timed waits are unsupported")
    ));

    let failed = orch
        .get_run(&run_id)
        .await
        .expect("query failed run after timed wait rejection");
    match failed {
        RunView::Failed { error, .. } => {
            assert!(error.contains("timed waits are unsupported"));
        }
        other => panic!("expected failed run after timed wait rejection, got {other:?}"),
    }

    let not_after = PortableWakeDeadline::parse("2026-03-12T09:00:00Z").unwrap();
    assert!(
        run_store
            .due_timers(&not_after, 10)
            .await
            .expect("list timers after timed wait rejection")
            .is_empty()
    );
}
