use rusqlite::{Connection, Error as SqliteError};
use serde_json::json;
use skg_run_core::{
    BackendRunRef, PendingResume, PendingSignal, PortableWakeDeadline, ResumeInput, RunId,
    RunOutcome, RunStatus, RunStore, RunStoreError, RunView, ScheduledTimer, StoreRunRecord,
    TimerStore, TimerStoreError, WaitPointId, WaitReason, WaitStore,
};
use skg_run_sqlite::SqliteRunStore;
use std::{env, fs, time::{SystemTime, UNIX_EPOCH}};

#[tokio::test]
async fn run_metadata_round_trips_and_replaces_previous_state() {
    let store = SqliteRunStore::open_in_memory().unwrap();
    let run_id = RunId::new("run-1");
    let wait_point = WaitPointId::new("wait-1");

    let initial = StoreRunRecord::new(
        RunView::waiting(run_id.clone(), wait_point.clone(), WaitReason::ExternalInput),
        Some(BackendRunRef::new("opaque-1")),
    );
    store.insert_run(initial.clone()).await.unwrap();

    assert_eq!(store.get_run(&run_id).await.unwrap(), Some(initial.clone()));
    assert!(matches!(
        store.insert_run(initial.clone()).await.unwrap_err(),
        RunStoreError::Conflict(_)
    ));

    let replacement = StoreRunRecord::new(
        RunView::terminal(
            run_id.clone(),
            RunOutcome::Completed {
                result: json!({ "ok": true }),
            },
        ),
        Some(BackendRunRef::new("opaque-2")),
    );
    store.put_run(replacement.clone()).await.unwrap();

    let fetched = store.get_run(&run_id).await.unwrap().unwrap();
    assert_eq!(fetched, replacement);
    assert_eq!(fetched.view.status(), RunStatus::Completed);
    assert_eq!(fetched.backend_ref, Some(BackendRunRef::new("opaque-2")));
    assert!(fetched.view.wait_point().is_none());
    assert!(store
        .get_run(&RunId::new("missing-run"))
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        store
            .put_run(StoreRunRecord::new(RunView::running(RunId::new("missing-run")), None))
            .await
            .unwrap_err(),
        RunStoreError::RunNotFound(found) if found == RunId::new("missing-run")
    ));
}

#[tokio::test]
async fn wait_store_keeps_resume_and_signals_distinct() {
    let store = SqliteRunStore::open_in_memory().unwrap();
    let run_id = RunId::new("run-2");
    let wait_point = WaitPointId::new("wait-2");

    let resume = PendingResume::new(
        run_id.clone(),
        wait_point.clone(),
        ResumeInput::new(json!({ "approved": true })).with_metadata("source", json!("human")),
    );
    store.save_resume(resume.clone()).await.unwrap();
    store
        .push_signal(PendingSignal::new(run_id.clone(), json!({ "kind": "poke" })))
        .await
        .unwrap();
    store
        .push_signal(PendingSignal::new(run_id.clone(), json!({ "kind": "cancel-check" })))
        .await
        .unwrap();

    assert_eq!(store.take_resume(&run_id, &wait_point).await.unwrap(), Some(resume));
    assert!(store.take_resume(&run_id, &wait_point).await.unwrap().is_none());

    let drained = store.drain_signals(&run_id).await.unwrap();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].signal, json!({ "kind": "poke" }));
    assert_eq!(drained[1].signal, json!({ "kind": "cancel-check" }));
    assert!(store.drain_signals(&run_id).await.unwrap().is_empty());
}

#[tokio::test]
async fn timers_can_be_replaced_listed_and_cancelled() {
    let store = SqliteRunStore::open_in_memory().unwrap();
    let run_id = RunId::new("run-3");
    let wait_point = WaitPointId::new("wait-3");
    let earlier = PortableWakeDeadline::parse("2026-03-12T08:00:00Z").unwrap();
    let later = PortableWakeDeadline::parse("2026-03-12T09:30:00Z").unwrap();
    let second_wait = WaitPointId::new("wait-4");

    store
        .schedule_timer(ScheduledTimer::new(
            run_id.clone(),
            wait_point.clone(),
            earlier.clone(),
        ))
        .await
        .unwrap();
    store
        .schedule_timer(ScheduledTimer::new(
            run_id.clone(),
            wait_point.clone(),
            later.clone(),
        ))
        .await
        .unwrap();
    store
        .schedule_timer(ScheduledTimer::new(
            run_id.clone(),
            second_wait.clone(),
            earlier.clone(),
        ))
        .await
        .unwrap();

    assert_eq!(store.due_timers(&earlier, 10).await.unwrap().len(), 1);
    let due = store.due_timers(&later, 10).await.unwrap();
    assert_eq!(due.len(), 2);
    assert_eq!(due[0].wait_point, second_wait);
    assert_eq!(due[1].wait_point, wait_point);
    assert_eq!(due[1].wake_at, later);

    store.cancel_timer(&run_id, &wait_point).await.unwrap();
    assert!(matches!(
        store.cancel_timer(&run_id, &wait_point).await.unwrap_err(),
        TimerStoreError::TimerNotFound { run_id: ref found_run, wait_point: ref found_wait }
            if *found_run == run_id && *found_wait == wait_point
    ));
    let remaining = store.due_timers(&later, 10).await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].wait_point, WaitPointId::new("wait-4"));
}

#[test]
fn open_rejects_future_schema_versions() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let path = env::temp_dir().join(format!("skg-run-sqlite-future-schema-{unique}.db"));

    let conn = Connection::open(&path).expect("create sqlite database");
    conn.pragma_update(None, "user_version", 2_u32)
        .expect("set future schema version");
    drop(conn);

    let error = SqliteRunStore::open(&path)
        .err()
        .expect("future schema version should be rejected");
    assert!(
        matches!(error, SqliteError::SqliteFailure(_, _)),
        "expected sqlite failure, got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("unsupported future schema version"),
        "expected future schema message, got {message}"
    );
    assert!(
        message.contains("found 2"),
        "expected found schema version in message, got {message}"
    );
    assert!(
        message.contains("current 1"),
        "expected current schema version in message, got {message}"
    );

    fs::remove_file(&path).expect("remove temp sqlite database");
}
