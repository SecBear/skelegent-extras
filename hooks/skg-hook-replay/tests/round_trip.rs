//! Round-trip integration tests: record through skg-hook-recorder, replay through skg-hook-replay.
//!
//! These tests verify that entries recorded by `DispatchRecorder` and `ExecRecorder`
//! carry non-Null Post payloads and that those payloads round-trip correctly through
//! `ReplayDispatcher`.
//!
//! # Dispatch payload limitation
//!
//! `DispatchRecorder` cannot capture the actual `OperatorOutput` in the Post phase because
//! `DispatchMiddleware` returns a `DispatchHandle` immediately — the real output arrives
//! asynchronously via `DispatchEvent::Completed`. The recorder's Post payload is therefore
//! a status marker (`{"status": "dispatched"}`), not an `OperatorOutput`.
//!
//! The round-trip test for dispatch works by:
//! 1. Recording through `DispatchRecorder` to capture well-formed Pre entries.
//! 2. Collecting each `DispatchHandle` to obtain the actual `OperatorOutput`.
//! 3. Building replay-ready Post entries by serializing the collected outputs.
//! 4. Feeding those entries to `ReplayDispatcher` and verifying the replayed outputs match.
//!
//! For `ExecRecorder` the fix is complete: the Post payload contains the serialized
//! `OperatorOutput` directly (exec is synchronous, so there is no handle gap).

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use layer0::dispatch_context::DispatchContext;
use layer0::environment::EnvironmentSpec;
use layer0::error::{EnvError, OrchError};
use layer0::id::{DispatchId, OperatorId};
use layer0::middleware::{DispatchMiddleware, DispatchNext, ExecMiddleware, ExecNext};
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput, TriggerType};
use layer0::Dispatcher;
use skg_hook_recorder::{
    Boundary, DispatchRecorder, ExecRecorder, InMemorySink, Phase, RecordEntry,
    SCHEMA_VERSION,
};
use skg_hook_replay::ReplayDispatcher;
use std::sync::Arc;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn make_output(text: &str) -> OperatorOutput {
    OperatorOutput::new(Content::text(text), ExitReason::Complete)
}

fn make_input(text: &str) -> OperatorInput {
    OperatorInput::new(Content::text(text), TriggerType::User)
}

fn make_ctx(op: &str, dispatch_id: &str) -> DispatchContext {
    DispatchContext::new(DispatchId::new(dispatch_id), OperatorId::new(op))
}

fn immediate_handle(output: OperatorOutput) -> DispatchHandle {
    let (handle, sender) = DispatchHandle::channel(DispatchId::new("mock"));
    tokio::spawn(async move {
        let _ = sender.send(DispatchEvent::Completed { output }).await;
    });
    handle
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MOCK DISPATCH NEXT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A mock `DispatchNext` terminal that returns a fixed sequence of outputs.
///
/// Each call consumes the next output from the list. Used to give the
/// recorder known values to record.
struct SequencedNext {
    outputs: std::sync::Mutex<std::collections::VecDeque<OperatorOutput>>,
}

impl SequencedNext {
    fn new(outputs: Vec<OperatorOutput>) -> Self {
        Self {
            outputs: std::sync::Mutex::new(outputs.into()),
        }
    }
}

#[async_trait]
impl DispatchNext for SequencedNext {
    async fn dispatch(
        &self,
        _ctx: &DispatchContext,
        _input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let output = self
            .outputs
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| OrchError::DispatchFailed("mock exhausted".into()))?;
        Ok(immediate_handle(output))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MOCK EXEC NEXT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A mock `ExecNext` terminal that returns a fixed sequence of outputs.
struct SequencedExecNext {
    outputs: std::sync::Mutex<std::collections::VecDeque<OperatorOutput>>,
}

impl SequencedExecNext {
    fn new(outputs: Vec<OperatorOutput>) -> Self {
        Self {
            outputs: std::sync::Mutex::new(outputs.into()),
        }
    }
}

#[async_trait]
impl ExecNext for SequencedExecNext {
    async fn run(
        &self,
        _input: OperatorInput,
        _spec: &EnvironmentSpec,
    ) -> Result<OperatorOutput, EnvError> {
        self.outputs
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| EnvError::ProvisionFailed("mock exhausted".into()))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DISPATCH ROUND-TRIP TEST
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verifies the full record-then-replay round-trip for dispatch operations.
///
/// Because `DispatchRecorder` cannot embed the actual `OperatorOutput` in its Post
/// entries (the handle is async), the round-trip reconstructs replay-ready entries by:
/// - Using the recorder's Pre entries (which carry the correct `RecordContext`).
/// - Serializing the outputs collected from the live handles into new Post entries.
///
/// This mirrors how a real replay system would store recordings: capture context and
/// outputs together, then replay deterministically.
#[tokio::test]
async fn dispatch_round_trip_three_cases() {
    // Known outputs our mock will produce.
    let expected_outputs = vec![
        make_output("alpha"),
        make_output("beta"),
        make_output("gamma"),
    ];

    // --- RECORD PHASE ---

    let sink = Arc::new(InMemorySink::new());
    let recorder = DispatchRecorder::new(sink.clone());

    let mock_next = SequencedNext::new(expected_outputs.clone());

    let ctxs = [
        make_ctx("op-a", "d-001"),
        make_ctx("op-b", "d-002"),
        make_ctx("op-c", "d-003"),
    ];

    // Dispatch 3 times through the recorder and collect each handle.
    let mut recorded_outputs: Vec<OperatorOutput> = Vec::new();
    for (i, ctx) in ctxs.iter().enumerate() {
        let input = make_input(&format!("input-{i}"));
        let handle = recorder
            .dispatch(ctx, input, &mock_next)
            .await
            .expect("recorder dispatch should succeed");
        let output = handle.collect().await.expect("handle collect should succeed");
        recorded_outputs.push(output);
    }

    // Verify we got the right outputs from the live dispatch.
    assert_eq!(
        recorded_outputs[0].message.as_text(),
        Some("alpha"),
        "first output mismatch"
    );
    assert_eq!(
        recorded_outputs[1].message.as_text(),
        Some("beta"),
        "second output mismatch"
    );
    assert_eq!(
        recorded_outputs[2].message.as_text(),
        Some("gamma"),
        "third output mismatch"
    );

    // Retrieve all entries from the sink: 3 dispatches × 2 phases = 6 entries.
    let raw_entries = sink.entries().await;
    assert_eq!(raw_entries.len(), 6, "expected 6 entries (3 pre + 3 post)");

    // Verify the recorder captured well-formed entries.
    for (i, entry) in raw_entries.iter().enumerate() {
        assert_eq!(
            entry.boundary,
            Boundary::Dispatch,
            "entry {i}: wrong boundary"
        );
        let expected_phase = if i % 2 == 0 { Phase::Pre } else { Phase::Post };
        assert_eq!(entry.phase, expected_phase, "entry {i}: wrong phase");
    }

    // Post entries must have non-Null payloads after the fix.
    // The dispatch recorder records a status marker (`{"status": "dispatched"}`) since
    // it cannot capture the async OperatorOutput directly.
    let post_entries: Vec<_> = raw_entries
        .iter()
        .filter(|e| e.phase == Phase::Post)
        .collect();
    assert_eq!(post_entries.len(), 3, "expected 3 post entries");
    for (i, post) in post_entries.iter().enumerate() {
        assert_ne!(
            post.payload_json,
            serde_json::Value::Null,
            "post entry {i}: payload_json must not be Null after the fix"
        );
    }

    // --- BUILD REPLAY ENTRIES ---
    //
    // The ReplayDispatcher needs Post entries whose payload_json is a valid OperatorOutput.
    // Since the DispatchRecorder cannot produce those (async handle gap), we construct
    // them here from the collected outputs paired with the recorded context fields.
    //
    // In production, a replay archive would record outputs out-of-band (e.g., by
    // listening on the DispatchHandle after the recorder layer).
    let pre_entries: Vec<_> = raw_entries
        .iter()
        .filter(|e| e.phase == Phase::Pre)
        .collect();
    assert_eq!(pre_entries.len(), 3, "expected 3 pre entries");

    let replay_entries: Vec<RecordEntry> = pre_entries
        .iter()
        .zip(recorded_outputs.iter())
        .map(|(pre, output)| RecordEntry {
            boundary: Boundary::Dispatch,
            phase: Phase::Post,
            context: pre.context.clone(),
            payload_json: serde_json::to_value(output).expect("serialize output"),
            duration_ms: Some(0),
            error: None,
            version: SCHEMA_VERSION,
        })
        .collect();

    // --- REPLAY PHASE ---

    let replay = ReplayDispatcher::new(replay_entries);
    let replay_ctx = make_ctx("any-op", "r-001");

    let replayed: Vec<OperatorOutput> = {
        let mut v = Vec::new();
        for i in 0..3usize {
            let input = make_input(&format!("replay-input-{i}"));
            let handle = replay
                .dispatch(&replay_ctx, input)
                .await
                .unwrap_or_else(|e| panic!("replay dispatch {i} failed: {e}"));
            let out = handle
                .collect()
                .await
                .unwrap_or_else(|e| panic!("replay collect {i} failed: {e}"));
            v.push(out);
        }
        v
    };

    // Replayed outputs must match the originally recorded outputs.
    let labels = ["alpha", "beta", "gamma"];
    for (i, (replayed_out, label)) in replayed.iter().zip(labels.iter()).enumerate() {
        assert_eq!(
            replayed_out.message.as_text(),
            Some(*label),
            "replay {i}: expected '{label}', got {:?}",
            replayed_out.message.as_text()
        );
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// EXEC RECORDER ROUND-TRIP TEST
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verifies that `ExecRecorder` Post entries carry non-Null, deserializable payloads.
///
/// For exec the fix is complete: `ExecMiddleware` is synchronous, so the recorder
/// can capture the `OperatorOutput` directly into the Post entry's `payload_json`.
///
/// No `ReplayExecRunner` exists yet, so this test stops at verifying the recorded
/// entries are correctly populated. Once a replay runner is added, a full exec
/// round-trip can be built on top of this.
#[tokio::test]
async fn exec_recorder_post_payload_non_null() {
    let expected_outputs = vec![
        make_output("exec-result-0"),
        make_output("exec-result-1"),
        make_output("exec-result-2"),
    ];

    let sink = Arc::new(InMemorySink::new());
    let recorder = ExecRecorder::new(sink.clone());
    let spec = EnvironmentSpec::default();

    let mock_exec = SequencedExecNext::new(expected_outputs.clone());

    // Run 3 executions through the recorder.
    let mut live_outputs = Vec::new();
    for i in 0..3usize {
        let input = make_input(&format!("exec-input-{i}"));
        let output = recorder
            .run(input, &spec, &mock_exec)
            .await
            .unwrap_or_else(|e| panic!("exec run {i} failed: {e}"));
        live_outputs.push(output);
    }

    // Verify live outputs match what the mock returned.
    for (i, out) in live_outputs.iter().enumerate() {
        let expected_text = format!("exec-result-{i}");
        assert_eq!(
            out.message.as_text(),
            Some(expected_text.as_str()),
            "exec live output {i} mismatch"
        );
    }

    // Retrieve entries: 3 executions × 2 phases = 6 entries.
    let entries = sink.entries().await;
    assert_eq!(entries.len(), 6, "expected 6 entries (3 pre + 3 post)");

    // All entries must be Exec boundary.
    for (i, entry) in entries.iter().enumerate() {
        assert_eq!(
            entry.boundary,
            Boundary::Exec,
            "entry {i}: expected Exec boundary"
        );
    }

    // Pre entries must have non-Null payloads (OperatorInput serialization).
    let pre_entries: Vec<_> = entries.iter().filter(|e| e.phase == Phase::Pre).collect();
    assert_eq!(pre_entries.len(), 3);
    for (i, pre) in pre_entries.iter().enumerate() {
        assert_ne!(
            pre.payload_json,
            serde_json::Value::Null,
            "pre entry {i}: payload_json should contain serialized OperatorInput"
        );
    }

    // Post entries must carry the serialized OperatorOutput — this is the fix being verified.
    let post_entries: Vec<_> = entries.iter().filter(|e| e.phase == Phase::Post).collect();
    assert_eq!(post_entries.len(), 3);
    for (i, post) in post_entries.iter().enumerate() {
        assert_ne!(
            post.payload_json,
            serde_json::Value::Null,
            "post entry {i}: payload_json must not be Null after the fix"
        );

        // The Post payload must deserialize back to an OperatorOutput that matches
        // what the live run produced.
        let replayed: OperatorOutput = serde_json::from_value(post.payload_json.clone())
            .unwrap_or_else(|e| panic!("post entry {i}: failed to deserialize payload: {e}"));
        let expected_text = format!("exec-result-{i}");
        assert_eq!(
            replayed.message.as_text(),
            Some(expected_text.as_str()),
            "post entry {i}: deserialized output mismatch"
        );
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DISPATCH ENTRY CONTEXT TEST
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verifies that `DispatchRecorder` entries carry the correct context fields
/// (operator_id, dispatch_id) so downstream systems can correlate recordings.
#[tokio::test]
async fn dispatch_recorder_entries_carry_correct_context() {
    let sink = Arc::new(InMemorySink::new());
    let recorder = DispatchRecorder::new(sink.clone());

    let outputs = vec![make_output("ctx-result")];
    let mock_next = SequencedNext::new(outputs);

    let ctx = make_ctx("my-operator", "dispatch-42");
    let input = make_input("some input");

    let handle = recorder
        .dispatch(&ctx, input, &mock_next)
        .await
        .expect("dispatch should succeed");
    let _ = handle.collect().await.expect("collect should succeed");

    let entries = sink.entries().await;
    assert_eq!(entries.len(), 2);

    // Both entries must carry operator_id and dispatch_id from the context.
    for entry in &entries {
        assert_eq!(
            entry.context.operator_id, "my-operator",
            "wrong operator_id in context"
        );
        assert_eq!(
            entry.context.dispatch_id, "dispatch-42",
            "wrong dispatch_id in context"
        );
    }

    // Pre entry payload contains the serialized OperatorInput (not Null).
    let pre = &entries[0];
    assert_eq!(pre.phase, Phase::Pre);
    assert_ne!(
        pre.payload_json,
        serde_json::Value::Null,
        "pre payload should be non-Null"
    );

    // Post entry must have timing set and a non-Null status payload.
    let post = &entries[1];
    assert_eq!(post.phase, Phase::Post);
    assert!(post.duration_ms.is_some(), "post entry must have duration");
    assert!(post.error.is_none(), "no error expected");
    assert_ne!(
        post.payload_json,
        serde_json::Value::Null,
        "post payload must not be Null after the fix"
    );
}
