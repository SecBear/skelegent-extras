//! Integration test: record → replay → eval pipeline.
//!
//! Proves that:
//! 1. A mock dispatcher produces predictable outputs for known inputs
//! 2. Those outputs can be serialised into [`RecordEntry`] replay entries
//! 3. [`ReplayDispatcher`] faithfully reproduces the recorded outputs without
//!    any real API calls
//! 4. [`EvalRunner`] scores are identical between the live and replay runs

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use layer0::dispatch_context::DispatchContext;
use layer0::error::OrchError;
use layer0::id::{DispatchId, OperatorId};
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput, TriggerType};
use layer0::Dispatcher;
use skg_eval::{ContainsMetric, EvalCase, EvalRunner, ExactMatchMetric, ExpectedOutput};
use skg_hook_recorder::{Boundary, Phase, RecordContext, RecordEntry, SCHEMA_VERSION};
use skg_hook_replay::ReplayDispatcher;
use std::sync::Arc;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MOCK DISPATCHER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A mock dispatcher that maps known input substrings to fixed responses.
///
/// If the input text contains "hello" → responds with "world".
/// If the input text contains "ping"  → responds with "pong ping".
/// If the input text contains "foo"   → responds with "foo bar baz".
/// Otherwise returns "default".
struct InputMappingDispatcher;

impl InputMappingDispatcher {
    fn response_for(input_text: &str) -> &'static str {
        if input_text.contains("hello") {
            "world"
        } else if input_text.contains("ping") {
            "pong ping"
        } else if input_text.contains("foo") {
            "foo bar baz"
        } else {
            "default"
        }
    }
}

#[async_trait]
impl Dispatcher for InputMappingDispatcher {
    async fn dispatch(
        &self,
        ctx: &DispatchContext,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let input_text = input.message.as_text().unwrap_or_default().to_owned();
        let response_text = Self::response_for(&input_text);
        let output = OperatorOutput::new(Content::text(response_text), ExitReason::Complete);

        let dispatch_id = ctx.dispatch_id.clone();
        let (handle, sender) = DispatchHandle::channel(dispatch_id);
        tokio::spawn(async move {
            let _ = sender.send(DispatchEvent::Completed { output }).await;
        });
        Ok(handle)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn make_input(text: &str) -> OperatorInput {
    OperatorInput::new(Content::text(text), TriggerType::User)
}

fn make_eval_cases() -> Vec<EvalCase> {
    vec![
        EvalCase::new(
            "hello-case",
            make_input("hello"),
            ExpectedOutput::ExactText("world".into()),
        ),
        EvalCase::new(
            "ping-case",
            make_input("ping"),
            ExpectedOutput::Contains(vec!["pong".into(), "ping".into()]),
        ),
        EvalCase::new(
            "foo-case",
            make_input("foo"),
            ExpectedOutput::ExactText("foo bar baz".into()),
        ),
    ]
}

/// Build a [`RecordEntry`] suitable for [`ReplayDispatcher`] from an output.
///
/// [`ReplayDispatcher`] filters for `Boundary::Dispatch + Phase::Post` entries
/// whose `payload_json` deserialises as an [`OperatorOutput`].
fn make_replay_entry(case_name: &str, output: &OperatorOutput) -> RecordEntry {
    RecordEntry {
        boundary: Boundary::Dispatch,
        phase: Phase::Post,
        context: RecordContext {
            trace_id: String::new(),
            operator_id: "mock-agent".into(),
            dispatch_id: format!("eval-{case_name}"),
        },
        payload_json: serde_json::to_value(output).expect("serialize OperatorOutput"),
        duration_ms: Some(0),
        error: None,
        version: SCHEMA_VERSION,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// INTEGRATION TEST
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Full record → replay → eval round-trip.
///
/// Steps:
/// 1. Run eval cases through a mock dispatcher (the "live" run).
/// 2. Collect the outputs produced for each case.
/// 3. Serialise those outputs into [`RecordEntry`] replay entries.
/// 4. Feed the entries to [`ReplayDispatcher`].
/// 5. Re-run the same eval cases through the replay dispatcher.
/// 6. Assert that live and replay scores are identical.
#[tokio::test]
async fn record_replay_eval_round_trip() {
    let operator_id = OperatorId::new("mock-agent");
    let cases = make_eval_cases();

    // ── LIVE RUN ──────────────────────────────────────────────────────────────
    //
    // Run the eval cases through the mock dispatcher and record each output so
    // we can build replay entries below.

    let live_dispatcher = Arc::new(InputMappingDispatcher);
    let live_runner = EvalRunner::new(live_dispatcher.clone(), operator_id.clone())
        .with_metric(ExactMatchMetric)
        .with_metric(ContainsMetric)
        .with_concurrency(1);

    let live_report = live_runner.run(cases.clone()).await;

    assert_eq!(
        live_report.total_cases(),
        3,
        "live run: expected 3 cases"
    );
    assert!(
        live_report.cases.iter().all(|c| c.error.is_none()),
        "live run: no case should have an error"
    );

    // ── BUILD REPLAY ENTRIES ──────────────────────────────────────────────────
    //
    // Drive the mock dispatcher once per case to get the outputs, then wrap
    // them in RecordEntry payloads for ReplayDispatcher.
    //
    // We drive the dispatcher directly (not through EvalRunner) so we can
    // capture the raw OperatorOutput for serialisation.

    let mut replay_entries: Vec<RecordEntry> = Vec::new();
    for case in &cases {
        let ctx = DispatchContext::new(
            DispatchId::new(format!("eval-{}", case.name)),
            operator_id.clone(),
        );
        let handle = live_dispatcher
            .dispatch(&ctx, case.input.clone())
            .await
            .unwrap_or_else(|e| panic!("live dispatch for '{}' failed: {e}", case.name));
        let output = handle
            .collect()
            .await
            .unwrap_or_else(|e| panic!("live collect for '{}' failed: {e}", case.name));
        replay_entries.push(make_replay_entry(&case.name, &output));
    }

    assert_eq!(
        replay_entries.len(),
        3,
        "expected one replay entry per case"
    );

    // ── REPLAY RUN ────────────────────────────────────────────────────────────
    //
    // Feed the recorded entries to ReplayDispatcher and run the same eval
    // cases.  No real API calls are made — all outputs come from the entries.

    let replay_dispatcher =
        Arc::new(ReplayDispatcher::new(replay_entries).expect("build ReplayDispatcher"));
    let replay_runner = EvalRunner::new(replay_dispatcher, operator_id)
        .with_metric(ExactMatchMetric)
        .with_metric(ContainsMetric)
        .with_concurrency(1);

    let replay_report = replay_runner.run(cases).await;

    assert_eq!(
        replay_report.total_cases(),
        3,
        "replay run: expected 3 cases"
    );
    assert!(
        replay_report.cases.iter().all(|c| c.error.is_none()),
        "replay run: no case should have an error"
    );

    // ── SCORE COMPARISON ──────────────────────────────────────────────────────
    //
    // Mean scores must be identical between live and replay runs.

    let live_exact = live_report.mean_score("exact_match");
    let replay_exact = replay_report.mean_score("exact_match");
    assert!(
        (live_exact - replay_exact).abs() < f64::EPSILON,
        "exact_match scores diverged: live={live_exact}, replay={replay_exact}"
    );

    let live_contains = live_report.mean_score("contains");
    let replay_contains = replay_report.mean_score("contains");
    assert!(
        (live_contains - replay_contains).abs() < f64::EPSILON,
        "contains scores diverged: live={live_contains}, replay={replay_contains}"
    );

    // Sanity-check the actual mean values so the test is self-documenting.
    //
    // Cases and their expected metric results:
    //   "hello-case"  — ExactText("world")          → exact_match=1.0, contains=0.0
    //   "ping-case"   — Contains(["pong", "ping"])   → exact_match=0.0, contains=1.0
    //   "foo-case"    — ExactText("foo bar baz")     → exact_match=1.0, contains=0.0
    //
    // ExactMatchMetric scores 0.0 when the expectation variant is not ExactText.
    // ContainsMetric scores 0.0 when the expectation variant is not Contains.
    // Mean exact_match = (1.0 + 0.0 + 1.0) / 3 ≈ 0.6667
    // Mean contains    = (0.0 + 1.0 + 0.0) / 3 ≈ 0.3333
    let expected_exact_mean = 2.0_f64 / 3.0;
    let expected_contains_mean = 1.0_f64 / 3.0;

    assert!(
        (live_exact - expected_exact_mean).abs() < 1e-10,
        "live exact_match mean should be ~{expected_exact_mean:.4}, got {live_exact}"
    );
    assert!(
        (live_contains - expected_contains_mean).abs() < 1e-10,
        "live contains mean should be ~{expected_contains_mean:.4}, got {live_contains}"
    );
}
