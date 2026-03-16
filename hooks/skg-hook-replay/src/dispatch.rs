//! [`ReplayDispatcher`] — deterministic replay of recorded dispatch operations.

use crate::{MatchStrategy, ReplayError};
use async_trait::async_trait;
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use layer0::dispatch_context::DispatchContext;
use layer0::error::OrchError;
use layer0::id::DispatchId;
use layer0::operator::{OperatorInput, OperatorOutput};
use skg_hook_recorder::{Boundary, Phase, RecordEntry};
use std::sync::atomic::{AtomicUsize, Ordering};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// REPLAY DISPATCHER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A [`layer0::Dispatcher`] that replays recorded dispatch outcomes.
///
/// Instead of routing to a real orchestrator, `ReplayDispatcher` feeds
/// pre-recorded [`RecordEntry`] responses back to the caller. Each call
/// to [`dispatch`](Self::dispatch) consumes the next matching recording.
///
/// Only [`Boundary::Dispatch`] + [`Phase::Post`] entries are used — pre-phase
/// entries and non-dispatch boundaries are filtered out at construction time.
///
/// # Example
///
/// ```rust,ignore
/// use skg_hook_replay::ReplayDispatcher;
/// use skg_hook_recorder::RecordEntry;
///
/// let entries = vec![/* recorded Post entries */];
/// let dispatcher = ReplayDispatcher::new(entries);
/// ```
pub struct ReplayDispatcher {
    recordings: Vec<RecordEntry>,
    index: AtomicUsize,
    strategy: MatchStrategy,
}

impl ReplayDispatcher {
    /// Create a new dispatcher from a recording sequence.
    ///
    /// Filters for [`Boundary::Dispatch`] + [`Phase::Post`] entries only.
    /// Other entries (Pre-phase, non-Dispatch boundaries) are discarded.
    pub fn new(recordings: Vec<RecordEntry>) -> Self {
        let recordings = recordings
            .into_iter()
            .filter(|e| e.boundary == Boundary::Dispatch && e.phase == Phase::Post)
            .collect();
        Self {
            recordings,
            index: AtomicUsize::new(0),
            strategy: MatchStrategy::Sequential,
        }
    }

    /// Override the match strategy.
    pub fn with_strategy(mut self, strategy: MatchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Find the next recording to use for the given dispatch context.
    ///
    /// Returns `(position, &RecordEntry)` or a [`ReplayError`].
    fn find_recording(
        &self,
        ctx: &DispatchContext,
        position: usize,
    ) -> Result<(usize, &RecordEntry), ReplayError> {
        match &self.strategy {
            MatchStrategy::Sequential | MatchStrategy::ByContentHash => {
                let entry = self
                    .recordings
                    .get(position)
                    .ok_or(ReplayError::RecordingExhausted { position })?;
                Ok((position, entry))
            }
            MatchStrategy::ByOperatorId => {
                let op_id = ctx.operator_id.to_string();
                let entry = self
                    .recordings
                    .iter()
                    .enumerate()
                    .skip(position)
                    .find(|(_, e)| e.context.operator_id == op_id)
                    .ok_or(ReplayError::RecordingExhausted { position })?;
                Ok(entry)
            }
        }
    }
}

#[async_trait]
impl layer0::Dispatcher for ReplayDispatcher {
    async fn dispatch(
        &self,
        ctx: &DispatchContext,
        _input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let position = self.index.fetch_add(1, Ordering::SeqCst);

        let (_, entry) = self
            .find_recording(ctx, position)
            .map_err(|e| OrchError::DispatchFailed(e.to_string()))?;

        // If the recording captured an error, replay it.
        if let Some(ref msg) = entry.error {
            return Err(OrchError::DispatchFailed(msg.clone()));
        }

        // Deserialize the recorded payload as OperatorOutput.
        let output: OperatorOutput = serde_json::from_value(entry.payload_json.clone())
            .map_err(|e| {
                OrchError::DispatchFailed(
                    ReplayError::PayloadError(e.to_string()).to_string(),
                )
            })?;

        // Build a DispatchHandle and immediately send the Completed event.
        let dispatch_id = DispatchId::new(entry.context.dispatch_id.clone());
        let (handle, sender) = DispatchHandle::channel(dispatch_id);
        tokio::spawn(async move {
            let _ = sender.send(DispatchEvent::Completed { output }).await;
        });

        Ok(handle)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TESTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::dispatch_context::DispatchContext;
    use layer0::id::{DispatchId, OperatorId};
    use layer0::operator::{ExitReason, OperatorInput, OperatorOutput, TriggerType};
    use layer0::Dispatcher;
    use skg_hook_recorder::{Boundary, Phase, RecordContext, RecordEntry, SCHEMA_VERSION};

    fn make_output(text: &str) -> OperatorOutput {
        OperatorOutput::new(Content::text(text), ExitReason::Complete)
    }

    fn make_dispatch_post_entry(
        operator_id: &str,
        dispatch_id: &str,
        output: &OperatorOutput,
    ) -> RecordEntry {
        RecordEntry {
            boundary: Boundary::Dispatch,
            phase: Phase::Post,
            context: RecordContext {
                trace_id: String::new(),
                operator_id: operator_id.to_owned(),
                dispatch_id: dispatch_id.to_owned(),
            },
            payload_json: serde_json::to_value(output).expect("serialize"),
            duration_ms: Some(10),
            error: None,
            version: SCHEMA_VERSION,
        }
    }

    fn make_error_post_entry(operator_id: &str) -> RecordEntry {
        RecordEntry {
            boundary: Boundary::Dispatch,
            phase: Phase::Post,
            context: RecordContext {
                trace_id: String::new(),
                operator_id: operator_id.to_owned(),
                dispatch_id: String::new(),
            },
            payload_json: serde_json::Value::Null,
            duration_ms: Some(5),
            error: Some("boom".into()),
            version: SCHEMA_VERSION,
        }
    }

    fn make_ctx(op: &str) -> DispatchContext {
        DispatchContext::new(DispatchId::new("test"), OperatorId::new(op))
    }

    fn make_input() -> OperatorInput {
        OperatorInput::new(Content::text("hello"), TriggerType::User)
    }

    #[tokio::test]
    async fn replay_dispatch_sequential() {
        let outputs = [
            make_output("first"),
            make_output("second"),
            make_output("third"),
        ];
        let entries: Vec<RecordEntry> = outputs
            .iter()
            .enumerate()
            .map(|(i, o)| make_dispatch_post_entry("op", &format!("d-{i}"), o))
            .collect();

        let dispatcher = ReplayDispatcher::new(entries);
        let ctx = make_ctx("op");

        let out0 = dispatcher
            .dispatch(&ctx, make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out0.message.as_text(), Some("first"));

        let out1 = dispatcher
            .dispatch(&ctx, make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out1.message.as_text(), Some("second"));

        let out2 = dispatcher
            .dispatch(&ctx, make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out2.message.as_text(), Some("third"));
    }

    #[tokio::test]
    async fn replay_dispatch_exhausted() {
        let entry = make_dispatch_post_entry("op", "d-0", &make_output("only"));
        let dispatcher = ReplayDispatcher::new(vec![entry]);
        let ctx = make_ctx("op");

        // First call succeeds.
        dispatcher
            .dispatch(&ctx, make_input())
            .await
            .expect("first call");

        // Second call must fail with exhausted error.
        let err = dispatcher
            .dispatch(&ctx, make_input())
            .await
            .expect_err("should be exhausted");
        assert!(
            err.to_string().contains("exhausted"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn replay_dispatch_error_entry() {
        let entry = make_error_post_entry("op");
        let dispatcher = ReplayDispatcher::new(vec![entry]);
        let ctx = make_ctx("op");

        let err = dispatcher
            .dispatch(&ctx, make_input())
            .await
            .expect_err("should replay error");
        assert!(err.to_string().contains("boom"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn replay_dispatch_filters_pre_and_non_dispatch() {
        // Pre-phase and non-Dispatch entries should be ignored.
        let pre = RecordEntry {
            boundary: Boundary::Dispatch,
            phase: Phase::Pre,
            context: RecordContext::empty(),
            payload_json: serde_json::Value::Null,
            duration_ms: None,
            error: None,
            version: SCHEMA_VERSION,
        };
        let real = make_dispatch_post_entry("op", "d-0", &make_output("real"));
        let dispatcher = ReplayDispatcher::new(vec![pre, real]);

        let ctx = make_ctx("op");
        // Only one valid entry, so the first call should succeed with "real".
        let out = dispatcher
            .dispatch(&ctx, make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out.message.as_text(), Some("real"));
    }
}
