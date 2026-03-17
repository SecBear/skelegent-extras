//! [`ReplayDispatcher`] — deterministic replay of recorded dispatch operations.

use crate::{MatchStrategy, ReplayError};
use async_trait::async_trait;
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use layer0::dispatch_context::DispatchContext;
use layer0::error::OrchError;
use layer0::id::DispatchId;
use layer0::operator::{OperatorInput, OperatorOutput};
use skg_hook_recorder::{Boundary, Phase, RecordEntry, SCHEMA_VERSION};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    /// Sequential index — used only by [`MatchStrategy::Sequential`] and [`MatchStrategy::ByContentHash`].
    sequential_index: AtomicUsize,
    /// Per-entry consumed flags — used only by [`MatchStrategy::ByOperatorId`].
    consumed: Vec<AtomicBool>,
    strategy: MatchStrategy,
}

impl ReplayDispatcher {
    /// Create a new dispatcher from a recording sequence.
    ///
    /// Filters for [`Boundary::Dispatch`] + [`Phase::Post`] entries only.
    /// Other entries (Pre-phase, non-Dispatch boundaries) are discarded.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::VersionMismatch`] if any entry's `version` does
    /// not match [`skg_hook_recorder::SCHEMA_VERSION`].
    pub fn new(recordings: Vec<RecordEntry>) -> Result<Self, ReplayError> {
        for entry in &recordings {
            if entry.version != SCHEMA_VERSION {
                return Err(ReplayError::VersionMismatch {
                    recorded: entry.version,
                    current: SCHEMA_VERSION,
                });
            }
        }
        let recordings: Vec<RecordEntry> = recordings
            .into_iter()
            .filter(|e| e.boundary == Boundary::Dispatch && e.phase == Phase::Post)
            .collect();
        let consumed = recordings.iter().map(|_| AtomicBool::new(false)).collect();
        Ok(Self {
            recordings,
            sequential_index: AtomicUsize::new(0),
            consumed,
            strategy: MatchStrategy::Sequential,
        })
    }

    /// Override the match strategy.
    pub fn with_strategy(mut self, strategy: MatchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Find the next recording to use for the given dispatch context.
    ///
    /// For [`MatchStrategy::Sequential`] and [`MatchStrategy::ByContentHash`]:
    /// uses the sequential index to find the next entry in order.
    ///
    /// For [`MatchStrategy::ByOperatorId`]: scans ALL entries (not just those
    /// after the current position), skips already-consumed entries, and returns
    /// the first unconsumed entry whose `operator_id` matches. This allows
    /// entries to be consumed out of order.
    ///
    /// Returns `(position, &RecordEntry)` or a [`ReplayError`].
    fn find_recording(
        &self,
        ctx: &DispatchContext,
    ) -> Result<(usize, &RecordEntry), ReplayError> {
        match &self.strategy {
            MatchStrategy::Sequential | MatchStrategy::ByContentHash => {
                let position = self.sequential_index.fetch_add(1, Ordering::SeqCst);
                let entry = self
                    .recordings
                    .get(position)
                    .ok_or(ReplayError::RecordingExhausted { position })?;
                Ok((position, entry))
            }
            MatchStrategy::ByOperatorId => {
                let op_id = ctx.operator_id.to_string();
                // Scan all entries, find first unconsumed match.
                let found = self
                    .recordings
                    .iter()
                    .zip(self.consumed.iter())
                    .enumerate()
                    .find(|(_, (e, consumed))| {
                        !consumed.load(Ordering::SeqCst) && e.context.operator_id == op_id
                    });
                match found {
                    Some((idx, _)) => {
                        // Mark as consumed before returning.
                        self.consumed[idx].store(true, Ordering::SeqCst);
                        Ok((idx, &self.recordings[idx]))
                    }
                    None => Err(ReplayError::RecordingExhausted {
                        position: self.sequential_index.load(Ordering::SeqCst),
                    }),
                }
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
        let (_, entry) = self
            .find_recording(ctx)
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

        let dispatcher = ReplayDispatcher::new(entries).unwrap();
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
        let dispatcher = ReplayDispatcher::new(vec![entry]).unwrap();
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
        let dispatcher = ReplayDispatcher::new(vec![entry]).unwrap();
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
        let dispatcher = ReplayDispatcher::new(vec![pre, real]).unwrap();

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

    #[tokio::test]
    async fn replay_dispatch_by_operator_id() {
        // Three entries with different operator_ids — consume out of order.
        let entries = vec![
            make_dispatch_post_entry("op-a", "d-a", &make_output("alpha")),
            make_dispatch_post_entry("op-b", "d-b", &make_output("beta")),
            make_dispatch_post_entry("op-c", "d-c", &make_output("gamma")),
        ];

        let dispatcher = ReplayDispatcher::new(entries)
            .unwrap()
            .with_strategy(MatchStrategy::ByOperatorId);

        // Consume op-c first (index 2), then op-a (index 0), then op-b (index 1).
        let out_c = dispatcher
            .dispatch(&make_ctx("op-c"), make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out_c.message.as_text(), Some("gamma"));

        let out_a = dispatcher
            .dispatch(&make_ctx("op-a"), make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out_a.message.as_text(), Some("alpha"));

        let out_b = dispatcher
            .dispatch(&make_ctx("op-b"), make_input())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(out_b.message.as_text(), Some("beta"));

        // All consumed — next call for any op should be exhausted.
        let err = dispatcher
            .dispatch(&make_ctx("op-a"), make_input())
            .await
            .expect_err("should be exhausted");
        assert!(
            err.to_string().contains("exhausted"),
            "unexpected error: {err}"
        );
    }
}
