#![deny(missing_docs)]
//! Deterministic replay engine for skelegent recorded operations.
//!
//! Feed recorded [`RecordEntry`] sequences back through the system,
//! returning recorded responses instead of making real calls.
//!
//! # Overview
//!
//! The replay engine reads [`RecordEntry`] sequences produced by the
//! `skg-hook-recorder` crate and replays them deterministically.
//! Instead of making real dispatch or provider calls, the engine
//! returns pre-recorded responses.
//!
//! # Modules
//!
//! - [`dispatch`] — [`ReplayDispatcher`] implementing [`layer0::Dispatcher`]
//! - [`infer`] — [`ReplayProvider`] implementing [`skg_turn::provider::Provider`]

pub mod dispatch;
pub mod infer;

pub use dispatch::ReplayDispatcher;
pub use infer::ReplayProvider;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ERRORS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Errors produced by the replay engine.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// Recording schema version does not match the expected version.
    #[error("version mismatch: recorded={recorded}, current={current}")]
    VersionMismatch {
        /// Version found in the recording.
        recorded: u64,
        /// Version expected by this engine.
        current: u64,
    },

    /// The operator at this position does not match what was recorded.
    #[error("operator mismatch at position {position}: expected={expected}, got={got}")]
    OperatorMismatch {
        /// Position in the recording sequence.
        position: usize,
        /// Operator ID found in the recording.
        expected: String,
        /// Operator ID from the live dispatch context.
        got: String,
    },

    /// The recording has been exhausted — more calls than recorded entries.
    #[error("recording exhausted at position {position}")]
    RecordingExhausted {
        /// Position at which the recording ran out.
        position: usize,
    },

    /// Failed to deserialize the recorded payload.
    #[error("payload error: {0}")]
    PayloadError(String),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MATCH STRATEGY
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Strategy for matching live calls to recorded entries.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub enum MatchStrategy {
    /// Match the nth call to the nth recording in sequence.
    ///
    /// This is the default strategy, suitable for deterministic
    /// workflows where calls always happen in the same order
    /// (e.g., Temporal-style replay).
    #[default]
    Sequential,

    /// Match by the `operator_id` field in the recording context.
    ///
    /// Scans the recording for the first unused entry whose
    /// `context.operator_id` matches the live dispatch's operator ID.
    ByOperatorId,

    /// Content-hash matching for replay.
    ///
    /// Currently behaves as [`Sequential`](Self::Sequential) — matching by
    /// content hash will be implemented in a future version.
    ///
    /// When implemented, this will allow replay entries to be matched by
    /// their payload hash rather than position, enabling tolerance for
    /// operation reordering between recording and replay.
    ///
    /// # Note
    ///
    /// This variant is preserved for forward compatibility. Removing it
    /// would be a semver-breaking change.
    ByContentHash,
}
