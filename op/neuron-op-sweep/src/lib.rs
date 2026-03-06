#![deny(missing_docs)]
//! Sweep operator for architectural decision auditing.
//!
//! This crate implements a per-decision sweep agent that compares research
//! findings against the current decision position, and produces a structured
//! [`SweepVerdict`].
//!
//! # Pipeline
//!
//! [`run_sweep`] sequences [`CompareOperator`] through an [`layer0::Orchestrator`],
//! respecting the effects boundary. Research results are supplied by the caller.
//! The pipeline:
//!
//! 1. **Dedup check** — skip if swept too recently (read-only).
//! 2. **Budget guard** — return Skipped if no budget.
//! 3. **Processor selection** — choose tier from budget ratio and previous verdict.
//! 4. **Compare dispatch** — [`CompareOperator`] via the orchestrator.
//! 5. **Emit** — parse and return the final [`SweepVerdict`].
//!
//! Context assembly (decision card, prior deltas) is **turn-owned**:
//! [`CompareOperator`] reads from the state store during `execute()`.
//!
//! # Design
//!
//! [`CompareOperator`] is generic over [`neuron_turn::provider::Provider`] —
//! any LLM backend (Anthropic, OpenAI, Ollama) can be wired in.
//!
//! Research is orchestrator-owned: callers run their own research pipeline and
//! pass results directly to [`run_sweep`].
//!
//! # Example
//!
//! ```no_run
//! use neuron_op_sweep::{CompareOperator, SweepOperatorConfig};
//! use neuron_op_sweep::provider::ResearchResult;
//!
//! // 1. Create CompareOperator with your LLM provider.
//! // 2. Register it on a LocalOrchestrator.
//! // 3. Collect research results externally.
//! // 4. Call run_sweep(&orch, &compare_id, decision_id, &results, ...).await
//! ```

pub mod cost;
pub mod operator;
pub mod provider;
pub mod synthesis;
pub mod types;
pub mod workflow;

pub use cost::*;
pub use operator::*;
pub use provider::*;
pub use synthesis::*;
pub use types::*;
pub use workflow::run_sweep;
