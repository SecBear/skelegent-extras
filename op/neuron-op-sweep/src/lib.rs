#![deny(missing_docs)]
//! Sweep operator for the Golden Sweep System.
//!
//! This crate implements Component 4 of the Golden Sweep System: a per-decision
//! research agent that compares Parallel.ai research findings against the current
//! golden decision, and produces a structured [`SweepVerdict`].
//!
//! # Pipeline
//!
//! [`run_sweep`] sequences [`ResearchOperator`] and [`CompareOperator`] through
//! an [`layer0::Orchestrator`], respecting the effects boundary. The pipeline:
//!
//! 1. **Dedup check** — skip if swept too recently (read-only).
//! 2. **Budget guard** — return Skipped if no budget.
//! 3. **Processor selection** — choose tier from budget ratio and previous verdict.
//! 4. **Research dispatch** — [`ResearchOperator`] via the orchestrator.
//! 5. **Compare dispatch** — [`CompareOperator`] via the orchestrator.
//! 6. **Emit** — parse and return the final [`SweepVerdict`].
//!
//! # Design
//!
//! [`ResearchProvider`] is a trait abstraction over the research backend.
//! No HTTP calls are made inside this crate; callers supply a concrete
//! implementation (Parallel.ai, mock, etc.).
//!
//! [`CompareOperator`] is generic over [`neuron_turn::provider::Provider`] —
//! any LLM backend (Anthropic, OpenAI, Ollama) can be wired in.
//!
//! # Example
//!
//! ```no_run
//! use neuron_op_sweep::{ResearchOperator, CompareOperator, SweepOperatorConfig};
//! use neuron_op_sweep::{ResearchProvider, run_sweep};
//! use neuron_op_sweep::types::{ProcessorTier, VerdictStatus};
//!
//! // 1. Create operators with your research and LLM providers.
//! // 2. Register them on a LocalOrchestrator.
//! // 3. Call run_sweep(&orch, &research_id, &compare_id, ...).await
//! ```

pub mod cost;
pub mod operator;
pub mod provider;
pub mod synthesis;
pub mod types;
pub mod sweep_provider;
pub mod workflow;

pub use cost::*;
pub use operator::*;
pub use provider::*;
pub use synthesis::*;
pub use types::*;
pub use sweep_provider::*;
pub use workflow::run_sweep;
