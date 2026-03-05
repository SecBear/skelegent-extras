#![deny(missing_docs)]
//! Sweep operator for the Golden Sweep System.
//!
//! This crate implements Component 4 of the Golden Sweep System: a per-decision
//! research agent that compares Parallel.ai research findings against the current
//! golden decision, and produces a structured [`SweepVerdict`].
//!
//! # Pipeline
//!
//! [`SweepOperator::run`] executes an 8-step pipeline per decision:
//!
//! 1. **Dedup check** — skip if swept too recently.
//! 2. **Query plan** — build research queries.
//! 3. **Context assembly** — assemble packed context from state store.
//! 4. **Research** — execute queries via [`ResearchProvider`] with retry.
//! 5. **Artifact storage** — persist raw results (requires write access).
//! 6. **Compare** — LLM comparison of research against the decision.
//! 7. **State update** — write delta, card, and sweep metadata.
//! 8. **Emit** — return the structured [`SweepVerdict`].
//!
//! Alternatively, use [`run_sweep`] with [`ResearchOperator`] and
//! [`CompareOperator`] dispatched through an [`layer0::Orchestrator`] to
//! respect the effects boundary.
//!
//! # Design
//!
//! [`ResearchProvider`] is a trait abstraction over the research backend.
//! No HTTP calls are made inside this crate; callers supply a concrete
//! implementation (Parallel.ai, mock, etc.).
//!
//! # Example
//!
//! ```no_run
//! use neuron_op_sweep::{SweepOperator, SweepOperatorConfig, ResearchProvider};
//! use neuron_op_sweep::{ResearchResult, SweepVerdict, SweepError};
//! use neuron_op_sweep::types::{ProcessorTier, VerdictStatus};
//!
//! // Implement ResearchProvider for your backend, then:
//! // let op = SweepOperator::new(SweepOperatorConfig::default(), Box::new(your_provider));
//! // let verdict = op.run("D3B", None, 8.0, 10.0, &store).await?;
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
