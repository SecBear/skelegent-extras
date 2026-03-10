#![deny(missing_docs)]
//! Sweep operator for architectural decision auditing.
//!
//! This crate implements a per-decision sweep pipeline that compares research
//! findings against existing decisions, produces structured [`SweepVerdict`]s,
//! and synthesizes cross-decision patterns via [`SynthesisReport`].
//!
//! # Architecture
//!
//! The crate follows the neuron unified architecture:
//!
//! - **Operators** ([`CompareOperator`], [`SynthesisOperator`]) implement the
//!   [`Operator`](layer0::Operator) trait and receive capabilities via constructor
//!   injection. State access uses [`ScopedState`](neuron_orch_kit::ScopedState)
//!   for partition-isolated reads/writes.
//!
//! - **Composition** ([`run_sweep_cycle`]) is a plain async function that sequences
//!   operator dispatches through an [`Orchestrator`](layer0::Orchestrator), using
//!   [`BudgetTracker`](neuron_orch_kit::BudgetTracker) for cost enforcement and
//!   [`CompositionTrace`](neuron_orch_kit::CompositionTrace) for cycle detection.
//!
//! - **Fractal promotion** ([`SweepCycleOperator`]) wraps the composition function
//!   as an [`Operator`](layer0::Operator), enabling outer systems to dispatch entire
//!   sweep cycles over any transport.
//!
//! [`ResearchSource`] is the trait that backs research — implementations live
//!
//! # Example
//!
//! ```no_run
//! use neuron_op_sweep::{CompareOperator, CompareConfig};
//! use neuron_op_sweep::cycle::run_sweep_cycle;
//!
//! // 1. Create operators with ScopedState + Provider.
//! // 2. Register them on an Orchestrator.
//! // 3. Register ResearchOperator + CompareOperator on an Orchestrator.
//! // 4. Dispatch ResearchOperator, feed results into CompareOperator.
//! // 5. Or call run_sweep_cycle(&orch, &state, &budget, &trace, ...).await

pub mod compare;
pub mod cost;
pub mod cycle;
pub mod provider;
pub mod synthesis;
pub mod synthesis_operator;
pub mod types;
pub mod queries;
pub mod research_operator;

pub use compare::*;
pub use cost::*;
pub use cycle::{CycleReport, SweepCycleOperator, SweepDecision};
pub use provider::*;
pub use synthesis::*;
pub use synthesis_operator::{SynthesisInput, SynthesisOperator};
pub use types::*;
pub use queries::{DecisionQuery, next_query, queries_for};
pub use research_operator::ResearchOperator;
