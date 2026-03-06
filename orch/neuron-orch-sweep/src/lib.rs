#![deny(missing_docs)]
//! Sweep orchestrator for architectural decision auditing.
//!
//! This crate provides priority scheduling, daily budget
//! enforcement, and sweep-cycle management.  It is intentionally free of
//! scheduling (cron/timers); callers are responsible for invoking
//! [`cycle::run_cycle`] on their preferred cadence.
//!
//! # Modules
//!
//! - [`priority`] — [`priority::QueuedDecision`] and
//!   [`priority::compute_priority`] for ranking decisions.
//! - [`budget`] — [`budget::BudgetConfig`], [`budget::BudgetState`], and
//!   [`budget::DegradationLevel`] for cost enforcement.
//! - [`cycle`] — [`cycle::run_cycle`], [`cycle::OrchestratorConfig`], and
//!   [`cycle::CycleReport`] for executing one sweep cycle.

pub mod budget;
pub mod cycle;
pub mod priority;

pub use budget::{BudgetConfig, BudgetState, DegradationLevel};
pub use cycle::{CycleReport, OrchestratorConfig, RateLimiter, run_cycle};
pub use priority::{QueuedDecision, compute_priority};
