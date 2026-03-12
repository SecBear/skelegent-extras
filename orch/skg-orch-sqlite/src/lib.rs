#![deny(missing_docs)]
//! SQLite durable orchestrator assembly for skelegent.
//!
//! This crate assembles the portable durable run/control substrate from
//! [`skg_run_core`] with the SQLite persistence backend from [`skg_run_sqlite`]
//! and in-process operator dispatch. It is intentionally minimal: operators can
//! start, wait without wake deadlines, resume, complete, fail, and be queried or
//! cancelled durably. Unsupported semantics such as operator effects, timed waits,
//! replay history, or durable signal delivery are rejected explicitly rather than
//! implied.

mod controller;
mod driver;

pub use controller::SqliteDurableOrchestrator;
pub use driver::{DurableDirective, SqliteRunDriver};
