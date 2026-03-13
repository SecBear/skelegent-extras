#![deny(missing_docs)]
//! SQLite-backed durable run persistence seams for skelegent.
//!
//! This crate implements the lower-level durable run seams from
//! [`skg_run_core`] without assembling a full orchestrator.

mod schema;
mod store;
mod timer;

pub use store::SqliteRunStore;
