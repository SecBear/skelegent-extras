#![deny(missing_docs)]
//! A2A protocol server and client for skelegent.

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "client")]
pub mod client;
