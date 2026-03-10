#![deny(missing_docs)]
//! CozoDB-backed [`StateStore`] implementation for skelegent.
//!
//! This crate provides [`CozoStore`], a [`StateStore`] backed by CozoDB —
//! a transactional, embedded database with Datalog graph traversal, HNSW
//! vector search, and full-text search.
//!
//! # Backends
//!
//! | Feature | Backend | Native deps |
//! |---|---|---|
//! | *(none, default)* | In-memory [`HashMap`] | None |
//! | `rocksdb` | Real [`cozo::DbInstance`] via RocksDB | cmake + C++ |
//!
//! The default backend is suitable for testing and ephemeral agent memory.
//! Enable `--features rocksdb` for persistent on-disk storage.
//!
//! # Feature Flags
//!
//! - `rocksdb` *(disabled by default)*: activates a real CozoDB backend
//!   backed by RocksDB persistent storage. Pulls in the `cozo` crate
//!   (v0.7.6, MPL-2.0). Requires cmake and a C++ compiler.
//!   Use [`CozoStore::open`] for persistent storage or
//!   [`CozoStore::memory`] for ephemeral in-process use.
//!
//! Without `rocksdb`, `--features rocksdb` is not accepted and
//! [`CozoStore::open`] returns [`layer0::error::StateError::WriteFailed`].
//!
//! [`HashMap`]: std::collections::HashMap
//!
//! # Example
//!
//! ```no_run
//! use skg_state_cozo::CozoStore;
//! use layer0::effect::Scope;
//! use layer0::state::StateStore;
//! use serde_json::json;
//!
//! # tokio_test::block_on(async {
//! let store = CozoStore::memory().expect("in-memory store");
//! store.write(&Scope::Global, "key", json!("value")).await.unwrap();
//! let val = store.read(&Scope::Global, "key").await.unwrap();
//! assert_eq!(val, Some(json!("value")));
//! # });
//! ```
//!
//! [`StateStore`]: layer0::state::StateStore

pub mod engine;
pub mod schema;
pub mod scope;
pub mod store;

pub use engine::{CozoEngine, CozoError};
pub use store::CozoStore;
