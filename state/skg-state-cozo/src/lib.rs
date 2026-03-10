#![deny(missing_docs)]
//! CozoDB-backed [`StateStore`] implementation for skelegent.
//!
//! This crate provides [`CozoStore`], a [`StateStore`] backed by CozoDB —
//! a transactional, embedded database with Datalog graph traversal, HNSW
//! vector search, and full-text search.
//!
//! # Capabilities
//!
//! - **FTS search**: the `kv:fts_val` index enables BM25-ranked full-text
//!   search over stored values via [`StateStore::search`].
//! - **HNSW vector search**: the `node:emb_idx` index enables approximate
//!   nearest-neighbour search over 1536-dimensional F32 embeddings via
//!   [`CozoStore::vector_search`].
//! - **Hybrid search**: [`CozoStore::hybrid_search`] runs FTS and HNSW
//!   in parallel and fuses results with Reciprocal Rank Fusion (RRF).
//! - **Graph edges**: [`StateStore::link`] / [`StateStore::unlink`] store
//!   typed directed edges; [`StateStore::traverse`] performs level-batched
//!   BFS using Datalog `is_in` predicates.
//! - **Transient table**: [`StateStore::write_hinted`] with `Lifetime::Transient`
//!   writes to a dedicated `kv_transient` relation, cleared at turn boundaries
//!   by [`CozoStore::clear_transient`].
//! - **HashMap backend**: the default (no-feature) build uses an in-memory
//!   [`HashMap`] — no native dependencies, suitable for testing. Graph link/
//!   traverse works; FTS, HNSW, and hybrid search require the `cozo` feature.
//!
//! # Backends
//!
//! | Feature | Backend | Native deps |
//! |---|---|---|
//! | *(none, default)* | In-memory [`HashMap`] | None |
//! | `cozo` | Real [`cozo::DbInstance`] (in-process Datalog) | None |
//! | `rocksdb` | Persistent RocksDB storage (implies `cozo`) | cmake + C++ |
//!
//! The default backend is suitable for testing and ephemeral agent memory.
//! Enable `--features cozo` for the full Datalog/FTS/HNSW feature set.
//! Add `--features rocksdb` for on-disk persistence.
//!
//! # Feature Flags
//!
//! - `cozo` *(disabled by default)*: activates a real CozoDB backend with
//!   pure-Rust in-memory Datalog storage. Enables FTS, HNSW vector search,
//!   hybrid search, and graph traversal via Datalog queries.
//! - `rocksdb` *(disabled by default)*: adds persistent RocksDB storage on
//!   top of `cozo` (implies the `cozo` feature). Pulls in the `cozo` crate
//!   (v0.7.6, MPL-2.0). Requires cmake and a C++ compiler.
//!   Use [`CozoStore::open`] for persistent storage or
//!   [`CozoStore::memory`] for ephemeral in-process use.
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

pub mod search;

pub mod engine;
pub mod schema;
pub mod scope;
pub mod store;

pub use engine::{CozoEngine, CozoError};
pub use store::CozoStore;
