//! CozoDB relation DDL for the skg-state-cozo schema.
//!
//! These constants define the Datalog DDL scripts executed by
//! [`CozoEngine::ensure_schema`]. They are exposed for external tooling and
//! documentation; callers should use [`CozoEngine::ensure_schema`] rather than
//! executing these strings directly.
//!
//! # Relation Summary
//!
//! | Relation | Purpose |
//! |---|---|
//! | `kv` | Key-value store: CRUD operations |
//! | `node` | Graph nodes with salience and type metadata |
//! | `edge` | Typed directed edges between nodes |
//!
//! [`CozoEngine::ensure_schema`]: crate::engine::CozoEngine::ensure_schema

/// DDL for the key-value relation.
///
/// Stores arbitrary JSON values keyed by `(scope, key)`. The `created_at`
/// column is a Unix timestamp (seconds since epoch as a float).
pub const KV_DDL: &str =
    ":create kv { scope: String, key: String => value: String, created_at: Float }";

/// DDL for the node relation.
///
/// Stores typed graph nodes. `node_type` is an application-defined string
/// (e.g. `"source"`, `"claim"`, `"concept"`). `salience` is a float in
/// `[0.0, 1.0]` indicating importance; used for decay and pruning.
pub const NODE_DDL: &str = ":create node { scope: String, key: String => data: String, node_type: String, salience: Float, created_at: Float }";

/// DDL for the edge relation.
///
/// Stores typed directed edges between `(scope, from_key)` and
/// `(scope, to_key)`. `relation` is a string label (e.g. `"references"`,
/// `"supersedes"`). `metadata` is JSON-encoded edge properties.
pub const EDGE_DDL: &str = ":create edge { scope: String, from_key: String, to_key: String, relation: String => metadata: String, created_at: Float }";

/// Complete schema initialization script.
///
/// Runs all three DDL statements as a single batch. In CozoDB, `:create` is
/// idempotent — it creates the relation only if it does not exist. This script
/// can be executed multiple times on the same database safely.
///
/// This string is a verbatim concatenation of [`KV_DDL`], [`NODE_DDL`], and
/// [`EDGE_DDL`], separated by newlines. When the `rocksdb` feature is enabled,
/// pass this to `DbInstance::run_script` inside [`CozoEngine::ensure_schema`].
pub const DDL_INIT: &str = "\
:create kv { scope: String, key: String => value: String, created_at: Float }
:create node { scope: String, key: String => data: String, node_type: String, salience: Float, created_at: Float }
:create edge { scope: String, from_key: String, to_key: String, relation: String => metadata: String, created_at: Float }";
