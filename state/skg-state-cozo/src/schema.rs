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

/// DDL for the node relation (v1, without embedding).
///
/// Retained for reference and documentation only. Use [`NODE_V2_DDL`] for
/// new schema initializations.
pub const NODE_DDL: &str = ":create node { scope: String, key: String => data: String, node_type: String, salience: Float, created_at: Float }";

/// Default embedding dimension (OpenAI text-embedding-ada-002).
pub const DEFAULT_EMBEDDING_DIM: usize = 1536;

/// DDL for the node relation with embedding vector.
///
/// Stores typed graph nodes with optional embedding vectors for HNSW search.
/// `salience` is a float in `[0.0, 1.0]` for importance/decay.
pub const NODE_V2_DDL: &str = ":create node { scope: String, key: String => data: String, node_type: String, salience: Float, embedding: <F32; 1536>, created_at: Float }";

/// HNSW index on node embeddings for vector similarity search.
///
/// # Note
///
/// This DDL **must** be executed as a separate [`run_script`] call after the
/// relation DDL — CozoDB processes `::hnsw create` (index DDL) differently
/// from `:create` (relation DDL) and rejects mixed scripts.
pub const NODE_HNSW_DDL: &str = r#"::hnsw create node:emb_idx {
    dim: 1536,
    m: 16,
    dtype: F32,
    fields: [embedding],
    distance: Cosine,
    ef_construction: 200,
}"#;

/// DDL for the edge relation.
///
/// Stores typed directed edges between `(scope, from_key)` and
/// `(scope, to_key)`. `relation` is a string label (e.g. `"references"`,
/// `"supersedes"`). `metadata` is JSON-encoded edge properties.
pub const EDGE_DDL: &str = ":create edge { scope: String, from_key: String, to_key: String, relation: String => metadata: String, created_at: Float }";

/// FTS index on the kv relation's value field.
///
/// Uses Simple tokenizer with Lowercase filter for language-neutral text search.
///
/// # Note
///
/// This DDL **must** be executed as a separate [`run_script`] call from
/// [`DDL_INIT`] — CozoDB processes `::fts create` (index DDL) differently
/// from `:create` (relation DDL) and rejects mixed scripts.
pub const KV_FTS_DDL: &str = r#"::fts create kv:fts_val {
    extractor: value,
    tokenizer: Simple,
    filters: [Lowercase],
}"#;

/// DDL for the transient key-value relation.
///
/// Stores entries written with [`Lifetime::Transient`]. Cleared at turn
/// boundaries via [`CozoStore::clear_transient`]. Never promoted to the
/// durable `kv` relation.
///
/// [`Lifetime::Transient`]: layer0::state::Lifetime
/// [`CozoStore::clear_transient`]: crate::store::CozoStore
pub const TRANSIENT_DDL: &str =
    ":create kv_transient { scope: String, key: String => value: String, created_at: Float }";

/// Complete schema initialization script.
///
/// Runs all four relation DDL statements as a single batch. In CozoDB, `:create` is
/// idempotent — it creates the relation only if it does not exist. This script
/// can be executed multiple times on the same database safely.
///
/// This string is a verbatim concatenation of [`KV_DDL`], [`NODE_V2_DDL`],
/// [`EDGE_DDL`], and [`TRANSIENT_DDL`], separated by newlines. When the
/// `rocksdb` feature is enabled, pass this to `DbInstance::run_script` inside
/// [`CozoEngine::ensure_schema`].
///
/// **Note:** [`KV_FTS_DDL`] and [`NODE_HNSW_DDL`] are intentionally excluded —
/// index DDL must be run as separate script calls.
pub const DDL_INIT: &str = "\
:create kv { scope: String, key: String => value: String, created_at: Float }
:create node { scope: String, key: String => data: String, node_type: String, salience: Float, embedding: <F32; 1536>, created_at: Float }
:create edge { scope: String, from_key: String, to_key: String, relation: String => metadata: String, created_at: Float }
:create kv_transient { scope: String, key: String => value: String, created_at: Float }";