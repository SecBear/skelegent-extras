//! [`CozoStore`]: [`StateStore`] implementation backed by [`CozoEngine`].
//!
//! # v1 Implementation Notes
//!
//! - `search` performs basic substring matching on keys and JSON-encoded values.
//!   Full hybrid retrieval (HNSW + BM25 + Datalog graph expansion + RRF) is
//!   planned for v2 when the `cozo` feature is fully activated.
//! - `clear_transient` is a no-op in v1; transient entries share the durable
//!   store. A dedicated transient table is planned for v2.
//! - `read_hinted` and `write_hinted` delegate to `read`/`write`, ignoring
//!   advisory hints. Backends may honour hints in future revisions.

#[cfg(not(feature = "cozo"))]
use crate::engine::EdgeRecord;
use crate::engine::CozoEngine;
use crate::scope::scope_prefix;
#[cfg(not(feature = "cozo"))]
use crate::scope::{composite_key, extract_key};
use async_trait::async_trait;
use layer0::effect::Scope;
use layer0::error::StateError;
use layer0::state::{MemoryLink, SearchOptions, SearchResult, StateStore, StoreOptions};
use std::collections::{HashSet, VecDeque};
use tracing::instrument;

/// CozoDB-backed [`StateStore`] with graph and hybrid search support.
///
/// # Backends
///
/// - **Default** (no `cozo` feature): wraps an in-memory [`HashMap`] suitable
///   for testing and single-process use without durability.
/// - **`cozo` feature**: wraps a real CozoDB `DbInstance` with in-memory
///   Datalog storage, enabling Datalog graph traversal, HNSW vector search,
///   and FTS.
/// - **`rocksdb` feature**: adds persistent RocksDB storage on top of `cozo`.
///
/// `CozoStore` is cheaply cloneable when needed — both the store and its
/// underlying [`CozoEngine`] share the same storage via [`Arc`].
///
/// # Example
///
/// ```no_run
/// use skg_state_cozo::CozoStore;
/// use layer0::effect::Scope;
/// use layer0::state::StateStore;
/// use serde_json::json;
///
/// # tokio_test::block_on(async {
/// let store = CozoStore::memory().unwrap();
/// store.write(&Scope::Global, "key", json!("value")).await.unwrap();
/// let val = store.read(&Scope::Global, "key").await.unwrap();
/// assert_eq!(val, Some(json!("value")));
/// # });
/// ```
///
/// [`HashMap`]: std::collections::HashMap
/// [`Arc`]: std::sync::Arc
pub struct CozoStore {
    engine: CozoEngine,
}

impl CozoStore {
    /// Create a `CozoStore` from an existing [`CozoEngine`].
    ///
    /// The engine should already have its schema initialized via
    /// [`CozoEngine::ensure_schema`].
    pub fn new(engine: CozoEngine) -> Self {
        Self { engine }
    }

    /// Create an in-memory `CozoStore`.
    ///
    /// Initializes a fresh in-memory engine and applies the schema. Safe to
    /// call multiple times — each call produces an independent store.
    ///
    /// # Errors
    ///
    /// Returns a [`StateError`] if the engine or schema initialization fails.
    pub fn memory() -> Result<Self, StateError> {
        let engine = CozoEngine::memory().map_err(StateError::from)?;
        engine.ensure_schema().map_err(StateError::from)?;
        Ok(Self { engine })
    }

    /// Open a persistent [`CozoStore`] at the given filesystem path.
    ///
    /// Initializes or reopens a CozoDB RocksDB database and applies the schema.
    ///
    /// **Requires the `rocksdb` feature.** Without it, this always returns
    /// [`StateError::WriteFailed`] with `"rocksdb feature not enabled"`.
    ///
    /// # Errors
    ///
    /// Returns a [`StateError`] if the engine cannot be opened or the schema
    /// initialization fails.
    pub fn open(_path: &str) -> Result<Self, StateError> {
        #[cfg(not(feature = "rocksdb"))]
        return Err(StateError::WriteFailed(
            "rocksdb feature not enabled".to_string(),
        ));
        #[cfg(feature = "rocksdb")]
        {
            let engine = CozoEngine::open(_path).map_err(StateError::from)?;
            engine.ensure_schema().map_err(StateError::from)?;
            Ok(Self { engine })
        }
    }
}

impl std::fmt::Debug for CozoStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CozoStore").finish_non_exhaustive()
    }
}

// ── Tier 2: CozoStore-only methods ───────────────────────────────────────────

/// Stub impl for the HashMap backend: vector operations are not supported.
#[cfg(not(feature = "cozo"))]
impl CozoStore {
    /// Write a node with embedding vector (Tier 2 — CozoStore only).
    ///
    /// The HashMap backend does not support HNSW. Returns [`StateError::WriteFailed`].
    pub async fn write_node(
        &self,
        _scope: &Scope,
        _key: &str,
        _data: serde_json::Value,
        _node_type: &str,
        _salience: f64,
        _embedding: &[f32],
    ) -> Result<(), StateError> {
        Err(StateError::WriteFailed(
            "HNSW requires cozo feature".to_string(),
        ))
    }

    /// Vector similarity search using HNSW index (Tier 2 — CozoStore only).
    ///
    /// The HashMap backend does not support HNSW. Returns [`StateError::WriteFailed`].
    pub async fn vector_search(
        &self,
        _scope: &Scope,
        _query_vector: &[f32],
        _limit: usize,
    ) -> Result<Vec<SearchResult>, StateError> {
        Err(StateError::WriteFailed(
            "HNSW requires cozo feature".to_string(),
        ))
    }
}

/// Helper: build a `BTreeMap<String, DataValue>` from key-value pairs.
#[cfg(feature = "cozo")]
macro_rules! cozo_params {
    ($($key:expr => $val:expr),* $(,)?) => {{
        let mut m = BTreeMap::<String, DataValue>::new();
        $( m.insert($key.to_string(), $val); )*
        m
    }};
}

/// Real HNSW impl for the CozoDB backend.
#[cfg(feature = "cozo")]
impl CozoStore {
    /// Write a node with embedding vector (Tier 2 — CozoStore only).
    ///
    /// Stores data alongside a 1536-dimensional F32 embedding for HNSW
    /// similarity search via [`CozoStore::vector_search`].
    pub async fn write_node(
        &self,
        scope: &Scope,
        key: &str,
        data: serde_json::Value,
        node_type: &str,
        salience: f64,
        embedding: &[f32],
    ) -> Result<(), StateError> {
        let sp = scope_prefix(scope);
        let data_str = serde_json::to_string(&data)
            .map_err(|e| StateError::Serialization(e.to_string()))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let emb_vec = DataValue::Vec(Vector::F32(ndarray::Array1::from_vec(
            embedding.to_vec(),
        )));
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "key" => DataValue::Str(key.to_string().into()),
            "data" => DataValue::Str(data_str.into()),
            "node_type" => DataValue::Str(node_type.to_string().into()),
            "salience" => DataValue::from(salience),
            "embedding" => emb_vec,
            "now" => DataValue::from(now),
        };
        self.engine
            .run_mutation(
                "?[scope, key, data, node_type, salience, embedding, created_at] <- \
                [[$scope, $key, $data, $node_type, $salience, $embedding, $now]] \
                :put node {scope, key => data, node_type, salience, embedding, created_at}",
                params,
            )
            .map_err(StateError::from)?;
        Ok(())
    }

    /// Vector similarity search using HNSW index (Tier 2 — CozoStore only).
    ///
    /// Returns the `limit` nearest neighbors by cosine distance. Scoped to
    /// `scope` — results from other scopes are excluded.
    pub async fn vector_search(
        &self,
        scope: &Scope,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, StateError> {
        if limit == 0 {
            return Ok(vec![]);
        }
        let sp = scope_prefix(scope);
        let query_dv = DataValue::Vec(Vector::F32(ndarray::Array1::from_vec(
            query_vector.to_vec(),
        )));
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "vec" => query_dv,
            "limit" => DataValue::from(limit as i64),
        };
        let q = concat!(
            "?[key, data, dist] := ",
            "~node:emb_idx {scope, key, data | query: $vec, k: $limit, ef: 64, bind_distance: dist},",
            " scope == $scope\n",
            ":order dist\n",
            ":limit $limit",
        );
        let rows = self
            .engine
            .run_query(q, params)
            .map_err(StateError::from)?;
        let results: Vec<SearchResult> = rows
            .rows
            .iter()
            .filter_map(|row| {
                let key = dv_as_string(&row[0])?;
                let data_str = dv_as_string(&row[1])?;
                let dist = dv_as_f64(&row[2]).unwrap_or(1.0);
                // Score: 1 − distance (cosine distance ∈ [0, 2]; 0 = identical).
                let score = 1.0 - dist;
                let mut sr = SearchResult::new(key, score);
                sr.snippet = Some(data_str.chars().take(120).collect());
                Some(sr)
            })
            .collect();
        Ok(results)
    }
}

// ── HashMap backend (default, no native deps) ─────────────────────────────────

#[cfg(not(feature = "cozo"))]
#[async_trait]
impl StateStore for CozoStore {
    #[instrument(skip_all, fields(scope = ?scope, key = %key))]
    async fn read(
        &self,
        scope: &Scope,
        key: &str,
    ) -> Result<Option<serde_json::Value>, StateError> {
        let ck = composite_key(scope, key);
        let inner = self.engine.inner.read().await;
        match inner.kv.get(&ck) {
            None => Ok(None),
            Some(raw) => {
                let val: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|e| StateError::Serialization(e.to_string()))?;
                Ok(Some(val))
            }
        }
    }

    #[instrument(skip_all, fields(scope = ?scope, key = %key))]
    async fn write(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), StateError> {
        let ck = composite_key(scope, key);
        let raw = serde_json::to_string(&value)
            .map_err(|e| StateError::Serialization(e.to_string()))?;
        let mut inner = self.engine.inner.write().await;
        inner.kv.insert(ck, raw);
        Ok(())
    }

    #[instrument(skip_all, fields(scope = ?scope, key = %key))]
    async fn delete(&self, scope: &Scope, key: &str) -> Result<(), StateError> {
        let ck = composite_key(scope, key);
        let mut inner = self.engine.inner.write().await;
        inner.kv.remove(&ck);
        Ok(())
    }

    #[instrument(skip_all, fields(scope = ?scope))]
    async fn list(&self, scope: &Scope, prefix: &str) -> Result<Vec<String>, StateError> {
        let scope_pfx = scope_prefix(scope);
        let inner = self.engine.inner.read().await;
        let keys = inner
            .kv
            .keys()
            .filter_map(|ck| {
                extract_key(ck, &scope_pfx).and_then(|k| {
                    if k.starts_with(prefix) {
                        Some(k.to_string())
                    } else {
                        None
                    }
                })
            })
            .collect();
        Ok(keys)
    }

    /// Search for entries whose key or JSON-encoded value contains `query`
    /// (case-insensitive substring match).
    ///
    /// Results are sorted by score (descending). Keys that match both the key
    /// name and the value receive a score of `1.0`; single-field matches
    /// receive `0.5`. The full hybrid retrieval pipeline (HNSW + BM25 +
    /// Datalog graph expansion) is planned for v2.
    #[instrument(skip_all, fields(scope = ?scope))]
    async fn search(
        &self,
        scope: &Scope,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StateError> {
        if query.is_empty() || limit == 0 {
            return Ok(vec![]);
        }
        let scope_pfx = scope_prefix(scope);
        let query_lower = query.to_lowercase();
        let inner = self.engine.inner.read().await;

        let mut results: Vec<SearchResult> = inner
            .kv
            .iter()
            .filter_map(|(ck, raw)| {
                let user_key = extract_key(ck, &scope_pfx)?;
                let key_hit = user_key.to_lowercase().contains(&query_lower);
                let val_hit = raw.to_lowercase().contains(&query_lower);
                if key_hit || val_hit {
                    let score = if key_hit && val_hit { 1.0 } else { 0.5 };
                    let mut sr = SearchResult::new(user_key.to_string(), score);
                    // Snippet: first 120 characters of the raw JSON value.
                    sr.snippet = Some(raw.chars().take(120).collect());
                    Some(sr)
                } else {
                    None
                }
            })
            .collect();

        // Stable descending sort by score.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    async fn read_hinted(
        &self,
        scope: &Scope,
        key: &str,
        _options: &StoreOptions,
    ) -> Result<Option<serde_json::Value>, StateError> {
        self.read(scope, key).await
    }

    async fn write_hinted(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
        _options: &StoreOptions,
    ) -> Result<(), StateError> {
        self.write(scope, key, value).await
    }

    /// No-op in v1.
    ///
    /// CozoDB v1 does not maintain a separate transient table. Entries written
    /// with [`Lifetime::Transient`] are stored durably and not cleared here.
    /// A dedicated transient store is planned for v2.
    ///
    /// [`Lifetime::Transient`]: layer0::state::Lifetime
    fn clear_transient(&self) {}

    async fn link(&self, scope: &Scope, link: &MemoryLink) -> Result<(), StateError> {
        let scope_pfx = scope_prefix(scope);
        let record = EdgeRecord {
            scope_prefix: scope_pfx.clone(),
            from_key: link.from_key.clone(),
            to_key: link.to_key.clone(),
            relation: link.relation.clone(),
            metadata: link.metadata.clone(),
        };
        let mut inner = self.engine.inner.write().await;
        // Deduplicate: remove any pre-existing edge with the same identity.
        inner.edges.retain(|e| {
            !(e.scope_prefix == scope_pfx
                && e.from_key == link.from_key
                && e.to_key == link.to_key
                && e.relation == link.relation)
        });
        inner.edges.push(record);
        Ok(())
    }

    async fn unlink(
        &self,
        scope: &Scope,
        from_key: &str,
        to_key: &str,
        relation: &str,
    ) -> Result<(), StateError> {
        let scope_pfx = scope_prefix(scope);
        let mut inner = self.engine.inner.write().await;
        inner.edges.retain(|e| {
            !(e.scope_prefix == scope_pfx
                && e.from_key == from_key
                && e.to_key == to_key
                && e.relation == relation)
        });
        Ok(())
    }

    /// Traverse links from `from_key` using breadth-first search.
    ///
    /// Returns all keys reachable within `max_depth` hops via edges matching
    /// `relation` (or any relation if `relation` is `None`). The starting key
    /// is never included in the result. Cycle-safe: each key is visited at
    /// most once.
    async fn traverse(
        &self,
        scope: &Scope,
        from_key: &str,
        relation: Option<&str>,
        max_depth: u32,
    ) -> Result<Vec<String>, StateError> {
        if max_depth == 0 {
            return Ok(vec![]);
        }
        let scope_pfx = scope_prefix(scope);
        let inner = self.engine.inner.read().await;

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(from_key.to_string());

        // Queue entries: (current_key, hops_taken_to_reach_it)
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((from_key.to_string(), 0));

        let mut result: Vec<String> = Vec::new();

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for edge in &inner.edges {
                if edge.scope_prefix != scope_pfx || edge.from_key != current {
                    continue;
                }
                if let Some(rel) = relation
                    && edge.relation != rel
                {
                    continue;
                }
                if !visited.contains(&edge.to_key) {
                    visited.insert(edge.to_key.clone());
                    result.push(edge.to_key.clone());
                    queue.push_back((edge.to_key.clone(), depth + 1));
                }
            }
        }
        Ok(result)
    }

    async fn search_hinted(
        &self,
        scope: &Scope,
        query: &str,
        limit: usize,
        _options: &SearchOptions,
    ) -> Result<Vec<SearchResult>, StateError> {
        self.search(scope, query, limit).await
    }
}

// ── Real CozoDB backend (requires `--features cozo`) ──────────────────────────

#[cfg(feature = "cozo")]
use crate::engine::DataValue;
#[cfg(feature = "cozo")]
use cozo::Vector;
#[cfg(feature = "cozo")]
use std::collections::BTreeMap;

/// Extract a `String` from a `DataValue`, returning `None` for non-string values.
#[cfg(feature = "cozo")]
fn dv_as_string(dv: &DataValue) -> Option<String> {
    match dv {
        DataValue::Str(s) => Some(s.to_string()),
        _ => None,
    }
}

/// Extract an `f64` from a `DataValue::Num`, returning `None` for non-numeric values.
#[cfg(feature = "cozo")]
fn dv_as_f64(dv: &DataValue) -> Option<f64> {
    dv.get_float()
}

#[cfg(feature = "cozo")]
#[async_trait]
impl StateStore for CozoStore {
    #[instrument(skip_all, fields(scope = ?scope, key = %key))]
    async fn read(
        &self,
        scope: &Scope,
        key: &str,
    ) -> Result<Option<serde_json::Value>, StateError> {
        let sp = scope_prefix(scope);
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "key" => DataValue::Str(key.to_string().into()),
        };
        let rows = self
            .engine
            .run_query("?[value] := *kv{scope: $scope, key: $key, value}", params)
            .map_err(StateError::from)?;
        match rows.rows.first() {
            None => Ok(None),
            Some(row) => {
                let raw = dv_as_string(&row[0]).ok_or_else(|| {
                    StateError::Serialization("expected string value from kv".to_string())
                })?;
                let val: serde_json::Value = serde_json::from_str(&raw)
                    .map_err(|e| StateError::Serialization(e.to_string()))?;
                Ok(Some(val))
            }
        }
    }

    #[instrument(skip_all, fields(scope = ?scope, key = %key))]
    async fn write(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), StateError> {
        let sp = scope_prefix(scope);
        let raw = serde_json::to_string(&value)
            .map_err(|e| StateError::Serialization(e.to_string()))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "key" => DataValue::Str(key.to_string().into()),
            "value" => DataValue::Str(raw.into()),
            "now" => DataValue::from(now),
        };
        self.engine
            .run_mutation(
                "?[scope, key, value, created_at] <- [[$scope, $key, $value, $now]] :put kv {scope, key => value, created_at}",
                params,
            )
            .map_err(StateError::from)?;
        Ok(())
    }

    #[instrument(skip_all, fields(scope = ?scope, key = %key))]
    async fn delete(&self, scope: &Scope, key: &str) -> Result<(), StateError> {
        let sp = scope_prefix(scope);
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "key" => DataValue::Str(key.to_string().into()),
        };
        self.engine
            .run_mutation(
                "?[scope, key] <- [[$scope, $key]] :rm kv {scope, key}",
                params,
            )
            .map_err(StateError::from)?;
        Ok(())
    }

    #[instrument(skip_all, fields(scope = ?scope))]
    async fn list(&self, scope: &Scope, prefix: &str) -> Result<Vec<String>, StateError> {
        let sp = scope_prefix(scope);
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
        };
        let rows = self
            .engine
            .run_query("?[key] := *kv{scope: $scope, key}", params)
            .map_err(StateError::from)?;
        let keys: Vec<String> = rows
            .rows
            .iter()
            .filter_map(|row| {
                let k = dv_as_string(&row[0])?;
                if k.starts_with(prefix) {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();
        Ok(keys)
    }

    /// Search for entries whose value field matches the FTS query.
    ///
    /// Uses the `kv:fts_val` FTS index created at schema init time.
    /// Results are returned ranked by BM25 score (descending).
    #[instrument(skip_all, fields(scope = ?scope))]
    async fn search(
        &self,
        scope: &Scope,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StateError> {
        if query.is_empty() || limit == 0 {
            return Ok(vec![]);
        }
        let sp = scope_prefix(scope);
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "query" => DataValue::Str(query.to_string().into()),
            "limit" => DataValue::from(limit as i64),
        };
        let fts_q = concat!(
            "?[key, score] := ",
            "~kv:fts_val {scope, key | query: $query, k: $limit, bind_score: score},",
            " scope == $scope\n",
            ":order -score\n",
            ":limit $limit",
        );
        let rows = self
            .engine
            .run_query(fts_q, params)
            .map_err(StateError::from)?;
        let results: Vec<SearchResult> = rows
            .rows
            .iter()
            .filter_map(|row| {
                let key = dv_as_string(&row[0])?;
                let score = dv_as_f64(&row[1]).unwrap_or(0.5);
                Some(SearchResult::new(key, score))
            })
            .collect();
        Ok(results)
    }

    async fn read_hinted(
        &self,
        scope: &Scope,
        key: &str,
        _options: &StoreOptions,
    ) -> Result<Option<serde_json::Value>, StateError> {
        self.read(scope, key).await
    }

    async fn write_hinted(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
        _options: &StoreOptions,
    ) -> Result<(), StateError> {
        self.write(scope, key, value).await
    }

    /// No-op in v1.
    ///
    /// CozoDB v1 does not maintain a separate transient table. Entries written
    /// with [`Lifetime::Transient`] are stored durably and not cleared here.
    /// A dedicated transient store is planned for v2.
    ///
    /// [`Lifetime::Transient`]: layer0::state::Lifetime
    fn clear_transient(&self) {}

    async fn link(&self, scope: &Scope, link: &MemoryLink) -> Result<(), StateError> {
        let sp = scope_prefix(scope);
        let metadata_str = link
            .metadata
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "from_key" => DataValue::Str(link.from_key.clone().into()),
            "to_key" => DataValue::Str(link.to_key.clone().into()),
            "relation" => DataValue::Str(link.relation.clone().into()),
            "metadata" => DataValue::Str(metadata_str.into()),
            "now" => DataValue::from(now),
        };
        self.engine
            .run_mutation(
                "?[scope, from_key, to_key, relation, metadata, created_at] <- [[$scope, $from_key, $to_key, $relation, $metadata, $now]] :put edge {scope, from_key, to_key, relation => metadata, created_at}",
                params,
            )
            .map_err(StateError::from)?;
        Ok(())
    }

    async fn unlink(
        &self,
        scope: &Scope,
        from_key: &str,
        to_key: &str,
        relation: &str,
    ) -> Result<(), StateError> {
        let sp = scope_prefix(scope);
        let params = cozo_params! {
            "scope" => DataValue::Str(sp.into()),
            "from_key" => DataValue::Str(from_key.to_string().into()),
            "to_key" => DataValue::Str(to_key.to_string().into()),
            "relation" => DataValue::Str(relation.to_string().into()),
        };
        self.engine
            .run_mutation(
                "?[scope, from_key, to_key, relation] <- [[$scope, $from_key, $to_key, $relation]] :rm edge {scope, from_key, to_key, relation}",
                params,
            )
            .map_err(StateError::from)?;
        Ok(())
    }

    /// Traverse links from `from_key` using breadth-first search.
    ///
    /// Returns all keys reachable within `max_depth` hops via edges matching
    /// `relation` (or any relation if `relation` is `None`). The starting key
    /// is never included in the result. Cycle-safe: each key is visited at
    /// most once.
    async fn traverse(
        &self,
        scope: &Scope,
        from_key: &str,
        relation: Option<&str>,
        max_depth: u32,
    ) -> Result<Vec<String>, StateError> {
        if max_depth == 0 {
            return Ok(vec![]);
        }
        let sp = scope_prefix(scope);

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(from_key.to_string());

        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((from_key.to_string(), 0));

        let mut result: Vec<String> = Vec::new();

        // BFS using repeated edge queries — avoids complex recursive Datalog.
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let rows = match relation {
                Some(rel) => {
                    let params = cozo_params! {
                        "scope" => DataValue::Str(sp.clone().into()),
                        "from_key" => DataValue::Str(current.clone().into()),
                        "relation" => DataValue::Str(rel.to_string().into()),
                    };
                    self.engine
                        .run_query(
                            "?[to_key] := *edge{scope: $scope, from_key: $from_key, relation: $relation, to_key}",
                            params,
                        )
                        .map_err(StateError::from)?
                }
                None => {
                    let params = cozo_params! {
                        "scope" => DataValue::Str(sp.clone().into()),
                        "from_key" => DataValue::Str(current.clone().into()),
                    };
                    self.engine
                        .run_query(
                            "?[to_key] := *edge{scope: $scope, from_key: $from_key, to_key}",
                            params,
                        )
                        .map_err(StateError::from)?
                }
            };

            for row in &rows.rows {
                if let Some(to_key) = dv_as_string(&row[0])
                    && !visited.contains(&to_key)
                {
                    visited.insert(to_key.clone());
                    result.push(to_key.clone());
                    queue.push_back((to_key, depth + 1));
                }
            }
        }
        Ok(result)
    }

    async fn search_hinted(
        &self,
        scope: &Scope,
        query: &str,
        limit: usize,
        _options: &SearchOptions,
    ) -> Result<Vec<SearchResult>, StateError> {
        self.search(scope, query, limit).await
    }
}
