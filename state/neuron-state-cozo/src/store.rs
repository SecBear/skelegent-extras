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

use crate::engine::{CozoEngine, EdgeRecord};
use crate::scope::{composite_key, extract_key, scope_prefix};
use async_trait::async_trait;
use layer0::effect::Scope;
use layer0::error::StateError;
use layer0::state::{MemoryLink, SearchOptions, SearchResult, StateStore, StoreOptions};
use std::collections::{HashSet, VecDeque};

/// CozoDB-backed [`StateStore`] with graph and hybrid search support.
///
/// # Backends
///
/// - **Default** (no `rocksdb` feature): wraps an in-memory [`HashMap`] suitable
///   for testing and single-process use without durability.
/// - **`rocksdb` feature**: wraps a real CozoDB `DbInstance` backed by RocksDB,
///   enabling Datalog graph traversal, HNSW vector search, and FTS.
///
/// `CozoStore` is cheaply cloneable when needed — both the store and its
/// underlying [`CozoEngine`] share the same storage via [`Arc`].
///
/// # Example
///
/// ```no_run
/// use neuron_state_cozo::CozoStore;
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

#[async_trait]
impl StateStore for CozoStore {
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

    async fn delete(&self, scope: &Scope, key: &str) -> Result<(), StateError> {
        let ck = composite_key(scope, key);
        let mut inner = self.engine.inner.write().await;
        inner.kv.remove(&ck);
        Ok(())
    }

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
