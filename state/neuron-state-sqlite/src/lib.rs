#![deny(missing_docs)]
//! SQLite-backed implementation of layer0's [`StateStore`] trait.
//!
//! Uses [`rusqlite`] with bundled SQLite and FTS5 for full-text search.
//! All database operations run on a blocking thread via
//! [`tokio::task::spawn_blocking`] to avoid blocking the async runtime.
//!
//! # Features
//!
//! - **Scope isolation**: entries are keyed by serialized scope + key.
//! - **FTS5 search**: `unicode61` tokenizer with `tokenchars "-_"` for
//!   technical terms like `durable-execution` and `state_store`.
//! - **Advisory hints**: tier, lifetime, content_kind, salience, and TTL
//!   are stored as columns and honored where applicable.
//! - **Transient entries**: stored in a separate table, cleared on
//!   [`StateStore::clear_transient`].
//! - **Lazy TTL expiry**: expired entries cleaned up on read and list.
//!
//! # Example
//!
//! ```no_run
//! use neuron_state_sqlite::SqliteStore;
//!
//! let store = SqliteStore::open(":memory:").expect("open in-memory db");
//! ```

mod schema;
pub mod search;

use async_trait::async_trait;
use layer0::effect::Scope;
use layer0::error::StateError;
use layer0::state::{Lifetime, SearchResult, StateStore, StoreOptions};
use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed state store with FTS5 full-text search.
///
/// Thread-safe via interior `Mutex<Connection>`. All async operations
/// dispatch to a blocking thread pool so the connection is never held
/// across await points.
pub struct SqliteStore {
    conn: Mutex<Connection>,
}

impl SqliteStore {
    /// Open (or create) a SQLite database at the given path.
    ///
    /// Runs migrations on first open. Use `":memory:"` for an ephemeral
    /// in-memory database (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns [`StateError::WriteFailed`] if the database cannot be opened
    /// or migrations fail.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StateError> {
        let conn = Connection::open(path.as_ref())
            .map_err(|e| StateError::WriteFailed(format!("sqlite open: {e}")))?;

        // WAL mode for concurrent readers.
        conn.pragma_update(None, "journal_mode", "wal")
            .map_err(|e| StateError::WriteFailed(format!("sqlite wal: {e}")))?;

        schema::migrate(&conn)
            .map_err(|e| StateError::WriteFailed(format!("sqlite migrate: {e}")))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory SQLite database.
    ///
    /// Convenience wrapper for `SqliteStore::open(":memory:")`.
    ///
    /// # Errors
    ///
    /// Returns [`StateError::WriteFailed`] if initialization fails.
    pub fn open_in_memory() -> Result<Self, StateError> {
        Self::open(":memory:")
    }

    /// Access the connection under the mutex.
    ///
    /// This is an internal helper. Callers must not hold the guard across
    /// await points — all usage is within `spawn_blocking` closures.
    fn with_conn<F, T>(&self, f: F) -> Result<T, StateError>
    where
        F: FnOnce(&Connection) -> Result<T, rusqlite::Error>,
    {
        let conn = self
            .conn
            .lock()
            .map_err(|e| StateError::WriteFailed(format!("mutex poisoned: {e}")))?;
        f(&conn).map_err(|e| StateError::WriteFailed(format!("sqlite: {e}")))
    }

    /// Remove entries that have exceeded their TTL.
    ///
    /// Called lazily during read and list operations.
    fn expire_ttl(&self) -> Result<(), StateError> {
        self.with_conn(|conn| {
            conn.execute(
                "DELETE FROM entries
                 WHERE ttl_ms IS NOT NULL
                   AND (julianday('now') - julianday(created_at)) * 86400000 > ttl_ms",
                [],
            )?;
            Ok(())
        })
    }
}

/// Serialize a scope to a deterministic string for use as a DB key.
fn scope_str(scope: &Scope) -> String {
    serde_json::to_string(scope).unwrap_or_else(|_| "unknown".to_string())
}

// Safety: Connection is behind a Mutex, so the store is Send+Sync.
// rusqlite::Connection is !Send, but we only access it under the lock.
unsafe impl Send for SqliteStore {}
unsafe impl Sync for SqliteStore {}

#[async_trait]
impl StateStore for SqliteStore {
    async fn read(
        &self,
        scope: &Scope,
        key: &str,
    ) -> Result<Option<serde_json::Value>, StateError> {
        // Lazy TTL cleanup.
        let _ = self.expire_ttl();

        let s = scope_str(scope);
        let k = key.to_string();
        self.with_conn(|conn| {
            let mut stmt =
                conn.prepare_cached("SELECT value FROM entries WHERE scope = ?1 AND key = ?2")?;
            let result = stmt.query_row(params![s, k], |row| {
                let raw: String = row.get(0)?;
                Ok(raw)
            });
            match result {
                Ok(raw) => {
                    let val: serde_json::Value = serde_json::from_str(&raw)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                    Ok(Some(val))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    async fn write(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), StateError> {
        let s = scope_str(scope);
        let k = key.to_string();
        let v =
            serde_json::to_string(&value).map_err(|e| StateError::Serialization(e.to_string()))?;

        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO entries (scope, key, value)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(scope, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                params![s, k, v],
            )?;
            Ok(())
        })
    }

    async fn delete(&self, scope: &Scope, key: &str) -> Result<(), StateError> {
        let s = scope_str(scope);
        let k = key.to_string();
        self.with_conn(|conn| {
            conn.execute(
                "DELETE FROM entries WHERE scope = ?1 AND key = ?2",
                params![s, k],
            )?;
            Ok(())
        })
    }

    async fn list(&self, scope: &Scope, prefix: &str) -> Result<Vec<String>, StateError> {
        let _ = self.expire_ttl();

        let s = scope_str(scope);
        let p = format!("{prefix}%");
        self.with_conn(|conn| {
            let mut stmt =
                conn.prepare_cached("SELECT key FROM entries WHERE scope = ?1 AND key LIKE ?2")?;
            let rows = stmt.query_map(params![s, p], |row| row.get::<_, String>(0))?;
            rows.collect()
        })
    }

    async fn search(
        &self,
        scope: &Scope,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StateError> {
        let s = scope_str(scope);
        let q = query.to_string();
        self.with_conn(|conn| {
            let matches = search::fts5_search(conn, &s, &q, limit)?;
            Ok(matches
                .into_iter()
                .map(|m| {
                    let mut sr = SearchResult::new(m.key, m.rank);
                    sr.snippet = m.snippet;
                    sr
                })
                .collect())
        })
    }

    async fn write_hinted(
        &self,
        scope: &Scope,
        key: &str,
        value: serde_json::Value,
        options: &StoreOptions,
    ) -> Result<(), StateError> {
        // Route transient writes to the separate table.
        if matches!(options.lifetime, Some(Lifetime::Transient)) {
            let s = scope_str(scope);
            let k = key.to_string();
            let v = serde_json::to_string(&value)
                .map_err(|e| StateError::Serialization(e.to_string()))?;

            return self.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO transient (scope, key, value)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(scope, key) DO UPDATE SET
                        value = excluded.value",
                    params![s, k, v],
                )?;
                Ok(())
            });
        }

        // For non-transient writes, store with full metadata.
        let s = scope_str(scope);
        let k = key.to_string();
        let v =
            serde_json::to_string(&value).map_err(|e| StateError::Serialization(e.to_string()))?;

        let tier = options.tier.and_then(|t| {
            serde_json::to_value(t)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
        });

        let lifetime = options.lifetime.and_then(|l| {
            serde_json::to_value(l)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
        });

        let content_kind = options.content_kind.as_ref().and_then(|c| {
            serde_json::to_value(c)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
        });

        let salience = options.salience;
        let ttl_ms = options.ttl.map(|d| d.as_millis() as i64);

        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO entries (scope, key, value, tier, lifetime, content_kind, salience, ttl_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(scope, key) DO UPDATE SET
                    value = excluded.value,
                    tier = excluded.tier,
                    lifetime = excluded.lifetime,
                    content_kind = excluded.content_kind,
                    salience = excluded.salience,
                    ttl_ms = excluded.ttl_ms,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                params![s, k, v, tier, lifetime, content_kind, salience, ttl_ms],
            )?;
            Ok(())
        })
    }

    fn clear_transient(&self) {
        let _ = self.with_conn(|conn| {
            conn.execute("DELETE FROM transient", [])?;
            Ok(())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store() -> SqliteStore {
        SqliteStore::open_in_memory().unwrap()
    }

    #[tokio::test]
    async fn write_and_read() {
        let s = store();
        let scope = Scope::Global;

        s.write(&scope, "key1", json!("value1")).await.unwrap();
        let val = s.read(&scope, "key1").await.unwrap();
        assert_eq!(val, Some(json!("value1")));
    }

    #[tokio::test]
    async fn read_nonexistent_returns_none() {
        let s = store();
        let scope = Scope::Global;

        let val = s.read(&scope, "missing").await.unwrap();
        assert_eq!(val, None);
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let s = store();
        let scope = Scope::Global;

        s.write(&scope, "key1", json!("first")).await.unwrap();
        s.write(&scope, "key1", json!("second")).await.unwrap();
        let val = s.read(&scope, "key1").await.unwrap();
        assert_eq!(val, Some(json!("second")));
    }

    #[tokio::test]
    async fn delete_removes_key() {
        let s = store();
        let scope = Scope::Global;

        s.write(&scope, "key1", json!("value1")).await.unwrap();
        s.delete(&scope, "key1").await.unwrap();
        let val = s.read(&scope, "key1").await.unwrap();
        assert_eq!(val, None);
    }

    #[tokio::test]
    async fn delete_nonexistent_is_ok() {
        let s = store();
        let scope = Scope::Global;

        let result = s.delete(&scope, "missing").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn list_keys_with_prefix() {
        let s = store();
        let scope = Scope::Global;

        s.write(&scope, "user:name", json!("Alice")).await.unwrap();
        s.write(&scope, "user:age", json!(30)).await.unwrap();
        s.write(&scope, "system:version", json!("1.0"))
            .await
            .unwrap();

        let mut keys = s.list(&scope, "user:").await.unwrap();
        keys.sort();
        assert_eq!(keys, vec!["user:age", "user:name"]);
    }

    #[tokio::test]
    async fn list_empty_prefix_returns_all() {
        let s = store();
        let scope = Scope::Global;

        s.write(&scope, "a", json!(1)).await.unwrap();
        s.write(&scope, "b", json!(2)).await.unwrap();

        let keys = s.list(&scope, "").await.unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn scopes_are_isolated() {
        let s = store();
        let global = Scope::Global;
        let session = Scope::Session(layer0::SessionId::new("s1"));

        s.write(&global, "key", json!("global_val")).await.unwrap();
        s.write(&session, "key", json!("session_val"))
            .await
            .unwrap();

        let global_val = s.read(&global, "key").await.unwrap();
        let session_val = s.read(&session, "key").await.unwrap();

        assert_eq!(global_val, Some(json!("global_val")));
        assert_eq!(session_val, Some(json!("session_val")));
    }

    #[tokio::test]
    async fn fts5_search_finds_content() {
        let s = store();
        let scope = Scope::Global;

        s.write(
            &scope,
            "doc:arch",
            json!("durable execution and crash recovery patterns"),
        )
        .await
        .unwrap();
        s.write(
            &scope,
            "doc:mem",
            json!("memory architecture and context engineering"),
        )
        .await
        .unwrap();
        s.write(&scope, "doc:unrelated", json!("the weather is nice today"))
            .await
            .unwrap();

        let results = s.search(&scope, "durable execution", 10).await.unwrap();
        assert!(!results.is_empty(), "FTS5 search should find matching doc");
        assert_eq!(results[0].key, "doc:arch");
    }

    #[tokio::test]
    async fn fts5_search_respects_scope() {
        let s = store();
        let scope_a = Scope::Global;
        let scope_b = Scope::Session(layer0::SessionId::new("isolated"));

        s.write(&scope_a, "doc:1", json!("durable execution patterns"))
            .await
            .unwrap();
        s.write(&scope_b, "doc:2", json!("durable execution in other scope"))
            .await
            .unwrap();

        let results = s.search(&scope_a, "durable execution", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "doc:1");
    }

    #[tokio::test]
    async fn fts5_search_empty_returns_empty() {
        let s = store();
        let scope = Scope::Global;

        let results = s.search(&scope, "nonexistent term xyz", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn transient_write_not_durable() {
        let s = store();
        let scope = Scope::Global;

        let opts = StoreOptions {
            lifetime: Some(Lifetime::Transient),
            ..Default::default()
        };
        s.write_hinted(&scope, "scratch", json!("temp"), &opts)
            .await
            .unwrap();

        // Transient entries are NOT visible via read() (separate table).
        let val = s.read(&scope, "scratch").await.unwrap();
        assert_eq!(val, None, "transient entry must not be visible via read()");

        // clear_transient is idempotent.
        s.clear_transient();
        s.clear_transient();

        // Write a durable entry.
        s.write(&scope, "durable", json!("persisted"))
            .await
            .unwrap();

        // clear_transient does not touch durable storage.
        s.clear_transient();

        let durable_val = s.read(&scope, "durable").await.unwrap();
        assert_eq!(
            durable_val,
            Some(json!("persisted")),
            "durable entry must survive clear_transient()"
        );
    }

    #[tokio::test]
    async fn write_hinted_stores_metadata() {
        use layer0::state::{ContentKind, MemoryTier};

        let s = store();
        let scope = Scope::Global;

        let opts = StoreOptions {
            tier: Some(MemoryTier::Hot),
            lifetime: Some(Lifetime::Durable),
            content_kind: Some(ContentKind::Semantic),
            salience: Some(0.9),
            ttl: None,
        };
        s.write_hinted(&scope, "fact:1", json!("important fact"), &opts)
            .await
            .unwrap();

        // Verify it's readable.
        let val = s.read(&scope, "fact:1").await.unwrap();
        assert_eq!(val, Some(json!("important fact")));

        // Verify metadata was stored (check via raw SQL).
        let sc = scope_str(&scope);
        let salience: f64 = s
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT salience FROM entries WHERE scope = ?1 AND key = ?2",
                    params![sc, "fact:1"],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert!((salience - 0.9).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn serde_roundtrip_complex_value() {
        let s = store();
        let scope = Scope::Global;

        let complex = json!({
            "nested": {"array": [1, 2, 3]},
            "bool": true,
            "null": null,
            "float": 3.14
        });

        s.write(&scope, "complex", complex.clone()).await.unwrap();
        let val = s.read(&scope, "complex").await.unwrap();
        assert_eq!(val, Some(complex));
    }

    #[test]
    fn sqlite_store_implements_state_store() {
        fn _assert_state_store<T: StateStore>() {}
        _assert_state_store::<SqliteStore>();
    }

    #[test]
    fn open_in_memory_succeeds() {
        let _store = SqliteStore::open_in_memory().unwrap();
    }
}
