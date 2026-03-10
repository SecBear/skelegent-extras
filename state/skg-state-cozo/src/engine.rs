//! CozoDB engine wrapper.
//!
//! Provides [`CozoEngine`]: a typed wrapper over CozoDB's `DbInstance`
//! (or an in-memory HashMap placeholder when the `cozo` feature is disabled).
//!
//! Schema initialization via [`CozoEngine::ensure_schema`] is idempotent.

use thiserror::Error;

// HashMap backend imports — only needed when cozo feature is off.
#[cfg(not(feature = "cozo"))]
use std::collections::HashMap;
#[cfg(not(feature = "cozo"))]
use std::sync::Arc;
#[cfg(not(feature = "cozo"))]
use tokio::sync::RwLock;

// Real CozoDB imports — only needed when cozo feature is on.
#[cfg(feature = "cozo")]
use std::collections::BTreeMap;
#[cfg(feature = "cozo")]
pub(crate) use cozo::DataValue;
#[cfg(feature = "cozo")]
use cozo::NamedRows;

/// Error type for [`CozoEngine`] operations.
#[derive(Debug, Error)]
pub enum CozoError {
    /// A database operation failed.
    #[error("database error: {0}")]
    Database(String),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<CozoError> for layer0::error::StateError {
    fn from(e: CozoError) -> Self {
        layer0::error::StateError::WriteFailed(e.to_string())
    }
}

// ── HashMap backend (default, no native deps) ─────────────────────────────────

/// A directed edge stored in the graph.
#[cfg(not(feature = "cozo"))]
#[derive(Debug, Clone)]
pub(crate) struct EdgeRecord {
    /// Serialized scope prefix (from [`crate::scope::scope_prefix`]).
    pub(crate) scope_prefix: String,
    /// Key of the source node.
    pub(crate) from_key: String,
    /// Key of the target node.
    pub(crate) to_key: String,
    /// Relationship type label (e.g. `"references"`, `"supersedes"`).
    pub(crate) relation: String,
    /// Optional structured metadata for the edge.
    /// Stored now; returned in traversal results when the real CozoDB backend
    /// is active. Suppressed here because [`StateStore::traverse`] returns
    /// only keys in v1.
    ///
    /// [`StateStore::traverse`]: layer0::state::StateStore::traverse
    #[allow(dead_code)]
    pub(crate) metadata: Option<serde_json::Value>,
}

/// Heap-allocated inner storage for the in-memory backend.
#[cfg(not(feature = "cozo"))]
#[derive(Debug, Default)]
pub(crate) struct EngineInner {
    /// Key-value store: composite key → JSON-encoded value string.
    pub(crate) kv: HashMap<String, String>,
    /// All graph edges across all scopes.
    pub(crate) edges: Vec<EdgeRecord>,
}

/// Typed wrapper over CozoDB's storage engine.
///
/// When the `cozo` feature is **disabled** (default), this wraps an in-memory
/// [`HashMap`] — suitable for testing and development without any C dependencies.
///
/// Enable the `cozo` feature to swap in a real [`cozo::DbInstance`] with
/// in-memory Datalog storage. Add `rocksdb` for persistent on-disk storage.
///
/// `CozoEngine` is cheaply cloneable — clones share the same underlying storage
/// (via [`Arc`]).
///
/// Schema initialization is idempotent: calling [`ensure_schema`] multiple times
/// on the same engine is safe.
///
/// [`ensure_schema`]: CozoEngine::ensure_schema
/// [`HashMap`]: std::collections::HashMap
/// [`Arc`]: std::sync::Arc
#[cfg(not(feature = "cozo"))]
#[derive(Debug, Clone)]
pub struct CozoEngine {
    pub(crate) inner: Arc<RwLock<EngineInner>>,
}

#[cfg(not(feature = "cozo"))]
impl CozoEngine {
    /// Open or create a CozoDB database at the given path.
    ///
    /// Pass `":memory:"` to create an ephemeral in-memory database, or use
    /// [`CozoEngine::memory`] as a convenience constructor.
    ///
    /// When the `cozo` feature is disabled, `path` is ignored and an
    /// in-memory HashMap backend is always used.
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if the engine cannot be initialized.
    pub fn new(_path: &str) -> Result<Self, CozoError> {
        Ok(Self {
            inner: Arc::new(RwLock::new(EngineInner::default())),
        })
    }

    /// Create an in-memory database.
    ///
    /// Convenience wrapper equivalent to `CozoEngine::new(":memory:")`.
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if the engine cannot be initialized.
    pub fn memory() -> Result<Self, CozoError> {
        Self::new(":memory:")
    }

    /// Initialize the database schema (idempotent).
    ///
    /// Creates the `kv`, `node`, and `edge` Datalog relations if they do not
    /// already exist. Safe to call multiple times on the same engine.
    ///
    /// See [`crate::schema`] for the DDL strings.
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if DDL execution fails.
    pub fn ensure_schema(&self) -> Result<(), CozoError> {
        // Placeholder: no schema is needed for the HashMap backend.
        // With the real CozoDB backend, execute schema::DDL_INIT here.
        Ok(())
    }
}

// ── Real CozoDB backend (requires `--features cozo`) ──────────────────────────

/// Typed wrapper over a real CozoDB `DbInstance`.
///
/// Enabled by the `cozo` cargo feature. With `rocksdb` also enabled, persistent
/// RocksDB storage is available via [`CozoEngine::open`]. Without `rocksdb`,
/// only in-memory Datalog storage is used.
///
/// Use [`CozoStore::memory`] for ephemeral storage or [`CozoStore::open`] for
/// persistent on-disk storage.
///
/// Schema initialization is idempotent: calling [`ensure_schema`] multiple
/// times on the same engine is safe.
///
/// [`CozoStore::memory`]: crate::store::CozoStore::memory
/// [`CozoStore::open`]: crate::store::CozoStore::open
/// [`ensure_schema`]: CozoEngine::ensure_schema
#[cfg(feature = "cozo")]
pub struct CozoEngine {
    db: cozo::DbInstance,
}

#[cfg(feature = "cozo")]
impl std::fmt::Debug for CozoEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CozoEngine").finish_non_exhaustive()
    }
}

#[cfg(feature = "cozo")]
impl CozoEngine {
    /// Open or create a CozoDB database at the given path.
    ///
    /// With the `rocksdb` feature enabled, uses the RocksDB engine for
    /// persistent storage. Without `rocksdb`, uses the `mem` engine and
    /// `path` is ignored.
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if the engine cannot be initialized.
    pub fn new(path: &str) -> Result<Self, CozoError> {
        #[cfg(feature = "rocksdb")]
        {
            let db = cozo::DbInstance::new("rocksdb", path, Default::default())
                .map_err(|e| CozoError::Database(e.to_string()))?;
            Ok(Self { db })
        }
        #[cfg(not(feature = "rocksdb"))]
        {
            let _ = path;
            let db = cozo::DbInstance::new("mem", "", Default::default())
                .map_err(|e| CozoError::Database(e.to_string()))?;
            Ok(Self { db })
        }
    }

    /// Open or create a persistent CozoDB RocksDB database at the given path.
    ///
    /// The directory at `path` is created if it does not exist.
    ///
    /// **Requires the `rocksdb` feature.**
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if the RocksDB engine cannot be opened.
    #[cfg(feature = "rocksdb")]
    pub fn open(path: &str) -> Result<Self, CozoError> {
        let db = cozo::DbInstance::new("rocksdb", path, Default::default())
            .map_err(|e| CozoError::Database(e.to_string()))?;
        Ok(Self { db })
    }

    /// Create an ephemeral in-memory CozoDB database.
    ///
    /// Uses cozo's built-in `mem` engine — pure Rust, no disk I/O.
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if the engine cannot be initialized.
    pub fn memory() -> Result<Self, CozoError> {
        let db = cozo::DbInstance::new("mem", "", Default::default())
            .map_err(|e| CozoError::Database(e.to_string()))?;
        Ok(Self { db })
    }

    /// Initialize the database schema (idempotent).
    ///
    /// Executes [`crate::schema::DDL_INIT`] to create the `kv`, `node`, and
    /// `edge` relations if they do not already exist. Safe to call multiple
    /// times.
    ///
    /// # Errors
    ///
    /// Returns [`CozoError::Database`] if DDL execution fails.
    pub fn ensure_schema(&self) -> Result<(), CozoError> {
        // CozoDB requires each :create command in its own run_script call.
        for ddl in [
            crate::schema::KV_DDL,
            crate::schema::NODE_V2_DDL,
            crate::schema::EDGE_DDL,
        ] {
            self.db
                .run_script(ddl, Default::default(), cozo::ScriptMutability::Mutable)
                .map_err(|e| CozoError::Database(format!("{e:?}")))?;
        }
        // FTS index DDL must be a separate run_script call — CozoDB rejects
        // mixed :create / ::fts create scripts.
        self.db
            .run_script(
                crate::schema::KV_FTS_DDL,
                Default::default(),
                cozo::ScriptMutability::Mutable,
            )
            .map_err(|e| CozoError::Database(format!("{e:?}")))?;
        // HNSW index DDL must also be a separate run_script call.
        self.db
            .run_script(
                crate::schema::NODE_HNSW_DDL,
                Default::default(),
                cozo::ScriptMutability::Mutable,
            )
            .map_err(|e| CozoError::Database(format!("{e:?}")))?;
        Ok(())
    }

    /// Execute a read-only CozoScript query.
    pub(crate) fn run_query(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows, CozoError> {
        self.db
            .run_script(script, params, cozo::ScriptMutability::Immutable)
            .map_err(|e| CozoError::Database(format!("{e:?}")))
    }

    /// Execute a mutable CozoScript query (put/rm).
    pub(crate) fn run_mutation(
        &self,
        script: &str,
        params: BTreeMap<String, DataValue>,
    ) -> Result<NamedRows, CozoError> {
        self.db
            .run_script(script, params, cozo::ScriptMutability::Mutable)
            .map_err(|e| CozoError::Database(format!("{e:?}")))
    }
}
