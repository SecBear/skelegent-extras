//! SQLite schema management and migrations.
//!
//! Schema version is stored in `PRAGMA user_version` and migrations run
//! sequentially on startup. The FTS5 virtual table uses `unicode61` with
//! `tokenchars` configured for hyphenated technical terms.

use rusqlite::Connection;

/// Current schema version. Bump this and add a migration when the schema changes.
const CURRENT_VERSION: u32 = 1;

/// Run all pending migrations against the connection.
///
/// # Errors
///
/// Returns `rusqlite::Error` if any migration SQL fails.
pub(crate) fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    let version: u32 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;

    if version < 1 {
        migration_001(conn)?;
    }

    conn.pragma_update(None, "user_version", CURRENT_VERSION)?;
    Ok(())
}

/// Migration 001: Initial schema.
///
/// Creates the main entries table, the FTS5 virtual table, and triggers
/// to keep them in sync.
fn migration_001(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        -- Main entries table. Scope is serialized JSON for isolation.
        CREATE TABLE IF NOT EXISTS entries (
            scope       TEXT    NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            tier        TEXT,
            lifetime    TEXT,
            content_kind TEXT,
            salience    REAL,
            ttl_ms      INTEGER,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (scope, key)
        );

        -- Transient entries: separate table, cleared on turn boundaries.
        CREATE TABLE IF NOT EXISTS transient (
            scope       TEXT    NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (scope, key)
        );

        -- FTS5 virtual table for full-text search.
        -- unicode61 tokenizer with tokenchars for hyphens and underscores,
        -- preserving technical terms like 'durable-execution' and 'state_store'.
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
            scope,
            key,
            value,
            content = 'entries',
            content_rowid = 'rowid',
            tokenize = "unicode61 tokenchars '-_' remove_diacritics 0"
        );

        -- Triggers to keep FTS in sync with entries table.
        CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
            INSERT INTO entries_fts(rowid, scope, key, value)
                VALUES (new.rowid, new.scope, new.key, new.value);
        END;

        CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
            INSERT INTO entries_fts(entries_fts, rowid, scope, key, value)
                VALUES ('delete', old.rowid, old.scope, old.key, old.value);
        END;

        CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
            INSERT INTO entries_fts(entries_fts, rowid, scope, key, value)
                VALUES ('delete', old.rowid, old.scope, old.key, old.value);
            INSERT INTO entries_fts(rowid, scope, key, value)
                VALUES (new.rowid, new.scope, new.key, new.value);
        END;

        -- Index for TTL cleanup scans.
        CREATE INDEX IF NOT EXISTS idx_entries_ttl
            ON entries(lifetime, created_at)
            WHERE ttl_ms IS NOT NULL;

        -- Index for scope+prefix list queries.
        CREATE INDEX IF NOT EXISTS idx_entries_scope_key
            ON entries(scope, key);
        "#,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_creates_tables() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        // Verify tables exist
        let count: u32 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='entries'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn migrate_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        migrate(&conn).unwrap();

        let version: u32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, CURRENT_VERSION);
    }

    #[test]
    fn fts5_table_exists() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        // FTS5 virtual tables show as 'table' in sqlite_master
        let count: u32 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name='entries_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
