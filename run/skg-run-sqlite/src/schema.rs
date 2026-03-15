//! SQLite schema for durable run persistence.

use rusqlite::{Connection, Error as SqliteError};

/// Current schema version.
const CURRENT_VERSION: u32 = 1;

/// Apply any pending schema migrations.
pub(crate) fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    let version: u32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

    if version > CURRENT_VERSION {
        return Err(future_schema_error(version));
    }

    if version < 1 {
        migration_001(conn)?;
    }

    Ok(())
}

fn future_schema_error(version: u32) -> SqliteError {
    SqliteError::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_SCHEMA),
        Some(format!(
            "unsupported future schema version for skg-run-sqlite: found {version}, current {CURRENT_VERSION}"
        )),
    )
}

fn migration_001(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS run_records (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            record_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS run_resumes (
            run_id TEXT NOT NULL,
            wait_point TEXT NOT NULL,
            input_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (run_id, wait_point)
        );

        CREATE TABLE IF NOT EXISTS run_signals (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            signal_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_run_signals_run_id_signal_id
            ON run_signals (run_id, signal_id);

        CREATE TABLE IF NOT EXISTS run_timers (
            run_id TEXT NOT NULL,
            wait_point TEXT NOT NULL,
            wake_at TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (run_id, wait_point)
        );

        CREATE INDEX IF NOT EXISTS idx_run_timers_wake_at
            ON run_timers (wake_at, run_id, wait_point);

        CREATE TABLE IF NOT EXISTS checkpoints (
            id          TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL,
            step        INTEGER NOT NULL,
            operator_id TEXT NOT NULL,
            state       TEXT NOT NULL,
            parent_id   TEXT,
            created_at  INTEGER NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES checkpoints(id)
        );

        CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id, step);

        PRAGMA user_version = 1;
        "#,
    )?;

    let version: u32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    debug_assert_eq!(version, CURRENT_VERSION);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::migrate;
    use rusqlite::Connection;

    #[test]
    fn migrate_creates_all_tables() {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        migrate(&conn).expect("migrate schema");

        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN ('run_records', 'run_resumes', 'run_signals', 'run_timers', 'checkpoints')",
                [],
                |row| row.get(0),
            )
            .expect("count tables");
        assert_eq!(count, 5);
    }

    #[test]
    fn migrate_is_idempotent() {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        migrate(&conn).expect("migrate schema first time");
        migrate(&conn).expect("migrate schema second time");

        let version: u32 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .expect("read schema version");
        assert_eq!(version, 1);
    }
}
