//! SQLite durable run and wait-store implementation.

use async_trait::async_trait;
use rusqlite::{Connection, Error as SqliteError, ErrorCode, OptionalExtension, Transaction, params};
use serde_json::Value;
use skg_run_core::{
    PendingResume, PendingSignal, RunId, RunStatus, RunStore, RunStoreError, StoreRunRecord,
    WaitPointId, WaitStore, WaitStoreError,
};
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed durable run, wait, and timer persistence store.
///
/// A single SQLite connection is protected by an interior [`Mutex`] so the same
/// instance can satisfy all lower-level backend persistence seams.
pub struct SqliteRunStore {
    pub(crate) conn: Mutex<Connection>,
}

impl SqliteRunStore {
    /// Open or create a SQLite durable run store at `path`.
    ///
    /// The schema is migrated automatically on open.
    ///
    /// # Errors
    ///
    /// Returns any SQLite open or migration error.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SqliteError> {
        let conn = Connection::open(path.as_ref())?;
        conn.pragma_update(None, "journal_mode", "wal")?;
        conn.pragma_update(None, "foreign_keys", "on")?;
        crate::schema::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory SQLite durable run store.
    ///
    /// # Errors
    ///
    /// Returns any SQLite open or migration error.
    pub fn open_in_memory() -> Result<Self, SqliteError> {
        Self::open(":memory:")
    }

    pub(crate) fn with_conn<T, E>(
        &self,
        map_lock_error: impl FnOnce(String) -> E,
        f: impl FnOnce(&mut Connection) -> Result<T, E>,
    ) -> Result<T, E> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|error| map_lock_error(format!("mutex poisoned: {error}")))?;
        f(&mut conn)
    }
}

// Safety: the only connection is protected by a Mutex and never shared across await points.
unsafe impl Send for SqliteRunStore {}
// Safety: the only connection is protected by a Mutex and never shared across await points.
unsafe impl Sync for SqliteRunStore {}

pub(crate) fn is_constraint_violation(error: &SqliteError) -> bool {
    matches!(
        error,
        SqliteError::SqliteFailure(code, _) if code.code == ErrorCode::ConstraintViolation
    )
}

pub(crate) fn sqlite_backend_error(prefix: &str, error: SqliteError) -> String {
    format!("{prefix}: {error}")
}

pub(crate) fn encode_json(value: &Value) -> Result<String, serde_json::Error> {
    serde_json::to_string(value)
}

pub(crate) fn decode_json(raw: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(raw)
}

pub(crate) fn encode_record(run: &StoreRunRecord) -> Result<String, serde_json::Error> {
    serde_json::to_string(run)
}

pub(crate) fn decode_record(raw: &str) -> Result<StoreRunRecord, serde_json::Error> {
    serde_json::from_str(raw)
}

pub(crate) fn encode_resume_input(resume: &PendingResume) -> Result<String, serde_json::Error> {
    serde_json::to_string(&resume.input)
}

pub(crate) fn decode_resume(
    run_id: &RunId,
    wait_point: &WaitPointId,
    raw: &str,
) -> Result<PendingResume, serde_json::Error> {
    Ok(PendingResume::new(
        run_id.clone(),
        wait_point.clone(),
        serde_json::from_str(raw)?,
    ))
}

pub(crate) fn decode_signal(run_id: &RunId, raw: &str) -> Result<PendingSignal, serde_json::Error> {
    Ok(PendingSignal::new(run_id.clone(), decode_json(raw)?))
}

pub(crate) fn tx_take_resume(
    tx: &Transaction<'_>,
    run_id: &RunId,
    wait_point: &WaitPointId,
) -> Result<Option<String>, SqliteError> {
    let input_json = tx
        .query_row(
            "SELECT input_json FROM run_resumes WHERE run_id = ?1 AND wait_point = ?2",
            params![run_id.as_str(), wait_point.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()?;

    if input_json.is_some() {
        tx.execute(
            "DELETE FROM run_resumes WHERE run_id = ?1 AND wait_point = ?2",
            params![run_id.as_str(), wait_point.as_str()],
        )?;
    }

    Ok(input_json)
}

pub(crate) fn tx_drain_signals(
    tx: &Transaction<'_>,
    run_id: &RunId,
) -> Result<Vec<String>, SqliteError> {
    let mut stmt = tx.prepare(
        "SELECT signal_json FROM run_signals WHERE run_id = ?1 ORDER BY signal_id ASC",
    )?;
    let rows = stmt.query_map(params![run_id.as_str()], |row| row.get::<_, String>(0))?;
    let signals = rows.collect::<Result<Vec<_>, _>>()?;
    drop(stmt);

    if !signals.is_empty() {
        tx.execute(
            "DELETE FROM run_signals WHERE run_id = ?1",
            params![run_id.as_str()],
        )?;
    }

    Ok(signals)
}

fn status_name(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Running => "running",
        RunStatus::Waiting => "waiting",
        RunStatus::Completed => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
        _ => "unknown",
    }
}

#[async_trait]
impl RunStore for SqliteRunStore {
    async fn insert_run(&self, run: StoreRunRecord) -> Result<(), RunStoreError> {
        let run_id = run.view.run_id().clone();
        let record_json = encode_record(&run)
            .map_err(|error| RunStoreError::Backend(format!("serialize run record: {error}")))?;
        let status = status_name(run.view.status());

        self.with_conn(RunStoreError::Backend, move |conn| {
            match conn.execute(
                "INSERT INTO run_records (run_id, status, record_json) VALUES (?1, ?2, ?3)",
                params![run_id.as_str(), status, record_json],
            ) {
                Ok(_) => Ok(()),
                Err(error) if is_constraint_violation(&error) => {
                    Err(RunStoreError::Conflict(format!("run record already exists: {run_id}")))
                }
                Err(error) => Err(RunStoreError::Backend(sqlite_backend_error(
                    "insert run record",
                    error,
                ))),
            }
        })
    }

    async fn get_run(&self, run_id: &RunId) -> Result<Option<StoreRunRecord>, RunStoreError> {
        let run_id = run_id.clone();
        self.with_conn(RunStoreError::Backend, move |conn| {
            let record_json = conn
                .query_row(
                    "SELECT record_json FROM run_records WHERE run_id = ?1",
                    params![run_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|error| RunStoreError::Backend(sqlite_backend_error("load run record", error)))?;

            record_json
                .map(|raw| {
                    decode_record(&raw).map_err(|error| {
                        RunStoreError::Backend(format!("deserialize run record for {run_id}: {error}"))
                    })
                })
                .transpose()
        })
    }

    async fn put_run(&self, run: StoreRunRecord) -> Result<(), RunStoreError> {
        let run_id = run.view.run_id().clone();
        let record_json = encode_record(&run)
            .map_err(|error| RunStoreError::Backend(format!("serialize run record: {error}")))?;
        let status = status_name(run.view.status());

        self.with_conn(RunStoreError::Backend, move |conn| {
            let updated = conn
                .execute(
                    "UPDATE run_records
                     SET status = ?2,
                         record_json = ?3,
                         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                     WHERE run_id = ?1",
                    params![run_id.as_str(), status, record_json],
                )
                .map_err(|error| RunStoreError::Backend(sqlite_backend_error("update run record", error)))?;

            if updated == 0 {
                return Err(RunStoreError::RunNotFound(run_id));
            }

            Ok(())
        })
    }
}

#[async_trait]
impl WaitStore for SqliteRunStore {
    async fn save_resume(&self, resume: PendingResume) -> Result<(), WaitStoreError> {
        let run_id = resume.run_id.clone();
        let wait_point = resume.wait_point.clone();
        let input_json = encode_resume_input(&resume)
            .map_err(|error| WaitStoreError::Backend(format!("serialize resume input: {error}")))?;

        self.with_conn(WaitStoreError::Backend, move |conn| {
            conn.execute(
                "INSERT INTO run_resumes (run_id, wait_point, input_json)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(run_id, wait_point) DO UPDATE SET
                     input_json = excluded.input_json,
                     updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                params![run_id.as_str(), wait_point.as_str(), input_json],
            )
            .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("save resume", error)))?;
            Ok(())
        })
    }

    async fn take_resume(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
    ) -> Result<Option<PendingResume>, WaitStoreError> {
        let run_id = run_id.clone();
        let wait_point = wait_point.clone();

        self.with_conn(WaitStoreError::Backend, move |conn| {
            let tx = conn
                .transaction()
                .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("begin take resume", error)))?;
            let input_json = tx_take_resume(&tx, &run_id, &wait_point)
                .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("take resume", error)))?;
            tx.commit()
                .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("commit take resume", error)))?;

            input_json
                .map(|raw| {
                    decode_resume(&run_id, &wait_point, &raw).map_err(|error| {
                        WaitStoreError::Backend(format!(
                            "deserialize resume for run {} wait point {}: {error}",
                            run_id, wait_point
                        ))
                    })
                })
                .transpose()
        })
    }

    async fn push_signal(&self, signal: PendingSignal) -> Result<(), WaitStoreError> {
        let run_id = signal.run_id.clone();
        let signal_json = encode_json(&signal.signal)
            .map_err(|error| WaitStoreError::Backend(format!("serialize signal: {error}")))?;

        self.with_conn(WaitStoreError::Backend, move |conn| {
            conn.execute(
                "INSERT INTO run_signals (run_id, signal_json) VALUES (?1, ?2)",
                params![run_id.as_str(), signal_json],
            )
            .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("push signal", error)))?;
            Ok(())
        })
    }

    async fn drain_signals(&self, run_id: &RunId) -> Result<Vec<PendingSignal>, WaitStoreError> {
        let run_id = run_id.clone();

        self.with_conn(WaitStoreError::Backend, move |conn| {
            let tx = conn
                .transaction()
                .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("begin drain signals", error)))?;
            let raw_signals = tx_drain_signals(&tx, &run_id)
                .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("drain signals", error)))?;
            tx.commit()
                .map_err(|error| WaitStoreError::Backend(sqlite_backend_error("commit drain signals", error)))?;

            raw_signals
                .into_iter()
                .map(|raw| {
                    decode_signal(&run_id, &raw).map_err(|error| {
                        WaitStoreError::Backend(format!("deserialize signal for run {}: {error}", run_id))
                    })
                })
                .collect()
        })
    }
}
