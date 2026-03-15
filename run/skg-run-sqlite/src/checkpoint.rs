//! [`CheckpointStore`] implementation for [`SqliteRunStore`].

use async_trait::async_trait;
use layer0::id::OperatorId;
use rusqlite::{OptionalExtension, params};
use skg_run_core::checkpoint::{Checkpoint, CheckpointError, CheckpointStore};
use skg_run_core::id::{CheckpointId, RunId};

use crate::store::{sqlite_backend_error, SqliteRunStore};

/// Convert a serialisation error into [`CheckpointError::Store`].
fn ser_err(e: serde_json::Error) -> CheckpointError {
    CheckpointError::Store(Box::new(e))
}

/// Convert a rusqlite error with a context prefix into [`CheckpointError::Store`].
fn db_err(prefix: &str, e: rusqlite::Error) -> CheckpointError {
    CheckpointError::Store(sqlite_backend_error(prefix, e).into())
}

/// Map a single database row to a [`Checkpoint`].
///
/// Column order must match every SELECT in this file:
/// `id, run_id, step, operator_id, state, parent_id, created_at`
fn row_to_checkpoint(row: &rusqlite::Row<'_>) -> rusqlite::Result<Checkpoint> {
    let id: String = row.get(0)?;
    let run_id: String = row.get(1)?;
    let step: u32 = row.get(2)?;
    let operator_id: String = row.get(3)?;
    let state_json: String = row.get(4)?;
    let parent_id: Option<String> = row.get(5)?;
    let created_at: u64 = row.get(6)?;

    // state is stored as JSON text; parse it back into a Value.
    // Map the serde error through rusqlite's InvalidParameterName variant as a
    // transport — the only available error constructor that accepts a String.
    let state: serde_json::Value = serde_json::from_str(&state_json).map_err(|e| {
        rusqlite::Error::InvalidParameterName(format!("state JSON: {e}"))
    })?;

    Ok(Checkpoint {
        id: CheckpointId::new(id),
        run_id: RunId::new(run_id),
        step,
        operator_id: OperatorId::new(operator_id),
        state,
        parent: parent_id.map(CheckpointId::new),
        created_at,
    })
}

#[async_trait]
impl CheckpointStore for SqliteRunStore {
    async fn save_checkpoint(&self, checkpoint: Checkpoint) -> Result<CheckpointId, CheckpointError> {
        let id = checkpoint.id.clone();
        let state_json = serde_json::to_string(&checkpoint.state).map_err(ser_err)?;

        self.with_conn(
            |msg| CheckpointError::Store(msg.into()),
            move |conn| {
                conn.execute(
                    "INSERT INTO checkpoints (id, run_id, step, operator_id, state, parent_id, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        checkpoint.id.as_str(),
                        checkpoint.run_id.as_str(),
                        checkpoint.step,
                        checkpoint.operator_id.as_str(),
                        state_json,
                        checkpoint.parent.as_ref().map(CheckpointId::as_str),
                        checkpoint.created_at,
                    ],
                )
                .map_err(|e| db_err("save checkpoint", e))?;
                Ok(id)
            },
        )
    }

    async fn get_checkpoint(&self, id: &CheckpointId) -> Result<Option<Checkpoint>, CheckpointError> {
        let id = id.clone();
        self.with_conn(
            |msg| CheckpointError::Store(msg.into()),
            move |conn| {
                conn.query_row(
                    "SELECT id, run_id, step, operator_id, state, parent_id, created_at
                     FROM checkpoints WHERE id = ?1",
                    params![id.as_str()],
                    row_to_checkpoint,
                )
                .optional()
                .map_err(|e| db_err("get checkpoint", e))
            },
        )
    }

    async fn list_checkpoints(&self, run_id: &RunId) -> Result<Vec<Checkpoint>, CheckpointError> {
        let run_id = run_id.clone();
        self.with_conn(
            |msg| CheckpointError::Store(msg.into()),
            move |conn| {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, run_id, step, operator_id, state, parent_id, created_at
                         FROM checkpoints WHERE run_id = ?1 ORDER BY step ASC",
                    )
                    .map_err(|e| db_err("prepare list checkpoints", e))?;

                let rows = stmt
                    .query_map(params![run_id.as_str()], row_to_checkpoint)
                    .map_err(|e| db_err("query list checkpoints", e))?;

                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| db_err("collect list checkpoints", e))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SqliteRunStore;
    use skg_run_core::checkpoint::CheckpointStore;

    fn store() -> SqliteRunStore {
        SqliteRunStore::open_in_memory().expect("open in-memory store")
    }

    /// Build a checkpoint with a predictable created_at so tests are deterministic.
    fn cp(id: &str, run_id: &str, step: u32, parent: Option<&str>) -> Checkpoint {
        let mut c = Checkpoint::new(id, run_id, step, "op-a", serde_json::json!({"step": step}));
        if let Some(p) = parent {
            c = c.with_parent(p);
        }
        c
    }

    #[tokio::test]
    async fn save_and_get_checkpoint() {
        let store = store();
        let checkpoint = cp("cp-1", "run-1", 0, None);
        store.save_checkpoint(checkpoint.clone()).await.expect("save");

        let got = store.get_checkpoint(&CheckpointId::new("cp-1")).await.expect("get");
        let got = got.expect("should be Some");
        assert_eq!(got.id.as_str(), "cp-1");
        assert_eq!(got.run_id.as_str(), "run-1");
        assert_eq!(got.step, 0);
        assert_eq!(got.operator_id.as_str(), "op-a");
        assert!(got.parent.is_none());
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let store = store();
        let result = store.get_checkpoint(&CheckpointId::new("does-not-exist")).await.expect("no error");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_checkpoints_ordered_by_step() {
        let store = store();

        // Save in order so foreign key constraints (parent_id) are satisfied.
        store.save_checkpoint(cp("cp-0", "run-2", 0, None)).await.expect("save cp-0");
        store.save_checkpoint(cp("cp-1", "run-2", 1, Some("cp-0"))).await.expect("save cp-1");
        store.save_checkpoint(cp("cp-2", "run-2", 2, Some("cp-1"))).await.expect("save cp-2");

        let list = store.list_checkpoints(&RunId::new("run-2")).await.expect("list");
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].id.as_str(), "cp-0");
        assert_eq!(list[1].id.as_str(), "cp-1");
        assert_eq!(list[2].id.as_str(), "cp-2");

        // Verify parent chain is preserved.
        assert!(list[0].parent.is_none());
        assert_eq!(list[1].parent.as_ref().unwrap().as_str(), "cp-0");
        assert_eq!(list[2].parent.as_ref().unwrap().as_str(), "cp-1");
    }

    #[tokio::test]
    async fn list_checkpoints_for_nonexistent_run_returns_empty() {
        let store = store();
        let list = store.list_checkpoints(&RunId::new("no-such-run")).await.expect("list");
        assert!(list.is_empty());
    }
}
