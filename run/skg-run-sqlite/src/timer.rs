//! SQLite durable timer-store implementation.

use async_trait::async_trait;
use rusqlite::params;
use skg_run_core::{
    PortableWakeDeadline, RunId, ScheduledTimer, TimerStore, TimerStoreError, WaitPointId,
};

use crate::SqliteRunStore;
use crate::store::{is_constraint_violation, sqlite_backend_error};

#[async_trait]
impl TimerStore for SqliteRunStore {
    async fn schedule_timer(&self, timer: ScheduledTimer) -> Result<(), TimerStoreError> {
        let run_id = timer.run_id.clone();
        let wait_point = timer.wait_point.clone();
        let wake_at = timer.wake_at.clone();

        self.with_conn(TimerStoreError::Backend, move |conn| {
            match conn.execute(
                "INSERT INTO run_timers (run_id, wait_point, wake_at)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(run_id, wait_point) DO UPDATE SET
                     wake_at = excluded.wake_at,
                     updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                params![run_id.as_str(), wait_point.as_str(), wake_at.as_str()],
            ) {
                Ok(_) => Ok(()),
                Err(error) if is_constraint_violation(&error) => {
                    Err(TimerStoreError::Conflict(format!(
                        "timer write conflicted for run {} wait point {}",
                        run_id, wait_point
                    )))
                }
                Err(error) => Err(TimerStoreError::Backend(sqlite_backend_error(
                    "schedule timer",
                    error,
                ))),
            }
        })
    }

    async fn cancel_timer(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
    ) -> Result<(), TimerStoreError> {
        let run_id = run_id.clone();
        let wait_point = wait_point.clone();

        self.with_conn(TimerStoreError::Backend, move |conn| {
            let deleted = conn
                .execute(
                    "DELETE FROM run_timers WHERE run_id = ?1 AND wait_point = ?2",
                    params![run_id.as_str(), wait_point.as_str()],
                )
                .map_err(|error| TimerStoreError::Backend(sqlite_backend_error("cancel timer", error)))?;

            if deleted == 0 {
                return Err(TimerStoreError::TimerNotFound { run_id, wait_point });
            }

            Ok(())
        })
    }

    async fn due_timers(
        &self,
        not_after: &PortableWakeDeadline,
        limit: usize,
    ) -> Result<Vec<ScheduledTimer>, TimerStoreError> {
        let not_after = not_after.clone();

        self.with_conn(TimerStoreError::Backend, move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT run_id, wait_point, wake_at
                     FROM run_timers
                     WHERE wake_at <= ?1
                     ORDER BY wake_at ASC, run_id ASC, wait_point ASC
                     LIMIT ?2",
                )
                .map_err(|error| TimerStoreError::Backend(sqlite_backend_error("prepare due timers", error)))?;
            let rows = stmt
                .query_map(params![not_after.as_str(), limit as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(|error| TimerStoreError::Backend(sqlite_backend_error("query due timers", error)))?;
            let raw = rows
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| TimerStoreError::Backend(sqlite_backend_error("read due timers", error)))?;

            raw.into_iter()
                .map(|(run_id, wait_point, wake_at)| {
                    let wake_at = PortableWakeDeadline::parse(&wake_at).map_err(|error| {
                        TimerStoreError::Backend(format!(
                            "deserialize timer deadline for run {} wait point {}: {error}",
                            run_id, wait_point
                        ))
                    })?;
                    Ok(ScheduledTimer::new(
                        RunId::new(run_id),
                        WaitPointId::new(wait_point),
                        wake_at,
                    ))
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skg_run_core::TimerStore;

    #[tokio::test]
    async fn missing_timer_returns_typed_not_found_error() {
        let store = SqliteRunStore::open_in_memory().expect("open store");
        let run_id = RunId::new("run-missing");
        let wait_point = WaitPointId::new("wait-missing");

        let error = store
            .cancel_timer(&run_id, &wait_point)
            .await
            .expect_err("missing timer should return typed error");

        assert!(matches!(
            error,
            TimerStoreError::TimerNotFound { run_id: found_run, wait_point: found_wait }
                if found_run == run_id && found_wait == wait_point
        ));
    }
}
