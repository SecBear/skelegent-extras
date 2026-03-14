use crate::driver::SqliteRunDriver;
use async_trait::async_trait;
use layer0::{Operator, OperatorId};
use serde_json::{Value, json};
use skg_run_core::{
    BackendRunRef, DriverError, DriverRequest, OrchestrationCommand, PendingResume, ResumeAction,
    ResumeInput, RunControlError, RunController, RunDriver, RunEvent, RunId, RunKernel,
    RunObserver, RunStarter, RunStatus, RunStore, RunStoreError, RunSubscription, RunUpdate,
    RunView, StoreRunRecord, TimerStore, TimerStoreError, WaitPointId, WaitStore, WaitStoreError,
};
use skg_run_sqlite::SqliteRunStore;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// SQLite-backed durable orchestrator assembled from portable run/control seams,
/// SQLite persistence, and in-process operator dispatch.
pub struct SqliteDurableOrchestrator {
    store: Arc<SqliteRunStore>,
    driver: SqliteRunDriver,
    run_counter: AtomicU64,
    /// Serializes crash-recovery replay and state-mutating operations so that
    /// concurrent `get_run`, `resume_run`, and `cancel_run` calls cannot race
    /// through the same replay path. Intentionally coarse — correctness over
    /// throughput.
    resume_lock: tokio::sync::Mutex<()>,
    /// Per-run broadcast channels for [`RunUpdate`] events.
    update_channels: std::sync::Mutex<HashMap<RunId, tokio::sync::broadcast::Sender<RunUpdate>>>,
}

const PENDING_RESUME_BACKEND_REF_PREFIX: &str = "sqlite:pending-resume:";

impl SqliteDurableOrchestrator {
    /// Create a new SQLite durable orchestrator.
    pub fn new(store: Arc<SqliteRunStore>) -> Self {
        Self {
            store,
            driver: SqliteRunDriver::new(),
            run_counter: AtomicU64::new(0),
            resume_lock: tokio::sync::Mutex::new(()),
            update_channels: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Register an operator for durable dispatch.
    pub fn register(&mut self, operator_id: OperatorId, operator: Arc<dyn Operator>) {
        self.driver.register(operator_id, operator);
    }

    /// Start a durable run for a specific registered operator.
    pub async fn start_operator_run(
        &self,
        operator_id: OperatorId,
        input: Value,
    ) -> Result<RunId, RunControlError> {
        self.start_run(json!({
            "operator_id": operator_id.as_str(),
            "input": input,
        }))
        .await
    }

    fn next_run_id(&self) -> RunId {
        let counter = self.run_counter.fetch_add(1, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        RunId::new(format!("sqlite-run-{now}-{counter}"))
    }

    /// Get or create a broadcast channel for a run.
    fn get_or_create_channel(&self, run_id: &RunId) -> tokio::sync::broadcast::Sender<RunUpdate> {
        let mut channels = self.update_channels.lock().expect("channel lock poisoned");
        channels
            .entry(run_id.clone())
            .or_insert_with(|| {
                let (tx, _) = tokio::sync::broadcast::channel(64);
                tx
            })
            .clone()
    }

    /// Emit a [`RunUpdate`] to subscribers, if any exist.
    fn emit_update(&self, update: RunUpdate) {
        let run_id = match &update {
            RunUpdate::StatusChanged { run_id, .. } => run_id,
            RunUpdate::ArtifactProduced { run_id, .. } => run_id,
            _ => return, // future non_exhaustive variants
        };
        let channels = self.update_channels.lock().expect("channel lock poisoned");
        if let Some(tx) = channels.get(run_id) {
            let _ = tx.send(update); // no receivers = no-op
        }
    }

    /// Clean up a channel after a run reaches a terminal state.
    fn cleanup_channel(&self, run_id: &RunId) {
        let mut channels = self.update_channels.lock().expect("channel lock poisoned");
        channels.remove(run_id);
    }

    /// Emit a status-changed update derived from a [`RunView`], and clean up
    /// the channel if the run has reached a terminal state.
    fn emit_status_from_view(&self, view: &RunView) {
        self.emit_update(RunUpdate::StatusChanged {
            run_id: view.run_id().clone(),
            status: view.status(),
        });
        match view.status() {
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled => {
                self.cleanup_channel(view.run_id());
            }
            _ => {}
        }
    }

    /// Pure read — fetches the persisted record without triggering replay.
    async fn get_record(&self, run_id: &RunId) -> Result<StoreRunRecord, RunControlError> {
        self.store
            .get_run(run_id)
            .await
            .map_err(map_run_store_error)?
            .ok_or_else(|| RunControlError::RunNotFound(run_id.clone()))
    }

    /// Acquires `resume_lock`, reads the record, then replays any pending
    /// crash-recovery resume. This is the ONLY path that may trigger replay.
    async fn load_and_replay(&self, run_id: &RunId) -> Result<StoreRunRecord, RunControlError> {
        let _guard = self.resume_lock.lock().await;
        let record = self.get_record(run_id).await?;
        self.replay_pending_resume_if_any(record).await
    }

    async fn replay_pending_resume_if_any(
        &self,
        record: StoreRunRecord,
    ) -> Result<StoreRunRecord, RunControlError> {
        let pending = match &record.view {
            RunView::Waiting {
                run_id, wait_point, ..
            } => self
                .store
                .load_resume(run_id, wait_point)
                .await
                .map_err(map_wait_store_error)?,
            RunView::Running { run_id } => match pending_resume_wait_point(record.backend_ref.as_ref()) {
                Some(wait_point) => self
                    .store
                    .load_resume(run_id, &wait_point)
                    .await
                    .map_err(map_wait_store_error)?,
                None => None,
            },
            _ => None,
        };

        match pending {
            Some(pending) => self.dispatch_persisted_resume(record, pending).await,
            None => Ok(record),
        }
    }

    async fn dispatch_persisted_resume(
        &self,
        current_record: StoreRunRecord,
        pending_resume: PendingResume,
    ) -> Result<StoreRunRecord, RunControlError> {
        let run_id = pending_resume.run_id.clone();
        let wait_point = pending_resume.wait_point.clone();
        let dispatch_payload = skg_run_core::DispatchPayload::Resume {
            wait_point: wait_point.clone(),
            input: pending_resume.input.clone(),
        };

        let running_record = match &current_record.view {
            RunView::Waiting { .. } => {
                let resume_transition = RunKernel::apply(
                    Some(&current_record.view),
                    RunEvent::Resume {
                        wait_point: wait_point.clone(),
                        input: pending_resume.input.clone(),
                        action: ResumeAction::Continue,
                    },
                )
                .map_err(map_kernel_error)?;

                for command in &resume_transition.commands {
                    if let OrchestrationCommand::CancelWake { run_id, wait_point } = command {
                        match self.store.cancel_timer(run_id, wait_point).await {
                            Ok(()) | Err(TimerStoreError::TimerNotFound { .. }) => {}
                            Err(error) => return Err(map_timer_store_error(error)),
                        }
                    }
                }

                let base_backend_ref = current_record.backend_ref.as_ref().ok_or_else(|| {
                    RunControlError::Conflict(format!(
                        "missing backend ref for run {} before replaying persisted resume",
                        run_id
                    ))
                })?;
                let running_record = StoreRunRecord::new(
                    resume_transition.next,
                    Some(pending_resume_backend_ref(&wait_point, base_backend_ref)),
                );
                self.persist_record(Some(&current_record), &running_record).await?;
                self.emit_status_from_view(&running_record.view);
                running_record
            }
            RunView::Running { .. } => current_record,
            other => {
                return Err(RunControlError::Conflict(format!(
                    "cannot replay pending resume for run {} while in status {:?}",
                    run_id,
                    other.status()
                )))
            }
        };

        let response = match self
            .driver
            .drive_run(DriverRequest::new(
                run_id.clone(),
                dispatch_payload,
                strip_pending_resume_backend_ref(running_record.backend_ref.as_ref()),
            ))
            .await
        {
            Ok(response) => response,
            Err(error) => {
                let (failure_summary, control_error) = split_driver_error(error);
                self.persist_driver_failure(
                    &StoreRunRecord::new(
                        running_record.view.clone(),
                        strip_pending_resume_backend_ref(running_record.backend_ref.as_ref()),
                    ),
                    failure_summary,
                )
                .await?;
                self.clear_resume_if_wait_resolved(&run_id, &wait_point).await?;
                return Err(control_error);
            }
        };

        let updated_running = StoreRunRecord::new(
            running_record.view,
            response
                .backend_ref
                .or_else(|| strip_pending_resume_backend_ref(running_record.backend_ref.as_ref())),
        );
        let result = self.process_event(Some(updated_running), response.next_event).await;
        self.clear_resume_if_wait_resolved(&run_id, &wait_point).await?;
        result
    }


    async fn persist_record(
        &self,
        previous: Option<&StoreRunRecord>,
        next: &StoreRunRecord,
    ) -> Result<(), RunControlError> {
        if let Some(_previous) = previous {
            self.store
                .put_run(next.clone())
                .await
                .map_err(map_run_store_error)
        } else {
            self.store
                .insert_run(next.clone())
                .await
                .map_err(map_run_store_error)
        }
    }

    async fn persist_driver_failure(
        &self,
        current: &StoreRunRecord,
        error: String,
    ) -> Result<(), RunControlError> {
        let transition = RunKernel::apply(
            Some(&current.view),
            RunEvent::Fail {
                error: error.clone(),
            },
        )
        .map_err(map_kernel_error)?;
        let failed_record = StoreRunRecord::new(transition.next, current.backend_ref.clone());
        self.persist_record(Some(current), &failed_record).await
    }

    async fn clear_resume_if_wait_resolved(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
    ) -> Result<(), RunControlError> {
        let Some(record) = self.store.get_run(run_id).await.map_err(map_run_store_error)? else {
            return Ok(());
        };
        if record.view.wait_point() == Some(wait_point) {
            return Ok(());
        }

        if let Err(error) = self.store.take_resume(run_id, wait_point).await {
            eprintln!(
                "warning: sqlite durable orchestrator left a stale pending resume after resolving the wait for run {} wait point {}: {}",
                run_id, wait_point, error
            );
        }
        Ok(())
    }

    async fn process_event(
        &self,
        current: Option<StoreRunRecord>,
        event: RunEvent,
    ) -> Result<StoreRunRecord, RunControlError> {
        let mut record = current;
        let persisted = record.is_some();
        let mut next_event = Some(event);
        let mut commands = VecDeque::new();

        loop {
            if let Some(event) = next_event.take() {
                let transition =
                    RunKernel::apply(record.as_ref().map(|current| &current.view), event)
                        .map_err(map_kernel_error)?;
                let next_record = StoreRunRecord::new(
                    transition.next,
                    record
                        .as_ref()
                        .and_then(|current| current.backend_ref.clone()),
                );
                if persisted {
                    self.persist_record(record.as_ref(), &next_record).await?;
                }
                commands.extend(transition.commands);
                record = Some(next_record);
                // Notify subscribers of the state transition.
                if let Some(ref rec) = record {
                    self.emit_status_from_view(&rec.view);
                }
                continue;
            }

            let Some(command) = commands.pop_front() else {
                let final_record = record.ok_or_else(|| {
                    RunControlError::Backend("run processing lost current record".into())
                })?;
                if !persisted {
                    self.persist_record(None, &final_record).await?;
                }
                return Ok(final_record);
            };

            match command {
                OrchestrationCommand::DispatchOperator { run_id, payload } => {
                    let current_record = record.clone().ok_or_else(|| {
                        RunControlError::Backend("dispatch missing current record".into())
                    })?;
                    let response = match self
                        .driver
                        .drive_run(DriverRequest::new(
                            run_id,
                            payload,
                            current_record.backend_ref.clone(),
                        ))
                        .await
                    {
                        Ok(response) => response,
                        Err(error) => {
                            let (failure_summary, control_error) = split_driver_error(error);
                            if persisted {
                                self.persist_driver_failure(&current_record, failure_summary)
                                    .await?;
                            }
                            return Err(control_error);
                        }
                    };
                    let updated_record = StoreRunRecord::new(
                        current_record.view,
                        response.backend_ref.or(current_record.backend_ref),
                    );
                    if persisted {
                        self.store
                            .put_run(updated_record.clone())
                            .await
                            .map_err(map_run_store_error)?;
                    }
                    record = Some(updated_record);
                    next_event = Some(response.next_event);
                }
                OrchestrationCommand::EnterWaitPoint { .. }
                | OrchestrationCommand::CompleteRun { .. }
                | OrchestrationCommand::FailRun { .. }
                | OrchestrationCommand::CancelRun { .. } => {}
                OrchestrationCommand::CancelWake { run_id, wait_point } => {
                    match self.store.cancel_timer(&run_id, &wait_point).await {
                        Ok(()) | Err(TimerStoreError::TimerNotFound { .. }) => {}
                        Err(error) => return Err(map_timer_store_error(error)),
                    }
                }
                OrchestrationCommand::ScheduleWake {
                    run_id,
                    wait_point,
                    wake_at,
                } => {
                    let current_record = record.clone().ok_or_else(|| {
                        RunControlError::Backend("schedule wake missing current record".into())
                    })?;
                    let message = format!(
                        "timed waits are unsupported by sqlite durable orchestrator until wake scheduling is durably coordinated: run {} wait point {} wake_at {}",
                        run_id, wait_point, wake_at
                    );
                    if persisted {
                        self.persist_driver_failure(&current_record, message.clone())
                            .await?;
                    }
                    return Err(RunControlError::InvalidInput(message));
                }
                _ => {
                    let current_record = record.clone().ok_or_else(|| {
                        RunControlError::Backend("unknown command missing current record".into())
                    })?;
                    let message = "unsupported orchestration command for sqlite durable orchestrator".to_string();
                    if persisted {
                        self.persist_driver_failure(&current_record, message.clone())
                            .await?;
                    }
                    return Err(RunControlError::Backend(message));
                }
            }
        }
    }
}

#[async_trait]
impl RunStarter for SqliteDurableOrchestrator {
    /// Start a durable run.
    ///
    /// The generic portable start contract currently expects a JSON envelope of:
    /// `{ "operator_id": string, "input": <value> }`.
    async fn start_run(&self, input: Value) -> Result<RunId, RunControlError> {
        let run_id = self.next_run_id();
        self.process_event(
            None,
            RunEvent::Start {
                run_id: run_id.clone(),
                input,
            },
        )
        .await?;
        Ok(run_id)
    }
}

#[async_trait]
impl RunController for SqliteDurableOrchestrator {
    async fn get_run(&self, run_id: &RunId) -> Result<RunView, RunControlError> {
        Ok(self.load_and_replay(run_id).await?.view)
    }

    async fn signal_run(&self, run_id: &RunId, _signal: Value) -> Result<(), RunControlError> {
        let record = self.get_record(run_id).await?;
        if !matches!(
            record.view,
            RunView::Running { .. } | RunView::Waiting { .. }
        ) {
            return Err(RunControlError::Conflict(format!(
                "cannot signal terminal run {} with status {:?}",
                run_id,
                record.view.status()
            )));
        }

        Err(RunControlError::Backend(format!(
            "signal delivery is unsupported by sqlite durable orchestrator for run {}",
            run_id
        )))
    }

    async fn resume_run(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
        input: ResumeInput,
    ) -> Result<(), RunControlError> {
        let record = self.load_and_replay(run_id).await?;
        let active_wait = match &record.view {
            RunView::Waiting { wait_point, .. } => wait_point,
            other => {
                return Err(RunControlError::Conflict(format!(
                    "cannot resume run {} while in status {:?}",
                    run_id,
                    other.status()
                )));
            }
        };
        if active_wait != wait_point {
            return Err(RunControlError::WaitPointNotFound(wait_point.clone()));
        }

        // Persist the accepted resume while the durable record still says
        // `Waiting`. We only clear that row after a later durable state has been
        // recorded, so a crash cannot lose an accepted resume between durable
        // checkpoints.
        let pending_resume = PendingResume::new(
            run_id.clone(),
            wait_point.clone(),
            input,
        );
        self.store
            .save_resume(pending_resume.clone())
            .await
            .map_err(map_wait_store_error)?;

        self.dispatch_persisted_resume(record, pending_resume)
            .await
            .map(|_| ())?;
        Ok(())
    }

    async fn cancel_run(&self, run_id: &RunId) -> Result<(), RunControlError> {
        let record = self.load_and_replay(run_id).await?;
        self.process_event(Some(record), RunEvent::Cancel).await?;
        Ok(())
    }
}

#[async_trait]
impl RunObserver for SqliteDurableOrchestrator {
    async fn subscribe(&self, run_id: &RunId) -> Result<RunSubscription, RunControlError> {
        // Verify the run exists before creating a channel.
        let _record = self.get_record(run_id).await?;
        let tx = self.get_or_create_channel(run_id);
        Ok(RunSubscription::new(tx.subscribe()))
    }
}

fn map_run_store_error(error: RunStoreError) -> RunControlError {
    match error {
        RunStoreError::RunNotFound(run_id) => RunControlError::RunNotFound(run_id),
        RunStoreError::Conflict(message) => RunControlError::Conflict(message),
        RunStoreError::Backend(message) => RunControlError::Backend(message),
    }
}

fn map_wait_store_error(error: WaitStoreError) -> RunControlError {
    match error {
        WaitStoreError::Conflict(message) => RunControlError::Conflict(message),
        WaitStoreError::Backend(message) => RunControlError::Backend(message),
    }
}

fn pending_resume_backend_ref(
    wait_point: &WaitPointId,
    backend_ref: &BackendRunRef,
) -> BackendRunRef {
    let payload = serde_json::json!({
        "wait_point": wait_point.as_str(),
        "backend_ref": backend_ref.as_str()
    });
    BackendRunRef::new(format!(
        "{}{}",
        PENDING_RESUME_BACKEND_REF_PREFIX,
        payload
    ))
}

fn pending_resume_wait_point(backend_ref: Option<&BackendRunRef>) -> Option<WaitPointId> {
    let value = backend_ref?.as_str().strip_prefix(PENDING_RESUME_BACKEND_REF_PREFIX)?;
    let parsed: serde_json::Value = serde_json::from_str(value).ok()?;
    let wp = parsed.get("wait_point")?.as_str()?;
    Some(WaitPointId::new(wp))
}

fn strip_pending_resume_backend_ref(
    backend_ref: Option<&BackendRunRef>,
) -> Option<BackendRunRef> {
    match backend_ref {
        Some(value) if value.as_str().starts_with(PENDING_RESUME_BACKEND_REF_PREFIX) => {
            let json_str = value.as_str().strip_prefix(PENDING_RESUME_BACKEND_REF_PREFIX)?;
            let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
            let original = parsed.get("backend_ref")?.as_str()?;
            Some(BackendRunRef::new(original))
        }
        Some(value) => Some(value.clone()),
        None => None,
    }
}


fn map_timer_store_error(error: TimerStoreError) -> RunControlError {
    match error {
        TimerStoreError::TimerNotFound { wait_point, .. } => {
            RunControlError::WaitPointNotFound(wait_point)
        }
        TimerStoreError::Conflict(message) => RunControlError::Conflict(message),
        TimerStoreError::Backend(message) => RunControlError::Backend(message),
    }
}

fn split_driver_error(error: DriverError) -> (String, RunControlError) {
    match error {
        DriverError::Conflict(message) => (message.clone(), RunControlError::Conflict(message)),
        DriverError::InvalidInput(message) => {
            (message.clone(), RunControlError::InvalidInput(message))
        }
        DriverError::Backend(message) => (message.clone(), RunControlError::Backend(message)),
    }
}

fn map_kernel_error(error: skg_run_core::KernelError) -> RunControlError {
    match error {
        skg_run_core::KernelError::InvalidTransition { status, event } => {
            RunControlError::Conflict(format!(
                "invalid durable transition from {status} via {event}"
            ))
        }
        skg_run_core::KernelError::InvalidResumeToken {
            expected, found, ..
        } => RunControlError::Conflict(format!(
            "resume token mismatch: expected {expected}, found {found}"
        )),
        other => RunControlError::Backend(format!("unsupported kernel error: {other}")),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{Content, DispatchContext, Effect, ExitReason, OperatorError, OperatorInput, OperatorOutput, Scope};
    use serde_json::json;
    use layer0::dispatch::EffectEmitter;

    struct UnsupportedEffectOnStartOperator;

    #[async_trait]
    impl Operator for UnsupportedEffectOnStartOperator {
        async fn execute(&self, _input: OperatorInput, _ctx: &DispatchContext, _emitter: &EffectEmitter) -> Result<OperatorOutput, OperatorError> {
            let mut output =
                OperatorOutput::new(Content::text("start should fail"), ExitReason::Complete);
            output.effects = vec![Effect::WriteMemory {
                scope: Scope::Global,
                key: "approval:result".into(),
                value: json!("approved"),
                tier: None,
                lifetime: None,
                content_kind: None,
                salience: None,
                ttl: None,
            }];
            Ok(output)
        }
    }

    struct TimedWaitOnStartOperator;

    #[async_trait]
    impl Operator for TimedWaitOnStartOperator {
        async fn execute(&self, _input: OperatorInput, _ctx: &DispatchContext, _emitter: &EffectEmitter) -> Result<OperatorOutput, OperatorError> {
            Ok(OperatorOutput::new(
                Content::text(
                    json!({
                        "kind": "wait",
                        "wait_point": "timed-token",
                        "reason": "timer",
                        "wake_at": "2026-03-12T08:15:30Z"
                    })
                    .to_string(),
                ),
                ExitReason::Complete,
            ))
        }
    }

    #[tokio::test]
    async fn start_driver_failure_does_not_create_durable_run() {
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open run store"));
        let mut orch = SqliteDurableOrchestrator::new(store);
        orch.register(
            OperatorId::new("unsupported_effect_on_start"),
            Arc::new(UnsupportedEffectOnStartOperator),
        );
        let run_id = RunId::new("run-start-driver-failure");

        let err = orch
            .process_event(
                None,
                RunEvent::Start {
                    run_id: run_id.clone(),
                    input: json!({
                        "operator_id": "unsupported_effect_on_start",
                        "input": { "ticket": 21 }
                    }),
                },
            )
            .await
            .expect_err("surface unsupported effect failure");
        assert!(matches!(
            &err,
            RunControlError::InvalidInput(message)
                if message.contains("operator effects are unsupported")
                    && message.contains("write_memory")
        ));
        assert!(matches!(
            orch.get_run(&run_id).await,
            Err(RunControlError::RunNotFound(found)) if found == run_id
        ));
    }

    #[tokio::test]
    async fn start_timed_wait_failure_does_not_create_durable_run() {
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open run store"));
        let mut orch = SqliteDurableOrchestrator::new(store);
        orch.register(
            OperatorId::new("timed_wait_on_start"),
            Arc::new(TimedWaitOnStartOperator),
        );
        let run_id = RunId::new("run-start-timed-wait-failure");

        let err = orch
            .process_event(
                None,
                RunEvent::Start {
                    run_id: run_id.clone(),
                    input: json!({
                        "operator_id": "timed_wait_on_start",
                        "input": { "ticket": 22 }
                    }),
                },
            )
            .await
            .expect_err("surface unsupported timed wait failure");
        assert!(matches!(
            &err,
            RunControlError::InvalidInput(message)
                if message.contains("timed waits are unsupported")
        ));
        assert!(matches!(
            orch.get_run(&run_id).await,
            Err(RunControlError::RunNotFound(found)) if found == run_id
        ));
    }
}