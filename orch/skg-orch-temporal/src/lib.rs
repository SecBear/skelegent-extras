#![deny(missing_docs)]
//! Temporal-backed orchestration implementation for skelegent.
//!
//! # Feature flags
//!
//! | Flag | Effect |
//! |------|--------|
//! | `temporal-sdk` | Compiles the real Temporal durable run-control client. [`TemporalOrch::new()`](TemporalOrch::new) remains mock-backed; [`TemporalOrch::connect()`](TemporalOrch::connect) opts into the real server-backed path. |
//!
//! # Public surface
//!
//! This crate exports only configuration types plus [`TemporalOrch`]. The
//! orchestrator itself implements multiple public contracts from core crates:
//! - [`Dispatcher`] for immediate operator dispatch
//! - [`Signalable`] and [`Queryable`] for workflow signal/query operations
//! - [`RunStarter`] and [`RunController`] for durable run lifecycle control
//!
//! # Durable run-control honesty
//!
//! Without `temporal-sdk`, or when callers construct the orchestrator with
//! [`TemporalOrch::new()`](TemporalOrch::new), durable run control is
//! intentionally mock-backed and limited to the top-level portable surface:
//! start a run, inspect its current view, signal it, resume a waiting run, or
//! cancel it. This path does not claim Temporal replay/history fidelity; it
//! exists to exercise the public contract without native dependencies.
//!
//! Enabling `temporal-sdk` only compiles the real Temporal client path. The
//! Temporal-native backend becomes active only when callers use
//! [`TemporalOrch::connect()`](TemporalOrch::connect).
//!
//! With `temporal-sdk` plus `TemporalOrch::connect()`, durable run control maps
//! onto Temporal-native client operations against a deployed workflow
//! implementation:
//! - `start_run` starts the configured generic durable-run workflow type with
//!   the existing JSON start envelope, using the portable [`RunId`] as the
//!   Temporal workflow ID for this backend.
//! - `get_run` queries that workflow for a [`RunView`].
//! - `signal_run` sends a control-plane signal to the workflow.
//! - `resume_run` uses a distinct workflow update to satisfy a wait point.
//! - `cancel_run` requests workflow cancellation.
//!
//! # Constitution compliance
//!
//! Concrete service clients are private inside the crate and never exported.
//! Consumers programme against the [`Dispatcher`] trait from `layer0`,
//! [`Signalable`]/[`Queryable`] from `skg-effects-core`, and durable run-control
//! traits from `skg_run_core`.

mod client;
pub mod config;

pub use config::{RetryPolicy, TemporalConfig};

use async_trait::async_trait;
use client::{MockTemporalClient, TemporalClient, TemporalError};
use layer0::dispatch::Dispatcher;
use layer0::effect::SignalPayload;
use layer0::error::OrchError;
use layer0::id::{OperatorId, WorkflowId};
use layer0::operator::{Operator, OperatorInput, OperatorOutput};
use serde_json::{json, Value};
use skg_effects_core::{QueryPayload, Queryable, Signalable};
use skg_run_core::{
    ResumeInput, RunControlError, RunController, RunId, RunOutcome, RunStarter, RunView,
    WaitPointId, WaitReason,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

#[cfg(feature = "temporal-sdk")]
use temporalio_client::{
    Client as TemporalSdkClient, ClientOptions as TemporalClientOptions,
    Connection as TemporalConnection, ConnectionOptions as TemporalConnectionOptions,
    WorkflowCancelOptions, WorkflowExecuteUpdateOptions, WorkflowQueryOptions,
    WorkflowSignalOptions, WorkflowStartOptions,
};
#[cfg(feature = "temporal-sdk")]
use temporalio_client::errors::{
    ClientConnectError, ClientNewError, WorkflowInteractionError, WorkflowQueryError,
    WorkflowStartError, WorkflowUpdateError,
};
#[cfg(feature = "temporal-sdk")]
use temporalio_common::{QueryDefinition, SignalDefinition, UpdateDefinition, WorkflowDefinition};
#[cfg(feature = "temporal-sdk")]
use url::Url;
#[cfg(feature = "temporal-sdk")]
use uuid::Uuid;

#[cfg_attr(not(feature = "temporal-sdk"), allow(dead_code))]
const RUN_VIEW_QUERY_NAME: &str = "skg.run.view";
const RUN_CONTROL_SIGNAL_NAME: &str = "skg.run.control";
#[cfg_attr(not(feature = "temporal-sdk"), allow(dead_code))]
const RUN_RESUME_UPDATE_NAME: &str = "skg.run.resume";

/// Indicates which run-control path is active for this orchestrator instance.
#[cfg_attr(not(feature = "temporal-sdk"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunControlBackend {
    /// Default, dependency-light mock path used in tests.
    MockLocal,
    /// Temporal-native client path backed by a deployed workflow implementation.
    TemporalGrpc,
}

/// Mock-backed durable run record used by the default non-`temporal-sdk` path.
#[derive(Debug, Clone)]
struct DurableRunRecord {
    start: Value,
    view: RunView,
    signals: Vec<Value>,
}

#[cfg(feature = "temporal-sdk")]
#[derive(Clone)]
struct TemporalRunControlClient {
    client: TemporalSdkClient,
}

#[cfg(feature = "temporal-sdk")]
impl std::fmt::Debug for TemporalRunControlClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemporalRunControlClient").finish_non_exhaustive()
    }
}

#[cfg(feature = "temporal-sdk")]
#[derive(Debug, Clone)]
struct TemporalRunWorkflow {
    name: String,
}

#[cfg(feature = "temporal-sdk")]
impl WorkflowDefinition for TemporalRunWorkflow {
    type Input = Value;
    type Output = Value;

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(feature = "temporal-sdk")]
#[derive(Debug, Clone, Copy)]
struct TemporalRunViewQuery;

#[cfg(feature = "temporal-sdk")]
impl QueryDefinition for TemporalRunViewQuery {
    type Workflow = TemporalRunWorkflow;
    type Input = Value;
    type Output = RunView;

    fn name(&self) -> &str {
        RUN_VIEW_QUERY_NAME
    }
}

#[cfg(feature = "temporal-sdk")]
#[derive(Debug, Clone, Copy)]
struct TemporalRunSignal;

#[cfg(feature = "temporal-sdk")]
impl SignalDefinition for TemporalRunSignal {
    type Workflow = TemporalRunWorkflow;
    type Input = Value;

    fn name(&self) -> &str {
        RUN_CONTROL_SIGNAL_NAME
    }
}

#[cfg(feature = "temporal-sdk")]
#[derive(Debug, Clone, Copy)]
struct TemporalResumeUpdate;

#[cfg(feature = "temporal-sdk")]
impl UpdateDefinition for TemporalResumeUpdate {
    type Workflow = TemporalRunWorkflow;
    type Input = Value;
    type Output = Value;

    fn name(&self) -> &str {
        RUN_RESUME_UPDATE_NAME
    }
}

/// Temporal-backed orchestrator implementing [`Dispatcher`], [`Signalable`],
/// [`Queryable`], [`RunStarter`], and [`RunController`].
///
/// By default this uses an in-process [`MockTemporalClient`][crate::client]
/// so the crate can be compiled and tested without native dependencies.
/// Building with `temporal-sdk` does not change that default; callers must use
/// [`connect()`](TemporalOrch::connect) to activate the real Temporal backend.
///
/// The mock-backed run-control path is intentionally narrow and honest: it
/// offers the public durable run lifecycle surface, but does not claim
/// server-backed Temporal replay/history semantics.
///
/// When callers opt into [`connect()`](TemporalOrch::connect), run control
/// assumes the configured `workflow_type` points at a deployed generic
/// durable-run workflow that understands:
/// - query handler [`RUN_VIEW_QUERY_NAME`]
/// - signal handler [`RUN_CONTROL_SIGNAL_NAME`]
/// - update handler [`RUN_RESUME_UPDATE_NAME`]
///
/// # Example
///
/// ```rust,no_run
/// use skg_orch_temporal::{TemporalConfig, TemporalOrch};
/// use layer0::id::OperatorId;
/// use layer0::test_utils::EchoOperator;
/// use std::sync::Arc;
///
/// let mut orch = TemporalOrch::new(TemporalConfig::default());
/// orch.register(OperatorId::new("echo"), Arc::new(EchoOperator));
/// ```
pub struct TemporalOrch {
    client: Arc<dyn TemporalClient>,
    config: TemporalConfig,
    run_control_backend: RunControlBackend,
    /// Process-local counter used only by the mock-backed `TemporalOrch::new()` path.
    run_counter: AtomicU64,
    /// Shared with `MockTemporalClient` so the mock can dispatch to operators.
    agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>>,
    /// Per-workflow signal journal for `signal`/`query` semantics.
    workflow_signals: RwLock<HashMap<String, Vec<SignalPayload>>>,
    /// Mock-backed durable run read models for the portable run-control surface.
    durable_runs: RwLock<HashMap<String, DurableRunRecord>>,
    #[cfg(feature = "temporal-sdk")]
    temporal_run_control: Option<Arc<TemporalRunControlClient>>,
}

impl TemporalOrch {
    /// Create a new `TemporalOrch` with the given configuration.
    ///
    /// Uses [`MockTemporalClient`][crate::client] by default, even when the
    /// crate is compiled with `temporal-sdk`. Call [`connect()`](Self::connect)
    /// to activate the real Temporal server-backed run-control path.
    pub fn new(config: TemporalConfig) -> Self {
        let agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let client: Arc<dyn TemporalClient> = Arc::new(MockTemporalClient::new(Arc::clone(&agents)));
        Self {
            client,
            config,
            run_control_backend: RunControlBackend::MockLocal,
            run_counter: AtomicU64::new(0),
            agents,
            workflow_signals: RwLock::new(HashMap::new()),
            durable_runs: RwLock::new(HashMap::new()),
            #[cfg(feature = "temporal-sdk")]
            temporal_run_control: None,
        }
    }

    /// Connect to a real Temporal server.
    ///
    /// Only available when the crate is compiled with `temporal-sdk`. Unlike
    /// [`new()`](Self::new), this opts the orchestrator into the real Temporal
    /// durable run-control backend. In this backend the portable [`RunId`] is
    /// realized as the Temporal workflow ID, while Temporal's server-assigned
    /// run ID remains internal.
    ///
    /// The returned orchestrator uses the existing internal gRPC client for the
    /// dispatch surface and a Temporal-native client for durable run control.
    /// Operators registered via [`register()`](TemporalOrch::register) are used
    /// for local dispatch only on the mock-backed [`new()`](Self::new) path.
    #[cfg(feature = "temporal-sdk")]
    pub async fn connect(config: TemporalConfig) -> Result<Self, OrchError> {
        use client::GrpcTemporalClient;

        let agents = Arc::new(Mutex::new(HashMap::new()));
        let grpc_client = GrpcTemporalClient::connect(&config)
            .await
            .map_err(|e| OrchError::DispatchFailed(format!("temporal connect: {e}")))?;
        let temporal_run_control = TemporalRunControlClient::connect(&config)
            .await
            .map_err(|e| OrchError::DispatchFailed(format!("temporal run control connect: {e}")))?;
        Ok(Self {
            client: Arc::new(grpc_client),
            config,
            run_control_backend: RunControlBackend::TemporalGrpc,
            run_counter: AtomicU64::new(0),
            agents,
            workflow_signals: RwLock::new(HashMap::new()),
            durable_runs: RwLock::new(HashMap::new()),
            temporal_run_control: Some(Arc::new(temporal_run_control)),
        })
    }

    /// Register an operator under the given agent ID.
    ///
    /// After registration, `dispatch(&operator_id, …)` will route to `op`.
    /// Registering the same ID twice replaces the previous operator.
    pub fn register(&mut self, id: OperatorId, op: Arc<dyn Operator>) {
        self.agents
            .lock()
            .expect("agents mutex poisoned")
            .insert(id.to_string(), op);
    }

    /// Return a reference to the server configuration.
    pub fn config(&self) -> &TemporalConfig {
        &self.config
    }

    fn next_mock_portable_run_id(&self) -> RunId {
        let counter = self.run_counter.fetch_add(1, Ordering::Relaxed);
        RunId::new(format!("temporal-run-{counter}"))
    }

    #[cfg(feature = "temporal-sdk")]
    fn next_temporal_workflow_backed_run_id() -> RunId {
        // Portable RunId stays external and backend-agnostic; this backend realizes it
        // as the Temporal workflow ID rather than exposing Temporal's server run ID.
        RunId::new(Uuid::new_v4().to_string())
    }

    fn default_wait_point(run_id: &RunId) -> WaitPointId {
        WaitPointId::new(format!("{}:external-input", run_id.as_str()))
    }

    fn validate_start_input(input: &Value) -> Result<(), RunControlError> {
        let object = input.as_object().ok_or_else(|| {
            RunControlError::InvalidInput(
                "TemporalOrch start_run expects an object like {\"operator_id\": string, \"input\": <value> }"
                    .into(),
            )
        })?;
        let operator_id = object
            .get("operator_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RunControlError::InvalidInput(
                    "TemporalOrch start_run requires a string field `operator_id`".into(),
                )
            })?;
        if operator_id.is_empty() {
            return Err(RunControlError::InvalidInput(
                "TemporalOrch start_run requires a non-empty `operator_id`".into(),
            ));
        }
        if !object.contains_key("input") {
            return Err(RunControlError::InvalidInput(
                "TemporalOrch start_run requires an `input` field".into(),
            ));
        }
        Ok(())
    }

    async fn load_run_record(&self, run_id: &RunId) -> Result<DurableRunRecord, RunControlError> {
        self.durable_runs
            .read()
            .await
            .get(run_id.as_str())
            .cloned()
            .ok_or_else(|| RunControlError::RunNotFound(run_id.clone()))
    }

    fn terminal_signal_conflict(run_id: &RunId, status: impl std::fmt::Debug) -> RunControlError {
        RunControlError::Conflict(format!(
            "cannot signal terminal run {} with status {:?}",
            run_id, status
        ))
    }

    fn terminal_cancel_conflict(run_id: &RunId, status: impl std::fmt::Debug) -> RunControlError {
        RunControlError::Conflict(format!(
            "cannot cancel terminal run {} with status {:?}",
            run_id, status
        ))
    }

    #[cfg(feature = "temporal-sdk")]
    fn temporal_backend_contract_message(&self, operation: &str) -> String {
        format!(
            "{operation} requires a deployed Temporal workflow type '{}' that accepts the generic durable-run contract via query '{}' / signal '{}' / update '{}'",
            self.config.workflow_type,
            RUN_VIEW_QUERY_NAME,
            RUN_CONTROL_SIGNAL_NAME,
            RUN_RESUME_UPDATE_NAME,
        )
    }

    #[cfg(feature = "temporal-sdk")]
    fn temporal_run_control_client(&self) -> Result<&TemporalRunControlClient, RunControlError> {
        self.temporal_run_control
            .as_deref()
            .ok_or_else(|| RunControlError::Backend("temporal run-control client is not configured".into()))
    }

    #[cfg(feature = "temporal-sdk")]
    async fn start_run_temporal(&self, input: Value) -> Result<RunId, RunControlError> {
        let run_id = Self::next_temporal_workflow_backed_run_id();
        self.temporal_run_control_client()?
            .start_run(&self.config, &run_id, input)
            .await
            .map_err(|error| match error {
                WorkflowStartError::AlreadyStarted { run_id: existing, .. } => RunControlError::Conflict(
                    format!(
                        "run {} already exists in Temporal{}",
                        run_id,
                        existing
                            .as_deref()
                            .map(|value| format!(" (server-assigned run id {value})"))
                            .unwrap_or_default()
                    ),
                ),
                other => RunControlError::Backend(format!(
                    "{}: {other}",
                    self.temporal_backend_contract_message("start_run")
                )),
            })?;
        Ok(run_id)
    }

    #[cfg(feature = "temporal-sdk")]
    async fn get_run_temporal(&self, run_id: &RunId) -> Result<RunView, RunControlError> {
        self.temporal_run_control_client()?
            .get_run(run_id)
            .await
            .map_err(|error| match error {
                WorkflowQueryError::NotFound(_) => RunControlError::RunNotFound(run_id.clone()),
                other => RunControlError::Backend(format!(
                    "{}: {other}",
                    self.temporal_backend_contract_message("get_run")
                )),
            })
    }

    #[cfg(feature = "temporal-sdk")]
    async fn signal_run_temporal(&self, run_id: &RunId, signal: Value) -> Result<(), RunControlError> {
        match self.get_run_temporal(run_id).await? {
            RunView::Running { .. } | RunView::Waiting { .. } => {}
            other => return Err(Self::terminal_signal_conflict(run_id, other.status())),
        }
        self.temporal_run_control_client()?
            .signal_run(run_id, signal)
            .await
            .map_err(|error| match error {
                WorkflowInteractionError::NotFound(_) => RunControlError::RunNotFound(run_id.clone()),
                other => RunControlError::Backend(format!(
                    "{}: {other}",
                    self.temporal_backend_contract_message("signal_run")
                )),
            })
    }

    #[cfg(feature = "temporal-sdk")]
    async fn resume_run_temporal(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
        input: ResumeInput,
    ) -> Result<(), RunControlError> {
        let active_wait = match self.get_run_temporal(run_id).await? {
            RunView::Waiting { wait_point, .. } => wait_point,
            other => {
                return Err(RunControlError::Conflict(format!(
                    "cannot resume run {} while in status {:?}",
                    run_id,
                    other.status()
                )));
            }
        };
        if active_wait != *wait_point {
            return Err(RunControlError::WaitPointNotFound(wait_point.clone()));
        }
        self.temporal_run_control_client()?
            .resume_run(run_id, wait_point, input)
            .await
            .map_err(|error| match error {
                WorkflowUpdateError::NotFound(_) => RunControlError::RunNotFound(run_id.clone()),
                other => RunControlError::Backend(format!(
                    "{}: {other}",
                    self.temporal_backend_contract_message("resume_run")
                )),
            })
    }

    #[cfg(feature = "temporal-sdk")]
    async fn cancel_run_temporal(&self, run_id: &RunId) -> Result<(), RunControlError> {
        match self.get_run_temporal(run_id).await? {
            RunView::Running { .. } | RunView::Waiting { .. } => {}
            other => return Err(Self::terminal_cancel_conflict(run_id, other.status())),
        }
        self.temporal_run_control_client()?
            .cancel_run(run_id)
            .await
            .map_err(|error| match error {
                WorkflowInteractionError::NotFound(_) => RunControlError::RunNotFound(run_id.clone()),
                other => RunControlError::Backend(format!(
                    "{}: {other}",
                    self.temporal_backend_contract_message("cancel_run")
                )),
            })
    }
}

#[cfg(feature = "temporal-sdk")]
impl TemporalRunControlClient {
    async fn connect(config: &TemporalConfig) -> Result<Self, String> {
        let target = temporal_server_url(config).map_err(|error| error.to_string())?;
        let connection = TemporalConnection::connect(
            TemporalConnectionOptions::new(target)
                .identity(temporal_identity(config))
                .build(),
        )
        .await
        .map_err(connection_error_message)?;
        let client = TemporalSdkClient::new(
            connection,
            TemporalClientOptions::new(config.namespace.clone()).build(),
        )
        .map_err(client_new_error_message)?;
        Ok(Self { client })
    }

    async fn start_run(
        &self,
        config: &TemporalConfig,
        run_id: &RunId,
        input: Value,
    ) -> Result<(), WorkflowStartError> {
        self.client
            .start_workflow(
                TemporalRunWorkflow {
                    name: config.workflow_type.clone(),
                },
                input,
                WorkflowStartOptions::new(config.task_queue.clone(), run_id.as_str().to_string()).build(),
            )
            .await
            .map(|_| ())
    }

    async fn get_run(&self, run_id: &RunId) -> Result<RunView, WorkflowQueryError> {
        self.client
            .get_workflow_handle::<TemporalRunWorkflow>(run_id.as_str())
            .query(TemporalRunViewQuery, Value::Null, WorkflowQueryOptions::default())
            .await
    }

    async fn signal_run(
        &self,
        run_id: &RunId,
        signal: Value,
    ) -> Result<(), WorkflowInteractionError> {
        self.client
            .get_workflow_handle::<TemporalRunWorkflow>(run_id.as_str())
            .signal(TemporalRunSignal, signal, WorkflowSignalOptions::default())
            .await
    }

    async fn resume_run(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
        input: ResumeInput,
    ) -> Result<(), WorkflowUpdateError> {
        self.client
            .get_workflow_handle::<TemporalRunWorkflow>(run_id.as_str())
            .execute_update(
                TemporalResumeUpdate,
                json!({
                    "wait_point": wait_point.as_str(),
                    "payload": input.payload,
                    "metadata": input.metadata,
                }),
                WorkflowExecuteUpdateOptions::default(),
            )
            .await
            .map(|_| ())
    }

    async fn cancel_run(&self, run_id: &RunId) -> Result<(), WorkflowInteractionError> {
        self.client
            .get_workflow_handle::<TemporalRunWorkflow>(run_id.as_str())
            .cancel(WorkflowCancelOptions::default())
            .await
    }
}

#[cfg(feature = "temporal-sdk")]
fn temporal_server_url(config: &TemporalConfig) -> Result<Url, url::ParseError> {
    let server = config.server_url.trim();
    let normalized = if server.contains("://") {
        server.to_string()
    } else {
        format!("http://{server}")
    };
    Url::parse(&normalized)
}

#[cfg(feature = "temporal-sdk")]
fn temporal_identity(config: &TemporalConfig) -> String {
    if config.identity.is_empty() {
        "skg-orch-temporal".to_string()
    } else {
        config.identity.clone()
    }
}

#[cfg(feature = "temporal-sdk")]
fn connection_error_message(error: ClientConnectError) -> String {
    error.to_string()
}

#[cfg(feature = "temporal-sdk")]
fn client_new_error_message(error: ClientNewError) -> String {
    error.to_string()
}

// ── Dispatcher impl ──────────────────────────────────────────────────────

#[async_trait]
impl Dispatcher for TemporalOrch {
    async fn dispatch(
        &self,
        operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<OperatorOutput, OrchError> {
        let bytes = serde_json::to_vec(&input)
            .map_err(|e| OrchError::DispatchFailed(format!("serialization: {e}")))?;

        let result_bytes = self
            .client
            .execute_activity(operator.as_str(), bytes)
            .await
            .map_err(|e| match e {
                TemporalError::WorkflowNotFound(msg) => OrchError::OperatorNotFound(msg),
                other => OrchError::DispatchFailed(other.to_string()),
            })?;

        serde_json::from_slice(&result_bytes)
            .map_err(|e| OrchError::DispatchFailed(format!("deserialization: {e}")))
    }
}

// ── Signalable + Queryable impl ──────────────────────────────────────────

#[async_trait]
impl Signalable for TemporalOrch {
    async fn signal(&self, target: &WorkflowId, signal: SignalPayload) -> Result<(), OrchError> {
        let bytes = serde_json::to_vec(&signal)
            .map_err(|e| OrchError::SignalFailed(format!("serialization: {e}")))?;

        self.client
            .signal_workflow(target.as_str(), &signal.signal_type, bytes)
            .await
            .map_err(|e| OrchError::SignalFailed(e.to_string()))?;

        let mut workflows = self.workflow_signals.write().await;
        workflows
            .entry(target.to_string())
            .or_default()
            .push(signal);
        Ok(())
    }
}

#[async_trait]
impl Queryable for TemporalOrch {
    async fn query(
        &self,
        target: &WorkflowId,
        query: QueryPayload,
    ) -> Result<serde_json::Value, OrchError> {
        if query.query_type == "run-view" {
            let run_id = RunId::new(target.as_str());
            let run = match self.run_control_backend {
                RunControlBackend::MockLocal => self.load_run_record(&run_id).await.map(|record| record.view),
                #[cfg(feature = "temporal-sdk")]
                RunControlBackend::TemporalGrpc => self.get_run_temporal(&run_id).await,
                #[cfg(not(feature = "temporal-sdk"))]
                RunControlBackend::TemporalGrpc => unreachable!("temporal-sdk gated variant"),
            }
            .map_err(|error| match error {
                RunControlError::RunNotFound(_) => OrchError::WorkflowNotFound(target.to_string()),
                other => OrchError::DispatchFailed(other.to_string()),
            })?;
            return serde_json::to_value(&run)
                .map_err(|e| OrchError::DispatchFailed(format!("serialization: {e}")));
        }

        let workflows = self.workflow_signals.read().await;
        let count = workflows
            .get(target.as_str())
            .map(std::vec::Vec::len)
            .unwrap_or(0);
        Ok(json!({ "signals": count }))
    }
}

// ── Durable run-control impl ─────────────────────────────────────────────

#[async_trait]
impl RunStarter for TemporalOrch {
    async fn start_run(&self, input: Value) -> Result<RunId, RunControlError> {
        Self::validate_start_input(&input)?;
        match self.run_control_backend {
            RunControlBackend::MockLocal => {
                let run_id = self.next_mock_portable_run_id();
                let wait_point = Self::default_wait_point(&run_id);
                let bytes = serde_json::to_vec(&input)
                    .map_err(|e| RunControlError::InvalidInput(format!("serialization: {e}")))?;
                self.client
                    .start_workflow(run_id.as_str(), &self.config.task_queue, bytes)
                    .await
                    .map_err(|e| RunControlError::Backend(e.to_string()))?;

                let record = DurableRunRecord {
                    start: input,
                    view: RunView::waiting(run_id.clone(), wait_point, WaitReason::ExternalInput),
                    signals: Vec::new(),
                };
                self.durable_runs
                    .write()
                    .await
                    .insert(run_id.to_string(), record);
                Ok(run_id)
            }
            #[cfg(feature = "temporal-sdk")]
            RunControlBackend::TemporalGrpc => self.start_run_temporal(input).await,
            #[cfg(not(feature = "temporal-sdk"))]
            RunControlBackend::TemporalGrpc => unreachable!("temporal-sdk gated variant"),
        }
    }
}

#[async_trait]
impl RunController for TemporalOrch {
    async fn get_run(&self, run_id: &RunId) -> Result<RunView, RunControlError> {
        match self.run_control_backend {
            RunControlBackend::MockLocal => Ok(self.load_run_record(run_id).await?.view),
            #[cfg(feature = "temporal-sdk")]
            RunControlBackend::TemporalGrpc => self.get_run_temporal(run_id).await,
            #[cfg(not(feature = "temporal-sdk"))]
            RunControlBackend::TemporalGrpc => unreachable!("temporal-sdk gated variant"),
        }
    }

    async fn signal_run(&self, run_id: &RunId, signal: Value) -> Result<(), RunControlError> {
        match self.run_control_backend {
            RunControlBackend::MockLocal => {
                let mut runs = self.durable_runs.write().await;
                let record = runs
                    .get_mut(run_id.as_str())
                    .ok_or_else(|| RunControlError::RunNotFound(run_id.clone()))?;
                if !matches!(record.view, RunView::Running { .. } | RunView::Waiting { .. }) {
                    return Err(Self::terminal_signal_conflict(run_id, record.view.status()));
                }

                let bytes = serde_json::to_vec(&signal)
                    .map_err(|e| RunControlError::InvalidInput(format!("serialization: {e}")))?;
                self.client
                    .signal_workflow(run_id.as_str(), RUN_CONTROL_SIGNAL_NAME, bytes)
                    .await
                    .map_err(|e| RunControlError::Backend(e.to_string()))?;
                record.signals.push(signal);
                Ok(())
            }
            #[cfg(feature = "temporal-sdk")]
            RunControlBackend::TemporalGrpc => self.signal_run_temporal(run_id, signal).await,
            #[cfg(not(feature = "temporal-sdk"))]
            RunControlBackend::TemporalGrpc => unreachable!("temporal-sdk gated variant"),
        }
    }

    async fn resume_run(
        &self,
        run_id: &RunId,
        wait_point: &WaitPointId,
        input: ResumeInput,
    ) -> Result<(), RunControlError> {
        match self.run_control_backend {
            RunControlBackend::MockLocal => {
                let mut runs = self.durable_runs.write().await;
                let record = runs
                    .get_mut(run_id.as_str())
                    .ok_or_else(|| RunControlError::RunNotFound(run_id.clone()))?;
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

                let result = json!({
                    "source": "resume",
                    "start": record.start.clone(),
                    "resume": input.payload,
                    "resume_metadata": input.metadata,
                    "signals": record.signals.clone(),
                });
                record.view = RunView::terminal(
                    run_id.clone(),
                    RunOutcome::Completed { result },
                );
                Ok(())
            }
            #[cfg(feature = "temporal-sdk")]
            RunControlBackend::TemporalGrpc => self.resume_run_temporal(run_id, wait_point, input).await,
            #[cfg(not(feature = "temporal-sdk"))]
            RunControlBackend::TemporalGrpc => unreachable!("temporal-sdk gated variant"),
        }
    }

    async fn cancel_run(&self, run_id: &RunId) -> Result<(), RunControlError> {
        match self.run_control_backend {
            RunControlBackend::MockLocal => {
                let mut runs = self.durable_runs.write().await;
                let record = runs
                    .get_mut(run_id.as_str())
                    .ok_or_else(|| RunControlError::RunNotFound(run_id.clone()))?;
                if !matches!(record.view, RunView::Running { .. } | RunView::Waiting { .. }) {
                    return Err(Self::terminal_cancel_conflict(run_id, record.view.status()));
                }
                record.view = RunView::terminal(run_id.clone(), RunOutcome::Cancelled);
                Ok(())
            }
            #[cfg(feature = "temporal-sdk")]
            RunControlBackend::TemporalGrpc => self.cancel_run_temporal(run_id).await,
            #[cfg(not(feature = "temporal-sdk"))]
            RunControlBackend::TemporalGrpc => unreachable!("temporal-sdk gated variant"),
        }
    }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn mock_portable_run_ids_remain_process_local_and_deterministic() {
		let orch = TemporalOrch::new(TemporalConfig::default());
		assert_eq!(orch.next_mock_portable_run_id().as_str(), "temporal-run-0");
		assert_eq!(orch.next_mock_portable_run_id().as_str(), "temporal-run-1");
	}

	#[cfg(feature = "temporal-sdk")]
	#[test]
	fn temporal_workflow_backed_run_ids_are_restart_safe_uuids() {
		let first = TemporalOrch::next_temporal_workflow_backed_run_id();
		let second = TemporalOrch::next_temporal_workflow_backed_run_id();

		assert_ne!(first, second);
		assert!(uuid::Uuid::parse_str(first.as_str()).is_ok());
		assert!(uuid::Uuid::parse_str(second.as_str()).is_ok());
	}
}
