#![deny(missing_docs)]
//! Temporal-backed [`Orchestrator`] implementation for neuron.
//!
//! # Feature flags
//!
//! | Flag | Effect |
//! |------|--------|
//! | `temporal-sdk` | Enables the real Temporal gRPC client (requires cmake + protobuf). Without this flag the crate compiles and tests with a [`MockTemporalClient`][crate::client] that runs operators in-process. |
//!
//! # Public surface
//!
//! Only three types are exported:
//! - [`TemporalOrch`] — the `Orchestrator` implementation
//! - [`TemporalConfig`] — server connection settings
//! - [`RetryPolicy`] — retry/backoff parameters
//!
//! Everything else (Temporal gRPC client, error types, mock) is private.
//!
//! # Constitution compliance
//!
//! Concrete service clients are private inside the crate and never exported.
//! Consumers programme against the [`Orchestrator`] trait from `layer0`.

mod client;
pub mod config;

pub use config::{RetryPolicy, TemporalConfig};

use async_trait::async_trait;
use client::{MockTemporalClient, TemporalClient, TemporalError};
use layer0::effect::SignalPayload;
use layer0::error::OrchError;
use layer0::id::{OperatorId, WorkflowId};
use layer0::operator::{Operator, OperatorInput, OperatorOutput};
use layer0::orchestrator::{Orchestrator, QueryPayload};
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

// ── TemporalOrch ───────────────────────────────────────────────────────────

/// Temporal-backed orchestrator implementing the layer0 [`Orchestrator`] trait.
///
/// By default (no `temporal-sdk` feature) this uses an in-process
/// [`MockTemporalClient`][crate::client] so the crate can be compiled and
/// tested without any native dependencies. Enable the `temporal-sdk` feature
/// to connect to a real Temporal server.
///
/// # Example
///
/// ```rust,no_run
/// use neuron_orch_temporal::{TemporalConfig, TemporalOrch};
/// use layer0::test_utils::EchoOperator;
/// use layer0::id::OperatorId;
/// use std::sync::Arc;
///
/// let mut orch = TemporalOrch::new(TemporalConfig::default());
/// orch.register(OperatorId::new("echo"), Arc::new(EchoOperator));
/// ```
pub struct TemporalOrch {
    client: Arc<dyn TemporalClient>,
    config: TemporalConfig,
    /// Shared with `MockTemporalClient` so the mock can dispatch to operators.
    agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>>,
    /// Per-workflow signal journal for `signal`/`query` semantics.
    workflow_signals: RwLock<HashMap<String, Vec<SignalPayload>>>,
}

impl TemporalOrch {
    /// Create a new `TemporalOrch` with the given configuration.
    ///
    /// Uses [`MockTemporalClient`][crate::client] by default (no native deps).
    /// Enable the `temporal-sdk` feature to use a real Temporal server.
    pub fn new(config: TemporalConfig) -> Self {
        let agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let client: Arc<dyn TemporalClient> =
            Arc::new(MockTemporalClient::new(Arc::clone(&agents)));
        Self {
            client,
            config,
            agents,
            workflow_signals: RwLock::new(HashMap::new()),
        }
    }

    /// Connect to a real Temporal server.
    ///
    /// Only available with the `temporal-sdk` feature. Returns an error if
    /// the server is unreachable.
    ///
    /// The returned orchestrator uses a gRPC client for all operations.
    /// Operators registered via [`register()`](TemporalOrch::register) are
    /// used for local dispatch only when running without the feature flag.
    #[cfg(feature = "temporal-sdk")]
    pub async fn connect(config: TemporalConfig) -> Result<Self, OrchError> {
        use client::GrpcTemporalClient;
        let agents = Arc::new(Mutex::new(HashMap::new()));
        let grpc_client = GrpcTemporalClient::connect(&config)
            .await
            .map_err(|e| OrchError::DispatchFailed(format!("temporal connect: {e}")))?;
        Ok(Self {
            client: Arc::new(grpc_client),
            config,
            agents,
            workflow_signals: RwLock::new(HashMap::new()),
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
}

// ── Orchestrator impl ──────────────────────────────────────────────────────

#[async_trait]
impl Orchestrator for TemporalOrch {
    async fn dispatch(
        &self,
        operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<OperatorOutput, OrchError> {
        // Serialise input for the client abstraction layer.
        let bytes = serde_json::to_vec(&input)
            .map_err(|e| OrchError::DispatchFailed(format!("serialization: {e}")))?;

        // Dispatch through the client (mock or real).
        let result_bytes = self
            .client
            .execute_activity(operator.as_str(), bytes)
            .await
            .map_err(|e| match e {
                TemporalError::WorkflowNotFound(msg) => OrchError::OperatorNotFound(msg),
                other => OrchError::DispatchFailed(other.to_string()),
            })?;

        // Deserialise the response.
        serde_json::from_slice(&result_bytes)
            .map_err(|e| OrchError::DispatchFailed(format!("deserialization: {e}")))
    }

    async fn dispatch_many(
        &self,
        tasks: Vec<(OperatorId, OperatorInput)>,
    ) -> Vec<Result<OperatorOutput, OrchError>> {
        // Phase 1: sequential dispatch. Temporal handles true parallelism in
        // Phase 2 via child workflows / activity scheduling.
        let mut results = Vec::with_capacity(tasks.len());
        for (operator_id, input) in tasks {
            results.push(self.dispatch(&operator_id, input).await);
        }
        results
    }

    async fn signal(&self, target: &WorkflowId, signal: SignalPayload) -> Result<(), OrchError> {
        // Serialise the payload for the client.
        let bytes = serde_json::to_vec(&signal)
            .map_err(|e| OrchError::SignalFailed(format!("serialization: {e}")))?;

        // Forward to the client (no-op in mock; durable in real SDK).
        self.client
            .signal_workflow(target.as_str(), &signal.signal_type, bytes)
            .await
            .map_err(|e| OrchError::SignalFailed(e.to_string()))?;

        // Always record locally — the journal is the source of truth for
        // `query()` in Phase 1, and acts as a local cache in Phase 2.
        let mut workflows = self.workflow_signals.write().await;
        workflows
            .entry(target.to_string())
            .or_default()
            .push(signal);
        Ok(())
    }

    async fn query(
        &self,
        target: &WorkflowId,
        _query: QueryPayload,
    ) -> Result<serde_json::Value, OrchError> {
        // Phase 1: read from the local in-memory signal journal.
        // Phase 2: call client.query_workflow() and fall back to journal on failure.
        let workflows = self.workflow_signals.read().await;
        let count = workflows
            .get(target.as_str())
            .map(|v| v.len())
            .unwrap_or(0);
        Ok(json!({ "signals": count }))
    }
}
