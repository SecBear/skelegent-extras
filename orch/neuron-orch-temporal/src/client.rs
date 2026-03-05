//! Internal Temporal client abstraction.
//!
//! This module is crate-private. Consumers interact with the Temporal backend
//! exclusively through the [`layer0::Orchestrator`] trait, never through these
//! types directly.

use async_trait::async_trait;
use layer0::operator::{Operator, OperatorInput};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors surfaced by Temporal client operations.
#[derive(Debug, Error)]
pub(crate) enum TemporalError {
    #[allow(dead_code)]
    /// Failed to establish or maintain a connection to the Temporal server.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// The referenced workflow execution was not found on the server.
    #[error("workflow not found: {0}")]
    WorkflowNotFound(String),

    #[allow(dead_code)]
    /// The operation exceeded its deadline.
    #[error("timeout: {0}")]
    Timeout(String),

    /// Serialization or deserialization of a payload failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// An unclassified error occurred.
    #[error("other: {0}")]
    Other(String),
}

/// Internal abstraction over Temporal server communication.
///
/// NOT exported — consumers use the `Orchestrator` trait from layer0.
/// Having this trait makes it possible to swap in a `MockTemporalClient`
/// for tests without any native dependencies.
#[async_trait]
pub(crate) trait TemporalClient: Send + Sync {
    #[allow(dead_code)]
    /// Start a new workflow execution and return the run ID.
    async fn start_workflow(
        &self,
        workflow_id: &str,
        task_queue: &str,
        input: Vec<u8>,
    ) -> Result<String, TemporalError>;

    /// Send a fire-and-forget signal to a running workflow.
    async fn signal_workflow(
        &self,
        workflow_id: &str,
        signal_name: &str,
        input: Vec<u8>,
    ) -> Result<(), TemporalError>;

    #[allow(dead_code)]
    /// Perform a synchronous read-only query against a running workflow.
    async fn query_workflow(
        &self,
        workflow_id: &str,
        query_type: &str,
        args: Vec<u8>,
    ) -> Result<Vec<u8>, TemporalError>;

    /// Execute an activity (operator) by agent ID with serialised input bytes.
    ///
    /// Returns the serialised [`layer0::OperatorOutput`] on success.
    async fn execute_activity(
        &self,
        activity_id: &str,
        input: Vec<u8>,
    ) -> Result<Vec<u8>, TemporalError>;
}

// ── MockTemporalClient ─────────────────────────────────────────────────────

/// A single recorded activity call.
#[allow(dead_code)]
pub(crate) struct ActivityCall {
    /// The agent/activity ID that was invoked.
    pub(crate) activity_id: String,
    /// The raw serialised input that was passed.
    pub(crate) input: Vec<u8>,
}

/// In-process mock Temporal client used when the `temporal-sdk` feature is off.
///
/// Executes activities by dispatching to locally registered [`Operator`]
/// implementations. All calls are recorded so tests can assert on call counts
/// and inputs without making real network connections.
pub(crate) struct MockTemporalClient {
    /// Shared operator registry (same `Arc` as `TemporalOrch.agents`).
    agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>>,
    /// Append-only log of every `execute_activity` invocation.
    call_log: Mutex<Vec<ActivityCall>>,
}

impl MockTemporalClient {
    /// Create a new mock client that shares the given agent registry.
    pub(crate) fn new(agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>>) -> Self {
        Self {
            agents,
            call_log: Mutex::new(Vec::new()),
        }
    }

    /// Return the number of `execute_activity` calls recorded so far.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn call_count(&self) -> usize {
        self.call_log.lock().expect("call_log mutex poisoned").len()
    }
}

#[async_trait]
impl TemporalClient for MockTemporalClient {
    async fn start_workflow(
        &self,
        workflow_id: &str,
        _task_queue: &str,
        _input: Vec<u8>,
    ) -> Result<String, TemporalError> {
        Ok(workflow_id.to_string())
    }

    async fn signal_workflow(
        &self,
        _workflow_id: &str,
        _signal_name: &str,
        _input: Vec<u8>,
    ) -> Result<(), TemporalError> {
        Ok(())
    }

    async fn query_workflow(
        &self,
        _workflow_id: &str,
        _query_type: &str,
        _args: Vec<u8>,
    ) -> Result<Vec<u8>, TemporalError> {
        // The real client would ask the server; the mock defers to the local
        // signal journal maintained by TemporalOrch, so just return a neutral
        // value.  TemporalOrch::query() always reads the local journal anyway.
        let response = serde_json::json!({ "signals": 0 });
        serde_json::to_vec(&response).map_err(|e| TemporalError::Serialization(e.to_string()))
    }

    async fn execute_activity(
        &self,
        activity_id: &str,
        input: Vec<u8>,
    ) -> Result<Vec<u8>, TemporalError> {
        // ── 1. Record the call (mutex held only for the push) ──────────────
        {
            let mut log = self.call_log.lock().expect("call_log mutex poisoned");
            log.push(ActivityCall {
                activity_id: activity_id.to_string(),
                input: input.clone(),
            });
        }

        // ── 2. Resolve the operator (lock released before any await) ───────
        let op = {
            let agents = self.agents.lock().expect("agents mutex poisoned");
            agents.get(activity_id).cloned()
        };
        let op = op.ok_or_else(|| TemporalError::WorkflowNotFound(activity_id.to_string()))?;

        // ── 3. Deserialise input ────────────────────────────────────────────
        let operator_input: OperatorInput = serde_json::from_slice(&input)
            .map_err(|e| TemporalError::Serialization(e.to_string()))?;

        // ── 4. Execute (async; all locks released) ─────────────────────────
        let output = op
            .execute(operator_input)
            .await
            .map_err(|e| TemporalError::Other(e.to_string()))?;

        // ── 5. Serialise output ────────────────────────────────────────────
        serde_json::to_vec(&output).map_err(|e| TemporalError::Serialization(e.to_string()))
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::operator::{OperatorInput, TriggerType};
    use layer0::test_utils::EchoOperator;

    fn make_input(msg: &str) -> Vec<u8> {
        let input = OperatorInput::new(Content::text(msg), TriggerType::User);
        serde_json::to_vec(&input).expect("serialise input")
    }

    #[tokio::test]
    async fn mock_client_records_activity_calls() {
        let agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        agents
            .lock()
            .unwrap()
            .insert("echo".to_string(), Arc::new(EchoOperator) as Arc<dyn Operator>);

        let client = MockTemporalClient::new(Arc::clone(&agents));
        assert_eq!(client.call_count(), 0, "no calls yet");

        client.execute_activity("echo", make_input("one")).await.unwrap();
        assert_eq!(client.call_count(), 1);

        client.execute_activity("echo", make_input("two")).await.unwrap();
        client.execute_activity("echo", make_input("three")).await.unwrap();
        assert_eq!(client.call_count(), 3, "three calls recorded");
    }

    #[tokio::test]
    async fn mock_client_unknown_agent_returns_workflow_not_found() {
        let agents: Arc<Mutex<HashMap<String, Arc<dyn Operator>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let client = MockTemporalClient::new(Arc::clone(&agents));

        let err = client
            .execute_activity("ghost", make_input("x"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, TemporalError::WorkflowNotFound(_)),
            "expected WorkflowNotFound, got {err}"
        );
        // Call is still recorded even on failure
        assert_eq!(client.call_count(), 1);
    }
}
