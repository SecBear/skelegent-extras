//! Route-based orchestration — dispatch to backend orchestrators by agent name.
//!
//! [`RoutingOrchestrator`] consults a routing table on each dispatch. When the
//! agent name matches a registered route, the dispatch is forwarded to that
//! backend. Unmatched names fall through to the fallback orchestrator.
//!
//! Signal and query operations always delegate to the fallback; routing is
//! applied at the dispatch boundary only.

use async_trait::async_trait;
use layer0::{
    effect::SignalPayload, AgentId, OperatorInput, OperatorOutput, OrchError, Orchestrator,
    QueryPayload, WorkflowId,
};
use std::collections::HashMap;
use std::sync::Arc;

/// An orchestrator that routes dispatches to backend orchestrators by agent name.
///
/// Built with [`RoutingOrchestrator::new`] and configured with the builder
/// method [`route`](RoutingOrchestrator::route).
///
/// ```rust,ignore
/// let router = RoutingOrchestrator::new(fallback)
///     .route("planner", planner_orch)
///     .route("executor", executor_orch);
/// ```
///
/// Registering the same name twice replaces the earlier entry.
pub struct RoutingOrchestrator {
    routes: HashMap<String, Arc<dyn Orchestrator>>,
    fallback: Arc<dyn Orchestrator>,
}

impl std::fmt::Debug for RoutingOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingOrchestrator")
            .field("route_count", &self.routes.len())
            .finish_non_exhaustive()
    }
}

impl RoutingOrchestrator {
    /// Create a new routing orchestrator. All unmatched agent names dispatch to `fallback`.
    pub fn new(fallback: Arc<dyn Orchestrator>) -> Self {
        Self {
            routes: HashMap::new(),
            fallback,
        }
    }

    /// Register a route. Dispatches whose agent name matches `name` are forwarded to `orch`.
    ///
    /// Returns `self` for chaining. If `name` is already registered, the new entry replaces it.
    pub fn route(mut self, name: impl Into<String>, orch: Arc<dyn Orchestrator>) -> Self {
        self.routes.insert(name.into(), orch);
        self
    }

    fn resolve(&self, agent: &AgentId) -> &dyn Orchestrator {
        self.routes
            .get(agent.as_str())
            .map(Arc::as_ref)
            .unwrap_or(Arc::as_ref(&self.fallback))
    }
}

#[async_trait]
impl Orchestrator for RoutingOrchestrator {
    async fn dispatch(
        &self,
        agent: &AgentId,
        input: OperatorInput,
    ) -> Result<OperatorOutput, OrchError> {
        self.resolve(agent).dispatch(agent, input).await
    }

    async fn dispatch_many(
        &self,
        tasks: Vec<(AgentId, OperatorInput)>,
    ) -> Vec<Result<OperatorOutput, OrchError>> {
        let mut results = Vec::with_capacity(tasks.len());
        for (agent, input) in tasks {
            results.push(self.resolve(&agent).dispatch(&agent, input).await);
        }
        results
    }

    async fn signal(
        &self,
        target: &WorkflowId,
        signal: SignalPayload,
    ) -> Result<(), OrchError> {
        self.fallback.signal(target, signal).await
    }

    async fn query(
        &self,
        target: &WorkflowId,
        query: QueryPayload,
    ) -> Result<serde_json::Value, OrchError> {
        self.fallback.query(target, query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{operator::TriggerType, Content, ExitReason};
    use std::sync::Mutex;

    /// Mock orchestrator that records the agent name for every dispatch it receives.
    struct RecordingOrchestrator {
        dispatched: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingOrchestrator {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let log = Arc::new(Mutex::new(Vec::new()));
            (Self { dispatched: Arc::clone(&log) }, log)
        }
    }

    #[async_trait]
    impl Orchestrator for RecordingOrchestrator {
        async fn dispatch(
            &self,
            agent: &AgentId,
            _input: OperatorInput,
        ) -> Result<OperatorOutput, OrchError> {
            self.dispatched.lock().unwrap().push(agent.as_str().to_owned());
            Ok(OperatorOutput::new(Content::text("ok"), ExitReason::Complete))
        }

        async fn dispatch_many(
            &self,
            tasks: Vec<(AgentId, OperatorInput)>,
        ) -> Vec<Result<OperatorOutput, OrchError>> {
            let mut results = Vec::new();
            for (agent, input) in tasks {
                results.push(self.dispatch(&agent, input).await);
            }
            results
        }

        async fn signal(
            &self,
            _target: &WorkflowId,
            _signal: SignalPayload,
        ) -> Result<(), OrchError> {
            Ok(())
        }

        async fn query(
            &self,
            _target: &WorkflowId,
            _query: QueryPayload,
        ) -> Result<serde_json::Value, OrchError> {
            Ok(serde_json::Value::Null)
        }
    }

    fn make_input() -> OperatorInput {
        OperatorInput::new(Content::text("test"), TriggerType::Task)
    }

    #[tokio::test]
    async fn routes_to_registered_backend() {
        let (mock_a, log_a) = RecordingOrchestrator::new();
        let (mock_b, log_b) = RecordingOrchestrator::new();

        let router = RoutingOrchestrator::new(Arc::new(mock_b))
            .route("alpha", Arc::new(mock_a));

        router.dispatch(&AgentId::new("alpha"), make_input()).await.unwrap();
        router.dispatch(&AgentId::new("beta"), make_input()).await.unwrap();

        assert_eq!(*log_a.lock().unwrap(), vec!["alpha"]);
        assert_eq!(*log_b.lock().unwrap(), vec!["beta"]);
    }

    #[tokio::test]
    async fn dispatch_many_routes_each() {
        let (mock_a, log_a) = RecordingOrchestrator::new();
        let (mock_b, log_b) = RecordingOrchestrator::new();

        let router = RoutingOrchestrator::new(Arc::new(mock_b))
            .route("alpha", Arc::new(mock_a));

        let tasks = vec![
            (AgentId::new("alpha"), make_input()),
            (AgentId::new("beta"), make_input()),
            (AgentId::new("alpha"), make_input()),
        ];

        let results = router.dispatch_many(tasks).await;
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));

        assert_eq!(*log_a.lock().unwrap(), vec!["alpha", "alpha"]);
        assert_eq!(*log_b.lock().unwrap(), vec!["beta"]);
    }

    #[tokio::test]
    async fn fallback_handles_unknown() {
        let (mock_a, log_a) = RecordingOrchestrator::new();
        let (mock_b, log_b) = RecordingOrchestrator::new();

        let router = RoutingOrchestrator::new(Arc::new(mock_b))
            .route("alpha", Arc::new(mock_a));

        router.dispatch(&AgentId::new("gamma"), make_input()).await.unwrap();

        assert!(log_a.lock().unwrap().is_empty(), "route for alpha must not fire");
        assert_eq!(*log_b.lock().unwrap(), vec!["gamma"]);
    }
}
