//! Route-based dispatch — dispatch to backend dispatchers by operator name.
//!
//! [`RoutingDispatcher`] consults a routing table on each dispatch. When the
//! operator name matches a registered route, the dispatch is forwarded to that
//! backend. Unmatched names fall through to the fallback dispatcher.

use async_trait::async_trait;
use layer0::{
    OperatorId, OperatorInput, OperatorOutput, OrchError,
    dispatch::Dispatcher,
};
use std::collections::HashMap;
use std::sync::Arc;

/// A dispatcher that routes dispatches to backend dispatchers by operator name.
///
/// Built with [`RoutingDispatcher::new`] and configured with the builder
/// method [`route`](RoutingDispatcher::route).
///
/// ```rust,ignore
/// let router = RoutingDispatcher::new(fallback)
///     .route("planner", planner_dispatcher)
///     .route("executor", executor_dispatcher);
/// ```
///
/// Registering the same name twice replaces the earlier entry.
pub struct RoutingDispatcher {
    routes: HashMap<String, Arc<dyn Dispatcher>>,
    fallback: Arc<dyn Dispatcher>,
}

impl std::fmt::Debug for RoutingDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingDispatcher")
            .field("route_count", &self.routes.len())
            .finish_non_exhaustive()
    }
}

impl RoutingDispatcher {
    /// Create a new routing dispatcher. All unmatched operator names dispatch to `fallback`.
    pub fn new(fallback: Arc<dyn Dispatcher>) -> Self {
        Self {
            routes: HashMap::new(),
            fallback,
        }
    }

    /// Register a route. Dispatches whose operator name matches `name` are forwarded to `dispatcher`.
    ///
    /// Returns `self` for chaining. If `name` is already registered, the new entry replaces it.
    pub fn route(mut self, name: impl Into<String>, dispatcher: Arc<dyn Dispatcher>) -> Self {
        self.routes.insert(name.into(), dispatcher);
        self
    }

    fn resolve(&self, operator: &OperatorId) -> &dyn Dispatcher {
        self.routes
            .get(operator.as_str())
            .map(Arc::as_ref)
            .unwrap_or(Arc::as_ref(&self.fallback))
    }
}

#[async_trait]
impl Dispatcher for RoutingDispatcher {
    async fn dispatch(
        &self,
        operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<OperatorOutput, OrchError> {
        self.resolve(operator).dispatch(operator, input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{operator::TriggerType, Content, ExitReason, dispatch::Dispatcher};
    use std::sync::Mutex;

    /// Mock dispatcher that records the operator name for every dispatch it receives.
    struct RecordingDispatcher {
        dispatched: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingDispatcher {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let log = Arc::new(Mutex::new(Vec::new()));
            (Self { dispatched: Arc::clone(&log) }, log)
        }
    }

    #[async_trait]
    impl Dispatcher for RecordingDispatcher {
        async fn dispatch(
            &self,
            operator: &OperatorId,
            _input: OperatorInput,
        ) -> Result<OperatorOutput, OrchError> {
            self.dispatched.lock().unwrap().push(operator.as_str().to_owned());
            Ok(OperatorOutput::new(Content::text("ok"), ExitReason::Complete))
        }
    }

    fn make_input() -> OperatorInput {
        OperatorInput::new(Content::text("test"), TriggerType::Task)
    }

    #[tokio::test]
    async fn routes_to_registered_backend() {
        let (mock_a, log_a) = RecordingDispatcher::new();
        let (mock_b, log_b) = RecordingDispatcher::new();

        let router = RoutingDispatcher::new(Arc::new(mock_b))
            .route("alpha", Arc::new(mock_a));

        router.dispatch(&OperatorId::new("alpha"), make_input()).await.unwrap();
        router.dispatch(&OperatorId::new("beta"), make_input()).await.unwrap();

        assert_eq!(*log_a.lock().unwrap(), vec!["alpha"]);
        assert_eq!(*log_b.lock().unwrap(), vec!["beta"]);
    }

    #[tokio::test]
    async fn fallback_handles_unknown() {
        let (mock_a, log_a) = RecordingDispatcher::new();
        let (mock_b, log_b) = RecordingDispatcher::new();

        let router = RoutingDispatcher::new(Arc::new(mock_b))
            .route("alpha", Arc::new(mock_a));

        router.dispatch(&OperatorId::new("gamma"), make_input()).await.unwrap();

        assert!(log_a.lock().unwrap().is_empty(), "route for alpha must not fire");
        assert_eq!(*log_b.lock().unwrap(), vec!["gamma"]);
    }
}
