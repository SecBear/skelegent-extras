//! A2A protocol server wrapping skelegent dispatch and run control.

mod handlers;
mod stream;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Extension;
use layer0::dispatch::Dispatcher;
use layer0::id::OperatorId;
use skg_a2a_core::AgentCard;
use skg_run_core::{RunController, RunObserver, RunStarter};

/// Shared state available to all A2A request handlers.
pub(crate) struct A2aServerState {
    pub dispatcher: Arc<dyn Dispatcher>,
    pub default_operator: OperatorId,
    pub run_starter: Option<Arc<dyn RunStarter>>,
    pub run_controller: Option<Arc<dyn RunController>>,
    pub run_observer: Option<Arc<dyn RunObserver>>,
    #[allow(dead_code)] // Retained for agent card endpoint access.
    pub card: AgentCard,
}

/// A2A protocol server wrapping skelegent dispatch and run control.
///
/// Use the builder methods to configure optional durable run support,
/// then call [`into_router`](Self::into_router) to produce an [`axum::Router`].
pub struct A2aServer {
    dispatcher: Arc<dyn Dispatcher>,
    default_operator: OperatorId,
    run_starter: Option<Arc<dyn RunStarter>>,
    run_controller: Option<Arc<dyn RunController>>,
    run_observer: Option<Arc<dyn RunObserver>>,
    card: AgentCard,
}

impl A2aServer {
    /// Create a new server with the required dispatcher and agent card.
    pub fn new(
        dispatcher: Arc<dyn Dispatcher>,
        default_operator: OperatorId,
        card: AgentCard,
    ) -> Self {
        Self {
            dispatcher,
            default_operator,
            run_starter: None,
            run_controller: None,
            run_observer: None,
            card,
        }
    }

    /// Enable durable run lifecycle (start + control).
    pub fn with_durable(
        mut self,
        starter: Arc<dyn RunStarter>,
        controller: Arc<dyn RunController>,
    ) -> Self {
        self.run_starter = Some(starter);
        self.run_controller = Some(controller);
        self
    }

    /// Enable streaming via run observation.
    pub fn with_observer(mut self, observer: Arc<dyn RunObserver>) -> Self {
        self.run_observer = Some(observer);
        self
    }

    /// Consume the builder and produce an [`axum::Router`].
    ///
    /// Routes:
    /// - `GET /.well-known/agent.json` — agent card discovery
    /// - `POST /` — JSON-RPC dispatch
    /// - `POST /:tenant` — JSON-RPC dispatch with tenant path param
    pub fn into_router(self) -> axum::Router {
        let card = self.card.clone();
        let state = Arc::new(A2aServerState {
            dispatcher: self.dispatcher,
            default_operator: self.default_operator,
            run_starter: self.run_starter,
            run_controller: self.run_controller,
            run_observer: self.run_observer,
            card: self.card,
        });

        axum::Router::new()
            .route(
                "/.well-known/agent.json",
                get({
                    move || async move { axum::Json(card) }
                }),
            )
            .route("/", post(handlers::handle_jsonrpc))
            .route("/{tenant}", post(handlers::handle_jsonrpc))
            .layer(Extension(state))
    }
}
