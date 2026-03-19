//! A2A protocol server wrapping skelegent dispatch.

mod handlers;
mod stream;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::routing::{get, post};
use axum::Extension;
use layer0::dispatch::Dispatcher;
use layer0::id::OperatorId;
use skg_a2a_core::push::PushNotificationStore;
use skg_a2a_core::{AgentCard, TaskState};
/// In-memory registry of active dispatches for A2A task operations.
///
/// Maps task_id → cancel token + latest state. Entries are cleaned up
/// when terminal events arrive via the streaming pump.
pub(crate) struct DispatchRegistry {
    active: Mutex<HashMap<String, ActiveDispatch>>,
}

struct ActiveDispatch {
    cancel: tokio::sync::watch::Sender<bool>,
    state: TaskState,
    /// Broadcast channel for multicasting SSE events to subscribers.
    events_tx: tokio::sync::broadcast::Sender<serde_json::Value>,
}

impl DispatchRegistry {
    fn new() -> Self {
        Self {
            active: Mutex::new(HashMap::new()),
        }
    }

    /// Register a dispatch's cancel token with initial `Submitted` state.
    pub fn register(&self, task_id: String, cancel_tx: tokio::sync::watch::Sender<bool>) {
        let (events_tx, _) = tokio::sync::broadcast::channel(64);
        self.active.lock().unwrap().insert(
            task_id,
            ActiveDispatch {
                cancel: cancel_tx,
                state: TaskState::Submitted,
                events_tx,
            },
        );
    }

    /// Update the state of a tracked dispatch.
    pub fn update_state(&self, task_id: &str, state: TaskState) {
        if let Some(entry) = self.active.lock().unwrap().get_mut(task_id) {
            entry.state = state;
        }
    }

    /// Get current state. Returns `None` if not tracked.
    pub fn get_state(&self, task_id: &str) -> Option<TaskState> {
        self.active.lock().unwrap().get(task_id).map(|e| e.state)
    }

    /// Cancel a dispatch. Returns `false` if not found.
    pub fn cancel(&self, task_id: &str) -> bool {
        if let Some(entry) = self.active.lock().unwrap().get(task_id) {
            entry.cancel.send(true).ok();
            true
        } else {
            false
        }
    }

    /// Remove a completed/failed/cancelled dispatch entry.
    pub fn remove(&self, task_id: &str) {
        self.active.lock().unwrap().remove(task_id);
    }

    /// Broadcast an event value to all subscribers of a dispatch.
    pub fn broadcast(&self, task_id: &str, value: serde_json::Value) {
        if let Some(entry) = self.active.lock().unwrap().get(task_id) {
            // Ignore send errors — means no receivers are listening.
            let _ = entry.events_tx.send(value);
        }
    }

    /// Subscribe to events for an existing dispatch.
    /// Returns None if the task is not found.
    pub fn subscribe(&self, task_id: &str) -> Option<tokio::sync::broadcast::Receiver<serde_json::Value>> {
        self.active.lock().unwrap().get(task_id).map(|e| e.events_tx.subscribe())
    }

    /// List all active task IDs with their current states.
    pub fn list(&self) -> Vec<(String, TaskState)> {
        self.active.lock().unwrap().iter().map(|(id, e)| (id.clone(), e.state)).collect()
    }
}

/// Shared state available to all A2A request handlers.
pub(crate) struct A2aServerState {
    pub dispatcher: Arc<dyn Dispatcher>,
    pub default_operator: OperatorId,
    #[allow(dead_code)] // Retained for agent card endpoint access.
    pub card: AgentCard,
    pub registry: DispatchRegistry,
    pub push_store: Option<Arc<dyn PushNotificationStore>>,
    pub auth_validator: Option<Arc<dyn skg_hook_security::auth::TokenValidator>>,
}

/// A2A protocol server wrapping skelegent dispatch.
///
/// Call [`into_router`](Self::into_router) to produce an [`axum::Router`].
pub struct A2aServer {
    dispatcher: Arc<dyn Dispatcher>,
    default_operator: OperatorId,
    card: AgentCard,
    push_store: Option<Arc<dyn PushNotificationStore>>,
    auth_validator: Option<Arc<dyn skg_hook_security::auth::TokenValidator>>,
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
            card,
            push_store: None,
            auth_validator: None,
        }
    }

    /// Enable push notification support with the given store.
    pub fn with_push(mut self, store: Arc<dyn PushNotificationStore>) -> Self {
        self.push_store = Some(store);
        self
    }

    /// Enable token-based authentication for JSON-RPC endpoints.
    ///
    /// When configured, all JSON-RPC requests must carry a valid
    /// `Authorization: Bearer <token>` header. The `/.well-known/agent.json`
    /// discovery endpoint remains public.
    pub fn with_auth(mut self, validator: Arc<dyn skg_hook_security::auth::TokenValidator>) -> Self {
        self.auth_validator = Some(validator);
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
            card: self.card,
            registry: DispatchRegistry::new(),
            push_store: self.push_store,
            auth_validator: self.auth_validator,
        });

        axum::Router::new()
            .route(
                "/.well-known/agent.json",
                get({
                    move || async move {
                        (
                            [(axum::http::header::CACHE_CONTROL, "max-age=3600, public")],
                            axum::Json(card),
                        )
                    }
                }),
            )
            .route("/", post(handlers::handle_jsonrpc))
            .route("/{tenant}", post(handlers::handle_jsonrpc))
            .layer(Extension(state))
    }
}
