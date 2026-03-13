//! JSON-RPC request dispatch for the A2A server.

use std::sync::Arc;

use axum::response::IntoResponse;
use axum::{Extension, Json};
use skg_a2a_core::convert::{a2a_message_to_operator_input, operator_output_to_a2a_message};
use skg_a2a_core::jsonrpc::methods;
use skg_a2a_core::types::{
    CancelTaskRequest, GetTaskRequest, ListTasksRequest, ListTasksResponse, SendMessageRequest,
    SendMessageResponse, SubscribeToTaskRequest, TaskStatus,
};
use skg_a2a_core::{
    A2aError, A2aTask, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse, TaskState,
};

use super::stream::{stream_dispatch, stream_subscription};
use super::A2aServerState;

/// Handle a JSON-RPC request and dispatch to the appropriate method.
pub(crate) async fn handle_jsonrpc(
    Extension(state): Extension<Arc<A2aServerState>>,
    Json(request): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    let id = request.id.clone();
    match request.method.as_str() {
        methods::SEND_MESSAGE => handle_send_message(state, request).await,
        methods::SEND_STREAMING_MESSAGE => handle_stream_message(state, request).await,
        methods::GET_TASK => handle_get_task(state, request).await,
        methods::CANCEL_TASK => handle_cancel_task(state, request).await,
        methods::LIST_TASKS => handle_list_tasks(state, request).await,
        methods::SUBSCRIBE_TO_TASK => handle_subscribe_task(state, request).await,
        methods::GET_EXTENDED_AGENT_CARD => handle_extended_agent_card(state, request).await,
        _ => {
            let err = A2aError::MethodNotFound {
                method: request.method,
            };
            jsonrpc_error_response(id, &err)
        }
    }
}

async fn handle_send_message(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    let id = request.id.clone();
    let send_req: SendMessageRequest = match serde_json::from_value(request.params) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    let input = a2a_message_to_operator_input(&send_req.message);
    let output = match state.dispatcher.dispatch(&state.default_operator, input).await {
        Ok(handle) => match handle.collect().await {
            Ok(o) => o,
            Err(e) => {
                return jsonrpc_error_response(
                    id,
                    &A2aError::Internal {
                        reason: e.to_string(),
                    },
                );
            }
        },
        Err(e) => {
            return jsonrpc_error_response(
                id,
                &A2aError::Internal {
                    reason: e.to_string(),
                },
            );
        }
    };

    let reply = operator_output_to_a2a_message(&output);
    let resp = SendMessageResponse::Message { message: reply };
    jsonrpc_success_response(id, &resp)
}

async fn handle_stream_message(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    let id = request.id.clone();
    let send_req: SendMessageRequest = match serde_json::from_value(request.params) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    let input = a2a_message_to_operator_input(&send_req.message);
    let handle = match state.dispatcher.dispatch(&state.default_operator, input).await {
        Ok(h) => h,
        Err(e) => {
            return jsonrpc_error_response(
                id,
                &A2aError::Internal {
                    reason: e.to_string(),
                },
            );
        }
    };

    let task_id = handle.id.to_string();
    let context_id = send_req
        .message
        .context_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Create a cancel channel for the registry; the streaming pump
    // watches this and calls handle.cancel() when triggered.
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    state.registry.register(task_id.clone(), cancel_tx);

    stream_dispatch(handle, task_id, context_id, state, cancel_rx).into_response()
}

async fn handle_get_task(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    let id = request.id.clone();
    let get_req: GetTaskRequest = match serde_json::from_value(request.params) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    match state.registry.get_state(&get_req.id) {
        Some(task_state) => {
            let mut task = A2aTask::new(TaskStatus::new(task_state));
            task.id = get_req.id;
            jsonrpc_success_response(id, &task)
        }
        None => jsonrpc_error_response(
            id,
            &A2aError::TaskNotFound {
                task_id: get_req.id,
            },
        ),
    }
}

async fn handle_cancel_task(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    let id = request.id.clone();
    let cancel_req: CancelTaskRequest = match serde_json::from_value(request.params) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    if state.registry.cancel(&cancel_req.id) {
        let mut task = A2aTask::new(TaskStatus::new(TaskState::Canceled));
        task.id = cancel_req.id;
        jsonrpc_success_response(id, &task)
    } else {
        jsonrpc_error_response(
            id,
            &A2aError::TaskNotCancelable {
                task_id: cancel_req.id,
            },
        )
    }
}

/// Build a JSON-RPC success response.
fn jsonrpc_success_response(
    id: Option<serde_json::Value>,
    result: &impl serde::Serialize,
) -> axum::response::Response {
    let val = serde_json::to_value(result).expect("serialization should not fail");
    let resp = JsonRpcResponse::success(id, val);
    Json(serde_json::to_value(resp).expect("serialization should not fail")).into_response()
}

/// Build a JSON-RPC error response from an [`A2aError`].
fn jsonrpc_error_response(
    id: Option<serde_json::Value>,
    err: &A2aError,
) -> axum::response::Response {
    let resp = JsonRpcErrorResponse::error(id, err.code(), err.to_string());
    Json(serde_json::to_value(resp).expect("serialization should not fail")).into_response()
}

async fn handle_list_tasks(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    let id = request.id.clone();
    let _list_req: ListTasksRequest = match serde_json::from_value(request.params) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    let active = state.registry.list();
    let tasks: Vec<A2aTask> = active
        .into_iter()
        .map(|(task_id, task_state)| {
            let mut task = A2aTask::new(TaskStatus::new(task_state));
            task.id = task_id;
            task
        })
        .collect();

    let resp = ListTasksResponse::new(tasks);
    jsonrpc_success_response(id, &resp)
}

async fn handle_subscribe_task(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    let id = request.id.clone();
    let sub_req: SubscribeToTaskRequest = match serde_json::from_value(request.params) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    match state.registry.subscribe(&sub_req.id) {
        Some(rx) => stream_subscription(rx, sub_req.id, state).into_response(),
        None => jsonrpc_error_response(
            id,
            &A2aError::TaskNotFound {
                task_id: sub_req.id,
            },
        ),
    }
}

async fn handle_extended_agent_card(
    state: Arc<A2aServerState>,
    request: JsonRpcRequest,
) -> axum::response::Response {
    jsonrpc_success_response(request.id, &state.card)
}
