//! JSON-RPC request dispatch for the A2A server.

use std::sync::Arc;

use axum::response::IntoResponse;
use axum::{Extension, Json};
use skg_a2a_core::jsonrpc::methods;
use skg_a2a_core::types::{
    CancelTaskRequest, GetTaskRequest, SendMessageRequest, SendMessageResponse, TaskStatus,
};
use skg_a2a_core::{
    A2aError, A2aTask, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse,
};
use skg_a2a_core::convert::{
    a2a_message_to_operator_input, operator_output_to_a2a_message,
    run_status_to_task_state,
};
use skg_run_core::RunId;

use super::stream::stream_run_updates;
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
    let send_req: SendMessageRequest = match serde_json::from_value(request.params.clone()) {
        Ok(r) => r,
        Err(e) => return jsonrpc_error_response(id, &A2aError::ParseError { reason: e.to_string() }),
    };

    // If we have run starter + observer, use the streaming path.
    let (starter, observer) = match (&state.run_starter, &state.run_observer) {
        (Some(s), Some(o)) => (s.clone(), o.clone()),
        _ => {
            // Fall back to synchronous dispatch.
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
            return jsonrpc_success_response(id, &resp);
        }
    };

    let run_id = match starter.start_run(request.params).await {
        Ok(rid) => rid,
        Err(e) => {
            return jsonrpc_error_response(
                id,
                &A2aError::Internal {
                    reason: e.to_string(),
                },
            );
        }
    };

    let subscription = match observer.subscribe(&run_id).await {
        Ok(s) => s,
        Err(e) => {
            return jsonrpc_error_response(
                id,
                &A2aError::Internal {
                    reason: e.to_string(),
                },
            );
        }
    };

    let task_id = run_id.as_str().to_owned();
    let context_id = send_req
        .message
        .context_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    stream_run_updates(subscription, task_id, context_id)
        .into_response()
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

    let controller = match &state.run_controller {
        Some(c) => c,
        None => {
            return jsonrpc_error_response(
                id,
                &A2aError::UnsupportedOperation {
                    operation: "tasks/get".into(),
                },
            );
        }
    };

    let run_id = RunId::new(&get_req.id);
    let view = match controller.get_run(&run_id).await {
        Ok(v) => v,
        Err(_) => {
            return jsonrpc_error_response(
                id,
                &A2aError::TaskNotFound {
                    task_id: get_req.id,
                },
            );
        }
    };

    let task_state = run_status_to_task_state(view.status());
    let mut task = A2aTask::new(TaskStatus::new(task_state));
    task.id = run_id.as_str().to_owned();

    jsonrpc_success_response(id, &task)
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

    let controller = match &state.run_controller {
        Some(c) => c,
        None => {
            return jsonrpc_error_response(
                id,
                &A2aError::UnsupportedOperation {
                    operation: "tasks/cancel".into(),
                },
            );
        }
    };

    let run_id = RunId::new(&cancel_req.id);
    if controller.cancel_run(&run_id).await.is_err() {
        return jsonrpc_error_response(
            id,
            &A2aError::TaskNotCancelable {
                task_id: cancel_req.id,
            },
        );
    }

    // Return the task in its current state after cancellation.
    let view = match controller.get_run(&run_id).await {
        Ok(v) => v,
        Err(_) => {
            // Successfully cancelled but can't fetch — return minimal cancelled task.
            let mut task = A2aTask::new(TaskStatus::new(
                skg_a2a_core::TaskState::Canceled,
            ));
            task.id = cancel_req.id;
            return jsonrpc_success_response(id, &task);
        }
    };

    let task_state = run_status_to_task_state(view.status());
    let mut task = A2aTask::new(TaskStatus::new(task_state));
    task.id = run_id.as_str().to_owned();

    jsonrpc_success_response(id, &task)
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
