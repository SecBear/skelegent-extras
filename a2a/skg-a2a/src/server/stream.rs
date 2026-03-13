//! SSE streaming from [`DispatchHandle`].

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use skg_a2a_core::convert::content_to_parts;
use skg_a2a_core::types::A2aArtifact;
use skg_a2a_core::TaskState;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;

use super::A2aServerState;

/// Convert a [`DispatchHandle`] into an SSE stream of A2A events.
///
/// Registers the dispatch in the server's [`DispatchRegistry`] and spawns a
/// background task that reads events from the handle, forwarding them as SSE.
/// The stream ends on terminal events or external cancellation via the registry.
pub(crate) fn stream_dispatch(
    handle: DispatchHandle,
    task_id: String,
    context_id: String,
    state: Arc<A2aServerState>,
    cancel_rx: watch::Receiver<bool>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(pump_dispatch_events(
        handle, task_id, context_id, tx, state, cancel_rx,
    ));
    Sse::new(ReceiverStream::new(rx))
}

async fn pump_dispatch_events(
    mut handle: DispatchHandle,
    task_id: String,
    context_id: String,
    tx: mpsc::Sender<Result<Event, Infallible>>,
    state: Arc<A2aServerState>,
    mut cancel_rx: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            event = handle.recv() => {
                let Some(event) = event else { break };
                let sse_event = match event {
                    DispatchEvent::Progress { content } => {
                        state.registry.update_state(&task_id, TaskState::Working);
                        let parts = content_to_parts(&content);
                        let evt_value = serde_json::json!({
                            "type": "status_update",
                            "task_id": task_id,
                            "context_id": context_id,
                            "status": {
                                "state": TaskState::Working,
                                "message": {
                                    "role": "agent",
                                    "parts": parts,
                                },
                            },
                        });
                        state.registry.broadcast(&task_id, evt_value.clone());
                        match Event::default().json_data(evt_value) {
                            Ok(e) => e,
                            Err(_) => continue,
                        }
                    }
                    DispatchEvent::ArtifactProduced { artifact } => {
                        let a2a_artifact = dispatch_artifact_to_a2a(&artifact);
                        let evt_value = serde_json::json!({
                            "type": "artifact_update",
                            "task_id": task_id,
                            "context_id": context_id,
                            "artifact": a2a_artifact,
                        });
                        state.registry.broadcast(&task_id, evt_value.clone());
                        match Event::default().json_data(evt_value) {
                            Ok(e) => e,
                            Err(_) => continue,
                        }
                    }
                    DispatchEvent::Completed { output } => {
                        state.registry.update_state(&task_id, TaskState::Completed);
                        let parts = content_to_parts(&output.message);
                        let evt_value = serde_json::json!({
                            "type": "status_update",
                            "task_id": task_id,
                            "context_id": context_id,
                            "final": true,
                            "status": {
                                "state": TaskState::Completed,
                                "message": {
                                    "role": "agent",
                                    "parts": parts,
                                },
                            },
                        });
                        state.registry.broadcast(&task_id, evt_value.clone());
                        let sse = match Event::default().json_data(evt_value) {
                            Ok(e) => e,
                            Err(_) => break,
                        };
                        let _ = tx.send(Ok(sse)).await;
                        break;
                    }
                    DispatchEvent::Failed { error } => {
                        state.registry.update_state(&task_id, TaskState::Failed);
                        let evt_value = serde_json::json!({
                            "type": "status_update",
                            "task_id": task_id,
                            "context_id": context_id,
                            "final": true,
                            "status": {
                                "state": TaskState::Failed,
                                "message": {
                                    "role": "agent",
                                    "parts": [{"type": "text", "text": error.to_string()}],
                                },
                            },
                        });
                        state.registry.broadcast(&task_id, evt_value.clone());
                        let sse = match Event::default().json_data(evt_value) {
                            Ok(e) => e,
                            Err(_) => break,
                        };
                        let _ = tx.send(Ok(sse)).await;
                        break;
                    }
                    _ => continue,
                };

                if tx.send(Ok(sse_event)).await.is_err() {
                    break;
                }
            }
            _ = cancel_rx.changed() => {
                handle.cancel();
                state.registry.update_state(&task_id, TaskState::Canceled);
                let evt_value = serde_json::json!({
                    "type": "status_update",
                    "task_id": task_id,
                    "context_id": context_id,
                    "final": true,
                    "status": { "state": TaskState::Canceled },
                });
                state.registry.broadcast(&task_id, evt_value.clone());
                if let Ok(sse) = Event::default().json_data(evt_value) {
                    let _ = tx.send(Ok(sse)).await;
                }
                break;
            }
        }
    }
    state.registry.remove(&task_id);
}

/// Convert a `layer0::dispatch::Artifact` to an A2A artifact.
fn dispatch_artifact_to_a2a(artifact: &layer0::dispatch::Artifact) -> A2aArtifact {
    let parts = artifact.parts.iter().flat_map(content_to_parts).collect();
    let mut a2a = A2aArtifact::new(parts);
    a2a.artifact_id = artifact.id.clone();
    a2a.name = artifact.name.clone();
    a2a.description = artifact.description.clone();
    a2a.metadata = artifact.metadata.clone();
    a2a
}


/// Stream events for an existing dispatch from a broadcast receiver.
pub(crate) fn stream_subscription(
    mut rx: tokio::sync::broadcast::Receiver<serde_json::Value>,
    task_id: String,
    state: Arc<A2aServerState>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, sse_rx) = mpsc::channel(32);
    tokio::spawn(async move {
        // Send initial status snapshot.
        if let Some(current_state) = state.registry.get_state(&task_id) {
            let evt = serde_json::json!({
                "type": "status_update",
                "task_id": task_id,
                "status": { "state": current_state },
            });
            if let Ok(sse) = Event::default().json_data(evt)
                && tx.send(Ok(sse)).await.is_err()
            {
                return;
            }
        }
        // Forward broadcast events until closed or sender dropped.
        loop {
            match rx.recv().await {
                Ok(value) => {
                    match Event::default().json_data(value) {
                        Ok(sse) => {
                            if tx.send(Ok(sse)).await.is_err() {
                                break;
                            }
                        }
                        Err(_) => continue,
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Sse::new(ReceiverStream::new(sse_rx))
}