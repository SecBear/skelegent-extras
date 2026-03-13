//! SSE streaming from [`RunSubscription`].

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use skg_a2a_core::convert::{run_artifact_to_a2a_artifact, run_status_to_task_state};
use skg_run_core::{RunSubscription, RunUpdate};
use tokio_stream::wrappers::ReceiverStream;

/// Convert a [`RunSubscription`] into an SSE stream of A2A [`StreamResponse`] events.
///
/// Spawns a background task that reads from the subscription and forwards
/// events as SSE. The stream ends when the subscription closes (run terminal).
pub(crate) fn stream_run_updates(
    subscription: RunSubscription,
    task_id: String,
    context_id: String,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(pump_updates(subscription, task_id, context_id, tx));

    Sse::new(ReceiverStream::new(rx))
}

async fn pump_updates(
    mut subscription: RunSubscription,
    task_id: String,
    context_id: String,
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    while let Some(update) = subscription.recv().await {
        let event = match update {
            RunUpdate::StatusChanged { status, .. } => {
                let a2a_state = run_status_to_task_state(status);
                let evt_value = serde_json::json!({
                    "type": "status_update",
                    "task_id": task_id,
                    "context_id": context_id,
                    "status": {
                        "state": a2a_state,
                    },
                });
                match Event::default().json_data(evt_value) {
                    Ok(e) => e,
                    Err(_) => continue,
                }
            }
            RunUpdate::ArtifactProduced { artifact, .. } => {
                let a2a_artifact = run_artifact_to_a2a_artifact(&artifact);
                let evt_value = serde_json::json!({
                    "type": "artifact_update",
                    "task_id": task_id,
                    "context_id": context_id,
                    "artifact": a2a_artifact,
                });
                match Event::default().json_data(evt_value) {
                    Ok(e) => e,
                    Err(_) => continue,
                }
            }
            _ => continue,
        };

        if tx.send(Ok(event)).await.is_err() {
            break;
        }
    }
}
