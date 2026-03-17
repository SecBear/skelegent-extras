//! Streaming SSE client integration tests.
//!
//! Starts an in-process A2A server backed by a mock dispatcher that emits
//! progress events before completing, then verifies that the streaming client
//! receives events in the expected order.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle, Dispatcher};
use layer0::error::OrchError;
use layer0::id::{DispatchId, OperatorId};
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use layer0::DispatchContext;
use skg_a2a::client::dispatch_streaming;
use skg_a2a::server::A2aServer;
use skg_a2a_core::{AgentCard, AgentSkill};
use std::net::SocketAddr;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Mock dispatcher: emits progress then completes
// ---------------------------------------------------------------------------

/// Dispatcher that emits two progress events then sends a final completion.
struct ProgressDispatcher;

#[async_trait]
impl Dispatcher for ProgressDispatcher {
    async fn dispatch(
        &self,
        _ctx: &DispatchContext,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let text = input.message.as_text().unwrap_or("(empty)").to_owned();
        let (handle, sender) = DispatchHandle::channel(DispatchId::new("stream-test"));
        tokio::spawn(async move {
            let _ = sender
                .send(DispatchEvent::Progress {
                    content: Content::text("step 1"),
                })
                .await;
            let _ = sender
                .send(DispatchEvent::Progress {
                    content: Content::text("step 2"),
                })
                .await;
            let _ = sender
                .send(DispatchEvent::Completed {
                    output: OperatorOutput::new(
                        Content::text(format!("done: {text}")),
                        ExitReason::Complete,
                    ),
                })
                .await;
        });
        Ok(handle)
    }
}

// ---------------------------------------------------------------------------
// Server helper
// ---------------------------------------------------------------------------

fn streaming_card(port: u16) -> AgentCard {
    AgentCard::builder("stream-test", "Streaming test agent")
        .version("0.1.0")
        .interface(
            format!("http://127.0.0.1:{port}"),
            "JSONRPC",
            "0.3",
        )
        .skill(AgentSkill::new("progress", "Progress", "Emits progress events"))
        .streaming(true)
        .input_mode("text/plain")
        .output_mode("text/plain")
        .build()
}

async fn start_streaming_server(dispatcher: Arc<dyn Dispatcher>) -> (SocketAddr, AgentCard) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");
    let addr = listener.local_addr().expect("local_addr failed");
    let card = streaming_card(addr.port());

    let router = A2aServer::new(
        dispatcher,
        OperatorId::new("stream-test"),
        card.clone(),
    )
    .into_router();

    tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .expect("server error");
    });

    (addr, card)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Human-readable label for a `DispatchEvent` variant (since it doesn't
/// implement `Debug`).
fn dispatch_event_label(event: &DispatchEvent) -> &'static str {
    match event {
        DispatchEvent::Progress { .. } => "Progress",
        DispatchEvent::ArtifactProduced { .. } => "ArtifactProduced",
        DispatchEvent::EffectEmitted { .. } => "EffectEmitted",
        DispatchEvent::Completed { .. } => "Completed",
        DispatchEvent::Failed { .. } => "Failed",
        _ => "Unknown",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that the streaming client receives progress events in order followed
/// by the terminal completion event.
#[tokio::test]
async fn streaming_round_trip() {
    let dispatcher: Arc<dyn Dispatcher> = Arc::new(ProgressDispatcher);
    let (addr, _card) = start_streaming_server(dispatcher).await;

    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", addr.port());
    let input = OperatorInput::new(
        Content::text("hello streaming"),
        layer0::operator::TriggerType::User,
    );

    let mut handle = dispatch_streaming(&http, &url, input)
        .await
        .expect("dispatch_streaming failed");

    // Collect all events.
    let mut events: Vec<DispatchEvent> = Vec::new();
    while let Some(event) = handle.recv().await {
        let terminal = matches!(
            event,
            DispatchEvent::Completed { .. } | DispatchEvent::Failed { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    // Should have: Progress("step 1"), Progress("step 2"), Completed("done: hello streaming")
    assert_eq!(events.len(), 3, "expected 3 events, got {}", events.len());

    match &events[0] {
        DispatchEvent::Progress { content } => {
            assert_eq!(content.as_text().unwrap(), "step 1");
        }
        _ => panic!("event[0]: expected Progress"),
    }

    match &events[1] {
        DispatchEvent::Progress { content } => {
            assert_eq!(content.as_text().unwrap(), "step 2");
        }
        _ => panic!("event[1]: expected Progress"),
    }

    match &events[2] {
        DispatchEvent::Completed { output } => {
            assert_eq!(output.message.as_text().unwrap(), "done: hello streaming");
            assert_eq!(output.exit_reason, ExitReason::Complete);
        }
        _ => panic!("event[2]: expected Completed"),
    }
}

/// Verify that a dispatcher-emitted failure flows through the server as
/// `TaskState::Failed` SSE and the streaming client maps it to
/// `DispatchEvent::Failed`.
#[tokio::test]
async fn streaming_failed_round_trip() {
    /// Dispatcher that sends a single `Failed` event.
    struct FailedDispatcher;

    #[async_trait]
    impl Dispatcher for FailedDispatcher {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("fail-test"));
            tokio::spawn(async move {
                let _ = sender
                    .send(DispatchEvent::Failed {
                        error: OrchError::DispatchFailed("something went wrong".into()),
                    })
                    .await;
            });
            Ok(handle)
        }
    }

    let dispatcher: Arc<dyn Dispatcher> = Arc::new(FailedDispatcher);
    let (addr, _card) = start_streaming_server(dispatcher).await;

    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", addr.port());
    let input = OperatorInput::new(
        Content::text("trigger failure"),
        layer0::operator::TriggerType::User,
    );

    let mut handle = dispatch_streaming(&http, &url, input)
        .await
        .expect("dispatch_streaming failed");

    // Collect events until terminal.
    let mut events: Vec<DispatchEvent> = Vec::new();
    while let Some(event) = handle.recv().await {
        let terminal = matches!(
            event,
            DispatchEvent::Completed { .. } | DispatchEvent::Failed { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert_eq!(events.len(), 1, "expected 1 event, got {}", events.len());
    match &events[0] {
        DispatchEvent::Failed { error } => {
            let msg = error.to_string();
            assert!(
                msg.contains("something went wrong"),
                "expected error to contain 'something went wrong', got: {msg}"
            );
        }
        other => panic!("event[0]: expected Failed, got {}", dispatch_event_label(other)),
    }
}

/// Verify that canceling a task via the server's cancel endpoint produces a
/// `TaskState::Canceled` SSE event that the streaming client maps to
/// `DispatchEvent::Failed`.
#[tokio::test]
async fn streaming_canceled_round_trip() {
    /// Dispatcher that blocks until its channel is closed (simulates a
    /// long-running task that can be canceled).
    struct SlowDispatcher;

    #[async_trait]
    impl Dispatcher for SlowDispatcher {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("cancel-test"));
            tokio::spawn(async move {
                // Send one progress event so we know the stream is active,
                // then block until the channel is dropped.
                let _ = sender
                    .send(DispatchEvent::Progress {
                        content: Content::text("working"),
                    })
                    .await;
                // Hold the sender open — the server cancel mechanism will
                // break the SSE pump independently.
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            });
            Ok(handle)
        }
    }

    let dispatcher: Arc<dyn Dispatcher> = Arc::new(SlowDispatcher);
    let (addr, _card) = start_streaming_server(dispatcher).await;

    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", addr.port());
    let input = OperatorInput::new(
        Content::text("cancel me"),
        layer0::operator::TriggerType::User,
    );

    let mut handle = dispatch_streaming(&http, &url, input)
        .await
        .expect("dispatch_streaming failed");

    // Wait for the first progress event to confirm the stream is live.
    let first = handle.recv().await.expect("expected at least one event");
    assert!(
        matches!(first, DispatchEvent::Progress { .. }),
        "expected Progress, got a non-Progress event"
    );

    // Send a cancel request to the server.  The task_id is the DispatchId
    // we set in SlowDispatcher ("cancel-test").
    let cancel_rpc = skg_a2a_core::JsonRpcRequest::new(
        "tasks/cancel",
        serde_json::json!({ "id": "cancel-test" }),
    );
    let cancel_resp = http
        .post(&url)
        .json(&cancel_rpc)
        .send()
        .await
        .expect("cancel request failed");
    assert!(
        cancel_resp.status().is_success(),
        "cancel response status: {}",
        cancel_resp.status()
    );

    // The SSE stream should now deliver a terminal Canceled event (mapped
    // to DispatchEvent::Failed by the client).
    let mut got_canceled = false;
    while let Some(event) = handle.recv().await {
        if let DispatchEvent::Failed { error } = &event {
            let msg = error.to_string();
            assert!(
                msg.contains("canceled"),
                "expected 'canceled' in error message, got: {msg}"
            );
            got_canceled = true;
            break;
        }
    }
    assert!(got_canceled, "never received Canceled terminal event");
}

/// Verify that a `TaskState::Rejected` SSE status_update is mapped to
/// `DispatchEvent::Failed` by the streaming client.
///
/// The A2A server does not currently produce Rejected from any dispatcher
/// event, so this test uses a lightweight mock HTTP server that sends a
/// hand-crafted rejected SSE response to exercise the client parse path.
#[tokio::test]
async fn streaming_rejected_round_trip() {
    use axum::body::Body;
    use axum::response::IntoResponse;
    use axum::routing::post;

    /// Build an SSE response body containing a rejected status_update.
    async fn rejected_handler() -> impl IntoResponse {
        let sse_payload = serde_json::json!({
            "type": "status_update",
            "task_id": "reject-test",
            "context_id": "ctx-1",
            "final": true,
            "status": {
                "state": "TASK_STATE_REJECTED",
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "task rejected by agent"}],
                },
            },
        });

        // Format as SSE: data: <json>\n\n
        let body = format!("data: {}\n\n", sse_payload);

        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(body))
            .unwrap()
    }

    let app = axum::Router::new().route("/", post(rejected_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");
    let addr = listener.local_addr().expect("local_addr failed");

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server error");
    });

    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", addr.port());
    let input = OperatorInput::new(
        Content::text("reject me"),
        layer0::operator::TriggerType::User,
    );

    let mut handle = dispatch_streaming(&http, &url, input)
        .await
        .expect("dispatch_streaming failed");

    // Collect events until terminal.
    let mut events: Vec<DispatchEvent> = Vec::new();
    while let Some(event) = handle.recv().await {
        let terminal = matches!(
            event,
            DispatchEvent::Completed { .. } | DispatchEvent::Failed { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert_eq!(events.len(), 1, "expected 1 event, got {}", events.len());
    match &events[0] {
        DispatchEvent::Failed { error } => {
            let msg = error.to_string();
            assert!(
                msg.contains("task rejected by agent"),
                "expected 'task rejected by agent' in error, got: {msg}"
            );
        }
        other => panic!("event[0]: expected Failed, got {}", dispatch_event_label(other)),
    }
}

/// `dispatch_auto` uses streaming when the card advertises it.
#[tokio::test]
async fn dispatch_auto_uses_streaming_when_available() {
    use skg_a2a::client::A2aDispatcher;

    let dispatcher: Arc<dyn Dispatcher> = Arc::new(ProgressDispatcher);
    let (_addr, card) = start_streaming_server(dispatcher).await;

    // Card has streaming = true.
    assert_eq!(card.capabilities.streaming, Some(true));

    let client = A2aDispatcher::new(card);
    let ctx = DispatchContext::new(
        DispatchId::new("auto-test"),
        OperatorId::new("stream-test"),
    );
    let input = OperatorInput::new(
        Content::text("auto dispatch"),
        layer0::operator::TriggerType::User,
    );

    let output = client
        .dispatch_auto(&ctx, input)
        .await
        .expect("dispatch_auto failed")
        .collect()
        .await
        .expect("collect failed");

    assert_eq!(output.message.as_text().unwrap(), "done: auto dispatch");
    assert_eq!(output.exit_reason, ExitReason::Complete);
}
