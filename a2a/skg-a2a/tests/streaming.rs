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
