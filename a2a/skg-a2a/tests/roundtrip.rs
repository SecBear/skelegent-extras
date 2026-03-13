//! A2A server ↔ client round-trip integration tests.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle, Dispatcher};
use layer0::error::OrchError;
use layer0::id::{DispatchId, OperatorId};
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use skg_a2a::client::A2aDispatcher;
use skg_a2a::server::A2aServer;
use skg_a2a_core::{AgentCard, AgentSkill};
use std::net::SocketAddr;
use std::sync::Arc;

/// Mock operator that echoes the input text back with a prefix.
struct EchoDispatcher;

#[async_trait]
impl Dispatcher for EchoDispatcher {
    async fn dispatch(
        &self,
        _operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let text = input.message.as_text().unwrap_or("(empty)").to_owned();
        let response = Content::text(format!("echo: {text}"));
        let output = OperatorOutput::new(response, ExitReason::Complete);
        let (handle, sender) = DispatchHandle::channel(DispatchId::new("test"));
        tokio::spawn(async move {
            let _ = sender.send(DispatchEvent::Completed { output }).await;
        });
        Ok(handle)
    }
}

fn test_card(port: u16) -> AgentCard {
    AgentCard::builder("test-echo", "Echo agent for testing")
        .version("0.1.0")
        .interface(
            format!("http://127.0.0.1:{port}"),
            "JSONRPC",
            "0.3",
        )
        .skill(AgentSkill::new("echo", "Echo", "Echoes input back"))
        .input_mode("text/plain")
        .output_mode("text/plain")
        .build()
}

async fn start_server(dispatcher: Arc<dyn Dispatcher>) -> (SocketAddr, AgentCard) {
    // Bind to random port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");
    let addr = listener.local_addr().expect("local_addr failed");
    let card = test_card(addr.port());

    let router = A2aServer::new(
        dispatcher,
        OperatorId::new("echo"),
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

#[tokio::test]
async fn a2a_round_trip_echo() {
    let dispatcher: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher);
    let (_addr, card) = start_server(dispatcher).await;

    // Create client from card
    let client = A2aDispatcher::new(card);

    // Dispatch through A2A
    let input = OperatorInput::new(
        Content::text("hello world"),
        layer0::operator::TriggerType::User,
    );
    let output = client
        .dispatch(&OperatorId::new("ignored"), input)
        .await
        .expect("dispatch failed")
        .collect()
        .await
        .expect("collect failed");

    assert_eq!(
        output.message.as_text().unwrap(),
        "echo: hello world"
    );
    assert_eq!(output.exit_reason, ExitReason::Complete);
}

#[tokio::test]
async fn a2a_agent_card_discovery() {
    let dispatcher: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher);
    let (addr, _card) = start_server(dispatcher).await;

    // Discover via well-known URL
    let url = format!("http://127.0.0.1:{}", addr.port());
    let client = A2aDispatcher::discover(&url).await.expect("discover failed");

    // Verify dispatch works through discovered client
    let input = OperatorInput::new(
        Content::text("discovery test"),
        layer0::operator::TriggerType::User,
    );
    let output = client
        .dispatch(&OperatorId::new("any"), input)
        .await
        .expect("dispatch failed")
        .collect()
        .await
        .expect("collect failed");

    assert_eq!(
        output.message.as_text().unwrap(),
        "echo: discovery test"
    );
}

#[tokio::test]
async fn a2a_multiblock_content() {
    let dispatcher: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher);
    let (_addr, card) = start_server(dispatcher).await;

    let client = A2aDispatcher::new(card);

    // Send blocks content — server should see the text
    let input = OperatorInput::new(
        Content::Blocks(vec![
            layer0::content::ContentBlock::Text {
                text: "block one".into(),
            },
            layer0::content::ContentBlock::Text {
                text: "block two".into(),
            },
        ]),
        layer0::operator::TriggerType::User,
    );

    let output = client
        .dispatch(&OperatorId::new("echo"), input)
        .await
        .expect("dispatch failed")
        .collect()
        .await
        .expect("collect failed");

    // EchoDispatcher uses as_text() which returns first text block
    assert_eq!(
        output.message.as_text().unwrap(),
        "echo: block one"
    );
}
