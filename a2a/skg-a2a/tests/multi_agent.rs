//! Multi-agent A2A integration tests.
//!
//! Verifies the end-to-end chain: client → Agent A → (A2A) → Agent B → response.
//! This proves "remote agent = local operator" — the defining DX proposition.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle, Dispatcher};
use layer0::error::OrchError;
use layer0::id::{DispatchId, OperatorId};
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use skg_a2a::client::A2aDispatcher;
use skg_a2a::server::A2aServer;
use skg_a2a_core::{AgentCard, AgentSkill, JsonRpcRequest};
use std::net::SocketAddr;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Test dispatchers
// ---------------------------------------------------------------------------

/// Echoes input with a prefix.
struct EchoDispatcher {
    prefix: &'static str,
}

impl EchoDispatcher {
    fn new(prefix: &'static str) -> Self {
        Self { prefix }
    }
}

#[async_trait]
impl Dispatcher for EchoDispatcher {
    async fn dispatch(
        &self,
        _operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let text = input.message.as_text().unwrap_or("(empty)").to_owned();
        let response = Content::text(format!("{}: {text}", self.prefix));
        let output = OperatorOutput::new(response, ExitReason::Complete);
        let (handle, sender) = DispatchHandle::channel(DispatchId::new("echo"));
        tokio::spawn(async move {
            let _ = sender.send(DispatchEvent::Completed { output }).await;
        });
        Ok(handle)
    }
}

/// Transforms input through a local function, then delegates to a remote agent.
struct DelegatingDispatcher {
    remote: Arc<A2aDispatcher>,
    tag: &'static str,
}

#[async_trait]
impl Dispatcher for DelegatingDispatcher {
    async fn dispatch(
        &self,
        _operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let text = input.message.as_text().unwrap_or("(empty)").to_owned();
        // Tag the message before forwarding
        let tagged = Content::text(format!("[{}] {text}", self.tag));
        let remote_input = OperatorInput::new(tagged, layer0::operator::TriggerType::User);

        // Dispatch to remote agent via A2A — it's just a Dispatcher
        self.remote.dispatch(&OperatorId::new("remote"), remote_input).await
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_card(port: u16, name: &str, description: &str) -> AgentCard {
    AgentCard::builder(name, description)
        .version("0.1.0")
        .interface(
            format!("http://127.0.0.1:{port}"),
            "JSONRPC",
            "0.3",
        )
        .skill(AgentSkill::new("default", name, description))
        .input_mode("text/plain")
        .output_mode("text/plain")
        .build()
}

async fn start_server(
    dispatcher: Arc<dyn Dispatcher>,
    name: &str,
    description: &str,
) -> (SocketAddr, AgentCard) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");
    let addr = listener.local_addr().expect("local_addr failed");
    let card = make_card(addr.port(), name, description);

    let router = A2aServer::new(
        dispatcher,
        OperatorId::new("default"),
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

/// Two-agent delegation: Client → Agent A → (A2A) → Agent B → response.
///
/// Agent B echoes with prefix "B". Agent A tags with "[A]" before forwarding.
/// Final output should be "B: [A] hello from client".
#[tokio::test]
async fn multi_agent_delegation() {
    // Start Agent B (echo server)
    let agent_b: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher::new("B"));
    let (_addr_b, card_b) = start_server(agent_b, "agent-b", "Echo agent B").await;

    // Create Agent A that delegates to Agent B
    let remote_client = A2aDispatcher::new(card_b);
    let agent_a: Arc<dyn Dispatcher> = Arc::new(DelegatingDispatcher {
        remote: Arc::new(remote_client),
        tag: "A",
    });
    let (_addr_a, card_a) = start_server(agent_a, "agent-a", "Delegating agent A").await;

    // Client dispatches to Agent A
    let client = A2aDispatcher::new(card_a);
    let input = OperatorInput::new(
        Content::text("hello from client"),
        layer0::operator::TriggerType::User,
    );
    let output = client
        .dispatch(&OperatorId::new("any"), input)
        .await
        .expect("dispatch failed")
        .collect()
        .await
        .expect("collect failed");

    assert_eq!(output.message.as_text().unwrap(), "B: [A] hello from client");
    assert_eq!(output.exit_reason, ExitReason::Complete);
}

/// Three-agent chain: Client → A → B → C → response.
/// Verifies deep delegation works.
#[tokio::test]
async fn three_agent_chain() {
    // Agent C: the leaf echo agent
    let agent_c: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher::new("C"));
    let (_addr_c, card_c) = start_server(agent_c, "agent-c", "Leaf agent C").await;

    // Agent B: delegates to C
    let b_remote = A2aDispatcher::new(card_c);
    let agent_b: Arc<dyn Dispatcher> = Arc::new(DelegatingDispatcher {
        remote: Arc::new(b_remote),
        tag: "B",
    });
    let (_addr_b, card_b) = start_server(agent_b, "agent-b", "Relay agent B").await;

    // Agent A: delegates to B
    let a_remote = A2aDispatcher::new(card_b);
    let agent_a: Arc<dyn Dispatcher> = Arc::new(DelegatingDispatcher {
        remote: Arc::new(a_remote),
        tag: "A",
    });
    let (_addr_a, card_a) = start_server(agent_a, "agent-a", "Entry agent A").await;

    // Client → A → B → C → response
    let client = A2aDispatcher::new(card_a);
    let input = OperatorInput::new(
        Content::text("deep call"),
        layer0::operator::TriggerType::User,
    );
    let output = client
        .dispatch(&OperatorId::new("any"), input)
        .await
        .expect("dispatch failed")
        .collect()
        .await
        .expect("collect failed");

    assert_eq!(output.message.as_text().unwrap(), "C: [B] [A] deep call");
}

/// Verify `tasks/list` returns active dispatches and `extendedAgentCard/get`
/// returns the agent card through JSON-RPC.
#[tokio::test]
async fn jsonrpc_list_and_extended_card() {
    let agent: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher::new("test"));
    let (addr, _card) = start_server(agent, "test-agent", "Test agent").await;

    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", addr.port());

    // tasks/list should return empty (no active streams)
    let rpc = JsonRpcRequest::new("tasks/list", serde_json::json!({}));
    let resp: serde_json::Value = http
        .post(&url)
        .json(&rpc)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let tasks = resp["result"]["tasks"].as_array().expect("should have tasks array");
    assert!(tasks.is_empty(), "no active dispatches expected");

    // extendedAgentCard/get should return the card
    let rpc = JsonRpcRequest::new("extendedAgentCard/get", serde_json::json!({}));
    let resp: serde_json::Value = http
        .post(&url)
        .json(&rpc)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let name = resp["result"]["name"].as_str().expect("should have name");
    assert_eq!(name, "test-agent");
}

/// Verify unknown methods return MethodNotFound error.
#[tokio::test]
async fn unknown_method_returns_error() {
    let agent: Arc<dyn Dispatcher> = Arc::new(EchoDispatcher::new("test"));
    let (addr, _card) = start_server(agent, "test-agent", "Test agent").await;

    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", addr.port());

    let rpc = JsonRpcRequest::new("bogus/method", serde_json::json!({}));
    let resp: serde_json::Value = http
        .post(&url)
        .json(&rpc)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert!(resp.get("error").is_some(), "should have error: {resp}");
    let msg = resp["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("bogus/method"), "error should name the method: {msg}");
}
