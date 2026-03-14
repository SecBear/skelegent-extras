//! Integration tests for push notification handlers and auth middleware.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle, Dispatcher};
use layer0::error::OrchError;
use layer0::id::{DispatchId, OperatorId};
use layer0::DispatchContext;
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use skg_a2a::server::A2aServer;
use skg_a2a_core::jsonrpc::methods;
use skg_a2a_core::push::InMemoryPushStore;
use skg_a2a_core::{AgentCard, AgentSkill, JsonRpcRequest};
use skg_hook_security::auth::{AuthIdentity, StaticKeyValidator};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct NoopDispatcher;

#[async_trait]
impl Dispatcher for NoopDispatcher {
    async fn dispatch(
        &self,
        _ctx: &DispatchContext,
        _input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let output = OperatorOutput::new(Content::text("ok"), ExitReason::Complete);
        let (handle, sender) = DispatchHandle::channel(DispatchId::new("noop"));
        tokio::spawn(async move {
            let _ = sender.send(DispatchEvent::Completed { output }).await;
        });
        Ok(handle)
    }
}

fn test_card(port: u16) -> AgentCard {
    AgentCard::builder("test-push", "Push test agent")
        .version("0.1.0")
        .interface(
            format!("http://127.0.0.1:{port}"),
            "JSONRPC",
            "0.3",
        )
        .skill(AgentSkill::new("noop", "Noop", "Does nothing"))
        .input_mode("text/plain")
        .output_mode("text/plain")
        .build()
}

/// Start a server with push store and optional auth.
async fn start_server_with_push(
    push: Arc<InMemoryPushStore>,
    auth: Option<Arc<dyn skg_hook_security::auth::TokenValidator>>,
) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed");
    let addr = listener.local_addr().expect("local_addr failed");
    let card = test_card(addr.port());

    let mut server = A2aServer::new(
        Arc::new(NoopDispatcher) as Arc<dyn Dispatcher>,
        OperatorId::new("noop"),
        card,
    )
    .with_push(push);

    if let Some(v) = auth {
        server = server.with_auth(v);
    }

    let router = server.into_router();

    tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .expect("server error");
    });

    addr
}

/// POST a JSON-RPC request to the server with optional auth header.
async fn post_jsonrpc(
    addr: SocketAddr,
    request: &JsonRpcRequest,
    auth_header: Option<&str>,
) -> serde_json::Value {
    let url = format!("http://127.0.0.1:{}/", addr.port());
    let client = reqwest::Client::new();
    let mut builder = client.post(&url).json(request);
    if let Some(h) = auth_header {
        builder = builder.header("Authorization", h);
    }
    builder
        .send()
        .await
        .expect("request failed")
        .json::<serde_json::Value>()
        .await
        .expect("json parse failed")
}

// ---------------------------------------------------------------------------
// Push notification handler tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn push_create_and_get() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, None).await;

    // Create
    let create_req = JsonRpcRequest::new(
        methods::CREATE_PUSH_CONFIG,
        serde_json::json!({
            "task_id": "t-42",
            "config": {
                "url": "https://hooks.example.com/push",
                "token": "secret-abc"
            }
        }),
    );
    let resp = post_jsonrpc(addr, &create_req, None).await;
    assert!(
        resp.get("result").is_some(),
        "expected success response: {resp}"
    );
    let result = &resp["result"];
    assert_eq!(result["task_id"], "t-42");
    assert_eq!(result["url"], "https://hooks.example.com/push");
    assert_eq!(result["token"], "secret-abc");

    // Get
    let get_req = JsonRpcRequest::new(
        methods::GET_PUSH_CONFIG,
        serde_json::json!({ "task_id": "t-42" }),
    );
    let resp = post_jsonrpc(addr, &get_req, None).await;
    assert!(
        resp.get("result").is_some(),
        "expected success response: {resp}"
    );
    assert_eq!(resp["result"]["task_id"], "t-42");
    assert_eq!(resp["result"]["url"], "https://hooks.example.com/push");
}

#[tokio::test]
async fn push_list_and_delete() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, None).await;

    // Create two configs.
    for tid in &["t-1", "t-2"] {
        let req = JsonRpcRequest::new(
            methods::CREATE_PUSH_CONFIG,
            serde_json::json!({
                "task_id": tid,
                "config": { "url": format!("https://example.com/{tid}") }
            }),
        );
        let resp = post_jsonrpc(addr, &req, None).await;
        assert!(resp.get("result").is_some(), "create failed: {resp}");
    }

    // List should return 2.
    let list_req = JsonRpcRequest::new(methods::LIST_PUSH_CONFIGS, serde_json::json!({}));
    let resp = post_jsonrpc(addr, &list_req, None).await;
    let configs = resp["result"].as_array().expect("expected array");
    assert_eq!(configs.len(), 2);

    // Delete t-1.
    let del_req = JsonRpcRequest::new(
        methods::DELETE_PUSH_CONFIG,
        serde_json::json!({ "task_id": "t-1" }),
    );
    let resp = post_jsonrpc(addr, &del_req, None).await;
    assert!(resp.get("result").is_some(), "delete failed: {resp}");

    // List should return 1.
    let resp = post_jsonrpc(addr, &list_req, None).await;
    let configs = resp["result"].as_array().expect("expected array");
    assert_eq!(configs.len(), 1);
    assert_eq!(configs[0]["task_id"], "t-2");
}

#[tokio::test]
async fn push_get_nonexistent_returns_error() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, None).await;

    let req = JsonRpcRequest::new(
        methods::GET_PUSH_CONFIG,
        serde_json::json!({ "task_id": "nope" }),
    );
    let resp = post_jsonrpc(addr, &req, None).await;
    assert!(resp.get("error").is_some(), "expected error: {resp}");
    assert_eq!(resp["error"]["code"], -32001); // TaskNotFound
}

// ---------------------------------------------------------------------------
// Auth middleware tests
// ---------------------------------------------------------------------------

fn test_validator() -> Arc<dyn skg_hook_security::auth::TokenValidator> {
    Arc::new(
        StaticKeyValidator::new(HashMap::new()).with_key(
            "good-key",
            AuthIdentity {
                id: "test-user".into(),
                scopes: vec!["a2a".into()],
                metadata: HashMap::new(),
            },
        ),
    )
}

#[tokio::test]
async fn auth_rejects_missing_token() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, Some(test_validator())).await;

    let req = JsonRpcRequest::new(
        methods::LIST_PUSH_CONFIGS,
        serde_json::json!({}),
    );
    let resp = post_jsonrpc(addr, &req, None).await;
    assert!(resp.get("error").is_some(), "expected auth error: {resp}");
    let msg = resp["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("Authorization"),
        "expected auth-related error: {msg}"
    );
}

#[tokio::test]
async fn auth_rejects_bad_token() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, Some(test_validator())).await;

    let req = JsonRpcRequest::new(
        methods::LIST_PUSH_CONFIGS,
        serde_json::json!({}),
    );
    let resp = post_jsonrpc(addr, &req, Some("Bearer wrong-key")).await;
    assert!(resp.get("error").is_some(), "expected auth error: {resp}");
}

#[tokio::test]
async fn auth_allows_valid_token() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, Some(test_validator())).await;

    let req = JsonRpcRequest::new(
        methods::LIST_PUSH_CONFIGS,
        serde_json::json!({}),
    );
    let resp = post_jsonrpc(addr, &req, Some("Bearer good-key")).await;
    assert!(
        resp.get("result").is_some(),
        "expected success with valid token: {resp}"
    );
}

#[tokio::test]
async fn auth_does_not_block_agent_card() {
    let store = Arc::new(InMemoryPushStore::new());
    let addr = start_server_with_push(store, Some(test_validator())).await;

    // GET /.well-known/agent.json should work without auth.
    let url = format!(
        "http://127.0.0.1:{}/.well-known/agent.json",
        addr.port()
    );
    let resp = reqwest::get(&url)
        .await
        .expect("request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json parse failed");
    assert_eq!(body["name"], "test-push");
}
