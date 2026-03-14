//! A2A client that implements [`Dispatcher`] by calling a remote A2A agent.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle, Dispatcher};
use layer0::error::OrchError;
use layer0::id::{DispatchId, OperatorId};
use layer0::DispatchContext;
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use skg_a2a_core::convert::{content_to_parts, parts_to_content};
use skg_a2a_core::jsonrpc::methods;
use skg_a2a_core::types::SendMessageResponse;
use skg_a2a_core::{
    A2aMessage, A2aRole, AgentCard, JsonRpcRequest, JsonRpcResponse,
};

/// Client-side errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum A2aClientError {
    /// HTTP request failed.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    /// JSON parse error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Agent card has no interfaces.
    #[error("agent card has no interfaces")]
    NoInterfaces,
}

/// A2A client that implements [`Dispatcher`] by calling a remote A2A agent.
///
/// The remote agent is identified by its [`AgentCard`]. The `operator` parameter
/// in [`Dispatcher::dispatch`] is ignored — routing is determined by the card's
/// interface URL.
pub struct A2aDispatcher {
    card: AgentCard,
    http: reqwest::Client,
}

impl A2aDispatcher {
    /// Create from a pre-fetched agent card.
    pub fn new(card: AgentCard) -> Self {
        Self {
            card,
            http: reqwest::Client::new(),
        }
    }

    /// Discover agent card from well-known URL and create client.
    pub async fn discover(base_url: &str) -> Result<Self, A2aClientError> {
        let http = reqwest::Client::new();
        let card_url = format!(
            "{}/.well-known/agent.json",
            base_url.trim_end_matches('/')
        );
        let card: AgentCard = http.get(&card_url).send().await?.json().await?;
        Ok(Self { card, http })
    }

    /// Return the agent card.
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    fn endpoint_url(&self) -> Result<&str, OrchError> {
        self.card
            .supported_interfaces
            .first()
            .map(|i| i.url.as_str())
            .ok_or_else(|| {
                OrchError::DispatchFailed("agent card has no interfaces".into())
            })
    }
}

#[async_trait]
impl Dispatcher for A2aDispatcher {
    async fn dispatch(
        &self,
        _ctx: &DispatchContext,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let url = self.endpoint_url()?.to_owned();

        // Build A2A message from operator input.
        let a2a_msg = A2aMessage::new(A2aRole::User, content_to_parts(&input.message));

        // Build SendMessageRequest via serde round-trip because the struct
        // is #[non_exhaustive] and has no public constructor.
        let send_req_value = serde_json::json!({
            "message": a2a_msg,
        });

        // Wrap in JSON-RPC envelope.
        let rpc = JsonRpcRequest::new(methods::SEND_MESSAGE, send_req_value);

        let http = self.http.clone();
        let (handle, sender) = DispatchHandle::channel(DispatchId::new("a2a"));

        tokio::spawn(async move {
            let result: Result<OperatorOutput, OrchError> = async {
                // POST to remote agent.
                let resp = http
                    .post(&url)
                    .json(&rpc)
                    .send()
                    .await
                    .map_err(|e| OrchError::DispatchFailed(e.to_string()))?;

                // Parse JSON-RPC response.
                let rpc_resp: JsonRpcResponse = resp
                    .json()
                    .await
                    .map_err(|e| OrchError::DispatchFailed(e.to_string()))?;

                // Extract A2A response.
                let send_resp: SendMessageResponse = serde_json::from_value(rpc_resp.result)
                    .map_err(|e| OrchError::DispatchFailed(e.to_string()))?;

                let content = match send_resp {
                    SendMessageResponse::Message { message } => parts_to_content(&message.parts),
                    SendMessageResponse::Task { task } => {
                        if let Some(last) = task.history.last() {
                            parts_to_content(&last.parts)
                        } else if let Some(artifact) = task.artifacts.last() {
                            parts_to_content(&artifact.parts)
                        } else {
                            Content::text("Task completed")
                        }
                    }
                    _ => Content::text("Unknown response type"),
                };

                Ok(OperatorOutput::new(content, ExitReason::Complete))
            }.await;

            match result {
                Ok(output) => { let _ = sender.send(DispatchEvent::Completed { output }).await; }
                Err(err) => { let _ = sender.send(DispatchEvent::Failed { error: err }).await; }
            }
        });

        Ok(handle)
    }
}
