//! A2A client that implements [`Dispatcher`] by calling a remote A2A agent.

pub mod stream;

pub use stream::dispatch_streaming;

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle, Dispatcher};
use layer0::error::OrchError;
use layer0::id::DispatchId;
use layer0::DispatchContext;
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use skg_a2a_core::convert::{content_to_parts, parts_to_content};
use skg_a2a_core::jsonrpc::methods;
use skg_a2a_core::types::SendMessageResponse;
use skg_a2a_core::{
    A2aMessage, A2aRole, AgentCard, AgentInterface, JsonRpcRequest, JsonRpcResponse,
};
use std::sync::Arc;
use std::time::Duration;

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

impl A2aClientError {
    /// Extract the HTTP status code from the underlying error, if available.
    ///
    /// Returns `Some(status)` when the error wraps a [`reqwest::Error`] that
    /// carries a status code (e.g., 4xx/5xx responses). Returns `None` for
    /// non-HTTP errors or when the reqwest error has no status.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::Http(e) => e.status().map(|s| s.as_u16()),
            _ => None,
        }
    }
}

/// Selects which [`AgentInterface`] to use from the agent card.
///
/// The default implementation picks the first interface. Implement this trait
/// to add custom selection logic (e.g., prefer interfaces with a specific
/// protocol binding, or select by tenant).
pub trait InterfaceSelector: Send + Sync {
    /// Select an interface from the list.
    ///
    /// Returns `None` if no suitable interface is found.
    fn select<'a>(&self, interfaces: &'a [AgentInterface]) -> Option<&'a AgentInterface>;
}

/// Default selector that picks the first interface.
#[derive(Debug, Clone)]
struct FirstInterfaceSelector;

impl InterfaceSelector for FirstInterfaceSelector {
    fn select<'a>(&self, interfaces: &'a [AgentInterface]) -> Option<&'a AgentInterface> {
        interfaces.first()
    }
}

/// A2A client that implements [`Dispatcher`] by calling a remote A2A agent.
///
/// The remote agent is identified by its [`AgentCard`]. The `operator` parameter
/// in [`Dispatcher::dispatch`] is ignored — routing is determined by the card's
/// interface URL.
///
/// Use [`A2aDispatcherBuilder`] for advanced configuration (custom HTTP client,
/// timeout, interface selection).
pub struct A2aDispatcher {
    card: AgentCard,
    http: reqwest::Client,
    /// HTTP request timeout.
    timeout: Option<Duration>,
    /// Strategy for selecting which interface to use.
    selector: Arc<dyn InterfaceSelector>,
}

impl A2aDispatcher {
    /// Create from a pre-fetched agent card with default settings.
    pub fn new(card: AgentCard) -> Self {
        Self {
            card,
            http: reqwest::Client::new(),
            timeout: None,
            selector: Arc::new(FirstInterfaceSelector),
        }
    }

    /// Return a builder for advanced configuration.
    pub fn builder(card: AgentCard) -> A2aDispatcherBuilder {
        A2aDispatcherBuilder::new(card)
    }

    /// Discover agent card from well-known URL and create client.
    pub async fn discover(base_url: &str) -> Result<Self, A2aClientError> {
        let http = reqwest::Client::new();
        let card_url = format!(
            "{}/.well-known/agent.json",
            base_url.trim_end_matches('/')
        );
        let card: AgentCard = http.get(&card_url).send().await?.json().await?;
        Ok(Self {
            card,
            http,
            timeout: None,
            selector: Arc::new(FirstInterfaceSelector),
        })
    }

    /// Return the agent card.
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Return the configured request timeout, if any.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    fn endpoint_url(&self) -> Result<&str, OrchError> {
        self.selector
            .select(&self.card.supported_interfaces)
            .map(|i| i.url.as_str())
            .ok_or_else(|| {
                OrchError::DispatchFailed("agent card has no interfaces".into())
            })
    }

    /// Dispatch using streaming SSE if the agent card advertises streaming
    /// capability, falling back to the sync [`Dispatcher::dispatch`] otherwise.
    ///
    /// When streaming is available this calls [`dispatch_streaming`] with the
    /// first interface URL from the agent card.  When `capabilities.streaming`
    /// is `None` or `false`, this falls through to the synchronous
    /// [`Dispatcher::dispatch`] implementation.
    pub async fn dispatch_auto(
        &self,
        ctx: &DispatchContext,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let streaming_supported = self
            .card
            .capabilities
            .streaming
            .unwrap_or(false);

        if streaming_supported {
            let url = self.endpoint_url()?.to_owned();
            stream::dispatch_streaming(&self.http, &url, input).await
        } else {
            self.dispatch(ctx, input).await
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BUILDER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Builder for [`A2aDispatcher`] with custom HTTP client, timeout, and
/// interface selection.
///
/// ```rust,ignore
/// use skg_a2a::client::{A2aDispatcherBuilder, InterfaceSelector};
///
/// let dispatcher = A2aDispatcherBuilder::new(card)
///     .with_timeout(Duration::from_secs(10))
///     .build();
/// ```
pub struct A2aDispatcherBuilder {
    card: AgentCard,
    http: Option<reqwest::Client>,
    timeout: Option<Duration>,
    selector: Option<Arc<dyn InterfaceSelector>>,
}

impl A2aDispatcherBuilder {
    /// Create a new builder for the given agent card.
    pub fn new(card: AgentCard) -> Self {
        Self {
            card,
            http: None,
            timeout: None,
            selector: None,
        }
    }

    /// Use a pre-configured HTTP client instead of the default.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    /// Set a request timeout for all HTTP calls.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set a custom interface selector.
    ///
    /// The default selector picks the first interface from the agent card.
    pub fn with_interface_selector(mut self, selector: Arc<dyn InterfaceSelector>) -> Self {
        self.selector = Some(selector);
        self
    }

    /// Build the [`A2aDispatcher`].
    pub fn build(self) -> A2aDispatcher {
        let http = self.http.unwrap_or_else(|| {
            let mut builder = reqwest::Client::builder();
            if let Some(timeout) = self.timeout {
                builder = builder.timeout(timeout);
            }
            builder.build().expect("failed to build reqwest client")
        });
        A2aDispatcher {
            card: self.card,
            http,
            timeout: self.timeout,
            selector: self.selector.unwrap_or_else(|| Arc::new(FirstInterfaceSelector)),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_card() -> AgentCard {
        AgentCard::builder("test-agent", "test agent")
            .version("0.1.0")
            .interface("http://localhost:9999/a2a", "jsonrpc/http", "0.1.0")
            .build()
    }

    fn test_card_two_interfaces() -> AgentCard {
        AgentCard::builder("test-agent", "test agent")
            .version("0.1.0")
            .interface("http://localhost:9999/a2a", "jsonrpc/http", "0.1.0")
            .interface("http://localhost:8888/a2a", "jsonrpc/http", "0.1.0")
            .build()
    }

    #[test]
    fn builder_stores_timeout() {
        let timeout = Duration::from_secs(42);
        let dispatcher = A2aDispatcherBuilder::new(test_card())
            .with_timeout(timeout)
            .build();
        assert_eq!(dispatcher.timeout(), Some(timeout));
    }

    #[test]
    fn builder_default_has_no_timeout() {
        let dispatcher = A2aDispatcher::new(test_card());
        assert_eq!(dispatcher.timeout(), None);
    }

    #[test]
    fn builder_custom_interface_selector() {
        /// Selector that picks the last interface.
        struct LastSelector;
        impl InterfaceSelector for LastSelector {
            fn select<'a>(&self, interfaces: &'a [AgentInterface]) -> Option<&'a AgentInterface> {
                interfaces.last()
            }
        }

        let card = test_card_two_interfaces();
        let dispatcher = A2aDispatcherBuilder::new(card)
            .with_interface_selector(Arc::new(LastSelector))
            .build();

        let url = dispatcher.endpoint_url().unwrap();
        assert_eq!(url, "http://localhost:8888/a2a");
    }

    #[test]
    fn builder_with_custom_http_client() {
        let client = reqwest::Client::builder()
            .user_agent("custom-agent")
            .build()
            .unwrap();
        let dispatcher = A2aDispatcherBuilder::new(test_card())
            .with_http_client(client)
            .with_timeout(Duration::from_secs(5))
            .build();
        // Timeout is stored even though we passed a custom client.
        assert_eq!(dispatcher.timeout(), Some(Duration::from_secs(5)));
    }
}
