//! [`ReplayProvider`] — deterministic replay of recorded infer/embed operations.

use crate::ReplayError;
use skg_hook_recorder::{Boundary, Phase, RecordEntry, SCHEMA_VERSION};
use skg_turn::embedding::{EmbedRequest, EmbedResponse};
use skg_turn::infer::{InferRequest, InferResponse};
use skg_turn::provider::{Provider, ProviderError};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// REPLAY PROVIDER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A [`Provider`] that replays recorded infer and embed outcomes.
///
/// Instead of making real LLM calls, `ReplayProvider` feeds pre-recorded
/// [`RecordEntry`] responses back to the caller sequentially.
///
/// Infer recordings are [`Boundary::Infer`] + [`Phase::Post`] entries.
/// Embed recordings are [`Boundary::Embed`] + [`Phase::Post`] entries.
/// Each boundary has its own independent sequential index.
///
/// The `payload_json` of each Post entry must contain a serialized
/// [`InferResponse`] or [`EmbedResponse`] respectively.
///
/// # Example
///
/// ```rust,ignore
/// use skg_hook_replay::ReplayProvider;
/// use skg_hook_recorder::RecordEntry;
///
/// let entries = vec![/* recorded Infer/Embed Post entries */];
/// let provider = ReplayProvider::new(entries);
/// ```
pub struct ReplayProvider {
    infer_recordings: Vec<RecordEntry>,
    embed_recordings: Vec<RecordEntry>,
    /// Per-entry consumed flags for infer recordings.
    infer_consumed: Vec<AtomicBool>,
    /// Per-entry consumed flags for embed recordings.
    embed_consumed: Vec<AtomicBool>,
}

impl ReplayProvider {
    /// Create a new provider from a recording sequence.
    ///
    /// Filters for [`Boundary::Infer`] / [`Boundary::Embed`] + [`Phase::Post`]
    /// entries. Other entries are discarded.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::VersionMismatch`] if any entry's `version` does
    /// not match [`skg_hook_recorder::SCHEMA_VERSION`].
    pub fn new(recordings: Vec<RecordEntry>) -> Result<Self, ReplayError> {
        for entry in &recordings {
            if entry.version != SCHEMA_VERSION {
                return Err(ReplayError::VersionMismatch {
                    recorded: entry.version,
                    current: SCHEMA_VERSION,
                });
            }
        }
        let mut infer_recordings = Vec::new();
        let mut embed_recordings = Vec::new();
        for entry in recordings {
            if entry.phase == Phase::Post {
                match entry.boundary {
                    Boundary::Infer => infer_recordings.push(entry),
                    Boundary::Embed => embed_recordings.push(entry),
                    _ => {}
                }
            }
        }
        let infer_consumed = infer_recordings.iter().map(|_| AtomicBool::new(false)).collect();
        let embed_consumed = embed_recordings.iter().map(|_| AtomicBool::new(false)).collect();
        Ok(Self {
            infer_recordings,
            embed_recordings,
            infer_consumed,
            embed_consumed,
        })
    }
}

impl Provider for ReplayProvider {
    fn infer(
        &self,
        _request: InferRequest,
    ) -> impl Future<Output = Result<InferResponse, ProviderError>> + Send {
        // Find the first unconsumed infer recording and mark it consumed.
        let found = self
            .infer_consumed
            .iter()
            .enumerate()
            .find(|(_, c)| !c.load(Ordering::SeqCst));
        let result = match found {
            None => {
                let position = self.infer_consumed.len();
                Err(ProviderError::InvalidRequest {
                    message: ReplayError::RecordingExhausted { position }.to_string(),
                    status: None,
                })
            }
            Some((idx, flag)) => {
                flag.store(true, Ordering::SeqCst);
                let entry = &self.infer_recordings[idx];
                if let Some(ref msg) = entry.error {
                    Err(ProviderError::InvalidRequest {
                        message: msg.clone(),
                        status: None,
                    })
                } else {
                    serde_json::from_value::<InferResponse>(entry.payload_json.clone())
                        .map_err(|e| {
                            ProviderError::InvalidResponse(
                                ReplayError::PayloadError(e.to_string()).to_string(),
                            )
                        })
                }
            }
        };
        std::future::ready(result)
    }

    fn embed(
        &self,
        _request: EmbedRequest,
    ) -> impl Future<Output = Result<EmbedResponse, ProviderError>> + Send {
        // Find the first unconsumed embed recording and mark it consumed.
        let found = self
            .embed_consumed
            .iter()
            .enumerate()
            .find(|(_, c)| !c.load(Ordering::SeqCst));
        let result = match found {
            None => {
                let position = self.embed_consumed.len();
                Err(ProviderError::InvalidRequest {
                    message: ReplayError::RecordingExhausted { position }.to_string(),
                    status: None,
                })
            }
            Some((idx, flag)) => {
                flag.store(true, Ordering::SeqCst);
                let entry = &self.embed_recordings[idx];
                if let Some(ref msg) = entry.error {
                    Err(ProviderError::InvalidRequest {
                        message: msg.clone(),
                        status: None,
                    })
                } else {
                    serde_json::from_value::<EmbedResponse>(entry.payload_json.clone())
                        .map_err(|e| {
                            ProviderError::InvalidResponse(
                                ReplayError::PayloadError(e.to_string()).to_string(),
                            )
                        })
                }
            }
        };
        std::future::ready(result)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TESTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use skg_hook_recorder::{Boundary, Phase, RecordContext, RecordEntry, SCHEMA_VERSION};
    use skg_turn::embedding::{EmbedRequest, EmbedResponse, Embedding};
    use skg_turn::infer::InferResponse;
    use skg_turn::types::{StopReason, TokenUsage};

    fn make_infer_response(text: &str) -> InferResponse {
        InferResponse {
            content: Content::text(text),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: "test-model".into(),
            cost: None,
            truncated: None,
        }
    }

    fn make_embed_response(vectors: Vec<Vec<f32>>) -> EmbedResponse {
        EmbedResponse {
            embeddings: vectors
                .into_iter()
                .map(|v| Embedding { vector: v })
                .collect(),
            model: "test-model".into(),
            usage: TokenUsage::default(),
        }
    }

    fn infer_post_entry(response: &InferResponse) -> RecordEntry {
        RecordEntry {
            boundary: Boundary::Infer,
            phase: Phase::Post,
            context: RecordContext::empty(),
            payload_json: serde_json::to_value(response).expect("serialize"),
            duration_ms: Some(50),
            error: None,
            version: SCHEMA_VERSION,
        }
    }

    fn embed_post_entry(response: &EmbedResponse) -> RecordEntry {
        RecordEntry {
            boundary: Boundary::Embed,
            phase: Phase::Post,
            context: RecordContext::empty(),
            payload_json: serde_json::to_value(response).expect("serialize"),
            duration_ms: Some(20),
            error: None,
            version: SCHEMA_VERSION,
        }
    }

    fn make_infer_request() -> InferRequest {
        use layer0::context::{Message, Role};
        InferRequest::new(vec![Message::new(Role::User, Content::text("hello"))])
    }

    fn make_embed_request() -> EmbedRequest {
        EmbedRequest::new(vec!["hello".into()])
    }

    #[tokio::test]
    async fn replay_provider_infer() {
        let r1 = make_infer_response("first response");
        let r2 = make_infer_response("second response");
        let entries = vec![infer_post_entry(&r1), infer_post_entry(&r2)];

        let provider = ReplayProvider::new(entries).unwrap();

        let resp0 = provider.infer(make_infer_request()).await.unwrap();
        assert_eq!(resp0.content.as_text(), Some("first response"));

        let resp1 = provider.infer(make_infer_request()).await.unwrap();
        assert_eq!(resp1.content.as_text(), Some("second response"));
    }

    #[tokio::test]
    async fn replay_provider_infer_exhausted() {
        let r = make_infer_response("only");
        let entries = vec![infer_post_entry(&r)];
        let provider = ReplayProvider::new(entries).unwrap();

        provider.infer(make_infer_request()).await.expect("first");

        let err = provider
            .infer(make_infer_request())
            .await
            .expect_err("should be exhausted");
        assert!(
            err.to_string().contains("exhausted"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn replay_provider_embed() {
        let r1 = make_embed_response(vec![vec![0.1, 0.2, 0.3]]);
        let r2 = make_embed_response(vec![vec![0.4, 0.5, 0.6], vec![0.7, 0.8, 0.9]]);
        let entries = vec![embed_post_entry(&r1), embed_post_entry(&r2)];

        let provider = ReplayProvider::new(entries).unwrap();

        let resp0 = provider.embed(make_embed_request()).await.unwrap();
        assert_eq!(resp0.embeddings.len(), 1);
        assert_eq!(resp0.embeddings[0].vector, vec![0.1f32, 0.2, 0.3]);

        let resp1 = provider.embed(make_embed_request()).await.unwrap();
        assert_eq!(resp1.embeddings.len(), 2);
    }

    #[tokio::test]
    async fn replay_provider_embed_exhausted() {
        let r = make_embed_response(vec![vec![0.1]]);
        let provider = ReplayProvider::new(vec![embed_post_entry(&r)]).unwrap();

        provider.embed(make_embed_request()).await.expect("first");

        let err = provider
            .embed(make_embed_request())
            .await
            .expect_err("should be exhausted");
        assert!(
            err.to_string().contains("exhausted"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn replay_provider_infer_and_embed_independent() {
        // Infer and embed recordings are independent — interleaved calls work correctly.
        let ir = make_infer_response("infer");
        let er = make_embed_response(vec![vec![1.0]]);
        let entries = vec![infer_post_entry(&ir), embed_post_entry(&er)];

        let provider = ReplayProvider::new(entries).unwrap();

        let embed_resp = provider.embed(make_embed_request()).await.unwrap();
        assert_eq!(embed_resp.embeddings.len(), 1);

        let infer_resp = provider.infer(make_infer_request()).await.unwrap();
        assert_eq!(infer_resp.content.as_text(), Some("infer"));
    }
}
