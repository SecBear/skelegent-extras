//! Research operator — gathers evidence for a single decision.
//!
//! [`ResearchOperator`] is generic over [`ResearchSource`], keeping the sweep
//! pipeline decoupled from any concrete HTTP client. The sweep runner binary
//! provides an adapter that wraps `ParallelClient`.
//!
//! # Architecture
//!
//! This is a leaf operator: it takes [`ResearchInput`], calls the injected
//! [`ResearchSource`], and returns serialized `Vec<ResearchResult>`. It does
//! not dispatch sub-operators or emit effects beyond its own scoped state.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use layer0::{Content, ExitReason, Operator, OperatorError, OperatorInput, OperatorOutput, DispatchContext};
use layer0::dispatch::EffectEmitter;
use tracing::{debug, info};

use crate::provider::{ResearchInput, ResearchMode, ResearchResult, ResearchSource, SweepError};

/// Operator that gathers research evidence for a single decision.
///
/// Generic over `R: ResearchSource` — the concrete research backend is injected
/// at construction and never imported directly.
pub struct ResearchOperator<R: ResearchSource> {
    source: Arc<R>,
}

impl<R: ResearchSource> ResearchOperator<R> {
    /// Create a new research operator with the given source.
    pub fn new(source: Arc<R>) -> Self {
        Self { source }
    }
}

#[async_trait]
impl<R: ResearchSource> Operator for ResearchOperator<R> {
    async fn execute(&self, input: OperatorInput, _ctx: &DispatchContext, _emitter: &EffectEmitter) -> Result<OperatorOutput, OperatorError> {
        let text = input.message.as_text().unwrap_or("{}");
        let req: ResearchInput = serde_json::from_str(text).map_err(|e| {
            OperatorError::non_retryable(format!("ResearchOperator: invalid input: {e}"))
        })?;

        info!(
            decision_id = %req.decision_id,
            mode = ?req.mode,
            processor = %req.processor,
            query = %req.query,
            query_angle = %req.query_angle,
            "research: starting"
        );

        let start = Instant::now();
        let results: Vec<ResearchResult> = match req.mode {
            ResearchMode::Search => {
                self.source.search(&req.query).await.map_err(|e| match e {
                    SweepError::Transient(msg) => OperatorError::retryable(format!(
                        "ResearchOperator: transient search error: {msg}"
                    )),
                    other => OperatorError::non_retryable(format!(
                        "ResearchOperator: search error: {other}"
                    )),
                })?
            }
            ResearchMode::Deep => {
                self.source
                    .deep_research(&req.query, &req.processor)
                    .await
                    .map_err(|e| match e {
                        SweepError::Transient(msg) => OperatorError::retryable(format!(
                            "ResearchOperator: transient deep error: {msg}"
                        )),
                        other => OperatorError::non_retryable(format!(
                            "ResearchOperator: deep error: {other}"
                        )),
                    })?
            }
        };

        let elapsed = start.elapsed();
        let count = results.len();

        info!(
            decision_id = %req.decision_id,
            mode = ?req.mode,
            processor = %req.processor,
            num_results = count,
            duration_ms = elapsed.as_millis() as u64,
            "research: completed"
        );

        // Log individual result URLs at debug level for traceability.
        for (i, r) in results.iter().enumerate() {
            debug!(
                decision_id = %req.decision_id,
                idx = i,
                url = %r.url,
                title = %r.title,
                snippet_chars = r.snippet.len(),
                "research: result"
            );
        }

        let body = serde_json::to_string(&results).map_err(|e| {
            OperatorError::non_retryable(format!("ResearchOperator: serialize error: {e}"))
        })?;

        Ok(OperatorOutput::new(Content::text(body), ExitReason::Complete))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{DispatchId, OperatorId};
    use crate::provider::ResearchResult;
    use layer0::operator::TriggerType;

    struct MockSource {
        results: Vec<ResearchResult>,
    }

    #[async_trait]
    impl ResearchSource for MockSource {
        async fn search(&self, _query: &str) -> Result<Vec<ResearchResult>, SweepError> {
            Ok(self.results.clone())
        }

        async fn deep_research(
            &self,
            _query: &str,
            _processor: &str,
        ) -> Result<Vec<ResearchResult>, SweepError> {
            Ok(self.results.clone())
        }
    }

    fn make_source(n: usize) -> Arc<MockSource> {
        let results: Vec<ResearchResult> = (0..n)
            .map(|i| ResearchResult {
                url: format!("https://example.com/{i}"),
                title: format!("Result {i}"),
                snippet: format!("Snippet {i}"),
                retrieved_at: "2026-03-07T00:00:00Z".to_string(),
            })
            .collect();
        Arc::new(MockSource { results })
    }

    fn make_input(mode: ResearchMode) -> OperatorInput {
        let req = ResearchInput {
            decision_id: "D1".to_string(),
            query: "test query".to_string(),
            query_angle: "event-model".to_string(),
            mode,
            processor: "ultra".to_string(),
        };
        OperatorInput::new(
            Content::text(serde_json::to_string(&req).unwrap()),
            TriggerType::Task,
        )
    }

    #[tokio::test]
    async fn search_mode_returns_results() {
        let op = ResearchOperator::new(make_source(3));
        let output = op.execute(make_input(ResearchMode::Search), &DispatchContext::new(DispatchId::new("test"), OperatorId::new("test")), &EffectEmitter::noop()).await.unwrap();
        let results: Vec<ResearchResult> =
            serde_json::from_str(output.message.as_text().unwrap()).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].url, "https://example.com/0");
    }

    #[tokio::test]
    async fn deep_mode_returns_results() {
        let op = ResearchOperator::new(make_source(5));
        let output = op.execute(make_input(ResearchMode::Deep), &DispatchContext::new(DispatchId::new("test"), OperatorId::new("test")), &EffectEmitter::noop()).await.unwrap();
        let results: Vec<ResearchResult> =
            serde_json::from_str(output.message.as_text().unwrap()).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn invalid_input_returns_error() {
        let op = ResearchOperator::new(make_source(0));
        let input = OperatorInput::new(Content::text("not json"), TriggerType::Task);
        let err = op.execute(input, &DispatchContext::new(DispatchId::new("test"), OperatorId::new("test")), &EffectEmitter::noop()).await.unwrap_err();
        assert!(err.to_string().contains("invalid input"));
    }

    #[tokio::test]
    async fn empty_results_ok() {
        let op = ResearchOperator::new(make_source(0));
        let output = op.execute(make_input(ResearchMode::Search), &DispatchContext::new(DispatchId::new("test"), OperatorId::new("test")), &EffectEmitter::noop()).await.unwrap();
        let results: Vec<ResearchResult> =
            serde_json::from_str(output.message.as_text().unwrap()).unwrap();
        assert!(results.is_empty());
    }
}
