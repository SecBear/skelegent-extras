//! Sweep comparison operator — compares research against decisions via LLM.
//!
//! ## v1 vs v2
//!
//! | Aspect | v1 | v2 |
//! |--------|----|----|
//! | Context building | `ProviderMessage { role: Role::User, content: vec![ContentPart::Text { text }] }` | `Message::new(Role::User, Content::text(text))` |
//! | Provider call | Build `ProviderRequest` with `Vec<ProviderMessage>` | Build `ProviderRequest` with `messages.iter().map(message_to_provider).collect()` |
//! | Response text | `response.content.iter().find_map(ContentPart::Text)` | `parts_to_content(&response.content).as_text()` |
//! | Context storage | `Vec<AnnotatedMessage>` with `Option<CompactionPolicy>` | `Vec<Message>` with `meta.policy: CompactionPolicy` |
//! | Compaction | `Box<dyn ContextStrategy>` trait object | `with_compactor(\|msgs\| ...)` closure |

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use layer0::content::Content;
use layer0::context::{Message, Role};
use layer0::operator::OperatorMetadata;
use layer0::state::MemoryTier;
use layer0::{Effect, ExitReason, Operator, OperatorError, OperatorInput, OperatorOutput, Scope};
use skg_orch_kit::ScopedState;
use skg_turn::infer::InferRequest;
use skg_turn::provider::{Provider, ProviderError};
use rust_decimal::Decimal;
use tracing::info;

use crate::provider::CompareInput;
use crate::types::{ProcessorTier, SweepVerdict, VerdictStatus};

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/// Comparison system prompt sent to the LLM.
pub(crate) const COMPARISON_SYSTEM_PROMPT: &str = r#"You are an architectural decision analyst. You will receive:
1. A <decision> section containing the current decision text
2. A <prior_findings> section with previous sweep results (may be empty)
3. A <research> section with new research findings

Compare the research findings against the existing decision and produce a structured JSON verdict matching this EXACT schema:

```json
{
  "decision_id": "<the decision ID from the input>",
  "status": "confirmed|refined|challenged|obsoleted|skipped",
  "confidence": 0.85,
  "num_supporting": 3,
  "num_contradicting": 0,
  "cost_usd": 0.0,
  "processor": "ultra",
  "duration_secs": 0.0,
  "swept_at": "2026-01-01T00:00:00Z",
  "evidence": [
    {
      "source_url": "https://example.com/paper",
      "title": "Relevant Paper Title",
      "stance": "supporting",
      "summary": "One sentence summary of how this source relates to the decision."
    }
  ],
  "narrative": "Markdown narrative explaining what was found and why.",
  "proposed_diff": null
}
```

Rules:
- "confirmed" requires >= 3 supporting sources and 0 contradicting
- "refined" means the core decision holds but evidence/wording needs updating
- "challenged" requires >= 2 independent contradicting sources
- "obsoleted" means the entire decision space has been superseded
- confidence MUST be a float between 0.0 and 1.0
- Cite every claim with a source URL
- Do NOT hallucinate sources
- Return ONLY the JSON object inside ```json fences"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`CompareOperator`].
#[derive(Debug, Clone)]
pub struct CompareConfig {
    /// Minimum time between sweeps of the same decision. Default: 20 hours.
    pub min_sweep_interval: Duration,
    /// Maximum research queries per sweep. Default: 5.
    pub max_queries: usize,
    /// Maximum artifacts to store. Default: 20.
    pub max_artifacts: usize,
    /// Model identifier. Default: `"claude-sonnet-4-20250514"`.
    pub model: String,
    /// Maximum tokens for the comparison response. Default: 4096.
    pub max_response_tokens: usize,
    /// Always use Ultra processor regardless of budget. Default: true.
    pub force_ultra: bool,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            min_sweep_interval: Duration::from_secs(20 * 3600),
            max_queries: 5,
            max_artifacts: 20,
            model: "claude-sonnet-4-20250514".to_string(),
            max_response_tokens: 4096,
            force_ultra: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Processor selection
// ---------------------------------------------------------------------------

/// Select research processor tier based on budget and previous verdict.
pub fn select_processor(
    budget_remaining_usd: f64,
    budget_total_usd: f64,
    previous_verdict: Option<&VerdictStatus>,
) -> ProcessorTier {
    if budget_total_usd <= 0.0 {
        return ProcessorTier::Base;
    }
    let budget_ratio = budget_remaining_usd / budget_total_usd;
    match previous_verdict {
        Some(VerdictStatus::Challenged) => {
            if budget_ratio > 0.5 {
                ProcessorTier::Ultra
            } else {
                ProcessorTier::Core
            }
        }
        Some(VerdictStatus::Refined) => ProcessorTier::Core,
        _ => {
            if budget_ratio > 0.8 {
                ProcessorTier::Core
            } else {
                ProcessorTier::Base
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON extraction helper
// ---------------------------------------------------------------------------

/// Strip optional ` ```json ` / ` ``` ` fence and return the inner JSON.
fn extract_json_block(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(inner) = trimmed.strip_prefix("```json") {
        inner.trim_start().trim_end_matches("```").trim()
    } else if let Some(inner) = trimmed.strip_prefix("```") {
        inner.trim_end_matches("```").trim()
    } else {
        trimmed
    }
}

// ---------------------------------------------------------------------------
// CompareOperator
// ---------------------------------------------------------------------------

/// Compares research findings against a decision via an LLM.
///
/// ## v2 differences from v1
///
/// Context assembly uses [`Message`] directly instead of [`ProviderMessage`].
/// The operator builds a `Vec<Message>` context, then converts to
/// `Vec<ProviderMessage>` only at the Provider call boundary via
/// [`message_to_provider`].
///
/// Response content is extracted via [`parts_to_content`] → [`Content::as_text`]
/// instead of manually matching on `ContentPart::Text`.
pub struct CompareOperator<P: Provider> {
    /// LLM provider for verdict comparison.
    llm: P,
    /// Scoped state for reading decision cards and prior findings.
    state: Arc<dyn ScopedState>,
    /// Configuration.
    pub config: CompareConfig,
}

impl<P: Provider> CompareOperator<P> {
    /// Create a new compare operator.
    pub fn new(llm: P, state: Arc<dyn ScopedState>, config: CompareConfig) -> Self {
        Self { llm, state, config }
    }
}

#[async_trait]
impl<P: Provider + 'static> Operator for CompareOperator<P> {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        let start = Instant::now();

        // Parse typed input.
        let msg_text = input.message.as_text().ok_or_else(|| {
            OperatorError::NonRetryable("CompareOperator: input must be text JSON".into())
        })?;
        let compare_in: CompareInput = serde_json::from_str(msg_text).map_err(|e| {
            OperatorError::NonRetryable(format!("CompareOperator: parse error: {e}"))
        })?;
        let research_json = serde_json::to_string(&compare_in.research_results)
            .unwrap_or_else(|_| "[]".to_string());
        let decision_id = compare_in.decision_id;

        info!(
            decision_id = %decision_id,
            num_research = compare_in.research_results.len(),
            model = %self.config.model,
            "compare: starting"
        );

        // --- Context assembly using unified Message type ---
        //
        // v1: Built ProviderMessage { role: Role::User, content: vec![ContentPart::Text { text }] }
        // v2: Build Message::new(Role::User, Content::text(text)) — cleaner, carries metadata

        let decision_text = match self
            .state
            .read(&format!("card:{}", decision_id))
            .await
            .unwrap_or(None)
        {
            Some(serde_json::Value::String(s)) => s,
            Some(other) => other.to_string(),
            None => "[Decision text not available]".to_string(),
        };

        let delta_keys = self
            .state
            .list(&format!("delta:{}:", decision_id))
            .await
            .unwrap_or_default();
        let mut prior_findings: Vec<String> = Vec::with_capacity(delta_keys.len());
        for key in &delta_keys {
            if let Ok(Some(val)) = self.state.read(key).await {
                prior_findings.push(match val {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                });
            }
        }
        let prior_json =
            serde_json::to_string(&prior_findings).unwrap_or_else(|_| "[]".to_string());

        let user_content = format!(
            "Decision ID: {decision_id}\n\n\
             <decision>\n{decision_text}\n</decision>\n\n\
             <prior_findings>\n{prior_json}\n</prior_findings>\n\n\
             <research>\n{research_json}\n</research>"
        );

        // v2: Build context as Vec<Message>, convert at provider boundary.
        //
        // This is the key architectural difference. Messages carry MessageMeta
        // (compaction policy, salience, source) which survives through the context
        // pipeline. At the Provider call boundary, metadata is stripped via
        // message_to_provider().
        let context = [Message::new(Role::User, Content::text(&user_content))];

        let request = InferRequest {
            model: Some(self.config.model.clone()),
            messages: context.to_vec(),
            tools: vec![],
            max_tokens: Some(self.config.max_response_tokens as u32),
            temperature: None,
            system: Some(COMPARISON_SYSTEM_PROMPT.to_string()),
            extra: serde_json::Value::Null,
        };

        let response = self.llm.infer(request).await.map_err(|e| match e {
            ProviderError::RateLimited => {
                OperatorError::Retryable("rate limited by LLM provider".into())
            }
            ProviderError::TransientError { message, .. } => OperatorError::Retryable(message),
            ProviderError::AuthFailed(msg) => {
                OperatorError::NonRetryable(format!("auth: {msg}"))
            }
            other => OperatorError::Model(other.to_string()),
        })?;

        // InferResponse.content is already Content — no conversion needed.
        let raw = response.content.as_text().unwrap_or("");

        let llm_elapsed = start.elapsed();
        info!(
            decision_id = %decision_id,
            tokens_in = response.usage.input_tokens,
            tokens_out = response.usage.output_tokens,
            cost_usd = %response.cost.unwrap_or(Decimal::ZERO),
            duration_ms = llm_elapsed.as_millis() as u64,
            "compare: response received"
        );

        // Parse JSON verdict (may be fenced).
        let mut verdict: SweepVerdict = {
            let json_str = extract_json_block(raw);
            serde_json::from_str(json_str).map_err(|e| {
                OperatorError::NonRetryable(format!(
                    "compare: verdict parse failed: {e}\nraw:\n{raw}"
                ))
            })?
        };

        // Fill in runtime fields.
        verdict.cost_usd = response
            .cost
            .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        verdict.duration_secs = llm_elapsed.as_secs_f64();
        verdict.swept_at = chrono::Utc::now().to_rfc3339();
        verdict.research_inputs = compare_in.research_results;
        verdict.query = compare_in.query;
        verdict.query_angle = compare_in.query_angle;

        // Write sweep timestamp to own scope.
        let _ = self
            .state
            .write(
                &format!("meta:{}:last_sweep", decision_id),
                serde_json::Value::String(verdict.swept_at.clone()),
            )
            .await;

        // Declare cross-scope effect for the delta entry.
        let delta_key = format!("delta:{}:{}", decision_id, verdict.swept_at);
        let delta_value = serde_json::to_value(&verdict)
            .unwrap_or_else(|_| serde_json::Value::String("serialization failed".into()));

        let effects = vec![Effect::WriteMemory {
            scope: Scope::Custom("sweep-results".into()),
            key: delta_key,
            value: delta_value,
            tier: Some(MemoryTier::Hot),
            lifetime: None,
            content_kind: None,
            salience: None,
            ttl: None,
        }];

        // v2: Return Content::text() directly — no ContentPart intermediary.
        let verdict_json = serde_json::to_string(&verdict).unwrap_or_default();
        let mut output = OperatorOutput::new(Content::text(&verdict_json), ExitReason::Complete);
        output.effects = effects;
        let mut meta = OperatorMetadata::default();
        meta.tokens_in = response.usage.input_tokens;
        meta.tokens_out = response.usage.output_tokens;
        meta.cost = response.cost.unwrap_or(Decimal::ZERO);
        meta.turns_used = 1;
        meta.duration = layer0::DurationMs::from(llm_elapsed);
        output.metadata = meta;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skg_turn::infer::InferResponse;
    use skg_turn::test_utils::TestProvider;
    use skg_turn::types::{StopReason, TokenUsage};

    struct NullState;

    #[async_trait]
    impl ScopedState for NullState {
        async fn read(
            &self,
            _key: &str,
        ) -> Result<Option<serde_json::Value>, layer0::error::StateError> {
            Ok(None)
        }
        async fn write(
            &self,
            _key: &str,
            _value: serde_json::Value,
        ) -> Result<(), layer0::error::StateError> {
            Ok(())
        }
        async fn delete(&self, _key: &str) -> Result<(), layer0::error::StateError> {
            Ok(())
        }
        async fn list(&self, _prefix: &str) -> Result<Vec<String>, layer0::error::StateError> {
            Ok(vec![])
        }
        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<layer0::state::SearchResult>, layer0::error::StateError> {
            Ok(vec![])
        }
    }

    fn verdict_response(verdict_json: &str) -> InferResponse {
        InferResponse {
            content: Content::text(format!("```json\n{verdict_json}\n```")),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            model: "mock".into(),
            cost: Some(Decimal::new(1, 4)),
            truncated: None,
        }
    }

    #[tokio::test]
    async fn compare_produces_verdict() {
        let verdict = r#"{
            "decision_id": "D1",
            "status": "confirmed",
            "confidence": 0.9,
            "num_supporting": 3,
            "num_contradicting": 0,
            "cost_usd": 0.0,
            "processor": "ultra",
            "duration_secs": 0.0,
            "swept_at": "2026-01-01T00:00:00Z",
            "evidence": [],
            "narrative": "All good.",
            "proposed_diff": null,
            "research_inputs": [],
            "query": "test query",
            "query_angle": "test angle"
        }"#;
        let provider = TestProvider::with_responses(vec![verdict_response(verdict)]);
        let op = CompareOperator::new(provider, Arc::new(NullState), CompareConfig::default());

        let input_json = serde_json::to_string(&CompareInput {
            research_results: vec![],
            decision_id: "D1".into(),
            query: "test query".into(),
            query_angle: "test angle".into(),
        })
        .unwrap();

        let input = OperatorInput::new(
            Content::text(&input_json),
            layer0::operator::TriggerType::Task,
        );
        let output = op.execute(input).await.unwrap();

        assert_eq!(output.exit_reason, ExitReason::Complete);
        let result: SweepVerdict =
            serde_json::from_str(output.message.as_text().unwrap()).unwrap();
        assert_eq!(result.status, VerdictStatus::Confirmed);
        assert_eq!(result.decision_id, "D1");
        assert!(!output.effects.is_empty());
    }

    #[test]
    fn extract_json_block_strips_fence() {
        let fenced = "```json\n{\"key\": \"value\"}\n```";
        assert_eq!(extract_json_block(fenced), r#"{"key": "value"}"#);
    }

    #[test]
    fn extract_json_block_passthrough() {
        let raw = r#"{"key": "value"}"#;
        assert_eq!(extract_json_block(raw), raw);
    }

    #[test]
    fn select_processor_challenged_high_budget() {
        let tier = select_processor(80.0, 100.0, Some(&VerdictStatus::Challenged));
        assert_eq!(tier, ProcessorTier::Ultra);
    }

    #[test]
    fn select_processor_low_budget() {
        let tier = select_processor(10.0, 100.0, None);
        assert_eq!(tier, ProcessorTier::Base);
    }
}
