//! Sweep operator pipeline — orchestrates research, comparison, and verdict.
//!
//! [`ResearchOperator`] and [`CompareOperator`] implement the pipeline defined in
//! spec section 4.2: research → artifact storage → compare → verdict.
//! Orchestrated by [`crate::workflow::run_sweep`].
//!
//! [`ResearchOperator`] and [`CompareOperator`] implement the [`Operator`] trait,
//! allowing them to be dispatched via an [`Orchestrator`] without violating the
//! effects boundary. Use [`crate::workflow::run_sweep`] to sequence them.

use std::time::Duration;

use layer0::state::{ContentKind, Lifetime, MemoryTier, StateReader, StateStore};
use layer0::{
    Content, DurationMs, Effect, ExitReason, Operator, OperatorError, OperatorInput,
    OperatorOutput, Scope,
};

use crate::provider::{ResearchProvider, ResearchResult, SweepError};
use crate::types::{ProcessorTier, SweepMeta, SweepVerdict, VerdictStatus};
use neuron_turn::provider::{Provider, ProviderError};
use neuron_turn::types::{ContentPart, ProviderMessage, ProviderRequest, Role};

// ---------------------------------------------------------------------------
// StoreAsReader adapter
// ---------------------------------------------------------------------------

/// Thin adapter that allows a `&dyn StateStore` to be used where a
/// `&dyn StateReader` is required (e.g. `ContextAssembler::assemble`).
///
/// The blanket impl `impl<T: StateStore> StateReader for T` does not apply
/// to unsized `dyn StateStore`, so this wrapper forwards explicitly.
pub(crate) struct StoreAsReader<'a>(pub(crate) &'a dyn StateStore);

#[async_trait::async_trait]
impl StateReader for StoreAsReader<'_> {
    async fn read(
        &self,
        scope: &Scope,
        key: &str,
    ) -> Result<Option<serde_json::Value>, layer0::error::StateError> {
        self.0.read(scope, key).await
    }
    async fn list(
        &self,
        scope: &Scope,
        prefix: &str,
    ) -> Result<Vec<String>, layer0::error::StateError> {
        self.0.list(scope, prefix).await
    }
    async fn search(
        &self,
        scope: &Scope,
        query: &str,
        limit: usize,
    ) -> Result<Vec<layer0::state::SearchResult>, layer0::error::StateError> {
        self.0.search(scope, query, limit).await
    }
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/// Comparison system prompt sent to the LLM in Step 6.
///
/// Instructs the model to produce a structured verdict by comparing research
/// findings against the existing decision text.
pub(crate) const COMPARISON_SYSTEM_PROMPT: &str = "\
You are an architectural decision analyst. Compare research findings against \
an existing decision and produce a structured verdict.

Rules:
- \"confirmed\" requires >= 3 supporting sources and 0 contradicting
- \"refined\" means the core decision holds but evidence/wording needs updating
- \"challenged\" requires >= 2 independent contradicting sources
- \"obsoleted\" means the entire decision space has been superseded
- Cite every claim with a source URL
- Do NOT hallucinate sources";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Static configuration shared by [`ResearchOperator`] and [`CompareOperator`].
#[derive(Debug, Clone)]
pub struct SweepOperatorConfig {
    /// Minimum time between sweeps of the same decision.
    ///
    /// A sweep that arrives sooner than this interval returns
    /// [`VerdictStatus::Skipped`] without performing any research.
    /// Default: 20 hours (allows daily sweeps with scheduling jitter).
    pub min_sweep_interval: Duration,

    /// Maximum research queries to execute per sweep run.
    ///
    /// Limits API spend. The adaptive plan step may generate more queries
    /// than this cap; excess queries are discarded.
    /// Default: 5.
    pub max_queries: usize,

    /// Maximum research results to store per sweep run.
    ///
    /// Results beyond this limit are dropped before artifact storage.
    /// Default: 20.
    pub max_artifacts: usize,

    /// Model identifier for plan generation and comparison LLM calls.
    ///
    /// Default: `"claude-sonnet-4-20250514"`.
    pub model: String,

    /// Maximum tokens for the comparison LLM response.
    ///
    /// Longer verdicts with detailed diffs may need a higher limit.
    /// Default: 4096.
    pub max_response_tokens: usize,
}

impl Default for SweepOperatorConfig {
    fn default() -> Self {
        Self {
            min_sweep_interval: Duration::from_secs(20 * 3600),
            max_queries: 5,
            max_artifacts: 20,
            model: "claude-sonnet-4-20250514".to_string(),
            max_response_tokens: 4096,
        }
    }
}

/// Build a keyword-based research query from the decision card text.
pub(crate) fn keyword_query(card_text: &str) -> String {
    format!(
        "{} agent architecture production systems frameworks best practices 2025 2026",
        card_text
            .split_whitespace()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ")
    )
}

/// Select the research processor tier based on budget and previous verdict.
pub(crate) fn select_processor(
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

const MAX_RESEARCH_RETRIES: usize = 3;
const BACKOFF_BASE_SECS: u64 = 1;
const BACKOFF_MULTIPLIER: u64 = 4;

// ---------------------------------------------------------------------------
// Internal retry helper (used by ResearchOperator)
// ---------------------------------------------------------------------------

/// Execute a research query with exponential-backoff retry on transient errors.
///
/// Retries up to [`MAX_RESEARCH_RETRIES`] times for [`SweepError::Transient`]
/// errors, with delays computed from [`BACKOFF_BASE_SECS`] and [`BACKOFF_MULTIPLIER`].
/// Permanent errors are returned immediately.
pub(crate) async fn research_with_retry_inner(
    provider: &dyn ResearchProvider,
    query: &str,
    processor: ProcessorTier,
) -> Result<Vec<ResearchResult>, SweepError> {
    let mut attempt: usize = 0;
    loop {
        match provider.search(query, processor.clone()).await {
            Ok(results) => return Ok(results),
            Err(e) if e.is_transient() && attempt < MAX_RESEARCH_RETRIES => {
                let delay_secs = BACKOFF_BASE_SECS
                    * BACKOFF_MULTIPLIER.pow(attempt as u32);
                tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// ResearchOperator
// ---------------------------------------------------------------------------

/// Operator that executes web research queries via the [`ResearchProvider`].
///
/// Dispatched by [`crate::workflow::run_sweep`] for step 4 of the sweep pipeline
/// (research). Declares [`Effect::WriteMemory`] for each research artifact (step 5)
/// rather than writing to state directly, preserving the effects boundary.
///
/// # Input
///
/// - `input.message` — the query string (text content).
/// - `input.metadata["processor"]` — [`ProcessorTier`] serialized as a JSON string
///   (`"base"`, `"core"`, or `"ultra"`). Defaults to `Base` if absent.
/// - `input.metadata["decision_id"]` — string identifier for the decision.
///
/// # Output
///
/// - `output.message` — JSON-serialized `Vec<ResearchResult>`, or
///   `"DECISION_NOT_FOUND:{id}"` if the provider returned [`SweepError::DecisionNotFound`].
/// - `output.effects` — one [`Effect::WriteMemory`] per research result (artifact storage).
pub struct ResearchOperator {
    /// Research provider backend (Parallel.ai or mock).
    provider: Box<dyn ResearchProvider>,
    /// Operator configuration.
    pub config: SweepOperatorConfig,
}

impl ResearchOperator {
    /// Create a new research operator.
    pub fn new(provider: Box<dyn ResearchProvider>, config: SweepOperatorConfig) -> Self {
        Self { provider, config }
    }
}

#[async_trait::async_trait]
impl Operator for ResearchOperator {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        // Extract query from the input message.
        let query = input
            .message
            .as_text()
            .ok_or_else(|| {
                OperatorError::NonRetryable(
                    "ResearchOperator: input.message must be text containing the query".into(),
                )
            })?
            .to_string();

        // Extract processor tier from metadata (defaults to Base).
        let processor: ProcessorTier = input
            .metadata
            .get("processor")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(ProcessorTier::Base);

        // Extract decision_id for artifact key generation.
        let decision_id = input
            .metadata
            .get("decision_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Execute research with exponential-backoff retry.
        let results =
            match research_with_retry_inner(&*self.provider, &query, processor).await {
                Ok(results) => results,
                Err(SweepError::DecisionNotFound(id)) => {
                    // Signal to the workflow that the decision was not found.
                    return Ok(OperatorOutput::new(
                        Content::text(format!("DECISION_NOT_FOUND:{id}")),
                        ExitReason::Complete,
                    ));
                }
                Err(SweepError::Transient(msg)) => return Err(OperatorError::Retryable(msg)),
                Err(SweepError::Permanent(msg)) => {
                    return Err(OperatorError::NonRetryable(msg));
                }
                Err(SweepError::LlmFailure(msg)) => return Err(OperatorError::Model(msg)),
                Err(SweepError::BudgetExhausted) => {
                    return Err(OperatorError::NonRetryable("budget exhausted".into()));
                }
            };

        // Truncate to the configured artifact limit.
        let results: Vec<_> = results.into_iter().take(self.config.max_artifacts).collect();

        // Declare WriteMemory effects for each research artifact.
        let scope = Scope::Custom("sweep".to_string());
        let mut effects: Vec<Effect> = Vec::with_capacity(results.len());

        for (i, result) in results.iter().enumerate() {
            let artifact_key = format!(
                "artifact:{}:{}:{}",
                decision_id,
                chrono::Utc::now().format("%Y%m%d"),
                i
            );
            if let Ok(value) = serde_json::to_value(result) {
                effects.push(Effect::WriteMemory {
                    scope: scope.clone(),
                    key: artifact_key,
                    value,
                    tier: Some(MemoryTier::Cold),
                    lifetime: Some(Lifetime::Durable),
                    content_kind: Some(ContentKind::Episodic),
                    salience: Some(0.3),
                    ttl: Some(DurationMs::from(Duration::from_secs(90 * 24 * 3600))),
                });
            }
        }

        // Serialize results as the output message.
        let results_json =
            serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
        let mut output = OperatorOutput::new(Content::text(results_json), ExitReason::Complete);
        output.effects = effects;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Verdict JSON extraction helper
// ---------------------------------------------------------------------------

/// Strip an optional `` ```json `` / `` ``` `` fence and return the inner JSON slice.
///
/// Many LLMs wrap JSON responses in a fenced code block. If no fence is found,
/// the input is returned unchanged.
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

/// Operator that compares research findings against a decision via an LLM [`Provider`].
///
/// Dispatched by [`crate::workflow::run_sweep`] for step 6 of the sweep pipeline
/// (comparison). Declares [`Effect::WriteMemory`] for sweep metadata and a delta
/// entry (step 7) rather than writing to state directly.
///
/// # Input
///
/// - `input.message` — JSON-serialized `Vec<ResearchResult>` produced by [`ResearchOperator`].
/// - `input.metadata["decision_id"]` — string identifier for the decision under review.
///
/// # Output
///
/// - `output.message` — JSON-serialized [`SweepVerdict`].
/// - `output.effects` — [`Effect::WriteMemory`] for the sweep metadata key
///   (`meta:{id}:last_sweep`) and a delta key (`delta:{id}:{timestamp}`).
pub struct CompareOperator<P: Provider> {
    /// LLM provider used for verdict comparison.
    llm: P,
    /// Operator configuration.
    pub config: SweepOperatorConfig,
}

impl<P: Provider> CompareOperator<P> {
    /// Create a new compare operator.
    ///
    /// Use [`neuron_provider_anthropic::AnthropicProvider::with_auth`] to wire
    /// pi coding agent OAuth credentials directly.
    pub fn new(llm: P, config: SweepOperatorConfig) -> Self {
        Self { llm, config }
    }
}

#[async_trait::async_trait]
impl<P: Provider + 'static> Operator for CompareOperator<P> {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        // Extract the research JSON from the input message.
        let research_json = input
            .message
            .as_text()
            .ok_or_else(|| {
                OperatorError::NonRetryable(
                    "CompareOperator: input.message must be text containing research JSON".into(),
                )
            })?
            .to_string();

        // Extract decision_id for effect keys and provider call.
        let decision_id = input
            .metadata
            .get("decision_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Build an LLM request using the configured model.
        let user_content = format!(
            "Decision ID: {}\n\n<research>\n{}\n</research>",
            decision_id, research_json
        );
        let request = ProviderRequest {
            model: Some(self.config.model.clone()),
            messages: vec![ProviderMessage {
                role: Role::User,
                content: vec![ContentPart::Text { text: user_content }],
            }],
            tools: vec![],
            max_tokens: Some(self.config.max_response_tokens as u32),
            temperature: None,
            system: Some(COMPARISON_SYSTEM_PROMPT.to_string()),
            extra: serde_json::Value::Null,
        };

        let response = self.llm.complete(request).await.map_err(|e| match e {
            ProviderError::RateLimited => {
                OperatorError::Retryable("rate limited by LLM provider".into())
            }
            ProviderError::TransientError { message, .. } => {
                OperatorError::Retryable(message)
            }
            ProviderError::AuthFailed(msg) => {
                OperatorError::NonRetryable(format!("auth: {msg}"))
            }
            other => OperatorError::Model(other.to_string()),
        })?;

        // Extract text content from the LLM response.
        let raw = response
            .content
            .iter()
            .find_map(|p| {
                if let ContentPart::Text { text } = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("");

        // Parse the JSON verdict (may be fenced ```json ... ```).
        let verdict: SweepVerdict = {
            let json_str = extract_json_block(raw);
            serde_json::from_str(json_str).map_err(|e| {
                OperatorError::Model(format!(
                    "CompareOperator: failed to parse LLM verdict JSON: {e}\nRaw: {raw}"
                ))
            })?
        };

        // Declare WriteMemory effects for sweep metadata and delta.
        let scope = Scope::Custom("sweep".to_string());
        let meta_key = format!("meta:{}:last_sweep", decision_id);
        let meta = SweepMeta {
            swept_at: chrono::Utc::now().to_rfc3339(),
            verdict: verdict.status.clone(),
            cost_usd: verdict.cost_usd,
        };

        let mut effects: Vec<Effect> = Vec::with_capacity(2);

        if let Ok(meta_value) = serde_json::to_value(&meta) {
            effects.push(Effect::WriteMemory {
                scope: scope.clone(),
                key: meta_key,
                value: meta_value,
                tier: Some(MemoryTier::Hot),
                lifetime: None,
                content_kind: None,
                salience: Some(1.0),
                ttl: None,
            });
        }

        let delta_key = format!(
            "delta:{}:{}",
            decision_id,
            chrono::Utc::now().format("%Y%m%dT%H%M%S")
        );
        let delta_value = serde_json::json!({
            "status": &verdict.status,
            "confidence": verdict.confidence,
            "num_supporting": verdict.num_supporting,
            "num_contradicting": verdict.num_contradicting,
            "narrative": &verdict.narrative,
        });
        effects.push(Effect::WriteMemory {
            scope,
            key: delta_key,
            value: delta_value,
            tier: Some(MemoryTier::Hot),
            lifetime: None,
            content_kind: None,
            salience: Some(0.8),
            ttl: None,
        });

        // Serialize the verdict as the output message.
        let verdict_json =
            serde_json::to_string(&verdict).unwrap_or_else(|_| "{}".to_string());
        let mut output = OperatorOutput::new(Content::text(verdict_json), ExitReason::Complete);
        output.effects = effects;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;
    use crate::types::{EvidenceItem, EvidenceStance, ProcessorTier};
    use neuron_turn::types::{ProviderResponse, StopReason, TokenUsage};
    use std::sync::Arc;
    use layer0::operator::TriggerType;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn dummy_verdict(decision_id: &str) -> SweepVerdict {
        SweepVerdict {
            decision_id: decision_id.to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.9,
            num_supporting: 3,
            num_contradicting: 0,
            cost_usd: 0.10,
            processor: ProcessorTier::Base,
            duration_secs: 2.0,
            swept_at: "2026-03-04T18:00:00Z".to_string(),
            evidence: vec![EvidenceItem {
                source_url: "https://example.com".into(),
                summary: "Supports decision".into(),
                stance: EvidenceStance::Supporting,
                retrieved_at: "2026-03-04T00:00:00Z".into(),
            }],
            narrative: "Confirmed by research".to_string(),
            proposed_diff: None,
        }
    }

    fn dummy_result() -> crate::provider::ResearchResult {
        crate::provider::ResearchResult {
            url: "https://example.com/paper".to_string(),
            title: "Agent Architecture 2026".to_string(),
            snippet: "Key findings about agent systems.".to_string(),
            retrieved_at: "2026-03-04T17:00:00Z".to_string(),
        }
    }


    /// Shared mock LLM provider that returns a fixed verdict as JSON.
    struct MockLlmProvider {
        verdict: SweepVerdict,
    }

    impl Provider for MockLlmProvider {
        fn complete(
            &self,
            _request: ProviderRequest,
        ) -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>>
               + Send {
            let json = serde_json::to_string(&self.verdict).expect("verdict serializable");
            async move {
                Ok(ProviderResponse {
                    content: vec![ContentPart::Text { text: json }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: "mock".into(),
                    cost: None,
                    truncated: None,
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // select_processor — all 5 cases from the spec
    // -----------------------------------------------------------------------

    #[test]
    fn select_processor_challenged_high_budget_returns_ultra() {
        let tier = select_processor(6.0, 10.0, Some(&VerdictStatus::Challenged));
        assert_eq!(tier, ProcessorTier::Ultra, "budget_ratio=0.6 > 0.5 → Ultra");
    }

    #[test]
    fn select_processor_challenged_low_budget_returns_core() {
        let tier = select_processor(4.0, 10.0, Some(&VerdictStatus::Challenged));
        assert_eq!(tier, ProcessorTier::Core, "budget_ratio=0.4 ≤ 0.5 → Core");
    }

    #[test]
    fn select_processor_refined_returns_core_regardless_of_budget() {
        for remaining in [0.1, 5.0, 9.9] {
            let tier =
                select_processor(remaining, 10.0, Some(&VerdictStatus::Refined));
            assert_eq!(
                tier,
                ProcessorTier::Core,
                "Refined → Core at budget {remaining}"
            );
        }
    }

    #[test]
    fn select_processor_default_high_budget_returns_core() {
        // None, Confirmed, Obsoleted, Skipped all take the default branch
        for verdict in [
            None,
            Some(&VerdictStatus::Confirmed),
            Some(&VerdictStatus::Obsoleted),
            Some(&VerdictStatus::Skipped),
        ] {
            let tier = select_processor(9.0, 10.0, verdict);
            assert_eq!(
                tier,
                ProcessorTier::Core,
                "budget_ratio=0.9 > 0.8 → Core for verdict {:?}",
                verdict
            );
        }
    }

    #[test]
    fn select_processor_default_low_budget_returns_base() {
        for verdict in [None, Some(&VerdictStatus::Confirmed)] {
            let tier = select_processor(7.0, 10.0, verdict);
            assert_eq!(
                tier,
                ProcessorTier::Base,
                "budget_ratio=0.7 ≤ 0.8 → Base for verdict {:?}",
                verdict
            );
        }
    }

    #[test]
    fn select_processor_zero_total_budget_returns_base() {
        let tier = select_processor(5.0, 0.0, Some(&VerdictStatus::Challenged));
        assert_eq!(tier, ProcessorTier::Base, "zero total budget → Base");
    }

    #[test]
    fn select_processor_challenged_exactly_50pct_returns_core() {
        // budget_ratio == 0.5 is NOT > 0.5, so falls to Core
        let tier = select_processor(5.0, 10.0, Some(&VerdictStatus::Challenged));
        assert_eq!(tier, ProcessorTier::Core);
    }

    #[test]
    fn select_processor_default_exactly_80pct_returns_base() {
        // budget_ratio == 0.8 is NOT > 0.8, so falls to Base
        let tier = select_processor(8.0, 10.0, None);
        assert_eq!(tier, ProcessorTier::Base);
    }

    // -----------------------------------------------------------------------
    // research_with_retry_inner — free function
    // -----------------------------------------------------------------------

    #[tokio::test(flavor = "current_thread")]
    async fn research_with_retry_inner_succeeds_on_first_attempt() {
        let mock = MockProvider::new(vec![dummy_result()]);
        let result = research_with_retry_inner(&mock, "query", ProcessorTier::Base)
            .await
            .expect("should succeed");
        assert_eq!(result.len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn research_with_retry_inner_fails_twice_then_succeeds() {
        tokio::time::pause();

        let mock = MockProvider::new_failing(2, vec![dummy_result()]);
        let handle = tokio::spawn(async move {
            research_with_retry_inner(&mock, "test query", ProcessorTier::Base).await
        });

        // Advance past the 1 s backoff after attempt 0.
        tokio::time::advance(Duration::from_secs(2)).await;
        // Advance past the 4 s backoff after attempt 1.
        tokio::time::advance(Duration::from_secs(5)).await;

        let result = handle.await.expect("task panicked");
        assert!(result.is_ok(), "expected success after 2 failures: {:?}", result);
        assert_eq!(result.unwrap().len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn research_with_retry_inner_permanent_error_fails_immediately() {
        struct PermanentErrorProvider;

        #[async_trait::async_trait]
        impl ResearchProvider for PermanentErrorProvider {
            async fn search(
                &self,
                _query: &str,
                _processor: ProcessorTier,
            ) -> Result<Vec<ResearchResult>, SweepError> {
                Err(SweepError::Permanent("400 bad request".to_string()))
            }
        }

        let err = research_with_retry_inner(&PermanentErrorProvider, "q", ProcessorTier::Base)
            .await
            .unwrap_err();
        assert!(!err.is_transient(), "Permanent error should not be retried");
        assert!(matches!(err, SweepError::Permanent(_)), "expected Permanent, got {:?}", err);
    }

    // -----------------------------------------------------------------------
    // ResearchOperator — Operator trait impl
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn research_operator_returns_results_json() {
        let mock = MockProvider::new(vec![dummy_result()]);
        let op = ResearchOperator::new(Box::new(mock), SweepOperatorConfig::default());

        let mut input = OperatorInput::new(Content::text("test query"), TriggerType::Task);
        input.metadata = serde_json::json!({
            "processor": "base",
            "decision_id": "topic-3b",
        });

        let output = op.execute(input).await.expect("execute should succeed");
        let text = output.message.as_text().expect("output should be text");
        let results: Vec<ResearchResult> =
            serde_json::from_str(text).expect("output should be valid JSON");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://example.com/paper");
    }

    #[tokio::test]
    async fn research_operator_declares_write_memory_effects() {
        let mock = MockProvider::new(vec![dummy_result(), dummy_result()]);
        let op = ResearchOperator::new(Box::new(mock), SweepOperatorConfig::default());

        let mut input = OperatorInput::new(Content::text("test query"), TriggerType::Task);
        input.metadata = serde_json::json!({
            "processor": "base",
            "decision_id": "topic-3b",
        });

        let output = op.execute(input).await.expect("execute should succeed");

        // Each result should produce one WriteMemory effect.
        assert_eq!(
            output.effects.len(),
            2,
            "expected 2 WriteMemory effects (one per result)"
        );
        for effect in &output.effects {
            assert!(
                matches!(effect, Effect::WriteMemory { .. }),
                "all effects must be WriteMemory, got {:?}",
                effect
            );
        }
    }

    #[tokio::test]
    async fn research_operator_returns_no_effects_for_empty_results() {
        let mock = MockProvider::new(vec![]);
        let op = ResearchOperator::new(Box::new(mock), SweepOperatorConfig::default());

        let mut input = OperatorInput::new(Content::text("test query"), TriggerType::Task);
        input.metadata = serde_json::json!({
            "processor": "base",
            "decision_id": "topic-3b",
        });

        let output = op.execute(input).await.expect("execute should succeed");
        assert!(
            output.effects.is_empty(),
            "no effects when research returns empty"
        );
        let text = output.message.as_text().unwrap_or("");
        let results: Vec<ResearchResult> = serde_json::from_str(text).unwrap_or_default();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn research_operator_returns_decision_not_found_sentinel() {
        struct NotFoundProvider;

        #[async_trait::async_trait]
        impl ResearchProvider for NotFoundProvider {
            async fn search(
                &self,
                _query: &str,
                _processor: ProcessorTier,
            ) -> Result<Vec<ResearchResult>, SweepError> {
                Err(SweepError::DecisionNotFound("D9X".to_string()))
            }
        }

        let op = ResearchOperator::new(Box::new(NotFoundProvider), SweepOperatorConfig::default());
        let mut input = OperatorInput::new(Content::text("q"), TriggerType::Task);
        input.metadata = serde_json::json!({"decision_id": "D9X"});

        let output = op.execute(input).await.expect("execute should not error");
        let text = output.message.as_text().unwrap_or("");
        assert!(
            text.starts_with("DECISION_NOT_FOUND:"),
            "expected sentinel, got: {text}"
        );
        assert!(
            output.effects.is_empty(),
            "no effects when decision not found"
        );
    }

    // -----------------------------------------------------------------------
    // CompareOperator — Operator trait impl
    // -----------------------------------------------------------------------


    #[tokio::test]
    async fn compare_operator_returns_verdict_json() {
        let expected = dummy_verdict("topic-3b");
        let mock = MockLlmProvider { verdict: expected.clone() };
        let op = CompareOperator::new(mock, SweepOperatorConfig::default());

        let research = serde_json::to_string(&vec![dummy_result()]).unwrap();
        let mut input = OperatorInput::new(Content::text(research), TriggerType::Task);
        input.metadata = serde_json::json!({"decision_id": "topic-3b"});

        let output = op.execute(input).await.expect("execute should succeed");
        let text = output.message.as_text().expect("output should be text");
        let verdict: SweepVerdict =
            serde_json::from_str(text).expect("output should be valid SweepVerdict JSON");
        assert_eq!(verdict.status, VerdictStatus::Confirmed);
    }

    #[tokio::test]
    async fn compare_operator_declares_write_memory_effects() {
        let mock = MockLlmProvider { verdict: dummy_verdict("topic-3b") };
        let op = CompareOperator::new(mock, SweepOperatorConfig::default());

        let research = serde_json::to_string(&vec![dummy_result()]).unwrap();
        let mut input = OperatorInput::new(Content::text(research), TriggerType::Task);
        input.metadata = serde_json::json!({"decision_id": "topic-3b"});

        let output = op.execute(input).await.expect("execute should succeed");

        // Expect exactly 2 effects: meta and delta.
        assert_eq!(
            output.effects.len(),
            2,
            "expected 2 WriteMemory effects (meta + delta)"
        );
        for effect in &output.effects {
            assert!(
                matches!(effect, Effect::WriteMemory { .. }),
                "all effects must be WriteMemory, got {:?}",
                effect
            );
        }

        // Verify meta key pattern.
        if let Effect::WriteMemory { key, .. } = &output.effects[0] {
            assert!(
                key.contains("last_sweep"),
                "first effect should be the meta key, got: {key}"
            );
        }
        // Verify delta key pattern.
        if let Effect::WriteMemory { key, .. } = &output.effects[1] {
            assert!(
                key.starts_with("delta:topic-3b:"),
                "second effect should be a delta key, got: {key}"
            );
        }
    }
}
