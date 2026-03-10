//! Sweep operator — comparison step.
//!
//! [`CompareOperator`] implements the comparison step of the sweep pipeline.
//! Research is orchestrator-owned: callers supply research results to
//! the sweep cycle function directly rather than dispatching a research
//! operator.
//!
//! [`CompareOperator`] implements the [`Operator`] trait, allowing it to be
//! dispatched via an [`Orchestrator`] without violating the effects boundary.
//! Use the sweep cycle composition function to sequence context assembly
//! and comparison.
//!
//! [`Orchestrator`]: layer0::Orchestrator

use std::sync::Arc;
use std::time::{Duration, Instant};

use layer0::context::{Message, Role};
use layer0::operator::OperatorMetadata;
use layer0::state::MemoryTier;
use layer0::{Content, Effect, ExitReason, Operator, OperatorError, OperatorInput, OperatorOutput, Scope};
use skg_orch_kit::ScopedState;
use skg_turn::infer::InferRequest;
use skg_turn::provider::{Provider, ProviderError};
use rust_decimal::Decimal;
use tracing::{debug, info};

use crate::provider::CompareInput;
use crate::types::{ProcessorTier, SweepMeta, SweepVerdict, VerdictStatus};

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/// Comparison system prompt sent to the LLM.
///
/// Instructs the model to compare research findings against the existing
/// decision text (all three sections supplied in the user message).
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
- status MUST be one of: confirmed, refined, challenged, obsoleted, skipped
- confidence MUST be a float between 0.0 and 1.0 (NOT a string like "high")
- decision_id MUST match the Decision ID from the input
- Cite every claim with a source URL
- Do NOT hallucinate sources
- Return ONLY the JSON object inside ```json fences, no other text
- If no research findings are provided, return status "skipped" with confidence 0.0"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`CompareOperator`].
#[derive(Debug, Clone)]
pub struct CompareConfig {
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

    /// When true, always use Ultra processor regardless of budget or prior verdict.
    /// Default: true.
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

/// Select the research processor tier based on budget and previous verdict.
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

/// Select processor tier, respecting the force_ultra override.
///
/// When `force_ultra` is true, always returns [`ProcessorTier::Ultra`].
/// Otherwise delegates to [`select_processor`] for adaptive selection.
pub fn resolve_processor(
    config: &CompareConfig,
    budget_remaining_usd: f64,
    budget_total_usd: f64,
    previous_verdict: Option<&VerdictStatus>,
) -> ProcessorTier {
    if config.force_ultra {
        ProcessorTier::Ultra
    } else {
        select_processor(budget_remaining_usd, budget_total_usd, previous_verdict)
    }
}

// ---------------------------------------------------------------------------
// Verdict JSON extraction helper
// ---------------------------------------------------------------------------

/// Strip an optional ` ```json ` / ` ``` ` fence and return the inner JSON slice.
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
/// Context assembly is **turn-owned**: the operator reads the decision card and
/// prior sweep deltas from the scoped state during `execute()`, then combines
/// them with the research results supplied in the input message.
///
/// Dispatched by the sweep cycle composition function for the comparison step.
///
/// # State access
///
/// - `card:{id}` — read-only. Decision card text.
/// - `delta:{id}:*` — read-only. Prior sweep findings listed by prefix.
/// - `meta:{id}:last_sweep` — written directly (own-scope).
///
/// # Cross-scope effect
///
/// Declares one [`Effect::WriteMemory`] for the delta entry into
/// `Scope::Custom("sweep-results")`.
///
/// # Input
///
/// - `input.message` — JSON-serialized [`CompareInput`] supplied by the caller.
///
/// # Output
///
/// - `output.message` — JSON-serialized [`SweepVerdict`].
/// - `output.effects` — [`Effect::WriteMemory`] for the cross-scope delta entry.
pub struct CompareOperator<P: Provider> {
    /// LLM provider used for verdict comparison.
    llm: P,
    /// Scoped state for reading decision cards and prior findings, and writing
    /// own-scope sweep metadata.
    state: Arc<dyn ScopedState>,
    /// Operator configuration.
    pub config: CompareConfig,
}

impl<P: Provider> CompareOperator<P> {
    /// Create a new compare operator.
    ///
    /// The `state` is scoped to this operator's partition. Own-scope metadata
    /// writes (`meta:{id}:last_sweep`) are performed directly. Cross-scope
    /// delta writes are declared as [`Effect::WriteMemory`].
    pub fn new(llm: P, state: Arc<dyn ScopedState>, config: CompareConfig) -> Self {
        Self { llm, state, config }
    }
}

#[async_trait::async_trait]
impl<P: Provider + 'static> Operator for CompareOperator<P> {
    async fn execute(&self, input: OperatorInput) -> Result<OperatorOutput, OperatorError> {
        let start = Instant::now();

        // Parse the typed input — both research results and decision_id come
        // from the CompareInput struct embedded in the message.
        let msg_text = input
            .message
            .as_text()
            .ok_or_else(|| {
                OperatorError::NonRetryable(
                    "CompareOperator: input.message must be text containing CompareInput JSON"
                        .into(),
                )
            })?;
        let compare_in: CompareInput = serde_json::from_str(msg_text).map_err(|e| {
            OperatorError::NonRetryable(format!(
                "CompareOperator: failed to parse CompareInput: {e}"
            ))
        })?;
        let research_json = serde_json::to_string(&compare_in.research_results)
            .unwrap_or_else(|_| "[]".to_string());
        let decision_id = compare_in.decision_id;
        let num_research = compare_in.research_results.len();

        info!(
            decision_id = %decision_id,
            num_research_results = num_research,
            query = compare_in.query.as_deref().unwrap_or(""),
            query_angle = compare_in.query_angle.as_deref().unwrap_or(""),
            model = %self.config.model,
            "compare: starting LLM comparison"
        );

        // --- Turn-owned context assembly ---
        // Read the decision card from own-scope state.
        let decision_text_owned: String = match self
            .state
            .read(&format!("card:{}", decision_id))
            .await
            .unwrap_or(None)
        {
            Some(val) => match val {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            },
            None => "[Decision text not available]".to_string(),
        };
        let decision_text = decision_text_owned.as_str();

        // Read prior sweep deltas by prefix, collect their text content.
        let delta_keys = self
            .state
            .list(&format!("delta:{}:", decision_id))
            .await
            .unwrap_or_default();
        let mut prior_findings: Vec<String> = Vec::with_capacity(delta_keys.len());
        for key in &delta_keys {
            if let Ok(Some(val)) = self.state.read(key).await {
                let text = match val {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                };
                prior_findings.push(text);
            }
        }
        let prior_findings_json =
            serde_json::to_string(&prior_findings).unwrap_or_else(|_| "[]".to_string());

        // Build an LLM request with full context.
        let user_content = format!(
            "Decision ID: {}\n\n<decision>\n{}\n</decision>\n\n<prior_findings>\n{}\n</prior_findings>\n\n<research>\n{}\n</research>",
            decision_id, decision_text, prior_findings_json, research_json
        );

        debug!(
            decision_id = %decision_id,
            prompt_chars = user_content.len(),
            system_prompt_chars = COMPARISON_SYSTEM_PROMPT.len(),
            "compare: full user prompt\n{user_content}"
        );
        let request = InferRequest::new(vec![Message::new(Role::User, Content::text(user_content))])
            .with_model(self.config.model.clone())
            .with_system(COMPARISON_SYSTEM_PROMPT)
            .with_max_tokens(self.config.max_response_tokens as u32);

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

        // Extract text content from the LLM response.
        let raw = response.text().unwrap_or("");

        let llm_elapsed = start.elapsed();
        info!(
            decision_id = %decision_id,
            tokens_in = response.usage.input_tokens,
            tokens_out = response.usage.output_tokens,
            cache_read = response.usage.cache_read_tokens,
            cache_create = response.usage.cache_creation_tokens,
            cost_usd = %response.cost.unwrap_or(Decimal::ZERO),
            model = %response.model,
            duration_ms = llm_elapsed.as_millis() as u64,
            "compare: LLM response received"
        );
        debug!(
            decision_id = %decision_id,
            raw_len = raw.len(),
            "compare: raw LLM output\n{raw}"
        );

        // Parse the JSON verdict (may be fenced ```json ... ```).
        let verdict: SweepVerdict = {
            let json_str = extract_json_block(raw);
            serde_json::from_str(json_str).map_err(|e| {
                OperatorError::Model(format!(
                    "CompareOperator: failed to parse LLM verdict JSON: {e}\nRaw: {raw}"
                ))
            })?
        };
        let verdict = SweepVerdict {
            research_inputs: compare_in.research_results,
            query: compare_in.query.unwrap_or_default(),
            query_angle: compare_in.query_angle.unwrap_or_default(),
            ..verdict
        };

        // Write own-scope sweep metadata directly (not an effect — same scope).
        let meta_key = format!("meta:{}:last_sweep", decision_id);
        let meta = SweepMeta {
            swept_at: chrono::Utc::now().to_rfc3339(),
            verdict: verdict.status.clone(),
            cost_usd: verdict.cost_usd,
            query: verdict.query.clone(),
            query_angle: verdict.query_angle.clone(),
            processor: verdict.processor.clone(),
        };
        if let Ok(meta_value) = serde_json::to_value(&meta) {
            // Best-effort: ignore write errors on own-scope metadata.
            let _ = self.state.write(&meta_key, meta_value).await;
        }

        // Declare a cross-scope WriteMemory effect for the delta entry.
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
        let effects = vec![Effect::WriteMemory {
            scope: Scope::Custom("sweep-results".to_string()),
            key: delta_key,
            value: delta_value,
            tier: Some(MemoryTier::Hot),
            lifetime: None,
            content_kind: None,
            salience: Some(0.8),
            ttl: None,
        }];

        // Serialize the verdict as the output message.
        let verdict_json =
            serde_json::to_string(&verdict).unwrap_or_else(|_| "{}".to_string());
        let mut output = OperatorOutput::new(Content::text(verdict_json), ExitReason::Complete);
        output.effects = effects;

        // Populate metadata from the LLM response (not LLM-generated guesses).
        let total_elapsed = start.elapsed();
        let duration = layer0::duration::DurationMs::from(total_elapsed);
        let mut metadata = OperatorMetadata::default();
        metadata.tokens_in = response.usage.input_tokens;
        metadata.tokens_out = response.usage.output_tokens;
        metadata.cost = response.cost.unwrap_or(Decimal::ZERO);
        metadata.turns_used = 1;
        metadata.duration = duration;
        output.metadata = metadata;

        info!(
            decision_id = %decision_id,
            status = ?verdict.status,
            confidence = verdict.confidence,
            num_supporting = verdict.num_supporting,
            num_contradicting = verdict.num_contradicting,
            cost_usd = %output.metadata.cost,
            tokens_in = output.metadata.tokens_in,
            tokens_out = output.metadata.tokens_out,
            duration_ms = total_elapsed.as_millis() as u64,
            "compare: verdict produced"
        );

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EvidenceItem, EvidenceStance, ProcessorTier};
    use layer0::operator::TriggerType;
    use layer0::StateError;
    use skg_turn::test_utils::TestProvider;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // -----------------------------------------------------------------------
    // Mock ScopedState
    // -----------------------------------------------------------------------

    /// Mock scoped state backed by an in-memory map. Records write calls.
    struct MockScopedState {
        data: Mutex<HashMap<String, serde_json::Value>>,
        write_calls: Mutex<Vec<String>>,
    }

    impl MockScopedState {
        fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
                write_calls: Mutex::new(Vec::new()),
            }
        }

        #[allow(dead_code)]
        fn with_entry(key: &str, value: serde_json::Value) -> Self {
            let s = Self::new();
            s.data.lock().unwrap().insert(key.to_string(), value);
            s
        }

        fn recorded_writes(&self) -> Vec<String> {
            self.write_calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl ScopedState for MockScopedState {
        async fn read(&self, key: &str) -> Result<Option<serde_json::Value>, StateError> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }

        async fn write(
            &self,
            key: &str,
            value: serde_json::Value,
        ) -> Result<(), StateError> {
            self.write_calls.lock().unwrap().push(key.to_string());
            self.data.lock().unwrap().insert(key.to_string(), value);
            Ok(())
        }

        async fn delete(&self, key: &str) -> Result<(), StateError> {
            self.data.lock().unwrap().remove(key);
            Ok(())
        }

        async fn list(&self, prefix: &str) -> Result<Vec<String>, StateError> {
            let keys: Vec<String> = self
                .data
                .lock()
                .unwrap()
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect();
            Ok(keys)
        }

        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<layer0::SearchResult>, StateError> {
            Ok(vec![])
        }
    }

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
                title: String::new(),
                summary: "Supports decision".into(),
                stance: EvidenceStance::Supporting,
                retrieved_at: "2026-03-04T00:00:00Z".into(),
            }],
            narrative: "Confirmed by research".to_string(),
            proposed_diff: None,
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),
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

    /// Build a TestProvider that returns the serialized verdict JSON.
    fn verdict_provider(verdict: &SweepVerdict) -> TestProvider {
        let json = serde_json::to_string(verdict).expect("verdict serializable");
        let provider = TestProvider::new();
        provider.respond_with_text(&json);
        provider
    }

    // -----------------------------------------------------------------------
    // select_processor — all 8 cases
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
            let tier = select_processor(remaining, 10.0, Some(&VerdictStatus::Refined));
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
    // CompareOperator — Operator trait impl
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn compare_operator_returns_verdict_json() {
        let expected = dummy_verdict("topic-3b");
        let mock = verdict_provider(&expected);
        let state = Arc::new(MockScopedState::new());
        let op = CompareOperator::new(mock, state, CompareConfig::default());

        let compare_in = crate::provider::CompareInput {
            research_results: vec![dummy_result()],
            decision_id: "topic-3b".to_string(),
            query: None,
            query_angle: None,
        };
        let msg = serde_json::to_string(&compare_in).unwrap();
        let input = OperatorInput::new(Content::text(msg), TriggerType::Task);

        let output = op.execute(input).await.expect("execute should succeed");
        let text = output.message.as_text().expect("output should be text");
        let verdict: SweepVerdict =
            serde_json::from_str(text).expect("output should be valid SweepVerdict JSON");
        assert_eq!(verdict.status, VerdictStatus::Confirmed);
    }

    #[tokio::test]
    async fn compare_operator_writes_meta_to_own_scope() {
        let mock = verdict_provider(&dummy_verdict("topic-3b"));
        let state = Arc::new(MockScopedState::new());
        let state_ref = state.clone();
        let op = CompareOperator::new(mock, state, CompareConfig::default());

        let compare_in = crate::provider::CompareInput {
            research_results: vec![dummy_result()],
            decision_id: "topic-3b".to_string(),
            query: None,
            query_angle: None,
        };
        let msg = serde_json::to_string(&compare_in).unwrap();
        let input = OperatorInput::new(Content::text(msg), TriggerType::Task);

        op.execute(input).await.expect("execute should succeed");

        // Verify the meta key was written to own-scope state.
        let writes = state_ref.recorded_writes();
        assert!(
            writes.iter().any(|k| k.contains("meta:topic-3b:last_sweep")),
            "expected meta:topic-3b:last_sweep in write calls, got: {:?}",
            writes
        );
    }

    #[tokio::test]
    async fn compare_operator_declares_delta_effect() {
        let mock = verdict_provider(&dummy_verdict("topic-3b"));
        let state = Arc::new(MockScopedState::new());
        let op = CompareOperator::new(mock, state, CompareConfig::default());

        let compare_in = crate::provider::CompareInput {
            research_results: vec![dummy_result()],
            decision_id: "topic-3b".to_string(),
            query: None,
            query_angle: None,
        };
        let msg = serde_json::to_string(&compare_in).unwrap();
        let input = OperatorInput::new(Content::text(msg), TriggerType::Task);

        let output = op.execute(input).await.expect("execute should succeed");

        // Expect exactly 1 cross-scope effect: the delta WriteMemory.
        assert_eq!(
            output.effects.len(),
            1,
            "expected exactly 1 WriteMemory effect (delta), got {}",
            output.effects.len()
        );

        let effect = &output.effects[0];
        assert!(
            matches!(effect, Effect::WriteMemory { .. }),
            "effect must be WriteMemory, got {:?}",
            effect
        );

        if let Effect::WriteMemory { scope, key, .. } = effect {
            assert_eq!(
                scope,
                &Scope::Custom("sweep-results".to_string()),
                "delta effect must target sweep-results scope"
            );
            assert!(
                key.starts_with("delta:topic-3b:"),
                "delta key must start with delta:topic-3b:, got: {key}"
            );
        }
    }
    #[test]
    fn resolve_processor_force_ultra_ignores_budget() {
        let config = CompareConfig { force_ultra: true, ..CompareConfig::default() };
        assert_eq!(
            resolve_processor(&config, 0.0, 10.0, Some(&VerdictStatus::Confirmed)),
            ProcessorTier::Ultra,
        );
    }

    #[test]
    fn resolve_processor_adaptive_when_not_forced() {
        let config = CompareConfig { force_ultra: false, ..CompareConfig::default() };
        assert_eq!(
            resolve_processor(&config, 1.0, 10.0, Some(&VerdictStatus::Confirmed)),
            ProcessorTier::Base,
        );
    }

}
