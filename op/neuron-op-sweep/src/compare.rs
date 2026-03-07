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
use std::time::Duration;

use layer0::state::MemoryTier;
use layer0::{Content, Effect, ExitReason, Operator, OperatorError, OperatorInput, OperatorOutput, Scope};
use neuron_orch_kit::ScopedState;
use neuron_turn::provider::{Provider, ProviderError};
use neuron_turn::types::{ContentPart, ProviderMessage, ProviderRequest, Role};

use crate::provider::CompareInput;
use crate::types::{ProcessorTier, SweepMeta, SweepVerdict, VerdictStatus};

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/// Comparison system prompt sent to the LLM.
///
/// Instructs the model to compare research findings against the existing
/// decision text (all three sections supplied in the user message).
pub(crate) const COMPARISON_SYSTEM_PROMPT: &str = "\
You are an architectural decision analyst. You will receive:
1. A <decision> section containing the current decision text
2. A <prior_findings> section with previous sweep results (may be empty)
3. A <research> section with new research findings

Compare the research findings against the existing decision and produce a \
structured JSON verdict.

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
}

impl Default for CompareConfig {
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
            ProviderError::TransientError { message, .. } => OperatorError::Retryable(message),
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

        // Write own-scope sweep metadata directly (not an effect — same scope).
        let meta_key = format!("meta:{}:last_sweep", decision_id);
        let meta = SweepMeta {
            swept_at: chrono::Utc::now().to_rfc3339(),
            verdict: verdict.status.clone(),
            cost_usd: verdict.cost_usd,
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
    use neuron_turn::provider::{Provider, ProviderError};
    use neuron_turn::types::{ProviderResponse, StopReason, TokenUsage};
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
        ) -> impl std::future::Future<Output = Result<ProviderResponse, ProviderError>> + Send
        {
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
        let mock = MockLlmProvider { verdict: expected.clone() };
        let state = Arc::new(MockScopedState::new());
        let op = CompareOperator::new(mock, state, CompareConfig::default());

        let compare_in = crate::provider::CompareInput {
            research_results: vec![dummy_result()],
            decision_id: "topic-3b".to_string(),
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
        let mock = MockLlmProvider { verdict: dummy_verdict("topic-3b") };
        let state = Arc::new(MockScopedState::new());
        let state_ref = state.clone();
        let op = CompareOperator::new(mock, state, CompareConfig::default());

        let compare_in = crate::provider::CompareInput {
            research_results: vec![dummy_result()],
            decision_id: "topic-3b".to_string(),
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
        let mock = MockLlmProvider { verdict: dummy_verdict("topic-3b") };
        let state = Arc::new(MockScopedState::new());
        let op = CompareOperator::new(mock, state, CompareConfig::default());

        let compare_in = crate::provider::CompareInput {
            research_results: vec![dummy_result()],
            decision_id: "topic-3b".to_string(),
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
}
