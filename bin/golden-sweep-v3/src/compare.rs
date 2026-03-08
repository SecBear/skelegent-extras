//! Per-decision comparison — implemented with context-engine primitives.
//!
//! # v2 vs v3
//!
//! | Aspect | v2 | v3 |
//! |--------|----|----|
//! | Interface | `CompareOperator` implementing `Operator` | `compare_decision()` free function |
//! | Context assembly | `&mut Vec<Message>` built manually | `Context::inject_system()` + `inject_message()` |
//! | Inference | Direct `InferRequest` construction | `ctx.compile(&config).infer(provider)` |
//! | State reads | Inside `Operator::execute()` | Caller reads; raw text passed in |
//! | Dispatch | `dispatch_typed` through `Orchestrator` | Direct function call |
//!
//! The LLM system prompt, JSON schema, and verdict parsing are identical to v2.

use std::time::Instant;

use layer0::content::Content;
use layer0::context::{Message, Role};
use neuron_context_engine::{CompileConfig, Context};
use neuron_turn::provider::Provider;
use tracing::info;

use crate::types::{ResearchResult, SweepVerdict};

// ---------------------------------------------------------------------------
// System prompt (verbatim copy from neuron-op-sweep-v2 for behavioural parity)
// ---------------------------------------------------------------------------

/// Comparison system prompt sent to the LLM.
///
/// Instructs the model to produce a structured JSON verdict matching the
/// [`SweepVerdict`] schema. Identical to the v2 `COMPARISON_SYSTEM_PROMPT` so
/// the two implementations remain behaviourally compatible.
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

/// Configuration for [`compare_decision`].
#[derive(Debug, Clone)]
pub struct CompareConfig {
    /// Model used for comparison. Default: `"claude-sonnet-4-20250514"`.
    pub model: String,
    /// Maximum output tokens. Default: 4096.
    pub max_response_tokens: u32,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            max_response_tokens: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON extraction helper
// ---------------------------------------------------------------------------

/// Strip optional `` ```json `` / `` ``` `` fence and return the inner JSON.
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
// compare_decision — the v3 core
// ---------------------------------------------------------------------------

/// Compare research findings against a decision card using context-engine.
///
/// # v3 approach
///
/// Instead of going through `dispatch_typed` → `CompareOperator::execute()`,
/// this function:
///
/// 1. Assembles a [`Context`] via the fluent API (`inject_system`, `inject_message`).
/// 2. Calls `ctx.compile(&config).infer(provider)` directly.
/// 3. Parses the JSON verdict from the response.
///
/// State reads (decision card, prior findings) happen in the caller
/// ([`crate::cycle::sweep_cycle`]) so this function is pure: provider call in,
/// verdict out.
///
/// # Errors
///
/// Returns an error if:
/// - The LLM call fails (network, auth, rate-limit).
/// - The response cannot be parsed as a [`SweepVerdict`] JSON object.
pub async fn compare_decision<P: Provider>(
    provider: &P,
    decision_id: &str,
    decision_text: &str,
    prior_findings: &[String],
    research_results: &[ResearchResult],
    config: &CompareConfig,
) -> Result<SweepVerdict, anyhow::Error> {
    let start = Instant::now();

    let prior_json =
        serde_json::to_string(prior_findings).unwrap_or_else(|_| "[]".to_string());
    let research_json =
        serde_json::to_string(research_results).unwrap_or_else(|_| "[]".to_string());

    info!(
        decision_id,
        num_research = research_results.len(),
        model = %config.model,
        "compare_decision: starting"
    );

    // ── Context assembly (v3 style: Context + fluent API) ─────────────────────
    //
    // v2 built Vec<Message> manually then constructed InferRequest directly.
    // v3 uses Context::inject_system + inject_message → compile() → infer().
    // Message content is identical; the difference is that rules fire on each
    // inject call, making the pipeline hookable.

    let user_content = format!(
        "Decision ID: {decision_id}\n\n\
         <decision>\n{decision_text}\n</decision>\n\n\
         <prior_findings>\n{prior_json}\n</prior_findings>\n\n\
         <research>\n{research_json}\n</research>"
    );

    let mut ctx = Context::new();
    ctx.inject_system(COMPARISON_SYSTEM_PROMPT).await?;
    ctx.inject_message(Message::new(Role::User, Content::text(user_content)))
        .await?;

    let compile_config = CompileConfig {
        model: Some(config.model.clone()),
        max_tokens: Some(config.max_response_tokens),
        ..Default::default()
    };

    // compile() snapshots the context; infer() crosses the network boundary.
    let result = ctx.compile(&compile_config).infer(provider).await?;
    let raw = result.text().unwrap_or_default();

    let llm_elapsed = start.elapsed();

    // ── Parse verdict ─────────────────────────────────────────────────────────
    let mut verdict: SweepVerdict = {
        let json_str = extract_json_block(raw);
        serde_json::from_str(json_str).map_err(|e| {
            anyhow::anyhow!("compare_decision: verdict parse failed: {e}\nraw:\n{raw}")
        })?
    };

    // Fill runtime fields the LLM cannot know.
    verdict.duration_secs = llm_elapsed.as_secs_f64();
    verdict.swept_at = chrono::Utc::now().to_rfc3339();
    verdict.research_inputs = research_results.to_vec();

    info!(
        decision_id,
        status = ?verdict.status,
        confidence = verdict.confidence,
        duration_ms = llm_elapsed.as_millis() as u64,
        "compare_decision: complete"
    );

    Ok(verdict)
}
