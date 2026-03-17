#![deny(missing_docs)]
//! Evaluation framework for skelegent agents.
//!
//! Run agents against test cases, score outputs, produce reports.
//!
//! # Usage
//!
//! ```rust,ignore
//! use skg_eval::{EvalCase, ExpectedOutput, ExactMatchMetric};
//!
//! let cases = vec![
//!     EvalCase::new("greeting", input, ExpectedOutput::Contains(vec!["hello".into()])),
//! ];
//! ```

use async_trait::async_trait;
use layer0::dispatch::Dispatcher;
use layer0::dispatch_context::DispatchContext;
use layer0::id::{DispatchId, OperatorId};
use layer0::operator::{OperatorInput, OperatorOutput, TriggerType};
use serde::{Deserialize, Serialize};
use skg_provider_router::DynProvider;
use skg_turn::embedding::EmbedRequest;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CORE TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A single evaluation case.
#[derive(Debug, Clone)]
pub struct EvalCase {
    /// Human-readable label for this case.
    pub name: String,
    /// Input to dispatch to the agent.
    pub input: OperatorInput,
    /// What correct output looks like.
    pub expected: ExpectedOutput,
}

impl EvalCase {
    /// Create a new eval case.
    pub fn new(
        name: impl Into<String>,
        input: OperatorInput,
        expected: ExpectedOutput,
    ) -> Self {
        Self {
            name: name.into(),
            input,
            expected,
        }
    }
}

/// What "correct" looks like for an eval case.
#[derive(Clone)]
pub enum ExpectedOutput {
    /// Output text matches exactly.
    ExactText(String),
    /// Output contains all of these substrings.
    Contains(Vec<String>),
    /// Output must have used these operators (by name, in order).
    ToolTrajectory(Vec<String>),
    /// Custom validator — receives the full output and returns a score.
    Custom(std::sync::Arc<dyn Fn(&OperatorOutput) -> Score + Send + Sync>),
}

impl std::fmt::Debug for ExpectedOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExactText(s) => f.debug_tuple("ExactText").field(s).finish(),
            Self::Contains(v) => f.debug_tuple("Contains").field(v).finish(),
            Self::ToolTrajectory(v) => f.debug_tuple("ToolTrajectory").field(v).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<fn>").finish(),
        }
    }
}

/// Score from an evaluation metric. Range: 0.0 (wrong) to 1.0 (perfect).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// 0.0 = completely wrong, 1.0 = perfect.
    pub value: f64,
    /// Optional human-readable explanation.
    pub explanation: Option<String>,
}

impl Score {
    /// Create a score with no explanation.
    pub fn new(value: f64) -> Self {
        Self { value, explanation: None }
    }

    /// Create a score with an explanation.
    pub fn with_explanation(value: f64, explanation: impl Into<String>) -> Self {
        Self { value, explanation: Some(explanation.into()) }
    }

    /// Perfect score (1.0).
    pub fn perfect() -> Self {
        Self::new(1.0)
    }

    /// Zero score (0.0).
    pub fn zero() -> Self {
        Self::new(0.0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// METRICS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scoring trait. Implement for custom evaluation criteria.
#[async_trait]
pub trait EvalMetric: Send + Sync {
    /// Metric name — used as the key in [`CaseResult::scores`].
    fn name(&self) -> &str;

    /// Score an agent's output against the expected output.
    async fn score(&self, output: &OperatorOutput, expected: &ExpectedOutput) -> Score;
}

/// Exact text match metric. Scores 1.0 only when the full output text equals
/// the expected string; 0.0 otherwise.
pub struct ExactMatchMetric;

#[async_trait]
impl EvalMetric for ExactMatchMetric {
    fn name(&self) -> &str {
        "exact_match"
    }

    async fn score(&self, output: &OperatorOutput, expected: &ExpectedOutput) -> Score {
        match expected {
            ExpectedOutput::ExactText(text) => {
                let output_text = output.message.as_text().unwrap_or_default();
                if output_text == text {
                    Score::perfect()
                } else {
                    Score::with_explanation(
                        0.0,
                        format!("expected '{}', got '{}'", text, output_text),
                    )
                }
            }
            _ => Score::with_explanation(
                0.0,
                "ExactMatchMetric only scores ExactText expectations",
            ),
        }
    }
}

/// Substring containment metric. Scores the fraction of expected substrings
/// that appear anywhere in the output text.
pub struct ContainsMetric;

#[async_trait]
impl EvalMetric for ContainsMetric {
    fn name(&self) -> &str {
        "contains"
    }

    async fn score(&self, output: &OperatorOutput, expected: &ExpectedOutput) -> Score {
        match expected {
            ExpectedOutput::Contains(substrings) => {
                let output_text = output.message.as_text().unwrap_or_default();
                let total = substrings.len();
                if total == 0 {
                    return Score::perfect();
                }
                let found = substrings
                    .iter()
                    .filter(|s| output_text.contains(s.as_str()))
                    .count();
                let ratio = found as f64 / total as f64;
                if found == total {
                    Score::perfect()
                } else {
                    let missing: Vec<&str> = substrings
                        .iter()
                        .filter(|s| !output_text.contains(s.as_str()))
                        .map(|s| s.as_str())
                        .collect();
                    Score::with_explanation(ratio, format!("missing: {:?}", missing))
                }
            }
            _ => Score::with_explanation(
                0.0,
                "ContainsMetric only scores Contains expectations",
            ),
        }
    }
}

/// Tool trajectory metric. Scores 1.0 only when the operator called the
/// expected sub-operators in exactly the expected order; 0.0 otherwise.
pub struct ToolTrajectoryMetric;

#[async_trait]
impl EvalMetric for ToolTrajectoryMetric {
    fn name(&self) -> &str {
        "tool_trajectory"
    }

    async fn score(&self, output: &OperatorOutput, expected: &ExpectedOutput) -> Score {
        match expected {
            ExpectedOutput::ToolTrajectory(expected_tools) => {
                let actual: Vec<&str> = output
                    .metadata
                    .sub_dispatches
                    .iter()
                    .map(|d| d.name.as_str())
                    .collect();
                let expected_refs: Vec<&str> =
                    expected_tools.iter().map(|s| s.as_str()).collect();
                if actual == expected_refs {
                    Score::perfect()
                } else {
                    Score::with_explanation(
                        0.0,
                        format!("expected {:?}, got {:?}", expected_refs, actual),
                    )
                }
            }
            _ => Score::with_explanation(
                0.0,
                "ToolTrajectoryMetric only scores ToolTrajectory expectations",
            ),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RESULTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Result of evaluating a single case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    /// Case name.
    pub name: String,
    /// Scores by metric name.
    pub scores: HashMap<String, Score>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Error message if the agent failed to produce output.
    pub error: Option<String>,
}

/// Aggregated evaluation report across all cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Per-case results.
    pub cases: Vec<CaseResult>,
}

impl EvalReport {
    /// Mean score for a specific metric across all cases that recorded it.
    /// Returns 0.0 when no cases recorded the metric.
    pub fn mean_score(&self, metric: &str) -> f64 {
        let scores: Vec<f64> = self
            .cases
            .iter()
            .filter_map(|c| c.scores.get(metric).map(|s| s.value))
            .collect();
        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Total number of cases.
    pub fn total_cases(&self) -> usize {
        self.cases.len()
    }

    /// Number of cases where every recorded metric achieved a perfect score (1.0).
    ///
    /// Cases with no scores (e.g., agent errored before metrics ran) are not counted.
    pub fn perfect_cases(&self) -> usize {
        self.cases
            .iter()
            .filter(|c| {
                !c.scores.is_empty()
                    && c.scores.values().all(|s| (s.value - 1.0).abs() < f64::EPSILON)
            })
            .count()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RUNNER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Runs evaluation cases against an agent via a [`Dispatcher`] and scores
/// them with one or more [`EvalMetric`]s.
///
/// # Example
///
/// ```rust,ignore
/// let runner = EvalRunner::new(dispatcher, OperatorId::new("my-agent"))
///     .with_metric(ExactMatchMetric)
///     .with_concurrency(8);
///
/// let report = runner.run(cases).await;
/// println!("mean exact_match: {}", report.mean_score("exact_match"));
/// ```
pub struct EvalRunner {
    dispatcher: Arc<dyn Dispatcher>,
    operator_id: OperatorId,
    metrics: Vec<Arc<dyn EvalMetric>>,
    concurrency: usize,
}

impl EvalRunner {
    /// Create a new runner that dispatches to `operator_id` via `dispatcher`.
    ///
    /// Default concurrency is 4. Add metrics with [`with_metric`](Self::with_metric).
    pub fn new(dispatcher: Arc<dyn Dispatcher>, operator_id: OperatorId) -> Self {
        Self {
            dispatcher,
            operator_id,
            metrics: Vec::new(),
            concurrency: 4,
        }
    }

    /// Add an evaluation metric. Builder-style — can be chained.
    pub fn with_metric(mut self, metric: impl EvalMetric + 'static) -> Self {
        self.metrics.push(Arc::new(metric));
        self
    }

    /// Set the maximum number of cases to run concurrently.
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    /// Run all cases and produce an aggregated [`EvalReport`].
    ///
    /// Cases are run with bounded concurrency (see [`with_concurrency`](Self::with_concurrency)).
    /// Up to `concurrency` cases execute concurrently; the rest wait in a queue.
    pub async fn run(&self, cases: Vec<EvalCase>) -> EvalReport {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.concurrency));
        let mut join_set = tokio::task::JoinSet::new();

        for case in cases {
            let sem = Arc::clone(&semaphore);
            let dispatcher = Arc::clone(&self.dispatcher);
            let operator_id = self.operator_id.clone();
            let metrics = self.metrics.clone();

            join_set.spawn(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore closed");
                run_case(dispatcher, operator_id, metrics, case).await
            });
        }

        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(case_result) => results.push(case_result),
                Err(e) => {
                    // A task panic — record it as an error result with an empty name.
                    results.push(CaseResult {
                        name: "<panicked>".into(),
                        scores: HashMap::new(),
                        duration_ms: 0,
                        error: Some(format!("task panicked: {e}")),
                    });
                }
            }
        }

        EvalReport { cases: results }
    }

    /// Run a single case. Useful for debugging individual cases.
    pub async fn run_one(&self, case: EvalCase) -> CaseResult {
        let start = Instant::now();
        let dispatch_id = DispatchId::new(format!("eval-{}", case.name));
        let ctx = DispatchContext::new(dispatch_id, self.operator_id.clone());

        let output_result = self
            .dispatcher
            .dispatch(&ctx, case.input)
            .await
            .map_err(|e| e.to_string());

        let duration_ms = start.elapsed().as_millis() as u64;

        let handle = match output_result {
            Err(e) => {
                return CaseResult {
                    name: case.name,
                    scores: HashMap::new(),
                    duration_ms,
                    error: Some(e),
                };
            }
            Ok(h) => h,
        };

        let output = match handle.collect().await {
            Ok(o) => o,
            Err(e) => {
                return CaseResult {
                    name: case.name,
                    scores: HashMap::new(),
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("collect failed: {e}")),
                };
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        let mut scores = HashMap::new();
        for metric in &self.metrics {
            let score = metric.score(&output, &case.expected).await;
            scores.insert(metric.name().to_owned(), score);
        }

        CaseResult {
            name: case.name,
            scores,
            duration_ms,
            error: None,
        }
    }
}

/// Free function that owns all resources needed to evaluate a single case.
/// Used by [`EvalRunner::run`] to spawn tasks without borrowing `self`.
async fn run_case(
    dispatcher: Arc<dyn Dispatcher>,
    operator_id: OperatorId,
    metrics: Vec<Arc<dyn EvalMetric>>,
    case: EvalCase,
) -> CaseResult {
    let start = Instant::now();
    let dispatch_id = DispatchId::new(format!("eval-{}", case.name));
    let ctx = DispatchContext::new(dispatch_id, operator_id);

    let output_result = dispatcher
        .dispatch(&ctx, case.input)
        .await
        .map_err(|e| e.to_string());

    let duration_ms = start.elapsed().as_millis() as u64;

    let handle = match output_result {
        Err(e) => {
            return CaseResult {
                name: case.name,
                scores: HashMap::new(),
                duration_ms,
                error: Some(e),
            };
        }
        Ok(h) => h,
    };

    let output = match handle.collect().await {
        Ok(o) => o,
        Err(e) => {
            return CaseResult {
                name: case.name,
                scores: HashMap::new(),
                duration_ms: start.elapsed().as_millis() as u64,
                error: Some(format!("collect failed: {e}")),
            };
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    let mut scores = HashMap::new();
    for metric in &metrics {
        let score = metric.score(&output, &case.expected).await;
        scores.insert(metric.name().to_owned(), score);
    }

    CaseResult {
        name: case.name,
        scores,
        duration_ms,
        error: None,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LLM JUDGE METRIC
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scores agent output by asking a judge LLM to evaluate it.
///
/// The judge is invoked via a [`Dispatcher`] targeting a configured operator.
/// It receives a prompt containing the rubric, expected output, and actual
/// output, and should respond with a float between 0.0 and 1.0 followed by
/// an explanation.
pub struct LlmJudgeMetric {
    dispatcher: Arc<dyn Dispatcher>,
    judge_operator: OperatorId,
    rubric: String,
}

impl LlmJudgeMetric {
    /// Create a new judge metric.
    ///
    /// - `dispatcher` — used to invoke the judge operator.
    /// - `judge_operator` — the operator ID of the LLM judge.
    /// - `rubric` — evaluation criteria shown to the judge.
    pub fn new(
        dispatcher: Arc<dyn Dispatcher>,
        judge_operator: OperatorId,
        rubric: impl Into<String>,
    ) -> Self {
        Self {
            dispatcher,
            judge_operator,
            rubric: rubric.into(),
        }
    }
}

#[async_trait]
impl EvalMetric for LlmJudgeMetric {
    fn name(&self) -> &str {
        "llm_judge"
    }

    async fn score(&self, output: &OperatorOutput, expected: &ExpectedOutput) -> Score {
        let expected_text = match expected {
            ExpectedOutput::ExactText(s) => s.clone(),
            ExpectedOutput::Contains(v) => v.join(", "),
            ExpectedOutput::ToolTrajectory(v) => format!("tools: {}", v.join(" -> ")),
            ExpectedOutput::Custom(_) => "<custom>".into(),
        };
        let actual_text = output.message.as_text().unwrap_or_default();

        let prompt = format!(
            "Given this rubric: {rubric}\nExpected: {expected}\nActual: {actual}\nScore 0.0-1.0 and explain.",
            rubric = self.rubric,
            expected = expected_text,
            actual = actual_text,
        );

        let input = OperatorInput::new(
            layer0::content::Content::text(prompt),
            TriggerType::Task,
        );
        let ctx = DispatchContext::new(
            DispatchId::new("eval-judge"),
            self.judge_operator.clone(),
        );

        let handle = match self.dispatcher.dispatch(&ctx, input).await {
            Ok(h) => h,
            Err(e) => {
                return Score::with_explanation(0.0, format!("judge dispatch failed: {e}"));
            }
        };

        let judge_output = match handle.collect().await {
            Ok(o) => o,
            Err(e) => {
                return Score::with_explanation(0.0, format!("judge collect failed: {e}"));
            }
        };

        let response_text = judge_output.message.as_text().unwrap_or_default();

        // Parse the first float found in the response.
        let score_value = parse_first_float(response_text);
        let clamped = score_value.clamp(0.0, 1.0);

        Score::with_explanation(clamped, response_text.to_owned())
    }
}

/// Extract the first floating-point number from a string.
/// Returns 0.0 if none is found.
fn parse_first_float(text: &str) -> f64 {
    let mut start = None;
    let mut has_dot = false;

    for (i, ch) in text.char_indices() {
        match ch {
            '0'..='9' => {
                if start.is_none() {
                    start = Some(i);
                }
            }
            '.' if start.is_some() && !has_dot => {
                has_dot = true;
            }
            _ => {
                if let Some(s) = start {
                    let candidate = &text[s..i];
                    if let Ok(v) = candidate.parse::<f64>() {
                        return v;
                    }
                    start = None;
                    has_dot = false;
                }
            }
        }
    }

    // Check at end of string.
    if let Some(s) = start {
        if let Ok(v) = text[s..].parse::<f64>() {
            return v;
        }
    }

    0.0
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SEMANTIC SIMILARITY METRIC
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Scores by computing cosine similarity between expected and actual embeddings.
///
/// Requires a [`DynProvider`] that supports the `embed` operation. The score
/// equals the cosine similarity (clamped to [0.0, 1.0]).
pub struct SemanticSimilarityMetric {
    provider: Arc<dyn DynProvider>,
    model: String,
    threshold: f64,
}

impl SemanticSimilarityMetric {
    /// Create a new semantic similarity metric using the given provider and model.
    ///
    /// Default threshold is 0.8 (scores below this threshold are returned as-is;
    /// the threshold does not gate the score but is available for callers that
    /// want to treat it as a pass/fail cutoff).
    pub fn new(provider: Arc<dyn DynProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            threshold: 0.8,
        }
    }

    /// Set the similarity threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Return the configured similarity threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[async_trait]
impl EvalMetric for SemanticSimilarityMetric {
    fn name(&self) -> &str {
        "semantic_similarity"
    }

    async fn score(&self, output: &OperatorOutput, expected: &ExpectedOutput) -> Score {
        let expected_text = match expected {
            ExpectedOutput::ExactText(s) => s.clone(),
            ExpectedOutput::Contains(v) => v.join(" "),
            ExpectedOutput::ToolTrajectory(v) => v.join(" "),
            ExpectedOutput::Custom(_) => {
                return Score::with_explanation(
                    0.0,
                    "SemanticSimilarityMetric cannot score Custom expectations",
                );
            }
        };
        let actual_text = output.message.as_text().unwrap_or_default().to_owned();

        let request = EmbedRequest::new(vec![expected_text.clone(), actual_text.clone()])
            .with_model(&self.model);

        let embed_response = match self.provider.embed_boxed(request).await {
            Ok(r) => r,
            Err(e) => {
                return Score::with_explanation(0.0, format!("embed failed: {e}"));
            }
        };

        if embed_response.embeddings.len() < 2 {
            return Score::with_explanation(
                0.0,
                "embed returned fewer than 2 embeddings",
            );
        }

        let a = &embed_response.embeddings[0].vector;
        let b = &embed_response.embeddings[1].vector;
        let similarity = cosine_similarity(a, b);
        let clamped = similarity.clamp(0.0, 1.0);

        Score::with_explanation(
            clamped,
            format!(
                "cosine similarity {clamped:.4} (threshold {threshold:.2})",
                threshold = self.threshold
            ),
        )
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero magnitude (to avoid NaN).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ERRORS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Errors from evaluation runs.
#[derive(Debug, Error)]
pub enum EvalError {
    /// Agent dispatch failed.
    #[error("dispatch failed: {0}")]
    DispatchFailed(String),
    /// Agent output collection failed.
    #[error("collect failed: {0}")]
    CollectFailed(String),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TESTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::operator::{ExitReason, SubDispatchRecord};
    use layer0::duration::DurationMs;

    fn test_output(text: &str) -> OperatorOutput {
        OperatorOutput::new(Content::text(text), ExitReason::Complete)
    }

    fn test_output_with_dispatches(text: &str, tools: &[&str]) -> OperatorOutput {
        let mut out = test_output(text);
        out.metadata.sub_dispatches = tools
            .iter()
            .map(|name| SubDispatchRecord::new(*name, DurationMs::ZERO, true))
            .collect();
        out
    }

    #[tokio::test]
    async fn exact_match_perfect() {
        let output = test_output("hello world");
        let expected = ExpectedOutput::ExactText("hello world".into());
        let score = ExactMatchMetric.score(&output, &expected).await;
        assert!((score.value - 1.0).abs() < f64::EPSILON);
        assert!(score.explanation.is_none());
    }

    #[tokio::test]
    async fn exact_match_mismatch() {
        let output = test_output("hello world");
        let expected = ExpectedOutput::ExactText("goodbye".into());
        let score = ExactMatchMetric.score(&output, &expected).await;
        assert!((score.value - 0.0).abs() < f64::EPSILON);
        assert!(score.explanation.is_some());
    }

    #[tokio::test]
    async fn exact_match_wrong_variant() {
        let output = test_output("hello");
        let expected = ExpectedOutput::Contains(vec!["hello".into()]);
        let score = ExactMatchMetric.score(&output, &expected).await;
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn contains_all_present() {
        let output = test_output("the quick brown fox jumps over the lazy dog");
        let expected =
            ExpectedOutput::Contains(vec!["quick".into(), "fox".into(), "dog".into()]);
        let score = ContainsMetric.score(&output, &expected).await;
        assert!((score.value - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn contains_partial() {
        let output = test_output("the quick brown fox");
        let expected = ExpectedOutput::Contains(vec!["quick".into(), "dog".into()]);
        let score = ContainsMetric.score(&output, &expected).await;
        assert!((score.value - 0.5).abs() < f64::EPSILON);
        assert!(score.explanation.is_some());
    }

    #[tokio::test]
    async fn contains_empty_substrings() {
        let output = test_output("anything");
        let expected = ExpectedOutput::Contains(vec![]);
        let score = ContainsMetric.score(&output, &expected).await;
        assert!((score.value - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn tool_trajectory_perfect() {
        let output = test_output_with_dispatches("done", &["search", "summarize"]);
        let expected =
            ExpectedOutput::ToolTrajectory(vec!["search".into(), "summarize".into()]);
        let score = ToolTrajectoryMetric.score(&output, &expected).await;
        assert!((score.value - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn tool_trajectory_mismatch() {
        let output = test_output_with_dispatches("done", &["search"]);
        let expected =
            ExpectedOutput::ToolTrajectory(vec!["search".into(), "summarize".into()]);
        let score = ToolTrajectoryMetric.score(&output, &expected).await;
        assert!((score.value - 0.0).abs() < f64::EPSILON);
        assert!(score.explanation.is_some());
    }

    #[test]
    fn eval_report_mean_score() {
        let report = EvalReport {
            cases: vec![
                CaseResult {
                    name: "a".into(),
                    scores: HashMap::from([("exact_match".into(), Score::new(1.0))]),
                    duration_ms: 100,
                    error: None,
                },
                CaseResult {
                    name: "b".into(),
                    scores: HashMap::from([("exact_match".into(), Score::new(0.5))]),
                    duration_ms: 200,
                    error: None,
                },
            ],
        };
        assert!((report.mean_score("exact_match") - 0.75).abs() < f64::EPSILON);
        assert_eq!(report.total_cases(), 2);
        assert_eq!(report.perfect_cases(), 1);
    }

    #[test]
    fn eval_report_missing_metric() {
        let report = EvalReport { cases: vec![] };
        assert!((report.mean_score("no_such_metric") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn score_constructors() {
        let s = Score::perfect();
        assert!((s.value - 1.0).abs() < f64::EPSILON);

        let s = Score::zero();
        assert!((s.value - 0.0).abs() < f64::EPSILON);

        let s = Score::with_explanation(0.7, "mostly right");
        assert!((s.value - 0.7).abs() < f64::EPSILON);
        assert_eq!(s.explanation.as_deref(), Some("mostly right"));
    }

    #[test]
    fn eval_case_new() {
        use layer0::operator::TriggerType;
        let input = OperatorInput::new(Content::text("hi"), TriggerType::User);
        let case = EvalCase::new("test", input, ExpectedOutput::ExactText("hi".into()));
        assert_eq!(case.name, "test");
    }

    // ── EvalRunner tests ────────────────────────────────────────────────────

    use layer0::dispatch::{DispatchEvent, DispatchHandle};
    use layer0::error::OrchError;

    /// Mock dispatcher that always returns a fixed OperatorOutput.
    struct MockDispatcher {
        output: OperatorOutput,
    }

    impl MockDispatcher {
        fn returning(text: &str) -> Arc<Self> {
            Arc::new(Self {
                output: test_output(text),
            })
        }
    }

    #[async_trait]
    impl Dispatcher for MockDispatcher {
        async fn dispatch(
            &self,
            ctx: &layer0::dispatch_context::DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let dispatch_id = ctx.dispatch_id.clone();
            let (handle, sender) = DispatchHandle::channel(dispatch_id);
            let output = self.output.clone();
            tokio::spawn(async move {
                let _ = sender
                    .send(DispatchEvent::Completed { output })
                    .await;
            });
            Ok(handle)
        }
    }

    /// Mock dispatcher that always returns a dispatch error.
    struct ErrorDispatcher;

    #[async_trait]
    impl Dispatcher for ErrorDispatcher {
        async fn dispatch(
            &self,
            _ctx: &layer0::dispatch_context::DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            Err(OrchError::DispatchFailed("mock error".into()))
        }
    }

    #[tokio::test]
    async fn eval_runner_runs_cases() {
        use layer0::operator::TriggerType;

        let dispatcher = MockDispatcher::returning("hello");
        let runner = EvalRunner::new(dispatcher, OperatorId::new("test-op"))
            .with_metric(ExactMatchMetric)
            .with_concurrency(2);

        let cases = vec![
            EvalCase::new(
                "case-a",
                OperatorInput::new(Content::text("hi"), TriggerType::User),
                ExpectedOutput::ExactText("hello".into()),
            ),
            EvalCase::new(
                "case-b",
                OperatorInput::new(Content::text("hi"), TriggerType::User),
                ExpectedOutput::ExactText("hello".into()),
            ),
        ];

        let report = runner.run(cases).await;
        assert_eq!(report.total_cases(), 2);
        assert_eq!(report.perfect_cases(), 2);
        assert!((report.mean_score("exact_match") - 1.0).abs() < f64::EPSILON);
        for case in &report.cases {
            assert!(case.error.is_none());
        }
    }

    #[tokio::test]
    async fn eval_runner_handles_dispatch_error() {
        use layer0::operator::TriggerType;

        let dispatcher = Arc::new(ErrorDispatcher);
        let runner = EvalRunner::new(dispatcher, OperatorId::new("test-op"))
            .with_metric(ExactMatchMetric);

        let cases = vec![EvalCase::new(
            "failing-case",
            OperatorInput::new(Content::text("hi"), TriggerType::User),
            ExpectedOutput::ExactText("anything".into()),
        )];

        let report = runner.run(cases).await;
        assert_eq!(report.total_cases(), 1);
        let case_result = &report.cases[0];
        assert!(case_result.error.is_some(), "expected an error to be recorded");
        assert!(case_result.scores.is_empty());
    }

    // ── Cosine similarity tests ─────────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0_f32, 0.0, 1.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "identical vectors → 1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9, "orthogonal vectors → 0.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-9, "zero vector → 0.0, got {sim}");
    }

    // ── Concurrency verification tests ────────────────────────────────────

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Dispatcher that sleeps for a configurable duration before completing.
    /// Used to verify wall-clock parallelism.
    struct SlowDispatcher {
        delay: std::time::Duration,
        output: OperatorOutput,
    }

    impl SlowDispatcher {
        fn new(delay: std::time::Duration) -> Arc<Self> {
            Arc::new(Self {
                delay,
                output: test_output("ok"),
            })
        }
    }

    #[async_trait]
    impl Dispatcher for SlowDispatcher {
        async fn dispatch(
            &self,
            ctx: &layer0::dispatch_context::DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let dispatch_id = ctx.dispatch_id.clone();
            let (handle, sender) = DispatchHandle::channel(dispatch_id);
            let output = self.output.clone();
            let delay = self.delay;
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let _ = sender
                    .send(DispatchEvent::Completed { output })
                    .await;
            });
            Ok(handle)
        }
    }

    /// Dispatcher that tracks how many dispatches are in-flight concurrently,
    /// recording the high-water mark.
    struct ConcurrencyTrackingDispatcher {
        delay: std::time::Duration,
        active: AtomicUsize,
        max_concurrent: AtomicUsize,
        output: OperatorOutput,
    }

    impl ConcurrencyTrackingDispatcher {
        fn new(delay: std::time::Duration) -> Arc<Self> {
            Arc::new(Self {
                delay,
                active: AtomicUsize::new(0),
                max_concurrent: AtomicUsize::new(0),
                output: test_output("tracked"),
            })
        }

        fn max_concurrent(&self) -> usize {
            self.max_concurrent.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Dispatcher for ConcurrencyTrackingDispatcher {
        async fn dispatch(
            &self,
            ctx: &layer0::dispatch_context::DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            // Increment active count and update high-water mark.
            let prev = self.active.fetch_add(1, Ordering::SeqCst);
            let current = prev + 1;
            self.max_concurrent
                .fetch_max(current, Ordering::SeqCst);

            let dispatch_id = ctx.dispatch_id.clone();
            let (handle, sender) = DispatchHandle::channel(dispatch_id);
            let output = self.output.clone();
            let delay = self.delay;

            // We need a raw pointer to decrement `active` after the sleep.
            // Safety: the Arc that owns this dispatcher outlives all spawned
            // tasks because EvalRunner holds the Arc until join_set drains.
            let active_ptr = &self.active as *const AtomicUsize;
            // SAFETY: The Arc<ConcurrencyTrackingDispatcher> is kept alive by the
            // EvalRunner (which holds Arc<dyn Dispatcher>) for the entire duration
            // of `run()`, including while it drains the JoinSet. Every spawned task
            // completes before `run()` returns, so the AtomicUsize is guaranteed to
            // be alive for the duration of every spawned task.
            let active_ref = unsafe { &*active_ptr };

            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                active_ref.fetch_sub(1, Ordering::SeqCst);
                let _ = sender
                    .send(DispatchEvent::Completed { output })
                    .await;
            });
            Ok(handle)
        }
    }

    fn make_cases(n: usize) -> Vec<EvalCase> {
        use layer0::operator::TriggerType;
        (0..n)
            .map(|i| {
                EvalCase::new(
                    format!("case-{i}"),
                    OperatorInput::new(Content::text("input"), TriggerType::Task),
                    ExpectedOutput::ExactText("ok".into()),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn eval_runner_executes_concurrently_wall_clock() {
        // 4 cases each sleeping 100ms, all running in parallel (concurrency=4).
        // Sequential would take ≥400ms; parallel should complete well under 250ms.
        let dispatcher = SlowDispatcher::new(std::time::Duration::from_millis(100));
        let runner = EvalRunner::new(dispatcher, OperatorId::new("test-op"))
            .with_concurrency(4);

        let cases = make_cases(4);
        let start = Instant::now();
        let report = runner.run(cases).await;
        let elapsed = start.elapsed();

        assert_eq!(report.total_cases(), 4);
        for case in &report.cases {
            assert!(case.error.is_none(), "unexpected error: {:?}", case.error);
        }
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "expected <250ms for 4 parallel 100ms tasks, got {}ms",
            elapsed.as_millis(),
        );
    }

    #[tokio::test]
    async fn eval_runner_respects_semaphore_bound() {
        // 6 cases, concurrency=2, each sleeping 50ms.
        // The tracking dispatcher records the high-water mark of in-flight
        // dispatches — it must never exceed the configured concurrency.
        let dispatcher =
            ConcurrencyTrackingDispatcher::new(std::time::Duration::from_millis(50));
        let dyn_dispatcher: Arc<dyn Dispatcher> = Arc::clone(&dispatcher) as _;
        let runner = EvalRunner::new(dyn_dispatcher, OperatorId::new("test-op"))
            .with_concurrency(2);

        let cases = make_cases(6);
        let report = runner.run(cases).await;

        assert_eq!(report.total_cases(), 6);
        for case in &report.cases {
            assert!(case.error.is_none(), "unexpected error: {:?}", case.error);
        }

        let max = dispatcher.max_concurrent();
        assert!(
            max <= 2,
            "max concurrent dispatches was {max}, expected ≤ 2",
        );
        // With 6 cases and concurrency=2 the max should actually reach 2.
        assert!(
            max == 2,
            "expected max concurrent to reach 2, but got {max}",
        );
    }

    #[tokio::test]
    async fn eval_runner_completes_all_cases_under_concurrency() {
        // Verify that the JoinSet correctly collects results from many cases
        // when concurrency is lower than the total case count. This exercises
        // the "queue" path where later tasks must wait for a semaphore permit.
        let dispatcher = MockDispatcher::returning("ok");
        let runner = EvalRunner::new(dispatcher, OperatorId::new("test-op"))
            .with_metric(ExactMatchMetric)
            .with_concurrency(2);

        let cases = make_cases(10);
        let report = runner.run(cases).await;

        assert_eq!(report.total_cases(), 10);
        assert_eq!(report.perfect_cases(), 10);
        assert!((report.mean_score("exact_match") - 1.0).abs() < f64::EPSILON);
        for case in &report.cases {
            assert!(case.error.is_none(), "unexpected error: {:?}", case.error);
        }
    }
}
