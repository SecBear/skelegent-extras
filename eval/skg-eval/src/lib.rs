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
use layer0::operator::{OperatorInput, OperatorOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
}
