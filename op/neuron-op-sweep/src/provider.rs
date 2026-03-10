//! Research provider types and error definitions.
//!
//! [`ResearchResult`] and [`SweepError`] are the core types used by the sweep
//! pipeline. [`ResearchSource`] is the trait that research operators use to
//! obtain evidence — implementations live in consumer crates.
use serde::{Deserialize, Serialize};
use thiserror::Error;
use async_trait::async_trait;

/// A single research result returned by the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    /// Source URL for this result.
    pub url: String,
    /// Page or article title.
    pub title: String,
    /// Relevant excerpt or summary from the source.
    pub snippet: String,
    /// RFC 3339 timestamp of when this result was retrieved.
    pub retrieved_at: String,
}

/// How the research operator should gather evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchMode {
    /// Quick synchronous search (~5s). Broad but shallow.
    Search,
    /// Deep async research via task API (5-25 min). Thorough analysis.
    #[default]
    Deep,
}


/// Input for [`crate::research_operator::ResearchOperator`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchInput {
    /// The decision being researched.
    pub decision_id: String,
    /// Search query text.
    pub query: String,
    /// Which angle/facet this query covers.
    pub query_angle: String,
    /// How deep to research.
    #[serde(default)]
    pub mode: ResearchMode,
    /// Processor tier for deep mode (e.g. "ultra", "core").
    #[serde(default = "default_processor")]
    pub processor: String,
}

fn default_processor() -> String {
    "ultra".to_string()
}

/// Abstraction over a research backend.
///
/// Implementations live outside this crate — the sweep pipeline only depends
/// on this trait, not on any concrete HTTP client.
///
/// # Contract
///
/// - `search` must return quickly (seconds). Used for broad coverage.
/// - `deep_research` may take minutes. Used for thorough per-decision analysis.
/// - Both must populate `retrieved_at` on every [`ResearchResult`].
#[async_trait]
pub trait ResearchSource: Send + Sync + 'static {
    /// Quick synchronous search.
    async fn search(&self, query: &str) -> Result<Vec<ResearchResult>, SweepError>;

    /// Deep async research with specified processor tier.
    async fn deep_research(
        &self,
        query: &str,
        processor: &str,
    ) -> Result<Vec<ResearchResult>, SweepError>;
}

/// Errors that can occur during a sweep operation.
#[derive(Debug, Clone, Serialize, Deserialize, Error)]
#[serde(rename_all = "snake_case")]
pub enum SweepError {
    /// Transient failure (rate limit, timeout, 5xx). Safe to retry.
    #[error("transient error: {0}")]
    Transient(String),

    /// Permanent failure (bad request, auth error). Do not retry.
    #[error("permanent error: {0}")]
    Permanent(String),

    /// The LLM comparison step failed.
    #[error("LLM failure: {0}")]
    LlmFailure(String),

    /// The per-day research budget has been exhausted.
    #[error("budget exhausted")]
    BudgetExhausted,

    /// The requested decision ID was not found in the store or file system.
    #[error("decision not found: {0}")]
    DecisionNotFound(String),
}

impl SweepError {
    /// Returns `true` if this error is safe to retry with exponential backoff.
    ///
    /// Only [`SweepError::Transient`] errors are retryable. All others fail
    /// immediately to avoid wasting budget on unrecoverable conditions.
    pub fn is_transient(&self) -> bool {
        matches!(self, SweepError::Transient(_))
    }
}

/// Typed input for [`crate::operator::CompareOperator`].
///
/// Bundles research results with the decision ID into a single serializable
/// struct. [`crate::workflow::run_sweep`] passes this to `dispatch_typed` so
/// both the message body and the `metadata["decision_id"]` field are derived
/// from a single source of truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareInput {
    /// Research results to compare against the decision.
    pub research_results: Vec<ResearchResult>,
    /// Identifier of the decision under review.
    pub decision_id: String,
    /// The search query that produced these research results.
    #[serde(default)]
    pub query: Option<String>,
    /// Which angle/facet this query covers.
    #[serde(default)]
    pub query_angle: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EvidenceStance;
    fn dummy_result() -> ResearchResult {
        ResearchResult {
            url: "https://example.com".to_string(),
            title: "Example Paper".to_string(),
            snippet: "Relevant finding about agent architecture.".to_string(),
            retrieved_at: "2026-03-04T17:00:00Z".to_string(),
        }
    }

    #[test]
    fn sweep_error_is_transient_only_for_transient_variant() {
        assert!(SweepError::Transient("rate limited".into()).is_transient());
        assert!(!SweepError::Permanent("bad request".into()).is_transient());
        assert!(!SweepError::LlmFailure("model error".into()).is_transient());
        assert!(!SweepError::BudgetExhausted.is_transient());
        assert!(!SweepError::DecisionNotFound("topic-3b".into()).is_transient());
    }

    #[test]
    fn sweep_error_serde_round_trip() {
        let errors = [
            SweepError::Transient("timeout".into()),
            SweepError::Permanent("403 forbidden".into()),
            SweepError::LlmFailure("invalid json".into()),
            SweepError::BudgetExhausted,
            SweepError::DecisionNotFound("topic-3b".into()),
        ];
        for err in &errors {
            let json = serde_json::to_string(err).expect("serialize");
            let back: SweepError = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                err.to_string(),
                back.to_string(),
                "round-trip display mismatch for {:?}",
                err
            );
        }
    }

    #[test]
    fn research_result_serde_round_trip() {
        let result = dummy_result();
        let json = serde_json::to_string(&result).expect("serialize");
        let back: ResearchResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.url, result.url);
        assert_eq!(back.title, result.title);
        assert_eq!(back.snippet, result.snippet);
    }

    // Verify EvidenceStance is accessible (used by callers assembling EvidenceItem)
    #[test]
    fn evidence_stance_accessible() {
        let _s = EvidenceStance::Supporting;
        let _c = EvidenceStance::Contradicting;
        let _n = EvidenceStance::Neutral;
    }

}