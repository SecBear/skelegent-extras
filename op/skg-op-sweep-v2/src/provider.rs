//! Research source abstraction for sweep operators.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A research finding from an external source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    /// URL of the source.
    pub url: String,
    /// Title of the source.
    pub title: String,
    /// Relevant excerpt from the source.
    pub snippet: String,
    /// RFC 3339 timestamp of when the result was retrieved.
    pub retrieved_at: String,
}

/// Research depth mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchMode {
    /// Quick synchronous search (~5s).
    Search,
    /// Deep async research (5-25 min).
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

/// Errors that can occur during sweep operations.
#[derive(Debug, Clone, Error)]
pub enum SweepError {
    /// Transient failure — safe to retry.
    #[error("transient: {0}")]
    Transient(String),
    /// Permanent failure — retrying will not help.
    #[error("permanent: {0}")]
    Permanent(String),
    /// Research operation timed out.
    #[error("timeout: {0}")]
    Timeout(String),
}

/// Input for a comparison operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareInput {
    /// Research results to compare against the decision.
    pub research_results: Vec<ResearchResult>,
    /// Decision identifier.
    pub decision_id: String,
    /// Research query used.
    pub query: String,
    /// Angle/facet of the query.
    pub query_angle: String,
}

/// Abstraction over research backends (Parallel.ai, web search, etc.).
#[async_trait]
pub trait ResearchSource: Send + Sync + 'static {
    /// Quick synchronous search.
    async fn search(&self, query: &str) -> Result<Vec<ResearchResult>, SweepError>;
    /// Deep async research with a processor tier hint.
    async fn deep_research(
        &self,
        query: &str,
        processor: &str,
    ) -> Result<Vec<ResearchResult>, SweepError>;
}
