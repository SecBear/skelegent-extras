//! Research provider abstraction and error types.
//!
//! [`ResearchProvider`] is the trait abstraction over Parallel.ai (or any
//! research backend). Production implementations make HTTP calls; test code
//! uses [`MockProvider`].

use crate::types::{ProcessorTier, SweepVerdict, VerdictStatus};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

/// Abstraction over a research backend (e.g. Parallel.ai).
///
/// Implementations are responsible for:
/// - Executing web research queries against the configured backend.
/// - Calling an LLM to compare research findings against a decision.
///
/// The separation of `search` and `compare` allows independent testing
/// of each step and future replacement of either backend.
#[async_trait]
pub trait ResearchProvider: Send + Sync {
    /// Execute a research query and return matching results.
    ///
    /// The `processor` hint controls which model tier the backend uses.
    /// Callers SHOULD use [`crate::operator::SweepOperator::select_processor`]
    /// to choose the tier based on budget and previous verdict.
    ///
    /// # Errors
    ///
    /// Returns [`SweepError::Transient`] for rate limits or timeouts (retryable).
    /// Returns [`SweepError::Permanent`] for auth or malformed request errors.
    async fn search(
        &self,
        query: &str,
        processor: ProcessorTier,
    ) -> Result<Vec<ResearchResult>, SweepError>;

    /// Compare research findings against the current decision text.
    ///
    /// The LLM receives the system prompt, packed context, serialized research
    /// results, and the full decision text, and produces a structured verdict.
    ///
    /// # Errors
    ///
    /// Returns [`SweepError::LlmFailure`] if the model fails or produces
    /// unparseable output. Returns [`SweepError::DecisionNotFound`] if the
    /// decision file cannot be read by the backend.
    async fn compare(
        &self,
        system: &str,
        context: &str,
        research: &str,
        current_decision: &str,
    ) -> Result<SweepVerdict, SweepError>;

    /// Generate targeted research queries from the decision context.
    ///
    /// Returns a list of 1-5 queries optimized for the research backend.
    /// The default implementation returns `None`, causing the operator to
    /// fall back to keyword-based query generation.
    ///
    /// Only called for Challenged/Refined verdicts (adaptive plan-first).
    async fn plan(
        &self,
        decision_id: &str,
        decision_text: &str,
        previous_verdict: Option<&VerdictStatus>,
    ) -> Result<Option<Vec<String>>, SweepError> {
        let _ = (decision_id, decision_text, previous_verdict);
        Ok(None)
    }

    /// Deep research via async task API.
    ///
    /// For Challenged/Refined verdicts where simple search isn't enough.
    /// Default: delegates to `search()` (backward compatible).
    async fn research(
        &self,
        query: &str,
        processor: ProcessorTier,
    ) -> Result<Vec<ResearchResult>, SweepError> {
        self.search(query, processor).await
    }

    /// Extract content from a URL.
    ///
    /// Returns cleaned/structured content from web pages, PDFs, etc.
    /// Default: returns empty result.
    async fn extract(
        &self,
        _url: &str,
    ) -> Result<Option<ResearchResult>, SweepError> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Configurable mock research provider for unit tests.
///
/// Can be configured to fail with a transient error for the first N search
/// calls, then succeed with pre-configured results.
#[cfg(test)]
pub struct MockProvider {
    /// Number of initial search calls to fail with a transient error.
    fail_count: usize,
    /// Tracks how many search calls have been made.
    call_count: std::sync::Mutex<usize>,
    /// Results returned after the failure window has passed.
    results: Vec<ResearchResult>,
    /// Verdict returned by `compare`.
    compare_response: SweepVerdict,
}

#[cfg(test)]
impl MockProvider {
    /// Create a provider that always succeeds immediately.
    pub fn new(results: Vec<ResearchResult>, compare_response: SweepVerdict) -> Self {
        Self::new_failing(0, results, compare_response)
    }

    /// Create a provider that returns transient errors for the first
    /// `fail_count` search calls, then succeeds.
    pub fn new_failing(
        fail_count: usize,
        results: Vec<ResearchResult>,
        compare_response: SweepVerdict,
    ) -> Self {
        Self {
            fail_count,
            call_count: std::sync::Mutex::new(0),
            results,
            compare_response,
        }
    }
}

#[cfg(test)]
#[async_trait]
impl ResearchProvider for MockProvider {
    async fn search(
        &self,
        _query: &str,
        _processor: ProcessorTier,
    ) -> Result<Vec<ResearchResult>, SweepError> {
        let n = {
            let mut count = self.call_count.lock().unwrap();
            let n = *count;
            *count += 1;
            n
        };
        if n < self.fail_count {
            Err(SweepError::Transient(format!(
                "mock transient error (call {})",
                n + 1
            )))
        } else {
            Ok(self.results.clone())
        }
    }

    async fn compare(
        &self,
        _system: &str,
        _context: &str,
        _research: &str,
        _current_decision: &str,
    ) -> Result<SweepVerdict, SweepError> {
        Ok(self.compare_response.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EvidenceStance, VerdictStatus};

    fn dummy_verdict(decision_id: &str) -> SweepVerdict {
        SweepVerdict {
            decision_id: decision_id.to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.9,
            num_supporting: 3,
            num_contradicting: 0,
            cost_usd: 0.10,
            processor: crate::types::ProcessorTier::Base,
            duration_secs: 2.0,
            swept_at: "2026-03-04T18:00:00Z".to_string(),
            evidence: vec![],
            narrative: "Confirmed by mock".to_string(),
            proposed_diff: None,
        }
    }

    fn dummy_result() -> ResearchResult {
        ResearchResult {
            url: "https://example.com".to_string(),
            title: "Example Paper".to_string(),
            snippet: "Relevant finding about agent architecture.".to_string(),
            retrieved_at: "2026-03-04T17:00:00Z".to_string(),
        }
    }

    #[tokio::test]
    async fn mock_provider_returns_configured_results() {
        let expected = vec![dummy_result()];
        let provider = MockProvider::new(expected.clone(), dummy_verdict("test"));
        let results = provider
            .search("query", ProcessorTier::Base)
            .await
            .expect("search should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, expected[0].url);
    }

    #[tokio::test]
    async fn mock_provider_returns_configured_verdict() {
        let verdict = dummy_verdict("topic-3b");
        let provider = MockProvider::new(vec![], verdict.clone());
        let result = provider
            .compare("sys", "ctx", "research", "decision")
            .await
            .expect("compare should succeed");
        assert_eq!(result.decision_id, "topic-3b");
        assert_eq!(result.status, VerdictStatus::Confirmed);
    }

    #[tokio::test]
    async fn mock_provider_fails_first_n_then_succeeds() {
        let results = vec![dummy_result()];
        let provider = MockProvider::new_failing(2, results.clone(), dummy_verdict("test"));

        // First two calls fail with Transient
        let e1 = provider.search("q", ProcessorTier::Base).await.unwrap_err();
        assert!(e1.is_transient(), "first call should be transient");

        let e2 = provider.search("q", ProcessorTier::Base).await.unwrap_err();
        assert!(e2.is_transient(), "second call should be transient");

        // Third call succeeds
        let r = provider
            .search("q", ProcessorTier::Base)
            .await
            .expect("third call should succeed");
        assert_eq!(r.len(), 1);
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

    #[tokio::test]
    async fn mock_provider_plan_returns_none_by_default() {
        let mock = MockProvider::new(vec![], dummy_verdict("test"));
        let result = mock
            .plan("topic-3b", "some decision text", Some(&VerdictStatus::Challenged))
            .await
            .expect("plan should succeed");
        assert!(result.is_none(), "default plan() should return None");
    }

    #[tokio::test]
    async fn research_delegates_to_search_by_default() {
        let expected = vec![dummy_result()];
        let provider = MockProvider::new(expected.clone(), dummy_verdict("test"));
        let results = provider
            .research("query", ProcessorTier::Base)
            .await
            .expect("research should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, expected[0].url);
    }

    #[tokio::test]
    async fn extract_returns_none_by_default() {
        let provider = MockProvider::new(vec![], dummy_verdict("test"));
        let result = provider
            .extract("https://example.com")
            .await
            .expect("extract should succeed");
        assert!(result.is_none(), "default extract() should return None");
    }
}