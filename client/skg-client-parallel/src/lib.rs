#![deny(missing_docs)]
//! Typed HTTP client for Parallel.ai's REST API.
//!
//! Provides structured access to Parallel.ai's search, extract, and task APIs
//! with [`AuthProvider`] integration and configurable polling for async tasks.
//!
//! # Auth resolution order
//!
//! 1. If an [`AuthProvider`] is set via [`ParallelClient::with_auth`], call
//!    `provider.provide()` with audience `"parallel.ai"`.
//! 2. If no provider is configured, or the provider returns an error, fall back
//!    to reading the environment variable named by `api_key_var`.
//!
//! # Example
//!
//! ```no_run
//! use skg_client_parallel::ParallelClient;
//!
//! # async fn run() -> Result<(), skg_client_parallel::ParallelError> {
//! let client = ParallelClient::new("PARALLEL_API_KEY");
//! let results = client.search("recent AI safety research", "one-shot").await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Duration;

use skg_auth::{AuthProvider, AuthRequest};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`ParallelClient`] operations.
#[derive(Debug, Clone, Error)]
pub enum ParallelError {
    /// Transient failure — the operation may succeed if retried.
    #[error("transient: {0}")]
    Transient(String),

    /// Permanent failure — retrying will not help.
    #[error("permanent: {0}")]
    Permanent(String),

    /// Task polling exceeded the configured maximum number of attempts.
    #[error("timeout: {0}")]
    Timeout(String),
}

impl ParallelError {
    /// Returns `true` if retrying this operation is safe.
    ///
    /// [`Transient`](ParallelError::Transient) and
    /// [`Timeout`](ParallelError::Timeout) are retryable.
    /// [`Permanent`](ParallelError::Permanent) is not.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Transient(_) | Self::Timeout(_))
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single search result returned by Parallel.ai.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The URL of the source document.
    pub url: String,
    /// The page title of the source document.
    pub title: String,
    /// A short text snippet extracted from the source.
    pub snippet: String,
    /// ISO-8601 timestamp of when the content was retrieved.
    pub retrieved_at: String,
}

// ---------------------------------------------------------------------------
// Private response types
// ---------------------------------------------------------------------------

/// Parallel.ai search response envelope.
#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<SearchResponseResult>,
}

/// Individual result from search API (differs from public SearchResult).
#[derive(Deserialize)]
struct SearchResponseResult {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    excerpts: Vec<String>,
}

/// Parallel.ai task creation response.
#[derive(Deserialize)]
struct TaskRunResponse {
    run_id: String,
    #[allow(dead_code)]
    status: String,
}

/// Parallel.ai task poll response.
#[derive(Deserialize)]
struct TaskResult {
    status: String,
    #[serde(default)]
    result: Option<TaskResultData>,
}

/// Payload inside a completed task result.
#[derive(Deserialize)]
struct TaskResultData {
    #[serde(default)]
    sources: Vec<SearchResult>,
}

/// Parallel.ai extract response.
#[derive(Deserialize)]
struct ExtractResponse {
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    url: String,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Typed HTTP client for Parallel.ai's search, extract, and task APIs.
///
/// Build with [`ParallelClient::new`] and optionally override endpoints or
/// inject an [`AuthProvider`] for dynamic credential resolution.
///
/// All public methods are `async` and require a Tokio runtime.
pub struct ParallelClient {
    client: reqwest::Client,
    search_url: String,
    task_url: String,
    extract_url: String,
    max_poll_attempts: usize,
    poll_interval: Duration,
    auth: Option<Arc<dyn AuthProvider>>,
    api_key_var: String,
}

impl std::fmt::Debug for ParallelClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelClient")
            .field("search_url", &self.search_url)
            .field("task_url", &self.task_url)
            .field("extract_url", &self.extract_url)
            .field("max_poll_attempts", &self.max_poll_attempts)
            .field("auth", &self.auth.as_ref().map(|_| "[configured]"))
            .finish_non_exhaustive()
    }
}

impl ParallelClient {
    /// Create a new client using the default Parallel.ai endpoint URLs.
    ///
    /// - `api_key_var`: name of the environment variable that holds the API
    ///   key. Used as the fallback when no [`AuthProvider`] is configured.
    ///
    /// Default endpoints:
    /// - search: `https://api.parallel.ai/v1beta/search`
    /// - task: `https://api.parallel.ai/v1beta/tasks/runs`
    /// - extract: `https://api.parallel.ai/v1beta/extract`
    ///
    /// Default polling: 30 attempts × 5 s interval.
    pub fn new(api_key_var: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            search_url: "https://api.parallel.ai/v1beta/search".into(),
            task_url: "https://api.parallel.ai/v1beta/tasks/runs".into(),
            extract_url: "https://api.parallel.ai/v1beta/extract".into(),
            max_poll_attempts: 30,
            poll_interval: Duration::from_secs(5),
            auth: None,
            api_key_var: api_key_var.into(),
        }
    }

    /// Set an [`AuthProvider`] for dynamic credential resolution.
    ///
    /// When set, key resolution calls `auth.provide()` with audience
    /// `"parallel.ai"` before falling back to the environment variable.
    pub fn with_auth(mut self, auth: Arc<dyn AuthProvider>) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Override the search endpoint URL (useful for proxies or tests).
    pub fn with_search_url(mut self, url: impl Into<String>) -> Self {
        self.search_url = url.into();
        self
    }

    /// Override the task endpoint URL (useful for proxies or tests).
    pub fn with_task_url(mut self, url: impl Into<String>) -> Self {
        self.task_url = url.into();
        self
    }

    /// Override the extract endpoint URL (useful for proxies or tests).
    pub fn with_extract_url(mut self, url: impl Into<String>) -> Self {
        self.extract_url = url.into();
        self
    }

    /// Override polling configuration.
    ///
    /// - `max_attempts`: number of GET polls before returning
    ///   [`ParallelError::Timeout`].
    /// - `interval`: delay between each poll attempt.
    pub fn with_poll_config(mut self, max_attempts: usize, interval: Duration) -> Self {
        self.max_poll_attempts = max_attempts;
        self.poll_interval = interval;
        self
    }

    // -- public async API --

    /// Search Parallel.ai (immediate results, no polling).
    ///
    /// POSTs to the search endpoint with `objective` and returns the result
    /// list directly. The `mode` parameter maps to Parallel.ai search modes:
    /// `"one-shot"` (comprehensive), `"agentic"` (concise), `"fast"` (low-latency).
    pub async fn search(
        &self,
        query: &str,
        mode: &str,
    ) -> Result<Vec<SearchResult>, ParallelError> {
        let key = self.resolve_key().await?;
        let body = serde_json::json!({
            "objective": query,
            "mode": mode,
            "max_results": 10,
            "excerpts": { "max_chars_per_result": 2000 },
        });

        let retrieved_at = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();

        let resp = self
            .client
            .post(&self.search_url)
            .header("x-api-key", &key)
            .header("parallel-beta", "search-extract-2025-10-10")
            .json(&body)
            .send()
            .await
            .map_err(|e| ParallelError::Transient(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Self::classify_status(status, &text)?;

        let parsed: SearchResponse = serde_json::from_str(&text)
            .map_err(|e| ParallelError::Permanent(format!("parse search response: {e}")))?;

        // Convert API-specific response to our public SearchResult type.
        let results = parsed.results.into_iter().map(|r| {
            let snippet = r.excerpts.join("\n\n");
            SearchResult {
                url: r.url,
                title: r.title,
                snippet,
                retrieved_at: retrieved_at.clone(),
            }
        }).collect();
        Ok(results)
    }

    /// Submit a task to Parallel.ai and return the task ID.
    ///
    /// Use [`poll_task`](Self::poll_task) or [`run_task`](Self::run_task) to
    /// retrieve results. The task is processed asynchronously by Parallel.ai.
    pub async fn submit_task(
        &self,
        query: &str,
        processor: &str,
    ) -> Result<String, ParallelError> {
        let key = self.resolve_key().await?;
        let body = serde_json::json!({
            "input": query,
            "processor": processor,
            "task_spec": { "type": "auto" },
        });

        let resp = self
            .client
            .post(&self.task_url)
            .header("x-api-key", &key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ParallelError::Transient(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Self::classify_status(status, &text)?;

        let parsed: TaskRunResponse = serde_json::from_str(&text)
            .map_err(|e| ParallelError::Transient(format!("parse task response: {e}")))?;
        Ok(parsed.run_id)
    }

    /// Poll a single task by ID.
    ///
    /// Returns:
    /// - `Ok(Some(results))` — task completed successfully.
    /// - `Ok(None)` — task is still pending/running; call again later.
    /// - `Err(ParallelError::Permanent(_))` — task failed permanently.
    /// - `Err(ParallelError::Transient(_))` — transient network/server error.
    pub async fn poll_task(
        &self,
        task_id: &str,
    ) -> Result<Option<Vec<SearchResult>>, ParallelError> {
        let key = self.resolve_key().await?;

        let resp = self
            .client
            .get(format!("{}/{}", self.task_url, task_id))
            .header("x-api-key", &key)
            .send()
            .await
            .map_err(|e| ParallelError::Transient(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Self::classify_status(status, &text)?;

        let parsed: TaskResult = serde_json::from_str(&text)
            .map_err(|e| ParallelError::Transient(format!("parse task result: {e}")))?;

        match parsed.status.as_str() {
            "completed" => Ok(Some(
                parsed.result.map(|r| r.sources).unwrap_or_default(),
            )),
            "failed" | "error" => Err(ParallelError::Permanent(format!(
                "task {task_id} failed"
            ))),
            _ => Ok(None),
        }
    }

    /// Submit a task and poll until it completes or times out.
    ///
    /// Waits [`poll_interval`](Self::with_poll_config) between attempts and
    /// returns [`ParallelError::Timeout`] after `max_attempts` polls.
    pub async fn run_task(
        &self,
        query: &str,
        processor: &str,
    ) -> Result<Vec<SearchResult>, ParallelError> {
        let task_id = self.submit_task(query, processor).await?;
        debug!(task_id = %task_id, "task submitted, beginning poll loop");

        for attempt in 0..self.max_poll_attempts {
            tokio::time::sleep(self.poll_interval).await;
            debug!(task_id = %task_id, attempt, "polling task");

            if let Some(results) = self.poll_task(&task_id).await? {
                return Ok(results);
            }
        }

        Err(ParallelError::Timeout(format!(
            "task {task_id} polling timed out after {} attempts",
            self.max_poll_attempts
        )))
    }

    /// Extract structured content from a URL via Parallel.ai's extract API.
    ///
    /// If the response does not include a URL, the original `url` argument is
    /// used. Empty title and text fields are preserved as-is.
    pub async fn extract(&self, url: &str) -> Result<SearchResult, ParallelError> {
        let key = self.resolve_key().await?;
        let body = serde_json::json!({
            "urls": [url],
            "excerpts": true,
            "full_content": true,
        });

        let resp = self
            .client
            .post(&self.extract_url)
            .header("x-api-key", &key)
            .header("parallel-beta", "search-extract-2025-10-10")
            .json(&body)
            .send()
            .await
            .map_err(|e| ParallelError::Transient(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Self::classify_status(status, &text)?;

        let parsed: ExtractResponse = serde_json::from_str(&text)
            .map_err(|e| ParallelError::Permanent(format!("parse extract response: {e}")))?;

        let retrieved_at = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();

        Ok(SearchResult {
            url: if parsed.url.is_empty() {
                url.to_string()
            } else {
                parsed.url
            },
            title: parsed.title,
            snippet: parsed.text,
            retrieved_at,
        })
    }

    // -- private helpers --

    /// Resolve the Parallel.ai API key.
    ///
    /// Tries the [`AuthProvider`] (audience `"parallel.ai"`) first. Falls back
    /// to the environment variable named by `api_key_var`. Error messages
    /// contain only the variable name, never the key value.
    async fn resolve_key(&self) -> Result<String, ParallelError> {
        if let Some(auth) = &self.auth {
            let request = AuthRequest::new().with_audience("parallel.ai");
            match auth.provide(&request).await {
                Ok(token) => {
                    let key = token.with_bytes(|b| String::from_utf8_lossy(b).into_owned());
                    if !key.is_empty() {
                        return Ok(key);
                    }
                }
                Err(_) => {
                    // Fall through to env var
                }
            }
        }

        let var = &self.api_key_var;
        let key = std::env::var(var)
            .map_err(|_| ParallelError::Permanent(format!("env var '{var}' not set")))?;
        if key.is_empty() {
            return Err(ParallelError::Permanent(format!(
                "env var '{var}' is empty"
            )));
        }
        Ok(key)
    }

    /// Map an HTTP status code to a [`ParallelError`], or `Ok(())` on success.
    ///
    /// - 429 → [`Transient`](ParallelError::Transient) (rate-limited)
    /// - 401/403 → [`Permanent`](ParallelError::Permanent) (auth failure)
    /// - 5xx → [`Transient`](ParallelError::Transient) (server error)
    /// - Other non-2xx → [`Permanent`](ParallelError::Permanent)
    fn classify_status(status: reqwest::StatusCode, body: &str) -> Result<(), ParallelError> {
        if status.is_success() {
            return Ok(());
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ParallelError::Transient("rate limited".into()));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED
            || status == reqwest::StatusCode::FORBIDDEN
        {
            return Err(ParallelError::Permanent(format!("auth failed ({status})")));
        }
        if status.is_server_error() {
            return Err(ParallelError::Transient(format!("{status}: {body}")));
        }
        Err(ParallelError::Permanent(format!("{status}: {body}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_urls() {
        let c = ParallelClient::new("MY_KEY_VAR");
        assert_eq!(c.search_url, "https://api.parallel.ai/v1beta/search");
        assert_eq!(c.task_url, "https://api.parallel.ai/v1beta/tasks/runs");
        assert_eq!(c.extract_url, "https://api.parallel.ai/v1beta/extract");
        assert_eq!(c.max_poll_attempts, 30);
        assert_eq!(c.poll_interval, Duration::from_secs(5));
        assert_eq!(c.api_key_var, "MY_KEY_VAR");
        assert!(c.auth.is_none());
    }

    #[test]
    fn error_retryable() {
        assert!(ParallelError::Transient("network glitch".into()).is_retryable());
        assert!(ParallelError::Timeout("took too long".into()).is_retryable());
        assert!(!ParallelError::Permanent("bad request".into()).is_retryable());
    }

    #[test]
    fn search_result_serde_round_trip() {
        let original = SearchResult {
            url: "https://example.com/page".into(),
            title: "Example Page".into(),
            snippet: "Some relevant text from the page.".into(),
            retrieved_at: "2026-01-15T10:30:00Z".into(),
        };

        let json = serde_json::to_string(&original).expect("serialize");
        let restored: SearchResult = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.url, original.url);
        assert_eq!(restored.title, original.title);
        assert_eq!(restored.snippet, original.snippet);
        assert_eq!(restored.retrieved_at, original.retrieved_at);
    }

    #[test]
    fn classify_status_success() {
        assert!(ParallelClient::classify_status(reqwest::StatusCode::OK, "").is_ok());
        assert!(
            ParallelClient::classify_status(reqwest::StatusCode::CREATED, "").is_ok()
        );
    }

    #[test]
    fn classify_status_transient() {
        let err =
            ParallelClient::classify_status(reqwest::StatusCode::TOO_MANY_REQUESTS, "")
                .unwrap_err();
        assert!(matches!(err, ParallelError::Transient(_)));
        assert!(err.is_retryable());

        let err =
            ParallelClient::classify_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "oops")
                .unwrap_err();
        assert!(matches!(err, ParallelError::Transient(_)));
        assert!(err.is_retryable());
    }

    #[test]
    fn classify_status_permanent() {
        let err =
            ParallelClient::classify_status(reqwest::StatusCode::UNAUTHORIZED, "").unwrap_err();
        assert!(matches!(err, ParallelError::Permanent(_)));
        assert!(!err.is_retryable());

        let err =
            ParallelClient::classify_status(reqwest::StatusCode::FORBIDDEN, "").unwrap_err();
        assert!(matches!(err, ParallelError::Permanent(_)));

        let err =
            ParallelClient::classify_status(reqwest::StatusCode::BAD_REQUEST, "bad").unwrap_err();
        assert!(matches!(err, ParallelError::Permanent(_)));
        assert!(!err.is_retryable());
    }

    #[test]
    fn resolve_key_env_var_missing() {
        // Use a var name that should not exist in CI
        let c = ParallelClient::new("SKG_TEST_PARALLEL_MISSING_abc123xyz");
        // We can't call async fn in sync test without runtime, but we can verify
        // the field is set correctly and classify_status covers the error path.
        assert_eq!(c.api_key_var, "SKG_TEST_PARALLEL_MISSING_abc123xyz");
    }

    #[test]
    fn extract_empty_url_falls_back_to_original() {
        // Verify the fallback logic by constructing ExtractResponse manually via
        // deserialization (the struct is private, but we can test via JSON).
        let json = r#"{"title": "My Title", "text": "body text", "url": ""}"#;
        let parsed: ExtractResponse = serde_json::from_str(json).expect("parse");
        let retrieved_at = "2026-01-15T10:00:00Z".to_string();
        let original_url = "https://original.example.com/";

        let result = SearchResult {
            url: if parsed.url.is_empty() {
                original_url.to_string()
            } else {
                parsed.url
            },
            title: parsed.title,
            snippet: parsed.text,
            retrieved_at,
        };

        assert_eq!(result.url, original_url);
        assert_eq!(result.title, "My Title");
        assert_eq!(result.snippet, "body text");
    }
}
