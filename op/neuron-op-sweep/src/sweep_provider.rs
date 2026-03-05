//! Production [`ResearchProvider`] composing Parallel.ai search and Anthropic LLM.
//!
//! `search()` calls Parallel.ai's search API (web research tool).
//! `compare()` and `plan()` call Anthropic's Messages API (LLM reasoning).
//!
//! See `rules/10-sweep-system-backends.md` for the design rationale.

use async_trait::async_trait;
use neuron_auth::{AuthProvider, AuthRequest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::provider::{ResearchProvider, ResearchResult, SweepError};
use crate::types::{ProcessorTier, SweepVerdict, VerdictStatus};

// ---------------------------------------------------------------------------
// Internal response types
// ---------------------------------------------------------------------------

/// Parallel.ai search response envelope.
#[derive(Deserialize)]
struct SearchResponse {
    results: Vec<ResearchResult>,
}

/// Anthropic Messages API request.
#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
}

/// Anthropic message.
#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

/// Anthropic Messages API response (subset).
#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    usage: AnthropicUsage,
}

/// Anthropic content block.
#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

/// Anthropic token usage.
#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

/// LLM response parsed as a plan.
#[derive(Deserialize)]
struct PlanResponse {
    queries: Vec<String>,
}

/// Parallel.ai task run request.
#[derive(Serialize)]
struct TaskRunRequest {
    query: String,
    processor: String,
}

/// Parallel.ai task run response (initial).
#[derive(Deserialize)]
struct TaskRunResponse {
    id: String,
    #[allow(dead_code)]
    status: String,
}

/// Parallel.ai task result (polling).
#[derive(Deserialize)]
struct TaskResult {
    status: String,
    #[serde(default)]
    result: Option<TaskResultData>,
}

/// Parallel.ai task result data.
#[derive(Deserialize)]
struct TaskResultData {
    #[serde(default)]
    sources: Vec<ResearchResult>,
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
// SweepProvider
// ---------------------------------------------------------------------------

/// Production [`ResearchProvider`] that composes two backends:
///
/// - **Parallel.ai** for `search()` — web research and evidence gathering.
/// - **Anthropic Claude** for `compare()` and `plan()` — LLM reasoning.
///
/// API keys are resolved via an optional [`AuthProvider`] first, then fall
/// back to environment variables at call time. Error messages contain only
/// variable names, never key material.
///
/// # Auth resolution order
///
/// 1. If an `AuthProvider` is configured (via [`with_auth`](Self::with_auth)),
///    call `provider.provide()` with the appropriate audience.
/// 2. If no `AuthProvider` or the provider returns an error, fall back to
///    the configured environment variable.
///
/// This allows zero-behavior-change adoption: existing callers that set env
/// vars continue to work. New callers can inject OAuth tokens, vault-resolved
/// secrets, or any other [`AuthProvider`] implementation.
#[derive(Clone)]
pub struct SweepProvider {
    client: reqwest::Client,
    /// Env var for Parallel.ai API key.
    parallel_key_var: String,
    /// Parallel.ai search endpoint.
    parallel_url: String,
    /// Env var for Anthropic API key.
    anthropic_key_var: String,
    /// Anthropic Messages API endpoint.
    anthropic_url: String,
    /// Anthropic API version header.
    anthropic_version: String,
    /// Model to use for LLM calls (e.g. `claude-sonnet-4-20250514`).
    model: String,
    /// Max tokens for LLM responses.
    max_tokens: usize,
    /// Parallel.ai task runs endpoint.
    parallel_task_url: String,
    /// Parallel.ai extract endpoint.
    parallel_extract_url: String,
    /// Max poll attempts for async tasks.
    max_poll_attempts: usize,
    /// Poll interval in seconds.
    poll_interval_secs: u64,
    /// Optional auth provider for dynamic credential resolution.
    auth: Option<Arc<dyn AuthProvider>>,
}

impl std::fmt::Debug for SweepProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SweepProvider")
            .field("parallel_url", &self.parallel_url)
            .field("anthropic_url", &self.anthropic_url)
            .field("model", &self.model)
            .field("auth", &self.auth.as_ref().map(|_| "[configured]"))
            .finish_non_exhaustive()
    }
}

impl SweepProvider {
    /// Create a provider with default API endpoints.
    ///
    /// - `parallel_key_var`: env var holding the Parallel.ai API key.
    /// - `anthropic_key_var`: env var holding the Anthropic API key.
    /// - `model`: Anthropic model identifier.
    pub fn new(
        parallel_key_var: impl Into<String>,
        anthropic_key_var: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            parallel_key_var: parallel_key_var.into(),
            parallel_url: "https://api.parallel.ai/v1/search".into(),
            anthropic_key_var: anthropic_key_var.into(),
            anthropic_url: "https://api.anthropic.com/v1/messages".into(),
            anthropic_version: "2023-06-01".into(),
            model: model.into(),
            max_tokens: 4096,
            parallel_task_url: "https://api.parallel.ai/v1/tasks/runs".into(),
            parallel_extract_url: "https://api.parallel.ai/v1beta/extract".into(),
            max_poll_attempts: 30,
            poll_interval_secs: 5,
            auth: None,
        }
    }

    /// Override API endpoints (for testing or proxies).
    pub fn with_urls(
        mut self,
        parallel_url: impl Into<String>,
        anthropic_url: impl Into<String>,
    ) -> Self {
        self.parallel_url = parallel_url.into();
        self.anthropic_url = anthropic_url.into();
        self
    }

    /// Override max tokens for LLM responses.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override task run and extract endpoints (for testing or proxies).
    pub fn with_task_urls(
        mut self,
        task_url: impl Into<String>,
        extract_url: impl Into<String>,
    ) -> Self {
        self.parallel_task_url = task_url.into();
        self.parallel_extract_url = extract_url.into();
        self
    }

    /// Set an [`AuthProvider`] for dynamic credential resolution.
    ///
    /// When set, key resolution tries `auth.provide()` before falling back
    /// to environment variables. This enables OAuth tokens, vault secrets,
    /// or any other credential source without changing call sites.
    pub fn with_auth(mut self, auth: Arc<dyn AuthProvider>) -> Self {
        self.auth = Some(auth);
        self
    }

    // -- private helpers --

    /// Resolve the Parallel.ai API key.
    ///
    /// Tries [`AuthProvider`] first (audience: `parallel.ai`), then falls
    /// back to the configured environment variable.
    async fn resolve_parallel_key(&self) -> Result<String, SweepError> {
        self.resolve_key("parallel.ai", &self.parallel_key_var).await
    }

    /// Resolve the Anthropic API key.
    ///
    /// Tries [`AuthProvider`] first (audience: `api.anthropic.com`), then
    /// falls back to the configured environment variable.
    async fn resolve_anthropic_key(&self) -> Result<String, SweepError> {
        self.resolve_key("api.anthropic.com", &self.anthropic_key_var).await
    }

    /// Resolve a key by trying the auth provider first, then env var.
    async fn resolve_key(&self, audience: &str, env_var: &str) -> Result<String, SweepError> {
        if let Some(auth) = &self.auth {
            let request = AuthRequest::new().with_audience(audience);
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
        resolve_env_key(env_var)
    }

    /// Map a ProcessorTier to the Parallel.ai processor string.
    fn tier_str(tier: &ProcessorTier) -> &'static str {
        match tier {
            ProcessorTier::Base => "base",
            ProcessorTier::Core => "core",
            ProcessorTier::Ultra => "ultra",
        }
    }

    /// Call Anthropic Messages API and return the text response.
    async fn anthropic_call(
        &self,
        system: &str,
        user_content: &str,
    ) -> Result<(String, usize), SweepError> {
        let key = self.resolve_anthropic_key().await?;
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system: Some(system.to_string()),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: user_content.to_string(),
            }],
        };

        let resp = self
            .client
            .post(&self.anthropic_url)
            .header("x-api-key", &key)
            .header("anthropic-version", &self.anthropic_version)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| SweepError::Transient(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(SweepError::Transient("rate limited".into()));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(SweepError::Permanent(format!(
                "auth failed ({})",
                self.anthropic_key_var
            )));
        }
        if status.is_server_error() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Transient(format!("{status}: {body}")));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Permanent(format!("{status}: {body}")));
        }

        let api_resp: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| SweepError::LlmFailure(e.to_string()))?;

        let total_tokens = api_resp.usage.input_tokens + api_resp.usage.output_tokens;

        let text = api_resp
            .content
            .into_iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return Err(SweepError::LlmFailure("empty response".into()));
        }

        Ok((text, total_tokens))
    }
}

/// Resolve an API key from an environment variable.
///
/// Returns `SweepError::Permanent` with the variable name (not the key)
/// if the variable is unset or empty.
fn resolve_env_key(var_name: &str) -> Result<String, SweepError> {
    let key = std::env::var(var_name)
        .map_err(|_| SweepError::Permanent(format!("env var '{}' not set", var_name)))?;
    if key.is_empty() {
        return Err(SweepError::Permanent(format!(
            "env var '{}' is empty",
            var_name
        )));
    }
    Ok(key)
}

#[async_trait]
impl ResearchProvider for SweepProvider {
    async fn search(
        &self,
        query: &str,
        processor: ProcessorTier,
    ) -> Result<Vec<ResearchResult>, SweepError> {
        let key = self.resolve_parallel_key().await?;

        let body = serde_json::json!({
            "query": query,
            "processor": Self::tier_str(&processor),
        });

        let resp = self
            .client
            .post(&self.parallel_url)
            .header("Authorization", format!("Bearer {key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| SweepError::Transient(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(SweepError::Transient("rate limited".into()));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(SweepError::Permanent(format!(
                "auth failed ({})",
                self.parallel_key_var
            )));
        }
        if status.is_server_error() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Transient(format!("{status}: {body}")));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Permanent(format!("{status}: {body}")));
        }

        let search_resp: SearchResponse = resp
            .json()
            .await
            .map_err(|e| SweepError::LlmFailure(format!("parse search response: {e}")))?;

        Ok(search_resp.results)
    }

    async fn compare(
        &self,
        system: &str,
        context: &str,
        research: &str,
        current_decision: &str,
    ) -> Result<SweepVerdict, SweepError> {
        let user_content = format!(
            "## Context\n{context}\n\n## Research Findings\n{research}\n\n## Current Decision\n{current_decision}\n\nProduce a JSON verdict."
        );

        let (text, _tokens) = self.anthropic_call(system, &user_content).await?;

        // Extract JSON from the response (may be wrapped in markdown fences)
        let json_str = extract_json_block(&text);
        serde_json::from_str::<SweepVerdict>(json_str)
            .map_err(|e| SweepError::LlmFailure(format!("parse verdict: {e}")))
    }

    async fn plan(
        &self,
        decision_id: &str,
        decision_text: &str,
        previous_verdict: Option<&VerdictStatus>,
    ) -> Result<Option<Vec<String>>, SweepError> {
        let verdict_str = match previous_verdict {
            Some(v) => serde_json::to_string(v).unwrap_or_default(),
            None => "none".into(),
        };

        let system = "Generate 3-5 targeted research queries for evaluating an architectural \
            decision against current industry practice. Return ONLY a JSON object: \
            {\"queries\": [\"query1\", \"query2\", ...]}";

        let user_content = format!(
            "Decision ID: {decision_id}\nPrevious verdict: {verdict_str}\n\n{decision_text}"
        );

        // Plan failures are non-fatal — return None to fall back to keyword queries
        match self.anthropic_call(system, &user_content).await {
            Ok((text, _tokens)) => {
                let json_str = extract_json_block(&text);
                match serde_json::from_str::<PlanResponse>(json_str) {
                    Ok(plan) if !plan.queries.is_empty() => Ok(Some(plan.queries)),
                    _ => Ok(None),
                }
            }
            Err(_) => Ok(None),
        }
    }

    async fn research(
        &self,
        query: &str,
        processor: ProcessorTier,
    ) -> Result<Vec<ResearchResult>, SweepError> {
        let key = self.resolve_parallel_key().await?;
        let req = TaskRunRequest {
            query: query.to_string(),
            processor: Self::tier_str(&processor).to_string(),
        };
        let resp = self
            .client
            .post(&self.parallel_task_url)
            .header("Authorization", format!("Bearer {key}"))
            .json(&req)
            .send()
            .await
            .map_err(|e| SweepError::Transient(e.to_string()))?;
        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(SweepError::Transient("rate limited".into()));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(SweepError::Permanent(format!(
                "auth failed ({})",
                self.parallel_key_var
            )));
        }
        if status.is_server_error() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Transient(format!("{status}: {body}")));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Permanent(format!("{status}: {body}")));
        }
        let task_resp: TaskRunResponse = resp
            .json()
            .await
            .map_err(|e| SweepError::Transient(format!("parse task response: {e}")))?;
        let task_id = task_resp.id;
        for _ in 0..self.max_poll_attempts {
            tokio::time::sleep(std::time::Duration::from_secs(self.poll_interval_secs)).await;
            let poll_resp = self
                .client
                .get(format!("{}/{}", self.parallel_task_url, task_id))
                .header("Authorization", format!("Bearer {key}"))
                .send()
                .await
                .map_err(|e| SweepError::Transient(e.to_string()))?;
            let poll_status = poll_resp.status();
            if poll_status.is_server_error() {
                let body = poll_resp.text().await.unwrap_or_default();
                return Err(SweepError::Transient(format!("{poll_status}: {body}")));
            }
            if !poll_status.is_success() {
                let body = poll_resp.text().await.unwrap_or_default();
                return Err(SweepError::Permanent(format!("{poll_status}: {body}")));
            }
            let task_result: TaskResult = poll_resp
                .json()
                .await
                .map_err(|e| SweepError::Transient(format!("parse task result: {e}")))?;
            match task_result.status.as_str() {
                "completed" => {
                    return Ok(task_result.result.map(|r| r.sources).unwrap_or_default());
                }
                "failed" | "error" => {
                    return Err(SweepError::Permanent(format!("task {task_id} failed")));
                }
                _ => {}
            }
        }
        Err(SweepError::Transient(format!(
            "task {task_id} polling timed out after {} attempts",
            self.max_poll_attempts
        )))
    }

    async fn extract(
        &self,
        url: &str,
    ) -> Result<Option<ResearchResult>, SweepError> {
        let key = self.resolve_parallel_key().await?;
        let body = serde_json::json!({ "url": url });
        let resp = self
            .client
            .post(&self.parallel_extract_url)
            .header("Authorization", format!("Bearer {key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| SweepError::Transient(e.to_string()))?;
        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(SweepError::Transient("rate limited".into()));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(SweepError::Permanent(format!(
                "auth failed ({})",
                self.parallel_key_var
            )));
        }
        if status.is_server_error() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Transient(format!("{status}: {body}")));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SweepError::Permanent(format!("{status}: {body}")));
        }
        let extract_resp: ExtractResponse = resp
            .json()
            .await
            .map_err(|e| SweepError::Permanent(format!("parse extract response: {e}")))?;
        let retrieved_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        Ok(Some(ResearchResult {
            url: if extract_resp.url.is_empty() {
                url.to_string()
            } else {
                extract_resp.url
            },
            title: extract_resp.title,
            snippet: extract_resp.text,
            retrieved_at,
        }))
    }
}

/// Extract the first JSON block from LLM output, stripping markdown fences.
fn extract_json_block(text: &str) -> &str {
    let trimmed = text.trim();
    // Try to extract from ```json ... ``` fences
    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + 7..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim();
        }
    }
    // Try ``` ... ``` fences
    if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim();
        }
    }
    // Try to find raw JSON object
    if let Some(start) = trimmed.find('{')
        && let Some(end) = trimmed.rfind('}')
    {
        return &trimmed[start..=end];
    }
    trimmed
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use neuron_auth::{AuthError, AuthToken};

    #[test]
    fn tier_str_mapping() {
        assert_eq!(SweepProvider::tier_str(&ProcessorTier::Base), "base");
        assert_eq!(SweepProvider::tier_str(&ProcessorTier::Core), "core");
        assert_eq!(SweepProvider::tier_str(&ProcessorTier::Ultra), "ultra");
    }

    #[test]
    fn resolve_env_key_missing() {
        let err = resolve_env_key("NEURON_TEST_MISSING_KEY_a7b3c9").unwrap_err();
        assert!(matches!(err, SweepError::Permanent(_)));
        let msg = err.to_string();
        assert!(msg.contains("NEURON_TEST_MISSING_KEY_a7b3c9"));
        // Must not contain any actual secret
        assert!(!msg.contains("sk-"));
    }

    #[test]
    fn resolve_env_key_present() {
        let var = "NEURON_TEST_KEY_PRESENT_x9y2z1";
        unsafe { std::env::set_var(var, "test-key-value") };
        let key = resolve_env_key(var).expect("should succeed");
        assert_eq!(key, "test-key-value");
        unsafe { std::env::remove_var(var) };
    }

    #[test]
    fn resolve_env_key_empty() {
        let var = "NEURON_TEST_KEY_EMPTY_m4n5o6";
        unsafe { std::env::set_var(var, "") };
        let err = resolve_env_key(var).unwrap_err();
        assert!(matches!(err, SweepError::Permanent(_)));
        assert!(err.to_string().contains("empty"));
        unsafe { std::env::remove_var(var) };
    }

    #[test]
    fn constructor_default_urls() {
        let p = SweepProvider::new("PAR_KEY", "ANT_KEY", "claude-sonnet-4-20250514");
        assert_eq!(p.parallel_url, "https://api.parallel.ai/v1/search");
        assert_eq!(p.anthropic_url, "https://api.anthropic.com/v1/messages");
        assert_eq!(p.model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn constructor_custom_urls() {
        let p = SweepProvider::new("P", "A", "m")
            .with_urls("http://local:8080/search", "http://local:9090/messages");
        assert_eq!(p.parallel_url, "http://local:8080/search");
        assert_eq!(p.anthropic_url, "http://local:9090/messages");
    }

    #[test]
    fn search_response_deserialization() {
        let json = r#"{"results": [{"url": "https://ex.com", "title": "T", "snippet": "S", "retrieved_at": "2026-01-01T00:00:00Z"}]}"#;
        let resp: SearchResponse = serde_json::from_str(json).expect("parse");
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].url, "https://ex.com");
    }

    #[test]
    fn plan_response_deserialization() {
        let json = r#"{"queries": ["q1", "q2", "q3"]}"#;
        let resp: PlanResponse = serde_json::from_str(json).expect("parse");
        assert_eq!(resp.queries.len(), 3);
    }

    #[test]
    fn extract_json_block_fenced() {
        let text = "Here is the verdict:\n```json\n{\"status\": \"confirmed\"}\n```\nDone.";
        assert_eq!(extract_json_block(text), "{\"status\": \"confirmed\"}");
    }

    #[test]
    fn extract_json_block_bare() {
        let text = "{\"queries\": [\"a\", \"b\"]}";
        assert_eq!(extract_json_block(text), "{\"queries\": [\"a\", \"b\"]}");
    }

    #[test]
    fn extract_json_block_with_preamble() {
        let text = "Sure! Here is the result:\n{\"status\": \"refined\"}";
        assert_eq!(extract_json_block(text), "{\"status\": \"refined\"}");
    }

    #[test]
    fn extract_json_block_generic_fence() {
        let text = "```\n{\"x\": 1}\n```";
        assert_eq!(extract_json_block(text), "{\"x\": 1}");
    }

    #[test]
    fn task_run_request_serialization() {
        let req = TaskRunRequest {
            query: "rust async patterns".to_string(),
            processor: "base".to_string(),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("\"query\""));
        assert!(json.contains("\"processor\""));
        assert!(json.contains("rust async patterns"));
        assert!(json.contains("base"));
    }

    #[test]
    fn task_run_response_deserialization() {
        let json = r#"{"id": "task-abc123", "status": "pending"}"#;
        let resp: TaskRunResponse = serde_json::from_str(json).expect("parse");
        assert_eq!(resp.id, "task-abc123");
        assert_eq!(resp.status, "pending");
    }

    #[test]
    fn task_result_deserialization_completed() {
        let json = r#"{"status": "completed", "result": {"sources": [{"url": "https://ex.com", "title": "T", "snippet": "S", "retrieved_at": "2026-01-01T00:00:00Z"}]}}"#;
        let result: TaskResult = serde_json::from_str(json).expect("parse");
        assert_eq!(result.status, "completed");
        assert_eq!(result.result.unwrap().sources.len(), 1);
    }

    #[test]
    fn task_result_deserialization_failed() {
        let json = r#"{"status": "failed"}"#;
        let result: TaskResult = serde_json::from_str(json).expect("parse");
        assert_eq!(result.status, "failed");
        assert!(result.result.is_none());
    }

    #[test]
    fn extract_response_deserialization() {
        let json = r#"{"title": "My Page", "text": "Content here", "url": "https://ex.com/page"}"#;
        let resp: ExtractResponse = serde_json::from_str(json).expect("parse");
        assert_eq!(resp.title, "My Page");
        assert_eq!(resp.text, "Content here");
        assert_eq!(resp.url, "https://ex.com/page");
    }

    #[test]
    fn constructor_default_task_urls() {
        let p = SweepProvider::new("PAR_KEY", "ANT_KEY", "claude-sonnet-4-20250514");
        assert_eq!(p.parallel_task_url, "https://api.parallel.ai/v1/tasks/runs");
        assert_eq!(p.parallel_extract_url, "https://api.parallel.ai/v1beta/extract");
        assert_eq!(p.max_poll_attempts, 30);
        assert_eq!(p.poll_interval_secs, 5);
    }

    // -- AuthProvider integration tests --

    struct StaticAuthProvider {
        token: Vec<u8>,
    }

    #[async_trait]
    impl AuthProvider for StaticAuthProvider {
        async fn provide(&self, _request: &AuthRequest) -> Result<AuthToken, AuthError> {
            Ok(AuthToken::permanent(self.token.clone()))
        }
    }

    struct FailingAuthProvider;

    #[async_trait]
    impl AuthProvider for FailingAuthProvider {
        async fn provide(&self, _request: &AuthRequest) -> Result<AuthToken, AuthError> {
            Err(AuthError::AuthFailed("intentional failure".into()))
        }
    }

    #[test]
    fn with_auth_sets_provider() {
        let p = SweepProvider::new("P", "A", "m")
            .with_auth(Arc::new(StaticAuthProvider { token: b"tok".to_vec() }));
        assert!(p.auth.is_some());
    }

    #[test]
    fn default_auth_is_none() {
        let p = SweepProvider::new("P", "A", "m");
        assert!(p.auth.is_none());
    }

    #[tokio::test]
    async fn resolve_key_uses_auth_provider_when_set() {
        let p = SweepProvider::new("NEURON_TEST_NONEXISTENT_zq9", "A", "m")
            .with_auth(Arc::new(StaticAuthProvider {
                token: b"auth-provided-key".to_vec(),
            }));
        let key = p.resolve_parallel_key().await.unwrap();
        assert_eq!(key, "auth-provided-key");
    }

    #[tokio::test]
    async fn resolve_key_falls_back_to_env_on_auth_failure() {
        let var = "NEURON_TEST_AUTH_FALLBACK_k3j";
        unsafe { std::env::set_var(var, "env-key-value") };
        let p = SweepProvider::new(var, "A", "m")
            .with_auth(Arc::new(FailingAuthProvider));
        let key = p.resolve_parallel_key().await.unwrap();
        assert_eq!(key, "env-key-value");
        unsafe { std::env::remove_var(var) };
    }

    #[tokio::test]
    async fn resolve_key_env_var_when_no_auth() {
        let var = "NEURON_TEST_NO_AUTH_r8p";
        unsafe { std::env::set_var(var, "plain-env-key") };
        let p = SweepProvider::new(var, "A", "m");
        let key = p.resolve_parallel_key().await.unwrap();
        assert_eq!(key, "plain-env-key");
        unsafe { std::env::remove_var(var) };
    }

    #[tokio::test]
    async fn resolve_key_auth_empty_token_falls_back() {
        let var = "NEURON_TEST_EMPTY_AUTH_w2x";
        unsafe { std::env::set_var(var, "fallback-key") };
        let p = SweepProvider::new(var, "A", "m")
            .with_auth(Arc::new(StaticAuthProvider { token: vec![] }));
        let key = p.resolve_parallel_key().await.unwrap();
        assert_eq!(key, "fallback-key");
        unsafe { std::env::remove_var(var) };
    }

    #[test]
    fn debug_impl_redacts_auth() {
        let p = SweepProvider::new("P", "A", "m")
            .with_auth(Arc::new(StaticAuthProvider { token: b"secret".to_vec() }));
        let debug = format!("{:?}", p);
        assert!(debug.contains("[configured]"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn debug_impl_no_auth() {
        let p = SweepProvider::new("P", "A", "m");
        let debug = format!("{:?}", p);
        assert!(debug.contains("None"));
    }
}
