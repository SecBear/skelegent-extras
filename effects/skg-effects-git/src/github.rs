//! GitHub REST API client for the sweep system.
//!
//! Implements [`GitHubClient`] using reqwest against the GitHub v3 REST API.
//! The API token is resolved from an environment variable at each call to
//! support token rotation without restart.

use async_trait::async_trait;
use serde::Deserialize;

use crate::client::{GitHubClient, PullRequest};
use crate::types::GitError;

/// Production GitHub API client using reqwest.
///
/// Resolves the API token from an environment variable at each call
/// to support token rotation without restart.
#[derive(Debug, Clone)]
pub struct GitHubApiClient {
    client: reqwest::Client,
    token_var: String,
    owner: String,
    repo: String,
    api_base: String,
}

impl GitHubApiClient {
    /// Create a client for `owner/repo`, reading the GitHub token from `token_var` env var.
    pub fn new(
        owner: impl Into<String>,
        repo: impl Into<String>,
        token_var: impl Into<String>,
    ) -> Self {
        Self::with_api_base(owner, repo, token_var, "https://api.github.com")
    }

    /// Create a client with a custom API base URL (for GitHub Enterprise or testing).
    pub fn with_api_base(
        owner: impl Into<String>,
        repo: impl Into<String>,
        token_var: impl Into<String>,
        api_base: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            token_var: token_var.into(),
            owner: owner.into(),
            repo: repo.into(),
            api_base: api_base.into(),
        }
    }
}

impl GitHubApiClient {
    fn resolve_token(&self) -> Result<String, GitError> {
        std::env::var(&self.token_var)
            .map_err(|_| GitError::ApiError(format!("{} not set", self.token_var)))
    }

    fn api_url(&self, path: &str) -> String {
        format!(
            "{}/repos/{}/{}/{}",
            self.api_base, self.owner, self.repo, path
        )
    }

    fn map_status(status: reqwest::StatusCode, body: &str) -> GitError {
        match status.as_u16() {
            404 => GitError::NotFound(body.to_string()),
            422 => GitError::BranchConflict(body.to_string()),
            429 => GitError::RateLimit(body.to_string()),
            _ => GitError::ApiError(format!("{}: {}", status, body)),
        }
    }

    async fn send_json(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, GitError> {
        let resp = builder
            .send()
            .await
            .map_err(|e| GitError::ApiError(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Self::map_status(status, &body));
        }
        Ok(resp)
    }

    fn authed_request(
        &self,
        builder: reqwest::RequestBuilder,
        token: &str,
    ) -> reqwest::RequestBuilder {
        builder
            .bearer_auth(token)
            .header("Accept", "application/vnd.github.v3+json")
            .header("User-Agent", "skg-effects-git/0.4.0")
    }
}

/// GitHub API pull request response (subset of fields we need).
#[derive(Deserialize)]
struct GhPullResponse {
    number: u64,
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    labels: Vec<GhLabel>,
    created_at: String,
}

#[derive(Deserialize)]
struct GhLabel {
    name: String,
}

/// GitHub API issue/PR creation response.
#[derive(Deserialize)]
struct GhCreateResponse {
    number: u64,
    html_url: String,
}

impl GhPullResponse {
    fn into_pull_request(self) -> PullRequest {
        PullRequest {
            number: self.number,
            url: self.html_url,
            body: self.body.unwrap_or_default(),
            labels: self.labels.into_iter().map(|l| l.name).collect(),
            created_at: self.created_at,
        }
    }
}

#[async_trait]
impl GitHubClient for GitHubApiClient {
    async fn find_open_pr(&self, decision_id: &str) -> Result<Option<PullRequest>, GitError> {
        let token = self.resolve_token()?;
        let url = self.api_url(&format!(
            "pulls?state=open&head={}:sweep/{}",
            self.owner, decision_id
        ));
        let req = self.authed_request(self.client.get(&url), &token);
        let resp = self.send_json(req).await?;
        let prs: Vec<GhPullResponse> = resp
            .json()
            .await
            .map_err(|e| GitError::ApiError(e.to_string()))?;
        Ok(prs
            .into_iter()
            .next()
            .map(GhPullResponse::into_pull_request))
    }

    async fn create_pr(
        &self,
        branch: &str,
        title: &str,
        body: &str,
        draft: bool,
        labels: &[String],
    ) -> Result<String, GitError> {
        let token = self.resolve_token()?;
        let url = self.api_url("pulls");
        let payload = serde_json::json!({
            "head": branch,
            "base": "main",
            "title": title,
            "body": body,
            "draft": draft,
        });
        let req = self
            .authed_request(self.client.post(&url), &token)
            .json(&payload);
        let resp = self.send_json(req).await?;
        let created: GhCreateResponse = resp
            .json()
            .await
            .map_err(|e| GitError::ApiError(e.to_string()))?;

        if !labels.is_empty() {
            let labels_url = self.api_url(&format!("issues/{}/labels", created.number));
            let labels_payload = serde_json::json!({ "labels": labels });
            let labels_req = self
                .authed_request(self.client.post(&labels_url), &token)
                .json(&labels_payload);
            // Fire-and-forget style: if labeling fails we still return the PR URL,
            // but we propagate the error to the caller.
            self.send_json(labels_req).await?;
        }

        Ok(created.html_url)
    }

    async fn update_pr(&self, number: u64, body: &str) -> Result<(), GitError> {
        let token = self.resolve_token()?;
        let url = self.api_url(&format!("pulls/{}", number));
        let payload = serde_json::json!({ "body": body });
        let req = self
            .authed_request(self.client.patch(&url), &token)
            .json(&payload);
        self.send_json(req).await?;
        Ok(())
    }

    async fn close_pr(&self, number: u64, comment: &str) -> Result<(), GitError> {
        let token = self.resolve_token()?;

        // Close the PR.
        let url = self.api_url(&format!("pulls/{}", number));
        let payload = serde_json::json!({ "state": "closed" });
        let req = self
            .authed_request(self.client.patch(&url), &token)
            .json(&payload);
        self.send_json(req).await?;

        // Post a comment.
        let comment_url = self.api_url(&format!("issues/{}/comments", number));
        let comment_payload = serde_json::json!({ "body": comment });
        let comment_req = self
            .authed_request(self.client.post(&comment_url), &token)
            .json(&comment_payload);
        self.send_json(comment_req).await?;

        Ok(())
    }

    async fn create_issue(
        &self,
        title: &str,
        body: &str,
        labels: &[String],
    ) -> Result<String, GitError> {
        let token = self.resolve_token()?;
        let url = self.api_url("issues");
        let payload = serde_json::json!({
            "title": title,
            "body": body,
            "labels": labels,
        });
        let req = self
            .authed_request(self.client.post(&url), &token)
            .json(&payload);
        let resp = self.send_json(req).await?;
        let created: GhCreateResponse = resp
            .json()
            .await
            .map_err(|e| GitError::ApiError(e.to_string()))?;
        Ok(created.html_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> GitHubApiClient {
        GitHubApiClient::new("owner", "repo", "SKG_TEST_GH_TOKEN")
    }

    #[test]
    fn map_status_404_is_not_found() {
        let status = reqwest::StatusCode::from_u16(404).unwrap();
        let err = GitHubApiClient::map_status(status, "nope");
        assert!(matches!(err, GitError::NotFound(_)));
    }

    #[test]
    fn map_status_422_is_branch_conflict() {
        let status = reqwest::StatusCode::from_u16(422).unwrap();
        let err = GitHubApiClient::map_status(status, "conflict");
        assert!(matches!(err, GitError::BranchConflict(_)));
    }

    #[test]
    fn map_status_429_is_rate_limit() {
        let status = reqwest::StatusCode::from_u16(429).unwrap();
        let err = GitHubApiClient::map_status(status, "slow down");
        assert!(matches!(err, GitError::RateLimit(_)));
    }

    #[test]
    fn map_status_500_is_api_error() {
        let status = reqwest::StatusCode::from_u16(500).unwrap();
        let err = GitHubApiClient::map_status(status, "boom");
        assert!(matches!(err, GitError::ApiError(_)));
    }

    #[test]
    fn resolve_token_missing() {
        let client = GitHubApiClient::new("owner", "repo", "SKG_TEST_GH_TOKEN_MISSING_8f3a2c");
        let err = client.resolve_token().unwrap_err();
        match err {
            GitError::ApiError(msg) => {
                assert!(msg.contains("SKG_TEST_GH_TOKEN_MISSING_8f3a2c"));
            }
            other => panic!("expected ApiError, got {:?}", other),
        }
    }

    #[test]
    fn api_url_construction() {
        let client = test_client();
        assert_eq!(
            client.api_url("pulls"),
            "https://api.github.com/repos/owner/repo/pulls"
        );
    }

    #[test]
    fn gh_pull_response_deserialization() {
        let json = r#"{
            "number": 99,
            "html_url": "https://github.com/owner/repo/pull/99",
            "body": "Some PR body",
            "labels": [{"name": "sweep"}, {"name": "automated"}],
            "created_at": "2026-03-04T12:00:00Z"
        }"#;
        let pr: GhPullResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(pr.number, 99);
        assert_eq!(pr.html_url, "https://github.com/owner/repo/pull/99");
        assert_eq!(pr.body.as_deref(), Some("Some PR body"));
        assert_eq!(pr.labels.len(), 2);
        assert_eq!(pr.labels[0].name, "sweep");
        assert_eq!(pr.created_at, "2026-03-04T12:00:00Z");
    }

    #[test]
    fn gh_pull_response_into_pull_request() {
        let gh = GhPullResponse {
            number: 42,
            html_url: "https://github.com/owner/repo/pull/42".to_string(),
            body: None,
            labels: vec![GhLabel {
                name: "sweep".to_string(),
            }],
            created_at: "2026-03-04T00:00:00Z".to_string(),
        };
        let pr = gh.into_pull_request();
        assert_eq!(pr.number, 42);
        assert_eq!(pr.url, "https://github.com/owner/repo/pull/42");
        assert_eq!(pr.body, ""); // None → empty string
        assert_eq!(pr.labels, vec!["sweep"]);
        assert_eq!(pr.created_at, "2026-03-04T00:00:00Z");
    }

    #[test]
    fn constructor_sets_default_api_base() {
        let client = GitHubApiClient::new("owner", "repo", "TOKEN_VAR");
        assert_eq!(client.api_base, "https://api.github.com");
    }

    #[test]
    fn constructor_custom_api_base() {
        let client = GitHubApiClient::with_api_base(
            "owner",
            "repo",
            "TOKEN_VAR",
            "https://github.example.com/api/v3",
        );
        assert_eq!(client.api_base, "https://github.example.com/api/v3");
    }
}
