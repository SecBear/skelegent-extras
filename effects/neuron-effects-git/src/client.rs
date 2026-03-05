//! GitHub API abstraction trait and mock implementation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::types::GitError;

/// A pull request as returned by the GitHub API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    /// PR number.
    pub number: u64,
    /// HTML URL of the PR.
    pub url: String,
    /// Current body (markdown) of the PR.
    pub body: String,
    /// Labels attached to the PR.
    pub labels: Vec<String>,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
}

/// Abstraction over GitHub API operations needed by the PR generator.
///
/// Implementations must be `Send + Sync` to be used across async task boundaries.
/// The real implementation performs GitHub REST API calls; the mock is used in tests.
#[async_trait]
pub trait GitHubClient: Send + Sync {
    /// Find an open pull request associated with the given decision ID.
    ///
    /// Returns `None` if no open PR exists for this decision.
    async fn find_open_pr(&self, decision_id: &str) -> Result<Option<PullRequest>, GitError>;

    /// Create a new pull request.
    ///
    /// Returns the HTML URL of the newly created PR.
    async fn create_pr(
        &self,
        branch: &str,
        title: &str,
        body: &str,
        draft: bool,
        labels: &[String],
    ) -> Result<String, GitError>;

    /// Update the body of an existing pull request.
    async fn update_pr(&self, number: u64, body: &str) -> Result<(), GitError>;

    /// Close a pull request and post a comment.
    async fn close_pr(&self, number: u64, comment: &str) -> Result<(), GitError>;

    /// Create a GitHub issue.
    ///
    /// Returns the HTML URL of the created issue.
    async fn create_issue(
        &self,
        title: &str,
        body: &str,
        labels: &[String],
    ) -> Result<String, GitError>;
}

/// Mock GitHub client for use in tests.
///
/// Configurable via its builder-style fields. Panics on unexpected calls
/// if the corresponding response has not been set.
#[cfg(test)]
pub struct MockGitHubClient {
    /// Pre-canned response for `find_open_pr`.
    pub open_pr: Option<PullRequest>,
    /// Error to return from `find_open_pr`, if any.
    pub find_error: Option<GitError>,
    /// URL returned from `create_pr`.
    pub created_pr_url: String,
    /// URL returned from `create_issue`.
    pub created_issue_url: String,
}

#[cfg(test)]
impl Default for MockGitHubClient {
    fn default() -> Self {
        Self {
            open_pr: None,
            find_error: None,
            created_pr_url: "https://github.com/owner/repo/pull/1".to_string(),
            created_issue_url: "https://github.com/owner/repo/issues/1".to_string(),
        }
    }
}

#[cfg(test)]
#[async_trait]
impl GitHubClient for MockGitHubClient {
    async fn find_open_pr(&self, _decision_id: &str) -> Result<Option<PullRequest>, GitError> {
        if let Some(ref e) = self.find_error {
            return Err(e.clone());
        }
        Ok(self.open_pr.clone())
    }

    async fn create_pr(
        &self,
        _branch: &str,
        _title: &str,
        _body: &str,
        _draft: bool,
        _labels: &[String],
    ) -> Result<String, GitError> {
        Ok(self.created_pr_url.clone())
    }

    async fn update_pr(&self, _number: u64, _body: &str) -> Result<(), GitError> {
        Ok(())
    }

    async fn close_pr(&self, _number: u64, _comment: &str) -> Result<(), GitError> {
        Ok(())
    }

    async fn create_issue(
        &self,
        _title: &str,
        _body: &str,
        _labels: &[String],
    ) -> Result<String, GitError> {
        Ok(self.created_issue_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_find_open_pr_returns_none_by_default() {
        let client = MockGitHubClient::default();
        let result = client.find_open_pr("D3B").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn mock_find_open_pr_returns_configured_pr() {
        let pr = PullRequest {
            number: 42,
            url: "https://github.com/owner/repo/pull/42".to_string(),
            body: "## Sweep Verdict: confirmed".to_string(),
            labels: vec!["sweep".to_string()],
            created_at: "2026-03-04T12:00:00Z".to_string(),
        };
        let client = MockGitHubClient {
            open_pr: Some(pr.clone()),
            ..Default::default()
        };
        let result = client.find_open_pr("D3B").await.unwrap();
        assert!(result.is_some());
        let got = result.unwrap();
        assert_eq!(got.number, 42);
        assert_eq!(got.url, pr.url);
    }

    #[tokio::test]
    async fn mock_find_open_pr_propagates_error() {
        let client = MockGitHubClient {
            find_error: Some(GitError::ApiError("500".to_string())),
            ..Default::default()
        };
        let result = client.find_open_pr("D3B").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn mock_create_pr_returns_url() {
        let client = MockGitHubClient::default();
        let url = client
            .create_pr("branch", "title", "body", true, &[])
            .await
            .unwrap();
        assert_eq!(url, "https://github.com/owner/repo/pull/1");
    }

    #[tokio::test]
    async fn mock_create_issue_returns_url() {
        let client = MockGitHubClient::default();
        let url = client.create_issue("title", "body", &[]).await.unwrap();
        assert_eq!(url, "https://github.com/owner/repo/issues/1");
    }

    #[tokio::test]
    async fn mock_update_and_close_pr_succeed() {
        let client = MockGitHubClient::default();
        client.update_pr(1, "new body").await.unwrap();
        client.close_pr(1, "stale").await.unwrap();
    }
}
