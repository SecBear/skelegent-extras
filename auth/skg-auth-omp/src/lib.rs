//! [`AuthProvider`] that reads credentials from `~/.omp/agent/agent.db`.
//!
//! # What this does
//!
//! OMP stores API keys and OAuth tokens in a SQLite database (`agent.db`).
//! This provider reads those credentials **read-only** — it does not refresh
//! OAuth tokens. Use [`skg-auth-pi`] as the first chain link for providers
//! that support refresh (Anthropic, OpenAI). Use this as the fallback for
//! credentials that only exist in OMP (e.g., Parallel.ai API keys).
//!
//! # Credential mapping
//!
//! | Audience                   | DB row                                    |
//! |----------------------------|-------------------------------------------|
//! | contains `"parallel"`      | `provider="parallel" type="apiKey"`       |
//! | contains `"anthropic"`     | `provider="anthropic" type="oauth"`       |
//! | contains `"openai"`        | `provider="openai-codex" type="oauth"`    |
//!
//! # Token format
//!
//! The `data` column contains JSON. Two shapes are handled:
//! - `{"access": "..."}` — OAuth access token
//! - `{"apiKey": "..."}` — static API key
//!
//! # DB location
//!
//! Resolved in order:
//! 1. `$PI_CODING_AGENT_DIR/agent.db`
//! 2. `$HOME/.omp/agent/agent.db`

use async_trait::async_trait;
use skg_auth::{AuthError, AuthProvider, AuthRequest, AuthToken};
use rusqlite::OptionalExtension;
use std::path::PathBuf;

/// Reads credentials from OMP's `agent.db` credential store (read-only).
///
/// Intended as a fallback in an [`skg_auth::AuthProviderChain`] after
/// [`skg-auth-pi`]. Handles credentials that are only available in OMP
/// (e.g., Parallel.ai API keys) and serves as a fallback for users who have
/// OMP but not pi installed.
///
/// Returns [`AuthError::ScopeUnavailable`] when no matching credential exists,
/// allowing the chain to continue to the next provider.
#[derive(Debug, Clone)]
pub struct OmpAuthProvider {
    db_path: PathBuf,
}

impl OmpAuthProvider {
    /// Create a provider pointing at a specific `agent.db` path.
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self { db_path: db_path.into() }
    }

    /// Resolve the agent.db path from the environment and return a provider,
    /// or `None` if no OMP installation is found.
    ///
    /// Checks `$PI_CODING_AGENT_DIR/agent.db` then `$HOME/.omp/agent/agent.db`.
    pub fn from_env() -> Option<Self> {
        let path = agent_db_path()?;
        if path.exists() { Some(Self::new(path)) } else { None }
    }

    /// Query `agent.db` for the most-recently-updated active credential.
    ///
    /// Returns the value of the `"access"` or `"apiKey"` field from the
    /// stored JSON.
    fn lookup(&self, provider: &str, credential_type: &str) -> Result<String, AuthError> {
        let conn = rusqlite::Connection::open_with_flags(
            &self.db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| AuthError::BackendError(format!("cannot open agent.db: {e}")))?;

        let data: Option<String> = conn
            .query_row(
                "SELECT data FROM auth_credentials \
                 WHERE provider = ?1 AND credential_type = ?2 AND disabled = 0 \
                 ORDER BY updated_at DESC, id DESC \
                 LIMIT 1",
                rusqlite::params![provider, credential_type],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| AuthError::BackendError(format!("agent.db query failed: {e}")))?;

        let data = data.ok_or_else(|| {
            AuthError::ScopeUnavailable(format!(
                "OmpAuthProvider: no active {credential_type} credential for '{provider}' in agent.db"
            ))
        })?;

        extract_token(&data).ok_or_else(|| {
            AuthError::BackendError(format!(
                "OmpAuthProvider: credential JSON for '{provider}' has no 'access' or 'apiKey' field"
            ))
        })
    }
}

#[async_trait]
impl AuthProvider for OmpAuthProvider {
    /// Provide a token by reading from `agent.db` (read-only, no refresh).
    ///
    /// The `audience` field of the request determines which credential row
    /// is looked up. Returns [`AuthError::ScopeUnavailable`] when no matching
    /// row exists.
    async fn provide(&self, request: &AuthRequest) -> Result<AuthToken, AuthError> {
        let audience = request.audience.as_deref().unwrap_or("");

        let token = if audience.contains("parallel") {
            self.lookup("parallel", "apiKey")?
        } else if audience.contains("anthropic") {
            self.lookup("anthropic", "oauth")?
        } else if audience.contains("openai") {
            self.lookup("openai-codex", "oauth")?
        } else {
            return Err(AuthError::ScopeUnavailable(format!(
                "OmpAuthProvider: no credential mapping for audience '{audience}'"
            )));
        };

        Ok(AuthToken::permanent(token.into_bytes()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn agent_db_path() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PI_CODING_AGENT_DIR") {
        return Some(PathBuf::from(dir).join("agent.db"));
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".omp/agent/agent.db"))
}

/// Extract the token string from OMP credential JSON.
///
/// Handles two shapes:
/// - `{"access": "..."}` — OAuth tokens
/// - `{"apiKey": "..."}` — API keys (Parallel.ai)
fn extract_token(json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    ["access", "apiKey"]
        .iter()
        .find_map(|field| v.get(*field)?.as_str().filter(|s| !s.is_empty()))
        .map(str::to_owned)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_token_access_field() {
        assert_eq!(
            extract_token(r#"{"access": "sk-ant-oat01-abc"}"#).as_deref(),
            Some("sk-ant-oat01-abc")
        );
    }

    #[test]
    fn extract_token_api_key_field() {
        assert_eq!(
            extract_token(r#"{"apiKey": "par-key-xyz"}"#).as_deref(),
            Some("par-key-xyz")
        );
    }

    #[test]
    fn extract_token_prefers_access_over_api_key() {
        assert_eq!(
            extract_token(r#"{"access": "oauth-tok", "apiKey": "api-key"}"#).as_deref(),
            Some("oauth-tok")
        );
    }

    #[test]
    fn extract_token_empty_is_none() {
        assert!(extract_token(r#"{"access": ""}"#).is_none());
    }

    #[test]
    fn extract_token_missing_fields_is_none() {
        assert!(extract_token(r#"{"other": "value"}"#).is_none());
    }

    #[test]
    fn extract_token_invalid_json_is_none() {
        assert!(extract_token("not json").is_none());
    }

    #[tokio::test]
    async fn unknown_audience_returns_scope_unavailable() {
        let provider = OmpAuthProvider::new("/nonexistent/agent.db");
        let req = AuthRequest::new().with_audience("unknown.example.com");
        let err = provider.provide(&req).await.unwrap_err();
        assert!(matches!(err, skg_auth::AuthError::ScopeUnavailable(_)));
    }
}
