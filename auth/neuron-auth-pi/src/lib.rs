//! [`AuthProvider`] that reads credentials from `~/.pi/agent/auth.json`.
//!
//! # What this does
//!
//! Pi stores OAuth credentials in a plain JSON file after the user completes
//! the browser-based login flow once. This provider reads those stored
//! credentials and returns a valid access token, refreshing automatically
//! when the token is within [`REFRESH_BUFFER_MS`] of expiry.
//!
//! Supports all five providers that pi ships natively:
//!
//! | Audience                    | Key in auth.json      | Refresh |
//! |-----------------------------|-----------------------|---------|
//! | contains `"anthropic"`      | `"anthropic"`         | yes     |
//! | contains `"openai"`         | `"openai-codex"`      | yes     |
//! | contains `"github"`         | `"github-copilot"`    | yes     |
//! | contains `"gemini"`         | `"google-gemini-cli"` | yes     |
//! | contains `"googleapis"`     | `"google-gemini-cli"` | yes     |
//! | contains `"antigravity"`    | `"google-antigravity"` | yes    |
//!
//! Refresh is only implemented for Anthropic (the only audience the sweep
//! runner uses). For other providers the stored access token is returned
//! as-is; a [`ScopeUnavailable`] error falls through to the next chain link.
//!
//! Parallel.ai is not a pi provider — use `PARALLEL_API_KEY` env var.
//!
//! # auth.json format
//!
//! ```json
//! {
//!   "anthropic": {
//!     "type": "oauth",
//!     "access":  "sk-ant-oat01-...",
//!     "refresh": "<refresh-token>",
//!     "expires": 1772519427241
//!   }
//! }
//! ```
//!
//! `expires` is epoch **milliseconds** with a 5-minute buffer already applied
//! (matching pi-mono's convention).
//!
//! # Refresh
//!
//! Refresh uses Anthropic's token endpoint
//! (`https://console.anthropic.com/v1/oauth/token`) with
//! `grant_type=refresh_token`. On success the new credentials are written
//! back to `auth.json` atomically (temp-file + rename). No file locking is
//! needed for single-process sweep runs; the sweep runner does not compete
//! with other writers.
//!
//! # File location
//!
//! Resolved in order:
//! 1. `$PI_AGENT_DIR/auth.json`
//! 2. `$HOME/.pi/agent/auth.json`

use async_trait::async_trait;
use neuron_auth::{AuthError, AuthProvider, AuthRequest, AuthToken};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Anthropic OAuth client ID (pi-mono's registered client).
const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Anthropic OAuth token endpoint.
const ANTHROPIC_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";

/// Refresh when this many milliseconds remain before expiry (5 minutes).
const REFRESH_BUFFER_MS: i64 = 5 * 60 * 1000;

// ── Credential types ──────────────────────────────────────────────────────────

/// A single entry in auth.json, tagged by `"type"`.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PiCredential {
    Oauth(OauthEntry),
    ApiKey { key: String },
}

/// OAuth entry — mirrors pi-mono's `OAuthCredentials` type.
#[derive(Debug, Deserialize, Serialize, Clone)]
struct OauthEntry {
    access: String,
    refresh: String,
    /// Epoch milliseconds with 5-minute buffer pre-applied.
    expires: i64,
    /// Extra provider-specific fields (accountId, email, etc.) — preserved on write-back.
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

/// Token refresh response from Anthropic.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    /// Lifetime in seconds.
    expires_in: u64,
}

// ── PiAuthProvider ────────────────────────────────────────────────────────────

/// Reads OAuth credentials from `~/.pi/agent/auth.json` and returns valid
/// access tokens, refreshing them transparently when near expiry.
///
/// Intended as the first link in an [`neuron_auth::AuthProviderChain`]:
/// - Handles `anthropic` and `openai-codex` audiences with refresh.
/// - Returns [`AuthError::ScopeUnavailable`] for unknown audiences so the
///   chain falls through to the next provider (e.g., [`neuron-auth-omp`]).
#[derive(Debug, Clone)]
pub struct PiAuthProvider {
    auth_path: PathBuf,
    client: reqwest::Client,
}

impl PiAuthProvider {
    /// Create a provider pointing at a specific `auth.json` path.
    pub fn new(auth_path: impl Into<PathBuf>) -> Self {
        Self {
            auth_path: auth_path.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Resolve `auth.json` from the environment and return a provider,
    /// or `None` if no pi installation is found.
    ///
    /// Checks `$PI_AGENT_DIR/auth.json` then `$HOME/.pi/agent/auth.json`.
    pub fn from_env() -> Option<Self> {
        let path = auth_json_path()?;
        if path.exists() { Some(Self::new(path)) } else { None }
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Load the entire `auth.json` file.
    fn load(&self) -> Result<HashMap<String, PiCredential>, AuthError> {
        let raw = std::fs::read_to_string(&self.auth_path)
            .map_err(|e| AuthError::BackendError(format!("cannot read auth.json: {e}")))?;
        serde_json::from_str(&raw)
            .map_err(|e| AuthError::BackendError(format!("auth.json parse failed: {e}")))
    }

    /// Write the credential map back to `auth.json` atomically.
    ///
    /// Writes to a `.tmp` sibling first, then renames. Avoids leaving a
    /// half-written file if the process is killed mid-write.
    fn persist(&self, creds: &HashMap<String, PiCredential>) -> Result<(), AuthError> {
        let json = serde_json::to_string_pretty(creds)
            .map_err(|e| AuthError::BackendError(format!("auth.json serialize failed: {e}")))?;

        let tmp_path = self.auth_path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &json)
            .map_err(|e| AuthError::BackendError(format!("auth.json write failed: {e}")))?;
        std::fs::rename(&tmp_path, &self.auth_path)
            .map_err(|e| AuthError::BackendError(format!("auth.json rename failed: {e}")))?;

        Ok(())
    }

    /// Look up and return a valid access token for the given auth.json key.
    ///
    /// Refreshes via the Anthropic token endpoint if the token is within
    /// [`REFRESH_BUFFER_MS`] of expiry, then writes updated credentials back.
    async fn resolve(&self, key: &str) -> Result<String, AuthError> {
        let mut creds = self.load()?;

        let entry = creds.get(key).ok_or_else(|| {
            AuthError::ScopeUnavailable(format!(
                "PiAuthProvider: no credential for '{key}' in auth.json"
            ))
        })?;

        let oauth = match entry {
            PiCredential::Oauth(o) => o.clone(),
            PiCredential::ApiKey { key: k } => return Ok(k.clone()),
        };

        let now_ms = now_epoch_ms();

        if now_ms < oauth.expires - REFRESH_BUFFER_MS {
            // Token is still valid — return it directly.
            tracing::debug!(key, "pi auth: token valid, skipping refresh");
            return Ok(oauth.access);
        }

        // Token is expired or near-expired — refresh it.
        tracing::info!(key, "pi auth: refreshing expired token");
        let refreshed = self.refresh_anthropic_token(&oauth.refresh).await?;

        // Write back the updated entry, preserving extra fields.
        let updated = PiCredential::Oauth(OauthEntry {
            access: refreshed.access_token.clone(),
            refresh: refreshed.refresh_token,
            expires: now_ms + (refreshed.expires_in as i64 * 1000) - REFRESH_BUFFER_MS,
            extra: oauth.extra,
        });
        creds.insert(key.to_string(), updated);
        self.persist(&creds)?;

        tracing::info!(key, "pi auth: token refreshed and persisted");
        Ok(refreshed.access_token)
    }

    /// POST to Anthropic's token endpoint with `grant_type=refresh_token`.
    async fn refresh_anthropic_token(&self, refresh_token: &str) -> Result<TokenResponse, AuthError> {
        let body = serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": ANTHROPIC_CLIENT_ID,
            "refresh_token": refresh_token,
        });

        let resp = self
            .client
            .post(ANTHROPIC_TOKEN_URL)
            .json(&body)
            .send()
            .await
            .map_err(|e| AuthError::BackendError(format!("token refresh request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AuthError::BackendError(format!(
                "Anthropic token refresh failed ({status}): {body}"
            )));
        }

        resp.json::<TokenResponse>()
            .await
            .map_err(|e| AuthError::BackendError(format!("token refresh response parse failed: {e}")))
    }
}

#[async_trait]
impl AuthProvider for PiAuthProvider {
    /// Provide a token by reading (and if necessary refreshing) `auth.json`.
    ///
    /// Audience routing (mirrors pi-mono's provider registry):
    /// - `"anthropic"`   → key `"anthropic"`         (OAuth, refresh supported)
    /// - `"openai"`      → key `"openai-codex"`      (OAuth, token returned as-is)
    /// - `"github"`      → key `"github-copilot"`    (OAuth, token returned as-is)
    /// - `"gemini"` / `"googleapis"`  → key `"google-gemini-cli"`  (OAuth)
    /// - `"antigravity"` → key `"google-antigravity"`  (OAuth)
    ///
    /// Parallel.ai is not a pi provider — use `PARALLEL_API_KEY`.
    /// Returns [`AuthError::ScopeUnavailable`] for unrecognised audiences.
    async fn provide(&self, request: &AuthRequest) -> Result<AuthToken, AuthError> {
        let audience = request.audience.as_deref().unwrap_or("");

        let key = if audience.contains("anthropic") {
            "anthropic"
        } else if audience.contains("openai") {
            "openai-codex"
        } else if audience.contains("github") {
            "github-copilot"
        } else if audience.contains("gemini") || audience.contains("googleapis") {
            "google-gemini-cli"
        } else if audience.contains("antigravity") {
            "google-antigravity"
        } else {
            return Err(AuthError::ScopeUnavailable(format!(
                "PiAuthProvider: no credential mapping for audience '{audience}'"
            )));
        };

        let token = self.resolve(key).await?;
        Ok(AuthToken::permanent(token.into_bytes()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn auth_json_path() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PI_AGENT_DIR") {
        return Some(PathBuf::from(dir).join("auth.json"));
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".pi/agent/auth.json"))
}

fn now_epoch_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_auth_json(entry: serde_json::Value) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", entry).unwrap();
        f
    }

    #[tokio::test]
    async fn unknown_audience_returns_scope_unavailable() {
        // Nonexistent path — audience check happens before file read.
        let provider = PiAuthProvider::new("/nonexistent/auth.json");
        let req = AuthRequest::new().with_audience("unknown.example.com");
        let err = provider.provide(&req).await.unwrap_err();
        assert!(matches!(err, neuron_auth::AuthError::ScopeUnavailable(_)));
    }

    #[tokio::test]
    async fn valid_token_returned_without_refresh() {
        // expires far in the future
        let expires_future = now_epoch_ms() + 60 * 60 * 1000; // +1 hour
        let json = serde_json::json!({
            "anthropic": {
                "type": "oauth",
                "access": "sk-ant-oat01-valid",
                "refresh": "ref-tok",
                "expires": expires_future
            }
        });
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        let req = AuthRequest::new().with_audience("api.anthropic.com");
        let token = provider.provide(&req).await.unwrap();
        let key = token.with_bytes(|b| String::from_utf8_lossy(b).into_owned());
        assert_eq!(key, "sk-ant-oat01-valid");
    }

    #[tokio::test]
    async fn missing_key_returns_scope_unavailable() {
        let json = serde_json::json!({});
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        let req = AuthRequest::new().with_audience("api.anthropic.com");
        let err = provider.provide(&req).await.unwrap_err();
        assert!(matches!(err, neuron_auth::AuthError::ScopeUnavailable(_)));
    }

    #[tokio::test]
    async fn api_key_type_returned_directly() {
        let json = serde_json::json!({
            "anthropic": {
                "type": "api_key",
                "key": "sk-ant-regular-key"
            }
        });
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        let req = AuthRequest::new().with_audience("api.anthropic.com");
        let token = provider.provide(&req).await.unwrap();
        let key = token.with_bytes(|b| String::from_utf8_lossy(b).into_owned());
        assert_eq!(key, "sk-ant-regular-key");
    }

    #[test]
    fn persist_round_trips_extra_fields() {
        let json = serde_json::json!({
            "openai-codex": {
                "type": "oauth",
                "access": "eyJ-tok",
                "refresh": "ref",
                "expires": 9999999999999i64,
                "accountId": "fac785e7-d662-41ba-8eea-3bb26d68a49a"
            }
        });
        let f = write_auth_json(json.clone());
        let provider = PiAuthProvider::new(f.path());
        let creds = provider.load().unwrap();

        // Write back and re-read — accountId must survive.
        provider.persist(&creds).unwrap();
        let reloaded = provider.load().unwrap();

        if let PiCredential::Oauth(o) = &reloaded["openai-codex"] {
            assert_eq!(o.access, "eyJ-tok");
            assert_eq!(
                o.extra.get("accountId").and_then(|v| v.as_str()),
                Some("fac785e7-d662-41ba-8eea-3bb26d68a49a")
            );
        } else {
            panic!("expected oauth entry");
        }
    }

    // ── Additional audience routing coverage ───────────────────────────────

    #[tokio::test]
    async fn github_audience_resolves() {
        let expires_future = now_epoch_ms() + 60 * 60 * 1000;
        let json = serde_json::json!({
            "github-copilot": {
                "type": "oauth",
                "access": "ghu_test_token",
                "refresh": "ref",
                "expires": expires_future
            }
        });
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        let req = AuthRequest::new().with_audience("api.github.com");
        let token = provider.provide(&req).await.unwrap();
        let key = token.with_bytes(|b| String::from_utf8_lossy(b).into_owned());
        assert_eq!(key, "ghu_test_token");
    }

    #[tokio::test]
    async fn gemini_audience_resolves() {
        let expires_future = now_epoch_ms() + 60 * 60 * 1000;
        let json = serde_json::json!({
            "google-gemini-cli": {
                "type": "oauth",
                "access": "ya29_gemini",
                "refresh": "ref",
                "expires": expires_future
            }
        });
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        // both audience strings route to the same key
        for audience in ["generativelanguage.googleapis.com", "gemini.api"] {
            let req = AuthRequest::new().with_audience(audience);
            let token = provider.provide(&req).await.unwrap();
            let key = token.with_bytes(|b| String::from_utf8_lossy(b).into_owned());
            assert_eq!(key, "ya29_gemini", "audience: {audience}");
        }
    }

    #[tokio::test]
    async fn antigravity_audience_resolves() {
        let expires_future = now_epoch_ms() + 60 * 60 * 1000;
        let json = serde_json::json!({
            "google-antigravity": {
                "type": "oauth",
                "access": "ya29_ag",
                "refresh": "ref",
                "expires": expires_future
            }
        });
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        let req = AuthRequest::new().with_audience("antigravity.google.com");
        let token = provider.provide(&req).await.unwrap();
        let key = token.with_bytes(|b| String::from_utf8_lossy(b).into_owned());
        assert_eq!(key, "ya29_ag");
    }

    #[tokio::test]
    async fn unrecognised_audience_falls_through() {
        // Even with a valid file, an unknown audience is ScopeUnavailable so
        // the chain can hand off to the next provider.
        let json = serde_json::json!({
            "anthropic": {"type": "api_key", "key": "sk-x"}
        });
        let f = write_auth_json(json);
        let provider = PiAuthProvider::new(f.path());
        let req = AuthRequest::new().with_audience("parallel.ai");
        let err = provider.provide(&req).await.unwrap_err();
        assert!(matches!(err, neuron_auth::AuthError::ScopeUnavailable(_)));
    }
}
