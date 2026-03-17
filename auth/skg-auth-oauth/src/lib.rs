#![deny(missing_docs)]
//! OAuth 2.0 Device Authorization Flow provider for [`skg_auth::AuthProvider`].
//!
//! This crate implements [RFC 8628](https://tools.ietf.org/html/rfc8628) — the
//! OAuth 2.0 Device Authorization Grant. It is designed for headless or CLI
//! environments where browser-based redirect flows are impractical.
//!
//! # Usage
//!
//! 1. Create an [`OAuthDeviceFlowProvider`] with vendor-specific [`OAuthConfig`]
//! 2. Call [`start_device_auth()`] to get a user code and verification URL
//! 3. Display the code/URL to the user
//! 4. Call [`poll_for_token()`] to wait for user authorization
//! 5. Use the provider as an [`AuthProvider`] — tokens are cached and refreshed
//!
//! No vendor-specific values (client IDs, URLs) are hardcoded. All configuration
//! is injected via [`OAuthConfig`].
//!
//! [`start_device_auth()`]: OAuthDeviceFlowProvider::start_device_auth
//! [`poll_for_token()`]: OAuthDeviceFlowProvider::poll_for_token
//! [`AuthProvider`]: skg_auth::AuthProvider

use async_trait::async_trait;
use skg_auth::{AuthError, AuthProvider, AuthRequest, AuthToken};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Configuration for the OAuth 2.0 Device Authorization Flow.
///
/// All fields are vendor-specific and must be provided by the caller.
/// No defaults are assumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    /// OAuth 2.0 client identifier.
    pub client_id: String,
    /// Device authorization endpoint URL.
    pub device_auth_url: String,
    /// Token endpoint URL.
    pub token_url: String,
    /// Scopes to request during device authorization.
    pub scopes: Vec<String>,
    /// Optional audience parameter (used by some providers like Auth0).
    pub audience: Option<String>,
}

/// Response from the device authorization endpoint (RFC 8628 Section 3.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAuthResponse {
    /// The device verification code.
    pub device_code: String,
    /// The end-user verification code.
    pub user_code: String,
    /// The end-user verification URI.
    pub verification_uri: String,
    /// Optional verification URI that includes the user code.
    pub verification_uri_complete: Option<String>,
    /// Lifetime in seconds of the device code.
    pub expires_in: u64,
    /// Minimum polling interval in seconds (default 5 per RFC 8628).
    pub interval: Option<u64>,
}

/// Cached token state held internally.
struct CachedToken {
    /// Raw access token bytes.
    access_token: Vec<u8>,
    /// Optional refresh token for token renewal.
    refresh_token: Option<String>,
    /// When the access token expires, if known.
    expires_at: Option<SystemTime>,
}

/// OAuth 2.0 Device Authorization Flow provider.
///
/// Implements [`AuthProvider`] by caching and refreshing tokens obtained
/// through the device authorization grant. The provider is not interactive —
/// callers must run the device flow via [`start_device_auth`] and
/// [`poll_for_token`] before using `provide()`.
///
/// [`start_device_auth`]: Self::start_device_auth
/// [`poll_for_token`]: Self::poll_for_token
/// [`AuthProvider`]: skg_auth::AuthProvider
#[derive(Clone)]
pub struct OAuthDeviceFlowProvider {
    config: OAuthConfig,
    http: reqwest::Client,
    cached: Arc<RwLock<Option<CachedToken>>>,
}

impl std::fmt::Debug for OAuthDeviceFlowProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthDeviceFlowProvider")
            .field("config", &self.config)
            .field("cached", &"[REDACTED]")
            .finish()
    }
}

/// Token endpoint JSON response for successful grants.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    #[allow(dead_code)]
    token_type: Option<String>,
}

/// Token endpoint JSON error response (RFC 8628 Section 3.5).
#[derive(Deserialize)]
struct TokenErrorResponse {
    error: String,
}

/// Buffer before expiry at which we consider a token "near expiry".
const NEAR_EXPIRY_BUFFER: Duration = Duration::from_secs(5 * 60);

impl OAuthDeviceFlowProvider {
    /// Create a new provider with the given configuration and a default HTTP client.
    pub fn new(config: OAuthConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
            cached: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a new provider with the given configuration and a custom HTTP client.
    pub fn with_http_client(config: OAuthConfig, client: reqwest::Client) -> Self {
        Self {
            config,
            http: client,
            cached: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the device authorization flow (RFC 8628 Section 3.1).
    ///
    /// POSTs to the device authorization endpoint and returns the
    /// [`DeviceAuthResponse`] containing the user code and verification URL.
    pub async fn start_device_auth(&self) -> Result<DeviceAuthResponse, AuthError> {
        let scope = self.config.scopes.join(" ");

        let mut params = vec![
            ("client_id", self.config.client_id.as_str()),
            ("scope", &scope),
        ];

        // Some providers (e.g. Auth0) require audience.
        let audience_val;
        if let Some(ref aud) = self.config.audience {
            audience_val = aud.clone();
            params.push(("audience", &audience_val));
        }

        let resp = self
            .http
            .post(&self.config.device_auth_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| AuthError::BackendError(format!("device auth request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(AuthError::BackendError(format!(
                "device auth endpoint returned status {}",
                resp.status()
            )));
        }

        let body = resp
            .json::<DeviceAuthResponse>()
            .await
            .map_err(|e| AuthError::BackendError(format!("failed to parse device auth response: {e}")))?;

        debug!(
            user_code = %body.user_code,
            verification_uri = %body.verification_uri,
            "device authorization started"
        );

        Ok(body)
    }

    /// Poll the token endpoint until the user authorizes the device (RFC 8628 Section 3.4).
    ///
    /// This blocks (asynchronously) until one of:
    /// - The user authorizes and a token is obtained (stored in cache)
    /// - The device code expires (`expires_in` timeout)
    /// - An unrecoverable error occurs
    ///
    /// `interval` is the initial polling interval from the device auth response.
    /// `expires_in` is the total lifetime of the device code.
    pub async fn poll_for_token(
        &self,
        device_code: &str,
        interval: Duration,
        expires_in: Duration,
    ) -> Result<(), AuthError> {
        let deadline = tokio::time::Instant::now() + expires_in;
        let mut poll_interval = interval;

        loop {
            tokio::time::sleep(poll_interval).await;

            if tokio::time::Instant::now() >= deadline {
                return Err(AuthError::AuthFailed(
                    "device code expired before user authorization".into(),
                ));
            }

            let resp = self
                .http
                .post(&self.config.token_url)
                .form(&[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", device_code),
                    ("client_id", &self.config.client_id),
                ])
                .send()
                .await
                .map_err(|e| AuthError::BackendError(format!("token request failed: {e}")))?;

            let status = resp.status();
            let body = resp
                .bytes()
                .await
                .map_err(|e| AuthError::BackendError(format!("failed to read token response: {e}")))?;

            if status.is_success() {
                let token: TokenResponse = serde_json::from_slice(&body).map_err(|e| {
                    AuthError::BackendError(format!("failed to parse token response: {e}"))
                })?;

                let expires_at = token
                    .expires_in
                    .map(|secs| SystemTime::now() + Duration::from_secs(secs));

                let cached = CachedToken {
                    access_token: token.access_token.into_bytes(),
                    refresh_token: token.refresh_token,
                    expires_at,
                };

                *self.cached.write().await = Some(cached);
                debug!("device flow token obtained successfully");
                return Ok(());
            }

            // Error response — check the error code per RFC 8628 Section 3.5.
            let err: TokenErrorResponse = serde_json::from_slice(&body).map_err(|e| {
                AuthError::BackendError(format!("failed to parse token error response: {e}"))
            })?;

            match err.error.as_str() {
                "authorization_pending" => {
                    debug!("authorization pending, continuing to poll");
                    continue;
                }
                "slow_down" => {
                    poll_interval += Duration::from_secs(5);
                    debug!(?poll_interval, "slow_down received, increased interval");
                    continue;
                }
                "expired_token" => {
                    return Err(AuthError::AuthFailed("device code expired".into()));
                }
                "access_denied" => {
                    return Err(AuthError::AuthFailed("user denied authorization".into()));
                }
                other => {
                    return Err(AuthError::BackendError(format!(
                        "token endpoint error: {other}"
                    )));
                }
            }
        }
    }

    /// Manually inject a token into the cache.
    ///
    /// Useful for tests and when a token has been obtained through
    /// an external mechanism.
    pub async fn set_token(
        &self,
        access_token: String,
        refresh_token: Option<String>,
        expires_in: Option<u64>,
    ) {
        let expires_at = expires_in.map(|secs| SystemTime::now() + Duration::from_secs(secs));

        *self.cached.write().await = Some(CachedToken {
            access_token: access_token.into_bytes(),
            refresh_token,
            expires_at,
        });
    }

    /// Attempt to refresh the access token using a stored refresh token.
    async fn try_refresh(&self, refresh_token: &str) -> Result<(), AuthError> {
        let resp = self
            .http
            .post(&self.config.token_url)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", &self.config.client_id),
            ])
            .send()
            .await
            .map_err(|e| AuthError::BackendError(format!("refresh request failed: {e}")))?;

        if !resp.status().is_success() {
            // Clear cache on refresh failure — the refresh token is likely revoked.
            *self.cached.write().await = None;
            return Err(AuthError::AuthFailed("token refresh failed".into()));
        }

        let token: TokenResponse = resp.json().await.map_err(|e| {
            AuthError::BackendError(format!("failed to parse refresh response: {e}"))
        })?;

        let expires_at = token
            .expires_in
            .map(|secs| SystemTime::now() + Duration::from_secs(secs));

        *self.cached.write().await = Some(CachedToken {
            access_token: token.access_token.into_bytes(),
            refresh_token: token.refresh_token,
            expires_at,
        });

        debug!("token refreshed successfully");
        Ok(())
    }

    /// Check whether a cached token is near expiry or already expired.
    fn is_near_expiry(expires_at: Option<SystemTime>) -> bool {
        match expires_at {
            Some(exp) => {
                let threshold = SystemTime::now() + NEAR_EXPIRY_BUFFER;
                exp <= threshold
            }
            None => false, // No expiry → never near expiry.
        }
    }
}

#[async_trait]
impl AuthProvider for OAuthDeviceFlowProvider {
    /// Provide a cached OAuth token.
    ///
    /// This method is **not** interactive. If no token has been obtained via
    /// the device flow, it returns [`AuthError::AuthFailed`].
    ///
    /// If the cached token is near expiry and a refresh token is available,
    /// a refresh is attempted automatically.
    async fn provide(&self, _request: &AuthRequest) -> Result<AuthToken, AuthError> {
        let guard = self.cached.read().await;
        let cached = guard
            .as_ref()
            .ok_or_else(|| AuthError::AuthFailed("no token available — run device flow first".into()))?;

        // If token is not near expiry, return it directly.
        if !Self::is_near_expiry(cached.expires_at) {
            return Ok(AuthToken::new(
                cached.access_token.clone(),
                cached.expires_at,
            ));
        }

        // Token is near expiry — try to refresh if we have a refresh token.
        let refresh_token = cached.refresh_token.clone();
        drop(guard); // Release read lock before write.

        if let Some(ref rt) = refresh_token {
            self.try_refresh(rt).await?;
            let guard = self.cached.read().await;
            let cached = guard.as_ref().ok_or_else(|| {
                AuthError::AuthFailed("token lost during refresh".into())
            })?;
            return Ok(AuthToken::new(
                cached.access_token.clone(),
                cached.expires_at,
            ));
        }

        // No refresh token — return the near-expiry token anyway (best effort).
        warn!("token is near expiry but no refresh token is available");
        let guard = self.cached.read().await;
        let cached = guard.as_ref().ok_or_else(|| {
            AuthError::AuthFailed("token lost".into())
        })?;
        Ok(AuthToken::new(
            cached.access_token.clone(),
            cached.expires_at,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(server_url: &str) -> OAuthConfig {
        OAuthConfig {
            client_id: "test-client".into(),
            device_auth_url: format!("{server_url}/device/code"),
            token_url: format!("{server_url}/oauth/token"),
            scopes: vec!["openid".into(), "profile".into()],
            audience: Some("https://api.example.com".into()),
        }
    }

    fn token_success_body() -> serde_json::Value {
        serde_json::json!({
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token"
        })
    }

    #[tokio::test]
    async fn test_device_auth_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_code": "dev-code-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://example.com/activate",
                "verification_uri_complete": "https://example.com/activate?user_code=ABCD-1234",
                "expires_in": 900,
                "interval": 5
            })))
            .mount(&server)
            .await;

        let config = test_config(&server.uri());
        let provider = OAuthDeviceFlowProvider::new(config);
        let resp = provider.start_device_auth().await.unwrap();

        assert_eq!(resp.device_code, "dev-code-123");
        assert_eq!(resp.user_code, "ABCD-1234");
        assert_eq!(resp.verification_uri, "https://example.com/activate");
        assert_eq!(
            resp.verification_uri_complete.as_deref(),
            Some("https://example.com/activate?user_code=ABCD-1234")
        );
        assert_eq!(resp.expires_in, 900);
        assert_eq!(resp.interval, Some(5));
    }

    #[tokio::test]
    async fn test_poll_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(token_success_body()))
            .mount(&server)
            .await;

        let config = test_config(&server.uri());
        let provider = OAuthDeviceFlowProvider::new(config);

        provider
            .poll_for_token("dev-code", Duration::from_millis(10), Duration::from_secs(30))
            .await
            .unwrap();

        // Verify token is cached.
        let guard = provider.cached.read().await;
        let cached = guard.as_ref().unwrap();
        assert_eq!(cached.access_token, b"test-access-token");
        assert_eq!(cached.refresh_token.as_deref(), Some("test-refresh-token"));
        assert!(cached.expires_at.is_some());
    }

    #[tokio::test]
    async fn test_poll_authorization_pending_then_success() {
        let server = MockServer::start().await;

        // First call: authorization_pending.
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(serde_json::json!({"error": "authorization_pending"})),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;

        // Second call: success.
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(token_success_body()))
            .mount(&server)
            .await;

        let config = test_config(&server.uri());
        let provider = OAuthDeviceFlowProvider::new(config);

        provider
            .poll_for_token("dev-code", Duration::from_millis(10), Duration::from_secs(30))
            .await
            .unwrap();

        let guard = provider.cached.read().await;
        assert!(guard.is_some());
    }

    #[tokio::test]
    async fn test_poll_slow_down() {
        let server = MockServer::start().await;

        // First call: slow_down.
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(serde_json::json!({"error": "slow_down"})),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;

        // Second call: success.
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(token_success_body()))
            .mount(&server)
            .await;

        let config = test_config(&server.uri());
        let provider = OAuthDeviceFlowProvider::new(config);

        // Use a very short initial interval so the test doesn't take long.
        // After slow_down, interval should increase by 5 seconds — but we
        // verify the logic works by the fact that the second poll succeeds.
        provider
            .poll_for_token("dev-code", Duration::from_millis(10), Duration::from_secs(30))
            .await
            .unwrap();

        let guard = provider.cached.read().await;
        assert!(guard.is_some());
    }

    #[tokio::test]
    async fn test_poll_expired_token() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(serde_json::json!({"error": "expired_token"})),
            )
            .mount(&server)
            .await;

        let config = test_config(&server.uri());
        let provider = OAuthDeviceFlowProvider::new(config);

        let err = provider
            .poll_for_token("dev-code", Duration::from_millis(10), Duration::from_secs(30))
            .await
            .unwrap_err();

        match err {
            AuthError::AuthFailed(msg) => assert!(msg.contains("expired"), "got: {msg}"),
            other => panic!("expected AuthFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_provide_cached_token() {
        let config = OAuthConfig {
            client_id: "c".into(),
            device_auth_url: "http://unused".into(),
            token_url: "http://unused".into(),
            scopes: vec![],
            audience: None,
        };
        let provider = OAuthDeviceFlowProvider::new(config);

        // Set a token that expires in 1 hour (well outside near-expiry buffer).
        provider
            .set_token("my-token".into(), None, Some(3600))
            .await;

        let request = AuthRequest::new();
        let token = provider.provide(&request).await.unwrap();

        token.with_bytes(|b| {
            assert_eq!(b, b"my-token");
        });
        assert!(!token.is_expired());
    }

    #[tokio::test]
    async fn test_provide_no_token_error() {
        let config = OAuthConfig {
            client_id: "c".into(),
            device_auth_url: "http://unused".into(),
            token_url: "http://unused".into(),
            scopes: vec![],
            audience: None,
        };
        let provider = OAuthDeviceFlowProvider::new(config);

        let err = provider.provide(&AuthRequest::new()).await.unwrap_err();
        match err {
            AuthError::AuthFailed(msg) => {
                assert!(msg.contains("no token available"), "got: {msg}");
            }
            other => panic!("expected AuthFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_provide_expired_tries_refresh() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "refreshed-token",
                "token_type": "Bearer",
                "expires_in": 3600
            })))
            .mount(&server)
            .await;

        let config = test_config(&server.uri());
        let provider = OAuthDeviceFlowProvider::new(config);

        // Set a token that is already expired (expires_at in the past).
        {
            let mut guard = provider.cached.write().await;
            *guard = Some(CachedToken {
                access_token: b"old-token".to_vec(),
                refresh_token: Some("my-refresh-token".into()),
                expires_at: Some(SystemTime::now() - Duration::from_secs(60)),
            });
        }

        let token = provider.provide(&AuthRequest::new()).await.unwrap();
        token.with_bytes(|b| {
            assert_eq!(b, b"refreshed-token");
        });
    }

    #[tokio::test]
    async fn test_config_serde_roundtrip() {
        let config = OAuthConfig {
            client_id: "my-client".into(),
            device_auth_url: "https://auth.example.com/device/code".into(),
            token_url: "https://auth.example.com/oauth/token".into(),
            scopes: vec!["openid".into(), "offline_access".into()],
            audience: Some("https://api.example.com".into()),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OAuthConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.client_id, config.client_id);
        assert_eq!(deserialized.device_auth_url, config.device_auth_url);
        assert_eq!(deserialized.token_url, config.token_url);
        assert_eq!(deserialized.scopes, config.scopes);
        assert_eq!(deserialized.audience, config.audience);
    }

    #[tokio::test]
    async fn test_clone() {
        let config = OAuthConfig {
            client_id: "c".into(),
            device_auth_url: "http://unused".into(),
            token_url: "http://unused".into(),
            scopes: vec!["openid".into()],
            audience: None,
        };
        let provider = OAuthDeviceFlowProvider::new(config);
        provider
            .set_token("my-token".into(), None, Some(3600))
            .await;

        let cloned = provider.clone();

        // Cloned provider should share the same token cache (Arc<RwLock>).
        let token = cloned.provide(&AuthRequest::new()).await.unwrap();
        token.with_bytes(|b| {
            assert_eq!(b, b"my-token");
        });
    }

    #[tokio::test]
    async fn test_object_safety() {
        let config = OAuthConfig {
            client_id: "c".into(),
            device_auth_url: "http://unused".into(),
            token_url: "http://unused".into(),
            scopes: vec![],
            audience: None,
        };
        let provider = OAuthDeviceFlowProvider::new(config);

        // Prove the provider can be used as a trait object.
        let _dyn_provider: Arc<dyn AuthProvider> = Arc::new(provider);
    }
}
