//! Configuration types for the Temporal orchestrator.

use serde::{Deserialize, Serialize};

/// Configuration for connecting to a Temporal server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// gRPC address of the Temporal server (e.g. `"localhost:7233"`).
    ///
    /// `TemporalOrch::connect()` accepts either a bare host:port pair or a full
    /// URL such as `http://localhost:7233`.
    pub server_url: String,
    /// Temporal namespace to use (e.g. `"default"`).
    pub namespace: String,
    /// Task queue name used when dispatching work to Temporal workers.
    pub task_queue: String,
    /// Identity string sent to Temporal for observability and audit purposes.
    pub identity: String,
    /// Workflow type name for the deployed generic durable-run workflow.
    ///
    /// Relevant only when the crate is built with `temporal-sdk` and callers
    /// opt into the real backend via `TemporalOrch::connect()`. In that backend,
    /// the portable `RunId` is used as the Temporal workflow ID for this workflow
    /// type; Temporal's server-assigned run ID remains internal. The workflow
    /// must expose the expected run-view query, control signal, and resume
    /// update handlers.
    #[serde(default = "default_workflow_type")]
    pub workflow_type: String,
}

impl Default for TemporalConfig {
    /// Returns a config pointing at a local Temporal server with sensible defaults.
    fn default() -> Self {
        Self {
            server_url: "localhost:7233".to_string(),
            namespace: "default".to_string(),
            task_queue: "skg-worker".to_string(),
            identity: String::new(),
            workflow_type: "skg.generic.durable-run".to_string(),
        }
    }
}

fn default_workflow_type() -> String {
    "skg.generic.durable-run".to_string()
}

impl TemporalConfig {
    /// Return the server URL in a form accepted by HTTP/gRPC clients.
    #[cfg(any(test, feature = "temporal-sdk"))]
    pub(crate) fn normalized_server_url(&self) -> String {
        let server = self.server_url.trim();
        if server.contains("://") {
            server.to_string()
        } else {
            format!("http://{server}")
        }
    }
}

/// Retry policy applied to activity and workflow executions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Initial backoff delay in milliseconds before the first retry.
    pub initial_interval_ms: u64,
    /// Maximum backoff delay in milliseconds; caps exponential growth.
    pub max_interval_ms: u64,
    /// Maximum number of attempts (including the first). `0` means unlimited.
    pub max_attempts: u32,
    /// Multiplier applied to the interval after each failure.
    pub backoff_coefficient: f64,
}

impl Default for RetryPolicy {
    /// Returns a retry policy with 3 attempts and exponential backoff up to 60 s.
    fn default() -> Self {
        Self {
            initial_interval_ms: 1000,
            max_interval_ms: 60_000,
            max_attempts: 3,
            backoff_coefficient: 2.0,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_server_url_adds_http_scheme_for_bare_host_port() {
        let config = TemporalConfig::default();
        assert_eq!(config.normalized_server_url(), "http://localhost:7233");
    }

    #[test]
    fn normalized_server_url_preserves_explicit_scheme() {
        let config = TemporalConfig {
            server_url: "https://temporal.internal:7233".to_string(),
            ..TemporalConfig::default()
        };
        assert_eq!(
            config.normalized_server_url(),
            "https://temporal.internal:7233"
        );
    }
}