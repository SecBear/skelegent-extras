//! Configuration types for the Temporal orchestrator.

use serde::{Deserialize, Serialize};

/// Configuration for connecting to a Temporal server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// gRPC address of the Temporal server (e.g. `"localhost:7233"`).
    pub server_url: String,
    /// Temporal namespace to use (e.g. `"default"`).
    pub namespace: String,
    /// Task queue name used when dispatching work to Temporal workers.
    pub task_queue: String,
    /// Identity string sent to Temporal for observability and audit purposes.
    pub identity: String,
}

impl Default for TemporalConfig {
    /// Returns a config pointing at a local Temporal server with sensible defaults.
    fn default() -> Self {
        Self {
            server_url: "localhost:7233".to_string(),
            namespace: "default".to_string(),
            task_queue: "neuron-worker".to_string(),
            identity: String::new(),
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
