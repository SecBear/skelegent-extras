//! Query rotation for per-decision sweep runs.
//!
//! Each decision has research angles. The sweep runner rotates through them
//! using the sweep counter so successive runs cover different facets before
//! repeating.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Types ────────────────────────────────────────────────────────────────────

/// A single research query paired with its angle label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionQuery {
    /// The verbatim search query sent to the research provider.
    pub query: String,
    /// Short label identifying the angle this query covers (e.g. `"scalability"`).
    pub angle: String,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Builds a [`DecisionQuery`] by appending the standard recency suffix to `keywords`.
pub fn build_query(angle: &str, keywords: &str) -> DecisionQuery {
    DecisionQuery {
        angle: angle.to_owned(),
        query: format!("{keywords} agent architecture production systems 2025 2026"),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns the full list of query angles registered for `decision_id`, or
/// `None` if the ID is unknown.
pub fn queries_for<'a>(
    registry: &'a HashMap<String, Vec<DecisionQuery>>,
    decision_id: &str,
) -> Option<&'a [DecisionQuery]> {
    registry.get(decision_id).map(|v| v.as_slice())
}

/// Selects the next query for `decision_id` using round-robin rotation over
/// `sweep_count`.
///
/// Returns `None` if `decision_id` is not registered.
pub fn next_query<'a>(
    registry: &'a HashMap<String, Vec<DecisionQuery>>,
    decision_id: &str,
    sweep_count: usize,
) -> Option<&'a DecisionQuery> {
    let qs = queries_for(registry, decision_id)?;
    Some(&qs[sweep_count % qs.len()])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> HashMap<String, Vec<DecisionQuery>> {
        let mut m = HashMap::new();
        m.insert("test-decision".to_string(), vec![
            build_query("event-model",     "event-driven trigger model LLM tool-calls agents"),
            build_query("scheduling",      "async task scheduling priority queues agent backpressure"),
            build_query("signal-handling", "signal interrupt cancellation agent lifecycle handling"),
        ]);
        m
    }

    #[test]
    fn queries_for_known_id() {
        let reg = make_registry();
        let qs = queries_for(&reg, "test-decision");
        assert!(qs.is_some());
        assert!(qs.unwrap().len() >= 3);
    }

    #[test]
    fn next_query_rotates() {
        let reg = make_registry();
        let q0 = next_query(&reg, "test-decision", 0).unwrap();
        let q1 = next_query(&reg, "test-decision", 1).unwrap();
        let q2 = next_query(&reg, "test-decision", 2).unwrap();
        let q3 = next_query(&reg, "test-decision", 3).unwrap();
        assert_ne!(q0.angle, q1.angle);
        assert_eq!(q0.angle, q3.angle);
        // keep q2 used to satisfy the compiler while matching the spec's test body
        let _ = q2;
    }

    #[test]
    fn unknown_decision_returns_none() {
        let reg = make_registry();
        assert!(next_query(&reg, "NONEXISTENT", 0).is_none());
    }
}
