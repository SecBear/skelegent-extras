//! Query rotation for per-decision sweep runs.
//!
//! Each decision has a set of research angles. The sweep runner rotates
//! through them using the sweep counter so successive runs cover different
//! facets before repeating.
//!
//! The registry (a `HashMap<String, Vec<DecisionQuery>>`) is owned by the
//! caller — typically the binary — so this module provides only the mechanism
//! (types + rotation logic), not the data.

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
fn dq(angle: &str, keywords: &str) -> DecisionQuery {
    DecisionQuery {
        angle: angle.to_owned(),
        query: format!("{keywords} agent architecture production systems 2025 2026"),
    }
}

/// Builds a [`DecisionQuery`] by appending the standard recency suffix to `keywords`.
///
/// Public alias for the internal `dq` helper. Use this to populate a registry
/// with the same query-formatting convention used throughout the pipeline.
pub fn build_query(angle: &str, keywords: &str) -> DecisionQuery {
    dq(angle, keywords)
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

    fn test_registry() -> HashMap<String, Vec<DecisionQuery>> {
        let mut m = HashMap::new();
        m.insert(
            "topic-1".to_string(),
            vec![
                dq("event-model", "event-driven trigger model LLM tool-calls agents"),
                dq("scheduling", "async task scheduling priority queues agent backpressure"),
                dq("signal-handling", "signal interrupt cancellation agent lifecycle handling"),
            ],
        );
        m.insert(
            "topic-2".to_string(),
            vec![
                dq("storage", "agent memory storage vector DB retrieval backends"),
                dq("context-window", "context window LLM token limits management strategies"),
                dq("retrieval", "retrieval augmented generation RAG agent memory recall"),
            ],
        );
        m
    }

    #[test]
    fn queries_for_returns_registered_entries() {
        let reg = test_registry();
        let qs = queries_for(&reg, "topic-1");
        assert!(qs.is_some(), "missing queries for topic-1");
        assert!(qs.unwrap().len() >= 3, "topic-1 has fewer than 3 query angles");
    }

    #[test]
    fn next_query_rotates() {
        let reg = test_registry();
        let q0 = next_query(&reg, "topic-1", 0).unwrap();
        let q1 = next_query(&reg, "topic-1", 1).unwrap();
        let q2 = next_query(&reg, "topic-1", 2).unwrap();
        let q3 = next_query(&reg, "topic-1", 3).unwrap();
        assert_ne!(q0.angle, q1.angle);
        assert_eq!(q0.angle, q3.angle);
        // keep q2 used to satisfy the compiler while matching the spec's test body
        let _ = q2;
    }

    #[test]
    fn unknown_decision_returns_none() {
        let reg = test_registry();
        assert!(next_query(&reg, "NONEXISTENT", 0).is_none());
    }

    #[test]
    fn query_lengths_within_limit() {
        let reg = test_registry();
        for (id, qs) in &reg {
            for q in qs {
                assert!(
                    q.query.len() <= 120,
                    "query too long ({} chars) for {id}/{}: {:?}",
                    q.query.len(),
                    q.angle,
                    q.query,
                );
            }
        }
    }

    #[test]
    fn build_query_produces_same_format_as_internal_dq() {
        let via_build = build_query("scalability", "some keywords");
        let via_dq = dq("scalability", "some keywords");
        assert_eq!(via_build.query, via_dq.query);
        assert_eq!(via_build.angle, via_dq.angle);
    }
}
