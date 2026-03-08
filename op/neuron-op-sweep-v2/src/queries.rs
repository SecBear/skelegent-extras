//! Query rotation for per-decision sweep runs.
//!
//! Each architectural decision has three research angles. The sweep runner
//! rotates through them using the sweep counter so successive runs cover
//! different facets before repeating.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

// ── Types ────────────────────────────────────────────────────────────────────

/// A single research query paired with its angle label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionQuery {
    /// The verbatim search query sent to the research provider.
    pub query: String,
    /// Short label identifying the angle this query covers (e.g. `"scalability"`).
    pub angle: String,
}

// ── Registry ─────────────────────────────────────────────────────────────────

/// Builds a [`DecisionQuery`] by appending the standard recency suffix to `keywords`.
fn dq(angle: &str, keywords: &str) -> DecisionQuery {
    DecisionQuery {
        angle: angle.to_owned(),
        query: format!("{keywords} agent architecture production systems 2025 2026"),
    }
}

static REGISTRY: LazyLock<HashMap<&'static str, Vec<DecisionQuery>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Vec<DecisionQuery>> = HashMap::new();

        m.insert("D1", vec![
            dq("event-model",      "event-driven trigger model LLM tool-calls agents"),
            dq("scheduling",       "async task scheduling priority queues agent backpressure"),
            dq("signal-handling",  "signal interrupt cancellation agent lifecycle handling"),
        ]);
        m.insert("D2A", vec![
            dq("identity",         "agent identity authentication authorization principal model"),
            dq("multi-tenant",     "multi-tenant agent isolation namespace partition segregation"),
            dq("session-mgmt",     "session management agent state persistence resumability"),
        ]);
        m.insert("D2B", vec![
            dq("storage",          "agent memory storage vector DB retrieval backends"),
            dq("context-window",   "context window LLM token limits management strategies"),
            dq("retrieval",        "retrieval augmented generation RAG agent memory recall"),
        ]);
        m.insert("D2C", vec![
            dq("tiered-memory",        "tiered memory hot warm cold agent knowledge store"),
            dq("knowledge-mgmt",       "knowledge management agent long-term memory distillation"),
            dq("context-engineering",  "context engineering prompt construction agent reasoning"),
        ]);
        m.insert("D2D", vec![
            dq("tool-patterns",    "tool use patterns function calling agents parallel"),
            dq("scalability",      "scalable agent tool registry dynamic capability discovery"),
            dq("mcp-ecosystem",    "MCP model context protocol server discovery tool loading"),
        ]);
        m.insert("D2E", vec![
            dq("budget-mgmt",      "token budget management LLM cost limits enforcement"),
            dq("compaction",       "context compaction summarization compression agent memory"),
            dq("cost-tracking",    "LLM cost tracking inference spend metering analytics"),
        ]);
        m.insert("D3A", vec![
            dq("model-routing",    "LLM routing model selection cost performance tradeoffs"),
            dq("multi-model",      "multi-model orchestration heterogeneous LLM backends agent"),
            dq("cost-perf",        "cost performance optimization inference agent pipeline"),
        ]);
        m.insert("D3B", vec![
            dq("durability",       "durable execution workflow persistence checkpointing agents"),
            dq("replay",           "event replay agent recovery state reconstruction"),
            dq("new-frameworks",   "agent framework temporal Inngest durable execution 2025"),
        ]);
        m.insert("D3C", vec![
            dq("retry-patterns",   "retry strategies exponential backoff LLM agent resilience"),
            dq("circuit-breaker",  "circuit breaker pattern agent LLM failure isolation"),
            dq("llm-errors",       "LLM error handling rate limits timeouts agent robustness"),
        ]);
        m.insert("D4A", vec![
            dq("isolation",           "agent isolation sandbox security boundaries enforcement"),
            dq("capability-model",    "capability model agent permissions least privilege"),
            dq("container-sandbox",   "container sandbox agent gVisor firecracker execution"),
        ]);
        m.insert("D4B", vec![
            dq("credentials",      "agent credential management secrets injection runtime"),
            dq("rotation",         "secret rotation agent credential refresh vault patterns"),
            dq("vault-patterns",   "HashiCorp Vault agent secrets dynamic credentials"),
        ]);
        m.insert("D4C", vec![
            dq("audit",            "agent audit logging compliance decision traceability"),
            dq("compliance",       "AI compliance governance SOC2 HIPAA agent regulatory"),
            dq("lineage",          "data lineage agent provenance tracking reproducibility"),
        ]);
        m.insert("D5", vec![
            dq("exit-conditions",  "agent termination conditions task completion criteria"),
            dq("composable-term",  "composable termination conditions agent pipeline"),
            dq("safety-stops",     "safety stops agent guardrails circuit breakers intervention"),
        ]);
        m.insert("C1", vec![
            dq("delegation",          "agent delegation sub-agent spawning task decomposition"),
            dq("context-passing",     "context passing parent child agent communication"),
            dq("isolation-vs-sharing","agent isolation vs context sharing tradeoffs"),
        ]);
        m.insert("C2", vec![
            dq("result-return",      "agent result return structured output formats"),
            dq("output-schema",      "output schema validation agent structured JSON"),
            dq("error-propagation",  "error propagation agent hierarchy failure handling"),
        ]);
        m.insert("C3", vec![
            dq("lifecycle",          "agent lifecycle management start stop health monitoring"),
            dq("health-checks",      "health check liveness readiness agent probes"),
            dq("graceful-shutdown",  "graceful shutdown agent task drain cleanup"),
        ]);
        m.insert("C4", vec![
            dq("protocols",       "agent communication protocols interoperability standards"),
            dq("a2a",             "agent-to-agent A2A Google protocol interaction patterns"),
            dq("mcp-evolution",   "MCP protocol evolution extensions capability negotiation"),
        ]);
        m.insert("C5", vec![
            dq("observability",   "agent observability distributed tracing metrics logging"),
            dq("opentelemetry",   "OpenTelemetry agent instrumentation spans traces OTEL"),
            dq("debugging",       "agent debugging inspection replay production troubleshoot"),
        ]);
        m.insert("L1", vec![
            dq("write-patterns",  "agent state write patterns append-only immutable log"),
            dq("consistency",     "consistency models agent distributed state convergence"),
            dq("scoped-state",    "scoped state management agent partition isolation"),
        ]);
        m.insert("L2", vec![
            dq("compaction",   "log compaction agent state size reduction strategies"),
            dq("strategies",   "agent memory compaction strategies LLM context reduction"),
            dq("benchmarks",   "agent memory compaction benchmarks performance measurement"),
        ]);
        m.insert("L3", vec![
            dq("crash-recovery",  "agent crash recovery restart resumption failure tolerance"),
            dq("checkpointing",   "checkpointing agent state persistence recovery points"),
            dq("event-sourcing",  "event sourcing agent state reconstruction CQRS"),
        ]);
        m.insert("L4", vec![
            dq("budget-tracking", "agent budget tracking token spend enforcement policies"),
            dq("metering",        "LLM metering agent usage analytics billing"),
            dq("limits",          "rate limiting agent resource quotas enforcement"),
        ]);
        m.insert("L5", vec![
            dq("tracing",      "distributed tracing agent spans context propagation Jaeger"),
            dq("otel",         "OpenTelemetry OTEL agent metrics traces logs integration"),
            dq("dashboards",   "agent observability dashboards Grafana metrics visualization"),
        ]);

        m
    });

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns the full list of query angles registered for `decision_id`, or
/// `None` if the ID is unknown.
pub fn queries_for(decision_id: &str) -> Option<&'static [DecisionQuery]> {
    REGISTRY.get(decision_id).map(|v| v.as_slice())
}

/// Selects the next query for `decision_id` using round-robin rotation over
/// `sweep_count`.
///
/// Returns `None` if `decision_id` is not registered.
pub fn next_query(decision_id: &str, sweep_count: usize) -> Option<&'static DecisionQuery> {
    let qs = queries_for(decision_id)?;
    Some(&qs[sweep_count % qs.len()])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_23_decisions_have_queries() {
        let expected = [
            "D1", "D2A", "D2B", "D2C", "D2D", "D2E",
            "D3A", "D3B", "D3C",
            "D4A", "D4B", "D4C",
            "D5",
            "C1", "C2", "C3", "C4", "C5",
            "L1", "L2", "L3", "L4", "L5",
        ];
        for id in &expected {
            let qs = queries_for(id);
            assert!(qs.is_some(), "missing queries for {id}");
            assert!(qs.unwrap().len() >= 3, "{id} has fewer than 3 query angles");
        }
    }

    #[test]
    fn next_query_rotates() {
        let q0 = next_query("D1", 0).unwrap();
        let q1 = next_query("D1", 1).unwrap();
        let q2 = next_query("D1", 2).unwrap();
        let q3 = next_query("D1", 3).unwrap();
        assert_ne!(q0.angle, q1.angle);
        assert_eq!(q0.angle, q3.angle);
        // keep q2 used to satisfy the compiler while matching the spec's test body
        let _ = q2;
    }

    #[test]
    fn unknown_decision_returns_none() {
        assert!(next_query("NONEXISTENT", 0).is_none());
    }

    #[test]
    fn query_lengths_within_limit() {
        for (id, qs) in REGISTRY.iter() {
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
}
