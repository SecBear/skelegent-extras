//! StateStore trait compliance tests for [`CozoStore`].
//!
//! All tests use the in-memory engine — no filesystem, no network.

use layer0::effect::Scope;
use layer0::id::{AgentId, SessionId, WorkflowId};
use layer0::state::{MemoryLink, StateStore};
use neuron_state_cozo::CozoStore;
use serde_json::json;

// ── helpers ──────────────────────────────────────────────────────────────────

fn store() -> CozoStore {
    CozoStore::memory().expect("in-memory CozoStore")
}

fn session(id: &str) -> Scope {
    Scope::Session(SessionId::new(id))
}

fn workflow(id: &str) -> Scope {
    Scope::Workflow(WorkflowId::new(id))
}

fn agent(wf: &str, ag: &str) -> Scope {
    Scope::Agent {
        workflow: WorkflowId::new(wf),
        agent: AgentId::new(ag),
    }
}

// ── basic CRUD ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn read_write_round_trip() {
    let s = store();
    let scope = Scope::Global;
    let value = json!({"name": "Alice", "age": 30});

    s.write(&scope, "user:alice", value.clone()).await.unwrap();
    let got = s.read(&scope, "user:alice").await.unwrap();
    assert_eq!(got, Some(value));
}

#[tokio::test]
async fn read_missing_returns_none() {
    let s = store();
    let got = s.read(&Scope::Global, "no-such-key").await.unwrap();
    assert_eq!(got, None);
}

#[tokio::test]
async fn write_overwrites_existing() {
    let s = store();
    let scope = Scope::Global;

    s.write(&scope, "k", json!("first")).await.unwrap();
    s.write(&scope, "k", json!("second")).await.unwrap();
    let got = s.read(&scope, "k").await.unwrap();
    assert_eq!(got, Some(json!("second")));
}

// ── delete ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn delete_removes_value() {
    let s = store();
    let scope = session("s1");

    s.write(&scope, "key", json!("val")).await.unwrap();
    s.delete(&scope, "key").await.unwrap();
    let got = s.read(&scope, "key").await.unwrap();
    assert_eq!(got, None, "deleted key must return None");
}

#[tokio::test]
async fn delete_nonexistent_is_noop() {
    let s = store();
    s.delete(&Scope::Global, "ghost").await.unwrap();
}

// ── list ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_by_prefix() {
    let s = store();
    let scope = workflow("wf-1");

    s.write(&scope, "doc:a", json!("alpha")).await.unwrap();
    s.write(&scope, "doc:b", json!("beta")).await.unwrap();
    s.write(&scope, "cfg:timeout", json!(30)).await.unwrap();

    let mut keys = s.list(&scope, "doc:").await.unwrap();
    keys.sort();
    assert_eq!(keys, vec!["doc:a", "doc:b"]);
}

#[tokio::test]
async fn list_empty_prefix_returns_all_in_scope() {
    let s = store();
    let scope = session("s-list");

    s.write(&scope, "a", json!(1)).await.unwrap();
    s.write(&scope, "b", json!(2)).await.unwrap();
    s.write(&scope, "c", json!(3)).await.unwrap();

    let mut keys = s.list(&scope, "").await.unwrap();
    keys.sort();
    assert_eq!(keys, vec!["a", "b", "c"]);
}

#[tokio::test]
async fn list_returns_empty_for_unknown_prefix() {
    let s = store();
    let scope = Scope::Global;

    s.write(&scope, "foo:bar", json!(1)).await.unwrap();

    let keys = s.list(&scope, "xyz:").await.unwrap();
    assert!(keys.is_empty());
}

// ── search ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn search_returns_results() {
    let s = store();
    let scope = Scope::Global;

    s.write(
        &scope,
        "doc:graph",
        json!("CozoDB graph search with Datalog"),
    )
    .await
    .unwrap();
    s.write(&scope, "doc:unrelated", json!("the weather is nice"))
        .await
        .unwrap();

    let results = s.search(&scope, "Datalog", 10).await.unwrap();
    assert!(!results.is_empty(), "search must find Datalog mention");
    assert!(
        results.iter().any(|r| r.key == "doc:graph"),
        "doc:graph must be in results"
    );
}

#[tokio::test]
async fn search_is_case_insensitive() {
    let s = store();
    let scope = Scope::Global;

    s.write(&scope, "item", json!("CamelCase Content")).await.unwrap();

    let results = s.search(&scope, "camelcase", 10).await.unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn search_empty_query_returns_empty() {
    let s = store();
    let scope = Scope::Global;

    s.write(&scope, "k", json!("v")).await.unwrap();

    let results = s.search(&scope, "", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn search_respects_limit() {
    let s = store();
    let scope = Scope::Global;

    for i in 0..10 {
        s.write(&scope, &format!("entry:{i}"), json!("match me"))
            .await
            .unwrap();
    }

    let results = s.search(&scope, "match", 3).await.unwrap();
    assert!(results.len() <= 3, "result count must not exceed limit");
}

// ── graph: link / traverse ────────────────────────────────────────────────────

#[tokio::test]
async fn link_and_traverse() {
    let s = store();
    let scope = agent("wf-1", "planner");

    // Ensure nodes exist in the KV store.
    s.write(&scope, "node:A", json!("source node")).await.unwrap();
    s.write(&scope, "node:B", json!("target node")).await.unwrap();

    let link = MemoryLink::new("node:A", "node:B", "references");
    s.link(&scope, &link).await.unwrap();

    let reached = s.traverse(&scope, "node:A", Some("references"), 1).await.unwrap();
    assert!(
        reached.contains(&"node:B".to_string()),
        "traverse must reach node:B from node:A"
    );
}

#[tokio::test]
async fn traverse_any_relation() {
    let s = store();
    let scope = Scope::Global;

    let link_a = MemoryLink::new("root", "child1", "contains");
    let link_b = MemoryLink::new("root", "child2", "references");
    s.link(&scope, &link_a).await.unwrap();
    s.link(&scope, &link_b).await.unwrap();

    let reached = s
        .traverse(&scope, "root", None, 1)
        .await
        .unwrap();
    assert!(reached.contains(&"child1".to_string()));
    assert!(reached.contains(&"child2".to_string()));
}

#[tokio::test]
async fn traverse_depth_zero_returns_empty() {
    let s = store();
    let scope = Scope::Global;

    let link = MemoryLink::new("a", "b", "rel");
    s.link(&scope, &link).await.unwrap();

    let reached = s.traverse(&scope, "a", None, 0).await.unwrap();
    assert!(reached.is_empty(), "depth=0 must return empty vec");
}

#[tokio::test]
async fn traverse_multi_hop() {
    let s = store();
    let scope = Scope::Global;

    // Chain: A → B → C
    s.link(&scope, &MemoryLink::new("A", "B", "next")).await.unwrap();
    s.link(&scope, &MemoryLink::new("B", "C", "next")).await.unwrap();

    let reached_1 = s.traverse(&scope, "A", Some("next"), 1).await.unwrap();
    assert_eq!(reached_1, vec!["B"]);

    let reached_2 = s.traverse(&scope, "A", Some("next"), 2).await.unwrap();
    assert!(reached_2.contains(&"B".to_string()));
    assert!(reached_2.contains(&"C".to_string()));
}

// ── graph: unlink ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn unlink_removes_edge() {
    let s = store();
    let scope = session("s-graph");

    let link = MemoryLink::new("src", "dst", "points_to");
    s.link(&scope, &link).await.unwrap();

    // Verify the edge exists.
    let before = s.traverse(&scope, "src", Some("points_to"), 1).await.unwrap();
    assert!(before.contains(&"dst".to_string()), "edge must exist before unlink");

    s.unlink(&scope, "src", "dst", "points_to").await.unwrap();

    let after = s.traverse(&scope, "src", Some("points_to"), 1).await.unwrap();
    assert!(after.is_empty(), "traverse must return empty after unlink");
}

#[tokio::test]
async fn unlink_nonexistent_is_noop() {
    let s = store();
    s.unlink(&Scope::Global, "x", "y", "rel").await.unwrap();
}

#[tokio::test]
async fn link_deduplicates() {
    let s = store();
    let scope = Scope::Global;

    let link = MemoryLink::new("a", "b", "r");
    s.link(&scope, &link).await.unwrap();
    s.link(&scope, &link).await.unwrap(); // second call must not duplicate

    let reached = s.traverse(&scope, "a", Some("r"), 1).await.unwrap();
    assert_eq!(reached.len(), 1, "duplicate links must be collapsed to one");
}

// ── scope isolation ───────────────────────────────────────────────────────────

#[tokio::test]
async fn scope_isolation() {
    let s = store();
    let scope_a = session("scope-A");
    let scope_b = session("scope-B");

    s.write(&scope_a, "key", json!("from-A")).await.unwrap();

    // key written in scope_a must NOT be visible in scope_b.
    let got = s.read(&scope_b, "key").await.unwrap();
    assert_eq!(got, None, "scope_b must not see scope_a writes");
}

#[tokio::test]
async fn scope_isolation_list() {
    let s = store();
    let scope_a = workflow("wf-A");
    let scope_b = workflow("wf-B");

    s.write(&scope_a, "doc:1", json!(1)).await.unwrap();
    s.write(&scope_b, "doc:2", json!(2)).await.unwrap();

    let keys_a = s.list(&scope_a, "").await.unwrap();
    let keys_b = s.list(&scope_b, "").await.unwrap();

    assert_eq!(keys_a, vec!["doc:1"]);
    assert_eq!(keys_b, vec!["doc:2"]);
}

#[tokio::test]
async fn graph_scope_isolation() {
    let s = store();
    let scope_a = session("graph-A");
    let scope_b = session("graph-B");

    // Link exists only in scope_a.
    s.link(&scope_a, &MemoryLink::new("x", "y", "r")).await.unwrap();

    let in_a = s.traverse(&scope_a, "x", Some("r"), 1).await.unwrap();
    let in_b = s.traverse(&scope_b, "x", Some("r"), 1).await.unwrap();

    assert!(in_a.contains(&"y".to_string()), "edge must be visible in scope_a");
    assert!(in_b.is_empty(), "edge must NOT be visible in scope_b");
}

// ── all Scope variants ────────────────────────────────────────────────────────

#[tokio::test]
async fn all_scope_variants_work() {
    let s = store();

    let scopes = vec![
        Scope::Global,
        session("s1"),
        workflow("wf-1"),
        agent("wf-1", "a1"),
        Scope::Custom("pipeline/stage-1".to_string()),
    ];

    for scope in &scopes {
        s.write(scope, "k", json!("v")).await.unwrap();
        let got = s.read(scope, "k").await.unwrap();
        assert_eq!(got, Some(json!("v")), "read must succeed for {:?}", scope);
    }
}

// ── object safety ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn object_safety_box_dyn_state_store() {
    let store: Box<dyn StateStore> = Box::new(store());
    let scope = Scope::Global;

    store.write(&scope, "k", json!("v")).await.unwrap();
    let val = store.read(&scope, "k").await.unwrap();
    assert_eq!(val, Some(json!("v")));
}

#[tokio::test]
async fn object_safety_arc_dyn_state_store() {
    use std::sync::Arc;
    let s: Arc<dyn StateStore> = Arc::new(store());
    let scope = Scope::Global;

    s.write(&scope, "k", json!("v")).await.unwrap();
    let val = s.read(&scope, "k").await.unwrap();
    assert_eq!(val, Some(json!("v")));
}

#[test]
fn cozo_store_implements_state_store() {
    fn _assert<T: StateStore>() {}
    _assert::<CozoStore>();
}

// ── complex values ────────────────────────────────────────────────────────────

#[tokio::test]
async fn complex_json_round_trip() {
    let s = store();
    let scope = Scope::Global;

    let complex = json!({
        "nested": {"array": [1, 2, 3]},
        "bool": true,
        "null_field": null,
        "float": 1.234
    });

    s.write(&scope, "complex", complex.clone()).await.unwrap();
    let got = s.read(&scope, "complex").await.unwrap();
    assert_eq!(got, Some(complex));
}

// ── cycle safety ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn traverse_cycle_does_not_loop() {
    let s = store();
    let scope = Scope::Global;

    // A ↔ B cycle.
    s.link(&scope, &MemoryLink::new("A", "B", "r")).await.unwrap();
    s.link(&scope, &MemoryLink::new("B", "A", "r")).await.unwrap();

    // Must terminate even with depth > 1.
    let reached = s.traverse(&scope, "A", Some("r"), 5).await.unwrap();
    assert!(reached.contains(&"B".to_string()), "B must be reachable");
    assert!(!reached.contains(&"A".to_string()), "A (start) must not appear in results");
}
