//! HNSW vector search tests for [`CozoStore`].
//!
//! Real HNSW tests are gated behind `#[cfg(feature = "cozo")]` because the
//! HashMap backend does not support proximity indices. The non-cozo tests
//! verify that the stub methods surface the expected error.

use layer0::effect::Scope;
use serde_json::json;
use skg_state_cozo::CozoStore;

// ── cozo backend: real HNSW behaviour ────────────────────────────────────────

#[cfg(feature = "cozo")]
#[tokio::test]
async fn write_and_vector_search_node() {
    let store = CozoStore::memory().unwrap();
    let scope = Scope::Global;

    // Two 1536-dim vectors that differ primarily in their first two components.
    let mut v_cat = vec![0.0f32; 1536];
    v_cat[0] = 0.9;
    v_cat[1] = 0.1;

    let mut v_dog = vec![0.0f32; 1536];
    v_dog[0] = 0.1;
    v_dog[1] = 0.9;

    store
        .write_node(&scope, "cat", json!("cats are fluffy"), "concept", 0.8, &v_cat)
        .await
        .unwrap();
    store
        .write_node(&scope, "dog", json!("dogs are loyal"), "concept", 0.7, &v_dog)
        .await
        .unwrap();

    // Query vector is close to v_cat.
    let mut query = vec![0.0f32; 1536];
    query[0] = 1.0;

    let results = store.vector_search(&scope, &query, 2).await.unwrap();
    assert!(!results.is_empty(), "vector search must return results");
    // Nearest neighbour must be 'cat' (cosine distance to v_cat is smaller).
    assert_eq!(
        results[0].key, "cat",
        "nearest neighbour must be 'cat', got {:?}",
        results.iter().map(|r| &r.key).collect::<Vec<_>>()
    );
}

#[cfg(feature = "cozo")]
#[tokio::test]
async fn vector_search_respects_scope() {
    use layer0::id::SessionId;

    let store = CozoStore::memory().unwrap();
    let scope_a = Scope::Global;
    let scope_b = Scope::Session(SessionId::new("s1"));

    let mut v = vec![0.0f32; 1536];
    v[0] = 1.0;

    store
        .write_node(&scope_a, "global_node", json!("in global scope"), "concept", 1.0, &v)
        .await
        .unwrap();

    // Search in scope_b — must not see the node written to scope_a.
    let results = store.vector_search(&scope_b, &v, 5).await.unwrap();
    assert!(
        results.is_empty(),
        "vector search must not cross scope boundaries, got {:?}",
        results.iter().map(|r| &r.key).collect::<Vec<_>>()
    );
}

#[cfg(feature = "cozo")]
#[tokio::test]
async fn vector_search_zero_limit_returns_empty() {
    let store = CozoStore::memory().unwrap();
    let scope = Scope::Global;

    let mut v = vec![0.0f32; 1536];
    v[0] = 1.0;

    store
        .write_node(&scope, "k", json!("data"), "concept", 1.0, &v)
        .await
        .unwrap();

    let results = store.vector_search(&scope, &v, 0).await.unwrap();
    assert!(results.is_empty(), "limit=0 must return empty vec");
}

// ── HashMap backend: stub error paths ────────────────────────────────────────

#[cfg(not(feature = "cozo"))]
#[tokio::test]
async fn write_node_returns_error_without_cozo() {
    let store = CozoStore::memory().unwrap();
    let scope = Scope::Global;
    let v = vec![0.0f32; 1536];

    let err = store
        .write_node(&scope, "k", json!("v"), "concept", 1.0, &v)
        .await;
    assert!(err.is_err(), "write_node must fail without cozo feature");
}

#[cfg(not(feature = "cozo"))]
#[tokio::test]
async fn vector_search_returns_error_without_cozo() {
    let store = CozoStore::memory().unwrap();
    let scope = Scope::Global;
    let v = vec![0.0f32; 1536];

    let err = store.vector_search(&scope, &v, 5).await;
    assert!(err.is_err(), "vector_search must fail without cozo feature");
}
