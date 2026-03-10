//! A-MEM memory graph evolution tests for [`CozoStore`].
//!
//! Validates that the CozoDB backend supports the full cognitive memory
//! architecture: node write with embeddings, HNSW vector search, link
//! creation, and Datalog graph traversal — all in a single evolving graph.

#[cfg(feature = "cozo")]
#[tokio::test]
async fn memory_note_graph_evolution() {
    use layer0::effect::Scope;
    use layer0::state::{MemoryLink, StateStore as _};
    use serde_json::json;
    use skg_state_cozo::CozoStore;

    let store = CozoStore::memory().unwrap();
    let scope = Scope::Global;

    // ── Step 1: build base embeddings (1536-dim, padded with 0.0) ────────────

    let mut emb_rust = vec![0.0f32; 1536];
    emb_rust[0] = 0.9;
    emb_rust[1] = 0.1;

    let mut emb_safety = vec![0.0f32; 1536];
    emb_safety[0] = 0.8;
    emb_safety[1] = 0.2;

    let mut emb_python = vec![0.0f32; 1536];
    emb_python[1] = 0.1;
    emb_python[2] = 0.9;

    // ── Step 2: store notes as nodes ─────────────────────────────────────────

    store
        .write_node(
            &scope,
            "note-rust",
            json!("Rust systems programming language"),
            "note",
            0.9,
            &emb_rust,
        )
        .await
        .unwrap();

    store
        .write_node(
            &scope,
            "note-safety",
            json!("Memory safety guarantees in Rust"),
            "note",
            0.85,
            &emb_safety,
        )
        .await
        .unwrap();

    store
        .write_node(
            &scope,
            "note-python",
            json!("Python data science ecosystem"),
            "note",
            0.7,
            &emb_python,
        )
        .await
        .unwrap();

    // ── Step 3: link related notes ───────────────────────────────────────────

    // note-rust is also a KV node so link() can resolve it.
    store
        .write(&scope, "note-rust", json!("Rust systems programming language"))
        .await
        .unwrap();
    store
        .write(&scope, "note-safety", json!("Memory safety guarantees in Rust"))
        .await
        .unwrap();

    let link_rust_safety = MemoryLink::new("note-rust", "note-safety", "related");
    store.link(&scope, &link_rust_safety).await.unwrap();

    // ── Step 4: vector search — rust first, safety second ────────────────────

    let mut query_vec = vec![0.0f32; 1536];
    query_vec[0] = 0.85;
    query_vec[1] = 0.15;

    let results = store.vector_search(&scope, &query_vec, 3).await.unwrap();
    assert!(
        !results.is_empty(),
        "vector search must return results for Rust-like query"
    );

    let keys: Vec<&str> = results.iter().map(|r| r.key.as_str()).collect();
    assert!(
        keys.contains(&"note-rust"),
        "note-rust must appear in vector search results, got {keys:?}"
    );
    assert!(
        keys.contains(&"note-safety"),
        "note-safety must appear in vector search results, got {keys:?}"
    );
    assert_eq!(
        results[0].key, "note-rust",
        "note-rust must be the nearest neighbour, got {keys:?}"
    );

    // ── Step 5: add note-borrow-checker ──────────────────────────────────────

    let mut emb_borrow = vec![0.0f32; 1536];
    emb_borrow[0] = 0.85;
    emb_borrow[1] = 0.15;
    emb_borrow[2] = 0.05;

    store
        .write_node(
            &scope,
            "note-borrow-checker",
            json!("Rust borrow checker enforces ownership"),
            "note",
            0.88,
            &emb_borrow,
        )
        .await
        .unwrap();

    store
        .write(
            &scope,
            "note-borrow-checker",
            json!("Rust borrow checker enforces ownership"),
        )
        .await
        .unwrap();

    // ── Step 6: link borrow-checker → rust and borrow-checker → safety ───────

    let link_bc_rust = MemoryLink::new("note-borrow-checker", "note-rust", "related");
    store.link(&scope, &link_bc_rust).await.unwrap();

    let link_bc_safety = MemoryLink::new("note-borrow-checker", "note-safety", "related");
    store.link(&scope, &link_bc_safety).await.unwrap();

    // ── Step 7: traverse from note-borrow-checker ────────────────────────────

    let from_bc = store
        .traverse(&scope, "note-borrow-checker", Some("related"), 1)
        .await
        .unwrap();

    assert!(
        from_bc.contains(&"note-rust".to_string()),
        "traverse from note-borrow-checker must reach note-rust, got {from_bc:?}"
    );
    assert!(
        from_bc.contains(&"note-safety".to_string()),
        "traverse from note-borrow-checker must reach note-safety, got {from_bc:?}"
    );

    // ── Step 8: traverse from note-rust ──────────────────────────────────────

    let from_rust = store
        .traverse(&scope, "note-rust", Some("related"), 1)
        .await
        .unwrap();

    assert!(
        from_rust.contains(&"note-safety".to_string()),
        "traverse from note-rust must reach note-safety, got {from_rust:?}"
    );

    // ── Step 9: note-python must NOT appear in any traversal ─────────────────

    assert!(
        !from_bc.contains(&"note-python".to_string()),
        "note-python must NOT be reachable from note-borrow-checker, got {from_bc:?}"
    );
    assert!(
        !from_rust.contains(&"note-python".to_string()),
        "note-python must NOT be reachable from note-rust, got {from_rust:?}"
    );
}
