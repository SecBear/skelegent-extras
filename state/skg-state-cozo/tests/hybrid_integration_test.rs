//! Full-stack integration test for the CozoDB capability suite.
//!
//! Exercises every Tier-2 capability in a single scenario:
//! HNSW vector search, BM25 FTS, RRF hybrid fusion, Datalog graph traversal,
//! and the transient scratchpad — all against the same in-memory store.

#[cfg(feature = "cozo")]
mod cozo_integration {
    use layer0::effect::Scope;
    use layer0::state::{Lifetime, MemoryLink, StateStore as _, StoreOptions};
    use serde_json::json;
    use skg_state_cozo::CozoStore;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a 1536-dim F32 unit vector with `1.0` at index `i` and `0.0`
    /// everywhere else. Used to construct orthogonal embeddings whose cosine
    /// distances are well-defined.
    fn unit_vec(i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 1536];
        v[i] = 1.0;
        v
    }

    // ── test ─────────────────────────────────────────────────────────────────

    /// Kitchen-sink integration test: all five capability axes exercised
    /// against a single in-memory store with five nodes, three edges, and one
    /// transient scratchpad entry.
    ///
    /// Pass conditions are documented inline for each assertion block.
    #[tokio::test]
    async fn full_capability_stack() {
        let store = CozoStore::memory().unwrap();
        let scope = Scope::Global;

        // ── 1. Write 5 nodes ─────────────────────────────────────────────────
        //
        // node_a: indexed by BOTH FTS (kv table) and HNSW (node table).
        //         Its embedding aligns with the [0]-axis — it will be the
        //         nearest neighbour for queries close to unit_vec(0).
        //         Its text contains "quantum", so it also wins FTS.
        //
        // node_b: HNSW + FTS. Embedding aligns with [1]-axis. Text contains
        //         "quantum" — ranked by FTS but farther from query vector.
        //
        // node_c: HNSW + FTS. Embedding at [2]-axis. Text is off-topic for
        //         "quantum" — found by vector proximity but not FTS.
        //
        // node_d: HNSW only — no KV entry, so FTS cannot find it.
        //
        // node_e: FTS only — no node entry, so HNSW cannot find it.

        let emb_a = unit_vec(0);
        let emb_b = unit_vec(1);
        let emb_c = unit_vec(2);
        let emb_d = unit_vec(3);

        // node_a: both FTS and HNSW
        store
            .write(&scope, "node_a", json!("quantum computing fundamentals"))
            .await
            .unwrap();
        store
            .write_node(
                &scope,
                "node_a",
                json!({"text": "quantum computing fundamentals"}),
                "concept",
                1.0,
                &emb_a,
            )
            .await
            .unwrap();

        // node_b: both FTS and HNSW
        store
            .write(&scope, "node_b", json!("quantum machine learning algorithms"))
            .await
            .unwrap();
        store
            .write_node(
                &scope,
                "node_b",
                json!({"text": "quantum machine learning algorithms"}),
                "concept",
                0.8,
                &emb_b,
            )
            .await
            .unwrap();

        // node_c: both FTS and HNSW — text does NOT contain "quantum"
        store
            .write(&scope, "node_c", json!("neural network architecture"))
            .await
            .unwrap();
        store
            .write_node(
                &scope,
                "node_c",
                json!({"text": "neural network architecture"}),
                "concept",
                0.7,
                &emb_c,
            )
            .await
            .unwrap();

        // node_d: HNSW only (no KV / FTS entry)
        store
            .write_node(
                &scope,
                "node_d",
                json!({"text": "isolated hnsw node"}),
                "concept",
                0.5,
                &emb_d,
            )
            .await
            .unwrap();

        // node_e: FTS only (no node / HNSW entry)
        store
            .write(&scope, "node_e", json!("quantum physics experiments"))
            .await
            .unwrap();

        // ── 2. Write edges: node_a→node_b, node_b→node_c, node_a→node_c ────

        store
            .link(&scope, &MemoryLink::new("node_a", "node_b", "relates_to"))
            .await
            .unwrap();
        store
            .link(&scope, &MemoryLink::new("node_b", "node_c", "relates_to"))
            .await
            .unwrap();
        store
            .link(&scope, &MemoryLink::new("node_a", "node_c", "relates_to"))
            .await
            .unwrap();

        // ── 3. Hybrid search: FTS + HNSW signals combined via RRF ───────────
        //
        // Query text = "quantum", query vector ≈ unit_vec(0) (alias of node_a).
        //
        // node_a matches BOTH FTS ("quantum" hit) and HNSW (nearest vector).
        // It must rank first because its RRF score is the sum of two positive
        // contributions — one from each list.

        let query_vec = unit_vec(0);
        let hybrid_results = store
            .hybrid_search(&scope, "quantum", &query_vec, 10)
            .await
            .unwrap();

        assert!(
            !hybrid_results.is_empty(),
            "hybrid_search must return at least one result"
        );
        assert_eq!(
            hybrid_results[0].key, "node_a",
            "node_a matches both FTS and HNSW — must rank first; got {:?}",
            hybrid_results.iter().map(|r| &r.key).collect::<Vec<_>>()
        );

        // node_b and node_e appear in the FTS list; node_b also appears in
        // HNSW (it has an embedding). All three must be present somewhere.
        let hybrid_keys: Vec<&str> = hybrid_results.iter().map(|r| r.key.as_str()).collect();
        assert!(
            hybrid_keys.contains(&"node_b"),
            "node_b must appear in hybrid results (FTS hit on 'quantum')"
        );
        assert!(
            hybrid_keys.contains(&"node_e"),
            "node_e must appear in hybrid results (FTS-only hit on 'quantum')"
        );

        // ── 4. Graph traversal from node_a ───────────────────────────────────
        //
        // Direct edges from node_a: → node_b, → node_c.
        // With max_depth=1 we see only immediate neighbours (node_b, node_c).
        // node_a itself must NOT appear in the result.

        let mut reached = store
            .traverse(&scope, "node_a", Some("relates_to"), 1)
            .await
            .unwrap();
        reached.sort();

        assert!(
            reached.contains(&"node_b".to_string()),
            "traverse from node_a must reach node_b; got {reached:?}"
        );
        assert!(
            reached.contains(&"node_c".to_string()),
            "traverse from node_a must reach node_c; got {reached:?}"
        );
        assert!(
            !reached.contains(&"node_a".to_string()),
            "starting node must not appear in traverse result"
        );

        // With max_depth=2 we additionally reach node_c via node_a→node_b→node_c.
        // The result is still {node_b, node_c} because node_c is already
        // reachable at depth 1 — BFS cycle-safety deduplicates it.
        let reached2 = store
            .traverse(&scope, "node_a", Some("relates_to"), 2)
            .await
            .unwrap();
        assert!(
            reached2.contains(&"node_b".to_string()) && reached2.contains(&"node_c".to_string()),
            "depth-2 traverse must still find node_b and node_c; got {reached2:?}"
        );

        // ── 5. Pure vector search: nearest-neighbour ordering ────────────────
        //
        // Query vector = unit_vec(0) → node_a is the exact match (distance 0).
        // node_b (unit_vec(1)) is orthogonal → farther away.
        // Result[0] must be node_a.

        let vec_results = store
            .vector_search(&scope, &query_vec, 5)
            .await
            .unwrap();

        assert!(
            !vec_results.is_empty(),
            "vector_search must return results"
        );
        assert_eq!(
            vec_results[0].key, "node_a",
            "nearest neighbour to unit_vec(0) must be node_a; got {:?}",
            vec_results.iter().map(|r| &r.key).collect::<Vec<_>>()
        );
        // node_d has an embedding (unit_vec(3)) so it must appear too.
        let vec_keys: Vec<&str> = vec_results.iter().map(|r| r.key.as_str()).collect();
        assert!(
            vec_keys.contains(&"node_d"),
            "node_d (HNSW-only) must appear in vector results"
        );
        // node_e has no embedding — it must NOT appear.
        assert!(
            !vec_keys.contains(&"node_e"),
            "node_e (FTS-only, no embedding) must not appear in vector results"
        );

        // ── 6. FTS search: text relevance ────────────────────────────────────
        //
        // "quantum" appears in node_a, node_b, and node_e KV values.
        // node_d has no KV entry → must not appear.

        let fts_results = store.search(&scope, "quantum", 10).await.unwrap();
        assert!(
            !fts_results.is_empty(),
            "FTS search for 'quantum' must return results"
        );
        let fts_keys: Vec<&str> = fts_results.iter().map(|r| r.key.as_str()).collect();
        assert!(
            fts_keys.contains(&"node_a"),
            "node_a must be in FTS results (value contains 'quantum')"
        );
        assert!(
            fts_keys.contains(&"node_b"),
            "node_b must be in FTS results (value contains 'quantum')"
        );
        assert!(
            fts_keys.contains(&"node_e"),
            "node_e must be in FTS results (FTS-only node with 'quantum')"
        );
        assert!(
            !fts_keys.contains(&"node_d"),
            "node_d must NOT appear in FTS results (no KV entry)"
        );

        // ── 7. Transient scratchpad ───────────────────────────────────────────
        //
        // Write a transient entry, verify it is immediately readable, then
        // call clear_transient and verify it disappears.

        let transient_opts = StoreOptions {
            lifetime: Some(Lifetime::Transient),
            ..Default::default()
        };

        store
            .write_hinted(&scope, "scratch:tmp", json!("ephemeral data"), &transient_opts)
            .await
            .unwrap();

        let before_clear = store.read(&scope, "scratch:tmp").await.unwrap();
        assert_eq!(
            before_clear,
            Some(json!("ephemeral data")),
            "transient entry must be readable before clear_transient"
        );

        store.clear_transient();

        let after_clear = store.read(&scope, "scratch:tmp").await.unwrap();
        assert_eq!(
            after_clear, None,
            "transient entry must be gone after clear_transient"
        );

        // Durable KV entries must survive clear_transient.
        let durable_after = store.read(&scope, "node_a").await.unwrap();
        assert!(
            durable_after.is_some(),
            "durable entry (node_a) must survive clear_transient"
        );
    }
}
