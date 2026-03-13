//! Hybrid FTS + HNSW search tests for [`CozoStore`].
//!
//! Tests requiring a real CozoDB backend are gated on `#[cfg(feature = "cozo")]`.
//! The `hybrid_search_stub_returns_error` test runs on the HashMap backend only.

// ── cozo backend: real hybrid search behaviour ────────────────────────────────

#[cfg(feature = "cozo")]
mod cozo_tests {
    use layer0::effect::Scope;
    use layer0::state::{SearchResult, StateStore as _};
    use serde_json::json;
    use skg_state_cozo::search::rrf_fuse;
    use skg_state_cozo::CozoStore;

    /// A node that matches both FTS and HNSW should rank above nodes that
    /// match only one index.
    #[tokio::test]
    async fn hybrid_search_fuses_fts_and_vector() {
        let store = CozoStore::memory().unwrap();
        let scope = Scope::Global;

        // "alpha" embedding — strongly in direction [1, 0, 0, …]
        let mut emb_alpha = vec![0.0f32; 1536];
        emb_alpha[0] = 1.0;

        // "beta" embedding — close to query but not as close as alpha
        let mut emb_beta = vec![0.0f32; 1536];
        emb_beta[0] = 0.9;
        emb_beta[1] = 0.2;

        // alpha: indexed by both FTS (kv table) and HNSW (node table)
        store
            .write(&scope, "alpha", json!("needle found here"))
            .await
            .unwrap();
        store
            .write_node(
                &scope,
                "alpha",
                json!({"text": "needle found here"}),
                "concept",
                1.0,
                &emb_alpha,
            )
            .await
            .unwrap();

        // beta: indexed only by HNSW — no "needle" text
        store
            .write_node(
                &scope,
                "beta",
                json!({"text": "unrelated content"}),
                "concept",
                0.5,
                &emb_beta,
            )
            .await
            .unwrap();

        // gamma: indexed only by FTS — no embedding
        store
            .write(&scope, "gamma", json!("needle in a haystack"))
            .await
            .unwrap();

        // Query vector aligns with emb_alpha.
        let mut query_vec = vec![0.0f32; 1536];
        query_vec[0] = 1.0;

        let results = store
            .hybrid_search(&scope, "needle", &query_vec, 10)
            .await
            .unwrap();

        assert!(!results.is_empty(), "hybrid search must return results");
        assert_eq!(
            results[0].key, "alpha",
            "alpha matches both FTS and HNSW — must rank first; got {:?}",
            results.iter().map(|r| &r.key).collect::<Vec<_>>()
        );
    }

    /// `rrf_fuse` must deduplicate keys that appear in multiple lists and
    /// assign them the sum of their per-list RRF scores.
    #[test]
    fn rrf_fuse_deduplicates() {
        let list_a = vec![
            SearchResult::new("x", 0.9),
            SearchResult::new("y", 0.7),
        ];
        let list_b = vec![
            SearchResult::new("x", 0.8),
            SearchResult::new("z", 0.6),
        ];

        let fused = rrf_fuse(&[list_a, list_b], 60.0);

        // "x" appears in both lists and must be deduplicated.
        let x_count = fused.iter().filter(|r| r.key == "x").count();
        assert_eq!(x_count, 1, "x must appear exactly once after deduplication");

        // "x" ranks first because its fused score is the sum of two RRF contributions.
        assert_eq!(
            fused[0].key, "x",
            "x must rank first; got {:?}",
            fused.iter().map(|r| &r.key).collect::<Vec<_>>()
        );

        // All three unique keys must be present.
        assert_eq!(fused.len(), 3, "must have 3 unique keys");
    }

    /// Edge cases: empty slice and slice of empty lists both return empty output.
    #[test]
    fn rrf_fuse_empty_lists() {
        // Empty slice of lists.
        let fused = rrf_fuse(&[], 60.0);
        assert!(fused.is_empty(), "empty input must produce empty output");

        // Slice containing empty lists.
        let fused2 = rrf_fuse(&[vec![], vec![]], 60.0);
        assert!(
            fused2.is_empty(),
            "empty sub-lists must produce empty output"
        );
    }
}

// ── HashMap backend: stub error path ─────────────────────────────────────────

#[cfg(not(feature = "cozo"))]
#[tokio::test]
async fn hybrid_search_stub_returns_error() {
    use layer0::effect::Scope;
    use skg_state_cozo::CozoStore;

    let store = CozoStore::memory().unwrap();
    let scope = Scope::Global;
    let v = vec![0.0f32; 1536];

    let err = store.hybrid_search(&scope, "query", &v, 5).await;
    assert!(err.is_err(), "hybrid_search must fail without cozo feature");
}
