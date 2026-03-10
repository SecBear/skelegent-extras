//! Reciprocal Rank Fusion for combining heterogeneous ranked result lists.

use layer0::state::SearchResult;
use std::collections::HashMap;

/// Reciprocal Rank Fusion constant. Standard value is 60.
pub const RRF_K: f64 = 60.0;

/// Fuse multiple ranked result lists using RRF.
///
/// RRF(d) = Σ 1/(k + rank_i(d)) for each ranking list i.
///
/// Ranks are 1-based: the first element in each list has rank 1.
/// Results are deduplicated by key, with RRF scores summed across lists.
/// The output is sorted descending by fused score.
pub fn rrf_fuse(ranked_lists: &[Vec<SearchResult>], k: f64) -> Vec<SearchResult> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for list in ranked_lists {
        for (zero_idx, result) in list.iter().enumerate() {
            // RRF uses 1-based rank: rank = zero_idx + 1.
            let rrf_score = 1.0 / (k + (zero_idx as f64) + 1.0);
            *scores.entry(result.key.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut fused: Vec<SearchResult> = scores
        .into_iter()
        .map(|(key, score)| SearchResult::new(key, score))
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_list_passthrough() {
        let list = vec![
            SearchResult::new("a", 0.9),
            SearchResult::new("b", 0.5),
        ];
        let fused = rrf_fuse(&[list], 60.0);
        // Both keys present; "a" ranked first (rank 1 → higher RRF score).
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].key, "a");
        assert_eq!(fused[1].key, "b");
    }

    #[test]
    fn score_is_strictly_positive() {
        let list = vec![SearchResult::new("x", 1.0)];
        let fused = rrf_fuse(&[list], 60.0);
        assert!(fused[0].score > 0.0);
    }
}
