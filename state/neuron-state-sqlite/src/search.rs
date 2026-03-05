//! FTS5 search helpers and similarity utilities.
//!
//! Provides BM25-ranked full-text search over entries and a term Jaccard
//! similarity function for deduplication in the sweep system.

use rusqlite::{Connection, params};
use std::collections::HashSet;

/// A raw FTS5 search result before mapping to `SearchResult`.
pub(crate) struct FtsMatch {
    /// The matched key.
    pub key: String,
    /// BM25 relevance score (lower is more relevant in SQLite FTS5).
    pub rank: f64,
    /// Snippet of the matched value.
    pub snippet: Option<String>,
}

/// Execute an FTS5 search within a scope.
///
/// Returns results ranked by BM25 (converted to descending score: higher = better).
/// The `query` is passed directly to FTS5 MATCH — callers should sanitize if needed.
///
/// # Errors
///
/// Returns `rusqlite::Error` on query failure.
pub(crate) fn fts5_search(
    conn: &Connection,
    scope_str: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<FtsMatch>, rusqlite::Error> {
    // FTS5 MATCH syntax. BM25 returns negative scores (more negative = more relevant).
    // We negate to get positive scores where higher = more relevant.
    let mut stmt = conn.prepare(
        "SELECT e.key, -rank AS score, snippet(entries_fts, 2, '»', '«', '…', 32) AS snip
         FROM entries_fts
         JOIN entries e ON entries_fts.rowid = e.rowid
         WHERE entries_fts MATCH ?1
           AND entries_fts.scope = ?2
         ORDER BY rank
         LIMIT ?3",
    )?;

    let rows = stmt.query_map(params![query, scope_str, limit as i64], |row| {
        Ok(FtsMatch {
            key: row.get(0)?,
            rank: row.get(1)?,
            snippet: row.get(2)?,
        })
    })?;

    rows.collect()
}

/// Compute term Jaccard similarity between two texts.
///
/// Tokenizes both texts into lowercase word sets and returns
/// `|intersection| / |union|`. Returns 0.0 if both are empty.
///
/// This is the v1 diversity metric for [`SaliencePackingStrategy`] and
/// deduplication in the PR generator (DECISION-010, DECISION-019).
pub fn term_jaccard(a: &str, b: &str) -> f64 {
    let set_a = tokenize(a);
    let set_b = tokenize(b);

    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Tokenize text into a set of lowercase words.
///
/// Splits on whitespace and punctuation (except hyphens and underscores,
/// matching the FTS5 `tokenchars` configuration). Strips remaining
/// punctuation from token edges.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '-' && c != '_'))
        .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '-' && c != '_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_texts() {
        let score = term_jaccard("hello world", "hello world");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_completely_different() {
        let score = term_jaccard("hello world", "foo bar");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_partial_overlap() {
        // "hello" is shared, total union = {hello, world, foo} = 3
        let score = term_jaccard("hello world", "hello foo");
        assert!((score - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn jaccard_empty_texts() {
        assert!((term_jaccard("", "") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_case_insensitive() {
        let score = term_jaccard("Hello World", "hello world");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_preserves_hyphens_underscores() {
        // "durable-execution" should be one token, not split
        let score = term_jaccard("durable-execution pattern", "durable-execution approach");
        // shared: durable-execution. union: durable-execution, pattern, approach = 3
        assert!((score - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn tokenize_technical_terms() {
        let tokens = tokenize("state_store and durable-execution");
        assert!(tokens.contains("state_store"));
        assert!(tokens.contains("durable-execution"));
        assert!(tokens.contains("and"));
    }

    #[test]
    fn tokenize_strips_edge_punctuation() {
        let tokens = tokenize("(hello) [world]");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
    }
}
