//! Three-layer deduplication for PR evidence novelty checks.
//!
//! Layer 1 compares source URLs; Layer 2 uses term-Jaccard similarity.
//! Layer 3 (LLM novelty check) is deferred to the caller.

use std::collections::HashSet;

use neuron_op_sweep::SweepVerdict;
use neuron_state_sqlite::search::term_jaccard;
use regex::Regex;

use crate::types::DedupResult;

/// Extract all URLs from a markdown text.
///
/// Handles both:
/// - Parenthesised markdown links: `[text](url)`
/// - Bare `http://` / `https://` URLs
///
/// Returns a [`HashSet`] of unique URL strings found in `text`.
pub fn extract_urls(text: &str) -> HashSet<String> {
    let mut urls = HashSet::new();

    // Markdown links: [text](url) — capture the URL inside the parens.
    let md_re = Regex::new(r"\[(?:[^\[\]]*)\]\((https?://[^\s)]+)\)").expect("valid regex");
    for cap in md_re.captures_iter(text) {
        if let Some(url) = cap.get(1) {
            urls.insert(url.as_str().to_string());
        }
    }

    // Bare URLs not already captured by the markdown link pattern.
    // We require that the URL is not immediately preceded by '(' (already matched above).
    let bare_re = Regex::new(r"(?:^|[^(])(https?://[^\s)>\]]+)").expect("valid regex");
    for cap in bare_re.captures_iter(text) {
        if let Some(url) = cap.get(1) {
            urls.insert(url.as_str().to_string());
        }
    }

    urls
}

/// Perform Layer 1 and Layer 2 deduplication checks.
///
/// # Layer 1 — Source URL comparison
/// Collects URLs from `verdict.evidence` and from `existing_pr_body`. If
/// every evidence URL is already present in the existing PR, returns
/// [`DedupResult::Redundant`].
///
/// # Layer 2 — Term-Jaccard similarity
/// Computes `term_jaccard(verdict.narrative, existing_pr_body)`:
/// - Above `jaccard_threshold` → [`DedupResult::Redundant`]
/// - Below `0.4` → [`DedupResult::New`]
/// - Between `0.4` and `jaccard_threshold` → [`DedupResult::New`]
///   (Layer 3 LLM check is deferred to the caller for this range)
///
/// # Empty existing body
/// If `existing_pr_body` is empty, always returns [`DedupResult::New`].
pub fn dedup_check_layers_1_2(
    verdict: &SweepVerdict,
    existing_pr_body: &str,
    jaccard_threshold: f64,
) -> DedupResult {
    // Empty existing body means there is nothing to deduplicate against.
    if existing_pr_body.is_empty() {
        return DedupResult::New;
    }

    // Layer 1: source URL set comparison.
    let existing_urls = extract_urls(existing_pr_body);
    let all_new_urls_already_seen = verdict
        .evidence
        .iter()
        .all(|e| existing_urls.contains(&e.source_url));

    if !verdict.evidence.is_empty() && all_new_urls_already_seen {
        return DedupResult::Redundant;
    }

    // Layer 2: term-Jaccard similarity of narrative vs existing body.
    let jaccard = term_jaccard(&verdict.narrative, existing_pr_body);

    if jaccard > jaccard_threshold {
        return DedupResult::Redundant;
    }

    // Jaccard <= threshold (including the 0.4–threshold ambiguous range).
    // Layer 3 (LLM check) is deferred to the caller; we return New here.
    DedupResult::New
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuron_op_sweep::{
        EvidenceItem, EvidenceStance, ProcessorTier, SweepVerdict, VerdictStatus,
    };

    fn make_verdict(narrative: &str, evidence_urls: &[&str]) -> SweepVerdict {
        SweepVerdict {
            decision_id: "topic-3b".to_string(),
            status: VerdictStatus::Refined,
            confidence: 0.8,
            num_supporting: evidence_urls.len(),
            num_contradicting: 0,
            cost_usd: 0.10,
            processor: ProcessorTier::Base,
            duration_secs: 3.0,
            swept_at: "2026-03-04T12:00:00Z".to_string(),
            evidence: evidence_urls
                .iter()
                .map(|url| EvidenceItem {
                    source_url: url.to_string(),
                    summary: "test evidence".to_string(),
                    stance: EvidenceStance::Supporting,
                    retrieved_at: "2026-03-04T11:55:00Z".to_string(),
                })
                .collect(),
            narrative: narrative.to_string(),
            proposed_diff: None,
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),
        }
    }

    // --- extract_urls tests ---

    #[test]
    fn extract_urls_from_markdown_links() {
        let text = "See [paper](https://example.com/paper) for details.";
        let urls = extract_urls(text);
        assert!(
            urls.contains("https://example.com/paper"),
            "markdown link not extracted"
        );
    }

    #[test]
    fn extract_urls_bare_url() {
        let text = "Read more at https://example.com/bare and then done.";
        let urls = extract_urls(text);
        assert!(
            urls.contains("https://example.com/bare"),
            "bare URL not extracted"
        );
    }

    #[test]
    fn extract_urls_mixed() {
        let text =
            "See [paper](https://example.com/md) and also https://example.com/bare for info.";
        let urls = extract_urls(text);
        assert!(urls.contains("https://example.com/md"));
        assert!(urls.contains("https://example.com/bare"));
    }

    #[test]
    fn extract_urls_no_urls() {
        let text = "No URLs here, just plain text.";
        let urls = extract_urls(text);
        assert!(urls.is_empty(), "expected empty set");
    }

    #[test]
    fn extract_urls_deduplicates() {
        let text = "First https://example.com/dup and second https://example.com/dup again.";
        let urls = extract_urls(text);
        assert_eq!(urls.len(), 1);
    }

    // --- dedup_check_layers_1_2 tests ---

    #[test]
    fn dedup_empty_existing_body_is_always_new() {
        let verdict = make_verdict("some narrative text", &["https://example.com/a"]);
        let result = dedup_check_layers_1_2(&verdict, "", 0.7);
        assert_eq!(result, DedupResult::New);
    }

    #[test]
    fn dedup_all_same_urls_is_redundant() {
        // All evidence URLs already exist in the PR body.
        let existing = "Prior evidence: [source](https://example.com/a) was found.\n\
                        Also [alt](https://example.com/b).";
        let verdict = make_verdict(
            "completely different narrative about other topics",
            &["https://example.com/a", "https://example.com/b"],
        );
        let result = dedup_check_layers_1_2(&verdict, existing, 0.7);
        assert_eq!(result, DedupResult::Redundant);
    }

    #[test]
    fn dedup_new_url_and_low_jaccard_is_new() {
        // New URL present AND narrative is clearly different from existing body.
        let existing = "The quick brown fox jumps over the lazy dog.";
        let verdict = make_verdict(
            "durable execution crash recovery patterns state machines",
            &["https://example.com/brand-new-source"],
        );
        let result = dedup_check_layers_1_2(&verdict, existing, 0.7);
        assert_eq!(result, DedupResult::New);
    }

    #[test]
    fn dedup_high_jaccard_is_redundant() {
        // Narrative is nearly identical to existing body, so Jaccard > 0.7.
        let text = "durable execution crash recovery patterns resilient systems design";
        let verdict = make_verdict(text, &["https://example.com/new-url-but-same-text"]);
        // existing body is the same text
        let result = dedup_check_layers_1_2(&verdict, text, 0.7);
        assert_eq!(result, DedupResult::Redundant);
    }

    #[test]
    fn dedup_no_evidence_items_skips_layer1() {
        // verdict has no evidence items — layer 1 can't trigger Redundant.
        // With high jaccard it should still be Redundant.
        let text = "identical narrative repeated here identical narrative repeated here";
        let verdict = make_verdict(text, &[]);
        let result = dedup_check_layers_1_2(&verdict, text, 0.7);
        assert_eq!(result, DedupResult::Redundant);
    }
}
