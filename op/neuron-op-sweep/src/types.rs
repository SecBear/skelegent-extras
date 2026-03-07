//! Data model for sweep operator verdicts and evidence.

use serde::{Deserialize, Serialize};

/// Classification of a sweep run's outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictStatus {
    /// Decision still holds; no changes needed.
    Confirmed,
    /// Decision holds but wording or evidence needs updating.
    Refined,
    /// New evidence contradicts the decision.
    Challenged,
    /// Decision is no longer relevant to the current landscape.
    Obsoleted,
    /// Too soon since last sweep, or dedup matched — no action taken.
    Skipped,
}

/// Stance of an evidence item relative to the decision under review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStance {
    /// Evidence supports the decision.
    Supporting,
    /// Evidence contradicts the decision.
    Contradicting,
    /// Evidence is tangentially related but neither supports nor contradicts.
    Neutral,
}

/// A single piece of supporting or contradicting evidence from research.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// Source URL (research report, paper, repository, or blog post).
    pub source_url: String,
    /// One-line summary of what this source says.
    pub summary: String,
    /// Whether this source supports, contradicts, or is neutral toward the decision.
    pub stance: EvidenceStance,
    /// RFC 3339 timestamp of when this source was retrieved.
    pub retrieved_at: String,
}

/// Parallel.ai processor tier used for a sweep run.
///
/// Higher tiers use more capable (and more expensive) models.
/// The operator selects the tier based on budget and previous verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorTier {
    /// Lowest cost. Used for routine sweeps when budget is below 80%.
    Base,
    /// Mid-tier. Used for refined decisions and budget-constrained challenged decisions.
    Core,
    /// Highest capability. Used for challenged decisions when budget allows.
    Ultra,
}

impl Default for ProcessorTier {
    fn default() -> Self {
        Self::Ultra
    }
}

/// Complete verdict produced by one sweep operator run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepVerdict {
    /// Which decision was swept.
    pub decision_id: String,
    /// The verdict classification.
    pub status: VerdictStatus,
    /// LLM-reported confidence in the verdict (0.0–1.0). Advisory only.
    pub confidence: f64,
    /// Count of supporting evidence sources found during research.
    pub num_supporting: usize,
    /// Count of contradicting evidence sources found during research.
    pub num_contradicting: usize,
    /// Total cost of this sweep run in USD.
    pub cost_usd: f64,
    /// Parallel.ai processor tier used for this run.
    pub processor: ProcessorTier,
    /// Wall-clock duration of the sweep in seconds.
    pub duration_secs: f64,
    /// RFC 3339 timestamp of when this sweep was completed.
    pub swept_at: String,
    /// Structured evidence items for machine processing and human review.
    pub evidence: Vec<EvidenceItem>,
    /// Markdown narrative: what was found, what changed, and why.
    ///
    /// This text is included verbatim in generated PR bodies.
    pub narrative: String,
    /// Proposed changes to the decision file, if `status == Refined`.
    ///
    /// Contains a unified diff or structured edit instructions.
    pub proposed_diff: Option<String>,
    /// Raw research results from Parallel.ai preserved for citation traceability.
    #[serde(default)]
    pub research_inputs: Vec<crate::provider::ResearchResult>,
    /// The search query sent to Parallel.ai for this sweep.
    #[serde(default)]
    pub query: String,
    /// Which facet/angle this query covers (e.g. "scalability", "security").
    #[serde(default)]
    pub query_angle: String,
}

/// Lightweight sweep metadata stored after each run for deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepMeta {
    /// RFC 3339 timestamp of when the sweep was completed.
    pub swept_at: String,
    /// Verdict from the sweep run.
    pub verdict: VerdictStatus,
    /// Cost of the sweep run in USD.
    pub cost_usd: f64,
    /// Search query used for this sweep.
    #[serde(default)]
    pub query: String,
    /// Query angle/facet identifier.
    #[serde(default)]
    pub query_angle: String,
    /// Processor tier used for research.
    #[serde(default)]
    pub processor: ProcessorTier,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_status_serde_round_trip() {
        let variants = [
            VerdictStatus::Confirmed,
            VerdictStatus::Refined,
            VerdictStatus::Challenged,
            VerdictStatus::Obsoleted,
            VerdictStatus::Skipped,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).expect("serialize");
            let back: VerdictStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(variant, &back, "round-trip failed for {:?}", variant);
        }
    }

    #[test]
    fn verdict_status_snake_case_encoding() {
        assert_eq!(
            serde_json::to_string(&VerdictStatus::Confirmed).unwrap(),
            "\"confirmed\""
        );
        assert_eq!(
            serde_json::to_string(&VerdictStatus::Challenged).unwrap(),
            "\"challenged\""
        );
    }

    #[test]
    fn sweep_verdict_serde_round_trip() {
        let verdict = SweepVerdict {
            decision_id: "topic-3b".to_string(),
            status: VerdictStatus::Confirmed,
            confidence: 0.85,
            num_supporting: 3,
            num_contradicting: 0,
            cost_usd: 0.12,
            processor: ProcessorTier::Base,
            duration_secs: 4.5,
            swept_at: "2026-03-04T18:00:00Z".to_string(),
            evidence: vec![EvidenceItem {
                source_url: "https://example.com/paper".to_string(),
                summary: "Confirms the approach".to_string(),
                stance: EvidenceStance::Supporting,
                retrieved_at: "2026-03-04T17:55:00Z".to_string(),
            }],
            narrative: "The decision is well-supported by current evidence.".to_string(),
            proposed_diff: None,
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),
        };

        let json = serde_json::to_string(&verdict).expect("serialize");
        let back: SweepVerdict = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.decision_id, verdict.decision_id);
        assert_eq!(back.status, verdict.status);
        assert!((back.confidence - verdict.confidence).abs() < f64::EPSILON);
        assert_eq!(back.num_supporting, verdict.num_supporting);
        assert_eq!(back.num_contradicting, verdict.num_contradicting);
        assert!((back.cost_usd - verdict.cost_usd).abs() < f64::EPSILON);
        assert_eq!(back.evidence.len(), 1);
        assert_eq!(back.evidence[0].source_url, verdict.evidence[0].source_url);
        assert_eq!(back.evidence[0].stance, verdict.evidence[0].stance);
        assert_eq!(back.proposed_diff, None);
    }

    #[test]
    fn evidence_item_serde_round_trip() {
        let item = EvidenceItem {
            source_url: "https://example.com".to_string(),
            summary: "Supports claim X".to_string(),
            stance: EvidenceStance::Contradicting,
            retrieved_at: "2026-03-04T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&item).expect("serialize");
        let back: EvidenceItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.source_url, item.source_url);
        assert_eq!(back.stance, item.stance);
    }

    #[test]
    fn sweep_verdict_with_proposed_diff() {
        let verdict = SweepVerdict {
            decision_id: "topic-1a".to_string(),
            status: VerdictStatus::Refined,
            confidence: 0.7,
            num_supporting: 2,
            num_contradicting: 0,
            cost_usd: 0.25,
            processor: ProcessorTier::Core,
            duration_secs: 7.2,
            swept_at: "2026-03-04T18:00:00Z".to_string(),
            evidence: vec![],
            narrative: "Minor update needed.".to_string(),
            proposed_diff: Some("--- a/decisions/topic-1a.md\n+++ b/decisions/topic-1a.md".to_string()),
            research_inputs: vec![],
            query: String::new(),
            query_angle: String::new(),
        };
        let json = serde_json::to_string(&verdict).unwrap();
        let back: SweepVerdict = serde_json::from_str(&json).unwrap();
        assert!(back.proposed_diff.is_some());
        assert_eq!(back.status, VerdictStatus::Refined);
    }
}
