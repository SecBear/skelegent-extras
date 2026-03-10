//! Synthesis agent types and helper logic for cross-decision structural analysis.
//!
//! Implements a two-pass analysis that detects structural shifts, contradictions,
//! emergent patterns, and candidate decisions after each full sweep cycle.
//!
//! # Two-Pass Architecture
//!
//! - **Pass 1 (broad scan):** All decision cards + latest deltas → flagged items.
//!   Use [`PASS_1_SYSTEM_PROMPT`] as the system prompt, replacing the
//!   `{min_flag_confidence}` placeholder with the configured threshold.
//!
//! - **Pass 2 (deep dive):** Each flagged item with full evidence →
//!   classified [`StructuralChange`] or [`CandidateDecision`].
//!   Use [`PASS_2_SYSTEM_PROMPT`] as the system prompt, replacing
//!   `{flag.summary}`, `{flag.decision_ids}`, and `{full_cards_and_deltas}`.

use serde::{Deserialize, Serialize};

/// Pass 1 system prompt for broad cross-decision scan.
///
/// Template variables (replace with [`str::replace`] or equivalent before
/// passing to an LLM call):
///
/// | Placeholder | Source |
/// |---|---|
/// | `{min_flag_confidence}` | [`SynthesisConfig::min_flag_confidence`] |
pub const PASS_1_SYSTEM_PROMPT: &str =
    "You are analyzing the architectural decision framework after a full \
sweep cycle. You have all 23 decision cards and their latest deltas.

For each decision, you see:
- Current verdict (confirmed/refined/challenged/obsoleted)
- Latest delta (what changed in this cycle)
- Decision card summary

Your task: Identify cross-decision patterns.

Output a JSON array of flagged items, each with:
- type: \"convergence\" | \"divergence\" | \"emergence\" | \"obsolescence\"
       | \"strengthening\" | \"weakening\"
- decision_ids: [list of 2+ decision IDs involved]
- summary: one-line description
- confidence: 0.0-1.0

Also identify potential new decisions:
- candidate_title: proposed title
- related_decisions: which existing decisions suggest this
- evidence_summary: why this should be a decision

Only flag items with confidence >= {min_flag_confidence}.
Only flag candidates observed by 2+ decisions in this cycle.";

/// Pass 2 system prompt for deep dive on a specific flagged pattern.
///
/// Template variables (replace before passing to an LLM call):
///
/// | Placeholder | Source |
/// |---|---|
/// | `{flag.summary}` | The flagged item's one-line summary |
/// | `{flag.decision_ids}` | Comma-separated list of decision IDs |
/// | `{full_cards_and_deltas}` | Full decision cards and sweep deltas for involved decisions |
pub const PASS_2_SYSTEM_PROMPT: &str = "Deep analysis of flagged cross-decision pattern.

FLAGGED PATTERN: {flag.summary}
DECISIONS INVOLVED: {flag.decision_ids}
FULL EVIDENCE FOR EACH DECISION:
{full_cards_and_deltas}

Produce detailed analysis:
1. Classify the change type precisely
2. Cite specific evidence from each decision's sweep results
3. Assess whether this is a one-cycle anomaly or persistent trend
4. For candidates: assess whether this merits a new formal decision
5. For relationship changes: specify exact edge updates needed

Output: StructuralChange or CandidateDecision JSON.";

/// Category of cross-decision structural change detected by synthesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralChangeType {
    /// Multiple decisions moving in the same direction.
    Convergence,
    /// Decisions pulling apart or developing tension.
    Divergence,
    /// New pattern spanning multiple decisions.
    Emergence,
    /// A decision or relationship becoming irrelevant.
    Obsolescence,
    /// Existing relationship between decisions getting stronger.
    Strengthening,
    /// Existing relationship getting weaker.
    Weakening,
}

/// A detected structural change across two or more decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralChange {
    /// Category of the structural change.
    pub change_type: StructuralChangeType,
    /// Decision IDs involved (2+ for a meaningful cross-decision change).
    pub decision_ids: Vec<String>,
    /// One-line summary of the change.
    pub summary: String,
    /// Detailed evidence narrative (markdown).
    pub evidence: String,
    /// Number of sweep cycles that have observed this change.
    pub observation_count: usize,
    /// Confidence in this classification (0.0–1.0).
    pub confidence: f64,
    /// Source URLs from research results that support this structural observation.
    #[serde(default)]
    pub source_urls: Vec<String>,
}

/// Lifecycle stage of a candidate decision in the promotion pipeline.
///
/// Transitions:
/// ```text
/// Observed → Staged   (automatic, via advance_candidate at threshold)
/// Staged   → Promoted (human action: file moved to decisions/)
/// Staged   → Dismissed (human action: rejection)
/// Observed → Dismissed (human action: early rejection)
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStage {
    /// First sighting — not yet confirmed across cycles.
    Observed,
    /// Seen in 3+ cycles; evidence files auto-created.
    Staged,
    /// Human moved the candidate to `decisions/`.
    Promoted,
    /// Human rejected the candidate.
    Dismissed,
}

/// A candidate decision identified by synthesis across one or more sweep cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDecision {
    /// Unique identifier for this candidate (e.g., `"candidate-L8"`).
    pub candidate_id: String,
    /// Proposed title for the candidate decision.
    pub title: String,
    /// Existing decision IDs whose sweep results suggest this candidate.
    pub related_decisions: Vec<String>,
    /// Accumulated evidence narrative (markdown).
    pub evidence: String,
    /// Number of sweep cycles in which this candidate was observed.
    pub observation_count: usize,
    /// Source URLs gathered across all observations.
    pub sources: Vec<String>,
    /// ISO 8601 timestamp of the first observation.
    pub first_observed: String,
    /// Current lifecycle stage of the candidate.
    pub stage: CandidateStage,
}

/// Kind of directed relationship between two decisions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipKind {
    /// One decision depends on another being true or stable.
    DependsOn,
    /// Two decisions are in direct contradiction.
    Contradicts,
    /// One decision supersedes another, making the other obsolete.
    Supersedes,
    /// Decisions are related without strict dependency or conflict.
    RelatedTo,
}

/// Action to apply to a relationship edge in the decision topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipAction {
    /// Create a new relationship edge.
    Add,
    /// Delete an existing relationship edge.
    Remove,
    /// Increase the weight or confidence of an existing edge.
    Strengthen,
    /// Decrease the weight or confidence of an existing edge.
    Weaken,
}

/// A proposed update to a directed relationship between two decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipUpdate {
    /// Source decision ID.
    pub from_decision: String,
    /// Target decision ID.
    pub to_decision: String,
    /// Kind of relationship between the two decisions.
    pub relationship: RelationshipKind,
    /// Action to apply to this edge.
    pub action: RelationshipAction,
    /// Evidence supporting this relationship update (markdown).
    pub evidence: String,
    /// Source URLs supporting this relationship change.
    #[serde(default)]
    pub source_urls: Vec<String>,
}

/// Full output of one synthesis agent run covering a complete sweep cycle.
///
/// An empty `structural_changes` and `candidates` list is a valid report
/// — it means the framework is stable with no cross-decision shifts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisReport {
    /// Structural changes detected across decisions this cycle.
    pub structural_changes: Vec<StructuralChange>,
    /// New or updated candidate decisions surfaced this cycle.
    pub candidates: Vec<CandidateDecision>,
    /// Proposed relationship edge updates for the decision topology.
    pub relationship_updates: Vec<RelationshipUpdate>,
    /// Human-readable health summary (markdown).
    pub health_summary: String,
    /// Total cost of both synthesis passes in USD.
    pub cost_usd: f64,
    /// Identifier of the sweep cycle this synthesis covers.
    pub cycle_id: String,
    /// RFC 3339 timestamp when synthesis completed.
    pub completed_at: String,
}

/// Configuration for the synthesis agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Minimum confidence to flag an item in Pass 1. Default: `0.6`.
    pub min_flag_confidence: f64,
    /// Maximum flagged items to deep-dive in Pass 2. Default: `10`.
    pub max_deep_dives: usize,
    /// Observation count threshold for auto-staging candidates. Default: `3`.
    pub candidate_staging_threshold: usize,
    /// Model name used for synthesis passes.
    ///
    /// Default: `"claude-sonnet-4-20250514"`.
    pub model: String,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            min_flag_confidence: 0.6,
            max_deep_dives: 10,
            candidate_staging_threshold: 3,
            model: "claude-sonnet-4-20250514".to_string(),
        }
    }
}

/// Returns `true` if `candidate` should be promoted to [`CandidateStage::Staged`].
///
/// Staging conditions (both must hold):
/// - `candidate.stage` is exactly [`CandidateStage::Observed`].
/// - `candidate.observation_count >= threshold`.
///
/// Already-staged, promoted, or dismissed candidates return `false` regardless
/// of observation count.
pub fn should_stage_candidate(candidate: &CandidateDecision, threshold: usize) -> bool {
    matches!(candidate.stage, CandidateStage::Observed) && candidate.observation_count >= threshold
}

/// Advances `candidate` to [`CandidateStage::Staged`] if the threshold is met.
///
/// This is a no-op when the candidate is already `Staged`, `Promoted`, or
/// `Dismissed`, or when `observation_count < threshold`.
pub fn advance_candidate(candidate: &mut CandidateDecision, threshold: usize) {
    if should_stage_candidate(candidate, threshold) {
        candidate.stage = CandidateStage::Staged;
    }
}

/// Merges a new observation into an existing candidate decision.
///
/// - Increments [`CandidateDecision::observation_count`] by one.
/// - Appends `new_evidence` to [`CandidateDecision::evidence`] (separated by
///   `"\n\n"` when the existing field is non-empty).
/// - Extends [`CandidateDecision::sources`] with the entries in `new_sources`.
///
/// This function does **not** advance the candidate's `stage`. Call
/// [`advance_candidate`] afterwards if promotion logic should run.
pub fn merge_candidate_observation(
    existing: &mut CandidateDecision,
    new_evidence: &str,
    new_sources: &[String],
) {
    existing.observation_count += 1;
    if !new_evidence.is_empty() {
        if !existing.evidence.is_empty() {
            existing.evidence.push_str("\n\n");
        }
        existing.evidence.push_str(new_evidence);
    }
    existing.sources.extend_from_slice(new_sources);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(stage: CandidateStage, observation_count: usize) -> CandidateDecision {
        CandidateDecision {
            candidate_id: "candidate-L8".to_string(),
            title: "Test Candidate".to_string(),
            related_decisions: vec!["topic-3b".to_string()],
            evidence: "Initial evidence.".to_string(),
            observation_count,
            sources: vec!["https://example.com".to_string()],
            first_observed: "2026-03-04T00:00:00Z".to_string(),
            stage,
        }
    }

    // --- Serde round-trip: StructuralChangeType (all 6 variants) ---

    #[test]
    fn structural_change_type_serde_round_trip() {
        let variants = [
            StructuralChangeType::Convergence,
            StructuralChangeType::Divergence,
            StructuralChangeType::Emergence,
            StructuralChangeType::Obsolescence,
            StructuralChangeType::Strengthening,
            StructuralChangeType::Weakening,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).expect("serialize");
            let back: StructuralChangeType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(v, &back, "round-trip failed for {:?}", v);
        }
    }

    #[test]
    fn structural_change_type_snake_case_encoding() {
        assert_eq!(
            serde_json::to_string(&StructuralChangeType::Convergence).unwrap(),
            "\"convergence\""
        );
        assert_eq!(
            serde_json::to_string(&StructuralChangeType::Obsolescence).unwrap(),
            "\"obsolescence\""
        );
        assert_eq!(
            serde_json::to_string(&StructuralChangeType::Strengthening).unwrap(),
            "\"strengthening\""
        );
    }

    // --- Serde round-trip: CandidateStage (all 4 variants) ---

    #[test]
    fn candidate_stage_serde_round_trip() {
        let variants = [
            CandidateStage::Observed,
            CandidateStage::Staged,
            CandidateStage::Promoted,
            CandidateStage::Dismissed,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).expect("serialize");
            let back: CandidateStage = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(v, &back, "round-trip failed for {:?}", v);
        }
    }

    #[test]
    fn candidate_stage_snake_case_encoding() {
        assert_eq!(
            serde_json::to_string(&CandidateStage::Observed).unwrap(),
            "\"observed\""
        );
        assert_eq!(
            serde_json::to_string(&CandidateStage::Dismissed).unwrap(),
            "\"dismissed\""
        );
    }

    // --- Serde round-trip: SynthesisReport ---

    #[test]
    fn synthesis_report_serde_round_trip() {
        let report = SynthesisReport {
            structural_changes: vec![StructuralChange {
                change_type: StructuralChangeType::Convergence,
                decision_ids: vec!["topic-1a".to_string(), "topic-3b".to_string()],
                summary: "Two decisions converging on async runtime model".to_string(),
                evidence: "Both reference tokio runtime primitives.".to_string(),
                observation_count: 2,
                confidence: 0.85,
                source_urls: vec![],
            }],
            candidates: vec![make_candidate(CandidateStage::Observed, 1)],
            relationship_updates: vec![RelationshipUpdate {
                from_decision: "topic-1a".to_string(),
                to_decision: "topic-3b".to_string(),
                relationship: RelationshipKind::DependsOn,
                action: RelationshipAction::Add,
                evidence: "topic-1a runtime relies on topic-3b crash recovery guarantees.".to_string(),
                source_urls: vec![],
            }],
            health_summary: "Framework stable.".to_string(),
            cost_usd: 0.08,
            cycle_id: "cycle-2026-03-04".to_string(),
            completed_at: "2026-03-04T18:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&report).expect("serialize");
        let back: SynthesisReport = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.structural_changes.len(), 1);
        assert_eq!(
            back.structural_changes[0].change_type,
            StructuralChangeType::Convergence
        );
        assert_eq!(back.structural_changes[0].decision_ids.len(), 2);
        assert!((back.structural_changes[0].confidence - 0.85).abs() < f64::EPSILON);
        assert_eq!(back.candidates.len(), 1);
        assert_eq!(back.candidates[0].stage, CandidateStage::Observed);
        assert_eq!(back.relationship_updates.len(), 1);
        assert_eq!(
            back.relationship_updates[0].relationship,
            RelationshipKind::DependsOn
        );
        assert_eq!(back.relationship_updates[0].action, RelationshipAction::Add);
        assert_eq!(back.cycle_id, "cycle-2026-03-04");
        assert!((back.cost_usd - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn synthesis_report_empty_is_valid() {
        let report = SynthesisReport {
            structural_changes: vec![],
            candidates: vec![],
            relationship_updates: vec![],
            health_summary: "framework stable, no cross-decision shifts".to_string(),
            cost_usd: 0.0,
            cycle_id: "cycle-2026-03-04".to_string(),
            completed_at: "2026-03-04T18:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let back: SynthesisReport = serde_json::from_str(&json).expect("deserialize");
        assert!(back.structural_changes.is_empty());
        assert!(back.candidates.is_empty());
        assert!(back.relationship_updates.is_empty());
    }

    // --- should_stage_candidate ---

    #[test]
    fn should_stage_below_threshold_returns_false() {
        let c = make_candidate(CandidateStage::Observed, 2);
        assert!(!should_stage_candidate(&c, 3));
    }

    #[test]
    fn should_stage_at_threshold_returns_true() {
        let c = make_candidate(CandidateStage::Observed, 3);
        assert!(should_stage_candidate(&c, 3));
    }

    #[test]
    fn should_stage_above_threshold_returns_true() {
        let c = make_candidate(CandidateStage::Observed, 5);
        assert!(should_stage_candidate(&c, 3));
    }

    #[test]
    fn should_stage_already_staged_returns_false() {
        let c = make_candidate(CandidateStage::Staged, 10);
        assert!(!should_stage_candidate(&c, 3));
    }

    #[test]
    fn should_stage_dismissed_returns_false() {
        let c = make_candidate(CandidateStage::Dismissed, 10);
        assert!(!should_stage_candidate(&c, 3));
    }

    #[test]
    fn should_stage_promoted_returns_false() {
        let c = make_candidate(CandidateStage::Promoted, 10);
        assert!(!should_stage_candidate(&c, 3));
    }

    // --- advance_candidate ---

    #[test]
    fn advance_candidate_observed_becomes_staged_at_threshold() {
        let mut c = make_candidate(CandidateStage::Observed, 3);
        advance_candidate(&mut c, 3);
        assert_eq!(c.stage, CandidateStage::Staged);
    }

    #[test]
    fn advance_candidate_below_threshold_stays_observed() {
        let mut c = make_candidate(CandidateStage::Observed, 2);
        advance_candidate(&mut c, 3);
        assert_eq!(c.stage, CandidateStage::Observed);
    }

    #[test]
    fn advance_candidate_dismissed_not_changed() {
        let mut c = make_candidate(CandidateStage::Dismissed, 10);
        advance_candidate(&mut c, 3);
        assert_eq!(c.stage, CandidateStage::Dismissed);
    }

    #[test]
    fn advance_candidate_staged_not_changed() {
        let mut c = make_candidate(CandidateStage::Staged, 10);
        advance_candidate(&mut c, 3);
        assert_eq!(c.stage, CandidateStage::Staged);
    }

    // --- merge_candidate_observation ---

    #[test]
    fn merge_increments_observation_count() {
        let mut c = make_candidate(CandidateStage::Observed, 1);
        merge_candidate_observation(&mut c, "More evidence.", &[]);
        assert_eq!(c.observation_count, 2);
    }

    #[test]
    fn merge_appends_evidence_with_separator() {
        let mut c = make_candidate(CandidateStage::Observed, 1);
        merge_candidate_observation(&mut c, "New finding.", &[]);
        assert!(c.evidence.contains("Initial evidence."));
        assert!(c.evidence.contains("New finding."));
        assert!(c.evidence.contains("\n\n"));
    }

    #[test]
    fn merge_empty_evidence_not_appended() {
        let mut c = make_candidate(CandidateStage::Observed, 1);
        let original = c.evidence.clone();
        merge_candidate_observation(&mut c, "", &[]);
        assert_eq!(c.evidence, original);
    }

    #[test]
    fn merge_extends_sources() {
        let mut c = make_candidate(CandidateStage::Observed, 1);
        let new_sources = vec!["https://new.example.com".to_string()];
        merge_candidate_observation(&mut c, "", &new_sources);
        assert_eq!(c.sources.len(), 2);
        assert!(c.sources.contains(&"https://new.example.com".to_string()));
    }

    #[test]
    fn merge_does_not_change_stage() {
        let mut c = make_candidate(CandidateStage::Observed, 1);
        merge_candidate_observation(&mut c, "Evidence.", &[]);
        assert_eq!(c.stage, CandidateStage::Observed);
    }

    #[test]
    fn merge_dismissed_does_not_change_stage() {
        let mut c = make_candidate(CandidateStage::Dismissed, 1);
        merge_candidate_observation(&mut c, "Evidence.", &[]);
        assert_eq!(c.stage, CandidateStage::Dismissed);
    }

    // --- SynthesisConfig defaults ---

    #[test]
    fn synthesis_config_default_values() {
        let config = SynthesisConfig::default();
        assert!(
            (config.min_flag_confidence - 0.6).abs() < f64::EPSILON,
            "min_flag_confidence default should be 0.6"
        );
        assert_eq!(config.max_deep_dives, 10);
        assert_eq!(config.candidate_staging_threshold, 3);
        assert_eq!(config.model, "claude-sonnet-4-20250514");
    }

    // --- Prompt constants ---

    #[test]
    fn pass1_prompt_non_empty_with_placeholder() {
        assert!(!PASS_1_SYSTEM_PROMPT.is_empty());
        assert!(
            PASS_1_SYSTEM_PROMPT.contains("{min_flag_confidence}"),
            "Pass 1 prompt must contain {{min_flag_confidence}} placeholder"
        );
    }

    #[test]
    fn pass2_prompt_non_empty_with_placeholders() {
        assert!(!PASS_2_SYSTEM_PROMPT.is_empty());
        assert!(
            PASS_2_SYSTEM_PROMPT.contains("{flag.summary}"),
            "Pass 2 prompt must contain {{flag.summary}} placeholder"
        );
        assert!(
            PASS_2_SYSTEM_PROMPT.contains("{flag.decision_ids}"),
            "Pass 2 prompt must contain {{flag.decision_ids}} placeholder"
        );
        assert!(
            PASS_2_SYSTEM_PROMPT.contains("{full_cards_and_deltas}"),
            "Pass 2 prompt must contain {{full_cards_and_deltas}} placeholder"
        );
    }
}
