//! Core types for sweep verdicts and evidence.

use serde::{Deserialize, Serialize};

/// Outcome of comparing research findings against a decision.
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
    /// Too soon since last sweep, or dedup matched.
    Skipped,
}

/// Stance of a piece of evidence relative to the decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStance {
    /// Evidence supports the decision.
    Supporting,
    /// Evidence contradicts the decision.
    Contradicting,
    /// Tangentially related but neither supports nor contradicts.
    Neutral,
}

/// A single piece of evidence cited in a verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// URL of the source.
    pub source_url: String,
    /// Title of the source.
    pub title: String,
    /// Stance relative to the decision.
    pub stance: EvidenceStance,
    /// One-sentence summary of how this source relates.
    pub summary: String,
}

/// Research depth tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorTier {
    /// Quick search (~5s).
    Base,
    /// Standard depth.
    Core,
    /// Deep research (5-25 min).
    #[default]
    Ultra,
}

impl ProcessorTier {
    /// Convert to the string used by the Parallel.ai API.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessorTier::Base => "base",
            ProcessorTier::Core => "core",
            ProcessorTier::Ultra => "ultra",
        }
    }
}

/// Structured verdict produced by [`CompareOperator`](crate::CompareOperator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepVerdict {
    /// Decision identifier (e.g. "trigger", "communication").
    pub decision_id: String,
    /// Verdict outcome.
    pub status: VerdictStatus,
    /// Confidence score 0.0–1.0.
    pub confidence: f64,
    /// Count of supporting evidence items.
    pub num_supporting: usize,
    /// Count of contradicting evidence items.
    pub num_contradicting: usize,
    /// USD cost of this comparison.
    pub cost_usd: f64,
    /// Processor tier used for research.
    pub processor: ProcessorTier,
    /// Wall-clock duration in seconds.
    pub duration_secs: f64,
    /// RFC 3339 timestamp of when this verdict was produced.
    pub swept_at: String,
    /// Cited evidence items.
    pub evidence: Vec<EvidenceItem>,
    /// Markdown narrative explaining the verdict.
    pub narrative: String,
    /// Optional diff proposing changes to the decision text.
    pub proposed_diff: Option<String>,
    /// Research inputs that were compared against.
    pub research_inputs: Vec<crate::provider::ResearchResult>,
    /// Query used for research.
    pub query: String,
    /// Angle/facet of the query.
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
