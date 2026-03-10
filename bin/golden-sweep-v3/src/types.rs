//! Core sweep types — re-exported from `neuron-op-sweep-v2` plus v3 additions.
//!
//! v3 re-uses verdict, evidence, and processor tier types from v2 for
//! compatibility with existing tooling. `SweepDecision` and `select_processor`
//! live here because they are pure data / pure logic that depend only on this
//! module's own types.

pub use neuron_op_sweep_v2::provider::ResearchResult;
pub use neuron_op_sweep_v2::types::{
    EvidenceItem, EvidenceStance, ProcessorTier, SweepMeta, SweepVerdict, VerdictStatus,
};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SweepDecision
// ---------------------------------------------------------------------------

/// Per-decision input for a sweep cycle.
///
/// The caller assembles one `SweepDecision` per decision under review and
/// passes the full slice to [`crate::cycle::sweep_cycle`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepDecision {
    /// Decision identifier (e.g. `"D1"`, `"C3"`).
    pub id: String,
    /// Research results for this decision (caller-supplied).
    ///
    /// When empty, the cycle short-circuits to `Confirmed(0.3)` without
    /// calling the LLM.
    pub research_results: Vec<ResearchResult>,
    /// Previous verdict, if any.
    ///
    /// Used by [`select_processor`] to upgrade the research tier for decisions
    /// that were recently `Challenged` or `Refined`.
    pub previous_verdict: Option<VerdictStatus>,
}

// ---------------------------------------------------------------------------
// select_processor
// ---------------------------------------------------------------------------

/// Select the Parallel.ai research processor tier for a decision.
///
/// Rules:
/// - `Challenged` + > 50% budget remaining → `Ultra`
/// - `Challenged` + ≤ 50% budget remaining → `Core`
/// - `Refined` → `Core`
/// - Anything else + > 80% budget remaining → `Core`
/// - Anything else + ≤ 80% budget remaining → `Base`
pub fn select_processor(
    budget_remaining_usd: f64,
    budget_total_usd: f64,
    previous_verdict: Option<&VerdictStatus>,
) -> ProcessorTier {
    if budget_total_usd <= 0.0 {
        return ProcessorTier::Base;
    }
    let ratio = budget_remaining_usd / budget_total_usd;
    match previous_verdict {
        Some(VerdictStatus::Challenged) => {
            if ratio > 0.5 {
                ProcessorTier::Ultra
            } else {
                ProcessorTier::Core
            }
        }
        Some(VerdictStatus::Refined) => ProcessorTier::Core,
        _ => {
            if ratio > 0.8 {
                ProcessorTier::Core
            } else {
                ProcessorTier::Base
            }
        }
    }
}
