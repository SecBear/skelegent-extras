//! Priority queue for sweep cycle scheduling.
//!
//! Decisions are ranked by a composite score that weights staleness, previous
//! verdict velocity, and a small round-robin tiebreaker so every decision gets
//! swept eventually.

use neuron_op_sweep::VerdictStatus;
use serde::{Deserialize, Serialize};

/// A decision queued for sweeping, with its computed priority score.
///
/// Higher `priority` values are swept sooner. The [`Ord`] implementation
/// orders by priority descending so a [`std::collections::BinaryHeap`] pops
/// the highest-priority decision first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedDecision {
    /// Unique identifier for the decision (e.g. `"topic-3b"`).
    pub decision_id: String,
    /// Computed priority score. Higher = sweep sooner. Range: ~0.0–2.0.
    pub priority: f64,
    /// How many days have elapsed since the last sweep.
    pub staleness_days: f64,
    /// Verdict from the most recent sweep run, if any.
    pub previous_verdict: Option<VerdictStatus>,
    /// Estimated cost in USD for this sweep, used for pre-flight budget checks.
    pub estimated_cost_usd: f64,
}

// Manual PartialEq/Eq because f64 does not implement Eq.
impl PartialEq for QueuedDecision {
    fn eq(&self, other: &Self) -> bool {
        self.priority.total_cmp(&other.priority) == std::cmp::Ordering::Equal
    }
}

impl Eq for QueuedDecision {}

impl PartialOrd for QueuedDecision {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Orders by priority ascending so that a max-heap ([`std::collections::BinaryHeap`])
/// pops the **highest**-priority decision first.
impl Ord for QueuedDecision {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

/// Compute the priority score for a decision.
///
/// Higher scores are swept sooner. Inputs:
///
/// - `staleness_days` — days since last sweep. Capped at 90 days and
///   normalized to `[0, 1]`.
/// - `previous_verdict` — the last known verdict. [`VerdictStatus::Challenged`]
///   and [`VerdictStatus::Obsoleted`] get a 2× velocity boost;
///   [`VerdictStatus::Refined`] gets 1.5×; all others 1×.
/// - `cycle_position` — 0-indexed position in the current ordering. Used to
///   compute a tiny (≤ 1%) deterministic tiebreaker that ensures round-robin
///   fairness across decisions with the same staleness.
/// - `total_decisions` — total number of decisions in this cycle. Must be > 0
///   for the tiebreaker to be meaningful; when 0 the tiebreaker defaults to 1.0.
///
/// Formula: `staleness_score × velocity_boost × round_robin`
pub fn compute_priority(
    staleness_days: f64,
    previous_verdict: Option<&VerdictStatus>,
    cycle_position: usize,
    total_decisions: usize,
) -> f64 {
    let staleness_score = staleness_days.min(90.0) / 90.0;

    let velocity_boost = match previous_verdict {
        Some(VerdictStatus::Challenged) => 2.0,
        Some(VerdictStatus::Refined) => 1.5,
        Some(VerdictStatus::Obsoleted) => 2.0,
        _ => 1.0,
    };

    let round_robin = if total_decisions > 0 {
        1.0 - (cycle_position as f64 / total_decisions as f64) * 0.01
    } else {
        1.0
    };

    staleness_score * velocity_boost * round_robin
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    fn queued(id: &str, priority: f64) -> QueuedDecision {
        QueuedDecision {
            decision_id: id.to_string(),
            priority,
            staleness_days: 30.0,
            previous_verdict: None,
            estimated_cost_usd: 0.30,
        }
    }

    // -----------------------------------------------------------------------
    // Ordering — BinaryHeap pops highest-priority first
    // -----------------------------------------------------------------------

    #[test]
    fn binary_heap_pops_highest_priority_first() {
        let mut heap = BinaryHeap::new();
        heap.push(queued("low", 0.3));
        heap.push(queued("high", 0.9));
        heap.push(queued("mid", 0.6));

        assert_eq!(heap.pop().unwrap().decision_id, "high");
        assert_eq!(heap.pop().unwrap().decision_id, "mid");
        assert_eq!(heap.pop().unwrap().decision_id, "low");
    }

    #[test]
    fn ord_higher_priority_greater() {
        let a = queued("a", 1.0);
        let b = queued("b", 0.5);
        assert!(a > b, "higher priority must compare as greater");
        assert!(b < a);
    }

    // -----------------------------------------------------------------------
    // compute_priority — staleness normalization
    // -----------------------------------------------------------------------

    #[test]
    fn staleness_caps_at_90_days() {
        let p90 = compute_priority(90.0, None, 0, 1);
        let p120 = compute_priority(120.0, None, 0, 1);
        // Both should yield the same score since staleness is capped at 90
        assert!(
            (p90 - p120).abs() < f64::EPSILON,
            "staleness beyond 90 days should be clamped"
        );
        assert!((p90 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn staleness_45_days_yields_half_score_no_boost() {
        let p = compute_priority(45.0, None, 0, 1);
        // staleness_score = 0.5, boost = 1.0, round_robin = 1.0
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn staleness_zero_yields_zero_priority() {
        let p = compute_priority(0.0, None, 0, 1);
        assert_eq!(p, 0.0);
    }

    // -----------------------------------------------------------------------
    // compute_priority — velocity boosts
    // -----------------------------------------------------------------------

    #[test]
    fn velocity_boost_challenged_is_2x() {
        let base = compute_priority(45.0, None, 0, 1);
        let boosted = compute_priority(45.0, Some(&VerdictStatus::Challenged), 0, 1);
        assert!((boosted - base * 2.0).abs() < 1e-9);
    }

    #[test]
    fn velocity_boost_obsoleted_is_2x() {
        let base = compute_priority(45.0, None, 0, 1);
        let boosted = compute_priority(45.0, Some(&VerdictStatus::Obsoleted), 0, 1);
        assert!((boosted - base * 2.0).abs() < 1e-9);
    }

    #[test]
    fn velocity_boost_refined_is_1_5x() {
        let base = compute_priority(45.0, None, 0, 1);
        let boosted = compute_priority(45.0, Some(&VerdictStatus::Refined), 0, 1);
        assert!((boosted - base * 1.5).abs() < 1e-9);
    }

    #[test]
    fn velocity_boost_confirmed_is_1x() {
        let base = compute_priority(45.0, None, 0, 1);
        let boosted = compute_priority(45.0, Some(&VerdictStatus::Confirmed), 0, 1);
        assert!((boosted - base).abs() < 1e-9);
    }

    #[test]
    fn velocity_boost_skipped_is_1x() {
        let base = compute_priority(45.0, None, 0, 1);
        let boosted = compute_priority(45.0, Some(&VerdictStatus::Skipped), 0, 1);
        assert!((boosted - base).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // compute_priority — round-robin tiebreaker
    // -----------------------------------------------------------------------

    #[test]
    fn round_robin_earlier_position_has_higher_priority() {
        let p0 = compute_priority(45.0, None, 0, 10);
        let p9 = compute_priority(45.0, None, 9, 10);
        assert!(p0 > p9, "earlier cycle position must yield higher priority");
    }

    #[test]
    fn round_robin_max_penalty_is_1_percent() {
        // position 0 vs position == total: max penalty is (1/total)*0.01
        let p_first = compute_priority(90.0, None, 0, 100);
        let p_last = compute_priority(90.0, None, 99, 100);
        let diff = p_first - p_last;
        // staleness=1.0, boost=1.0; diff = 1.0*(1.0 - 0/100*0.01) - 1.0*(1.0 - 99/100*0.01)
        //                                = (99/100)*0.01 ≈ 0.0099
        assert!(diff < 0.01 + 1e-9, "round-robin penalty must be < 1%");
        assert!(diff > 0.0, "earlier position must still win");
    }

    #[test]
    fn round_robin_zero_total_decisions_does_not_panic() {
        let p = compute_priority(45.0, None, 0, 0);
        assert!(p.is_finite());
    }

    // -----------------------------------------------------------------------
    // serde round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn queued_decision_serde_round_trip() {
        let d = QueuedDecision {
            decision_id: "topic-3b".to_string(),
            priority: 0.75,
            staleness_days: 30.0,
            previous_verdict: Some(VerdictStatus::Challenged),
            estimated_cost_usd: 0.40,
        };
        let json = serde_json::to_string(&d).expect("serialize");
        let back: QueuedDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.decision_id, d.decision_id);
        assert_eq!(back.previous_verdict, d.previous_verdict);
        assert!((back.priority - d.priority).abs() < f64::EPSILON);
    }
}
