//! Cost tracking for a single sweep run.

use serde::{Deserialize, Serialize};

/// Accumulated cost for one sweep operator run.
///
/// Each field tracks spending for a distinct phase of the pipeline.
/// The operator updates this struct as each phase completes and attaches
/// the totals to the emitted [`crate::types::SweepVerdict`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SweepCostTracker {
    /// Cost of the plan-generation LLM call (adaptive plan-first path only).
    ///
    /// Zero for the direct-query path where no planning LLM call is made.
    pub plan_cost_usd: f64,
    /// Cost of all research API calls (sum across all queries and retries).
    pub research_cost_usd: f64,
    /// Cost of the comparison LLM call that produces the verdict.
    pub comparison_cost_usd: f64,
    /// Total tokens consumed across all LLM calls in this run.
    pub total_tokens: usize,
}

impl SweepCostTracker {
    /// Return the total spend for this run in USD.
    ///
    /// This is the sum of plan, research, and comparison costs.
    /// Token counts are not included in the monetary total.
    pub fn total(&self) -> f64 {
        self.plan_cost_usd + self.research_cost_usd + self.comparison_cost_usd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_is_sum_of_all_three_fields() {
        let tracker = SweepCostTracker {
            plan_cost_usd: 0.05,
            research_cost_usd: 0.20,
            comparison_cost_usd: 0.10,
            total_tokens: 5000,
        };
        let total = tracker.total();
        assert!((total - 0.35).abs() < 1e-10, "expected 0.35, got {total}");
    }

    #[test]
    fn total_with_zero_plan_cost() {
        let tracker = SweepCostTracker {
            plan_cost_usd: 0.0,
            research_cost_usd: 0.15,
            comparison_cost_usd: 0.08,
            total_tokens: 2000,
        };
        let total = tracker.total();
        assert!((total - 0.23).abs() < 1e-10, "expected 0.23, got {total}");
    }

    #[test]
    fn total_all_zeros() {
        let tracker = SweepCostTracker::default();
        assert!(tracker.total().abs() < f64::EPSILON);
    }

    #[test]
    fn cost_tracker_serde_round_trip() {
        let tracker = SweepCostTracker {
            plan_cost_usd: 0.01,
            research_cost_usd: 0.30,
            comparison_cost_usd: 0.05,
            total_tokens: 8192,
        };
        let json = serde_json::to_string(&tracker).expect("serialize");
        let back: SweepCostTracker = serde_json::from_str(&json).expect("deserialize");
        assert!((back.plan_cost_usd - tracker.plan_cost_usd).abs() < f64::EPSILON);
        assert!((back.research_cost_usd - tracker.research_cost_usd).abs() < f64::EPSILON);
        assert!((back.comparison_cost_usd - tracker.comparison_cost_usd).abs() < f64::EPSILON);
        assert_eq!(back.total_tokens, tracker.total_tokens);
    }
}
