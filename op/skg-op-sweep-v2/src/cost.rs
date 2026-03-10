//! Cost tracking for sweep cycles.

use serde::{Deserialize, Serialize};

/// Tracks costs across the stages of a sweep cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SweepCostTracker {
    /// Cost of research queries (USD).
    pub research_cost: f64,
    /// Cost of LLM comparison calls (USD).
    pub comparison_cost: f64,
    /// Total tokens consumed.
    pub total_tokens: u64,
}

impl SweepCostTracker {
    /// Create a new zero-cost tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total USD cost across all stages.
    pub fn total_usd(&self) -> f64 {
        self.research_cost + self.comparison_cost
    }

    /// Accumulate research cost.
    pub fn add_research(&mut self, cost_usd: f64) {
        self.research_cost += cost_usd;
    }

    /// Accumulate comparison cost and tokens.
    pub fn add_comparison(&mut self, cost_usd: f64, tokens: u64) {
        self.comparison_cost += cost_usd;
        self.total_tokens += tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_accumulation() {
        let mut tracker = SweepCostTracker::new();
        tracker.add_research(0.10);
        tracker.add_comparison(0.05, 1000);
        tracker.add_comparison(0.03, 500);
        assert!((tracker.total_usd() - 0.18).abs() < f64::EPSILON);
        assert_eq!(tracker.total_tokens, 1500);
    }
}
