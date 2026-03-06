//! Budget enforcement for the sweep orchestrator.
//!
//! The budget system tracks daily USD spend and enforces three degradation
//! tiers — [`DegradationLevel::Normal`], [`DegradationLevel::Degraded`], and
//! [`DegradationLevel::HardStop`] — to protect against runaway costs.
//!
//! # Relationship to `neuron-orch-kit`
//!
//! `neuron-orch-kit` provides a generic [`neuron_orch_kit::BudgetTracker`] with
//! atomic microdollar tracking and a [`neuron_orch_kit::BudgetDecision`] enum.
//! The sweep budget is NOT migrated to use it because the two systems serve
//! different concerns:
//!
//! - **Sweep budget**: daily-reset USD cap, per-decision cost history,
//!   degradation levels that affect scheduler behaviour across multiple decisions.
//! - **orch-kit BudgetTracker**: per-session or per-operator atomic counters with
//!   a simple Allow/Degraded/Deny policy.
//!
//! The sweep [`DegradationLevel`] enum (Normal/Degraded/HardStop) is structurally
//! different from `BudgetDecision` (Allow/Degraded{reason}/Deny{reason}) and
//! the `reason` string field has no equivalent here. Forcing alignment would
//! require changing the public API of this crate and every caller without
//! meaningful benefit. Leave both systems independent.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Configuration for the daily budget enforcement system.
///
/// All thresholds are expressed as a fraction of [`daily_cap_usd`].
/// The defaults are designed for a $10/day limit with conservative margins.
///
/// [`daily_cap_usd`]: BudgetConfig::daily_cap_usd
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum daily spend in USD before a hard stop is issued. Default: 10.0.
    pub daily_cap_usd: f64,
    /// Fraction of the daily cap at which degraded mode begins.
    ///
    /// In degraded mode the orchestrator downgrades processors to `Base` and
    /// skips `Confirmed` decisions that are not sufficiently stale.
    /// Default: 0.80 (80% of the daily cap).
    pub degradation_threshold: f64,
    /// Fraction of the daily cap at which all sweeps are halted. Default: 0.95.
    pub hard_stop_threshold: f64,
    /// Maximum number of sweeps that may run concurrently. Default: 3.
    pub max_parallel: usize,
    /// Minimum staleness (days) required for a `Confirmed` decision to be swept
    /// while in [`DegradationLevel::Degraded`] mode. Default: 14 days.
    pub degraded_min_staleness_days: f64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            daily_cap_usd: 10.0,
            degradation_threshold: 0.80,
            hard_stop_threshold: 0.95,
            max_parallel: 3,
            degraded_min_staleness_days: 14.0,
        }
    }
}

/// Live budget state for a single UTC day.
///
/// This struct is persisted to the state database between cycles and reset
/// automatically at midnight UTC. Use [`BudgetState::should_reset`] to detect
/// when a reset is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetState {
    /// Running USD total accumulated today.
    pub daily_total_usd: f64,
    /// UTC date this counter is for, formatted as `"YYYY-MM-DD"`.
    pub date: String,
    /// Per-decision cost history for cost estimation.
    ///
    /// Keys are decision IDs; values are the last known sweep cost in USD.
    /// Used by [`BudgetState::estimate_cost`] to produce better pre-flight
    /// estimates than the fallback default.
    pub last_costs: HashMap<String, f64>,
}

impl BudgetState {
    /// Determine the current degradation level given the budget configuration.
    ///
    /// Returns [`DegradationLevel::HardStop`] when the spend ratio reaches
    /// [`BudgetConfig::hard_stop_threshold`], [`DegradationLevel::Degraded`]
    /// when it reaches [`BudgetConfig::degradation_threshold`], and
    /// [`DegradationLevel::Normal`] otherwise.
    pub fn degradation_mode(&self, config: &BudgetConfig) -> DegradationLevel {
        let ratio = self.daily_total_usd / config.daily_cap_usd;
        if ratio >= config.hard_stop_threshold {
            DegradationLevel::HardStop
        } else if ratio >= config.degradation_threshold {
            DegradationLevel::Degraded
        } else {
            DegradationLevel::Normal
        }
    }

    /// Estimate the cost of sweeping `decision_id` based on historical data.
    ///
    /// Returns the last recorded cost for this decision if available, or
    /// a conservative default of `$0.30` for decisions with no cost history.
    pub fn estimate_cost(&self, decision_id: &str) -> f64 {
        self.last_costs.get(decision_id).copied().unwrap_or(0.30)
    }

    /// Returns `true` if the budget counter should be reset.
    ///
    /// A reset is needed when [`BudgetState::date`] does not match today's
    /// UTC date. Callers should zero out [`BudgetState::daily_total_usd`] and
    /// update [`BudgetState::date`] before proceeding.
    pub fn should_reset(&self) -> bool {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        self.date != today
    }
}

/// Operational degradation level derived from current budget spend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationLevel {
    /// Normal operation. Use the operator-selected processor tier.
    Normal,
    /// Budget pressure — spend has crossed [`BudgetConfig::degradation_threshold`].
    ///
    /// In this mode the orchestrator downgrades processors to `Base` and skips
    /// `Confirmed` decisions whose staleness is below
    /// [`BudgetConfig::degraded_min_staleness_days`].
    Degraded,
    /// Near-cap halt — spend has crossed [`BudgetConfig::hard_stop_threshold`].
    ///
    /// All further sweep dispatch is suspended for the remainder of the UTC day.
    HardStop,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(daily_total_usd: f64) -> BudgetState {
        BudgetState {
            daily_total_usd,
            date: "2026-03-04".to_string(),
            last_costs: HashMap::new(),
        }
    }

    fn cfg() -> BudgetConfig {
        BudgetConfig::default() // cap=10, degraded=0.80, hard_stop=0.95
    }

    // -----------------------------------------------------------------------
    // degradation_mode — all three transitions
    // -----------------------------------------------------------------------

    #[test]
    fn normal_when_below_degradation_threshold() {
        // 79% spend → Normal
        assert_eq!(
            state(7.9).degradation_mode(&cfg()),
            DegradationLevel::Normal
        );
    }

    #[test]
    fn normal_at_zero_spend() {
        assert_eq!(
            state(0.0).degradation_mode(&cfg()),
            DegradationLevel::Normal
        );
    }

    #[test]
    fn degraded_at_exactly_80_percent() {
        // 80% exactly → Degraded (ratio >= threshold)
        assert_eq!(
            state(8.0).degradation_mode(&cfg()),
            DegradationLevel::Degraded
        );
    }

    #[test]
    fn degraded_between_thresholds() {
        // 87% → Degraded
        assert_eq!(
            state(8.7).degradation_mode(&cfg()),
            DegradationLevel::Degraded
        );
    }

    #[test]
    fn hard_stop_at_exactly_95_percent() {
        // 95% exactly → HardStop (ratio >= hard_stop_threshold)
        assert_eq!(
            state(9.5).degradation_mode(&cfg()),
            DegradationLevel::HardStop
        );
    }

    #[test]
    fn hard_stop_above_cap() {
        // Overspend → HardStop
        assert_eq!(
            state(12.0).degradation_mode(&cfg()),
            DegradationLevel::HardStop
        );
    }

    #[test]
    fn degraded_just_below_hard_stop() {
        // 94.9% → Degraded, not HardStop
        assert_eq!(
            state(9.49).degradation_mode(&cfg()),
            DegradationLevel::Degraded
        );
    }

    #[test]
    fn custom_config_thresholds() {
        let cfg = BudgetConfig {
            daily_cap_usd: 100.0,
            degradation_threshold: 0.50,
            hard_stop_threshold: 0.75,
            max_parallel: 2,
            degraded_min_staleness_days: 7.0,
        };
        assert_eq!(state(49.0).degradation_mode(&cfg), DegradationLevel::Normal);
        assert_eq!(
            state(50.0).degradation_mode(&cfg),
            DegradationLevel::Degraded
        );
        assert_eq!(
            state(75.0).degradation_mode(&cfg),
            DegradationLevel::HardStop
        );
    }

    // -----------------------------------------------------------------------
    // estimate_cost
    // -----------------------------------------------------------------------

    #[test]
    fn estimate_cost_returns_default_when_no_history() {
        let s = state(0.0);
        assert!((s.estimate_cost("topic-3b") - 0.30).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_cost_returns_known_value_from_history() {
        let mut s = state(0.0);
        s.last_costs.insert("topic-1a".to_string(), 0.55);
        assert!((s.estimate_cost("topic-1a") - 0.55).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_cost_falls_back_for_unknown_id() {
        let mut s = state(0.0);
        s.last_costs.insert("topic-1a".to_string(), 0.55);
        // topic-3b was never recorded
        assert!((s.estimate_cost("topic-3b") - 0.30).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // should_reset — date comparison
    // -----------------------------------------------------------------------

    #[test]
    fn should_reset_true_when_date_is_past() {
        // Any date in the past should trigger a reset
        let s = BudgetState {
            daily_total_usd: 5.0,
            date: "2000-01-01".to_string(),
            last_costs: HashMap::new(),
        };
        assert!(s.should_reset(), "past date must trigger reset");
    }

    #[test]
    fn should_reset_false_when_date_matches_today() {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let s = BudgetState {
            daily_total_usd: 5.0,
            date: today,
            last_costs: HashMap::new(),
        };
        assert!(!s.should_reset(), "current date must not trigger reset");
    }

    // -----------------------------------------------------------------------
    // serde round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn budget_config_serde_round_trip() {
        let cfg = BudgetConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: BudgetConfig = serde_json::from_str(&json).expect("deserialize");
        assert!((back.daily_cap_usd - cfg.daily_cap_usd).abs() < f64::EPSILON);
        assert!((back.degradation_threshold - cfg.degradation_threshold).abs() < f64::EPSILON);
        assert_eq!(back.max_parallel, cfg.max_parallel);
    }

    #[test]
    fn budget_state_serde_round_trip() {
        let mut s = state(4.5);
        s.last_costs.insert("topic-1a".to_string(), 0.25);
        let json = serde_json::to_string(&s).expect("serialize");
        let back: BudgetState = serde_json::from_str(&json).expect("deserialize");
        assert!((back.daily_total_usd - s.daily_total_usd).abs() < f64::EPSILON);
        assert_eq!(back.date, s.date);
        assert!((back.last_costs["topic-1a"] - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn degradation_level_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&DegradationLevel::HardStop).unwrap(),
            "\"hard_stop\""
        );
        assert_eq!(
            serde_json::to_string(&DegradationLevel::Degraded).unwrap(),
            "\"degraded\""
        );
        assert_eq!(
            serde_json::to_string(&DegradationLevel::Normal).unwrap(),
            "\"normal\""
        );
    }
}
