//! Generic budget enforcement with pluggable policies.
//!
//! [`BudgetTracker`] holds atomic budget state and delegates policy decisions
//! to a [`BudgetPolicy`] implementation. This separates the tracking mechanism
//! from domain-specific budget rules.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// A policy that decides whether a proposed spend should proceed.
///
/// Implement this for domain-specific budget rules (daily caps, per-operator
/// limits, degradation tiers, etc.).
pub trait BudgetPolicy: Send + Sync {
    /// Evaluate whether the proposed cost should proceed given the current state.
    fn check(&self, state: &BudgetState) -> BudgetDecision;
}

/// The result of a budget policy check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetDecision {
    /// Proceed normally.
    Allow,
    /// Proceed but with degraded behavior (caller decides what that means).
    Degraded {
        /// Why the budget is degraded.
        reason: String,
    },
    /// Do not proceed.
    Deny {
        /// Why the budget was denied.
        reason: String,
    },
}

/// Snapshot of current budget state, provided to [`BudgetPolicy::check`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetState {
    /// Total spend in USD (accumulated).
    pub total_usd: f64,
    /// Number of dispatches completed.
    pub dispatch_count: u64,
}

/// Thread-safe budget tracker with pluggable policy.
///
/// Tracks accumulated spend atomically. The policy is consulted before
/// each proposed spend via [`BudgetTracker::check`].
///
/// The spend amount is stored as integer micro-dollars (1 USD = 1_000_000)
/// for atomic operations without floating-point races.
pub struct BudgetTracker<P: BudgetPolicy> {
    /// Accumulated spend in micro-dollars.
    total_microdollars: AtomicU64,
    /// Number of dispatches recorded.
    dispatch_count: AtomicU64,
    /// The policy that decides allow/degrade/deny.
    policy: P,
}

impl<P: BudgetPolicy> std::fmt::Debug for BudgetTracker<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state();
        f.debug_struct("BudgetTracker")
            .field("total_usd", &state.total_usd)
            .field("dispatch_count", &state.dispatch_count)
            .finish()
    }
}

impl<P: BudgetPolicy> BudgetTracker<P> {
    /// Create a new tracker with the given policy.
    pub fn new(policy: P) -> Self {
        Self {
            total_microdollars: AtomicU64::new(0),
            dispatch_count: AtomicU64::new(0),
            policy,
        }
    }

    /// Record a completed spend.
    pub fn record(&self, cost_usd: f64) {
        let micros = (cost_usd * 1_000_000.0) as u64;
        self.total_microdollars.fetch_add(micros, Ordering::Relaxed);
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Check whether a proposed spend should proceed.
    pub fn check(&self) -> BudgetDecision {
        self.policy.check(&self.state())
    }

    /// Access the policy.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Current budget state snapshot.
    pub fn state(&self) -> BudgetState {
        let micros = self.total_microdollars.load(Ordering::Relaxed);
        BudgetState {
            total_usd: micros as f64 / 1_000_000.0,
            dispatch_count: self.dispatch_count.load(Ordering::Relaxed),
        }
    }

    /// Total USD spent so far.
    pub fn total_usd(&self) -> f64 {
        self.total_microdollars.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Reset the tracker to zero.
    pub fn reset(&self) {
        self.total_microdollars.store(0, Ordering::Relaxed);
        self.dispatch_count.store(0, Ordering::Relaxed);
    }
}

/// A simple cap-based budget policy.
///
/// Allows below `soft_cap`, degrades between `soft_cap` and `hard_cap`,
/// denies at or above `hard_cap`.
#[derive(Debug, Clone)]
pub struct CapPolicy {
    /// USD threshold for degraded mode.
    pub soft_cap: f64,
    /// USD threshold for hard stop.
    pub hard_cap: f64,
}

impl BudgetPolicy for CapPolicy {
    fn check(&self, state: &BudgetState) -> BudgetDecision {
        if state.total_usd >= self.hard_cap {
            BudgetDecision::Deny {
                reason: format!(
                    "hard cap ${:.2} reached (spent ${:.2})",
                    self.hard_cap, state.total_usd
                ),
            }
        } else if state.total_usd >= self.soft_cap {
            BudgetDecision::Degraded {
                reason: format!(
                    "soft cap ${:.2} reached (spent ${:.2})",
                    self.soft_cap, state.total_usd
                ),
            }
        } else {
            BudgetDecision::Allow
        }
    }
}

/// A budget policy that always allows (no enforcement).
#[derive(Debug, Clone)]
pub struct NoLimitPolicy;

impl BudgetPolicy for NoLimitPolicy {
    fn check(&self, _state: &BudgetState) -> BudgetDecision {
        BudgetDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_policy_normal_below_soft() {
        let policy = CapPolicy {
            soft_cap: 8.0,
            hard_cap: 10.0,
        };
        let tracker = BudgetTracker::new(policy);
        tracker.record(5.0);
        assert_eq!(tracker.check(), BudgetDecision::Allow);
    }

    #[test]
    fn cap_policy_degraded_at_soft() {
        let policy = CapPolicy {
            soft_cap: 8.0,
            hard_cap: 10.0,
        };
        let tracker = BudgetTracker::new(policy);
        tracker.record(8.0);
        assert!(matches!(tracker.check(), BudgetDecision::Degraded { .. }));
    }

    #[test]
    fn cap_policy_deny_at_hard() {
        let policy = CapPolicy {
            soft_cap: 8.0,
            hard_cap: 10.0,
        };
        let tracker = BudgetTracker::new(policy);
        tracker.record(10.0);
        assert!(matches!(tracker.check(), BudgetDecision::Deny { .. }));
    }

    #[test]
    fn record_accumulates() {
        let tracker = BudgetTracker::new(NoLimitPolicy);
        tracker.record(1.5);
        tracker.record(2.5);
        let total = tracker.total_usd();
        assert!((total - 4.0).abs() < 0.001);
        assert_eq!(tracker.state().dispatch_count, 2);
    }

    #[test]
    fn reset_zeroes_state() {
        let tracker = BudgetTracker::new(NoLimitPolicy);
        tracker.record(5.0);
        tracker.reset();
        assert!((tracker.total_usd()).abs() < 0.001);
        assert_eq!(tracker.state().dispatch_count, 0);
    }

    #[test]
    fn no_limit_always_allows() {
        let tracker = BudgetTracker::new(NoLimitPolicy);
        tracker.record(999_999.0);
        assert_eq!(tracker.check(), BudgetDecision::Allow);
    }

    #[test]
    fn cap_policy_between_soft_and_hard() {
        let policy = CapPolicy {
            soft_cap: 8.0,
            hard_cap: 10.0,
        };
        let tracker = BudgetTracker::new(policy);
        tracker.record(9.0);
        assert!(matches!(tracker.check(), BudgetDecision::Degraded { .. }));
    }

    #[test]
    fn thread_safety() {
        // BudgetTracker is Send + Sync — prove it compiles.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BudgetTracker<CapPolicy>>();
        assert_send_sync::<BudgetTracker<NoLimitPolicy>>();
    }
}
