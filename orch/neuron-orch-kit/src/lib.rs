//! # neuron-orch-kit — Composition utilities for neuron orchestration
//!
//! This crate provides the building blocks for composing operators into
//! application workflows:
//!
//! - [`ScopedState`] / [`ScopedStateView`] — scoped read/write access to a partition of a [`StateStore`]
//! - [`dispatch_typed`] — type-safe operator dispatch with automatic serde
//! - [`BudgetTracker`] / [`BudgetPolicy`] — generic budget enforcement with pluggable policies
//! - [`CompositionTrace`] — span-based tracing with cycle detection for nested dispatch
//! - [`EffectMiddleware`] / [`MiddlewareOrchestrator`] — intercept and transform effects at the dispatch boundary
//! - [`RoutingOrchestrator`] — route dispatches to backend orchestrators by agent name
//!
//! [`StateStore`]: layer0::StateStore

#![deny(missing_docs)]

pub mod budget;
pub mod dispatch;
pub mod middleware;
pub mod scoped_state;
pub mod trace;
pub mod routing;

pub use budget::{BudgetDecision, BudgetPolicy, BudgetTracker};
pub use dispatch::{dispatch_typed, DispatchError};
pub use middleware::{EffectMiddleware, MiddlewareOrchestrator};
pub use routing::RoutingOrchestrator;
pub use scoped_state::{ScopedState, ScopedStateView};
pub use trace::CompositionTrace;
