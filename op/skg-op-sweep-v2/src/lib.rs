#![deny(missing_docs)]
//! Sweep operator v2 — architectural decision auditing with unified Message type.
//!
//! This is the v2 rewrite using skelegent's unified architecture:
//! - [`Message`](layer0::Message) replaces `AnnotatedMessage`/`ProviderMessage` as the
//!   universal context unit
//! - Compaction via closures instead of `ContextStrategy` trait
//! - Direct `Content::text()` construction instead of `ContentPart` intermediaries
//!
//! # Modules
//!
//! - [`types`] — Verdict, evidence, and processor tier types
//! - [`compare`] — Per-decision comparison operator (implements [`Operator`](layer0::Operator))
//! - [`provider`] — Research source abstraction
//! - [`cost`] — Cost tracking for sweep cycles
//! - [`queries`] — Query registry and rotation for decision angles
//! - [`synthesis`] — Two-pass synthesis for cross-decision pattern detection
pub mod compare;
pub mod cost;
pub mod cycle;
pub mod provider;
pub mod queries;
pub mod research_operator;
pub mod types;
pub mod synthesis;
pub mod synthesis_operator;

pub use compare::{CompareConfig, CompareOperator};
pub use cost::SweepCostTracker;
pub use provider::{CompareInput, ResearchInput, ResearchMode, ResearchResult, ResearchSource, SweepError};
pub use queries::{build_query, DecisionQuery, next_query, queries_for};
pub use research_operator::ResearchOperator;
pub use types::*;
