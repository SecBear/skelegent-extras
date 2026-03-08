#![deny(missing_docs)]
//! Golden sweep v3 — built on neuron-context-engine primitives.
//!
//! Instead of implementing `Operator` and dispatching through an `Orchestrator`,
//! v3 uses `Context`, `ContextOp`, and `compile()→infer()` directly. The sweep
//! pipeline is a function that composes context engine primitives, not a framework
//! of interacting trait objects.
//!
//! ## Architecture comparison
//!
//! | Concern | v2 (Operator) | v3 (context-engine) |
//! |---------|---------------|---------------------|
//! | Compare | `CompareOperator` impl `Operator` | `compare_decision()` function |
//! | Synthesis | `SynthesisOperator` impl `Operator` | `synthesize()` function |
//! | Budget | External `BudgetTracker` checked in cycle | `BudgetGuard` rule on Context |
//! | Dispatch | `dispatch_typed` through `Orchestrator` | Direct `compile().infer()` |
//! | Context | Manual `Vec<Message>` + `InferRequest` | `Context` with fluent assembly |
//! | Cycle | `run_sweep_cycle()` with 9 steps | `sweep_cycle()` with same logic, less ceremony |

pub mod compare;
pub mod cycle;
pub mod dashboard;
pub mod decisions;
pub mod synthesis;
pub mod types;
