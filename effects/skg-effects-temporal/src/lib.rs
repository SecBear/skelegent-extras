#![deny(missing_docs)]
//! Temporal-backed effect executor for skelegent.
//!
//! Unlike [`skg_effects_local::LocalEffectExecutor`] which executes effects
//! in-process, this executor routes delegation and signalling through a
//! Temporal-backed [`Dispatcher`] / [`Signalable`], making those effects
//! durable and retriable.
//!
//! State effects (`WriteMemory`, `DeleteMemory`, `LinkMemory`, `UnlinkMemory`)
//! are **not** handled — they require a state-store activity worker that is
//! outside the scope of this crate. A warning is logged and execution
//! continues.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::dispatch::Dispatcher;
use layer0::effect::Effect;
use layer0::error::OrchError;
use layer0::id::DispatchId;
use layer0::operator::{OperatorInput, TriggerType};
use layer0::DispatchContext;
use serde_json::json;
use skg_effects_core::{EffectExecutor, Error, Signalable, UnknownEffectPolicy};
use std::sync::Arc;

/// Temporal-backed effect executor.
///
/// Routes `Delegate` and `Handoff` effects through a [`Dispatcher`] (typically
/// a [`TemporalOrch`](skg_orch_temporal::TemporalOrch)), which schedules them
/// as durable Temporal activities. `Signal` effects go through a [`Signalable`]
/// implementation.
///
/// State-mutation effects (`WriteMemory`, `DeleteMemory`, `LinkMemory`,
/// `UnlinkMemory`) log a warning and are skipped — they require a dedicated
/// state-store activity worker that is not yet implemented.
pub struct TemporalEffectExecutor {
    /// Dispatcher for delegation and handoff effects (typically a `TemporalOrch`).
    pub dispatcher: Arc<dyn Dispatcher>,
    /// Signaler for signal effects. When `None`, signal effects error.
    pub signaler: Option<Arc<dyn Signalable>>,
    /// Policy for effects this executor does not handle.
    pub unknown_policy: UnknownEffectPolicy,
}

impl TemporalEffectExecutor {
    /// Create a new executor with the default `IgnoreAndWarn` unknown-effect policy.
    pub fn new(
        dispatcher: Arc<dyn Dispatcher>,
        signaler: Option<Arc<dyn Signalable>>,
    ) -> Self {
        Self {
            dispatcher,
            signaler,
            unknown_policy: UnknownEffectPolicy::IgnoreAndWarn,
        }
    }

    /// Override the unknown/custom effect handling policy.
    pub fn with_unknown_policy(mut self, policy: UnknownEffectPolicy) -> Self {
        self.unknown_policy = policy;
        self
    }
}

#[async_trait]
impl EffectExecutor for TemporalEffectExecutor {
    async fn execute(&self, effects: &[Effect], ctx: &DispatchContext) -> Result<(), Error> {
        for effect in effects {
            match effect {
                // ── State effects: not supported without an activity worker ──
                Effect::WriteMemory { .. }
                | Effect::DeleteMemory { .. }
                | Effect::LinkMemory { .. }
                | Effect::UnlinkMemory { .. } => {
                    tracing::warn!(
                        "temporal executor: skipping state effect (requires state-store activity worker): {:?}",
                        std::mem::discriminant(effect),
                    );
                }

                // ── Dispatch effects: routed through Temporal dispatcher ──
                Effect::Delegate { operator, input } => {
                    let child_ctx =
                        ctx.child(DispatchId::new(operator.as_str()), operator.clone());
                    self.dispatcher
                        .dispatch(&child_ctx, (*input.clone()).clone())
                        .await?
                        .collect()
                        .await?;
                }

                Effect::Handoff { operator, state } => {
                    let mut input =
                        OperatorInput::new(Content::text(state.to_string()), TriggerType::Task);
                    input.metadata = json!({ "handoff": true });
                    let child_ctx =
                        ctx.child(DispatchId::new(operator.as_str()), operator.clone());
                    self.dispatcher
                        .dispatch(&child_ctx, input)
                        .await?
                        .collect()
                        .await?;
                }

                // ── Signals: routed through Signalable ──
                Effect::Signal { target, payload } => match &self.signaler {
                    Some(s) => s.signal(target, payload.clone()).await?,
                    None => {
                        return Err(Error::Dispatch(OrchError::DispatchFailed(
                            "signal requires a Signalable implementation".into(),
                        )));
                    }
                },

                // ── Non-executing effects: policy-based ──
                Effect::Log { .. } | Effect::Custom { .. } => match self.unknown_policy {
                    UnknownEffectPolicy::IgnoreAndWarn => {
                        tracing::warn!("ignoring unsupported effect: {:?}", effect);
                    }
                    UnknownEffectPolicy::Error => return Err(Error::UnknownEffect),
                },

                // Forward-compat: Effect is #[non_exhaustive].
                _ => match self.unknown_policy {
                    UnknownEffectPolicy::IgnoreAndWarn => {
                        tracing::warn!(
                            "ignoring forward-compatible effect variant: {:?}",
                            effect
                        );
                    }
                    UnknownEffectPolicy::Error => return Err(Error::UnknownEffect),
                },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::effect::{Effect, SignalPayload};
    use layer0::id::{DispatchId, OperatorId, WorkflowId};
    use layer0::operator::{OperatorInput, TriggerType};
    use layer0::test_utils::EchoOperator;
    use skg_effects_core::UnknownEffectPolicy;
    use skg_orch_temporal::{TemporalConfig, TemporalOrch};
    use std::sync::Arc;

    /// Build a TemporalOrch with a registered echo operator and return it as
    /// both `Arc<dyn Dispatcher>` and `Arc<dyn Signalable>`.
    fn make_orch() -> (Arc<TemporalOrch>, DispatchContext) {
        let mut orch = TemporalOrch::new(TemporalConfig::default());
        orch.register(OperatorId::new("echo"), Arc::new(EchoOperator));
        let ctx = DispatchContext::new(DispatchId::new("test"), OperatorId::new("test"));
        (Arc::new(orch), ctx)
    }

    #[tokio::test]
    async fn delegate_dispatches_through_temporal() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            Some(Arc::clone(&orch) as Arc<dyn Signalable>),
        );

        let effects = vec![Effect::Delegate {
            operator: OperatorId::new("echo"),
            input: Box::new(OperatorInput::new(Content::text("hello"), TriggerType::Task)),
        }];

        executor.execute(&effects, &ctx).await.unwrap();
    }

    #[tokio::test]
    async fn handoff_dispatches_through_temporal() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            Some(Arc::clone(&orch) as Arc<dyn Signalable>),
        );

        let effects = vec![Effect::Handoff {
            operator: OperatorId::new("echo"),
            state: serde_json::json!({ "key": "value" }),
        }];

        executor.execute(&effects, &ctx).await.unwrap();
    }

    #[tokio::test]
    async fn signal_goes_through_signalable() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            Some(Arc::clone(&orch) as Arc<dyn Signalable>),
        );

        let effects = vec![Effect::Signal {
            target: WorkflowId::new("wf-1"),
            payload: SignalPayload::new("test-signal", serde_json::json!({})),
        }];

        executor.execute(&effects, &ctx).await.unwrap();
    }

    #[tokio::test]
    async fn signal_without_signaler_errors() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            None,
        );

        let effects = vec![Effect::Signal {
            target: WorkflowId::new("wf-1"),
            payload: SignalPayload::new("test", serde_json::json!({})),
        }];

        let result = executor.execute(&effects, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn state_effects_are_skipped() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            None,
        );

        let effects = vec![Effect::WriteMemory {
            scope: layer0::effect::Scope::Global,
            key: "k".into(),
            value: serde_json::json!("v"),
            tier: None,
            lifetime: None,
            content_kind: None,
            salience: None,
            ttl: None,
        }];

        // Should succeed (warn-and-skip), not error.
        executor.execute(&effects, &ctx).await.unwrap();
    }

    #[tokio::test]
    async fn unknown_policy_error_rejects_custom() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            None,
        )
        .with_unknown_policy(UnknownEffectPolicy::Error);

        let effects = vec![Effect::Custom {
            effect_type: "something".into(),
            data: serde_json::json!({}),
        }];

        let result = executor.execute(&effects, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_policy_ignore_accepts_custom() {
        let (orch, ctx) = make_orch();
        let executor = TemporalEffectExecutor::new(
            Arc::clone(&orch) as Arc<dyn Dispatcher>,
            None,
        )
        .with_unknown_policy(UnknownEffectPolicy::IgnoreAndWarn);

        let effects = vec![Effect::Custom {
            effect_type: "something".into(),
            data: serde_json::json!({}),
        }];

        executor.execute(&effects, &ctx).await.unwrap();
    }
}
