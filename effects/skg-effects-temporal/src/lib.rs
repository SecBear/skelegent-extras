#![deny(missing_docs)]
//! Temporal-backed effect handler for skelegent.
//!
//! Unlike [`skg_effects_local::LocalEffectHandler`] which executes effects
//! in-process, this handler routes delegation and signalling through a
//! Temporal-backed [`Signalable`], making signal effects durable and retriable.
//! Delegate and Handoff effects are returned as [`EffectOutcome`] variants for
//! the caller to dispatch.
//!
//! State effects (`WriteMemory`, `DeleteMemory`, `LinkMemory`, `UnlinkMemory`)
//! are **not** handled — they require a state-store activity worker that is
//! outside the scope of this crate. A warning is logged and
//! [`EffectOutcome::Skipped`] is returned.

use async_trait::async_trait;
use layer0::content::Content;
use layer0::effect::Effect;
use layer0::error::OrchError;
use layer0::operator::{OperatorInput, TriggerType};
use layer0::DispatchContext;
use serde_json::Value;
use skg_effects_core::{EffectHandler, EffectOutcome, Error, Signalable, UnknownEffectPolicy};
use std::sync::Arc;

/// Temporal-backed effect handler.
///
/// `Delegate` and `Handoff` effects are returned as [`EffectOutcome`] variants
/// so the caller (e.g. an orchestrated runner) can schedule them as durable
/// Temporal activities. `Signal` effects go through a [`Signalable`]
/// implementation.
///
/// State-mutation effects (`WriteMemory`, `DeleteMemory`, `LinkMemory`,
/// `UnlinkMemory`) log a warning and return [`EffectOutcome::Skipped`] — they
/// require a dedicated state-store activity worker that is not yet implemented.
pub struct TemporalEffectHandler {
    /// Signaler for signal effects. When `None`, signal effects error.
    pub signaler: Option<Arc<dyn Signalable>>,
    /// Policy for effects this handler does not handle.
    pub unknown_policy: UnknownEffectPolicy,
}

impl TemporalEffectHandler {
    /// Create a new handler with the default `IgnoreAndWarn` unknown-effect policy.
    pub fn new(signaler: Option<Arc<dyn Signalable>>) -> Self {
        Self {
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
impl EffectHandler for TemporalEffectHandler {
    async fn handle(
        &self,
        effect: &Effect,
        _ctx: &DispatchContext,
    ) -> Result<EffectOutcome, Error> {
        match effect {
            // ── State effects: not supported without an activity worker ──
            Effect::WriteMemory { .. }
            | Effect::DeleteMemory { .. }
            | Effect::LinkMemory { .. }
            | Effect::UnlinkMemory { .. } => {
                tracing::warn!(
                    "temporal handler: skipping state effect (requires state-store activity worker): {:?}",
                    std::mem::discriminant(effect),
                );
                Ok(EffectOutcome::Skipped)
            }

            // ── Dispatch effects: returned as outcomes for caller to schedule ──
            Effect::Delegate { operator, input } => Ok(EffectOutcome::Delegate {
                operator: operator.clone(),
                input: (*input.clone()).clone(),
            }),

            Effect::Handoff { operator, state } => {
                let mut input =
                    OperatorInput::new(Content::text(state.to_string()), TriggerType::Task);
                input.metadata = Value::Null;
                Ok(EffectOutcome::Handoff {
                    operator: operator.clone(),
                    input,
                })
            }

            // ── Signals: routed through Signalable ──
            Effect::Signal { target, payload } => match &self.signaler {
                Some(s) => {
                    s.signal(target, payload.clone()).await?;
                    Ok(EffectOutcome::Applied)
                }
                None => Err(Error::Dispatch(OrchError::DispatchFailed(
                    "signal requires a Signalable implementation".into(),
                ))),
            },

            // ── Non-executing effects: policy-based ──
            Effect::Log { .. } | Effect::Custom { .. } => match self.unknown_policy {
                UnknownEffectPolicy::IgnoreAndWarn => {
                    tracing::warn!("ignoring unsupported effect: {:?}", effect);
                    Ok(EffectOutcome::Skipped)
                }
                UnknownEffectPolicy::Error => Err(Error::UnknownEffect),
            },

            // Forward-compat: Effect is #[non_exhaustive].
            _ => match self.unknown_policy {
                UnknownEffectPolicy::IgnoreAndWarn => {
                    tracing::warn!(
                        "ignoring forward-compatible effect variant: {:?}",
                        effect
                    );
                    Ok(EffectOutcome::Skipped)
                }
                UnknownEffectPolicy::Error => Err(Error::UnknownEffect),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::effect::{Effect, SignalPayload};
    use layer0::id::{DispatchId, OperatorId, WorkflowId};
    use layer0::operator::{OperatorInput, TriggerType};
    use skg_effects_core::UnknownEffectPolicy;
    use skg_orch_temporal::{TemporalConfig, TemporalOrch};
    use std::sync::Arc;

    /// Build a TemporalOrch (for its Signalable impl) and a DispatchContext.
    fn make_orch() -> (Arc<TemporalOrch>, DispatchContext) {
        let orch = TemporalOrch::new(TemporalConfig::default());
        let ctx = DispatchContext::new(DispatchId::new("test"), OperatorId::new("test"));
        (Arc::new(orch), ctx)
    }

    #[tokio::test]
    async fn delegate_returns_delegate_outcome() {
        let (orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(Some(Arc::clone(&orch) as Arc<dyn Signalable>));

        let effect = Effect::Delegate {
            operator: OperatorId::new("echo"),
            input: Box::new(OperatorInput::new(Content::text("hello"), TriggerType::Task)),
        };

        let outcome = handler.handle(&effect, &ctx).await.unwrap();
        assert!(matches!(outcome, EffectOutcome::Delegate { .. }));
    }

    #[tokio::test]
    async fn handoff_returns_handoff_outcome() {
        let (orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(Some(Arc::clone(&orch) as Arc<dyn Signalable>));

        let effect = Effect::Handoff {
            operator: OperatorId::new("echo"),
            state: serde_json::json!({ "key": "value" }),
        };

        let outcome = handler.handle(&effect, &ctx).await.unwrap();
        assert!(matches!(outcome, EffectOutcome::Handoff { .. }));
    }

    #[tokio::test]
    async fn signal_goes_through_signalable() {
        let (orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(Some(Arc::clone(&orch) as Arc<dyn Signalable>));

        let effect = Effect::Signal {
            target: WorkflowId::new("wf-1"),
            payload: SignalPayload::new("test-signal", serde_json::json!({})),
        };

        let outcome = handler.handle(&effect, &ctx).await.unwrap();
        assert!(matches!(outcome, EffectOutcome::Applied));
    }

    #[tokio::test]
    async fn signal_without_signaler_errors() {
        let (_orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(None);

        let effect = Effect::Signal {
            target: WorkflowId::new("wf-1"),
            payload: SignalPayload::new("test", serde_json::json!({})),
        };

        let result = handler.handle(&effect, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn state_effects_are_skipped() {
        let (_orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(None);

        let effect = Effect::WriteMemory {
            scope: layer0::effect::Scope::Global,
            key: "k".into(),
            value: serde_json::json!("v"),
            tier: None,
            lifetime: None,
            content_kind: None,
            salience: None,
            ttl: None,
        };

        let outcome = handler.handle(&effect, &ctx).await.unwrap();
        assert!(matches!(outcome, EffectOutcome::Skipped));
    }

    #[tokio::test]
    async fn unknown_policy_error_rejects_custom() {
        let (_orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(None)
            .with_unknown_policy(UnknownEffectPolicy::Error);

        let effect = Effect::Custom {
            effect_type: "something".into(),
            data: serde_json::json!({}),
        };

        let result = handler.handle(&effect, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_policy_ignore_accepts_custom() {
        let (_orch, ctx) = make_orch();
        let handler = TemporalEffectHandler::new(None)
            .with_unknown_policy(UnknownEffectPolicy::IgnoreAndWarn);

        let effect = Effect::Custom {
            effect_type: "something".into(),
            data: serde_json::json!({}),
        };

        let outcome = handler.handle(&effect, &ctx).await.unwrap();
        assert!(matches!(outcome, EffectOutcome::Skipped));
    }
}
