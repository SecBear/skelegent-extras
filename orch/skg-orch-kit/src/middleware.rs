//! Effect middleware — intercept and transform effects at the dispatch boundary.
//!
//! [`MiddlewareOrchestrator`] wraps an inner orchestrator and applies a chain
//! of [`EffectMiddleware`] transforms to effects returned by each dispatch.

use async_trait::async_trait;
use layer0::{
    effect::SignalPayload, OperatorId, Effect, OperatorInput, OperatorOutput, OrchError,
    Orchestrator, QueryPayload, WorkflowId, dispatch::Dispatcher,
};
use std::sync::Arc;

/// A middleware that can intercept and transform effects.
///
/// Applied to every effect returned by an operator dispatch. Middleware
/// runs left-to-right (first registered = first applied).
pub trait EffectMiddleware: Send + Sync {
    /// Transform a single effect. Return the effect unchanged to pass through,
    /// modify it, or return a different effect entirely.
    fn transform(&self, effect: Effect) -> Effect;
}

/// An orchestrator wrapper that applies [`EffectMiddleware`] to dispatch results.
///
/// Wraps an inner orchestrator. After each `dispatch` or `dispatch_many`, the
/// returned effects are transformed by the middleware chain (left-to-right).
/// Signal and query operations are passed through unchanged.
pub struct MiddlewareOrchestrator {
    inner: Arc<dyn Orchestrator>,
    middlewares: Vec<Arc<dyn EffectMiddleware>>,
}

impl std::fmt::Debug for MiddlewareOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareOrchestrator")
            .field("middleware_count", &self.middlewares.len())
            .finish_non_exhaustive()
    }
}

impl MiddlewareOrchestrator {
    /// Create a new middleware orchestrator.
    pub fn new(inner: Arc<dyn Orchestrator>, middlewares: Vec<Arc<dyn EffectMiddleware>>) -> Self {
        Self { inner, middlewares }
    }

    fn apply_middlewares(&self, effects: Vec<Effect>) -> Vec<Effect> {
        effects
            .into_iter()
            .map(|mut effect| {
                for mw in &self.middlewares {
                    effect = mw.transform(effect);
                }
                effect
            })
            .collect()
    }
}

#[async_trait]
impl Dispatcher for MiddlewareOrchestrator {
    async fn dispatch(
        &self,
        operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<OperatorOutput, OrchError> {
        let mut output = self.inner.dispatch(operator, input).await?;
        output.effects = self.apply_middlewares(output.effects);
        Ok(output)
    }
}

#[async_trait]
impl Orchestrator for MiddlewareOrchestrator {
    async fn dispatch_many(
        &self,
        tasks: Vec<(OperatorId, OperatorInput)>,
    ) -> Vec<Result<OperatorOutput, OrchError>> {
        let results = self.inner.dispatch_many(tasks).await;
        results
            .into_iter()
            .map(|r| {
                r.map(|mut output| {
                    output.effects = self.apply_middlewares(output.effects);
                    output
                })
            })
            .collect()
    }

    async fn signal(
        &self,
        target: &WorkflowId,
        signal: SignalPayload,
    ) -> Result<(), OrchError> {
        self.inner.signal(target, signal).await
    }

    async fn query(
        &self,
        target: &WorkflowId,
        query: QueryPayload,
    ) -> Result<serde_json::Value, OrchError> {
        self.inner.query(target, query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{operator::TriggerType, Content, ExitReason, Scope, dispatch::Dispatcher};

    /// Orchestrator that returns a fixed set of effects.
    struct EffectOrchestrator {
        effects: Vec<Effect>,
    }

    #[async_trait]
    impl Dispatcher for EffectOrchestrator {
        async fn dispatch(
            &self,
            _operator: &OperatorId,
            _input: OperatorInput,
        ) -> Result<OperatorOutput, OrchError> {
            Ok({
                            let mut out = OperatorOutput::new(Content::text("ok"), ExitReason::Complete);
                            out.effects = self.effects.clone();
                            out
                        })
        }
    }

    #[async_trait]
    impl Orchestrator for EffectOrchestrator {
        async fn dispatch_many(
            &self,
            tasks: Vec<(OperatorId, OperatorInput)>,
        ) -> Vec<Result<OperatorOutput, OrchError>> {
            let mut results = Vec::new();
            for (operator, input) in tasks {
                results.push(self.dispatch(&operator, input).await);
            }
            results
        }

        async fn signal(
            &self,
            _target: &WorkflowId,
            _signal: SignalPayload,
        ) -> Result<(), OrchError> {
            Ok(())
        }

        async fn query(
            &self,
            _target: &WorkflowId,
            _query: QueryPayload,
        ) -> Result<serde_json::Value, OrchError> {
            Ok(serde_json::Value::Null)
        }
    }

    /// Identity middleware — passes effects through unchanged.
    struct IdentityMiddleware;

    impl EffectMiddleware for IdentityMiddleware {
        fn transform(&self, effect: Effect) -> Effect {
            effect
        }
    }

    /// Middleware that redacts values in WriteMemory effects.
    struct RedactMiddleware;

    impl EffectMiddleware for RedactMiddleware {
        fn transform(&self, effect: Effect) -> Effect {
            match effect {
                Effect::WriteMemory {
                    scope,
                    key,
                    tier,
                    lifetime,
                    content_kind,
                    salience,
                    ttl,
                    ..
                } => Effect::WriteMemory {
                    scope,
                    key,
                    value: serde_json::json!("[REDACTED]"),
                    tier,
                    lifetime,
                    content_kind,
                    salience,
                    ttl,
                },
                other => other,
            }
        }
    }

    /// Middleware that uppercases WriteMemory keys.
    struct UppercaseKeyMiddleware;

    impl EffectMiddleware for UppercaseKeyMiddleware {
        fn transform(&self, effect: Effect) -> Effect {
            match effect {
                Effect::WriteMemory {
                    scope,
                    key,
                    value,
                    tier,
                    lifetime,
                    content_kind,
                    salience,
                    ttl,
                } => Effect::WriteMemory {
                    scope,
                    key: key.to_uppercase(),
                    value,
                    tier,
                    lifetime,
                    content_kind,
                    salience,
                    ttl,
                },
                other => other,
            }
        }
    }

    fn make_input() -> OperatorInput {
        OperatorInput::new(Content::text("test"), TriggerType::Task)
    }

    fn write_effect(key: &str, value: &str) -> Effect {
        Effect::WriteMemory {
            scope: Scope::Custom("test".into()),
            key: key.into(),
            value: serde_json::json!(value),
            tier: None,
            lifetime: None,
            content_kind: None,
            salience: None,
            ttl: None,
        }
    }

    #[tokio::test]
    async fn identity_middleware_passes_through() {
        let inner = Arc::new(EffectOrchestrator {
            effects: vec![write_effect("key1", "value1")],
        });
        let mw = MiddlewareOrchestrator::new(inner, vec![Arc::new(IdentityMiddleware)]);
        let operator = OperatorId::new("test");

        let output = mw.dispatch(&operator, make_input()).await.unwrap();
        assert_eq!(output.effects.len(), 1);

        if let Effect::WriteMemory { key, value, .. } = &output.effects[0] {
            assert_eq!(key, "key1");
            assert_eq!(value, &serde_json::json!("value1"));
        } else {
            panic!("expected WriteMemory");
        }
    }

    #[tokio::test]
    async fn redact_middleware_replaces_values() {
        let inner = Arc::new(EffectOrchestrator {
            effects: vec![write_effect("secret", "password123")],
        });
        let mw = MiddlewareOrchestrator::new(inner, vec![Arc::new(RedactMiddleware)]);
        let operator = OperatorId::new("test");

        let output = mw.dispatch(&operator, make_input()).await.unwrap();

        if let Effect::WriteMemory { key, value, .. } = &output.effects[0] {
            assert_eq!(key, "secret");
            assert_eq!(value, &serde_json::json!("[REDACTED]"));
        } else {
            panic!("expected WriteMemory");
        }
    }

    #[tokio::test]
    async fn middleware_chain_applies_left_to_right() {
        // UppercaseKey first, then Redact.
        // Input: key="secret", value="password"
        // After UppercaseKey: key="SECRET", value="password"
        // After Redact: key="SECRET", value="[REDACTED]"
        let inner = Arc::new(EffectOrchestrator {
            effects: vec![write_effect("secret", "password")],
        });
        let mw = MiddlewareOrchestrator::new(
            inner,
            vec![
                Arc::new(UppercaseKeyMiddleware),
                Arc::new(RedactMiddleware),
            ],
        );
        let operator = OperatorId::new("test");

        let output = mw.dispatch(&operator, make_input()).await.unwrap();

        if let Effect::WriteMemory { key, value, .. } = &output.effects[0] {
            assert_eq!(key, "SECRET");
            assert_eq!(value, &serde_json::json!("[REDACTED]"));
        } else {
            panic!("expected WriteMemory");
        }
    }
}
