//! Effect middleware — intercept and transform effects at the dispatch boundary.
//!
//! [`MiddlewareDispatcher`] wraps an inner dispatcher and applies a chain
//! of [`EffectMiddleware`] transforms to effects returned by each dispatch.

use async_trait::async_trait;
use layer0::{
    Effect, OperatorId, OperatorInput, OrchError,
    dispatch::{Dispatcher, DispatchEvent, DispatchHandle},
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

/// A dispatcher wrapper that applies [`EffectMiddleware`] to dispatch results.
///
/// Wraps an inner dispatcher. After each `dispatch`, the returned effects are
/// transformed by the middleware chain (left-to-right).
pub struct MiddlewareDispatcher {
    inner: Arc<dyn Dispatcher>,
    middlewares: Vec<Arc<dyn EffectMiddleware>>,
}

impl std::fmt::Debug for MiddlewareDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareDispatcher")
            .field("middleware_count", &self.middlewares.len())
            .finish_non_exhaustive()
    }
}

impl MiddlewareDispatcher {
    /// Create a new middleware dispatcher.
    pub fn new(inner: Arc<dyn Dispatcher>, middlewares: Vec<Arc<dyn EffectMiddleware>>) -> Self {
        Self { inner, middlewares }
    }
}

#[async_trait]
impl Dispatcher for MiddlewareDispatcher {
    async fn dispatch(
        &self,
        operator: &OperatorId,
        input: OperatorInput,
    ) -> Result<DispatchHandle, OrchError> {
        let mut inner_handle = self.inner.dispatch(operator, input).await?;
        let middlewares: Vec<Arc<dyn EffectMiddleware>> = self.middlewares.clone();
        let (handle, sender) = DispatchHandle::channel(inner_handle.id.clone());
        tokio::spawn(async move {
            while let Some(event) = inner_handle.recv().await {
                match event {
                    DispatchEvent::Completed { mut output } => {
                        output.effects = output
                            .effects
                            .into_iter()
                            .map(|mut effect| {
                                for mw in &middlewares {
                                    effect = mw.transform(effect);
                                }
                                effect
                            })
                            .collect();
                        let _ = sender.send(DispatchEvent::Completed { output }).await;
                    }
                    other => { let _ = sender.send(other).await; }
                }
            }
        });
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{operator::TriggerType, Content, ExitReason, OperatorOutput, Scope, dispatch::{DispatchEvent, DispatchHandle}};
    use layer0::id::DispatchId;

    /// Dispatcher that returns a fixed set of effects.
    struct EffectDispatcher {
        effects: Vec<Effect>,
    }

    #[async_trait]
    impl Dispatcher for EffectDispatcher {
        async fn dispatch(
            &self,
            _operator: &OperatorId,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let output = {
                let mut out = OperatorOutput::new(Content::text("ok"), ExitReason::Complete);
                out.effects = self.effects.clone();
                out
            };
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("test"));
            tokio::spawn(async move {
                let _ = sender.send(DispatchEvent::Completed { output }).await;
            });
            Ok(handle)
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
        let inner = Arc::new(EffectDispatcher {
            effects: vec![write_effect("key1", "value1")],
        });
        let mw = MiddlewareDispatcher::new(inner, vec![Arc::new(IdentityMiddleware)]);
        let operator = OperatorId::new("test");

        let output = mw.dispatch(&operator, make_input()).await.unwrap().collect().await.unwrap();
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
        let inner = Arc::new(EffectDispatcher {
            effects: vec![write_effect("secret", "password123")],
        });
        let mw = MiddlewareDispatcher::new(inner, vec![Arc::new(RedactMiddleware)]);
        let operator = OperatorId::new("test");

        let output = mw.dispatch(&operator, make_input()).await.unwrap().collect().await.unwrap();

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
        let inner = Arc::new(EffectDispatcher {
            effects: vec![write_effect("secret", "password")],
        });
        let mw = MiddlewareDispatcher::new(
            inner,
            vec![
                Arc::new(UppercaseKeyMiddleware),
                Arc::new(RedactMiddleware),
            ],
        );
        let operator = OperatorId::new("test");

        let output = mw.dispatch(&operator, make_input()).await.unwrap().collect().await.unwrap();

        if let Effect::WriteMemory { key, value, .. } = &output.effects[0] {
            assert_eq!(key, "SECRET");
            assert_eq!(value, &serde_json::json!("[REDACTED]"));
        } else {
            panic!("expected WriteMemory");
        }
    }
}
