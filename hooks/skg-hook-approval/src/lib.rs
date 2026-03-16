#![deny(missing_docs)]
//! Approval guard middleware for skelegent.
//!
//! Configurable guards that can block dispatch or inference operations
//! based on policy functions. Use as a Guard in middleware stacks for
//! human-in-the-loop approval, budget enforcement, or safety checks.
//!
//! # Examples
//!
//! ```rust,ignore
//! use skg_hook_approval::{DispatchApprovalGuard, InferApprovalGuard, PolicyDecision};
//! use layer0::middleware::DispatchStack;
//! use skg_turn::infer_middleware::InferStack;
//! use std::sync::Arc;
//!
//! // Block all dispatches to "dangerous-op"
//! let guard = DispatchApprovalGuard::block_operators(vec!["dangerous-op".into()]);
//! let stack = DispatchStack::builder()
//!     .guard(Arc::new(guard))
//!     .build();
//!
//! // Block inference calls that would cost more than $0.10
//! let infer_guard = InferApprovalGuard::budget_limit(0.10, 0.000_001);
//! let infer_stack = InferStack::builder()
//!     .guard(Arc::new(infer_guard))
//!     .build();
//! ```

use async_trait::async_trait;
use layer0::dispatch::DispatchHandle;
use layer0::dispatch_context::DispatchContext;
use layer0::error::OrchError;
use layer0::middleware::{DispatchMiddleware, DispatchNext};
use layer0::operator::OperatorInput;
use skg_turn::infer::{InferRequest, InferResponse};
use skg_turn::infer_middleware::{InferMiddleware, InferNext};
use skg_turn::provider::ProviderError;
use std::sync::Arc;

/// Type alias for a dispatch policy function stored in [`DispatchApprovalGuard`].
type DispatchPolicyFn =
    Arc<dyn Fn(&DispatchContext, &OperatorInput) -> PolicyDecision + Send + Sync>;

/// Type alias for an infer policy function stored in [`InferApprovalGuard`].
type InferPolicyFn = Arc<dyn Fn(&InferRequest) -> PolicyDecision + Send + Sync>;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// POLICY DECISION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Decision from a guard policy.
///
/// Returned by policy functions to indicate whether an operation
/// should proceed or be blocked.
#[derive(Debug, Clone)]
pub enum PolicyDecision {
    /// Allow the operation to proceed.
    Allow,
    /// Deny the operation with a human-readable reason.
    Deny(String),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DISPATCH APPROVAL GUARD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Guards dispatch operations with a policy function.
///
/// If the policy returns `Deny(reason)`, the dispatch is blocked and
/// [`OrchError::DispatchFailed`] is returned with the denial reason.
/// If it returns `Allow`, the dispatch proceeds to the next layer.
///
/// Use as a Guard in [`DispatchStack`](layer0::middleware::DispatchStack):
///
/// ```rust,ignore
/// use skg_hook_approval::{DispatchApprovalGuard, PolicyDecision};
/// use layer0::middleware::DispatchStack;
/// use std::sync::Arc;
///
/// let guard = DispatchApprovalGuard::new(|ctx, _input| {
///     if ctx.operator_id.as_str() == "restricted" {
///         PolicyDecision::Deny("operator restricted".into())
///     } else {
///         PolicyDecision::Allow
///     }
/// });
///
/// let stack = DispatchStack::builder()
///     .guard(Arc::new(guard))
///     .build();
/// ```
pub struct DispatchApprovalGuard {
    policy: DispatchPolicyFn,
}

impl DispatchApprovalGuard {
    /// Create a new dispatch approval guard with the given policy function.
    ///
    /// The policy receives the dispatch context and operator input, and returns
    /// either `Allow` to proceed or `Deny(reason)` to block the dispatch.
    pub fn new<F>(policy: F) -> Self
    where
        F: Fn(&DispatchContext, &OperatorInput) -> PolicyDecision + Send + Sync + 'static,
    {
        Self {
            policy: Arc::new(policy),
        }
    }

    /// Create a guard that blocks all dispatches to operators matching the given names.
    ///
    /// Any operator whose ID matches an entry in `names` will be denied.
    /// All other operators are allowed through.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use skg_hook_approval::DispatchApprovalGuard;
    ///
    /// let guard = DispatchApprovalGuard::block_operators(vec![
    ///     "file-writer".into(),
    ///     "shell-exec".into(),
    /// ]);
    /// ```
    pub fn block_operators(names: Vec<String>) -> Self {
        Self::new(move |ctx, _input| {
            if names.contains(&ctx.operator_id.to_string()) {
                PolicyDecision::Deny(format!(
                    "operator '{}' is blocked by approval policy",
                    ctx.operator_id
                ))
            } else {
                PolicyDecision::Allow
            }
        })
    }
}

impl std::fmt::Debug for DispatchApprovalGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchApprovalGuard").finish_non_exhaustive()
    }
}

#[async_trait]
impl DispatchMiddleware for DispatchApprovalGuard {
    /// Evaluate the policy and either proceed or return a dispatch error.
    ///
    /// On `Allow`: calls `next.dispatch()` and returns its result.
    /// On `Deny(reason)`: returns `OrchError::DispatchFailed(reason)` without
    /// calling `next`.
    async fn dispatch(
        &self,
        ctx: &DispatchContext,
        input: OperatorInput,
        next: &dyn DispatchNext,
    ) -> Result<DispatchHandle, OrchError> {
        match (self.policy)(ctx, &input) {
            PolicyDecision::Allow => next.dispatch(ctx, input).await,
            PolicyDecision::Deny(reason) => Err(OrchError::DispatchFailed(reason)),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// INFER APPROVAL GUARD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Guards inference operations with a policy function.
///
/// If the policy returns `Deny(reason)`, the inference is blocked and
/// [`ProviderError::ContentBlocked`] is returned with the denial reason.
/// If it returns `Allow`, the inference proceeds to the next layer.
///
/// Use as a Guard in [`InferStack`](skg_turn::infer_middleware::InferStack):
///
/// ```rust,ignore
/// use skg_hook_approval::{InferApprovalGuard, PolicyDecision};
/// use skg_turn::infer_middleware::InferStack;
/// use std::sync::Arc;
///
/// let guard = InferApprovalGuard::new(|request| {
///     if request.max_tokens.unwrap_or(0) > 10_000 {
///         PolicyDecision::Deny("token limit exceeded".into())
///     } else {
///         PolicyDecision::Allow
///     }
/// });
///
/// let stack = InferStack::builder()
///     .guard(Arc::new(guard))
///     .build();
/// ```
pub struct InferApprovalGuard {
    policy: InferPolicyFn,
}

impl InferApprovalGuard {
    /// Create a new infer approval guard with the given policy function.
    ///
    /// The policy receives the inference request and returns either `Allow`
    /// to proceed or `Deny(reason)` to block the inference.
    pub fn new<F>(policy: F) -> Self
    where
        F: Fn(&InferRequest) -> PolicyDecision + Send + Sync + 'static,
    {
        Self {
            policy: Arc::new(policy),
        }
    }

    /// Create a guard that blocks inference when the estimated cost exceeds the given threshold.
    ///
    /// Uses `max_tokens * cost_per_token` as the estimated cost. If the request's
    /// `max_tokens` is not set, the check is skipped and the request is allowed.
    ///
    /// # Parameters
    ///
    /// - `max_cost_usd`: Maximum allowed estimated cost in US dollars.
    /// - `cost_per_token`: Approximate cost per output token in US dollars.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use skg_hook_approval::InferApprovalGuard;
    ///
    /// // Block requests estimated to cost more than $0.10 at $0.000_001/token
    /// let guard = InferApprovalGuard::budget_limit(0.10, 0.000_001);
    /// ```
    pub fn budget_limit(max_cost_usd: f64, cost_per_token: f64) -> Self {
        debug_assert!(
            cost_per_token > 0.0,
            "zero cost_per_token means budget_limit will allow everything"
        );
        Self::new(move |request| {
            if let Some(max_tokens) = request.max_tokens {
                let estimated_cost = max_tokens as f64 * cost_per_token;
                if estimated_cost > max_cost_usd {
                    return PolicyDecision::Deny(format!(
                        "estimated cost ${estimated_cost:.6} exceeds budget ${max_cost_usd:.6}"
                    ));
                }
            }
            PolicyDecision::Allow
        })
    }
}

impl std::fmt::Debug for InferApprovalGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferApprovalGuard").finish_non_exhaustive()
    }
}

#[async_trait]
impl InferMiddleware for InferApprovalGuard {
    /// Evaluate the policy and either proceed or return a provider error.
    ///
    /// On `Allow`: calls `next.infer()` and returns its result.
    /// On `Deny(reason)`: returns `ProviderError::ContentBlocked { message: reason }`
    /// without calling `next`.
    async fn infer(
        &self,
        request: InferRequest,
        next: &dyn InferNext,
    ) -> Result<InferResponse, ProviderError> {
        match (self.policy)(&request) {
            PolicyDecision::Allow => next.infer(request).await,
            PolicyDecision::Deny(reason) => {
                Err(ProviderError::ContentBlocked { message: reason })
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TESTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::dispatch::{DispatchEvent, DispatchHandle};
    use layer0::id::{DispatchId, OperatorId};
    use layer0::operator::{ExitReason, OperatorOutput, TriggerType};
    use skg_turn::infer::InferResponse;
    use skg_turn::types::{StopReason, TokenUsage};

    // ── Dispatch helpers ─────────────────────────────────────────

    fn test_ctx(operator_id: &str) -> DispatchContext {
        DispatchContext::new(DispatchId::new("d1"), OperatorId::new(operator_id))
    }

    fn test_input() -> OperatorInput {
        OperatorInput::new(Content::text("hello"), TriggerType::User)
    }

    /// A `DispatchNext` terminal that always succeeds with an echo response.
    struct EchoNext;

    #[async_trait]
    impl DispatchNext for EchoNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let output = OperatorOutput::new(input.message, ExitReason::Complete);
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("echo"));
            tokio::spawn(async move {
                let _ = sender.send(DispatchEvent::Completed { output }).await;
            });
            Ok(handle)
        }
    }

    // ── Infer helpers ─────────────────────────────────────────────

    fn make_infer_request(max_tokens: Option<u32>) -> InferRequest {
        use layer0::context::{Message, Role};
        let mut req = InferRequest::new(vec![Message::new(Role::User, Content::text("hello"))]);
        req.max_tokens = max_tokens;
        req
    }

    fn make_infer_response() -> InferResponse {
        InferResponse {
            content: Content::text("ok"),
            tool_calls: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: "test".into(),
            cost: None,
            truncated: None,
        }
    }

    /// An `InferNext` terminal that always succeeds.
    struct EchoInferNext;

    #[async_trait]
    impl InferNext for EchoInferNext {
        async fn infer(&self, _request: InferRequest) -> Result<InferResponse, ProviderError> {
            Ok(make_infer_response())
        }
    }

    // ── Dispatch tests ────────────────────────────────────────────

    #[tokio::test]
    async fn dispatch_guard_allows() {
        let guard = DispatchApprovalGuard::new(|_ctx, _input| PolicyDecision::Allow);
        let ctx = test_ctx("my-op");
        let result = guard.dispatch(&ctx, test_input(), &EchoNext).await;
        assert!(result.is_ok(), "expected Allow to pass through: {result:?}");
        let output = result.unwrap().collect().await.unwrap();
        assert_eq!(output.message.as_text().unwrap(), "hello");
    }

    #[tokio::test]
    async fn dispatch_guard_denies() {
        let guard =
            DispatchApprovalGuard::new(|_ctx, _input| PolicyDecision::Deny("not permitted".into()));
        let ctx = test_ctx("my-op");
        let result = guard.dispatch(&ctx, test_input(), &EchoNext).await;
        assert!(result.is_err(), "expected Deny to block dispatch");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("not permitted"),
            "expected denial reason in error: {err}"
        );
    }

    #[tokio::test]
    async fn infer_guard_allows() {
        let guard = InferApprovalGuard::new(|_req| PolicyDecision::Allow);
        let request = make_infer_request(Some(100));
        let result = guard.infer(request, &EchoInferNext).await;
        assert!(result.is_ok(), "expected Allow to pass through: {result:?}");
    }

    #[tokio::test]
    async fn infer_guard_denies() {
        let guard =
            InferApprovalGuard::new(|_req| PolicyDecision::Deny("inference blocked".into()));
        let request = make_infer_request(Some(100));
        let result = guard.infer(request, &EchoInferNext).await;
        assert!(result.is_err(), "expected Deny to block inference");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("inference blocked"),
            "expected denial reason in error: {err}"
        );
    }

    #[tokio::test]
    async fn block_operators_filters_correctly() {
        let guard = DispatchApprovalGuard::block_operators(vec!["blocked-op".into()]);

        // Blocked operator is denied.
        let blocked_ctx = test_ctx("blocked-op");
        let result = guard.dispatch(&blocked_ctx, test_input(), &EchoNext).await;
        assert!(
            result.is_err(),
            "expected blocked-op to be denied: {result:?}"
        );
        assert!(result.unwrap_err().to_string().contains("blocked-op"));

        // Different operator is allowed.
        let allowed_ctx = test_ctx("allowed-op");
        let result = guard.dispatch(&allowed_ctx, test_input(), &EchoNext).await;
        assert!(
            result.is_ok(),
            "expected allowed-op to pass through: {result:?}"
        );
    }

    #[tokio::test]
    async fn budget_limit_blocks_expensive() {
        // cost = 200_000 tokens * $0.000_001/token = $0.20, exceeds $0.10 limit.
        let guard = InferApprovalGuard::budget_limit(0.10, 0.000_001);

        let expensive = make_infer_request(Some(200_000));
        let result = guard.infer(expensive, &EchoInferNext).await;
        assert!(result.is_err(), "expected expensive request to be blocked");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("budget"),
            "expected budget mention in error: {err}"
        );

        // A cheap request (50,000 tokens * $0.000_001 = $0.05) should pass.
        let cheap = make_infer_request(Some(50_000));
        let result = guard.infer(cheap, &EchoInferNext).await;
        assert!(
            result.is_ok(),
            "expected cheap request to pass: {result:?}"
        );
    }

    #[tokio::test]
    async fn budget_limit_allows_when_no_max_tokens() {
        // When max_tokens is not set, cost cannot be estimated — allow through.
        let guard = InferApprovalGuard::budget_limit(0.0, 0.000_001);
        let request = make_infer_request(None);
        let result = guard.infer(request, &EchoInferNext).await;
        assert!(
            result.is_ok(),
            "expected request without max_tokens to be allowed: {result:?}"
        );
    }
}
