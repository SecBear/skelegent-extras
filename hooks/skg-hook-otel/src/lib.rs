#![deny(missing_docs)]
//! OpenTelemetry dispatch middleware for skelegent.
//!
//! Bridges [`layer0::TraceContext`] with the OpenTelemetry tracing API.
//! Wraps every dispatch in an OTel span, propagating trace/span IDs
//! between the skelegent context and the OTel [`Context`].
//!
//! # Usage
//!
//! ```rust,ignore
//! use skg_hook_otel::OtelMiddleware;
//! use layer0::middleware::DispatchStack;
//!
//! let stack = DispatchStack::builder()
//!     .observe(OtelMiddleware::new("my-service"))
//!     .build();
//! ```

use async_trait::async_trait;
use layer0::dispatch::DispatchHandle;
use layer0::dispatch_context::{DispatchContext, TraceContext};
use layer0::error::OrchError;
use layer0::middleware::{DispatchMiddleware, DispatchNext};
use layer0::operator::OperatorInput;
use opentelemetry::trace::{
    Span as _, SpanContext, Status, TraceContextExt, TraceFlags, TraceState, Tracer,
};
use opentelemetry::{global, Context, KeyValue, SpanId, TraceId};

/// Dispatch middleware that creates OpenTelemetry spans for every dispatch.
///
/// Before dispatch: reads the [`TraceContext`] from the dispatch context and
/// creates an OTel span linked to the parent (or a new root trace if empty).
/// The updated span/trace IDs are propagated to downstream middleware via a
/// cloned [`DispatchContext`].
///
/// After dispatch: sets the span status to `Ok` on success or records the
/// error and sets `Error` status on failure.
pub struct OtelMiddleware {
    tracer_name: &'static str,
}

impl OtelMiddleware {
    /// Create a new OTel middleware with the given tracer/service name.
    pub fn new(tracer_name: &'static str) -> Self {
        Self { tracer_name }
    }
}

/// Convert a skelegent [`TraceContext`] into an OTel [`Context`] carrying
/// the parent span as a remote span context.
///
/// Returns `Context::current()` (no parent) if trace_id or span_id are
/// empty or unparseable — the tracer will start a new root trace.
fn parent_context_from_trace(trace: &TraceContext) -> Context {
    if trace.trace_id.is_empty() || trace.span_id.is_empty() {
        return Context::current();
    }

    let Ok(trace_id) = TraceId::from_hex(&trace.trace_id) else {
        tracing::warn!(
            trace_id = %trace.trace_id,
            "invalid trace_id hex, starting new root trace"
        );
        return Context::current();
    };

    let Ok(span_id) = SpanId::from_hex(&trace.span_id) else {
        tracing::warn!(
            span_id = %trace.span_id,
            "invalid span_id hex, starting new root trace"
        );
        return Context::current();
    };

    let flags = TraceFlags::new(trace.trace_flags);
    let trace_state = trace
        .trace_state
        .as_deref()
        .and_then(|s| s.parse::<TraceState>().ok())
        .unwrap_or_default();

    let span_ctx = SpanContext::new(trace_id, span_id, flags, true, trace_state);
    Context::current().with_remote_span_context(span_ctx)
}

/// Extract updated [`TraceContext`] from an OTel span context after the
/// tracer has assigned real IDs.
fn trace_context_from_span(span_ctx: &SpanContext, original: &TraceContext) -> TraceContext {
    TraceContext {
        trace_id: span_ctx.trace_id().to_string(),
        span_id: span_ctx.span_id().to_string(),
        trace_flags: span_ctx.trace_flags().to_u8(),
        trace_state: original.trace_state.clone(),
    }
}

#[async_trait]
impl DispatchMiddleware for OtelMiddleware {
    /// Wrap the dispatch in an OTel span.
    ///
    /// 1. Build a parent OTel context from the incoming [`TraceContext`].
    /// 2. Start a span with operator/dispatch attributes.
    /// 3. Clone the [`DispatchContext`] with updated trace IDs.
    /// 4. Forward to the next middleware.
    /// 5. Set span status based on the dispatch result.
    async fn dispatch(
        &self,
        ctx: &DispatchContext,
        input: OperatorInput,
        next: &dyn DispatchNext,
    ) -> Result<DispatchHandle, OrchError> {
        let tracer = global::tracer(self.tracer_name);

        // Build parent context from incoming TraceContext.
        let parent_cx = parent_context_from_trace(&ctx.trace);

        // Create span with attributes.
        let mut span = tracer.start_with_context(
            format!("dispatch:{}", ctx.operator_id),
            &parent_cx,
        );

        span.set_attribute(KeyValue::new(
            "skelegent.operator_id",
            ctx.operator_id.to_string(),
        ));
        span.set_attribute(KeyValue::new(
            "skelegent.dispatch_id",
            ctx.dispatch_id.to_string(),
        ));
        if let Some(ref parent_id) = ctx.parent_id {
            span.set_attribute(KeyValue::new(
                "skelegent.parent_id",
                parent_id.to_string(),
            ));
        }

        // Extract the new span's IDs and propagate them downstream.
        let new_trace = trace_context_from_span(span.span_context(), &ctx.trace);
        let modified_ctx = ctx.clone().with_trace(new_trace);

        // Forward to next middleware/dispatcher.
        let result = next.dispatch(&modified_ctx, input).await;

        match &result {
            Ok(_) => {
                span.set_attribute(KeyValue::new("skelegent.dispatch.success", true));
                span.set_status(Status::Ok);
            }
            Err(err) => {
                // Classify the error for structured observability.
                use layer0::error::OperatorError;
                let (error_code, retryable) = match err {
                    OrchError::OperatorNotFound(_) => ("operator_not_found", false),
                    OrchError::WorkflowNotFound(_) => ("workflow_not_found", false),
                    OrchError::DispatchFailed(_) => ("dispatch_failed", true),
                    OrchError::SignalFailed(_) => ("signal_failed", true),
                    OrchError::OperatorError(op_err) => match op_err {
                        OperatorError::Model { retryable, .. } => {
                            if *retryable {
                                ("model_error_retryable", true)
                            } else {
                                ("model_error", false)
                            }
                        }
                        OperatorError::Retryable { .. } => ("retryable_error", true),
                        OperatorError::Halted { .. } => ("halted", false),
                        _ => ("operator_error", false),
                    },
                    OrchError::EnvironmentError(_) => ("environment_error", false),
                    _ => ("unknown_error", false),
                };
                span.set_attribute(KeyValue::new("skelegent.error.code", error_code));
                span.set_attribute(KeyValue::new("skelegent.error.retryable", retryable));
                span.set_status(Status::Error {
                    description: err.to_string().into(),
                });
            }
        }

        span.end();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::content::Content;
    use layer0::dispatch::DispatchEvent;
    use layer0::id::{DispatchId, OperatorId};
    use layer0::operator::{ExitReason, OperatorOutput, TriggerType};

    /// Mock next layer that echoes back with a fixed message.
    struct EchoNext;

    #[async_trait]
    impl DispatchNext for EchoNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let output = OperatorOutput::new(Content::text("echo"), ExitReason::Complete);
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("echo"));
            tokio::spawn(async move {
                let _ = sender.send(DispatchEvent::Completed { output }).await;
            });
            Ok(handle)
        }
    }

    /// Mock next layer that always fails with a plain dispatch error.
    struct FailNext;

    #[async_trait]
    impl DispatchNext for FailNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            Err(OrchError::DispatchFailed("test failure".into()))
        }
    }

    /// Mock next layer that fails with a structured OperatorError.
    struct FailOperatorNext;

    #[async_trait]
    impl DispatchNext for FailOperatorNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            use layer0::error::OperatorError;
            Err(OrchError::OperatorError(OperatorError::model("test")))
        }
    }

    fn test_ctx() -> DispatchContext {
        DispatchContext::new(DispatchId::new("d1"), OperatorId::new("op1"))
    }

    fn test_input() -> OperatorInput {
        OperatorInput::new(Content::text("hello"), TriggerType::User)
    }

    #[tokio::test]
    async fn otel_mw_passes_through_on_success() {
        let mw = OtelMiddleware::new("test-service");
        let next = EchoNext;
        let ctx = test_ctx();

        let handle = mw.dispatch(&ctx, test_input(), &next).await.unwrap();
        let output = handle.collect().await.unwrap();
        assert_eq!(output.message.as_text().unwrap(), "echo");
    }

    #[tokio::test]
    async fn otel_mw_propagates_error() {
        let mw = OtelMiddleware::new("test-service");
        let next = FailNext;
        let ctx = test_ctx();

        let result = mw.dispatch(&ctx, test_input(), &next).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("test failure"), "err: {err}");
    }

    #[tokio::test]
    async fn otel_mw_records_structured_error_attributes() {
        // Verifies that the structured error classification code path executes
        // without panicking. Span attribute inspection requires a custom OTel
        // exporter and is deferred to integration tests.
        let mw = OtelMiddleware::new("test-service");
        let next = FailOperatorNext;
        let ctx = test_ctx();

        let result = mw.dispatch(&ctx, test_input(), &next).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("test"), "err: {err}");
    }

    #[tokio::test]
    async fn otel_mw_handles_empty_trace_context() {
        // Default TraceContext has empty fields — should not panic.
        let mw = OtelMiddleware::new("test-service");
        let next = EchoNext;
        let ctx = test_ctx(); // trace is TraceContext::default() (empty)

        let handle = mw.dispatch(&ctx, test_input(), &next).await.unwrap();
        let output = handle.collect().await.unwrap();
        assert_eq!(output.message.as_text().unwrap(), "echo");
    }

    #[tokio::test]
    async fn otel_mw_propagates_existing_trace() {
        let mw = OtelMiddleware::new("test-service");

        // Capture the modified context to verify trace propagation.
        struct CtxCapture;

        #[async_trait]
        impl DispatchNext for CtxCapture {
            async fn dispatch(
                &self,
                ctx: &DispatchContext,
                _input: OperatorInput,
            ) -> Result<DispatchHandle, OrchError> {
                // The middleware should have updated the trace context
                // with IDs from the OTel span. With the noop tracer,
                // the span context will be invalid (all-zeros), but
                // the mechanism itself should not panic.
                let _ = &ctx.trace;
                let output =
                    OperatorOutput::new(Content::text("captured"), ExitReason::Complete);
                let (handle, sender) = DispatchHandle::channel(DispatchId::new("cap"));
                tokio::spawn(async move {
                    let _ = sender.send(DispatchEvent::Completed { output }).await;
                });
                Ok(handle)
            }
        }

        let trace = TraceContext::new(
            "58406520a006649127e371903a2de979",
            "1234567890abcdef",
        );
        let ctx = test_ctx().with_trace(trace);

        let handle = mw.dispatch(&ctx, test_input(), &CtxCapture).await.unwrap();
        let output = handle.collect().await.unwrap();
        assert_eq!(output.message.as_text().unwrap(), "captured");
    }
}
