#![deny(missing_docs)]
//! OpenTelemetry middleware for skelegent.
//!
//! Provides two families of middleware:
//!
//! - [`OtelMiddleware`] — wraps dispatch calls in OTel spans with
//!   `skelegent.*` attributes (bridges [`layer0::TraceContext`]).
//! - [`OtelInferMiddleware`] — wraps provider inference calls in OTel spans
//!   following the `gen_ai.*` semantic conventions.
//! - [`OtelEmbedMiddleware`] — wraps provider embed calls in OTel spans
//!   following the `gen_ai.*` semantic conventions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use skg_hook_otel::{OtelMiddleware, OtelInferMiddleware, OtelEmbedMiddleware};
//! use layer0::middleware::DispatchStack;
//!
//! let stack = DispatchStack::builder()
//!     .observe(OtelMiddleware::new("my-service"))
//!     .build();
//! ```
//!
//! # Wiring tracing-opentelemetry
//!
//! This crate uses `tracing::info_span!` inside [`OtelMiddleware`] to mirror
//! each dispatch span into the `tracing` ecosystem. When a
//! [`tracing_opentelemetry::layer()`](https://docs.rs/tracing-opentelemetry)
//! is installed in the application's subscriber, those spans are automatically
//! forwarded to the configured OTel exporter — no code changes required here.
//!
//! ```rust,ignore
//! use tracing_subscriber::prelude::*;
//! use tracing_opentelemetry::layer as otel_layer;
//!
//! tracing_subscriber::registry()
//!     .with(otel_layer().with_tracer(my_tracer))
//!     .init();
//! ```
//!
//! The `tracing` spans emitted by this crate use the `skelegent.dispatch`
//! target, so they can be filtered independently of application spans.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use layer0::dispatch_context::{DispatchContext, TraceContext};
use layer0::error::OrchError;
use layer0::middleware::{DispatchMiddleware, DispatchNext};
use layer0::operator::OperatorInput;
use opentelemetry::trace::{
    Span as _, SpanContext, Status, TraceContextExt, TraceFlags, TraceState, Tracer,
};
use opentelemetry::{global, Context, KeyValue, SpanId, TraceId};
use skg_turn::embedding::{EmbedRequest, EmbedResponse};
use skg_turn::infer::{InferRequest, InferResponse};
use skg_turn::infer_middleware::{EmbedMiddleware, EmbedNext, InferMiddleware, InferNext};
use skg_turn::provider::ProviderError;

/// Dispatch middleware that creates OpenTelemetry spans for every dispatch.
///
/// Before dispatch: reads the [`TraceContext`] from the dispatch context and
/// creates an OTel span linked to the parent (or a new root trace if empty).
/// The updated span/trace IDs are propagated to downstream middleware via a
/// cloned [`DispatchContext`].
///
/// After dispatch: the span is kept open until the terminal
/// [`DispatchEvent::Completed`] or [`DispatchEvent::Failed`] event arrives.
/// On completion, result-level metrics from [`layer0::operator::OperatorMetadata`]
/// (`tokens_in`, `tokens_out`, `cost`, `turns_used`, tool-call counts) are
/// recorded as span attributes before the span is closed.
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

/// Classify an [`OrchError`] into a structured `(error_code, retryable)` pair
/// for OTel span attributes.
///
/// The `error_code` is a stable snake_case string suitable for OTel attribute
/// values and cardinality-safe dashboards. `retryable` signals whether the
/// caller should attempt a retry.
fn classify_orch_error(err: &OrchError) -> (&'static str, bool) {
    use layer0::error::OperatorError;
    match err {
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
    /// 5. If `next.dispatch()` returns an error, record it and close the span
    ///    immediately.
    /// 6. On success, keep the span open until [`DispatchEvent::Completed`]
    ///    arrives; then record result-level metrics from
    ///    [`layer0::operator::OperatorMetadata`] and close the span.
    ///    [`DispatchEvent::Failed`] is also handled: error attributes are set
    ///    and the span is closed.
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

        // Create a tracing span that mirrors the OTel span. Progress events
        // emitted inside this span's scope become OTel span events when a
        // tracing-opentelemetry subscriber is active.
        let tracing_span = tracing::info_span!(
            target: "skelegent.dispatch",
            "dispatch",
            operator_id = %ctx.operator_id,
            dispatch_id = %ctx.dispatch_id,
        );

        // Forward to next middleware/dispatcher.
        let result = next.dispatch(&modified_ctx, input).await;

        match result {
            Err(err) => {
                // Dispatch failed to start: classify error, record on span, and
                // close it immediately (no channel events will arrive).
                let (error_code, retryable) = classify_orch_error(&err);
                span.set_attribute(KeyValue::new("skelegent.dispatch.success", false));
                span.set_attribute(KeyValue::new("skelegent.error.code", error_code));
                span.set_attribute(KeyValue::new("skelegent.error.retryable", retryable));
                span.set_status(Status::Error {
                    description: err.to_string().into(),
                });
                span.end();
                Err(err)
            }
            Ok(handle) => {
                // The dispatch was accepted. Keep the span open until the
                // terminal event arrives so we can record result-level metrics
                // from OperatorMetadata.
                //
                // The intercept closure is `Fn` (not `FnMut`), so span access
                // requires interior mutability. The Mutex is only ever locked
                // for the brief duration of recording attributes at terminal
                // events; there is no contention in practice.
                let span_arc = Arc::new(Mutex::new(span));
                let span_for_intercept = Arc::clone(&span_arc);

                Ok(handle.intercept(move |event| {
                    match event {
                        DispatchEvent::Progress { content } => {
                            // Bridge progress events to the tracing ecosystem.
                            tracing_span.in_scope(|| {
                                tracing::info!(
                                    target: "skelegent.dispatch.progress",
                                    message = %content.as_text().unwrap_or(""),
                                    "dispatch progress"
                                );
                            });
                        }
                        DispatchEvent::Completed { output } => {
                            // Record result-level metrics from the operator's
                            // execution summary, then close the span.
                            let meta = &output.metadata;
                            let total_tools = meta.sub_dispatches.len() as i64;
                            let failed_tools = meta
                                .sub_dispatches
                                .iter()
                                .filter(|r| !r.success)
                                .count() as i64;

                            let mut s = span_for_intercept.lock().unwrap();
                            s.set_attribute(KeyValue::new("skelegent.dispatch.success", true));
                            s.set_attribute(KeyValue::new(
                                "skelegent.tokens_in",
                                meta.tokens_in as i64,
                            ));
                            s.set_attribute(KeyValue::new(
                                "skelegent.tokens_out",
                                meta.tokens_out as i64,
                            ));
                            s.set_attribute(KeyValue::new(
                                "skelegent.cost",
                                meta.cost.to_string(),
                            ));
                            s.set_attribute(KeyValue::new(
                                "skelegent.turns_used",
                                meta.turns_used as i64,
                            ));
                            s.set_attribute(KeyValue::new(
                                "skelegent.tool_calls_total",
                                total_tools,
                            ));
                            s.set_attribute(KeyValue::new(
                                "skelegent.tool_calls_failed",
                                failed_tools,
                            ));
                            s.set_status(Status::Ok);
                            s.end();
                        }
                        DispatchEvent::Failed { error } => {
                            // Terminal failure delivered through the channel
                            // (distinct from next.dispatch() returning Err above).
                            let (error_code, retryable) = classify_orch_error(error);
                            let mut s = span_for_intercept.lock().unwrap();
                            s.set_attribute(KeyValue::new("skelegent.dispatch.success", false));
                            s.set_attribute(KeyValue::new("skelegent.error.code", error_code));
                            s.set_attribute(KeyValue::new(
                                "skelegent.error.retryable",
                                retryable,
                            ));
                            s.set_status(Status::Error {
                                description: error.to_string().into(),
                            });
                            s.end();
                        }
                        // Other events (ArtifactProduced, EffectEmitted,
                        // AwaitingApproval) are forwarded transparently.
                        _ => {}
                    }
                }))
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// OtelInferMiddleware — gen_ai.* semconv for inference
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Infer middleware that creates OpenTelemetry spans following the
/// [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
///
/// Before inference: starts a `gen_ai.chat` span and records request
/// attributes (`gen_ai.operation.name`, `gen_ai.request.model`,
/// `gen_ai.request.max_tokens`, `gen_ai.request.temperature`).
///
/// After inference: records response attributes
/// (`gen_ai.response.model`, `gen_ai.usage.input_tokens`,
/// `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reason`) on
/// success, or sets an `Error` span status on failure.
pub struct OtelInferMiddleware {
    tracer_name: &'static str,
}

impl OtelInferMiddleware {
    /// Create a new OTel infer middleware with the given tracer/service name.
    pub fn new(tracer_name: &'static str) -> Self {
        Self { tracer_name }
    }
}

#[async_trait]
impl InferMiddleware for OtelInferMiddleware {
    /// Wrap an inference call in a `gen_ai.chat` OTel span.
    ///
    /// 1. Start a fresh `gen_ai.chat` span (no parent propagation).
    /// 2. Record pre-call `gen_ai.*` request attributes.
    /// 3. Forward to the next layer via `next.infer(request).await`.
    /// 4. Record post-call response attributes on success, or set
    ///    `Error` status on failure.
    async fn infer(
        &self,
        request: InferRequest,
        next: &dyn InferNext,
    ) -> Result<InferResponse, ProviderError> {
        let tracer = global::tracer(self.tracer_name);
        let mut span = tracer.start("gen_ai.chat");

        span.set_attribute(KeyValue::new("gen_ai.operation.name", "chat"));
        if let Some(ref model) = request.model {
            span.set_attribute(KeyValue::new("gen_ai.request.model", model.clone()));
        }
        if let Some(max_tokens) = request.max_tokens {
            span.set_attribute(KeyValue::new(
                "gen_ai.request.max_tokens",
                max_tokens as i64,
            ));
        }
        if let Some(temperature) = request.temperature {
            span.set_attribute(KeyValue::new(
                "gen_ai.request.temperature",
                temperature,
            ));
        }

        let result = next.infer(request).await;

        match &result {
            Ok(response) => {
                span.set_attribute(KeyValue::new(
                    "gen_ai.response.model",
                    response.model.clone(),
                ));
                span.set_attribute(KeyValue::new(
                    "gen_ai.usage.input_tokens",
                    response.usage.input_tokens as i64,
                ));
                span.set_attribute(KeyValue::new(
                    "gen_ai.usage.output_tokens",
                    response.usage.output_tokens as i64,
                ));
                span.set_attribute(KeyValue::new(
                    "gen_ai.response.finish_reason",
                    stop_reason_str(&response.stop_reason),
                ));
                span.set_status(Status::Ok);
            }
            Err(err) => {
                span.set_status(Status::Error {
                    description: err.to_string().into(),
                });
            }
        }

        span.end();
        result
    }
}

/// Format a [`StopReason`] as a string for OTel `gen_ai.response.finish_reason`.
///
/// Uses lowercase snake_case names matching the OTel semantic conventions:
/// `"end_turn"`, `"tool_use"`, `"max_tokens"`, `"content_filter"`.
fn stop_reason_str(reason: &skg_turn::types::StopReason) -> &'static str {
    use skg_turn::types::StopReason;
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::ToolUse => "tool_use",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ContentFilter => "content_filter",
        _ => "unknown",
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// OtelEmbedMiddleware — gen_ai.* semconv for embedding
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Embed middleware that creates OpenTelemetry spans following the
/// [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
///
/// Before embedding: starts a `gen_ai.embed` span and records request
/// attributes (`gen_ai.operation.name`, `gen_ai.request.model`).
///
/// After embedding: records `gen_ai.usage.input_tokens` on success, or
/// sets an `Error` span status on failure.
pub struct OtelEmbedMiddleware {
    tracer_name: &'static str,
}

impl OtelEmbedMiddleware {
    /// Create a new OTel embed middleware with the given tracer/service name.
    pub fn new(tracer_name: &'static str) -> Self {
        Self { tracer_name }
    }
}

#[async_trait]
impl EmbedMiddleware for OtelEmbedMiddleware {
    /// Wrap an embed call in a `gen_ai.embed` OTel span.
    ///
    /// 1. Start a fresh `gen_ai.embed` span (no parent propagation).
    /// 2. Record pre-call `gen_ai.*` request attributes.
    /// 3. Forward to the next layer via `next.embed(request).await`.
    /// 4. Record `gen_ai.usage.input_tokens` on success, or set `Error`
    ///    status on failure.
    async fn embed(
        &self,
        request: EmbedRequest,
        next: &dyn EmbedNext,
    ) -> Result<EmbedResponse, ProviderError> {
        let tracer = global::tracer(self.tracer_name);
        let mut span = tracer.start("gen_ai.embed");

        span.set_attribute(KeyValue::new("gen_ai.operation.name", "embed"));
        if let Some(ref model) = request.model {
            span.set_attribute(KeyValue::new("gen_ai.request.model", model.clone()));
        }

        let result = next.embed(request).await;

        match &result {
            Ok(response) => {
                span.set_attribute(KeyValue::new(
                    "gen_ai.response.model",
                    response.model.clone(),
                ));
                span.set_attribute(KeyValue::new(
                    "gen_ai.usage.input_tokens",
                    response.usage.input_tokens as i64,
                ));
                span.set_status(Status::Ok);
            }
            Err(err) => {
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
    use layer0::operator::{ExitReason, OperatorOutput, SubDispatchRecord, TriggerType};

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

    /// Mock next layer that sends a [`DispatchEvent::Failed`] through the
    /// channel rather than returning `Err` from `dispatch()`.
    struct ChannelFailNext;

    #[async_trait]
    impl DispatchNext for ChannelFailNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("fail-channel"));
            tokio::spawn(async move {
                let _ = sender
                    .send(DispatchEvent::Failed {
                        error: OrchError::DispatchFailed("channel failure".into()),
                    })
                    .await;
            });
            Ok(handle)
        }
    }

    /// Mock next layer that returns an [`OperatorOutput`] with specific
    /// [`layer0::operator::OperatorMetadata`] values for attribute assertions.
    struct MetadataNext {
        tokens_in: u64,
        tokens_out: u64,
        turns_used: u32,
        successful_tools: usize,
        failed_tools: usize,
    }

    #[async_trait]
    impl DispatchNext for MetadataNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            use layer0::duration::DurationMs;

            let mut output =
                OperatorOutput::new(Content::text("meta-output"), ExitReason::Complete);
            output.metadata.tokens_in = self.tokens_in;
            output.metadata.tokens_out = self.tokens_out;
            output.metadata.turns_used = self.turns_used;

            for _ in 0..self.successful_tools {
                output
                    .metadata
                    .sub_dispatches
                    .push(SubDispatchRecord::new("tool", DurationMs::ZERO, true));
            }
            for _ in 0..self.failed_tools {
                output
                    .metadata
                    .sub_dispatches
                    .push(SubDispatchRecord::new("tool", DurationMs::ZERO, false));
            }

            let (handle, sender) = DispatchHandle::channel(DispatchId::new("meta"));
            tokio::spawn(async move {
                let _ = sender.send(DispatchEvent::Completed { output }).await;
            });
            Ok(handle)
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

    /// Mock next layer that emits progress events before completing.
    struct ProgressNext;

    #[async_trait]
    impl DispatchNext for ProgressNext {
        async fn dispatch(
            &self,
            _ctx: &DispatchContext,
            _input: OperatorInput,
        ) -> Result<DispatchHandle, OrchError> {
            let output = OperatorOutput::new(Content::text("done"), ExitReason::Complete);
            let (handle, sender) = DispatchHandle::channel(DispatchId::new("progress"));
            tokio::spawn(async move {
                let _ = sender
                    .send(DispatchEvent::Progress {
                        content: Content::text("step 1"),
                    })
                    .await;
                let _ = sender
                    .send(DispatchEvent::Progress {
                        content: Content::text("step 2"),
                    })
                    .await;
                let _ = sender.send(DispatchEvent::Completed { output }).await;
            });
            Ok(handle)
        }
    }

    #[tokio::test]
    async fn otel_mw_intercepts_progress_events() {
        // Verify that OtelMiddleware attaches a progress intercept that is
        // transparent to the consumer — progress events still flow through
        // and the final output is correct.
        let mw = OtelMiddleware::new("test-service");
        let next = ProgressNext;
        let ctx = test_ctx();

        let handle = mw.dispatch(&ctx, test_input(), &next).await.unwrap();

        // collect_all preserves intermediate events, proving the intercept
        // forwards them transparently.
        let collected = handle.collect_all().await.unwrap();
        assert_eq!(collected.output.message.as_text().unwrap(), "done");
        assert_eq!(
            collected.events.len(),
            2,
            "expected 2 progress events to pass through the intercept"
        );

        // Verify the progress content is preserved.
        for (i, event) in collected.events.iter().enumerate() {
            if let DispatchEvent::Progress { content } = event {
                let text = content.as_text().unwrap();
                assert_eq!(text, format!("step {}", i + 1));
            } else {
                panic!("expected Progress event at index {i}");
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Result-level metadata attribute tests
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Verify that the dispatch passes through correctly when non-zero metadata
    /// is present. The middleware reads metadata to set OTel span attributes;
    /// this test confirms that path does not panic and the output is preserved.
    ///
    /// Full span attribute value inspection requires an OTel exporter
    /// (e.g., `InMemorySpanExporter` from `opentelemetry_sdk`) and is deferred
    /// to integration tests that can install a global tracer provider.
    #[tokio::test]
    async fn otel_mw_records_metadata_on_completion() {
        let mw = OtelMiddleware::new("test-service");
        let next = MetadataNext {
            tokens_in: 100,
            tokens_out: 50,
            turns_used: 3,
            successful_tools: 2,
            failed_tools: 1,
        };
        let ctx = test_ctx();

        let handle = mw.dispatch(&ctx, test_input(), &next).await.unwrap();
        let output = handle.collect().await.unwrap();

        // Output passes through unchanged: middleware is transparent to the caller.
        assert_eq!(output.message.as_text().unwrap(), "meta-output");
        // The metadata values we injected are preserved on the output.
        assert_eq!(output.metadata.tokens_in, 100);
        assert_eq!(output.metadata.tokens_out, 50);
        assert_eq!(output.metadata.turns_used, 3);
        assert_eq!(output.metadata.sub_dispatches.len(), 3);
        assert_eq!(
            output.metadata.sub_dispatches.iter().filter(|r| !r.success).count(),
            1
        );
    }

    #[tokio::test]
    async fn otel_mw_records_zero_tools_when_no_sub_dispatches() {
        let mw = OtelMiddleware::new("test-service");
        let next = MetadataNext {
            tokens_in: 10,
            tokens_out: 5,
            turns_used: 1,
            successful_tools: 0,
            failed_tools: 0,
        };
        let ctx = test_ctx();

        let handle = mw.dispatch(&ctx, test_input(), &next).await.unwrap();
        let output = handle.collect().await.unwrap();

        assert_eq!(output.message.as_text().unwrap(), "meta-output");
        assert_eq!(output.metadata.sub_dispatches.len(), 0);
    }

    /// Verify that a [`DispatchEvent::Failed`] arriving through the channel
    /// (not returned directly from `next.dispatch()`) is handled without panic.
    /// The handle's `collect()` should surface the error.
    #[tokio::test]
    async fn otel_mw_handles_channel_failure_event() {
        let mw = OtelMiddleware::new("test-service");
        let next = ChannelFailNext;
        let ctx = test_ctx();

        let handle = mw.dispatch(&ctx, test_input(), &next).await.unwrap();
        let result = handle.collect().await;

        assert!(result.is_err(), "expected collect to return Err on Failed event");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("channel failure"),
            "unexpected error: {err}"
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // OtelInferMiddleware tests
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    use skg_turn::embedding::{Embedding, EmbedRequest, EmbedResponse};
    use skg_turn::infer::{InferRequest, InferResponse};
    use skg_turn::infer_middleware::{EmbedNext, InferNext};
    use skg_turn::provider::ProviderError;
    use skg_turn::types::{StopReason, TokenUsage};

    /// Mock infer next that echoes back a fixed response.
    struct EchoInferNext;

    #[async_trait]
    impl InferNext for EchoInferNext {
        async fn infer(&self, _request: InferRequest) -> Result<InferResponse, ProviderError> {
            Ok(InferResponse {
                content: layer0::content::Content::text("echo"),
                tool_calls: vec![],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: None,
                    cache_creation_tokens: None,
                    reasoning_tokens: None,
                },
                model: "test-model".into(),
                cost: None,
                truncated: None,
            })
        }
    }

    /// Mock infer next that always fails.
    struct FailInferNext;

    #[async_trait]
    impl InferNext for FailInferNext {
        async fn infer(&self, _request: InferRequest) -> Result<InferResponse, ProviderError> {
            Err(ProviderError::TransientError {
                message: "test failure".into(),
                status: None,
            })
        }
    }

    /// Mock embed next that echoes back a fixed response.
    struct EchoEmbedNext;

    #[async_trait]
    impl EmbedNext for EchoEmbedNext {
        async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse, ProviderError> {
            Ok(EmbedResponse {
                embeddings: vec![Embedding {
                    vector: vec![0.1, 0.2, 0.3],
                }],
                model: "embed-model".into(),
                usage: TokenUsage {
                    input_tokens: 8,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_creation_tokens: None,
                    reasoning_tokens: None,
                },
            })
        }
    }

    fn test_infer_request() -> InferRequest {
        use layer0::context::{Message, Role};
        InferRequest::new(vec![Message::new(
            Role::User,
            layer0::content::Content::text("hello"),
        )])
        .with_model("gpt-4o")
        .with_max_tokens(512)
    }

    fn test_embed_request() -> EmbedRequest {
        EmbedRequest::new(vec!["hello world".into()]).with_model("text-embedding-3-small")
    }

    #[tokio::test]
    async fn otel_infer_mw_passes_through() {
        let mw = OtelInferMiddleware::new("test-service");
        let next = EchoInferNext;

        let result = mw.infer(test_infer_request(), &next).await.unwrap();
        assert_eq!(result.content.as_text().unwrap(), "echo");
        assert_eq!(result.model, "test-model");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn otel_infer_mw_propagates_error() {
        let mw = OtelInferMiddleware::new("test-service");
        let next = FailInferNext;

        let result = mw.infer(test_infer_request(), &next).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("test failure"), "err: {err}");
    }

    #[tokio::test]
    async fn otel_embed_mw_passes_through() {
        let mw = OtelEmbedMiddleware::new("test-service");
        let next = EchoEmbedNext;

        let result = mw.embed(test_embed_request(), &next).await.unwrap();
        assert_eq!(result.embeddings.len(), 1);
        assert_eq!(result.model, "embed-model");
        assert_eq!(result.usage.input_tokens, 8);
    }
}
