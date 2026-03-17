//! Streaming A2A client using Server-Sent Events.
//!
//! This module provides [`dispatch_streaming`], which POSTs a `message/stream`
//! JSON-RPC request and consumes the SSE response, forwarding events as
//! [`DispatchEvent`]s through a [`DispatchHandle`] channel.

use layer0::content::Content;
use layer0::dispatch::{DispatchEvent, DispatchHandle};
use layer0::error::OrchError;
use layer0::id::DispatchId;
use layer0::operator::{ExitReason, OperatorInput, OperatorOutput};
use skg_a2a_core::convert::{a2a_artifact_to_artifact, content_to_parts, parts_to_content};
use skg_a2a_core::jsonrpc::methods;
use skg_a2a_core::types::TaskState;
use skg_a2a_core::{A2aMessage, A2aRole, JsonRpcRequest};

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------

/// A parsed Server-Sent Event frame.
#[derive(Debug, Default)]
struct SseFrame {
    /// The `event:` field, if present.
    pub event_type: Option<String>,
    /// Accumulated `data:` lines joined by `\n`.
    pub data: Option<String>,
}

impl SseFrame {
    fn is_empty(&self) -> bool {
        self.event_type.is_none() && self.data.is_none()
    }
}

/// Minimal SSE line parser.
///
/// Processes a single line from an SSE response and mutates the current
/// in-progress [`SseFrame`]:
///
/// - Lines starting with `:` are comments and are ignored.
/// - `event: <value>` sets the event type.
/// - `data: <value>` appends to the data buffer (newline-joined for multi-line).
/// - Empty lines signal end of frame and return `Some(frame)`, resetting state.
/// - All other lines are ignored per the SSE spec.
///
/// # Reconnection limitations
///
/// The `retry:` field is parsed per the SSE specification but reconnection
/// with exponential backoff is **not** implemented. The parsed value is
/// discarded.
///
/// The `id:` field (Last-Event-ID) is parsed per the SSE specification but
/// reconnection with the `Last-Event-ID` request header is **not**
/// implemented. The parsed value is discarded.
///
/// Callers that need automatic reconnection must implement it externally by
/// catching stream termination and re-invoking [`dispatch_streaming`] with
/// appropriate backoff.
///
/// Returns `Some(frame)` when an empty line terminates a non-empty frame,
/// `None` otherwise.
fn parse_sse_line(line: &str, frame: &mut SseFrame) -> Option<SseFrame> {
    if line.is_empty() {
        // Empty line: dispatch the current frame if non-empty.
        if !frame.is_empty() {
            return Some(std::mem::take(frame));
        }
        return None;
    }

    if line.starts_with(':') {
        // SSE comment — ignore.
        return None;
    }

    if let Some(rest) = line.strip_prefix("event:") {
        frame.event_type = Some(rest.trim().to_owned());
        return None;
    }

    if let Some(rest) = line.strip_prefix("data:") {
        // SSE spec: strip at most one leading space after the colon.
        let chunk = rest.strip_prefix(' ').unwrap_or(rest);
        match frame.data.as_mut() {
            Some(existing) => {
                existing.push('\n');
                existing.push_str(chunk);
            }
            None => {
                frame.data = Some(chunk.to_owned());
            }
        }
        return None;
    }

    // Unknown field — ignore per spec.
    None
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Dispatch to a remote A2A agent using streaming (SSE).
///
/// Unlike the sync dispatch in [`super::A2aDispatcher`], this uses the
/// `message/stream` JSON-RPC method and consumes Server-Sent Events for
/// real-time progress updates.
///
/// The function POSTs a JSON-RPC request with `Accept: text/event-stream`,
/// spawns a background task that reads the response as SSE, and returns a
/// [`DispatchHandle`] immediately. Events arrive asynchronously:
///
/// - `status_update` with non-terminal state → [`DispatchEvent::Progress`]
/// - `artifact_update` → [`DispatchEvent::ArtifactProduced`]
/// - `status_update` with `state == completed` → [`DispatchEvent::Completed`]
/// - `status_update` with `state == failed/canceled` → [`DispatchEvent::Failed`]
///
/// # Errors
///
/// Returns [`OrchError::DispatchFailed`] if the HTTP POST itself fails before
/// the stream starts. Errors occurring inside the stream are forwarded as
/// [`DispatchEvent::Failed`].
pub async fn dispatch_streaming(
    http: &reqwest::Client,
    endpoint_url: &str,
    input: OperatorInput,
) -> Result<DispatchHandle, OrchError> {
    // Build the A2A message.
    let a2a_msg = A2aMessage::new(A2aRole::User, content_to_parts(&input.message));
    let send_req_value = serde_json::json!({ "message": a2a_msg });

    // Wrap in JSON-RPC envelope using the streaming method.
    let rpc = JsonRpcRequest::new(methods::SEND_STREAMING_MESSAGE, send_req_value);

    // POST with Accept: text/event-stream so the server knows we want SSE.
    let response = http
        .post(endpoint_url)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .json(&rpc)
        .send()
        .await
        .map_err(|e| OrchError::DispatchFailed(e.to_string()))?;

    let (handle, sender) = DispatchHandle::channel(DispatchId::new("a2a-stream"));

    tokio::spawn(async move {
        pump_sse(response, sender).await;
    });

    Ok(handle)
}

/// Read SSE bytes from `response` and forward [`DispatchEvent`]s to `sender`.
///
/// Returns after forwarding a terminal event ([`DispatchEvent::Completed`] or
/// [`DispatchEvent::Failed`]).  If the HTTP stream closes before a terminal SSE
/// event is received, a synthetic `Failed` event is sent.
async fn pump_sse(
    mut response: reqwest::Response,
    sender: layer0::dispatch::DispatchSender,
) {
    let mut buffer = String::new();
    let mut frame = SseFrame::default();
    let mut terminal_sent = false;

    // Accumulate bytes into lines, parse SSE frames, dispatch events.
    'outer: loop {
        let chunk = match response.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                let _ = sender
                    .send(DispatchEvent::Failed {
                        error: OrchError::DispatchFailed(e.to_string()),
                    })
                    .await;
                return;
            }
        };

        // Decode as UTF-8 best-effort.
        let text = match std::str::from_utf8(&chunk) {
            Ok(s) => s.to_owned(),
            Err(_) => String::from_utf8_lossy(&chunk).into_owned(),
        };
        buffer.push_str(&text);

        // Process all complete lines in the buffer.
        while let Some(newline_pos) = buffer.find('\n') {
            // Extract the line (strip \r\n or \n).
            let line = {
                let raw = &buffer[..newline_pos];
                raw.trim_end_matches('\r').to_owned()
            };
            buffer.drain(..=newline_pos);

            if let Some(completed_frame) = parse_sse_line(&line, &mut frame) {
                let done = handle_frame(completed_frame, &sender).await;
                if done {
                    terminal_sent = true;
                    break 'outer;
                }
            }
        }
    }

    // Only send a synthetic failure if no terminal event was sent yet.
    // This avoids poisoning `DispatchHandle::collect()` callers who already
    // received a `Completed` event — `collect()` reads until channel close,
    // so a spurious `Failed` after `Completed` would override the success.
    if !terminal_sent {
        let _ = sender
            .send(DispatchEvent::Failed {
                error: OrchError::DispatchFailed("SSE stream ended without terminal event".into()),
            })
            .await;
    }
}

/// Process one completed SSE frame.
///
/// Parses the data field as a JSON object and dispatches [`DispatchEvent`]s
/// based on the `"type"` field. Uses raw JSON value extraction rather than
/// strict typed parsing so that server-side omissions (e.g. missing
/// `message_id` in inline status messages) don't silently drop events.
///
/// Returns `true` if a terminal event was sent and the pump should stop.
async fn handle_frame(
    frame: SseFrame,
    sender: &layer0::dispatch::DispatchSender,
) -> bool {
    let data = match frame.data {
        Some(d) => d,
        None => return false,
    };

    // Parse as a raw JSON object so we can inspect the "type" field
    // before attempting strongly-typed deserialization.
    let value: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("SSE frame parse failed: {}", e);
            return false;
        }
    };

    let event_type = match value.get("type").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return false,
    };

    match event_type {
        "status_update" => {
            // Extract the state from .status.state
            let state_str = value
                .pointer("/status/state")
                .and_then(|v| v.as_str())
                .unwrap_or("TASK_STATE_UNSPECIFIED");
            let state: TaskState = serde_json::from_value(
                serde_json::Value::String(state_str.to_owned()),
            )
            .unwrap_or(TaskState::Unspecified);

            // Extract message content from .status.message.parts if present.
            let content = extract_message_content(&value, "/status/message/parts")
                .unwrap_or_else(|| Content::text(format!("state: {state:?}")));

            if state.is_terminal() {
                let dispatch_event = match state {
                    TaskState::Completed => DispatchEvent::Completed {
                        output: OperatorOutput::new(content, ExitReason::Complete),
                    },
                    TaskState::Failed | TaskState::Rejected => DispatchEvent::Failed {
                        error: OrchError::DispatchFailed(
                            content.as_text().unwrap_or("Task failed").to_owned(),
                        ),
                    },
                    TaskState::Canceled => DispatchEvent::Failed {
                        error: OrchError::DispatchFailed("Task was canceled".into()),
                    },
                    _ => DispatchEvent::Failed {
                        error: OrchError::DispatchFailed(format!(
                            "unexpected terminal state: {state:?}"
                        )),
                    },
                };
                let _ = sender.send(dispatch_event).await;
                return true;
            }

            let _ = sender
                .send(DispatchEvent::Progress { content })
                .await;
        }

        "artifact_update" => {
            // Parse the artifact field directly.
            if let Some(artifact_val) = value.get("artifact")
                && let Ok(a2a_artifact) =
                    serde_json::from_value::<skg_a2a_core::types::A2aArtifact>(artifact_val.clone())
            {
                let artifact = a2a_artifact_to_artifact(&a2a_artifact);
                let _ = sender
                    .send(DispatchEvent::ArtifactProduced { artifact })
                    .await;
            }
        }

        "task" => {
            // Full task snapshot — treat terminal states as terminal events.
            if let Some(task_val) = value.get("task") {
                let state_str = task_val
                    .pointer("/status/state")
                    .and_then(|v| v.as_str())
                    .unwrap_or("TASK_STATE_UNSPECIFIED");
                let state: TaskState =
                    serde_json::from_value(serde_json::Value::String(state_str.to_owned()))
                        .unwrap_or(TaskState::Unspecified);

                if state.is_terminal() {
                    let content = extract_message_content(task_val, "/status/message/parts")
                        .unwrap_or_else(|| Content::text("Task completed"));

                    let dispatch_event = if state == TaskState::Completed {
                        DispatchEvent::Completed {
                            output: OperatorOutput::new(content, ExitReason::Complete),
                        }
                    } else {
                        DispatchEvent::Failed {
                            error: OrchError::DispatchFailed(format!("Task ended: {state:?}")),
                        }
                    };
                    let _ = sender.send(dispatch_event).await;
                    return true;
                }
            }
        }

        "message" => {
            // Standalone message: treat as terminal completed.
            if let Some(msg_val) = value.get("message") {
                let content = extract_message_content(msg_val, "/parts")
                    .unwrap_or_else(|| Content::text(""));
                let _ = sender
                    .send(DispatchEvent::Completed {
                        output: OperatorOutput::new(content, ExitReason::Complete),
                    })
                    .await;
                return true;
            }
        }

        _ => {
            // Unknown event type — skip.
        }
    }

    false
}

/// Extract [`Content`] from a JSON pointer path that points to an array of A2A parts.
///
/// Returns `None` if the path doesn't exist or parsing fails.
fn extract_message_content(
    value: &serde_json::Value,
    parts_pointer: &str,
) -> Option<Content> {
    let parts_val = value.pointer(parts_pointer)?;
    let parts: Vec<skg_a2a_core::types::Part> =
        serde_json::from_value(parts_val.clone()).ok()?;
    if parts.is_empty() {
        return None;
    }
    Some(parts_to_content(&parts))
}

// ---------------------------------------------------------------------------
// Unit tests — SSE parser
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_simple_data_frame() {
        let mut frame = SseFrame::default();
        assert!(parse_sse_line("data: hello", &mut frame).is_none());
        let completed = parse_sse_line("", &mut frame).unwrap();
        assert_eq!(completed.data.as_deref(), Some("hello"));
        assert!(completed.event_type.is_none());
    }

    #[test]
    fn sse_multiline_data() {
        let mut frame = SseFrame::default();
        parse_sse_line("data: line one", &mut frame);
        parse_sse_line("data: line two", &mut frame);
        let completed = parse_sse_line("", &mut frame).unwrap();
        assert_eq!(completed.data.as_deref(), Some("line one\nline two"));
    }

    #[test]
    fn sse_event_type() {
        let mut frame = SseFrame::default();
        parse_sse_line("event: status_update", &mut frame);
        parse_sse_line("data: {}", &mut frame);
        let completed = parse_sse_line("", &mut frame).unwrap();
        assert_eq!(completed.event_type.as_deref(), Some("status_update"));
        assert_eq!(completed.data.as_deref(), Some("{}"));
    }

    #[test]
    fn sse_comment_ignored() {
        let mut frame = SseFrame::default();
        parse_sse_line(": this is a comment", &mut frame);
        assert!(frame.data.is_none());
        assert!(frame.event_type.is_none());
    }

    #[test]
    fn sse_empty_line_without_data_does_not_dispatch() {
        let mut frame = SseFrame::default();
        assert!(parse_sse_line("", &mut frame).is_none());
    }

    #[test]
    fn sse_data_colon_in_value() {
        let mut frame = SseFrame::default();
        parse_sse_line(r#"data: {"key":"value"}"#, &mut frame);
        let completed = parse_sse_line("", &mut frame).unwrap();
        assert_eq!(completed.data.as_deref(), Some(r#"{"key":"value"}"#));
    }

    #[test]
    fn sse_data_space_trimmed() {
        // "data: text" and "data:text" should both yield "text"
        let cases = ["data: text", "data:text"];
        for input in cases {
            let mut frame = SseFrame::default();
            parse_sse_line(input, &mut frame);
            let completed = parse_sse_line("", &mut frame).unwrap();
            assert_eq!(completed.data.as_deref(), Some("text"), "input: {input:?}");
        }
    }

    #[test]
    fn sse_reset_after_dispatch() {
        let mut frame = SseFrame::default();
        parse_sse_line("data: first", &mut frame);
        parse_sse_line("", &mut frame).unwrap();

        // Frame should have been reset
        assert!(frame.is_empty());

        parse_sse_line("data: second", &mut frame);
        let completed = parse_sse_line("", &mut frame).unwrap();
        assert_eq!(completed.data.as_deref(), Some("second"));
    }
}
