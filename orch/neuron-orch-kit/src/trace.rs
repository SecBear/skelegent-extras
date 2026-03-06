//! Span-based composition tracing with cycle detection.
//!
//! [`CompositionTrace`] tracks nested dispatch calls as a stack of spans.
//! It detects cycles (A → B → A) and enforces maximum dispatch depth.

use layer0::AgentId;
use std::sync::Mutex;
use thiserror::Error;

/// Errors from the composition trace.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum TraceError {
    /// A dispatch cycle was detected (agent already on the stack).
    #[error("cycle detected: {agent} is already on the dispatch stack")]
    CycleDetected {
        /// The agent that caused the cycle.
        agent: AgentId,
    },
    /// Maximum dispatch depth exceeded.
    #[error("max depth {max_depth} exceeded (current depth: {current_depth})")]
    MaxDepthExceeded {
        /// The configured maximum depth.
        max_depth: usize,
        /// The current stack depth.
        current_depth: usize,
    },
}

/// A span in the composition trace representing a single dispatch.
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// The agent being dispatched to.
    pub agent_id: AgentId,
    /// Depth in the dispatch stack (0-indexed).
    pub depth: usize,
}

/// Tracks nested dispatch calls for cycle detection and depth limiting.
///
/// Thread-safe — uses interior mutability via [`Mutex`].
/// Clone produces a new trace with the same configuration but empty stack.
pub struct CompositionTrace {
    stack: Mutex<Vec<AgentId>>,
    max_depth: usize,
}

impl std::fmt::Debug for CompositionTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stack = self.stack.lock().unwrap();
        f.debug_struct("CompositionTrace")
            .field("depth", &stack.len())
            .field("max_depth", &self.max_depth)
            .field("stack", &*stack)
            .finish()
    }
}

impl CompositionTrace {
    /// Create a new trace with the given maximum dispatch depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            stack: Mutex::new(Vec::new()),
            max_depth,
        }
    }

    /// Enter a dispatch to the given agent.
    ///
    /// Returns a [`TraceSpan`] on success. The caller must call [`exit`] when
    /// the dispatch completes.
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::CycleDetected`] if the agent is already on the stack.
    /// Returns [`TraceError::MaxDepthExceeded`] if the stack depth would exceed `max_depth`.
    ///
    /// [`exit`]: CompositionTrace::exit
    pub fn enter(&self, agent_id: &AgentId) -> Result<TraceSpan, TraceError> {
        let mut stack = self.stack.lock().unwrap();

        if stack.contains(agent_id) {
            return Err(TraceError::CycleDetected {
                agent: agent_id.clone(),
            });
        }

        if stack.len() >= self.max_depth {
            return Err(TraceError::MaxDepthExceeded {
                max_depth: self.max_depth,
                current_depth: stack.len(),
            });
        }

        let depth = stack.len();
        stack.push(agent_id.clone());

        Ok(TraceSpan {
            agent_id: agent_id.clone(),
            depth,
        })
    }

    /// Exit a dispatch. Pops the agent from the stack.
    ///
    /// # Panics
    ///
    /// Panics if the stack is empty or the top doesn't match the span's agent.
    /// This indicates a programming error (mismatched enter/exit calls).
    pub fn exit(&self, span: &TraceSpan) {
        let mut stack = self.stack.lock().unwrap();
        let popped = stack.pop().expect("exit called on empty trace stack");
        assert_eq!(
            popped, span.agent_id,
            "trace stack mismatch: expected {:?}, got {:?}",
            span.agent_id, popped
        );
    }

    /// Current dispatch depth.
    pub fn depth(&self) -> usize {
        self.stack.lock().unwrap().len()
    }

    /// Check if an agent is currently on the dispatch stack.
    pub fn contains(&self, agent_id: &AgentId) -> bool {
        self.stack.lock().unwrap().contains(agent_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(name: &str) -> AgentId {
        AgentId::new(name)
    }

    #[test]
    fn happy_path_nesting() {
        let trace = CompositionTrace::new(10);

        let span_a = trace.enter(&agent("a")).unwrap();
        assert_eq!(span_a.depth, 0);
        assert_eq!(trace.depth(), 1);

        let span_b = trace.enter(&agent("b")).unwrap();
        assert_eq!(span_b.depth, 1);
        assert_eq!(trace.depth(), 2);

        trace.exit(&span_b);
        assert_eq!(trace.depth(), 1);

        trace.exit(&span_a);
        assert_eq!(trace.depth(), 0);
    }

    #[test]
    fn cycle_detection() {
        let trace = CompositionTrace::new(10);

        let span_a = trace.enter(&agent("a")).unwrap();
        let _span_b = trace.enter(&agent("b")).unwrap();

        // a is already on the stack
        let result = trace.enter(&agent("a"));
        assert!(matches!(result, Err(TraceError::CycleDetected { .. })));

        // After exiting, a can be entered again
        trace.exit(&_span_b);
        trace.exit(&span_a);
        let _span_a2 = trace.enter(&agent("a")).unwrap();
    }

    #[test]
    fn max_depth_exceeded() {
        let trace = CompositionTrace::new(2);

        let _s1 = trace.enter(&agent("a")).unwrap();
        let _s2 = trace.enter(&agent("b")).unwrap();

        let result = trace.enter(&agent("c"));
        assert!(matches!(result, Err(TraceError::MaxDepthExceeded { .. })));
    }

    #[test]
    fn contains_check() {
        let trace = CompositionTrace::new(10);

        assert!(!trace.contains(&agent("a")));
        let span = trace.enter(&agent("a")).unwrap();
        assert!(trace.contains(&agent("a")));
        assert!(!trace.contains(&agent("b")));
        trace.exit(&span);
        assert!(!trace.contains(&agent("a")));
    }

    #[test]
    fn reentry_after_exit() {
        let trace = CompositionTrace::new(10);

        let span = trace.enter(&agent("a")).unwrap();
        trace.exit(&span);

        // Same agent can be entered again after exit
        let span2 = trace.enter(&agent("a")).unwrap();
        assert_eq!(span2.depth, 0);
        trace.exit(&span2);
    }

    #[test]
    fn thread_safety() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CompositionTrace>();
    }
}
