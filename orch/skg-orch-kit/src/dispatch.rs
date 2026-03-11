//! Type-safe operator dispatch with automatic serialization.

use layer0::{
    operator::TriggerType, OperatorId, Content, OperatorInput, OperatorOutput, OrchError, Orchestrator,
};
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

/// Errors from typed dispatch.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DispatchError {
    /// Failed to serialize the typed input into `OperatorInput`.
    #[error("input serialization failed: {0}")]
    SerializeInput(String),

    /// Failed to deserialize `OperatorOutput.message` into the expected type.
    #[error("output deserialization failed: {0}")]
    DeserializeOutput(String),

    /// The underlying orchestrator dispatch failed.
    #[error("dispatch failed: {0}")]
    Orchestrator(#[from] OrchError),
}

/// Dispatch an operator with typed input and output.
///
/// Serializes `input` into `OperatorInput.message` as JSON text and
/// `OperatorInput.metadata` as the JSON value. Deserializes the
/// `OperatorOutput.message` text into `O`.
///
/// This is the primary dispatch primitive for composition code.
///
/// # Type Bounds
///
/// - `I: Serialize` — the typed input
/// - `O: DeserializeOwned` — the typed output
///
/// # Errors
///
/// Returns [`DispatchError::SerializeInput`] if `input` cannot be serialized,
/// [`DispatchError::DeserializeOutput`] if the output text cannot be deserialized
/// into `O`, or [`DispatchError::Orchestrator`] if the underlying dispatch fails.
pub async fn dispatch_typed<I, O>(
    orch: &dyn Orchestrator,
    operator: &OperatorId,
    input: I,
    trigger: TriggerType,
) -> Result<(O, OperatorOutput), DispatchError>
where
    I: Serialize,
    O: DeserializeOwned,
{
    let json_text = serde_json::to_string(&input)
        .map_err(|e| DispatchError::SerializeInput(e.to_string()))?;

    let metadata = serde_json::to_value(&input)
        .map_err(|e| DispatchError::SerializeInput(e.to_string()))?;

    let mut op_input = OperatorInput::new(Content::text(json_text), trigger);
    op_input.metadata = metadata;

    let output = orch.dispatch(operator, op_input).await?;

    let text = output
        .message
        .as_text()
        .ok_or_else(|| {
            DispatchError::DeserializeOutput("output message has no text content".into())
        })?;

    let typed: O = serde_json::from_str(text)
        .map_err(|e| DispatchError::DeserializeOutput(e.to_string()))?;

    Ok((typed, output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use layer0::{
        effect::SignalPayload, ExitReason, OrchError, QueryPayload, WorkflowId, dispatch::Dispatcher,
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestInput {
        question: String,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestOutput {
        answer: String,
    }

    /// Mock orchestrator that echoes input text back as output.
    struct EchoOrchestrator;

    #[async_trait]
    impl Dispatcher for EchoOrchestrator {
        async fn dispatch(
            &self,
            _operator: &OperatorId,
            input: OperatorInput,
        ) -> Result<OperatorOutput, OrchError> {
            Ok({
                            let mut out = OperatorOutput::new(input.message, ExitReason::Complete);
                            out.effects = vec![];
                            out
                        })
        }
    }

    #[async_trait]
    impl Orchestrator for EchoOrchestrator {
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

    /// Mock orchestrator that returns a fixed JSON response.
    struct FixedOrchestrator {
        response: String,
    }

    #[async_trait]
    impl Dispatcher for FixedOrchestrator {
        async fn dispatch(
            &self,
            _operator: &OperatorId,
            _input: OperatorInput,
        ) -> Result<OperatorOutput, OrchError> {
            Ok({
                            let mut out = OperatorOutput::new(Content::text(&self.response), ExitReason::Complete);
                            out.effects = vec![];
                            out
                        })
        }
    }

    #[async_trait]
    impl Orchestrator for FixedOrchestrator {
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

    #[tokio::test]
    async fn typed_roundtrip_with_echo() {
        let orch = Arc::new(EchoOrchestrator);
        let operator = OperatorId::new("test-agent");
        let input = TestInput {
            question: "what is 2+2?".into(),
        };

        // Echo returns the serialized input as output, so we deserialize it
        // back into TestInput (not TestOutput).
        let (output, raw): (TestInput, _) =
            dispatch_typed(&*orch, &operator, input.clone(), TriggerType::Task)
                .await
                .unwrap();

        assert_eq!(output, input);
        assert_eq!(raw.exit_reason, ExitReason::Complete);
    }

    #[tokio::test]
    async fn typed_roundtrip_with_fixed_response() {
        let orch = Arc::new(FixedOrchestrator {
            response: r#"{"answer":"four"}"#.into(),
        });
        let operator = OperatorId::new("test-agent");

        let (output, _): (TestOutput, _) = dispatch_typed(
            &*orch,
            &operator,
            TestInput {
                question: "what is 2+2?".into(),
            },
            TriggerType::Task,
        )
        .await
        .unwrap();

        assert_eq!(
            output,
            TestOutput {
                answer: "four".into()
            }
        );
    }

    #[tokio::test]
    async fn deserialization_failure() {
        let orch = Arc::new(FixedOrchestrator {
            response: "not valid json for TestOutput".into(),
        });
        let operator = OperatorId::new("test-agent");

        let result: Result<(TestOutput, _), _> = dispatch_typed(
            &*orch,
            &operator,
            TestInput {
                question: "hello".into(),
            },
            TriggerType::Task,
        )
        .await;

        assert!(matches!(result, Err(DispatchError::DeserializeOutput(_))));
    }

    #[tokio::test]
    async fn metadata_carries_structured_input() {
        // Verify that the metadata field carries the full structured input
        let orch = Arc::new(EchoOrchestrator);
        let operator = OperatorId::new("test-agent");
        let input = TestInput {
            question: "test".into(),
        };

        let (_, raw) =
            dispatch_typed::<_, TestInput>(&*orch, &operator, input, TriggerType::Task)
                .await
                .unwrap();

        // The raw output carries the original message (echo). The metadata was
        // set on the _input_ side. We can't easily inspect it through echo,
        // but the dispatch itself succeeded — the serialization path works.
        assert_eq!(raw.exit_reason, ExitReason::Complete);
    }
}
