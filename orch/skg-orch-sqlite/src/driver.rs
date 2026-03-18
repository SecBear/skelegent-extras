use async_trait::async_trait;
use layer0::operator::TriggerType;
use layer0::{Content, DispatchContext, Effect, ExitReason, Operator, OperatorInput};
use layer0::id::{DispatchId, OperatorId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use skg_run_core::{
    BackendRunRef, DispatchPayload, DriverError, DriverRequest, DriverResponse,
    PortableWakeDeadline, RunDriver, RunEvent, RunId, WaitPointId, WaitReason,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Minimal durable directive emitted by operators executed through [`SqliteRunDriver`].
///
/// This crate intentionally supports only a small honest contract: operators may
/// ask to wait without a wake deadline, complete with a result, or fail with an
/// error. Timed waits, replay-oriented semantics, and operator effects are out of
/// scope for this assembled SQLite orchestrator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DurableDirective {
    /// Suspend the run at a specific wait point.
    Wait {
        /// Wait point token that must later be resumed.
        wait_point: WaitPointId,
        /// Why the run is blocked.
        reason: WaitReason,
        /// Optional wake deadline for timer-backed waiting.
        ///
        /// The assembled SQLite orchestrator rejects timed waits until wake
        /// scheduling can be coordinated atomically with durable run state.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wake_at: Option<PortableWakeDeadline>,
    },
    /// Finish the run successfully with a result payload.
    Complete {
        /// Terminal result payload.
        result: Value,
    },
    /// Finish the run with a terminal error.
    Fail {
        /// Human-readable terminal failure summary.
        error: String,
    },
}

impl DurableDirective {
    fn into_event(self) -> RunEvent {
        match self {
            Self::Wait {
                wait_point,
                reason,
                wake_at,
            } => RunEvent::Wait {
                wait_point,
                reason,
                wake_at,
            },
            Self::Complete { result } => RunEvent::Complete { result },
            Self::Fail { error } => RunEvent::Fail { error },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BackendLink {
    operator_id: String,
}

#[derive(Debug, Deserialize)]
struct StartEnvelope {
    operator_id: String,
    input: Value,
}

/// SQLite durable run driver that dispatches registered operators and projects
/// their outputs back into portable durable kernel events.
pub struct SqliteRunDriver {
    operators: RwLock<HashMap<String, Arc<dyn Operator>>>,
}

impl SqliteRunDriver {
    /// Create a new empty durable run driver.
    pub fn new() -> Self {
        Self {
            operators: RwLock::new(HashMap::new()),
        }
    }

    /// Register or replace an operator by identifier.
    pub fn register(&self, operator_id: OperatorId, operator: Arc<dyn Operator>) {
        self.operators
            .write()
            .expect("operator registry lock poisoned")
            .insert(operator_id.to_string(), operator);
    }

    fn operator_for(&self, operator_id: &OperatorId) -> Result<Arc<dyn Operator>, DriverError> {
        self.operators
            .read()
            .map_err(|error| {
                DriverError::Backend(format!("operator registry lock poisoned: {error}"))
            })?
            .get(operator_id.as_str())
            .cloned()
            .ok_or_else(|| {
                DriverError::InvalidInput(format!("unregistered operator: {operator_id}"))
            })
    }

    fn resolve_dispatch(
        &self,
        request: &DriverRequest,
    ) -> Result<(OperatorId, Value, BackendRunRef), DriverError> {
        match &request.payload {
            DispatchPayload::Start { input } => {
                let envelope: StartEnvelope = serde_json::from_value(input.clone()).map_err(|error| {
                    DriverError::InvalidInput(
                        format!(
                            "start input must be {{\"operator_id\": string, \"input\": value}}: {error}"
                        ),
                    )
                })?;
                let operator_id = OperatorId::new(envelope.operator_id.clone());
                let backend_ref = encode_backend_ref(&operator_id)?;
                Ok((operator_id, envelope.input, backend_ref))
            }
            DispatchPayload::Resume { input, .. } => {
                let backend_ref = request.backend_ref.clone().ok_or_else(|| {
                    DriverError::Conflict(format!("missing backend ref for run {}", request.run_id))
                })?;
                let operator_id = decode_backend_ref(&backend_ref)?;
                Ok((operator_id, input.payload.clone(), backend_ref))
            }
            _ => Err(DriverError::InvalidInput(
                "unsupported dispatch payload for sqlite durable orchestrator".into(),
            )),
        }
    }

    fn event_from_output(&self, output: &layer0::OperatorOutput) -> Result<RunEvent, DriverError> {
        if let Some(effect) = output.effects.first() {
            return Err(DriverError::InvalidInput(format!(
                "operator effects are unsupported by sqlite durable orchestrator until atomic durable coordination exists: {}",
                effect_kind(effect)
            )));
        }

        if let Some(text) = output.message.as_text() {
            if let Ok(directive) = serde_json::from_str::<DurableDirective>(text) {
                if let DurableDirective::Wait {
                    wake_at: Some(_), ..
                } = &directive
                {
                    return Err(DriverError::InvalidInput(
                        "timed waits are unsupported by sqlite durable orchestrator until wake scheduling is durably coordinated".into(),
                    ));
                }
                return Ok(directive.into_event());
            }
            if matches!(output.exit_reason, ExitReason::Complete) {
                return Ok(RunEvent::Complete {
                    result: serde_json::from_str(text)
                        .unwrap_or_else(|_| Value::String(text.to_owned())),
                });
            }
            return Ok(RunEvent::Fail {
                error: format!("operator exited {:?}: {}", output.exit_reason, text),
            });
        }

        Ok(match output.exit_reason {
            ExitReason::Complete => RunEvent::Complete {
                result: Value::String(format!("{:?}", output.message)),
            },
            _ => RunEvent::Fail {
                error: format!("operator exited {:?}", output.exit_reason),
            },
        })
    }

    fn build_operator_input(
        &self,
        run_id: &RunId,
        payload: &DispatchPayload,
        value: Value,
    ) -> Result<OperatorInput, DriverError> {
        let message = Content::text(serde_json::to_string(&value).map_err(|error| {
            DriverError::InvalidInput(format!("serialize operator input: {error}"))
        })?);
        let trigger = match payload {
            DispatchPayload::Start { .. } => TriggerType::Task,
            DispatchPayload::Resume { .. } => TriggerType::SystemEvent,
            _ => TriggerType::SystemEvent,
        };
        let mut input = OperatorInput::new(message, trigger);
        input.metadata = serde_json::json!({
            "durable_run_id": run_id.as_str(),
            "scope": "sqlite_durable_orchestrator"
        });
        Ok(input)
    }
}

impl Default for SqliteRunDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RunDriver for SqliteRunDriver {
    async fn drive_run(&self, request: DriverRequest) -> Result<DriverResponse, DriverError> {
        let (operator_id, payload_value, backend_ref) = self.resolve_dispatch(&request)?;
        let operator = self.operator_for(&operator_id)?;
        let input = self.build_operator_input(&request.run_id, &request.payload, payload_value)?;
        let ctx = DispatchContext::new(DispatchId::new(request.run_id.as_str()), operator_id.clone());
        let output = operator.execute(input, &ctx).await.map_err(|error| {
            DriverError::Backend(format!(
                "execute operator {} for run {}: {error}",
                operator_id, request.run_id
            ))
        })?;
        Ok(DriverResponse::new(
            self.event_from_output(&output)?,
            Some(backend_ref),
        ))
    }
}

fn encode_backend_ref(operator_id: &OperatorId) -> Result<BackendRunRef, DriverError> {
    serde_json::to_string(&BackendLink {
        operator_id: operator_id.to_string(),
    })
    .map(BackendRunRef::new)
    .map_err(|error| DriverError::Backend(format!("serialize backend ref: {error}")))
}

fn decode_backend_ref(backend_ref: &BackendRunRef) -> Result<OperatorId, DriverError> {
    let link: BackendLink = serde_json::from_str(backend_ref.as_str())
        .map_err(|error| DriverError::Backend(format!("deserialize backend ref: {error}")))?;
    Ok(OperatorId::new(link.operator_id))
}

fn effect_kind(effect: &Effect) -> &'static str {
    match effect {
        Effect::WriteMemory { .. } => "write_memory",
        Effect::DeleteMemory { .. } => "delete_memory",
        Effect::Signal { .. } => "signal",
        Effect::Delegate { .. } => "delegate",
        Effect::Handoff { .. } => "handoff",
        Effect::LinkMemory { .. } => "link_memory",
        Effect::UnlinkMemory { .. } => "unlink_memory",
        Effect::ToolApprovalRequired { .. } => "tool_approval_required",
        Effect::Custom { .. } => "custom",
        _ => "unknown",
    }
}
