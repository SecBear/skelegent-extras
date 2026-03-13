//! Scope-to-key-prefix mapping for CozoDB isolation.
//!
//! Every [`Scope`] variant maps to a unique string prefix that is prepended to
//! all keys stored in that scope. This guarantees full isolation between scopes
//! without requiring separate tables or databases.
//!
//! | Scope variant | Key prefix |
//! |---|---|
//! | `Global` | `"global"` |
//! | `Session(id)` | `"session:<id>"` |
//! | `Workflow(id)` | `"workflow:<id>"` |
//! | `Operator { workflow, operator }` | `"operator:<workflow>:<operator>"` |
//! | `Custom(s)` | `"custom:<s>"` |

use layer0::effect::Scope;

/// Map a [`Scope`] to a CozoDB key prefix string.
///
/// The returned string uniquely identifies the scope. It is used as a composite
/// key prefix for all KV entries and as the scope discriminator for all graph
/// edges belonging to that scope.
pub fn scope_prefix(scope: &Scope) -> String {
    match scope {
        Scope::Global => "global".to_string(),
        Scope::Session(id) => format!("session:{id}"),
        Scope::Workflow(id) => format!("workflow:{id}"),
        Scope::Operator { workflow, operator } => format!("operator:{workflow}:{operator}"),
        Scope::Custom(s) => format!("custom:{s}"),
        // Forward-compatible: serialize unknown variants to JSON.
        _ => serde_json::to_string(scope).unwrap_or_else(|_| "unknown".to_string()),
    }
}

/// Build a composite key from a scope and a user-facing key.
///
/// The null byte (`\0`) separator ensures no scope prefix can collide with a
/// key that happens to start with the same characters.
#[cfg(not(feature = "cozo"))]
pub(crate) fn composite_key(scope: &Scope, key: &str) -> String {
    format!("{}\0{}", scope_prefix(scope), key)
}

/// Extract the user-facing key from a composite key, if it belongs to the
/// given scope prefix.
///
/// Returns `None` if the composite key does not belong to the scope.
#[cfg(not(feature = "cozo"))]
pub(crate) fn extract_key<'a>(composite: &'a str, scope_pfx: &str) -> Option<&'a str> {
    composite
        .strip_prefix(scope_pfx)
        .and_then(|rest| rest.strip_prefix('\0'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::id::{OperatorId, SessionId, WorkflowId};

    #[test]
    fn global_prefix() {
        assert_eq!(scope_prefix(&Scope::Global), "global");
    }

    #[test]
    fn session_prefix() {
        let s = Scope::Session(SessionId::new("abc"));
        assert_eq!(scope_prefix(&s), "session:abc");
    }

    #[test]
    fn workflow_prefix() {
        let s = Scope::Workflow(WorkflowId::new("wf-1"));
        assert_eq!(scope_prefix(&s), "workflow:wf-1");
    }

    #[test]
    fn operator_prefix() {
        let s = Scope::Operator {
            workflow: WorkflowId::new("wf-1"),
            operator: OperatorId::new("planner"),
        };
        assert_eq!(scope_prefix(&s), "operator:wf-1:planner");
    }

    #[test]
    fn custom_prefix() {
        let s = Scope::Custom("research/topicX".to_string());
        assert_eq!(scope_prefix(&s), "custom:research/topicX");
    }

    #[cfg(not(feature = "cozo"))]
    #[test]
    fn composite_key_uses_null_separator() {
        let ck = composite_key(&Scope::Global, "some:key");
        assert_eq!(ck, "global\0some:key");
    }

    #[cfg(not(feature = "cozo"))]
    #[test]
    fn extract_key_round_trips() {
        let scope = Scope::Session(SessionId::new("s1"));
        let pfx = scope_prefix(&scope);
        let ck = composite_key(&scope, "my:key");
        assert_eq!(extract_key(&ck, &pfx), Some("my:key"));
    }

    #[cfg(not(feature = "cozo"))]
    #[test]
    fn extract_key_wrong_scope_returns_none() {
        let pfx_a = scope_prefix(&Scope::Global);
        let ck = composite_key(&Scope::Session(SessionId::new("s1")), "key");
        assert_eq!(extract_key(&ck, &pfx_a), None);
    }
}
