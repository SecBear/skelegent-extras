//! Scoped state access — read/write a single partition of a [`StateStore`].
//!
//! [`StateStore`]: layer0::StateStore

use async_trait::async_trait;
use layer0::{Scope, SearchResult, StateError, StateStore, StoreOptions};
use std::sync::Arc;

/// Async trait for scoped state access. All operations are confined to a
/// single scope — callers cannot read or write outside their partition.
///
/// This is the capability injected into operators and composition code.
/// Cross-scope writes go through [`Effect::WriteMemory`] instead.
///
/// [`Effect::WriteMemory`]: layer0::Effect::WriteMemory
#[async_trait]
pub trait ScopedState: Send + Sync {
    /// Read a value by key within this scope.
    async fn read(&self, key: &str) -> Result<Option<serde_json::Value>, StateError>;

    /// Write a value within this scope. Creates or overwrites.
    async fn write(&self, key: &str, value: serde_json::Value) -> Result<(), StateError>;

    /// Delete a value within this scope. No-op if key doesn't exist.
    async fn delete(&self, key: &str) -> Result<(), StateError>;

    /// List keys under a prefix within this scope.
    async fn list(&self, prefix: &str) -> Result<Vec<String>, StateError>;

    /// Semantic search within this scope.
    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StateError>;

    /// Write with advisory storage hints. Default: ignores options.
    async fn write_hinted(
        &self,
        key: &str,
        value: serde_json::Value,
        options: &StoreOptions,
    ) -> Result<(), StateError> {
        let _ = options;
        self.write(key, value).await
    }
}

/// A [`ScopedState`] backed by a [`StateStore`] pinned to a single [`Scope`].
///
/// Every method delegates to the underlying store with `self.scope`.
/// Two `ScopedStateView` instances on the same store with different scopes
/// cannot see each other's data.
#[derive(Clone)]
pub struct ScopedStateView {
    store: Arc<dyn StateStore>,
    scope: Scope,
}

impl std::fmt::Debug for ScopedStateView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedStateView")
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl ScopedStateView {
    /// Create a new scoped view over the given store.
    pub fn new(store: Arc<dyn StateStore>, scope: Scope) -> Self {
        Self { store, scope }
    }

    /// The scope this view is pinned to.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }
}

#[async_trait]
impl ScopedState for ScopedStateView {
    async fn read(&self, key: &str) -> Result<Option<serde_json::Value>, StateError> {
        self.store.read(&self.scope, key).await
    }

    async fn write(&self, key: &str, value: serde_json::Value) -> Result<(), StateError> {
        self.store.write(&self.scope, key, value).await
    }

    async fn delete(&self, key: &str) -> Result<(), StateError> {
        self.store.delete(&self.scope, key).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, StateError> {
        self.store.list(&self.scope, prefix).await
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StateError> {
        self.store.search(&self.scope, query, limit).await
    }

    async fn write_hinted(
        &self,
        key: &str,
        value: serde_json::Value,
        options: &StoreOptions,
    ) -> Result<(), StateError> {
        self.store.write_hinted(&self.scope, key, value, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use layer0::{test_utils::InMemoryStore, Scope};

    fn store() -> Arc<dyn StateStore> {
        Arc::new(InMemoryStore::new())
    }

    #[tokio::test]
    async fn write_read_roundtrip() {
        let store = store();
        let view = ScopedStateView::new(store, Scope::Custom("test".into()));

        view.write("key1", serde_json::json!({"a": 1})).await.unwrap();
        let val = view.read("key1").await.unwrap();
        assert_eq!(val, Some(serde_json::json!({"a": 1})));
    }

    #[tokio::test]
    async fn read_nonexistent_returns_none() {
        let store = store();
        let view = ScopedStateView::new(store, Scope::Custom("test".into()));

        let val = view.read("missing").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn scope_isolation() {
        let store = store();
        let view_a = ScopedStateView::new(store.clone(), Scope::Custom("scope_a".into()));
        let view_b = ScopedStateView::new(store, Scope::Custom("scope_b".into()));

        view_a.write("shared_key", serde_json::json!("a_value")).await.unwrap();
        view_b.write("shared_key", serde_json::json!("b_value")).await.unwrap();

        assert_eq!(
            view_a.read("shared_key").await.unwrap(),
            Some(serde_json::json!("a_value"))
        );
        assert_eq!(
            view_b.read("shared_key").await.unwrap(),
            Some(serde_json::json!("b_value"))
        );
    }

    #[tokio::test]
    async fn delete_removes_key() {
        let store = store();
        let view = ScopedStateView::new(store, Scope::Custom("test".into()));

        view.write("key1", serde_json::json!(1)).await.unwrap();
        view.delete("key1").await.unwrap();
        assert!(view.read("key1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_filters_by_prefix() {
        let store = store();
        let view = ScopedStateView::new(store, Scope::Custom("test".into()));

        view.write("card:1", serde_json::json!(1)).await.unwrap();
        view.write("card:2", serde_json::json!(2)).await.unwrap();
        view.write("meta:1", serde_json::json!(3)).await.unwrap();

        let mut keys = view.list("card:").await.unwrap();
        keys.sort();
        assert_eq!(keys, vec!["card:1", "card:2"]);
    }

    #[tokio::test]
    async fn list_does_not_cross_scopes() {
        let store = store();
        let view_a = ScopedStateView::new(store.clone(), Scope::Custom("a".into()));
        let view_b = ScopedStateView::new(store, Scope::Custom("b".into()));

        view_a.write("key:1", serde_json::json!(1)).await.unwrap();
        view_b.write("key:2", serde_json::json!(2)).await.unwrap();

        let keys_a = view_a.list("key:").await.unwrap();
        assert_eq!(keys_a, vec!["key:1"]);

        let keys_b = view_b.list("key:").await.unwrap();
        assert_eq!(keys_b, vec!["key:2"]);
    }
}
