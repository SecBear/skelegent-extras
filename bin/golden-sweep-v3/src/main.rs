//! Golden decision sweep runner v3.
//!
//! Uses `neuron-context-engine` primitives directly. No `Operator`,
//! no `Orchestrator`, no `dispatch_typed`. The compare and synthesis steps are
//! plain async functions that call `Context::compile().infer(provider)`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;
use layer0::effect::Scope;
use layer0::state::StateStore;
use layer0::test_utils::InMemoryStore;
use neuron_auth::AuthProviderChain;
use neuron_auth_omp::OmpAuthProvider;
use neuron_auth_pi::PiAuthProvider;
use neuron_client_parallel::ParallelClient;
use neuron_orch_kit::{ScopedState, ScopedStateView};
use neuron_provider_anthropic::AnthropicProvider;

use golden_sweep_v3::compare::CompareConfig;
use golden_sweep_v3::cycle::sweep_cycle;
use golden_sweep_v3::decisions;
use golden_sweep_v3::synthesis::SynthesisConfig;
use golden_sweep_v3::types::{ResearchResult, SweepDecision};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Golden decision sweep runner v3 — context-engine primitives, no Operator.
#[derive(Parser, Debug)]
#[command(name = "golden-sweep-v3", about = "Sweep decisions via context-engine")]
struct Cli {
    /// Path to the golden repository root.
    #[arg(long, env = "GOLDEN_PATH")]
    golden: PathBuf,

    /// Comma-separated decision IDs to sweep (default: all 23).
    #[arg(long)]
    decisions: Option<String>,

    /// Path to SQLite state database.
    #[arg(long, default_value = "sweep-state.db")]
    state_db: PathBuf,

    /// Daily budget cap in USD.
    #[arg(long, default_value = "10.0")]
    budget_cap: f64,

    /// Plan only — print what would be swept without making API calls.
    #[arg(long)]
    plan_only: bool,

    /// Use in-memory state instead of SQLite (for testing).
    #[arg(long)]
    ephemeral: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the auth provider chain: Pi (with refresh) → OMP (fallback).
fn build_auth_chain() -> Arc<AuthProviderChain> {
    let mut chain = AuthProviderChain::new();

    if let Some(pi) = PiAuthProvider::from_env() {
        tracing::info!("auth: pi provider found (with refresh)");
        chain.add(Arc::new(pi));
    }

    if let Some(omp) = OmpAuthProvider::from_env() {
        tracing::info!("auth: omp provider found (fallback)");
        chain.add(Arc::new(omp));
    }

    Arc::new(chain)
}

/// Convert a `ParallelClient` [`SearchResult`] to a sweep [`ResearchResult`].
fn convert_result(sr: neuron_client_parallel::SearchResult) -> ResearchResult {
    ResearchResult {
        url: sr.url,
        title: sr.title,
        snippet: sr.snippet,
        retrieved_at: sr.retrieved_at,
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "golden_sweep_v3=info".into()),
        )
        .init();

    let cli = Cli::parse();
    tracing::info!(golden = %cli.golden.display(), "golden-sweep-v3 starting");

    let decision_ids = decisions::parse_decision_ids(cli.decisions.as_deref())
        .map_err(|e| anyhow::anyhow!(e))?;

    tracing::info!(count = decision_ids.len(), "decisions to sweep");

    if cli.plan_only {
        println!("Plan: {} decisions to sweep", decision_ids.len());
        for id in &decision_ids {
            let q = neuron_op_sweep_v2::next_query(id, 0);
            println!(
                "  {:<6}  {}",
                id,
                q.map(|q| q.query.as_str()).unwrap_or("NO QUERY")
            );
        }
        return Ok(());
    }

    // ── Verify golden path ────────────────────────────────────────────────────
    let decisions_dir = cli.golden.join("decisions");
    if !decisions_dir.is_dir() {
        anyhow::bail!(
            "golden decisions directory not found: {}",
            decisions_dir.display()
        );
    }

    // ── Auth ──────────────────────────────────────────────────────────────────
    let auth = build_auth_chain();

    // ── State ─────────────────────────────────────────────────────────────────
    let store: Arc<dyn StateStore> = if cli.ephemeral {
        tracing::info!("state: using in-memory store");
        Arc::new(InMemoryStore::new())
    } else {
        tracing::info!(path = %cli.state_db.display(), "state: using sqlite");
        Arc::new(
            neuron_state_sqlite::SqliteStore::open(&cli.state_db)
                .context("failed to open state database")?,
        )
    };

    let scope = Scope::Custom("sweep".into());
    let scoped = Arc::new(ScopedStateView::new(Arc::clone(&store), scope));

    // ── Seed decision cards ───────────────────────────────────────────────────
    for id in &decision_ids {
        let card = decisions::load_card(&cli.golden, id).await?;
        let key = format!("card:{id}");
        scoped
            .write(&key, serde_json::Value::String(card))
            .await
            .context("failed to seed decision card into state")?;
        tracing::debug!(id, "seeded decision card");
    }
    tracing::info!("seeded {} decision cards into state", decision_ids.len());

    // ── Research via Parallel.ai ──────────────────────────────────────────────
    let parallel = ParallelClient::new("PARALLEL_API_KEY").with_auth(Arc::clone(&auth) as _);

    tracing::info!(
        "phase 1: researching {} decisions via Parallel.ai (ultra)",
        decision_ids.len()
    );

    let mut sweep_decisions: Vec<SweepDecision> = Vec::new();

    for id in &decision_ids {
        let sweep_count = 0usize; // TODO: read from state for rotation
        let dq = match neuron_op_sweep_v2::next_query(id, sweep_count) {
            Some(q) => q,
            None => {
                tracing::warn!(id, "no query registered, skipping");
                continue;
            }
        };

        tracing::info!(id, angle = %dq.angle, "submitting research task");

        match parallel.run_task(&dq.query, "ultra").await {
            Ok(results) => {
                let converted: Vec<ResearchResult> =
                    results.into_iter().map(convert_result).collect();
                tracing::info!(
                    id,
                    count = converted.len(),
                    angle = %dq.angle,
                    "research complete"
                );
                sweep_decisions.push(SweepDecision {
                    id: id.to_string(),
                    research_results: converted,
                    previous_verdict: None, // TODO: read from state
                });
            }
            Err(e) => {
                tracing::error!(id, error = %e, "research failed");
            }
        }
    }

    if sweep_decisions.is_empty() {
        anyhow::bail!("no research results returned for any decision");
    }

    // ── Compare + Synthesize — direct sweep_cycle() call ─────────────────────
    //
    // v3 key difference from sweep-runner:
    // - No LocalOrchestrator, no CompareOperator/SynthesisOperator registration
    // - No BudgetTracker, no CompositionTrace
    // - sweep_cycle() calls compare_decision() directly, which uses
    //   Context.compile().infer() with no intermediary dispatch layer
    tracing::info!(
        "phase 2: comparing {} decisions via Sonnet (context-engine)",
        sweep_decisions.len()
    );

    let provider = AnthropicProvider::with_auth(Arc::clone(&auth) as _);

    let report = sweep_cycle(
        &provider,
        sweep_decisions,
        scoped.as_ref(),
        cli.budget_cap,
        &CompareConfig::default(),
        &SynthesisConfig::default(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("sweep cycle failed: {e}"))?;

    // ── Dashboard ─────────────────────────────────────────────────────────────
    golden_sweep_v3::dashboard::print_report(&report);

    Ok(())
}
