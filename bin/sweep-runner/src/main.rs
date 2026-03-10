//! Golden decision sweep runner.
//!
//! Reads decision cards from the golden repository, researches each via
//! Parallel.ai, compares findings against decisions via concurrent Sonnet
//! calls, synthesizes cross-decision patterns, and outputs a dashboard.

mod dashboard;
mod decisions;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use layer0::effect::Scope;
use layer0::id::OperatorId;
use layer0::state::StateStore;
use layer0::test_utils::InMemoryStore;
use neuron_auth::AuthProviderChain;
use neuron_auth_omp::OmpAuthProvider;
use neuron_auth_pi::PiAuthProvider;
use neuron_client_parallel::ParallelClient;
use neuron_op_sweep::cycle::{SweepDecision, run_sweep_cycle};
use neuron_op_sweep::provider::ResearchResult;
use neuron_op_sweep::{CompareConfig, CompareOperator};
use neuron_op_sweep::synthesis_operator::SynthesisOperator;
use neuron_orch_kit::{
    BudgetTracker, CompositionTrace, ScopedState, ScopedStateView,
    budget::CapPolicy,
};
use neuron_provider_anthropic::AnthropicProvider;

/// Golden decision sweep runner.
#[derive(Parser, Debug)]
#[command(name = "sweep-runner", about = "Sweep decisions against current research")]
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

/// Convert Parallel.ai SearchResult to sweep ResearchResult.
fn convert_result(sr: neuron_client_parallel::SearchResult) -> ResearchResult {
    ResearchResult {
        url: sr.url,
        title: sr.title,
        snippet: sr.snippet,
        retrieved_at: sr.retrieved_at,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sweep_runner=info".into()),
        )
        .init();

    let cli = Cli::parse();
    tracing::info!(golden = %cli.golden.display(), "sweep runner starting");

    let decision_ids = decisions::parse_decision_ids(cli.decisions.as_deref())
        .map_err(|e| anyhow::anyhow!(e))?;

    tracing::info!(count = decision_ids.len(), "decisions to sweep");

    if cli.plan_only {
        println!("Plan: {} decisions to sweep", decision_ids.len());
        for id in &decision_ids {
            let q = neuron_op_sweep::next_query(id, 0);
            println!(
                "  {:<6}  {}",
                id,
                q.map(|q| q.query.as_str()).unwrap_or("NO QUERY")
            );
        }
        return Ok(());
    }

    // Verify golden path exists and has decisions/
    let decisions_dir = cli.golden.join("decisions");
    if !decisions_dir.is_dir() {
        anyhow::bail!(
            "golden decisions directory not found: {}",
            decisions_dir.display()
        );
    }

    // ── Auth ──────────────────────────────────────────────────────────────
    let auth = build_auth_chain();

    // ── State ─────────────────────────────────────────────────────────────
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

    // ── Seed decision cards into state ────────────────────────────────────
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

    // ── Clients ───────────────────────────────────────────────────────────
    let parallel = ParallelClient::new("PARALLEL_API_KEY")
        .with_auth(Arc::clone(&auth) as _);
    let compare_llm = AnthropicProvider::with_auth(Arc::clone(&auth) as _);
    let synthesis_llm = AnthropicProvider::with_auth(Arc::clone(&auth) as _);

    // ── Research (Parallel.ai Ultra) ──────────────────────────────────────
    tracing::info!("phase 1: researching {} decisions via Parallel.ai (ultra)", decision_ids.len());

    let mut sweep_decisions: Vec<SweepDecision> = Vec::new();

    for id in &decision_ids {
        let sweep_count = 0usize; // TODO: read from state for rotation
        let dq = match neuron_op_sweep::next_query(id, sweep_count) {
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
                tracing::info!(id, count = converted.len(), angle = %dq.angle, "research complete");
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

    // ── Compare + Synthesize ──────────────────────────────────────────────
    tracing::info!(
        "phase 2: comparing {} decisions via Sonnet",
        sweep_decisions.len()
    );

    let compare_operator = OperatorId::new("compare");
    let synthesis_operator = OperatorId::new("synthesis");
    let compare_config = CompareConfig::default(); // force_ultra: true

    // Build a local orchestrator with the two operators registered.
    let mut orch = layer0::test_utils::LocalOrchestrator::new();
    orch.register(
        compare_operator.clone(),
        Arc::new(CompareOperator::new(
            compare_llm,
            Arc::clone(&scoped) as _,
            compare_config,
        )),
    );
    orch.register(
        synthesis_operator.clone(),
        Arc::new(SynthesisOperator::new(
            synthesis_llm,
            Arc::clone(&scoped) as _,
            Default::default(),
        )),
    );

    let budget = BudgetTracker::new(CapPolicy {
        soft_cap: cli.budget_cap * 0.8,
        hard_cap: cli.budget_cap,
    });
    let trace = CompositionTrace::new(20);

    let report = run_sweep_cycle(
        &orch,
        scoped.as_ref(),
        &budget,
        &trace,
        &compare_operator,
        &synthesis_operator,
        cli.budget_cap,
        sweep_decisions,
    )
    .await
    .map_err(|e| anyhow::anyhow!("sweep cycle failed: {e}"))?;

    // ── Dashboard ─────────────────────────────────────────────────────────
    dashboard::print_report(&report);

    Ok(())
}
