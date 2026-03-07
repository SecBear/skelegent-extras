//! Golden decision sweep runner.
//!
//! Reads decision cards from the golden repository, researches each via
//! Parallel.ai, compares findings against decisions via concurrent Sonnet
//! calls, synthesizes cross-decision patterns, and outputs a dashboard.

mod dashboard;
mod decisions;

use std::path::PathBuf;

use clap::Parser;

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

    // Load and verify all decision cards exist
    for id in &decision_ids {
        let card = decisions::load_card(&cli.golden, id).await?;
        tracing::debug!(id, bytes = card.len(), "loaded decision card");
    }
    tracing::info!("all {} decision cards verified", decision_ids.len());

    // TODO: Phase 3 Tasks 8-10 will implement:
    // 1. Initialize SQLite state store
    // 2. Seed decision cards into state
    // 3. Create ParallelClient for research
    // 4. Create AnthropicProvider for comparison
    // 5. Register CompareOperator + SynthesisOperator
    // 6. Run pipeline: research -> compare -> synthesize
    // 7. Print dashboard

    tracing::warn!("pipeline not yet implemented — use --plan-only for now");
    Ok(())
}
