//! Dashboard output for sweep results.

use neuron_op_sweep::cycle::CycleReport;
use neuron_op_sweep::types::VerdictStatus;

/// Print a human-readable dashboard table from the cycle report.
#[allow(dead_code)] // Wired when pipeline is implemented
pub fn print_report(report: &CycleReport) {
    println!();
    println!("{}", "=".repeat(72));
    println!("  Sweep Dashboard");
    println!("{}", "=".repeat(72));
    println!();
    println!(
        "  {:<8} {:<12} {:>10} {:>8} {:>8}  Angle",
        "Decision", "Status", "Confidence", "Support", "Contra"
    );
    println!(
        "  {:-<8} {:-<12} {:->10} {:->8} {:->8}  {:-<20}",
        "", "", "", "", "", ""
    );

    for v in &report.verdicts {
        let status = match &v.status {
            VerdictStatus::Confirmed => "Confirmed",
            VerdictStatus::Refined => "Refined",
            VerdictStatus::Challenged => "CHALLENGED",
            VerdictStatus::Obsoleted => "OBSOLETED",
            VerdictStatus::Skipped => "Skipped",
        };
        println!(
            "  {:<8} {:<12} {:>9.0}% {:>8} {:>8}  {}",
            v.decision_id,
            status,
            v.confidence * 100.0,
            v.num_supporting,
            v.num_contradicting,
            v.query_angle,
        );
    }

    if !report.skipped.is_empty() {
        println!();
        println!("  Skipped:");
        for (id, reason) in &report.skipped {
            println!("    {id}: {reason}");
        }
    }

    if let Some(ref synthesis) = report.synthesis {
        println!();
        println!(
            "  Synthesis: {} structural changes, {} candidates",
            synthesis.structural_changes.len(),
            synthesis.candidates.len(),
        );
        for sc in &synthesis.structural_changes {
            println!(
                "    [{:?}] {} (confidence: {:.0}%)",
                sc.change_type,
                sc.summary,
                sc.confidence * 100.0
            );
            for url in &sc.source_urls {
                println!("      -> {url}");
            }
        }
    }

    println!();
    let total_cost: f64 = report.verdicts.iter().map(|v| v.cost_usd).sum();
    println!(
        "  Total: {} verdicts, {} skipped, ${:.4} USD",
        report.verdicts.len(),
        report.skipped.len(),
        total_cost
    );
    println!("{}", "=".repeat(72));
}
