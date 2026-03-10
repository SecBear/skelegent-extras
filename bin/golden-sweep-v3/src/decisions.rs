//! Decision card loading from the golden repository.
//!
//! Provides the full list of the 23 canonical decision IDs, a mapping from
//! each ID to its filename, and helpers for parsing CLI input and loading
//! decision card text from disk.

use std::path::Path;

/// All 23 golden decision IDs in canonical order.
pub const ALL_DECISION_IDS: &[&str] = &[
    "D1", "D2A", "D2B", "D2C", "D2D", "D2E", "D3A", "D3B", "D3C", "D4A", "D4B", "D4C", "D5",
    "C1", "C2", "C3", "C4", "C5", "L1", "L2", "L3", "L4", "L5",
];

/// Map a decision ID to its filename in `golden/decisions/`.
///
/// Returns `None` for unknown IDs.
pub fn decision_filename(id: &str) -> Option<&'static str> {
    match id {
        "D1" => Some("D1-trigger.md"),
        "D2A" => Some("D2A-identity.md"),
        "D2B" => Some("D2B-history.md"),
        "D2C" => Some("D2C-memory.md"),
        "D2D" => Some("D2D-tools.md"),
        "D2E" => Some("D2E-budget.md"),
        "D3A" => Some("D3A-model.md"),
        "D3B" => Some("D3B-durability.md"),
        "D3C" => Some("D3C-retry.md"),
        "D4A" => Some("D4A-isolation.md"),
        "D4B" => Some("D4B-credentials.md"),
        "D4C" => Some("D4C-backfill.md"),
        "D5" => Some("D5-exit.md"),
        "C1" => Some("C1-child-context.md"),
        "C2" => Some("C2-result-return.md"),
        "C3" => Some("C3-lifecycle.md"),
        "C4" => Some("C4-communication.md"),
        "C5" => Some("C5-observation.md"),
        "L1" => Some("L1-memory-writes.md"),
        "L2" => Some("L2-compaction.md"),
        "L3" => Some("L3-crash-recovery.md"),
        "L4" => Some("L4-budget.md"),
        "L5" => Some("L5-observability.md"),
        _ => None,
    }
}

/// Parse a comma-separated decision ID list from the CLI.
///
/// Returns all 23 IDs when `arg` is `None`. Returns an error string for any
/// unrecognized ID.
pub fn parse_decision_ids(arg: Option<&str>) -> Result<Vec<&'static str>, String> {
    match arg {
        None => Ok(ALL_DECISION_IDS.to_vec()),
        Some(s) => {
            let mut ids = Vec::new();
            for raw in s.split(',') {
                let trimmed = raw.trim();
                match ALL_DECISION_IDS.iter().find(|&&id| id == trimmed) {
                    Some(&id) => ids.push(id),
                    None => return Err(format!("unknown decision ID: {trimmed}")),
                }
            }
            if ids.is_empty() {
                return Err("no decision IDs specified".into());
            }
            Ok(ids)
        }
    }
}

/// Load decision card markdown text from `golden/decisions/{filename}`.
///
/// # Errors
///
/// Returns `std::io::Error` when the ID is not recognised or the file cannot
/// be read.
pub async fn load_card(golden_root: &Path, id: &str) -> Result<String, std::io::Error> {
    let filename = decision_filename(id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("unknown decision ID: {id}"),
        )
    })?;
    let path = golden_root.join("decisions").join(filename);
    tokio::fs::read_to_string(&path).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_decisions_returns_23() {
        let ids = parse_decision_ids(None).unwrap();
        assert_eq!(ids.len(), 23);
    }

    #[test]
    fn parse_specific_decisions() {
        let ids = parse_decision_ids(Some("D1,D2D,L4")).unwrap();
        assert_eq!(ids, vec!["D1", "D2D", "L4"]);
    }

    #[test]
    fn parse_invalid_decision_fails() {
        let err = parse_decision_ids(Some("D1,FAKE")).unwrap_err();
        assert!(err.contains("FAKE"));
    }

    #[test]
    fn all_ids_have_filenames() {
        for id in ALL_DECISION_IDS {
            assert!(
                decision_filename(id).is_some(),
                "no filename mapping for {id}"
            );
        }
    }
}
