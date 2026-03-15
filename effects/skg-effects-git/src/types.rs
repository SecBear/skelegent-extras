//! Data model for the git/GitHub effects module.

use skg_op_sweep_v2::VerdictStatus;
use serde::{Deserialize, Serialize};

/// Result of the deduplication check for existing PR evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupResult {
    /// Materially new evidence. Append a new section to the PR.
    New,
    /// Redundant evidence. Skip PR update.
    Redundant,
    /// Same conclusion, additional support. Add footnote, bump counts.
    Strengthens,
}

/// Mode in which a pull request is opened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrMode {
    /// Draft PR — not ready for merge, requires human review.
    Draft,
    /// Ready-for-review PR.
    Ready,
}

/// Action taken on a pull request during a sweep cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrAction {
    /// A new PR was created.
    Created,
    /// An existing PR was updated with new evidence.
    Updated,
    /// No PR action taken (evidence was redundant or verdict was Skipped).
    Skipped,
    /// PR was automatically merged (Confirmed verdict direct-commit path).
    AutoMerged,
    /// Stale PR was closed automatically.
    StaleClosed,
}

/// Record of PR activity for a single decision in a sweep cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrActivity {
    /// The decision identifier (e.g. `"topic-3b"`).
    pub decision_id: String,
    /// What action was taken.
    pub action: PrAction,
    /// URL of the PR or issue, if one was created or updated.
    pub pr_url: Option<String>,
}

/// One-line digest entry for a single swept decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestEntry {
    /// The decision identifier.
    pub decision_id: String,
    /// Verdict from the sweep.
    pub verdict: VerdictStatus,
    /// One-line human-readable summary.
    pub summary: String,
}

/// Daily summary of sweep activity, costs, and PR actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyDigest {
    /// ISO 8601 date string (`YYYY-MM-DD`).
    pub date: String,
    /// Decisions swept today with verdicts.
    pub sweeps: Vec<DigestEntry>,
    /// PRs created or updated today.
    pub pr_activity: Vec<PrActivity>,
    /// Candidate decision IDs staged today.
    pub new_candidates: Vec<String>,
    /// Total sweep cost in USD for this day.
    pub total_cost_usd: f64,
    /// Remaining budget in USD after today's sweeps.
    pub budget_remaining_usd: f64,
}

/// Configuration for the PR generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrGeneratorConfig {
    /// GitHub repository in `owner/repo` format.
    pub repo: String,
    /// Labels applied to sweep PRs. Default: `["sweep", "automated"]`.
    pub labels: Vec<String>,
    /// Days before open sweep PRs are auto-closed as stale. Default: `30`.
    pub stale_days: usize,
    /// Term-Jaccard threshold for Layer 2 dedup. Default: `0.7`.
    pub dedup_jaccard_threshold: f64,
    /// Whether to auto-commit Confirmed date updates directly. Default: `true`.
    pub auto_commit_confirmed: bool,
    /// Maximum PRs to create per hour. Default: `5`.
    pub pr_hourly_limit: usize,
    /// Maximum direct commits per hour (Confirmed path). Default: `10`.
    pub commit_hourly_limit: usize,
}

impl Default for PrGeneratorConfig {
    fn default() -> Self {
        Self {
            repo: String::new(),
            labels: vec!["sweep".to_string(), "automated".to_string()],
            stale_days: 30,
            dedup_jaccard_threshold: 0.7,
            auto_commit_confirmed: true,
            pr_hourly_limit: 5,
            commit_hourly_limit: 10,
        }
    }
}

/// Errors produced by the git/GitHub effects module.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitError {
    /// GitHub API returned an error response.
    #[error("GitHub API error: {0}")]
    ApiError(String),
    /// Request rate limit exceeded.
    #[error("rate limit: {0}")]
    RateLimit(String),
    /// Branch conflict prevented the operation.
    #[error("branch conflict: {0}")]
    BranchConflict(String),
    /// Resource not found.
    #[error("not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_result_serde_round_trip() {
        let variants = [
            DedupResult::New,
            DedupResult::Redundant,
            DedupResult::Strengthens,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).expect("serialize");
            let back: DedupResult = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(v, &back);
        }
    }

    #[test]
    fn pr_action_serde_round_trip() {
        let variants = [
            PrAction::Created,
            PrAction::Updated,
            PrAction::Skipped,
            PrAction::AutoMerged,
            PrAction::StaleClosed,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).expect("serialize");
            let back: PrAction = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(v, &back);
        }
    }

    #[test]
    fn daily_digest_serde_round_trip() {
        let digest = DailyDigest {
            date: "2026-03-04".to_string(),
            sweeps: vec![DigestEntry {
                decision_id: "topic-3b".to_string(),
                verdict: VerdictStatus::Confirmed,
                summary: "Confirmed with strong evidence.".to_string(),
            }],
            pr_activity: vec![PrActivity {
                decision_id: "topic-3b".to_string(),
                action: PrAction::AutoMerged,
                pr_url: None,
            }],
            new_candidates: vec!["candidate-1".to_string()],
            total_cost_usd: 0.45,
            budget_remaining_usd: 9.55,
        };
        let json = serde_json::to_string(&digest).expect("serialize");
        let back: DailyDigest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.date, digest.date);
        assert_eq!(back.sweeps.len(), 1);
        assert_eq!(back.sweeps[0].verdict, VerdictStatus::Confirmed);
        assert_eq!(back.pr_activity[0].action, PrAction::AutoMerged);
    }

    #[test]
    fn pr_generator_config_defaults() {
        let cfg = PrGeneratorConfig::default();
        assert_eq!(cfg.stale_days, 30);
        assert!((cfg.dedup_jaccard_threshold - 0.7).abs() < f64::EPSILON);
        assert!(cfg.auto_commit_confirmed);
        assert_eq!(cfg.pr_hourly_limit, 5);
        assert_eq!(cfg.commit_hourly_limit, 10);
        assert_eq!(cfg.labels, vec!["sweep", "automated"]);
    }

    #[test]
    fn git_error_display() {
        let e = GitError::ApiError("500 Internal Server Error".to_string());
        assert!(e.to_string().contains("500 Internal Server Error"));
        let e2 = GitError::RateLimit("60/hour exceeded".to_string());
        assert!(e2.to_string().contains("60/hour exceeded"));
    }
}
