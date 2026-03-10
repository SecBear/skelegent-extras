#![deny(missing_docs)]
//! Git/GitHub effect module for the sweep system.
//!
//! Translates [`skg_op_sweep::SweepVerdict`] results into git operations:
//! pull requests, direct commits (for Confirmed verdicts), and daily digest issues.
//!
//! # Modules
//!
//! - [`types`] — data model (dedup result, PR actions, digest, config, errors)
//! - [`client`] — [`GitHubClient`] trait and [`MockGitHubClient`] for tests
//! - [`dedup`] — three-layer deduplication logic (Layers 1 and 2)
//! - [`templates`] — branch naming, PR titles, PR bodies, and digest bodies
//!
//! # Routing
//!
//! [`route_verdict`] maps a [`VerdictStatus`] to the appropriate [`PrAction`]:
//!
//! | Verdict     | Action       | Notes                             |
//! |-------------|--------------|-----------------------------------|
//! | Confirmed   | AutoMerged   | Direct commit, no PR needed       |
//! | Refined     | Created      | Draft PR with evidence            |
//! | Challenged  | Created      | Draft PR + human notification     |
//! | Obsoleted   | Created      | Draft PR + human notification     |
//! | Skipped     | Skipped      | No action                         |

pub mod client;
pub mod dedup;
pub mod templates;
pub mod types;
pub mod github;

pub use client::{GitHubClient, PullRequest};
pub use github::GitHubApiClient;
pub use dedup::{dedup_check_layers_1_2, extract_urls};
pub use templates::{branch_name, digest_body, pr_body, pr_title};
pub use types::{
    DailyDigest, DedupResult, DigestEntry, GitError, PrAction, PrActivity, PrGeneratorConfig,
    PrMode,
};

use skg_op_sweep::VerdictStatus;

/// Route a sweep verdict status to the appropriate PR action.
///
/// This is a pure mapping function with no side effects; callers use the
/// returned [`PrAction`] to drive the actual git/GitHub operations.
///
/// | `VerdictStatus`           | `PrAction`    |
/// |---------------------------|---------------|
/// | [`VerdictStatus::Confirmed`]  | [`PrAction::AutoMerged`] |
/// | [`VerdictStatus::Refined`]    | [`PrAction::Created`]    |
/// | [`VerdictStatus::Challenged`] | [`PrAction::Created`]    |
/// | [`VerdictStatus::Obsoleted`]  | [`PrAction::Created`]    |
/// | [`VerdictStatus::Skipped`]    | [`PrAction::Skipped`]    |
pub fn route_verdict(verdict: &VerdictStatus) -> PrAction {
    match verdict {
        VerdictStatus::Confirmed => PrAction::AutoMerged,
        VerdictStatus::Refined => PrAction::Created,
        VerdictStatus::Challenged | VerdictStatus::Obsoleted => PrAction::Created,
        VerdictStatus::Skipped => PrAction::Skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skg_op_sweep::VerdictStatus;

    #[test]
    fn route_confirmed_is_auto_merged() {
        assert_eq!(
            route_verdict(&VerdictStatus::Confirmed),
            PrAction::AutoMerged
        );
    }

    #[test]
    fn route_refined_is_created() {
        assert_eq!(route_verdict(&VerdictStatus::Refined), PrAction::Created);
    }

    #[test]
    fn route_challenged_is_created() {
        assert_eq!(route_verdict(&VerdictStatus::Challenged), PrAction::Created);
    }

    #[test]
    fn route_obsoleted_is_created() {
        assert_eq!(route_verdict(&VerdictStatus::Obsoleted), PrAction::Created);
    }

    #[test]
    fn route_skipped_is_skipped() {
        assert_eq!(route_verdict(&VerdictStatus::Skipped), PrAction::Skipped);
    }
}
