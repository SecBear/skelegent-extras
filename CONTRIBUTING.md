# Contributing to skelegent-extras

Thank you for your interest in contributing to skelegent-extras! This document covers the
process for contributing and the standards we maintain.

## Project overview

skelegent-extras is a Rust workspace providing composable operators, orchestration backends,
state-store implementations, effect handlers, and authentication providers for the skelegent
agentic AI framework. See the root `Cargo.toml` for the full list of workspace members.

## Getting started

### Prerequisites

- **Rust 1.85+** (edition 2024)
- A working internet connection for downloading crate dependencies

### With Nix (recommended)

If you have [Nix](https://nixos.org/) installed:

```bash
nix develop
```

This provides the full development environment including Rust, clippy, rustfmt,
cargo-deny, lychee, and mdbook.

### Without Nix

Install Rust via [rustup](https://rustup.rs/) and ensure you have the stable
toolchain with clippy and rustfmt components:

```bash
rustup component add clippy rustfmt
```

### Fork and branch workflow

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/skelegent-extras.git
   cd skelegent-extras
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```
4. Make your changes, following the conventions below.
5. Push your branch and open a Pull Request against `main`.

## Conventions

All coding conventions, architectural decisions, and design principles are
documented in [`AGENTS.md`](./AGENTS.md) at the repository root. Read it before
submitting your first PR. Key highlights:

### Rust standards

- **Edition 2024**, resolver 2, minimum Rust 1.85
- **`#[async_trait]`** for async trait methods (not native async traits)
- **`thiserror`** for error types, two levels of nesting maximum
- **`schemars`** for JSON Schema derivation on tool inputs
- No `unwrap()` in library code
- `#[must_use]` on Result-returning functions
- `#![deny(missing_docs)]` on all public items
- Public types must derive `Debug`, `Clone`, `Serialize`, and `Deserialize`

### Workspace structure

The workspace is organized in five core directories:

| Directory | Purpose |
|-----------|---------|
| `op/` | Sweep operators |
| `orch/` | Orchestration backends (sweep, Temporal) |
| `state/` | State-store implementations (SQLite, CozoDB) |
| `effects/` | Effect implementations (Git) |
| `auth/` | Auth providers (OAuth Device Flow, pi OAuth, OMP SQLite) |

### Documentation

- Inline `///` doc comments on **every** public item.
- Every trait must have a doc example.
- When adding or changing public API, update all documentation surfaces in the
  same commit: source doc comments, crate `AGENTS.md`, crate `README.md`,
  examples, root `AGENTS.md`, and `llms.txt` as applicable.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`.

Scope is typically the crate name or directory name (e.g., `op`, `orch`, `state`,
`effects`, `auth`). Use no scope for workspace-wide changes.

Examples:

```
feat(op): add new sweep operator for batch processing
fix(state): handle SQLite concurrent access correctly
docs: update workspace structure overview
chore: add release-please config and initial CHANGELOGs
```

## Running checks

Before submitting a PR, run full verification:

```bash
nix develop --command cargo test --workspace --all-targets
nix develop --command cargo clippy --workspace -- -D warnings
```

If `cargo doc` warnings matter for your change, also run:
`nix develop --command cargo doc --workspace --no-deps`.

## Pull request process

1. Fill out the PR template completely.
2. Ensure all CI checks pass.
3. Keep PRs focused -- one concern per PR.
4. If your change adds a public type or trait, confirm you have updated all
   documentation surfaces.
5. Add or update tests for any behavioral changes.

## Path dependencies and publication

Before publication, path dependencies on skelegent core crates will be replaced
with crates.io dependencies. Do not be concerned about path dependencies during
development; they enable rapid iteration.

## License

By contributing to skelegent-extras, you agree that your contributions will be dual
licensed under the [MIT License](./LICENSE-MIT) and the
[Apache License 2.0](./LICENSE-APACHE), at the user's option. This is the same
license used by the project itself.
