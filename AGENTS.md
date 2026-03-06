# Extras — Provider Ecosystem

Provider ecosystem for [neuron](../neuron/). Heavy-dependency implementations
that don't belong in neuron core, shipped as a separate repo (terraform provider
model). Each crate wraps one external system behind a neuron trait.

## Crate Layout

| Directory | Crate(s)       | Purpose                              |
|-----------|----------------|--------------------------------------|
| `op/`     | `sweep`        | Sweep operators                      |
| `orch/`   | `sweep`, `temporal` | Orchestration backends          |
| `state/`  | `sqlite`, `cozo`    | State-store implementations     |
| `effects/`| `git`          | Git effects                          |
| `auth/`   | `neuron-auth-oauth`, `neuron-auth-pi`, `neuron-auth-omp` | Auth providers (OAuth Device Flow, pi OAuth, OMP SQLite) |

## Build Commands

Default (no native deps):
```sh
nix develop --command cargo test --workspace --all-targets
```

Full (requires cmake/clang/protobuf):
```sh
nix develop .#full --command cargo test --workspace --all-targets --all-features
```

## Feature Gates

| Feature         | What it enables                        | Native deps        |
|-----------------|----------------------------------------|--------------------|
| `cozo`          | Real CozoDB Datalog backend            | cmake, clang       |
| `rocksdb`       | Persistent CozoDB (RocksDB engine)     | cmake, clang       |
| `temporal-sdk`  | Real Temporal gRPC client              | protobuf           |

Default features require **no** native dependencies.

## Quality Gates

Same bar as neuron:
- `#![deny(missing_docs)]` on every crate
- Edition 2024
- All public types derive `Debug`, `Clone`, `Serialize`, `Deserialize`

## Dependencies

Path-depends on neuron core crates during development (`path = "../neuron/..."`).
Published versions will use crates.io deps with compatible semver ranges.

## Golden Constraint

Zero golden decision framework references in code, docs, or commit messages.
Decision IDs, framework vocabulary, and planning artifacts belong elsewhere.
