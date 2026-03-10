# skelegent-extras — provider ecosystem for skelegent

## Stability Notice

**This repository is pre-alpha.** The public API, crate structure, and inter-crate interfaces are subject to change without notice between any two commits. There are no stability guarantees, no deprecation period, and no semver compatibility commitment at this stage. Do not take a dependency on these crates in production code.

## What This Is

Provider ecosystem for [skelegent](https://github.com/secbear/skelegent). Heavy-dependency implementations that don't belong in skelegent core — each crate wraps one external system behind a skelegent trait. Uses a terraform-provider-style model: one concern per crate, opt-in via Cargo features.

## Crate Map

| Directory | Crate(s) | Purpose |
|-----------|----------|---------|
| `op/` | `skg-op-sweep` | Sweep operators |
| `orch/` | `skg-orch-sweep`, `skg-orch-temporal` | Orchestration backends |
| `state/` | `skg-state-sqlite`, `skg-state-cozo` | State-store implementations |
| `effects/` | `skg-effects-git` | Git effect executor |
| `auth/` | `skg-auth-oauth`, `skg-auth-pi`, `skg-auth-omp` | Auth providers |

## Build

```sh
# Default (no native deps)
nix develop --command cargo test --workspace --all-targets

# Full (requires cmake/clang/protobuf)
nix develop .#full --command cargo test --workspace --all-targets --all-features
```

## Feature Gates

| Feature | What it enables | Native deps |
|---------|-----------------|-------------|
| `cozo` | CozoDB Datalog backend | cmake, clang |
| `rocksdb` | Persistent CozoDB (RocksDB engine) | cmake, clang |
| `temporal-sdk` | Temporal gRPC client | protobuf |

## License

MIT OR Apache-2.0
