# neuron-extras — provider ecosystem for neuron

## Stability Notice

**This repository is pre-alpha.** The public API, crate structure, and inter-crate interfaces are subject to change without notice between any two commits. There are no stability guarantees, no deprecation period, and no semver compatibility commitment at this stage. Do not take a dependency on these crates in production code.

## What This Is

Provider ecosystem for [neuron](https://github.com/secbear/neuron). Heavy-dependency implementations that don't belong in neuron core — each crate wraps one external system behind a neuron trait. Uses a terraform-provider-style model: one concern per crate, opt-in via Cargo features.

## Crate Map

| Directory | Crate(s) | Purpose |
|-----------|----------|---------|
| `op/` | `neuron-op-sweep` | Sweep operators |
| `orch/` | `neuron-orch-sweep`, `neuron-orch-temporal` | Orchestration backends |
| `state/` | `neuron-state-sqlite`, `neuron-state-cozo` | State-store implementations |
| `effects/` | `neuron-effects-git` | Git effect executor |
| `auth/` | `neuron-auth-oauth`, `neuron-auth-pi`, `neuron-auth-omp` | Auth providers |

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
