{
  description = "neuron-extras — provider ecosystem crates for neuron";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.treefmt-nix.flakeModule
      ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        {
          config,
          inputs',
          pkgs,
          lib,
          system,
          ...
        }:
        let
          rustToolchain = inputs'.fenix.packages.stable.withComponents [
            "cargo"
            "clippy"
            "rust-src"
            "rustc"
            "rustfmt"
            "rust-analyzer"
          ];

          # Packages shared by both devshells
          commonPackages =
            [
              rustToolchain
              pkgs.pkg-config
              pkgs.openssl
            ]
            ++ lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ]
            ++ lib.optionals pkgs.stdenv.isLinux [
              pkgs.cargo-tarpaulin
            ];

          commonEnv = {
            OPENSSL_NO_VENDOR = 1;
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.fenix.overlays.default ];
          };

          # ── treefmt ──────────────────────────────────────────────
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              rustfmt = {
                enable = true;
                package = rustToolchain;
              };
              nixfmt.enable = true;
              taplo.enable = true;
            };
          };

          # ── devShells ────────────────────────────────────────────
          devShells.default = pkgs.mkShell {
            buildInputs = [ config.treefmt.build.wrapper ] ++ commonPackages;

            inherit (commonEnv) OPENSSL_NO_VENDOR RUST_SRC_PATH;

            shellHook = ''
              echo "neuron-extras dev shell"
              echo ""
              echo "  rustc --version   — $(rustc --version)"
              echo "  cargo test        — run all tests"
              echo "  cargo clippy      — lint all crates"
              echo ""
            '';
          };

          devShells.full = pkgs.mkShell {
            buildInputs =
              [ config.treefmt.build.wrapper ]
              ++ commonPackages
              ++ [
                pkgs.cmake
                pkgs.clang
                pkgs.protobuf
              ];

            inherit (commonEnv) OPENSSL_NO_VENDOR RUST_SRC_PATH;

            CC = "${pkgs.clang}/bin/clang";
            CXX = "${pkgs.clang}/bin/clang++";

            shellHook = ''
              echo "neuron-extras FULL dev shell (native deps for optional features)"
              echo ""
              echo "  Extra: cmake, clang, protobuf"
              echo "  Enables: --features rocksdb, --features temporal-sdk"
              echo ""
              echo "  rustc --version   — $(rustc --version)"
              echo "  cmake --version   — $(cmake --version | head -1)"
              echo ""
            '';
          };

        };
    };
}
