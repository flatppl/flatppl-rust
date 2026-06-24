# FlatPPL — Rust implementation

Rust ecosystem for FlatPPL, the Flat Portable Probabilistic Language.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Components

This monorepo is a Cargo workspace; member crates live under `crates/`:

* [`flatppl-core`](crates/core) — the in-memory IR (extended-FlatPIR model)
* [`flatppl-syntax`](crates/syntax) — canonical FlatPPL surface syntax: parse + print
* [`flatppl-flatpir`](crates/flatpir) — FlatPIR S-expression reader + writer
* [`flatppl-infer`](crates/infer) — type, shape, and phase inference
* [`flatppl-hs3`](crates/hs3) — HS3 / pyhf → FlatPPL importer
* [`flatppl-fileaccess`](crates/fileaccess) — resolve a `source` (local path or `http`/`https` URL) to a local file, with the shared remote-content cache (native host layer)
* [`flatppl-cli`](crates/cli) — the `flatppl` command-line driver
* [`flatppl-lsp`](crates/lsp) — FlatPPL language server (diagnostics, hover, go-to-definition, completion)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design and the road ahead.

## Installation (early users)

Requires a Rust toolchain (≥ 1.85, e.g. via [rustup](https://rustup.rs)).
The crates are not published to crates.io yet; install the CLI straight from
the repository:

```sh
cargo install --git https://github.com/flatppl/flatppl-rust flatppl-cli
```

This places a `flatppl` binary on your Cargo bin path:

```sh
flatppl convert model.flatppl model.flatpir   # FlatPPL → FlatPIR
flatppl convert model.flatpir model.flatppl   # FlatPIR → FlatPPL
flatppl convert messy.flatppl tidy.flatppl    # canonicalize (same format)
flatppl infer model.flatppl typed.flatpir    # emit type/phase-annotated FlatPIR
flatppl infer --level=phase m.flatppl m.flatpir  # or: type, valueset, normalization, shape
```

Formats are inferred from the file extensions. FlatPPL output uses the full
surface syntax (operators, indexing, lambdas, `:=`); pass `--syntax minimal`
for the lowered function-call form instead.

## Building and testing (developers)

```sh
git clone https://github.com/flatppl/flatppl-rust
cd flatppl-rust
cargo build --workspace        # build all crates
cargo test --workspace         # run all test suites
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo run -p flatppl-cli -- convert model.flatppl model.flatpir
```

## License

[MIT](LICENSE)
