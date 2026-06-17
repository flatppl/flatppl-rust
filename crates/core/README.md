# flatppl-core — the in-memory IR

The extended-FlatPIR model that every other crate in the [flatppl-rust](../..) workspace reads
and writes.

A `Module` holds the node graph (`Node`, `Call`, `Ref`, `Scalar`), the binding graph, interned
`Symbol`s and `NodeId`s, and the type/phase annotation tables. It is syntax-agnostic — parsing and
printing live in `flatppl-syntax` and `flatppl-flatpir`, inference in `flatppl-infer`.

Pure library (no binary); compiles to `wasm32` and links cleanly into PyO3 / jlrs / cxx hosts.

See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for where this sits in the pipeline.
