# flatppl-flatpir — FlatPIR S-expression reader + writer

Read and write [`flatppl-core`](../core) modules from and to FlatPIR, the canonical S-expression
intermediate representation of FlatPPL (spec §11).

In FlatPIR, operators, field access, and indexing are lowered to function calls, and every call may
carry a `(%meta <type> <phase> <valueset>)` annotation.

* `read(text) -> Result<Module>` — parse `.flatpir`, with span-localized structural errors.
* `write(&module) -> String` — canonical FlatPIR, including any inference annotations.

This is the form term-rewriting and inference operate on. Pure library (no binary).

See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for where this sits in the pipeline.
