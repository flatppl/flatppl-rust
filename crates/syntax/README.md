# flatppl-syntax — canonical FlatPPL surface syntax

Parse `.flatppl` text into [`flatppl-core`](../core), and pretty-print a module back out (spec §05).

* `parse(text) -> Result<Module>` — lowers surface sugar to core calls: operators → `add`/`mul`/…,
  indexing and field access → `get`, `~` → `draw`, `[…]` → `vector`.
* `print_with(&module, Syntax)` — `Syntax::Full` re-applies the sugar; `Syntax::Minimal` emits the
  lowered function-call form.

`parse` then `print_with` round-trips. Pure library (no binary).

See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for where this sits in the pipeline.
