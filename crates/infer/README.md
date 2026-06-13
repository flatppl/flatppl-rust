# flatppl-infer — type, shape, and phase inference

The type-domain trace over [`flatppl-core`](../core) modules: annotate FlatPIR in place with
`(%meta <type> <phase> <valueset>)`.

* `infer(&mut module)` / `infer_with(&mut module, Level)` — annotate, returning diagnostics: errors
  for cycles and unresolved names, honest `%deferred` notes for ops without a rule yet.
* Levels form a hierarchy, each including the previous: `Phase` (binding classes only) ⊂ `Type` ⊂
  `Valueset` ⊂ `Normalization` (total-mass classes) ⊂ `Shape` (fixed-phase dims).

Phases follow the spec-§04 ancestor rule (`stochastic > parameterized > fixed`); the per-op rule
catalogue (`ops.rs`) is the one source of truth per op. Pure library (no binary).

See [`ARCHITECTURE.md`](../../ARCHITECTURE.md) for where this sits in the pipeline.
