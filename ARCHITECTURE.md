# flatppl-rust — Architecture & design decisions

**Status: design, pre-implementation.** This records the design decisions for the
`flatppl-rust` toolchain as they are made; it becomes the architecture reference as
crates land. For the language semantics, see the FlatPPL spec (`flatppl-design`).

## Big picture

`flatppl-rust` is a **codegen-first** toolchain for FlatPPL: read/convert, infer
types & shapes, then lower *deliberately* and emit to host-language and accelerator
targets. (The `flatppl-js` reference engine is *eager* by contrast; two independent
implementations cross-validate.)

The pipeline:

```
FlatPPL / FlatPIR text  ──read──►  flatppl-core (one multi-level IR)
                                        │  infer types/shapes/phases (high level)
                                        ▼
                                   typed core
                                        │  deliberate, shape-directed lowering
                                        ▼
                              target-profile-conforming core  ──►  codegen / emit
```

## `flatppl-core` — one multi-level IR

`flatppl-core` is the single in-memory IR — the extended-FlatPIR model — and it is
**multi-level** (MLIR-style): it can hold high-level constructs (`metricsum`,
measures, distributions) *and* their lowered forms. **Lowering is deliberate, never
automatic on read.**

- **Graph:** arena + integer `NodeId`/`BindingId` indices (not a pointer graph) —
  rewrite-friendly and borrow-checker-friendly; standard for compiler IRs.
- **Nodes:** a small enum (`lit`/`const`/`ref`/`hole`/`axis`/`call`). Literals are a
  typed `Scalar` enum (`Int`/`Real`/`Bool`/`Str`/`Complex`). Nested array types are
  kept (vec-of-vec ≠ matrix, spec §03); `Dim = Static | Dynamic`.
- **IR-proper vs annotations:** structural fields live on nodes; analyses
  (type/shape/phase, spans, axis metadata) live in **side-tables keyed by
  `NodeId`/`BindingId`, owned by the producing pass**. Serialization is a projection
  of the IR-proper, dropping side-tables.
- **Identifiers:** interned `Symbol`s (cheap copy/eq).
- A module also carries the multi-module / bundle shape (`load_module`) and the
  query-module wrapper (model + pinned externals + requested outputs).

## Sugar vs construct boundary (the `flatppl-syntax` contract)

Reading FlatPPL removes **only syntactic sugar**; **named constructs are retained.**
Writing FlatPPL **re-applies** sugar — a *canonicalizing* pretty-printer, so
FlatPPL→core→FlatPPL is semantically faithful and canonically formatted (idempotent
after the first print), not byte-preserving. FlatPIR↔core round-trips are
structure-preserving.

The printer has two syntax levels (`Syntax` in `flatppl-syntax`; CLI
`--syntax`): **full** (the default) re-applies every sugar form below, with
precedence-aware parenthesization; **minimal** emits the spec §04 lowered
linear form (call syntax only, plus `~` and array/tuple literals). Sugar is
re-applied only where the re-parse provably inverts it — a shadowed built-in
prints through `base.`, `get` on a module binding never prints as dot access,
non-`sum` aggregates keep the call form — everything else falls back to call
syntax. The semantic oracle is FlatPIR equality across print→re-parse.
The same projection serves as the cross-engine oracle: the parity harness
(`crates/syntax/tests/cross_engine.rs`) lowers both engines' fixture corpora
through `flatppl-js` and this workspace and compares canonical FlatPIR
structurally — both sides derived live per run, never golden-baked.

- **Sugar (removed on read / re-applied on write) = surface notation with no
  distinct FlatPIR head:** operators (`+`→`add`), indexing/field access
  (`a[i]`→`get`, `a.f`→`get`), array/tuple literals (→`vector`/`tuple`),
  dot-broadcast (`f.(x)`, `a .+ b`→`broadcast`), `~` (→`draw`), `:=` / `metric: …`
  (→`aggregate` / `metricsum` calls), lambda/`fn`/holes (→`functionof`).
- **Constructs (retained) = anything with its own FlatPIR head:** `metricsum`,
  `aggregate`, `draw`, `functionof`, `lawof`, **`kernelof`** (retained — its
  equivalence to `functionof(lawof(…))` is a deliberate lowering *rule*, not
  desugaring), the measure ops, distributions.

## `flatppl-infer` — inference at the high level, before lowering

Type + shape + phase inference runs on the **high-level** (pre-lowering) IR, and is
a **prerequisite for lowering** (lowering is shape-directed — e.g. `MvNormal`→
`pushfwd` needs a static `D = length(mu)`).

- `flatppl-infer` **owns the per-construct shape catalogue** (the single source of
  per-distribution variate-shape knowledge); lowering passes are *generic* over the
  typed IR. Adding a distribution = one shape signature + per-backend impls; the
  lowering/routing stays invariant.
- Inference is **structure-preserving**: it does demand-driven const-eval at shape
  positions ("resolve, don't rewrite" — compute shape integers into a side table,
  never mutate or lower constructs). The shape-const-eval folds in here (no separate
  `minieval` crate).
- Includes cross-module inference (`load_module`).

## Lowering — rule catalog + target profiles + legalization (not fixed levels)

Lowering is **not** a fixed linear stack of levels — targets want *orthogonal*
combinations of constructs (measures-or-not × `metricsum`-or-`aggregate` ×
`MvNormal`-or-`pushfwd` × …). Instead, the MLIR `ConversionTarget` model:

- **A catalog of semantics-preserving rewrite rules**, each a `core → core`
  (FlatPIR→FlatPIR) transform eliminating/transforming specific constructs
  (`metricsum`→`aggregate`, `MvNormal`→`pushfwd(affine, iid)`,
  `kernelof`→`functionof∘lawof`, measure-elimination, …). These are the spec's
  normative equivalences made executable; each is individually testable. Rules carry
  **guards** (often over inferred shape).
- **Per-target/host profiles** = the legal construct set for a target (its "legal
  ops"). **FlatPDL** — the deterministic profile (measures eliminated; deterministic
  ops + the primitive basis + explicit RNG) — is the floor; HS³/RooFit, Stan, Julia
  are **sibling** higher profiles. It is a *lattice*, not a line.
- **A legalization driver** applies rules until the module conforms to the target's
  profile (validated by the profile-conformance checker).

Targets tap in at the level matching their native vocabulary:

| Target class | Examples | Profile consumed |
|---|---|---|
| Deterministic array math | MLIR/StableHLO, raw JAX | **FlatPDL** (fully lowered) |
| Probabilistically-native | Julia (MeasureBase/Distributions), Stan, RooFit/HS³, NumPyro/PyMC/Pyro PPL layers | higher, measure/distribution-bearing profiles |

So you **lower only as far as the target's native vocabulary** (MLIR's "legalize to
the target dialect"). The **precompiler** = *"legalize to the FlatPDL profile"* — one
driver over the shared catalog, not a bespoke monolith; other targets are other
drivers reusing the same rules.

### Two regimes over one catalog

The same rule catalog drives two different engines, chosen by the *nature* of the
transformation:

- **Directional legalization** — compilation-target lowering (e.g. → FlatPDL for
  MLIR). Rules *eliminate* constructs toward the target's legal set, so there is a
  decreasing measure (construct-complexity); a **greedy worklist / ordered pipeline
  terminates**, and *any* legal form is correct. This is classic instruction
  selection / dialect legalization — **no e-graph needed.** The `precompiler` and
  codegen lowering use this. (An e-graph would only buy the *optimal* lowering among
  choices — a quality concern, deferrable; correctness never needs it.)
- **Equality saturation** — cross-paradigm raise/lower + optimization (e.g.
  draw-based ↔ measure-algebra; (Num)Pyro/Stan ↔ HS³/RooFit). The equivalences are
  **bidirectional** (`kernelof`↔`functionof∘lawof`, draw-graph↔`jointchain`, …) with
  **no decreasing measure** (greedy loops), and reaching a paradigm often requires
  **raising**, not just lowering. This is the textbook **e-graph / egglog** case:
  saturate (hold all equivalent forms — no phase-ordering), then **extract** the
  lowest-cost form conforming to the target profile. Conformance becomes "did
  extraction find a form in the profile?"; raising is *partial* (a continuous
  marginal won't close into measure-algebra), and "no conforming extraction" is the
  clean failure signal. This is the `flatppl-rewrite` work.

Tag each catalog rule **directional** (lowering-only) vs **equivalence**
(bidirectional). **Start with the greedy legalizer** — it covers the `precompiler` +
all codegen lowering; **add the egglog engine for cross-paradigm interop**, the
ambitious later piece. (egglog is a superset — it could do directional lowering too,
with a cost model forbidding illegal ops — but greedy is simpler, faster, and
predictable for that path.)

## Profiles — one mechanism, several uses

A profile is the legal-construct set of a FlatPPL subset. The same mechanism +
**one conformance checker** serves: FlatPDL and other compilation/lowering targets;
remote **server capability contracts** (FlatPPL/full, /density, /rand, …); and the
legalization targets above.

## Crates

Landed: `flatppl-core` (the IR) · `flatppl-syntax` (FlatPPL surface ↔ core) ·
`flatppl-flatpir` (FlatPIR S-expr ↔ core) · `flatppl-infer` (the type/phase
trace + per-op rule catalogue) · `flatppl-lint` (lint rules over the IR) ·
`flatppl-cli` (the `flatppl` driver binary — `convert`, `infer`, `fmt`, `lint`).
`syntax`/`flatpir`/`infer` depend on `core`; `core` depends on nothing. Library
crates stay **binary-free** (they compile to `wasm32` and link into PyO3 / jlrs /
cxx); all CLI surface lives in `flatppl-cli`.

**CLI model.** One driver binary (`flatppl`) with subcommands; capabilities are
compile-time cargo features of `flatppl-cli` (a verb's crates link only when its
feature is on — lean default build, opt-in weight; `infer` is light and rides
in the default set). Verbs map to library crates: `convert` → syntax + flatpir;
`infer` → `flatppl-infer`; later `lower` / `check` → the rule catalog + profile
checker. The crate can host additional `[[bin]]`s later (gated by
`required-features`); a second tool with its own heavy dependency stack would
split into its own crate instead.

**Formatter + linter.** `flatppl-lint` (binary-free lint rules over the IR) plus
a thin `fmt` layer over the `flatppl-syntax` canonicalizing printer are surfaced
two ways. The full `flatppl` driver gains `fmt` and `lint` subcommands (behind the
default `fmtlint` feature). A **standalone `flatppl-fmt` binary** (subcommands
`fmt` / `lint`, `required-features = ["fmtlint"]`) is the lean CI/editor tool: it
links only `core` + `syntax` + `infer` + `lint`, **not** the converter
(`flatpir`/`hs3`), so `cargo build -p flatppl-cli --no-default-features
--features fmtlint --bin flatppl-fmt` produces a minimal binary. `flatppl-cli` is
therefore a **lib + two bins**: shared logic (`read_module`, `format_text`,
`run_fmt`, `run_lint`, diagnostics) lives in the lib; each converter verb is its
own cargo feature (`convert`/`infer`/`hs3`, all enabling `flatpir`). Lint rules:
`unused-binding`, `shadows-builtin`, `missing-doc`, `not-canonical`, plus an
`flatppl-infer` bridge (`unresolved-name` / `inference-cycle` / `inference-gap`);
severities are leveled (allow/warn/deny) with CLI overrides
(`--deny`/`--warn`/`--allow`/`--deny-warnings`) and a `% flatppl-lint: allow RULE`
file-level suppression directive. The formatter is zero-config and idempotent.

Planned (later phases): the lowering rule-catalog (possibly merged with a
rewrite/egglog crate), per-target codegen crates, remote/server, …

Conventions: crate names are hyphenated and prefixed (`flatppl-…`), singular
concepts; workspace directories drop the prefix (`crates/core`, `crates/syntax`);
edition 2024 / resolver 3, toolchain pinned to stable.

## Open / not-yet-locked

- **Pipeline-stage modeling:** type-state `Module<Stage>` (gates which annotation
  tables exist; makes stage-ordering errors unrepresentable) **vs** a single
  `Module` + optional annotation tables + debug asserts. *Leaning the latter for v1*;
  add type-state only if drift forces it.
- Lowering **driver flavour** (ordered pipelines vs automatic vs egglog) — start
  simple.
- Naming of the binary-FlatPIR/wire crate and whether `flatppl-batched` is a separate
  crate — deferred.

## Relation to the spec

The FlatPPL spec (`flatppl-design`) is the source of truth for semantics, FlatPIR
(§11), the measure algebra (§06), distributions (§08), and profiles (§12). This
document records how the Rust toolchain realizes that; where the two ever disagree,
the spec wins.
