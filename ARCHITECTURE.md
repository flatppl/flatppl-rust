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

### The value domain: const-eval and the interpreter

Shape const-eval (`crates/infer/src/consteval.rs`) is the **value domain** of the
inference trace (engine-concepts §17): the spec lets shapes depend on fixed-phase
values (`iid(M, lengthof(data))`, `zeros(sizeof(M))`), so pure-structural inference
is incomplete. It is three-valued (§17.1 "fixed-value boundary"): `Val` resolved,
`Dynamic` genuinely unknowable (non-fixed ancestor, `external`/`load_data`, a
`%dynamic`-dim observer) → legitimately `%dynamic`, and `Gap` — a fixed value-typed
op it cannot fold — a **loud diagnostic**, never a silent `%dynamic`. `Dynamic`
dominates `Gap`.

It is the **seed of the Phase-3 `flatppl-interpreter`, not a parallel walker** —
but they are **not one crate**, even long-term. Three parts, only one shared:
the **pure value-op core** (`FixedValue` + `eval_*`, no inference state) is what the
interpreter lifts and reuses; the **driver** walks the graph; the **shape observers**
(`lengthof`/`sizeof`) read the inferred TYPE (the §17.1 laziness short-circuit), so
they are inference-specific — the interpreter, holding real values, reads the value.
When the interpreter lands (the genuine *second* consumer) the pure core lifts to a
neutral leaf crate both depend on (leaning `flatppl-eval-core`; not `flatppl-core`,
which stays data-only). It stays a *separate* crate because (a) inference must stay
lean — it rides the default CLI feature set and the `fmt`/LSP builds, and must not
transitively link the sampler / RNG / `builtin_*` kernels; (b) codegen-first: the
interpreter is one **backend** among siblings (Stan / StableHLO), downstream of
inference, not part of the frontend. "One op registry, many drivers" — not one crate.

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
the target dialect"). The **determiniser** (crate `flatppl-determinizer`) = *"legalize to the FlatPDL profile"* — one
driver over the shared catalog, not a bespoke monolith; other targets are other
drivers reusing the same rules.

### Two regimes over one catalog

The same rule catalog drives two different engines, chosen by the *nature* of the
transformation:

- **Directional legalization** — compilation-target lowering (e.g. → FlatPDL for
  MLIR). Rules *eliminate* constructs toward the target's legal set, so there is a
  decreasing measure (construct-complexity); a **greedy worklist / ordered pipeline
  terminates**, and *any* legal form is correct. This is classic instruction
  selection / dialect legalization — **no e-graph needed.** The `flatppl-determinizer` and
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
(bidirectional). **Start with the greedy legalizer** — it covers the `flatppl-determinizer` +
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
`flatppl-fileaccess` (resolve a `source` to a local file) ·
`flatppl-cli` (the `flatppl` driver binary — `convert`, `infer`, `fmt`, `lint`) ·
`flatppl-lsp` (the FlatPPL language server binary).
`syntax`/`flatpir`/`infer` depend on `core`; `core` depends on nothing. Library
crates stay **binary-free** (they compile to `wasm32` and link into PyO3 / jlrs /
cxx); all CLI surface lives in `flatppl-cli`.

**`flatppl-fileaccess` — source resolution + remote cache.** A thin file-access
abstraction for the host layer: a `Location` (local path | `http`/`https` URL)
with relative-`join` resolution (spec §04 path resolution + the URL analogue),
and a `Resolver` that returns a local file path — local paths pass through,
remote URLs are fetched and cached per spec §sec:url-cache (content-addressed by
URL hash under `<flatppl-cachedir>/v1/`, per-URL trust markers, atomic
publishes, never revalidated). It does **no** file-format decoding and does not
parse FlatPPL. Network is a seam: fetching goes through a `Fetcher` trait and
the trust decision through a `TrustOracle`, so the cache logic is unit-testable
without a network; the real `ureq` client is behind the `net` feature. **Unlike
the other libraries it is native-only** (fs + optional network), not
wasm-targeted — URL loading in a browser/wasm context is the JS host's job — but
it stays binary-free.

**`Location` is the zero-dep core; the cache is a feature.** The §04 path/URL
`join` (`Location`) needs no dependencies and is always compiled. The whole pull
layer — `Cache`/`Resolver`/`Fetcher`/`TrustOracle` and its deps
(sha2/serde/dirs/ureq) — lives behind the default-on `cache` feature
(`net` ⊃ `cache`). A consumer that only *resolves* (notably the LSP, below)
depends with `default-features = false` and links nothing heavy: pure §04
resolution, no cache, no TLS.

**Wired into the CLI (`crates/cli/src/resolve.rs`), package-manager style.**
CLI inputs are always **local files** (no URLs) — a model may `load_module`
remote deps, but the command operates on a local path, like `cargo build` on a
local crate. Fetching is a separate, explicit step:

- **`flatppl prepare <file>… [--update]`** is the *only* network-touching verb. It
  BFS-walks each local model's transitive `load_module` (+ `load_data`) graph —
  reading + parsing each module to discover its deps — and downloads the
  `http`/`https` ones into the shared cache. Relative deps of a URL-loaded module
  resolve against that module's URL (`Location::join`). Trust is **batched per
  BFS level** (one interactive prompt per discovery wave; non-interactive
  refuses untrusted URLs). `--update` re-fetches cached URLs.
- **`convert` / `infer` are local and offline** — a cache-only resolver
  (`OfflineFetcher`, no HTTP client) resolves deps from the cache + local files;
  an uncached remote dep is an error pointing at `flatppl prepare`. `infer`
  assembles the inference `ModuleBundle` from the walk (engine stays I/O-free),
  keyed by the directive string the engine looks up (a string denoting two
  different files across the graph is a hard error); it discovers but does **not**
  resolve `load_data` (inference never reads data). `convert` is a single-file
  transform.

Feature isolation makes `infer` **net-free by construction**: it links
`flatppl-fileaccess` *without* `net` (no TLS); only `prepare` enables
`flatppl-fileaccess/net`; the lean `flatppl-fmt` links neither. The
**reads-only / writes-direct** split holds (source reads via `fileaccess`,
output writes are plain `fs::write`).

**The LSP resolves URL deps but never fetches.** It links only `Location`
(`fileaccess` with the `cache` feature off — no `Fetcher`/`Cache`/TLS) and routes
all `load_module`/`load_data` path resolution through it (`crates/lsp/src/queries.rs`
`resolve_path`), replacing a bespoke lexical normalizer. It cannot fetch: §sec:url-cache
requires interactive trust approval (non-interactive tooling must refuse), and the
LSP — non-interactive over the protocol, resolving inside a pure salsa query —
has no way to prompt; the editor client is already the sole fetcher+truster.
Instead, remote content arrives as a salsa **input**: the client pushes what it
fetched via a `flatppl/urlSources` notification (`{ sources: [{uri, text}] }`,
keyed by the resolved URL), which the server merges into the reactive `FileSet`
(under a URL key, kept out of the editor-buffer map, so URL deps resolve for
cross-module inference but get no diagnostics). This is the reactive counterpart
of the CLI's pull cache: same `Location` resolver, content-as-input instead of
content-fetched. The matching VS Code extension feed-hook is tracked in
`flatppl-dev/TODO-flatppl-js.md`.

**Binary tool crates** — `flatppl-cli` and `flatppl-lsp` — are the two exceptions
to the binary-free rule: they are standalone tools, never wasm-linked. Both ship a
`[[bin]]`; this is intentional and does not violate the library-crates rule.
`flatppl-lsp` additionally exposes a `[lib]` target so integration tests can drive
the server in-process without spawning a subprocess.

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

## Testing conventions

Distilled from the nested-data work, where coincidental test inputs hid real
bugs:

- **Exercise the discriminating case, not just the happy path.** Inference bugs
  hid because every test used the *valid, symmetric* shape — equal inner/outer
  table row counts hid a row-count reconstruction bug, and the test that caught
  it was the *unequal-columns* (ill-formed) case. Use distinct, non-coincidental
  magic numbers per axis so a swapped or mis-derived dimension must show, and
  always test the boundary / ill-formed input, not only the well-formed one.
- **Assert against the spec rule, not the current output.** A type / value-set
  assertion should encode what the spec mandates (cite the §); pinning whatever
  the engine emits today lets a wrong behaviour "pass" — the `column_elem`
  double-strip and the tuple / table value-sets all shipped that way.
- **The §11 value-set refinement invariant is guarded**
  (`crates/infer/tests/value_set_invariant.rs`): every value-typed node's
  value-set must be a subset of `natural_of(type)`, the canonical type→value-set
  mapping. Extend its corpus when you add a value-set producer, so a new one
  cannot silently drift from the natural extent.

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
