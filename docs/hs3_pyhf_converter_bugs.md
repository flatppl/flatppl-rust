# HS3 / pyhf converter bugs

Found 2026-06-17 converting the `pyhf-uncorrelated_background` rosetta example
(statsmodel-rosetta-stone, `src/physics/high-energy/binned/pyhf-uncorrelated_background/`).

Command shape:

```
flatppl convert --from {pyhf,hs3} <input>.json <output>.flatppl
```

The `--from pyhf` path ran without errors (only the cosmetic `joint_likelihood` type-rule
warning), declared its free params, and is now **numerically verified** (see below). It still
has metadata fidelity gaps (Bug 3). The native-HS3 `hl` input hit a hard bug (Bug 1).

**Verification status (updated 2026-06-17):** the `--from pyhf` output is numerically correct.
Scored by the flatppl-js engine against pyhf `logpdf` at 5 parameter points (the repro harness in
the rosetta example dir), it matches to ~1e-9 — e.g. `logL = -15.3876271732` at `μ=1, γ=(1,1)`,
identical to pyhf and to the hand-written `ma-auxm` variant. So Bug 3 is confirmed **metadata-only**
(domain/POI), not a density error. (Before Bug 1 was fixed, the HS3-`hl` output was never
produced.)

## What needs to be done (action checklist)

Fixes implemented on branch `converter-hs3-fixes` (TDD; full `flatppl-hs3` suite green).

- [x] **Bug 1** — `crates/hs3/src/histfactory.rs` `json_array`: unwrap the `{ "vals": [...] }`
      shapesys-data form. *Oracle: converted `hl` now scores identically to pyhf at all 5 points.*
- [x] **Bug 3a** — NOT changed (deliberate): spec §12:206 keeps a `normfactor`'s support as `reals`.
      pyhf's `[0, 10]` is a *fit domain*, not measure support, and it is not even present in the
      workspace JSON (`config.parameters: []`) — it's an out-of-band pyhf code default. Faithful
      to the JSON, so left as `reals`.
- [x] **Bug 3b** — `crates/hs3/src/pyhf.rs` `emit_poi`: emit `config.poi` as a record binding
      `<measurement> = record(poi = <param>)` (FlatPPL has no POI construct). Empty-string poi
      skipped. *Density unchanged (still matches pyhf).*
- [x] **Bug 4** — `crates/hs3/src/convert.rs` `emit_histfactory_channels`: HS3 `shapesys` `vals` are
      RELATIVE uncertainties (RooFit/HS3 convention), so the native-HS3 path now scales them by the
      sample nominal (σ_abs = vals × nominal) before the shared assembler; the pyhf path keeps
      absolute vals untouched. *Oracle: converted `hl` now matches ROOT(hl) Δ(logL) exactly.* This
      is the convention split between the HS3 and pyhf paths.

(ROOT cannot load `hs3-ll.json` either, but that is an HS3-file encoding issue — undeclared
observables, missing `poisson_dist` `integer` flag, inline likelihood data — **not** a converter
bug, and out of scope here.)

## Bug 1 — `shapesys` `data` object form rejected (hard fail)

**Input:** `hs3-uncorrelated_background-hl.json`, with

```json
"data": { "vals": [3.0, 7.0] }
```

**Error:**

```
hs3: unsupported HS3 construct: shapesys `uncorr_bkguncrt` data: expected a JSON array of numbers
```

Importer only accepts a bare array (`"data": [3.0, 7.0]`). The `{ "vals": [...] }` wrapper is
valid HS3 — accept both forms. No output is produced on failure.

**What needs to be done:** the error is raised by `json_array` at
`crates/hs3/src/histfactory.rs:26`, called for shapesys data at `histfactory.rs:272`. Unwrap a
`{ "vals": [...] }` object to its inner array before the array check (either inside `json_array`
or at the shapesys call site). RooFit accepts both forms, so matching that is the target.

## Bug 3 — pyhf path drops parameter domain + POI

`--from pyhf` emits:

```
mu = elementof(reals)
uncorr_bkguncrt = elementof(cartpow(posreals, 2))
```

- **`mu` domain lost.** pyhf `normfactor` default bounds are `[0, 10]` (init 1.0); the converter
  emits unbounded `reals`, losing both bounds. Note the HS3-ll path *does* capture
  `interval(0.0, 10.0)` for `mu` — so the pyhf path is strictly worse here.
- **POI lost.** Source `measurements[].config.poi = "mu"` is not represented in the output; nothing
  marks `mu` as the parameter of interest.

The likelihood *structure* (obs `Poisson` + `ContinuedPoisson` aux, τ = (b/δb)²) is numerically
correct (verification note above); these are purely metadata losses.

**What needs to be done:** in `crates/hs3/src/pyhf.rs` (free-param declaration, ~line 250), when a
parameter is a `normfactor`, carry its bounds into the emitted domain — default pyhf `normfactor`
bounds are `[0, 10]`, so emit `mu = elementof(interval(0.0, 10.0))` rather than `elementof(reals)`
— and represent the `measurements[].config.poi` (mark `mu` as parameter-of-interest) so it survives
the conversion.
