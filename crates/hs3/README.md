# flatppl-hs3 — HS3 / pyhf → FlatPPL importer

Import HS3 (HEP Statistics Serialization Standard) and pyhf JSON models into
[`flatppl-core`](../core), following the HS³/RooFit profile in flatppl-design §12. Import only.

* `read_hs3(json)` — native HS3 documents (`distributions`, `functions`, `domains`,
  `parameter_points`, `likelihoods`).
* `read_pyhf(json)` — pyhf models or workspaces (top-level `channels`).
* `read_pyhf_with(json, &PyhfOptions)` — the same, with a selected measurement,
  external observations or external constraint centres.
* `read(json)` — dispatch on the `channels` key.

Covers most of the HS3 distribution catalogue, the `functions` block, and histfactory (both pyhf
`channels` and native `histfactory_dist`: normfactor / shapesys / normsys / histosys / lumi /
staterror / shapefactor). Out-of-scope constructs fail loud rather than mis-convert, and every
emitted module is re-parsed to validate it.

Drives the `flatppl convert --from hs3|pyhf` verb in [`flatppl-cli`](../cli).

Model-only JSON accepts `channels` and optional top-level `parameters`. It uses
the same modifier and auxiliary-constraint lowering as workspace JSON. Each
channel exposes an observed-count input, for example
`singlechannel_observed = external(cartpow(nonnegreals, 2))`.
Supply it through the host or an ordinary module load:

```flatppl
model = load_module("model.flatppl", singlechannel_observed = [5.0, 10.0])
```

These inputs accept integer or fractional counts, so model-only likelihoods use
`hepphys.ContinuedPoisson`. This is a density extension, not a sampling distribution.
Auxiliary observations use the modifier defaults or configured `auxdata`, just
as workspace imports do. Fit settings (`inits`, `bounds`, `fixed`) do not change
the likelihood. Model-only input declares no parameter of interest.

Workspace imports with bound observations require observations for every channel.
Integer-count workspaces keep `Poisson`; fractional-count channels use `ContinuedPoisson`.

`PyhfOptions` changes three things. `measurement` selects a workspace measurement
by name, where pyhf takes the first. `Observations::External` leaves a workspace's
observed counts as external inputs, like a model-only document, with
`ContinuedPoisson`. `Auxiliary::External` makes each constraint's centre a field
of one record external, for example
`pyhf_auxdata = external(cartprod(alpha = reals, gamma = cartpow(nonnegreals, 2)))`.
Its fields follow pyhf's `auxdata_order`, so its flat layout is pyhf's
`config.auxdata`. Gaussian centres are reals and Poisson centres are non-negative
reals. The nominal values are the settings record's `auxdata`. The CLI flags are
`--measurement NAME`, `--observations external` and `--auxiliary external` on
`flatppl convert --from pyhf`.

Each workspace measurement becomes a literal record named
`pyhf_measurement_<name>`, with the name sanitized like a channel name. A
model-only document gets one `pyhf_measurement`. No query reads the record, so
it never changes emitted StableHLO. Its fields:

* `name`, `selected` (the measurement the likelihood uses), and `poi`, the
  emitted parameter name as a string. `poi` is absent when none is declared.
* `parameters`, in pyhf's registration order (`config.par_order`). Each entry
  has `pyhf` (the original name), `init`, `lo`, `hi` and `fixed`, and for a
  constrained parameter `width` and nominal `auxdata`. pyhf's per-modifier
  defaults apply first, then the measurement's `inits`, `bounds`, `fixed`,
  `auxdata`, `sigmas` and `factors`. Scalar parameters have scalar values,
  vector parameters one value per component.
* `channels`: per channel, `name`, `bins`, and the `observed` and `expected`
  binding names.

The full pyhf likelihood batches consecutive channels with the same count
distribution. It preserves parameter order and shared nuisance parameters.
Named channel models and likelihoods remain available for individual queries.

[`ORACLES.md`](ORACLES.md) holds one independently-derived checkpoint per HS3 kind — what to reach
for when an end-to-end score moves and you need to know which construct moved.
