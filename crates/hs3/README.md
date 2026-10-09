# flatppl-hs3 — HS3 / pyhf → FlatPPL importer

Import HS3 (HEP Statistics Serialization Standard) and pyhf JSON models into
[`flatppl-core`](../core), following the HS³/RooFit profile in flatppl-design §12. Import only.

* `read_hs3(json)` — native HS3 documents (`distributions`, `functions`, `domains`,
  `parameter_points`, `likelihoods`).
* `read_pyhf(json)` — pyhf models or workspaces (top-level `channels`).
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

Workspace imports still require observations for every channel. Integer-count
workspaces keep `Poisson`; fractional-count channels use `ContinuedPoisson`.

The full pyhf likelihood batches consecutive channels with the same count
distribution. It preserves parameter order and shared nuisance parameters.
Named channel models and likelihoods remain available for individual queries.

[`ORACLES.md`](ORACLES.md) holds one independently-derived checkpoint per HS3 kind — what to reach
for when an end-to-end score moves and you need to know which construct moved.
