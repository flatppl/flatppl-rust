# flatppl-hs3 — HS3 / pyhf → FlatPPL importer

Import HS3 (HEP Statistics Serialization Standard) and pyhf JSON models into
[`flatppl-core`](../core), following the HS³/RooFit profile in flatppl-design §12. Import only.

* `read_hs3(json)` — native HS3 documents (`distributions`, `functions`, `domains`,
  `parameter_points`, `likelihoods`).
* `read_pyhf(json)` — pyhf workspaces (top-level `channels`).
* `read(json)` — dispatch on the `channels` key.

Covers most of the HS3 distribution catalogue, the `functions` block, and histfactory (both pyhf
`channels` and native `histfactory_dist`: normfactor / shapesys / normsys / histosys / lumi /
staterror / shapefactor). Out-of-scope constructs fail loud rather than mis-convert, and every
emitted module is re-parsed to validate it.

Drives the `flatppl convert --from hs3|pyhf` verb in [`flatppl-cli`](../cli).
