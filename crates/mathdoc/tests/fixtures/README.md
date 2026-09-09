# Corpus fixtures for the math renderer

Resilience copies (flatppl-dev `CONVENTIONS.md`, "Examples and test fixtures"):
the user-facing models of `flatppl-examples@a3f71d0` and, from
`statsmodel-rosetta-stone@e1812a7`, the HS3 reference conversions `gaussian`,
`histfactory` and `product`. (That repo's `resonance` and `phase-space` modules
are not copied: as of that commit neither parses under spec §05 — one has two
`%` lines before a binding, the other a `;` inside a `#` comment.) The sweep in
`../corpus.rs` renders every file here and every fixture under the repo's
`fixtures/flatppl/` and fails on any row that fell back to source text.
Refresh a copy from upstream when an example changes; cite the upstream commit.
