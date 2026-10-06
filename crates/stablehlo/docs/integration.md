# Numerical scalar integration

Numerical integration is an explicit StableHLO extension. Ordinary FlatPDL
conformance and default emission still require exact density lowering.

```rust
use flatppl_stablehlo::{EmitOptions, IntegrationOptions};

let options = EmitOptions {
    integration: Some(IntegrationOptions {
        rtol: 1e-5,
        atol: 0.0,
        max_intervals: 128,
    }),
    ..EmitOptions::default()
};
let lowering = options.lowering_options();
```

Pass `lowering` to the determinizer and `options` to the emitter. The host crate
does this through `LoadedModule::compile`. The CLI exposes `--numerical-integrals`,
`--integration-rtol`, `--integration-atol`, and `--integration-max-intervals`.

The determinizer runs exact rules first. The fallback integrates one named,
continuous scalar latent with an ancestor-free prior, or a positive scalar
normalizer. It admits known continuous constructors, not arbitrary measures
whose support happens to be real. Its private log-integral and bound-coordinate
nodes retain lexical scope through canonicalization and conformance checks.

The emitter uses adaptive Gauss 7 / Kronrod 15 quadrature with QUADPACK's
variation-based error correction and floating-point error floor. It splits the
panel with the largest estimated error. Finite, half-infinite and full-infinite
intervals map to the unit interval. Panel masses and errors stay in log space.
The stopping rule is `error <= max(atol, rtol * integral)` before taking logs.

Both tolerances must be finite and nonnegative, with at least one positive.
`max_intervals` must lie in `2..=i32::MAX` and fixes buffer sizes. Failure to meet
the estimated tolerance, invalid evaluations or interval collapse returns NaN.
A zero-mass normalizer also returns NaN. The error estimate is not a bound:
unresolved peaks and discontinuities can evade all evaluation nodes.

Enzyme compatibility mode uses the adaptive loop only to select a dyadic mesh.
It reevaluates the selected panels outside the loop for differentiation, while
keeping physical bounds live. This avoids Enzyme's adaptive-loop tape limits.
Unused panel slots are masked. The primal tolerance does not bound gradient error.

Derivatives require smooth integrands. Moving interior discontinuities need
explicit splitting, which is not implemented. Nested integrals, discrete or
mixed measures, explicit nonconjugate `kchain` integration, and integrals inside
FlatPPL broadcasts remain unsupported. Some composed truncations still require
an exact mass rule.

References: [Quadax differentiation](https://quadax.readthedocs.io/en/stable/differentiation.html),
[QUADPACK DQK15](https://www.netlib.org/quadpack/dqk15.f).
