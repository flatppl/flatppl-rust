//! `densityof(M, x)` lowering (spec §06: `densityof` is the density,
//! `logdensityof` its log — `densityof(M, x) = exp(logdensityof(M, x))`).
//!
//! FlatPPL has no `builtin_densityof` primitive (§07 lists six `builtin_*`
//! primitives, and `builtin_logdensityof` is the only density one), so
//! `densityof` must lower by reusing the existing `logdensityof` lowering and
//! wrapping the result in `exp(...)`.
//!
//! Pins three properties:
//! - `densityof_lowers_to_exp_of_logdensity` — the emitted FlatPDL wraps a
//!   single `builtin_logdensityof` term in `exp(`.
//! - `densityof_query_over_draw_binding_does_not_refuse_citing_draw` — the
//!   two-pass driver probe (`find_measure_node`) must fire for `densityof` the
//!   same way it does for `logdensityof`, so a `draw` binding consumed by a
//!   `densityof` query is legalised through the query rather than reached (and
//!   refused) by the general source-order scan first.
//! - `densityof_refuses_when_logdensityof_would` — `densityof` propagates the
//!   same refusal as `logdensityof` for a shape neither can lower (a
//!   non-conjugate continuous-latent `kchain`), rather than silently
//!   succeeding or emitting a partial density.

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// A self-contained `densityof` query over an independent-record-of-draws
/// prior (the same shape the `logdensityof` Task-3 goldens use).
const DENSITYOF_OVER_DRAW: &str = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
d = densityof(lawof(record(a = a)), record(a = 0.5))";

#[test]
fn densityof_lowers_to_exp_of_logdensity() {
    let out = determinize(&parse_infer(DENSITYOF_OVER_DRAW))
        .expect("densityof over a record-of-draws prior must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "expected exactly one builtin_logdensityof term; got:\n{pir}"
    );
    // FlatPIR prints calls in prefix S-expression form (`(exp ...)`, not
    // `exp(...)`), so look for the prefix form.
    assert!(
        pir.contains("(exp "),
        "densityof must lower to exp(<logdensity>); got:\n{pir}"
    );
    // The `(exp ` must actually wrap the density term, not merely coexist with
    // it elsewhere in the module.
    let exp_pos = pir.find("(exp ").expect("(exp  present (checked above)");
    let density_pos = pir
        .find("builtin_logdensityof")
        .expect("builtin_logdensityof present (checked above)");
    assert!(
        exp_pos < density_pos,
        "expected `(exp ` to wrap the builtin_logdensityof term; got:\n{pir}"
    );
}

#[test]
fn densityof_query_over_draw_binding_does_not_refuse_citing_draw() {
    // Regression: before the driver's priority probe recognised `densityof`,
    // this refused citing the bare `draw` binding (the general source-order
    // scan reached `a = draw(...)` before the query that would have legalised
    // it). It must lower (Ok), exactly as the equivalent `logdensityof` query
    // does.
    let err = determinize(&parse_infer(DENSITYOF_OVER_DRAW));
    assert!(
        err.is_ok(),
        "densityof over a draw binding must not refuse citing `draw`; got: {:?}",
        err.err().map(|e| format!("{e:?}"))
    );
}

/// The `determinize_refuses_with_exit_3` CLI shape (a continuous latent
/// feeding the likelihood SCALE — non-conjugate, no closed-form marginal),
/// wrapped in `densityof` instead of `logdensityof`.
const DENSITYOF_NONCONJUGATE_KCHAIN: &str = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(y = draw(Normal(mu = 1.0, sigma = z))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = densityof(pp, record(y = 0.5))";

#[test]
fn densityof_refuses_when_logdensityof_would() {
    let err = determinize(&parse_infer(DENSITYOF_NONCONJUGATE_KCHAIN))
        .expect_err("a non-conjugate continuous-latent kchain must refuse under densityof too");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("kchain") || msg.contains("kernel") || msg.contains("marginal"),
        "expected the refusal to name the non-conjugate kchain marginal; got: {msg}"
    );
}
