//! Type/shape inference for the FlatPDL measure-kernel evaluation primitives
//! (spec §07 §sec:measure-eval-prims). Type-level only — flatppl-rust does not
//! evaluate them; this checks the result types the converter emits type-check.

use flatppl_infer::infer;

fn infer_src(src: &str) -> (flatppl_core::Module, Vec<flatppl_infer::Diagnostic>) {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut module);
    (module, diags)
}

/// The trimmed FlatPIR line containing `needle` (its `%meta` annotation).
fn meta_of(src: &str, needle: &str) -> String {
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    out.lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line containing `{needle}` in:\n{out}"))
        .trim()
        .to_string()
}

#[test]
fn builtin_logdensityof_is_real_scalar() {
    let m = meta_of(
        "lp = builtin_logdensityof(Normal, record(mu = 0.0, sigma = 1.0), 0.0)",
        "builtin_logdensityof",
    );
    // The RESULT meta wrapping the call — `(%meta ((%scalar real) …) (builtin_logdensityof …))`
    // — must be a real scalar (not the nested record args' `(%scalar real)`).
    assert!(
        m.contains("(%meta ((%scalar real)"),
        "builtin_logdensityof result must type as a real scalar; got: {m}"
    );
    // The primitive absorbs stochasticity — it is not a stochastic seed (only
    // `draw` is, spec §04 Phases). Fixed args ⇒ not %stochastic.
    assert!(
        !m.contains("%stochastic"),
        "builtin_logdensityof must not be stochastic-phase; got: {m}"
    );
}

#[test]
fn builtin_sample_is_variate_rngstate_tuple() {
    let m = meta_of(
        "state = rnginit(0)\nxs, s2 = builtin_sample(state, Normal, record(mu = 0.0, sigma = 1.0))",
        "builtin_sample",
    );
    // Result meta is a (variate, new_rngstate) tuple.
    assert!(
        m.contains("(%meta ((%tuple") && m.contains("%rngstate"),
        "builtin_sample result must be a (variate, rngstate) tuple; got: {m}"
    );
    // The Normal kernel's variate is a real scalar.
    assert!(
        m.contains("(%tuple (%scalar real) %rngstate)"),
        "builtin_sample variate (Normal) must be a real scalar; got: {m}"
    );
}

#[test]
fn builtin_sample_mvnormal_variate_is_vector() {
    let m = meta_of(
        "state = rnginit(0)\nmu = [0.0, 0.0, 0.0]\ncov = eye(3)\nxs, s2 = builtin_sample(state, MvNormal, record(mu = mu, cov = cov))",
        "builtin_sample",
    );
    // variate is a length-3 real vector (dim read from kernel_input.mu). Assert the
    // RESULT tuple (the line also holds the nested `(mu (%array 1 (3) …))` arg).
    assert!(
        m.contains("(%tuple (%array 1 (3) (%scalar real)) %rngstate)"),
        "MvNormal sample variate must be array[3] of real in the result tuple; got: {m}"
    );
}

#[test]
fn builtin_sample_wishart_variate_is_matrix() {
    let m = meta_of(
        "state = rnginit(0)\nsc = eye(2)\nxs, s2 = builtin_sample(state, Wishart, record(nu = 3.0, scale = sc))",
        "builtin_sample",
    );
    // DynMatrix path: array[Dynamic, Dynamic] of real (dims NOT from the record).
    assert!(
        m.contains("(%tuple (%array 2 (%dynamic %dynamic) (%scalar real)) %rngstate)"),
        "Wishart sample variate must be array[Dynamic,Dynamic] of real in the result tuple; got: {m}"
    );
}

#[test]
fn builtin_sample_hepphys_argus_variate() {
    let m = meta_of(
        "hepphys = standard_module(\"particle-physics\", \"0.1\")\nstate = rnginit(0)\nxs, s2 = builtin_sample(state, hepphys.Argus, record(chi = 1.0, c = 2.0, p = 0.5))",
        "builtin_sample",
    );
    // Argus is a scalar-real-variate distribution (particle-physics.ron).
    assert!(
        m.contains("(%tuple (%scalar real) %rngstate)"),
        "hepphys.Argus sample variate must be a real scalar in the result tuple; got: {m}"
    );
}

#[test]
fn builtin_transports_are_variate_typed() {
    for op in [
        "builtin_touniform",
        "builtin_fromuniform",
        "builtin_tonormal",
        "builtin_fromnormal",
    ] {
        let src = format!("u = {op}(Normal, record(mu = 0.0, sigma = 1.0), 0.0)");
        let m = meta_of(&src, op);
        // The RESULT meta (the one wrapping the op call) is Normal's variate: real scalar.
        assert!(
            m.contains(&format!("(%scalar real) %fixed reals) ({op} ")),
            "{op} of a univariate continuous kernel must type as a real scalar; got: {m}"
        );
    }
}

#[test]
fn builtin_sample_non_kernel_diagnoses() {
    // A resolved-but-non-kernel `kernel` arg (here `3.0`) is a static error, not a
    // silent `%deferred`.
    let (_module, diags) =
        infer_src("state = rnginit(0)\nxs, s2 = builtin_sample(state, 3.0, record(x = 0.0))");
    assert!(
        diags
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error
                && d.message.contains("builtin_sample")),
        "a non-kernel kernel arg must emit an error diagnostic; got: {diags:?}"
    );
}
