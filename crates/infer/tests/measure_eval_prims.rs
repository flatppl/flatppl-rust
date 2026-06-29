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
