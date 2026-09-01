//! The StableHLO end of `PoissonProcess`'s extended-likelihood lowering.
//!
//! The determiniser leaves no `PoissonProcess` tag behind — the density becomes
//! `get0`/`logsumexp`/`add`/`sub` arithmetic over `builtin_logdensityof` leaves
//! plus `in`-gated constants — so the emitter needs no registry entry for it.
//! What the emitter DOES need is the support gate over a NAMED set, which the
//! `Lebesgue` and `truncate` lowerings both emit as `in(v, window)`.

use flatppl_stablehlo::{Mode, emit};

fn emit_logdensity(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "inference errors: {errors:?}");
    let d = flatppl_determinizer::determinize(&m).expect("must lower, not refuse");
    emit(&d, Mode::LogDensity, &Default::default()).expect("must emit")
}

/// A support set is normally a named binding, so `in(v, window)` carries a
/// `(%ref self window)` rather than the `interval` call. The emitter resolves one
/// ref level; before it did, this emitted "'in': only an interval(lo, hi),
/// posreals or nonnegreals set is supported".
#[test]
fn a_named_interval_support_set_emits() {
    let src = "\
flatppl_compat = \"0.1\"
window = interval(0.0, 10.0)
t = normalize(truncate(Normal(mu = 5.0, sigma = 0.8), window))
a = draw(t)
t_a = elementof(reals)
inputs = (t_a)
outputs = logdensityof(lawof(record(a = a)), record(a = t_a))";
    let out = emit_logdensity(src);
    // The gate lowers to the two interval comparisons ANDed together.
    assert!(
        out.contains("stablehlo.compare GE") && out.contains("stablehlo.compare LE"),
        "the named interval lowers to its two bound comparisons:\n{out}"
    );
}

/// §06's superposed-intensity idiom, end to end: an extended unbinned likelihood
/// over a latent-weighted signal plus a flat background emits with no residual
/// measure-layer symbol.
#[test]
fn superposed_intensity_extended_likelihood_emits() {
    let src = "\
flatppl_compat = \"0.1\"
window = interval(0.0, 10.0)
obs = [1.0, 5.0, 9.0]
s ~ Gamma(shape = 5.0, rate = 0.5)
b ~ Gamma(shape = 8.0, rate = 0.5)
sig = normalize(truncate(Normal(mu = 5.0, sigma = 0.8), window))
bkg = weighted(0.1, Lebesgue(support = window))
intensity = superpose(weighted(s, sig), weighted(b, bkg))
events ~ PoissonProcess(intensity = intensity)
L = likelihoodof(kernelof(record(events = events), s = s, b = b), record(events = obs))
posterior = bayesupdate(L, lawof(record(s = s, b = b)))
t_s = elementof(reals)
t_b = elementof(reals)
inputs = (t_s, t_b)
outputs = logdensityof(posterior, record(s = t_s, b = t_b))";
    let out = emit_logdensity(src);
    assert!(
        !out.contains("PoissonProcess") && !out.contains("Lebesgue"),
        "no measure-layer symbol reaches the emitter:\n{out}"
    );
    assert!(
        out.contains("func.func @logdensity"),
        "the logdensity entry is emitted:\n{out}"
    );
}
