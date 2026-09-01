// `PoissonProcess(intensity)` — spec §08's extended unbinned likelihood
// `Σ_j log λ(x_j) − Λ(R)` — and the limits it refuses instead of mis-lowering.

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

// §06's own superposed-intensity idiom: a latent-weighted signal shape over a
// flat background, scored at three observed events. The density is
// `Σ_j logsumexp([log s + log sig(x_j), log b + log bkg(x_j)]) − (s·1 + b·1)`,
// so the emitted form must carry one logsumexp per event and no measure layer.
#[test]
fn superposed_intensity_lowers_to_extended_likelihood() {
    let src = "\
window = interval(0.0, 10.0)
obs = [1.0, 5.0, 9.0]
s = elementof(posreals)
b = elementof(posreals)
sig = normalize(truncate(Normal(mu = 5.0, sigma = 0.8), window))
bkg = weighted(0.1, Lebesgue(support = window))
intensity = superpose(weighted(s, sig), weighted(b, bkg))
events ~ PoissonProcess(intensity = intensity)
lp = logdensityof(lawof(record(events = events)), record(events = obs))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    // One per-event logsumexp over the two intensity components.
    assert_eq!(
        pir.matches("logsumexp").count(),
        3,
        "one logsumexp per observed event:\n{pir}"
    );
    // The events are read off the variate positionally, one `get0` per event per
    // component gate (the truncate gate re-reads its own point).
    assert!(
        pir.contains("get0"),
        "events indexed off the variate:\n{pir}"
    );
    // `totalmass` is a measure query and must never be emitted; the total mass
    // is synthesized as `s · 1.0 + b · 1.0` instead.
    assert!(
        !pir.contains("totalmass"),
        "totalmass must not be emitted:\n{pir}"
    );
    assert!(
        !pir.contains("PoissonProcess")
            && !pir.contains("(superpose ")
            && !pir.contains("(Lebesgue ")
            && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// The `Λ` subtraction is what separates an extended likelihood from a plain
// product of intensities. A single-component `weighted(n, shape)` intensity —
// the HS3 `rate_extended_dist` shape — puts it in reach of an exact check: the
// mass is `n · 1.0` and appears under one `sub`.
#[test]
fn total_mass_of_a_weighted_shape_is_the_rate() {
    let src = "\
obs = [0.5]
n = elementof(posreals)
shape_dist = Normal(mu = 0.0, sigma = 1.0)
intensity = weighted(n, shape_dist)
events ~ PoissonProcess(intensity)
lp = logdensityof(lawof(record(events = events)), record(events = obs))";
    let out = determinize_src(src);
    let text = flatppl_syntax::print(&out);
    assert!(
        text.contains("n * 1.0"),
        "Λ = n · totalmass(shape) = n · 1.0:\n{text}"
    );
    assert!(
        text.contains(" - "),
        "the mass is SUBTRACTED from the events term:\n{text}"
    );
}

// A `Lebesgue` over a bounded interval has the closed-form mass `hi − lo`; the
// enclosing `weighted` scales it. The weight and the endpoints are all literals
// here, so canonicalization folds `0.3 · (4.0 − 0.0)` to the `1.2` asserted
// below — deliberately NOT a product that folds to 1, which would pass even if
// the interval length were dropped for a unit-mass shortcut.
#[test]
fn bounded_lebesgue_mass_is_the_interval_length() {
    let src = "\
window = interval(0.0, 4.0)
obs = [2.0]
b = elementof(posreals)
intensity = weighted(b, weighted(0.3, Lebesgue(support = window)))
events ~ PoissonProcess(intensity = intensity)
lp = logdensityof(lawof(record(events = events)), record(events = obs))";
    let out = determinize_src(src);
    let text = flatppl_syntax::print(&out);
    assert!(
        text.contains("b * 1.2"),
        "Λ = b · (0.3 · 4.0) = 1.2·b:\n{text}"
    );
}

// An UNBOUNDED intensity support has infinite mass, so §08's finite-mass
// precondition fails and there is no `Λ` to emit. Refuse rather than emit an
// infinity.
#[test]
fn unbounded_intensity_support_refuses() {
    let src = "\
obs = [1.0]
b = elementof(posreals)
intensity = weighted(b, Lebesgue(support = reals))
events ~ PoissonProcess(intensity = intensity)
lp = logdensityof(lawof(record(events = events)), record(events = obs))";
    let err = determinize(&parse_infer(src)).expect_err("infinite intensity mass must refuse");
    assert_eq!(err.construct, "PoissonProcess", "refusal names it: {err:?}");
    assert!(
        err.reason.contains("closed-form total mass"),
        "refusal names the missing mass rule: {err:?}"
    );
}

// The per-event sum is unrolled over the VARIATE's own length, so a variate whose
// length is not statically known has nothing to unroll over. Refuse rather than
// score a prefix.
#[test]
fn dynamic_length_variate_refuses() {
    let src = "\
obs = elementof(cartpow(reals, n))
n = 4
intensity = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
events ~ PoissonProcess(intensity = intensity)
lp = logdensityof(lawof(record(events = events)), record(events = sort(obs)))";
    let m = parse_infer(src);
    // Either inference resolves the length (then this lowers) or it does not
    // (then it refuses naming the variate limit) — never a silent partial score.
    match determinize(&m) {
        Ok(out) => {
            let pir = flatppl_flatpir::write(&out);
            assert!(
                !pir.contains("PoissonProcess"),
                "if it lowers, the measure layer is gone:\n{pir}"
            );
        }
        Err(err) => {
            assert_eq!(err.construct, "PoissonProcess", "{err:?}");
            assert!(
                err.reason.contains("statically-known length"),
                "refusal names the variate-length limit: {err:?}"
            );
        }
    }
}

// §08 admits record-valued points (a table variate), whose rows `get0` does not
// index. That form is out of the implemented fragment and must refuse.
#[test]
fn record_valued_points_refuse() {
    let src = "\
obs = table(x = [1.0, 2.0], y = [3.0, 4.0])
intensity = weighted(2.0, joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 0.0, sigma = 1.0)))
events ~ PoissonProcess(intensity = intensity)
lp = logdensityof(lawof(record(events = events)), record(events = obs))";
    let err = determinize(&parse_infer(src)).expect_err("a table variate must refuse");
    assert!(
        err.reason.contains("statically-known length")
            || err.reason.contains("closed-form total mass"),
        "refusal names an implemented limit: {err:?}"
    );
}

// `Lebesgue`'s own density is the support gate: 0 inside, −inf outside (§06
// "the `support` parameter specifies where the measure is nonzero; density is
// zero outside"). It must lower rather than leave a
// `builtin_logdensityof(Lebesgue, …)` leaf for a backend to implement.
#[test]
fn lebesgue_density_lowers_to_the_support_gate() {
    let src = "\
window = interval(0.0, 10.0)
m = Lebesgue(support = window)
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 2.0))";
    let out = determinize_src(src);
    let text = flatppl_syntax::print(&out);
    assert!(
        text.contains("in window") && text.contains("-inf"),
        "the density is the `in`-gated constant:\n{text}"
    );
    assert!(
        !text.contains("builtin_logdensityof"),
        "no residual Lebesgue density leaf:\n{text}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{text}"
    );
}
