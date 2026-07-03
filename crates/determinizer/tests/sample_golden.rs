use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

// rand(rng, lawof(record(x = draw(Normal)))) samples the one draw via
// builtin_sample, threads the rng, and eliminates the measure/stochastic layer.
#[test]
fn single_draw_samples_via_builtin_sample() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-draw rand must lower to builtin_sample");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_sample"),
        "emits builtin_sample:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_sample").count(),
        1,
        "one sample per draw:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("(lawof ") && !pir.contains("(rand "),
        "measure/sample-surface layer eliminated:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
    // Strengthened per the {pir} dump: the rng is threaded in as-is (no fresh
    // rngstate is fabricated), the constructor symbol survives bare (`Normal`,
    // not re-wrapped), and the sampled value is read out via `get0(sample, 0)`
    // (there is no separate `get1` primitive in this codebase — see
    // `sample::build_sample_term`).
    assert!(
        pir.contains("(builtin_sample (%ref self s) Normal ("),
        "builtin_sample threads the rng and carries the bare Normal ctor:\n{pir}"
    );
    assert!(
        pir.contains("(get0"),
        "sampled value is projected via get0(sample, 0):\n{pir}"
    );
    // The draw-binding `x` is now dead (the sampled value is a fresh inline
    // node, not a ref to `x`) and swept by `sweep_dead_measure_bindings`.
    assert!(
        pir.contains("(%bind x 0.0)"),
        "orphaned draw binding x is swept to a harmless literal:\n{pir}"
    );
}
