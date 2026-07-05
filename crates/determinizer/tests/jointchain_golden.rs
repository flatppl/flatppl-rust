use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

// jointchain(M, K) keeps BOTH variates; its density is the product of
// conditionals (spec §06, no marginalization). Base `a ~ Normal(0,1)`, kernel
// `b ~ Normal(mu = a, sigma = 0.5)`. At the point {a: 0.3, b: 0.7} the density
// is logdensityof(Normal(0,1), 0.3) + logdensityof(Normal(0.3, 0.5), 0.7):
// exactly TWO builtin_logdensityof terms summed, no measure layer.
#[test]
fn jointchain_record_single_step() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)
j = jointchain(lawof(record(a = a)), k)
lp = logdensityof(j, record(a = 0.3, b = 0.7))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two conditional density terms:\n{pir}"
    );
    assert!(
        pir.contains("(builtin_logdensityof Normal "),
        "both terms Normal:\n{pir}"
    );
    // Base mean 0.0, scored at a=0.3.
    assert!(pir.contains("(%field mu 0.0)"), "base mean 0.0:\n{pir}");
    // Kernel mean is the realized a-slice 0.3 (input `a` bound to point.a).
    assert!(
        pir.contains("(%field mu 0.3)"),
        "kernel mean = realized a = 0.3:\n{pir}"
    );
    // Kernel scored at b = 0.7.
    assert!(pir.contains(" 0.7)"), "kernel scored at b = 0.7:\n{pir}");
    assert!(
        !pir.contains("jointchain")
            && !pir.contains("kernelof")
            && !pir.contains("lawof")
            && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
}

// Keyword-form jointchain (named components) carries relabel semantics not yet
// lowered — refuse, don't mislower.
#[test]
fn jointchain_keyword_form_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)
j = jointchain(prior = lawof(record(a = a)), fwd = k)
lp = logdensityof(j, record(prior = record(a = 0.3), fwd = record(b = 0.7)))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("keyword-form jointchain must refuse");
    assert!(
        err.construct.contains("jointchain"),
        "names jointchain: {err:?}"
    );
    assert!(
        err.reason.contains("keyword-form"),
        "explains keyword-form: {err:?}"
    );
}

// Record-form base + a bare-scalar kernel step is a malformed/mixed shape —
// refuse (never panic). Regression for the jointchain.rs comp.field.expect bug.
#[test]
fn jointchain_record_base_scalar_kernel_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(draw(Normal(mu = a, sigma = 0.5)), a = a)
j = jointchain(lawof(record(a = a)), k)
lp = logdensityof(j, record(a = 0.3, b = 0.7))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("record base + scalar kernel step must refuse, not panic");
    assert!(
        err.construct.contains("jointchain"),
        "names jointchain: {err:?}"
    );
}

// Three-component chain a → b → c. Density = logdensityof(Normal(0,1), 0.3)
//   + logdensityof(Normal(0.3, 0.5), 0.7) + logdensityof(Normal(0.7, 0.25), 1.1).
// k2's input `b` is a PLACEHOLDER (`_b_`) referencing k1's variate field, so
// the substitution must target the ref symbol `_b_`, not the input name `b`.
#[test]
fn jointchain_record_multi_step() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k1 = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)
k2 = kernelof(record(c = draw(Normal(mu = _b_, sigma = 0.25))), b = _b_)
j = jointchain(lawof(record(a = a)), k1, k2)
lp = logdensityof(j, record(a = 0.3, b = 0.7, c = 1.1))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "three terms:\n{pir}"
    );
    assert!(pir.contains("(%field mu 0.0)"), "base mean 0.0:\n{pir}");
    assert!(
        pir.contains("(%field mu 0.3)"),
        "k1 mean = realized a = 0.3:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.7)"),
        "k2 mean = realized b = 0.7 (placeholder bound):\n{pir}"
    );
    assert!(pir.contains(" 1.1)"), "k2 scored at c = 1.1:\n{pir}");
    assert!(
        !pir.contains("jointchain") && !pir.contains("kernelof") && !pir.contains("_b_"),
        "measure layer + placeholder gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
}

// A kernel input that names no PRIOR variate field (a forward / unknown
// reference) cannot be bound — refuse rather than emit a dangling ref.
#[test]
fn jointchain_unbindable_input_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(b = draw(Normal(mu = _c_, sigma = 0.5))), c = _c_)
j = jointchain(lawof(record(a = a)), k)
lp = logdensityof(j, record(a = 0.3, b = 0.7))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("input naming a non-prior field must refuse");
    assert!(
        err.construct.contains("jointchain"),
        "names jointchain: {err:?}"
    );
    assert!(
        err.reason.contains("non-prior"),
        "explains the unbindable input: {err:?}"
    );
}
