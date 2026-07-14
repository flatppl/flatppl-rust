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

// Scalar-cat jointchain: output is a VECTOR variate; slices via get0. Base
// `a ~ Normal(0,1)` scored at get0(v,0); kernel `b ~ Normal(mu = a, 0.5)`
// scored at get0(v,1) with its input `a` bound to get0(v,0). The scored
// variate here is the LITERAL point vector `[0.3, 0.7]`, so canon Pass 3
// (`flatten_structural`) resolves every get0 slice all the way down to a bare
// literal — re-baselined for buffy #263 Pass 3 (this test used to pin the
// unresolved `get0(v, i)` shape; the flattened literals are a stronger pin,
// not a weaker one, since they also verify the kernel's mean correctly picks
// up the base slice's value).
#[test]
fn jointchain_scalar_single_step() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(Normal(mu = a, sigma = 0.5), a = a)
j = jointchain(lawof(a), k)
lp = logdensityof(j, [0.3, 0.7])";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two terms:\n{pir}"
    );
    assert!(pir.contains("(%field mu 0.0)"), "base mean 0.0:\n{pir}");
    // canon Pass 3 flattens the get0 slices of the literal point vector all
    // the way down: the kernel's mean (bound to the base slice) resolves to
    // the literal 0.3, and no residual get0 accessor survives.
    assert!(
        !pir.contains("(get0 "),
        "get0 slices of the literal vector are fully flattened:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.3)"),
        "kernel mean is the flattened base slice (0.3):\n{pir}"
    );
    assert!(
        !pir.contains("jointchain") && !pir.contains("kernelof") && !pir.contains("lawof"),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
}

// Scalar-cat 3-step: c ~ K2([a,b]). The single input `_ab_` binds to
// vector(get0(v,0), get0(v,1)); the body indexes it. The scored variate here
// is the LITERAL point vector `[0.3, 0.7, 1.1]`, so canon Pass 3
// (`flatten_structural`), together with Pass 1's const-fold, resolves the
// get0/vector wrapping and the resulting `add` all the way down to bare
// literals (`k2`'s mean folds to `0.3 + 0.7 = 1.0`) — re-baselined for buffy
// #263 Pass 3 (this test used to pin the unresolved `vector(get0(...),
// get0(...))` shape; the fully-flattened literal is a stronger pin, since it
// also verifies the arithmetic).
#[test]
fn jointchain_scalar_multi_step() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k1 = kernelof(Normal(mu = a, sigma = 0.5), a = a)
k2 = kernelof(Normal(mu = add(get0(_ab_, 0), get0(_ab_, 1)), sigma = 0.25), ab = _ab_)
j = jointchain(lawof(a), k1, k2)
lp = logdensityof(j, [0.3, 0.7, 1.1])";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "three terms:\n{pir}"
    );
    // canon flattens the prior cat's get0/vector wrapping (and folds the
    // resulting sum) down to the literal 1.0 mean for k2 — no residual get0
    // or vector call survives.
    assert!(
        !pir.contains("(get0 ") && !pir.contains("(vector "),
        "get0/vector wrapping of the literal point vector is fully flattened:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 1.0)"),
        "k2's mean folds the flattened base+kernel-1 slices (0.3 + 0.7 = 1.0):\n{pir}"
    );
    assert!(
        !pir.contains("_ab_"),
        "placeholder substituted away:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
}

// A record-form kernel in a scalar-cat chain (base is scalar, kernel body is a
// record draw) is a mixed family — refuse.
#[test]
fn jointchain_mixed_family_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)
j = jointchain(lawof(a), k)
lp = logdensityof(j, [0.3, 0.7])";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("mixed record/scalar families must refuse");
    assert!(
        err.construct.contains("jointchain"),
        "names jointchain: {err:?}"
    );
    assert!(
        err.reason.contains("mixed families"),
        "explains mixed families: {err:?}"
    );
}

// A jointchain base that is a measure-combinator (here superpose), not a single
// primitive draw, must refuse NAMING jointchain (design refuse-set: nullary-kernel
// / non-single-draw base) — not with the inner combinator's name.
#[test]
fn jointchain_combinator_base_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
m0 = lawof(record(a = a))
finite_m = superpose(m0, m0)
k = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)
j = jointchain(finite_m, k)
lp = logdensityof(j, record(a = 0.3, b = 0.7))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("combinator base must refuse");
    assert!(
        err.construct.contains("jointchain"),
        "refusal names jointchain: {err:?}"
    );
}
