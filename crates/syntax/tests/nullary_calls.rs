use flatppl_syntax::parse;

/// §05 requires CallArgs for ordinary and dotted calls, regardless of callee.
#[test]
fn calls_require_an_argument() {
    for source in [
        "x = Normal()",
        "x = Poisson()",
        "x = log()",
        "x = base.exp()",
        "x = m.CrystalBall()",
        "f(x) = x\ny = f()",
        "x = (y -> y)()",
        "x = exp.()",
        "x = record()",
        "x = table()",
        "x = vector()",
        "x = cat()",
        "x = rowstack()",
        "x = colstack()",
        "x = aggregate()",
        "x = metricsum()",
        "x = load_module()",
        "x = standard_module()",
        "x = load_data()",
        "x = broadcast()",
        "x = draw()",
        "x = elementof()",
        "x = joint()",
        "x = joint_likelihood()",
        "x = jointchain()",
        "x = cartprod()",
        "x = superpose()",
        "x = functionof()",
        "x = kernelof()",
    ] {
        let error = parse(source).expect_err(source);
        assert!(error.to_string().contains("nullary"), "{source}: {error}");
        assert!(
            error.span.is_some(),
            "{source}: diagnostic needs a location"
        );
    }
}

/// Empty output-axis lists and auto-reification boundaries are not empty calls.
#[test]
fn empty_axes_and_inferred_boundaries_remain_valid() {
    for source in [
        "x = [1, 2]\ns = aggregate(sum, [], x[.i])",
        "x = [1, 2]\ns[] := x[.i]",
        "f = functionof(1)",
        "x ~ Normal(0, 1)\nk = kernelof(x)",
        "r = (a = 1)",
        "x = Normal(mu = 0, sigma = 1)",
    ] {
        parse(source).expect(source);
    }
}
