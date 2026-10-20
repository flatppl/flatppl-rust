use flatppl_stablehlo::{EmitOptions, LOWERING_OPTIONS, Mode};

fn compile(source: &str, mode: Mode) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{source}\n{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &LOWERING_OPTIONS,
    )
    .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    let restrict_enzyme_compatible = matches!(mode, Mode::LogDensity);
    flatppl_stablehlo::emit(
        &model,
        mode,
        &EmitOptions {
            restrict_enzyme_compatible,
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{source}\n{error:?}"))
}

#[test]
fn scalar_closed_tilts_score_and_sample_without_integration() {
    // Each row changes a different family parameter. Runtime oracle checks
    // independently compare the emitted values, derivatives and RNG states.
    for (base, discrete) in [
        ("Normal(1, 2)", false),
        ("Gamma(3, 2)", false),
        ("Exponential(2)", false),
        ("ChiSquared(4)", false),
        ("Bernoulli(0.3)", true),
        ("Binomial(5, 0.3)", true),
        ("Poisson(2)", true),
        ("Geometric(0.6)", true),
        ("NegativeBinomial(2, 3)", true),
        ("NegativeBinomial2(2, 3)", true),
    ] {
        let (set, reference) = if discrete {
            ("integers", "Counting")
        } else {
            ("reals", "Lebesgue")
        };
        for law in [
            format!("normalize(logweighted(fn(theta * _ + 7), {base}))"),
            format!("tilt(logweighted(fn(theta * _ + 7), {reference}({set})), {base})"),
        ] {
            let source = format!(
                "theta = external(reals)\nx = external({set})\nq = {law}\n\
                 inputs = (theta, x)\noutputs = logdensityof(q, x)"
            );
            compile(&source, Mode::LogDensity);
            let sample = source.replace("logdensityof(q, x)", "rand(q, rnginit(1729))");
            let ir = compile(&sample, Mode::Sample);
            // No density integral, proposal rejection, or user weight survives
            // into sampling: it uses the transformed primitive's RNG path.
            assert!(ir.contains("stablehlo.rng_bit_generator"), "{base}");
        }
    }
}

#[test]
fn explicit_weight_parameters_keep_closed_tilts_and_rng_paths() {
    for law in [
        "normalize(logweighted((x, p) -> p*x, Normal(0, 1), theta))",
        "tilt(logweighted((x, p) -> p*x, Lebesgue(reals), theta), Normal(0, 1))",
    ] {
        // Completing the square gives N(theta,1). At theta=.7,x=.25,
        // log density is -1.0201885332046727 and d/dtheta is -.45.
        let source = format!(
            "theta = external(reals)\nx = external(reals)\nq = {law}\n\
             inputs = (theta, x)\noutputs = logdensityof(q, x)"
        );
        compile(&source, Mode::LogDensity);
        let sampled = compile(
            &source.replace("logdensityof(q, x)", "rand(q, rnginit(1729))"),
            Mode::Sample,
        );
        assert!(sampled.contains("stablehlo.rng_bit_generator"));
    }
}

#[test]
fn explicit_weights_keep_whole_records_and_exact_normalizers() {
    for source in [
        "p = external(cartprod(a=reals, b=reals))\n\
         x = external(cartprod(a=reals, b=reals))\n\
         M = logweighted((x,p) -> p.a*x.a+p.b*x.b, joint(a=Normal(0,1),b=Normal(0,1)), p)\n\
         inputs = (p,x)\noutputs = logdensityof(M,x)",
        "p = external(reals)\nx = external(reals)\n\
         M = normalize(logweighted((x,p) -> logdensityof(Normal(p,1),x), Normal(0,1), p))\n\
         inputs = (p,x)\noutputs = logdensityof(M,x)",
    ] {
        compile(source, Mode::LogDensity);
    }
}

#[test]
fn vector_linear_tilts_keep_covariance_and_live_coefficients() {
    for law in [
        "normalize(logweighted(weight, B))",
        "tilt(logweighted(weight, Lebesgue(cartpow(reals, 2))), B)",
    ] {
        let source = format!(
            "theta = external(cartpow(reals, 2))\n\
             x = external(cartpow(reals, 2))\n\
             v = elementof(cartpow(reals, 2))\n\
             weight = functionof(sum(theta .* v) + 2, value = v)\n\
             B = MvNormal([1, -1], [[2, 0.5], [0.5, 3]])\n\
             q = {law}\ninputs = (theta, x)\noutputs = logdensityof(q, x)"
        );
        compile(&source, Mode::LogDensity);
        compile(
            &source.replace("logdensityof(q, x)", "rand(q, rnginit(0))"),
            Mode::Sample,
        );
    }
}

#[test]
fn aliased_affine_boundaries_and_repeated_tilts_remain_callable() {
    let source = "theta = external(reals)\nx = external(reals)\n\
                  v = elementof(reals)\nlinear = theta * v\n\
                  weight = functionof(2 * linear / 3 + 5, value = v)\n\
                  first = normalize(logweighted(weight, Normal(0, 1)))\n\
                  q = normalize(logweighted(fn(-theta * _), first))\n\
                  inputs = (theta, x)\noutputs = logdensityof(q, x)";
    compile(source, Mode::LogDensity);
    compile(
        &source.replace("logdensityof(q, x)", "rand(iid(q, 4), rnginit(0))"),
        Mode::Sample,
    );
}
