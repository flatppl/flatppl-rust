use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str, mode: &Mode) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        match mode {
            Mode::LogDensity => Mode::LogDensity,
            Mode::Sample => Mode::Sample,
        },
        &EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn callback_normalization_retains_scalar_weight_and_base_proofs() {
    let prefix = "mu = external(reals)\nsigma = external(reals)\nx = external(reals)\n\
                  row = (l, s) -> normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), \
                    locscale(StudentT(4), l, ifelse(s > 0 && isfinite(s), s, 1))))\n\
                  inputs = (mu, sigma, x)\n";
    let base = "normalize(weighted(ifelse(sigma > 0 && isfinite(sigma), 1, 0), \
                locscale(StudentT(4), mu, ifelse(sigma > 0 && isfinite(sigma), sigma, 1))))";
    for (mode, output) in [
        (Mode::LogDensity, "logdensityof(row(mu, sigma), x)"),
        (Mode::Sample, "rand(row(mu, sigma), rnginit(19))"),
    ] {
        let source = format!("{prefix}outputs = {output}");
        assert_eq!(
            emit(&source, &mode),
            emit(&source.replace("row(mu, sigma)", base), &mode),
        );
    }
}

#[test]
fn callback_recovery_does_not_assume_constant_weight_or_probability_mass() {
    for body in [
        "normalize(weighted(s, Lebesgue(reals)))",
        "normalize(weighted(x -> x*x + 1, Normal(s, 1)))",
        "normalize(weighted(s, Poisson(2)))",
    ] {
        let source = format!(
            "s = external(reals)\nrow = s -> {body}\ninputs = s\noutputs = rand(row(s), rnginit(19))"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        flatppl_infer::infer(&mut module);
        assert!(flatppl_determinizer::determinize(&module).is_err());
    }
}

#[test]
fn applied_vector_prior_retains_its_static_shape_and_mass_guard() {
    for shape in ["3", "[2, 3]"] {
        let source = format!(
            "s = external(reals)\nshift ~ Normal(0, 1)\n\
             b ~ iid(normalize(weighted(ifelse(shift > 0 && isfinite(shift), 1, 0), \
                 locscale(StudentT(4), 0, ifelse(shift > 0 && isfinite(shift), shift, 1)))), {shape})\n\
             K = kernelof(b, shift = shift)\n\
             inputs = s\noutputs = rand(K(s), rnginit(19))"
        );
        let expected = format!(
            "s = external(reals)\ninputs = s\n\
             outputs = rand(iid(normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), \
                 locscale(StudentT(4), 0, ifelse(s > 0 && isfinite(s), s, 1)))), {shape}), rnginit(19))"
        );
        assert_eq!(emit(&source, &Mode::Sample), emit(&expected, &Mode::Sample));
    }
}
