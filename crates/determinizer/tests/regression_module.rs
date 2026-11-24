use flatppl_determinizer::{determinize, is_flatpdl};

#[test]
fn regression_vector_laws_lower_for_density_and_sampling() {
    for (name, args, response) in [
        ("BernoulliLogitGLM", "x, 0.1, b", "[0, 1, 1]"),
        ("BinomialLogitGLM", "x, 4, 0.1, b", "[0, 2, 4]"),
        (
            "CategoricalLogitGLM",
            "x, [0.1, -0.1], rowstack([[0.2, -0.3], [0.3, -0.1]])",
            "[1, 2, 1]",
        ),
        (
            "NormalGLM",
            "x = x, alpha = 0.1, beta = b, sigma = 1.0",
            "[0.1, 0.2, 0.3]",
        ),
        ("PoissonLogGLM", "x, 0.1, b", "[0, 1, 2]"),
        ("NegativeBinomial2LogGLM", "x, 0.1, b, 2.0", "[0, 1, 2]"),
        ("StudentTGLM", "x, 4.0, 0.1, b, 1.0", "[0.1, 0.2, 0.3]"),
        ("GammaLogGLM", "x, 0.1, b, 2.0", "[0.1, 0.2, 0.3]"),
        ("OrderedLogitGLM", "x, b, [-1.0, 1.0]", "[1, 2, 3]"),
        ("BetaLogitGLM", "x, 0.1, b, 4.0", "[0.1, 0.2, 0.3]"),
    ] {
        let source = format!(
            "r = standard_module(\"regression\", \"0.1\")\nx = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])\nb = [0.2, -0.3]\ny ~ r.{name}({args})\nlaw = lawof(y)\nscore = logdensityof(law, {response})\nsample = rand(law, rnginit(0))\noutputs = (score, sample)\n"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(
            !diagnostics
                .iter()
                .any(|d| d.severity == flatppl_infer::Severity::Error),
            "{name}: {diagnostics:?}"
        );
        let out = determinize(&module).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(is_flatpdl(&out).is_ok(), "{name}");
    }
}

#[test]
fn a_recognised_regression_law_lowers_for_density_and_sampling() {
    for (law, response) in [
        ("Normal.(0.1 .+ x * b, 1.0)", "[0.1, 0.2, 0.3]"),
        ("Bernoulli.(invlogit.(0.1 .+ x * b))", "[0, 1, 1]"),
        ("Poisson.(exp.(x * b .+ 0.1))", "[0, 1, 2]"),
    ] {
        let source = format!(
            "x = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])\nb = elementof(cartpow(reals, 2))\n\
             y ~ {law}\nlaw = lawof(y)\nscore = logdensityof(law, {response})\n\
             sample = rand(law, rnginit(0))\ninputs = b\noutputs = (score, sample)\n"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(
            !diagnostics
                .iter()
                .any(|d| d.severity == flatppl_infer::Severity::Error),
            "{law}: {diagnostics:?}"
        );
        let roots = [module.intern("inputs"), module.intern("outputs")];
        // Without the likelihood reductions the lowered query keeps the module law.
        let out = flatppl_determinizer::determinize_with_options(
            &module,
            &Default::default(),
            Some(&roots),
            &flatppl_determinizer::LoweringOptions {
                disable_sufficient_statistics: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{law}: {e:?}"));
        assert!(is_flatpdl(&out).is_ok(), "{law}");
        assert!(flatppl_syntax::print(&out).contains("GLM"), "{law}");
    }
}
