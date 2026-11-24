use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_infer::{ModuleBundle, Severity};

#[test]
fn location_scale_transports_accept_active_values_and_reordered_fields() {
    for call in [
        "builtin_touniform(Normal, record(sigma = scale, mu = location), x)",
        "builtin_touniform(Cauchy, record(scale = scale, location = location), x)",
        "builtin_tonormal(Normal, record(sigma = scale, mu = location), x)",
        "builtin_fromnormal(Normal, record(sigma = scale, mu = location), x)",
        "builtin_fromuniform(Cauchy, record(scale = scale, location = location), x)",
        "builtin_fromuniform(Exponential, record(rate = scale), x)",
        "builtin_fromuniform(Logistic, record(s = scale, mu = location), x)",
    ] {
        let source = format!(
            "x = elementof(reals)\n\
             location = elementof(reals)\n\
             scale = elementof(posreals)\n\
             inputs = (location, scale, x)\n\
             outputs = {call}\n"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(
            diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{call}: {diagnostics:?}"
        );
        let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
        for differentiate in [gradient, jvp, vjp] {
            let result = differentiate(&deterministic, &["location", "scale", "x"])
                .unwrap_or_else(|error| panic!("{call}: {error}"));
            let printed = flatppl_syntax::print(&result);
            let mut reparsed = flatppl_syntax::parse(&printed).unwrap();
            let diagnostics = flatppl_infer::infer(&mut reparsed);
            assert!(
                diagnostics.iter().all(|d| d.severity != Severity::Error),
                "{call}: {diagnostics:?}\n{printed}"
            );
            flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
        }
    }
}

#[test]
fn indexed_transport_parameters_keep_the_input_cotangent_shape() {
    let source = "
locations = elementof(cartpow(reals, 3))
scale = elementof(posreals)
x = elementof(reals)
first = builtin_touniform(Normal, record(mu = locations[1], sigma = scale), x)
second = builtin_touniform(Cauchy, record(location = locations[3], scale = scale), x)
inputs = (locations, scale, x)
outputs = first + second
";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
    let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
    let result = gradient(&deterministic, &["locations", "scale", "x"]).unwrap();
    let printed = flatppl_syntax::print(&result);
    let mut reparsed = flatppl_syntax::parse(&printed).unwrap();
    let diagnostics = flatppl_infer::infer(&mut reparsed);
    assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
    flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
}

#[test]
fn differentiates_retained_normal_log_mass() {
    use flatppl_autodiff::{determinize_query_with_options, gradient_with_options};
    use flatppl_determinizer::LoweringOptions;

    let source = "
mu = elementof(reals)
sigma = elementof(posreals)
lo = elementof(reals)
hi = elementof(reals)
x = elementof(reals)
law = normalize(truncate(Normal(mu, sigma), interval(lo, hi)))
inputs = (mu, sigma, lo, hi, x)
outputs = logdensityof(law, x)
";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
    for compact_log_cdf_difference in [false, true] {
        let options = LoweringOptions {
            compact_log_cdf_difference,
            retain_standard_functions: &[("special-functions", "log_ndtr")],
            ..Default::default()
        };
        let deterministic =
            determinize_query_with_options(&module, &ModuleBundle::new(), &options).unwrap();
        let derivative =
            gradient_with_options(&deterministic, &["mu", "sigma", "lo", "hi", "x"], &options)
                .unwrap();
        let text = flatppl_syntax::print(&derivative);
        assert!(text.contains(".log_ndtr("), "{text}");
        assert!(text.contains(".erfcx("), "{text}");
    }
}

#[test]
fn differentiates_standard_normal_cdf_and_quantile() {
    for function in ["invprobit", "probit"] {
        let mut module = flatppl_syntax::parse(&format!(
            "x = elementof(reals)\ninputs = x\noutputs = {function}(x)\n"
        ))
        .unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
        let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
        gradient(&deterministic, &["x"]).unwrap();
    }
}

#[test]
fn differentiates_incomplete_tails_in_the_integration_limit() {
    for (family, parameters) in [("gamma", "a"), ("beta", "a, b")] {
        for (prefix, suffix) in [("", ""), ("", "c"), ("log_", ""), ("log_", "c")] {
            let function = format!("{prefix}{family}inc{suffix}");
            let source = format!(
                "sp = standard_module(\"special-functions\", \"0.1\")\n\
                 a = elementof(posreals)\n\
                 b = elementof(posreals)\n\
                 x = elementof(reals)\n\
                 inputs = (a, b, x)\n\
                 outputs = sp.{function}({parameters}, x)\n"
            );
            let mut module = flatppl_syntax::parse(&source).unwrap();
            let diagnostics = flatppl_infer::infer(&mut module);
            assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
            let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
            gradient(&deterministic, &["x"]).unwrap_or_else(|error| panic!("{function}: {error}"));
            assert!(gradient(&deterministic, &["a"]).is_err(), "{function}");
        }
    }
}

#[test]
fn differentiates_beta_and_gamma_normalizing_constants() {
    for call in ["loggamma(a)", "sp.logbeta(a, b)", "sp.beta(a, b)"] {
        let source = format!(
            "sp = standard_module(\"special-functions\", \"0.1\")\n\
             a = elementof(posreals)\n\
             b = elementof(posreals)\n\
             inputs = (a, b)\n\
             outputs = {call}\n"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
        let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
        gradient(&deterministic, &["a", "b"]).unwrap_or_else(|error| panic!("{call}: {error}"));
    }
}
