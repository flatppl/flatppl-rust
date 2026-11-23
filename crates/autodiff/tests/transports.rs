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
