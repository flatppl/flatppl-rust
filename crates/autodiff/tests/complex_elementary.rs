use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_infer::{ModuleBundle, Severity};

#[test]
fn complex_elementary_products_round_trip_with_real_coordinate_inputs() {
    for expression in [
        "imag(asin(complex(0.3, 0.7) + z))",
        "real(cosh(z) * tan(z) + atan(z))",
        "abs(z ^ complex(1.4, -0.6))",
    ] {
        let source = format!(
            "x = elementof(reals)\ny = elementof(reals)\ninputs = (x, y)\n\
             z = complex(x, y)\noutputs = {expression}\n"
        );
        let lower = |source: &str| {
            let mut module = flatppl_syntax::parse(source).unwrap();
            let diagnostics = flatppl_infer::infer(&mut module);
            assert!(
                diagnostics.iter().all(|d| d.severity != Severity::Error),
                "{diagnostics:?}"
            );
            determinize_query(&module, &ModuleBundle::new()).unwrap()
        };
        let module = lower(&source);
        for differentiate in [gradient, jvp, vjp] {
            let result = differentiate(&module, &["x", "y"]).unwrap();
            let printed = flatppl_syntax::print(&result);
            lower(&printed);
        }
    }
}
