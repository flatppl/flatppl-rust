use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn lower(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}

#[test]
fn real_bessel_j_differentiates_its_argument_with_order_fixed() {
    let module = lower(
        r#"
sf = standard_module("special-functions", "0.1")
v = elementof(reals)
z = elementof(reals)
j = sf.bessel_j
value = j(z = z, v = v)
inputs = (v, z)
outputs = real(value)
"#,
    );
    for differentiate in [gradient, jvp, vjp] {
        let result = differentiate(&module, &["z"]).unwrap();
        let printed = flatppl_syntax::print(&result);
        assert_eq!(printed.matches("real(sf.bessel_j(").count(), 2, "{printed}");
        lower(&printed);

        for selected in [&["v"][..], &["v", "z"][..]] {
            let error = differentiate(&module, selected).unwrap_err();
            assert!(
                error.to_string().contains("respect to its order"),
                "{error}"
            );
        }
    }
}

#[test]
fn real_bessel_j_order_uses_inferred_input_bounds() {
    let module = lower(
        r#"
sf = standard_module("special-functions", "0.1")
v = elementof(unitinterval)
x = elementof(interval(0.125, 4.0))
inputs = (v, x)
outputs = real(sf.bessel_j(v, x))
"#,
    );
    for differentiate in [gradient, jvp, vjp] {
        let derivative = differentiate(&module, &["v", "x"]).unwrap();
        let reparsed = lower(&flatppl_syntax::print(&derivative));
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}
