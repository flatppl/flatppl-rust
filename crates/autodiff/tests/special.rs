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
        lower(&printed);

        for selected in [&["v"][..], &["v", "z"][..]] {
            let error = differentiate(&module, selected).unwrap_err();
            assert!(
                error.to_string().contains("with respect to order"),
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

#[test]
fn dotted_regression_replays_a_mixed_frame_with_array_cells() {
    let module = lower(
        r#"
reg = standard_module("regression", "0.1")
x = elementof(cartpow(cartpow(reals, [3, 2]), [2, 1]))
alpha = elementof(cartpow(reals, [1, 3]))
beta = elementof(cartpow(cartpow(reals, 2), [1, 1]))
sigma = elementof(posreals)
y = elementof(cartpow(cartpow(reals, 3), [2, 3]))
inputs = (x, alpha, beta, sigma, y)
outputs = logdensityof(reg.NormalGLM.(x, alpha, beta, sigma), y)
"#,
    );
    for differentiate in [gradient, jvp, vjp] {
        let derivative = differentiate(&module, &["x", "alpha", "beta", "sigma", "y"]).unwrap();
        let reparsed = lower(&flatppl_syntax::print(&derivative));
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn bessel_projections_remain_differentiable_after_complex_arithmetic() {
    let module = lower(
        r#"
sf = standard_module("special-functions", "0.1")
v = elementof(interval(0.0, 16.0))
z = elementof(interval(-4.0, -0.125))
inputs = (v, z)
outputs = imag(complex(2.0, 1.0) * sf.bessel_i_scaled(v, z))
"#,
    );
    for differentiate in [gradient, jvp, vjp] {
        let result = differentiate(&module, &["v", "z"]).unwrap();
        lower(&flatppl_syntax::print(&result));
    }
}

#[test]
fn airy_slope_round_trips_and_preserves_broadcast_shape() {
    let module = lower(
        r#"
sf = standard_module("special-functions", "0.1")
x = elementof(cartpow(reals, 3))
inputs = x
outputs = sum(sf.airy.(x))
"#,
    );
    for differentiate in [gradient, jvp, vjp] {
        let result = differentiate(&module, &["x"]).unwrap();
        lower(&flatppl_syntax::print(&result));
    }
}

#[test]
fn beta_shapes_and_bounded_bessel_orders_round_trip_with_numerical_integration() {
    use flatppl_autodiff::{
        NUMERICAL_SOURCE_OPTIONS, determinize_query_with_options, gradient_with_options,
        jvp_with_options, prepare_source_with_options, vjp_with_options,
    };

    for call in [
        "sf.log_betaincc(a, b, x)",
        "sf.bessel_y(a, x)",
        "sf.log_bessel_k(a, x)",
    ] {
        let source = format!(
            r#"
sf = standard_module("special-functions", "0.1")
a = elementof(interval(1.0, 3.0))
b = elementof(posreals)
x = elementof(interval(0.125, 1.0))
inputs = (a, b, x)
outputs = {call}
"#
        );
        let module = lower(&source);
        assert!(gradient(&module, &["a"]).is_err());
        for differentiate in [gradient_with_options, jvp_with_options, vjp_with_options] {
            let mut result = differentiate(&module, &["a"], &NUMERICAL_SOURCE_OPTIONS).unwrap();
            prepare_source_with_options(&mut result, &NUMERICAL_SOURCE_OPTIONS).unwrap();
            let mut reparsed = flatppl_syntax::parse(&flatppl_syntax::print(&result)).unwrap();
            flatppl_infer::infer(&mut reparsed);
            let reparsed = determinize_query_with_options(
                &reparsed,
                &ModuleBundle::new(),
                &NUMERICAL_SOURCE_OPTIONS,
            )
            .unwrap();
            flatppl_determinizer::is_flatpdl_with_options(&reparsed, &NUMERICAL_SOURCE_OPTIONS)
                .unwrap();
        }
    }
}
