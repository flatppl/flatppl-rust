use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str, mode: Mode) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_roots(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        mode,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn matrix_iid_density_maps_whole_events_without_unrolling() {
    for (measure, domain) in [
        ("Normal(mu, 2)", "reals"),
        ("MvNormal([mu, -1.0], eye(2))", "cartpow(reals, 2)"),
    ] {
        let source = format!(
            "mu = elementof(reals)\n\
            x = elementof(cartpow({domain}, [2, 3]))\n\
            z ~ iid({measure}, [2, 3])\n\
            P = lawof(record(z = z))\n\
            inputs = (mu, x)\n\
            outputs = logdensityof(P, record(z = x))"
        );
        let mlir = emit(&source, Mode::LogDensity);
        assert!(!mlir.contains("stablehlo.while"), "{mlir}");
        assert!(
            mlir.lines()
                .any(|line| line.contains("func.func @logdensity")
                    && line.ends_with("-> tensor<f32> {")),
            "{mlir}"
        );
        if domain != "reals" {
            assert!(
                mlir.contains("across dimensions = [2]"),
                "vector coordinates must contract before the product axes: {mlir}"
            );
        }
    }
}

#[test]
fn matrix_sampling_keeps_event_axes_and_existing_rng_consumption() {
    for (measure, event) in [
        ("Normal(0, 1)", ""),
        ("Gamma(2, 3)", ""),
        ("MvNormal([1.0, -1.0], eye(2))", "2x"),
    ] {
        let mapped = if event.is_empty() {
            "x .+ 1"
        } else {
            "sum.(x)"
        };
        let source = |size| {
            format!(
                "s = rnginit(0)\n\
            x, s2 = rand(iid({measure}, {size}), s)\n\
            y, s3 = rand(Normal(0, 1), s2)\n\
            outputs = (x, {mapped}, y, s3)"
            )
        };
        let matrix = emit(&source("[2, 3]"), Mode::Sample);
        let flat = emit(&source("6"), Mode::Sample);
        assert!(
            matrix.contains(&format!(
                "-> (tensor<2x3x{event}f32>, tensor<2x3xf32>, tensor<f32>, tensor<2xui64>)"
            )),
            "{matrix}"
        );
        for operation in ["stablehlo.rng_bit_generator", "stablehlo.while"] {
            assert_eq!(
                matrix.matches(operation).count(),
                flat.matches(operation).count(),
                "matrix shaping must not repeat sampling: {matrix}"
            );
        }
    }
}
