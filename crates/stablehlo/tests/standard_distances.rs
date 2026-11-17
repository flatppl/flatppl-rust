//! §09 distance calls retain callable semantics and nested-array axes.
//! Executed values and derivatives are checked against SciPy separately.

use flatppl_infer::{Level, ModuleBundle};
use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let (diagnostics, bundle) =
        flatppl_infer::infer_module_with_inputs(&mut module, ModuleBundle::new(), Level::Shape);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &module,
        &bundle,
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(&lowered, Mode::LogDensity, &EmitOptions::default()).unwrap()
}

#[test]
fn cross_distance_keeps_unequal_outer_extents_and_custom_callable() {
    let output = emit(
        r#"
        d = standard_module("distances", "0.1")
        t = external(reals)
        metric(a, b) = sum(abs.(a .- b))
        alias = metric
        x = [[t, 0.0], [0.0, 1.0]]
        y = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        inputs = t
        outputs = d.cross_distance(distance = alias, x = x, y = y)
    "#,
    );
    assert!(output.contains("tensor<2x3xf32>"));
}

#[test]
fn pairwise_distance_accepts_a_standard_member_alias() {
    let output = emit(
        r#"
        d = standard_module("distances", "0.1")
        t = external(reals)
        metric = d.euclidean
        x = [[t, 0.0], [0.0, 1.0], [1.0, 1.0]]
        inputs = t
        outputs = d.pairwise_distance(metric, x)
    "#,
    );
    assert!(output.contains("tensor<3x3xf32>"));
}

#[test]
fn scalar_distances_accept_live_arguments_and_keyword_order() {
    let output = emit(
        r#"
        d = standard_module("distances", "0.1")
        u = external(cartpow(reals, 3))
        v = external(cartpow(reals, 3))
        p = external(posreals)
        inputs = (u, v, p)
        outputs = (d.minkowski(v = v, p = p, u = u), d.jensenshannon(u, v))
    "#,
    );
    assert!(output.contains("tensor<3xf32>"));
}
