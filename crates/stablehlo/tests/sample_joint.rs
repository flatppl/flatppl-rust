//! A drawn positional `joint` (§06: the variate is the `cat` of the components) samples
//! in StableHLO, and a later draw reads one of its components.
use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

#[test]
fn a_positional_joint_and_its_dependent_draw_emit_a_sample() {
    let source = "mu0 = elementof(reals)\n\
                  v ~ joint(Gamma(2.0, 1.0), Gamma(3.0, 1.0))\n\
                  y ~ Normal(mu0, sqrt(v[1]))\n\
                  prior = lawof(record(v = v, y = y))\n\
                  draws = rand(prior, rnginit([0]))\n\
                  inputs = (mu0)\noutputs = (draws)\n";
    let mut model = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut model)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    let module = flatppl_stablehlo::emit(
        &model,
        Mode::Sample,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            dtype: Dtype::F64,
            ..Default::default()
        },
    )
    .unwrap();
    let signature = module
        .lines()
        .find(|line| line.contains("func.func @sample"))
        .unwrap();
    // The record's fields in order: `v` a 2-vector, `y` a scalar.
    assert!(
        signature.contains("-> (tensor<2xf64>, tensor<f64>"),
        "{signature}"
    );
}
