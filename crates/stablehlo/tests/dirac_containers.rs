//! §06 Dirac equality covers a whole variate; §04 broadcast retains each cell's score.

use flatppl_stablehlo::{EmitOptions, Mode};

fn compile(source: &str) -> Result<String, String> {
    let mut model = flatppl_syntax::parse(source).map_err(|e| format!("{e:?}"))?;
    let diagnostics = flatppl_infer::infer(&mut model);
    if !diagnostics.is_empty() {
        return Err(format!("{diagnostics:?}"));
    }
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let options = EmitOptions::default();
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .map_err(|e| format!("{e:?}"))?;
    flatppl_stablehlo::emit(&model, Mode::LogDensity, &options).map_err(|e| e.msg)
}

#[test]
fn tensor_atoms_reduce_every_event_axis() {
    for domain in [
        "cartpow(reals, 2)",
        "cartpow(reals, [2, 2])",
        "cartpow(cartpow(reals, 2), 2)",
    ] {
        let ir = compile(&format!(
            "atom = external({domain})\npoint = external({domain})\n\
             inputs = (atom, point)\noutputs = logdensityof(Dirac(atom), point)"
        ))
        .unwrap();
        assert!(
            ir.lines()
                .any(|l| l.contains("func.func") && l.ends_with("-> tensor<f32> {")),
            "{ir}"
        );
        assert!(
            ir.contains("applies stablehlo.and across dimensions = [0]"),
            "{ir}"
        );
    }
}

#[test]
fn flat_abi_and_nested_literal_records_compare_all_fields() {
    for source in [
        "atom = external(cartprod(a = reals, b = cartpow(reals, 2)))\n\
         point = external(cartprod(a = reals, b = cartpow(reals, 2)))\n\
         inputs = (atom, point)\noutputs = logdensityof(Dirac(atom), point)",
        "a = external(reals)\nb = external(cartpow(reals, 2))\n\
         x = external(reals)\ny = external(cartpow(reals, 2))\n\
         atom = record(a = a, nested = record(b = b))\n\
         point = record(a = x, nested = record(b = y))\n\
         inputs = (a, b, x, y)\noutputs = logdensityof(Dirac(atom), point)",
    ] {
        let ir = compile(source).unwrap();
        assert!(ir.contains("-> tensor<f32> {"), "{ir}");
        assert_eq!(ir.matches("stablehlo.compare EQ").count(), 2, "{ir}");
        assert!(ir.contains("applies stablehlo.and"), "{ir}");
        assert!(ir.contains(" = stablehlo.and "), "{ir}");
    }
}

#[test]
fn declared_literal_record_inputs_override_the_source_fields() {
    let ir = compile(
        "atom = record(a = 1.0, b = [2.0, 3.0])\n\
         point = record(a = 4.0, b = [5.0, 6.0])\n\
         inputs = (atom, point)\noutputs = logdensityof(Dirac(atom), point)",
    )
    .unwrap();
    assert!(ir.contains("stablehlo.compare EQ, %arg2, %arg0"), "{ir}");
    assert!(ir.contains("stablehlo.compare EQ, %arg3, %arg1"), "{ir}");
}

#[test]
fn record_order_and_collection_layers_are_not_broadcast_equality() {
    for source in [
        "a = external(reals)\nb = external(reals)\ninputs = (a, b)\n\
         outputs = logdensityof(Dirac(record(a = a, b = b)), record(b = b, a = a))",
        "atom = external(cartpow(reals, [2, 2]))\n\
         point = external(cartpow(cartpow(reals, 2), 2))\ninputs = (atom, point)\n\
         outputs = logdensityof(Dirac(atom), point)",
        "atom = external(cartpow(reals, 1))\npoint = external(cartpow(reals, 2))\n\
         inputs = (atom, point)\noutputs = logdensityof(Dirac(atom), point)",
    ] {
        assert!(
            compile(source).is_err(),
            "must reject a different variate type: {source}"
        );
    }
}

#[test]
fn iid_vector_atoms_preserve_equal_width_collection_axis() {
    let ir = compile(
        "atom = external(cartpow(reals, 2))\n\
         point = external(cartpow(cartpow(reals, 2), 2))\n\
         inputs = (atom, point)\noutputs = logdensityof(iid(Dirac(atom), 2), point)",
    )
    .unwrap();
    assert!(
        ir.contains("applies stablehlo.and across dimensions = [1]"),
        "{ir}"
    );
    assert!(
        ir.contains("applies stablehlo.add across dimensions = [0]"),
        "{ir}"
    );
}

#[test]
fn scalar_iid_keeps_separate_scores_before_sum() {
    let ir = compile(
        "atom = external(reals)\npoint = external(cartpow(reals, 2))\n\
         inputs = (atom, point)\noutputs = logdensityof(iid(Dirac(atom), 2), point)",
    )
    .unwrap();
    assert!(!ir.contains("applies stablehlo.and"), "{ir}");
    assert!(
        ir.contains("stablehlo.select") && ir.contains("applies stablehlo.add"),
        "{ir}"
    );
}

#[test]
fn iid_record_atoms_reduce_fields_within_each_row() {
    let ir = compile(
        "atom = external(cartprod(a = reals, b = cartpow(reals, 2)))\n\
         point = external(cartpow(cartprod(a = reals, b = cartpow(reals, 2)), 2))\n\
         inputs = (atom, point)\noutputs = logdensityof(iid(Dirac(atom), 2), point)",
    )
    .unwrap();
    assert!(ir.contains(" = stablehlo.and "), "{ir}");
    assert!(
        ir.contains("applies stablehlo.and across dimensions = [1]"),
        "{ir}"
    );
    assert!(
        ir.contains("applies stablehlo.add across dimensions = [0]"),
        "{ir}"
    );
}

#[test]
fn user_function_broadcast_preserves_vector_event_scores() {
    let ir = compile(
        "atom = external(cartpow(reals, 2))\n\
         point = external(cartpow(cartpow(reals, 2), 2))\n\
         inputs = (atom, point)\noutputs = broadcast(fn(logdensityof(Dirac(atom), _)), point)",
    )
    .unwrap();
    assert!(ir.contains("-> tensor<2xf32> {"), "{ir}");
    assert!(
        ir.contains("applies stablehlo.and across dimensions = [1]"),
        "{ir}"
    );
}
