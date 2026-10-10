use std::sync::Arc;

use flatppl_infer::ModuleBundle;

fn emit(source: &str, bundle: &ModuleBundle) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let (_, bundle) = flatppl_infer::infer_module_with_inputs(
        &mut model,
        bundle.clone(),
        flatppl_infer::Level::Shape,
    );
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &bundle,
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap()
}

#[test]
fn imported_function_broadcasts_at_root_and_inside_expression() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helpers.flatppl",
        Arc::new(flatppl_syntax::parse("f(x) = x + 1.0").unwrap()),
    );
    for output in ["h.f.(xs)", "sum(h.f.(xs))"] {
        let prefix = "xs = elementof(cartpow(reals, 3))\ninputs = xs\n";
        let actual = format!("h = load_module(\"helpers.flatppl\")\n{prefix}outputs = {output}");
        let expected = format!(
            "f(x) = x + 1.0\n{prefix}outputs = {}",
            output.replace("h.f", "f"),
        );
        assert_eq!(
            emit(&actual, &bundle),
            emit(&expected, &ModuleBundle::new())
        );
    }
}

#[test]
fn pyhf_helper_splats_an_opaque_record_by_name() {
    let prefix = "pyhf = standard_module(\"pyhf_helpers\", \"0.1\")\n\
                  pars = elementof(cartprod(alpha = reals, hi = posreals, lo = posreals))\n\
                  inputs = pars\noutputs = ";
    let actual = format!("{prefix}pyhf.normsys_factor(pars)");
    let expected = format!("{prefix}pyhf.normsys_factor(pars.lo, pars.hi, pars.alpha)");
    assert_eq!(
        emit(&actual, &ModuleBundle::new()),
        emit(&expected, &ModuleBundle::new())
    );
}

#[test]
fn pyhf_helpers_keep_dots_only_for_bin_broadcast_products() {
    let prefix = "pyhf = standard_module(\"pyhf_helpers\", \"0.1\")\n\
                  theta = elementof(cartpow(reals, 2))\n\
                  inputs = theta\n";
    // Per-row alphas broadcast over three bins in the three bin products.
    let histosys = emit(
        &format!(
            "{prefix}a = addaxes([theta[1], theta[2]], 0, 1)\n\
             outputs = sum(sum(broadcast(pyhf.histosys_shift, \
             rowstack([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]), \
             rowstack([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]), \
             rowstack([[3.0, 4.0, 5.0], [2.5, 3.5, 4.5]]), a)))"
        ),
        &ModuleBundle::new(),
    );
    assert_eq!(
        histosys.matches("stablehlo.dot_general").count(),
        3,
        "{histosys}"
    );
    // One alpha per lane leaves no broadcast product.
    let normsys = emit(
        &format!(
            "{prefix}outputs = sum(broadcast(pyhf.normsys_factor, \
             [0.9, 0.8], [1.1, 1.3], [theta[1], theta[2]]))"
        ),
        &ModuleBundle::new(),
    );
    assert!(!normsys.contains("stablehlo.dot_general"), "{normsys}");
}
