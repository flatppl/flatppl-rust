use flatppl_host::{
    Context, EmitOptions,
    flatppl_formula::{self, DataMode},
};

#[test]
fn formula_training_and_prediction_compile_from_distinct_snapshots()
-> Result<(), Box<dyn std::error::Error>> {
    let mut spec = flatppl_formula::read(include_str!(
        "../../formula/tests/fixtures/gaussian.formula.json"
    ))?;
    compile_from_distinct_snapshots(&spec)?;
    spec.priors.clear();
    compile_from_distinct_snapshots(&spec)
}

fn compile_from_distinct_snapshots(
    spec: &flatppl_formula::Spec,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.import_formula(spec, spec.data.as_ref().unwrap(), None, None)?;
    context.register("training.flatppl", &model.module)?;
    let score = context.parse(r#"
        m = load_module("training.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), sd_g = posreals, z_g = cartpow(reals, 2), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
    "#, None, None)?;
    let export = score.compile(&EmitOptions::default())?;
    assert_eq!(export.inputs.len(), 1);
    let mut data = spec.data.as_ref().unwrap().clone();
    data.columns.retain(|c| c.name != "y");
    let prediction = model.prepare(&mut context, &data, DataMode::Prediction)?;
    context.register("prediction.flatppl", &prediction)?;
    let forward = context.parse(r#"
        m = load_module("prediction.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), sd_g = posreals, z_g = cartpow(reals, 2), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [2.0, 4.0, 8.0]))
    "#, None, None)?;
    forward.compile(&EmitOptions::default())?;
    // Reusing the unbound template must not consume or mutate the training data.
    assert_eq!(
        score.compile(&EmitOptions::default())?.stablehlo,
        export.stablehlo
    );
    Ok(())
}
