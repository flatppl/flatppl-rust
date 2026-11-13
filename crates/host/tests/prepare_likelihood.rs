use flatppl_host::{Context, EmitOptions, Export, Schema};

#[test]
fn reduced_imported_posterior_keeps_its_scalar_score_type() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    let model = context.parse(
        "lam ~ Gamma(2.0, 1.0)\ny ~ iid(Poisson(lam), 3)\nL = likelihoodof(kernelof(record(y = y), lam = lam), record(y = [3, 1, 4]))\nposterior = bayesupdate(L, lawof(record(lam = lam)))",
        None,
        None,
    )?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        "model = load_module(\"model.flatppl\")\npoint = elementof(cartprod(lam = nonnegreals))\ninputs = point\noutputs = logdensityof(model.posterior, point)",
        None,
        None,
    )?;
    for sufficient_statistics in [false, true] {
        let options = EmitOptions {
            sufficient_statistics,
            ..Default::default()
        };
        query.compile(&options)?;
        context
            .transform(&query, true)?
            .unconstrained
            .compile(&options)?;
    }
    Ok(())
}

fn check_summary_schemas(
    prep: &Export,
    score: &Export,
    summaries: &[String],
) -> Result<(), serde_json::Error> {
    let outputs = match &prep.output {
        Schema::Tuple { items } => items.as_slice(),
        output => std::slice::from_ref(output),
    };
    assert_eq!(outputs.len(), summaries.len());
    for (output, name) in outputs.iter().zip(summaries) {
        let input = score
            .inputs
            .iter()
            .find(|field| field.name == *name)
            .unwrap();
        let mut output = serde_json::to_value(output)?;
        let mut input = serde_json::to_value(&input.value)?;
        output.as_object_mut().unwrap().remove("index");
        input.as_object_mut().unwrap().remove("index");
        assert_eq!(output, input, "preparation result and score input {name}");
    }
    Ok(())
}

#[test]
fn prepared_queries_keep_their_imports_and_compilable_summary_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let dir = std::env::temp_dir().join(format!("flatppl-prepared-data-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("scale.json"), "2.0")?;
    let model = context.parse(
        "scale = load_data(\"scale.json\", posreals)",
        None,
        dir.join("model.flatppl").to_str(),
    )?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        "m = load_module(\"model.flatppl\")\nmu = elementof(reals)\ny = external(cartpow(reals, 3))\nz = external(cartpow(reals, 3))\ninputs = (mu, y, z)\noutputs = (logdensityof(iid(Normal(mu, 2.0), 3), y), logdensityof(iid(Normal(mu, 1.0), 3), z), m.scale)",
        None,
        None,
    )?;
    let options = EmitOptions::default();
    let original = query.compile(&options)?;
    let prepared = context.prepare_likelihood(&query, &["y".into()])?;
    std::fs::remove_dir_all(dir)?;
    assert_eq!(
        original
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        ["mu", "y", "z"],
    );
    let prep = prepared.preparation.compile(&options)?;
    assert_eq!(prep.inputs.len(), 1);
    assert_eq!(prep.inputs[0].name, "y");
    let score = prepared.score.compile(&options)?;
    check_summary_schemas(&prep, &score, &prepared.summaries)?;
    let expected: Vec<_> = ["mu", "z"]
        .into_iter()
        .chain(prepared.summaries.iter().map(String::as_str))
        .collect();
    assert_eq!(
        score
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        expected,
    );
    assert!(!prepared.summaries.is_empty());
    assert_eq!(
        serde_json::to_value(&score.output)?,
        serde_json::to_value(&original.output)?,
    );
    let output = prepared
        .score
        .bindings()
        .into_iter()
        .find(|b| b.name == "outputs")
        .unwrap();
    let line = prepared
        .score
        .source()
        .lines()
        .nth(output.line.unwrap() as usize - 1)
        .unwrap();
    assert!(line.starts_with("outputs = "), "{line}");
    drop(context);
    assert_eq!(prepared.score.compile(&options)?.stablehlo, score.stablehlo);
    Ok(())
}

#[test]
fn preparing_preserves_raw_data_reached_by_another_output() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    let query = context.parse(
        "rate = elementof(posreals)\ny = load_data(\"missing.json\", cartpow(reals, 3))\ninputs = (rate, y)\noutputs = (logdensityof(iid(Exponential(rate), 3), y), y)",
        None,
        None,
    )?;
    let prepared = context.prepare_likelihood(&query, &["y".into()])?;
    let score = prepared.score.compile(&EmitOptions::default())?;
    assert!(score.inputs.iter().any(|field| field.name == "y"));
    assert!(score.multiple_outputs);
    assert_eq!(score.output_names.len(), 2);
    let prep = prepared.preparation.compile(&EmitOptions::default())?;
    check_summary_schemas(&prep, &score, &prepared.summaries)?;
    Ok(())
}

#[test]
fn prepared_boolean_counts_keep_their_integer_tensor_abi() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    let query = context.parse(
        "p = elementof(unitinterval)\ny = external(cartpow(booleans, 3))\ninputs = (p, y)\noutputs = logdensityof(iid(Bernoulli(p), 3), y)",
        None,
        None,
    )?;
    let prepared = context.prepare_likelihood(&query, &["y".into()])?;
    let options = EmitOptions::default();
    let prep = prepared.preparation.compile(&options)?;
    let score = prepared.score.compile(&options)?;
    check_summary_schemas(&prep, &score, &prepared.summaries)?;
    Ok(())
}
