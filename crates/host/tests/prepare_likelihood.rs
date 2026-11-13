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

#[test]
fn multivariate_preparation_replaces_observations_with_typed_event_summaries()
-> Result<(), Box<dyn std::error::Error>> {
    for (parameters, law, event) in [
        (
            "mu=elementof(cartpow(reals,2))\ncov=elementof(cartpow(reals,[2,2]))\ninputs=(mu,cov,y)",
            "MvNormal(mu,cov)",
            "cartpow(interval(-100,100),2)",
        ),
        (
            "alpha=elementof(cartpow(posreals,3))\ninputs=(alpha,y)",
            "Dirichlet(alpha)",
            "stdsimplex(3)",
        ),
        (
            "nu=elementof(posreals)\nscale=elementof(cartpow(reals,[2,2]))\ninputs=(nu,scale,y)",
            "Wishart(nu,scale)",
            "cartpow(interval(-100,100),[2,2])",
        ),
    ] {
        let mut context = Context::default();
        let query = context.parse(
            &format!(
                "{parameters}\ny=external(cartpow({event},4))\noutputs=logdensityof(iid({law},4),y)"
            ),
            None,
            None,
        )?;
        let prepared = context.prepare_likelihood(&query, &["y".into()])?;
        for dtype in [flatppl_host::Dtype::F32, flatppl_host::Dtype::F64] {
            let options = EmitOptions {
                dtype,
                ..Default::default()
            };
            let original = query.compile(&options)?;
            let prep = prepared.preparation.compile(&options)?;
            let score = prepared.score.compile(&options)?;
            check_summary_schemas(&prep, &score, &prepared.summaries)?;
            assert_eq!(prep.inputs.len(), 1);
            assert!(!score.inputs.iter().any(|field| field.name == "y"));
            assert_eq!(
                serde_json::to_value(original.output)?,
                serde_json::to_value(score.output)?
            );
            // Summary dimensions depend on the event, never the observation count.
            assert!(!score.stablehlo.contains("tensor<4x"), "{law}");
        }
    }
    Ok(())
}

#[test]
fn gaussian_preparation_pads_partial_blocks_and_underdetermined_scatter()
-> Result<(), Box<dyn std::error::Error>> {
    for (dimension, count) in [(2, 3), (3, 2)] {
        let mut context = Context::default();
        let source = format!(
            "mu=elementof(cartpow(reals,{dimension}))\ncov=elementof(cartpow(reals,[{dimension},{dimension}]))\ny=external(cartpow(cartpow(interval(-100,100),{dimension}),{count}))\ninputs=(mu,cov,y)\noutputs=logdensityof(iid(MvNormal(mu,cov),{count}),y)"
        );
        let query = context.parse(&source, None, None)?;
        let prepared = context.prepare_likelihood(&query, &["y".into()])?;
        let prep = prepared.preparation.compile(&EmitOptions::default())?;
        let score = prepared.score.compile(&EmitOptions::default())?;
        check_summary_schemas(&prep, &score, &prepared.summaries)?;
        assert!(!score.inputs.iter().any(|field| field.name == "y"));
    }
    Ok(())
}

#[test]
fn gaussian_preparation_keeps_shared_centers_outside_row_maps()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        "mu=elementof(cartpow(reals,2))\ncov=elementof(cartpow(reals,[2,2]))\ny=external(cartpow(cartpow(interval(-100,100),2),128))\ninputs=(mu,cov,y)\noutputs=logdensityof(iid(MvNormal(mu,cov),128),y)",
        None,
        None,
    )?;
    let prepared = context.prepare_likelihood(&query, &["y".into()])?;
    let prep = prepared.preparation.compile(&EmitOptions::default())?;
    // Recomputing a captured center in each mapped row creates an N-by-N tensor.
    assert!(!prep.stablehlo.contains("tensor<128x128x"));
    assert!(!prep.stablehlo.contains("tensor<128x128>"));
    // The QR workspace depends on the event dimension, not the row count.
    assert!(!prep.stablehlo.contains("tensor<128x2x2x"));
    Ok(())
}

#[test]
fn count_and_inverse_matrix_preparation_keep_the_scoring_abi()
-> Result<(), Box<dyn std::error::Error>> {
    for source in [
        "p=elementof(stdsimplex(3))\ny=external(cartpow(cartpow(integers,3),17))\ninputs=(p,y)\noutputs=logdensityof(iid(Multinomial(10,p),17),y)",
        "nu=elementof(posreals)\nscale=elementof(cartpow(reals,[2,2]))\ny=external(cartpow(cartpow(interval(-100,100),[2,2]),17))\ninputs=(nu,scale,y)\noutputs=logdensityof(iid(InverseWishart(nu,scale),17),y)",
        "a=elementof(posreals)\nb=elementof(posreals)\ny=external(cartpow(interval(0.1,10),17))\ninputs=(a,b,y)\noutputs=logdensityof(iid(InverseGamma(a,b),17),y)",
        "a=elementof(posreals)\nb=elementof(posreals)\ny=external(cartpow(interval(0.1,10),17))\ninputs=(a,b,y)\noutputs=logdensityof(iid(Pareto(a,b),17),y)",
        "a=elementof(posreals)\ny=external(cartpow(interval(0.1,10),17))\ninputs=(a,y)\noutputs=logdensityof(iid(Weibull(2.0,a),17),y)",
        "a=elementof(posreals)\ny=external(cartpow(nonnegintegers,17))\ninputs=(a,y)\noutputs=logdensityof(iid(NegativeBinomial(2.0,a),17),y)",
        "a=elementof(posreals)\ny=external(cartpow(nonnegintegers,17))\ninputs=(a,y)\noutputs=logdensityof(iid(NegativeBinomial2(a,2.0),17),y)",
    ] {
        let mut context = Context::default();
        let query = context.parse(source, None, None)?;
        let prepared = context.prepare_likelihood(&query, &["y".into()])?;
        for dtype in [flatppl_host::Dtype::F32, flatppl_host::Dtype::F64] {
            let options = EmitOptions {
                dtype,
                ..Default::default()
            };
            let prep = prepared.preparation.compile(&options)?;
            let score = prepared.score.compile(&options)?;
            check_summary_schemas(&prep, &score, &prepared.summaries)?;
            assert!(!score.inputs.iter().any(|field| field.name == "y"));
            assert!(!score.stablehlo.contains("tensor<17x"));
        }
    }
    Ok(())
}

#[test]
fn regression_preparation_keeps_matrix_and_affine_designs_outside_scoring()
-> Result<(), Box<dyn std::error::Error>> {
    for (dimension, count) in [(2, 5), (3, 2)] {
        let mut context = Context::default();
        let source = format!(
            "beta=elementof(cartpow(reals,{dimension}))\nsigma=elementof(posreals)\nX=external(cartpow(interval(-100,100),[{count},{dimension}]))\ny=external(cartpow(interval(-100,100),{count}))\ninputs=(beta,sigma,X,y)\noutputs=logdensityof(Normal.(X*beta,sigma),y)"
        );
        let query = context.parse(&source, None, None)?;
        let prepared = context.prepare_likelihood(&query, &["X".into(), "y".into()])?;
        let prep = prepared.preparation.compile(&EmitOptions::default())?;
        let score = prepared.score.compile(&EmitOptions::default())?;
        check_summary_schemas(&prep, &score, &prepared.summaries)?;
        assert_eq!(prepared.summaries.len(), 3);
        assert!(
            !score
                .inputs
                .iter()
                .any(|field| matches!(field.name.as_str(), "X" | "y"))
        );
        assert!(!prep.stablehlo.contains(&format!("tensor<{count}x{count}x")));
        // A design that remains a scoring input must retain ordinary scoring.
        assert!(context.prepare_likelihood(&query, &["y".into()]).is_err());
    }
    let mut context = Context::default();
    let query = context.parse("a=elementof(reals)\nb=elementof(reals)\nsigma=elementof(posreals)\nx=external(cartpow(interval(-100,100),6))\noffset=external(cartpow(interval(-100,100),6))\ny=external(cartpow(interval(-100,100),6))\ninputs=(a,b,sigma,x,offset,y)\noutputs=logdensityof(Normal.(offset .+ a .+ b .* x,sigma),y)", None, None)?;
    let prepared =
        context.prepare_likelihood(&query, &["x".into(), "offset".into(), "y".into()])?;
    let score = prepared.score.compile(&EmitOptions::default())?;
    assert!(
        !score
            .inputs
            .iter()
            .any(|field| matches!(field.name.as_str(), "x" | "offset" | "y"))
    );
    let constructed = context.parse("beta=elementof(cartpow(reals,2))\nsigma=elementof(posreals)\nx=external(cartpow(interval(-100,100),5))\ny=external(cartpow(interval(-100,100),5))\nX=colstack([fill(1.0,5),x])\ninputs=(beta,sigma,x,y)\noutputs=logdensityof(Normal.(X*beta,sigma),y)", None, None)?;
    let prepared = context.prepare_likelihood(&constructed, &["x".into(), "y".into()])?;
    prepared.preparation.compile(&EmitOptions::default())?;
    let score = prepared.score.compile(&EmitOptions::default())?;
    assert!(
        !score
            .inputs
            .iter()
            .any(|field| matches!(field.name.as_str(), "x" | "y"))
    );
    Ok(())
}
