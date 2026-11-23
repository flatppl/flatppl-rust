use flatppl_host::{BatchSpec, Context, Dtype, EmitOptions, Export, Schema};

fn summary_count(preparation: &Export) -> usize {
    match &preparation.output {
        Schema::Tuple { items } => items.len(),
        Schema::Tensor { .. } => 1,
        _ => panic!("summaries must be tensors"),
    }
}

fn main_arguments(source: &str) -> usize {
    let entry = source
        .lines()
        .find(|line| line.contains("func.func @main("))
        .unwrap();
    entry.split_once(" -> ").unwrap().0.matches("%arg").count()
}

#[test]
fn fixed_regression_uses_fewer_qr_levels_in_double_precision()
-> Result<(), Box<dyn std::error::Error>> {
    let values = (0..257)
        .map(|i| format!("{}.0", i % 31))
        .collect::<Vec<_>>()
        .join(",");
    let source = format!(
        "x=[{values}]\ndata=record(x=x,y=x)\nX=colstack([ones(257),data.x])\n\
         beta=elementof(cartpow(reals,2))\nsigma=elementof(posreals)\n\
         inputs=(beta,sigma)\noutputs=logdensityof(Normal.(X*beta,sigma),data.y)"
    );
    let query = Context::default().parse(&source, None, None)?;
    let mut loops = Vec::new();
    for dtype in [Dtype::F32, Dtype::F64] {
        let options = EmitOptions {
            dtype,
            ..Default::default()
        };
        let setup = query.compile_prepared(&options)?.preparation.unwrap();
        loops.push(setup.stablehlo.matches("stablehlo.while").count());
    }
    assert!(loops[1] > 0 && loops[1] < loops[0]);
    Ok(())
}

#[test]
fn fixed_regression_runs_qr_and_checks_the_solve_in_preparation()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("flatppl-fixed-regression-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("X.json"),
        "[[1,0],[1,1],[1,-1],[1,2],[1,3],[1,-2]]",
    )?;
    std::fs::write(dir.join("y.json"), "[0.5,1.4,-0.1,2.8,3.1,-2.0]")?;
    let mut context = Context::default();
    let query = context.parse(
        "X=load_data(\"X.json\",cartpow(interval(-10,10),[6,2]))\n\
         y=load_data(\"y.json\",cartpow(interval(-10,10),6))\n\
         beta=elementof(cartpow(reals,2))\nsigma=elementof(posreals)\n\
         inputs=(beta,sigma)\nscore=logdensityof(Normal.(X*beta,sigma),y)\noutputs=score",
        None,
        dir.join("query.flatppl").to_str(),
    )?;
    for dtype in [Dtype::F32, Dtype::F64] {
        let options = EmitOptions {
            dtype,
            check_parameters: true,
            ..Default::default()
        };
        let ordinary = query.compile(&options)?;
        let score = query.compile_prepared(&options)?;
        let setup = score.preparation.as_deref().unwrap();
        assert!(ordinary.stablehlo.contains("stablehlo.while"));
        assert!(setup.stablehlo.contains("stablehlo.while"));
        assert!(!score.stablehlo.contains("stablehlo.while"));
        assert!(setup.inputs.is_empty());
        assert_eq!(main_arguments(&setup.stablehlo), 0);
        assert_eq!(main_arguments(&score.stablehlo), 2 + summary_count(setup));
        assert_eq!(
            serde_json::to_value(&score.inputs)?,
            serde_json::to_value(&ordinary.inputs)?
        );
        assert_eq!(score.output_names, ordinary.output_names);
        let solve = setup.runtime_status.as_ref().unwrap();
        assert!(solve.codes.contains(&4));
        assert_eq!(solve.index, summary_count(setup));
        assert_eq!(score.runtime_status.as_ref().unwrap().codes, [9]);
    }
    let disabled = EmitOptions {
        sufficient_statistics: false,
        ..Default::default()
    };
    assert_eq!(
        serde_json::to_value(query.compile_prepared(&disabled)?)?,
        serde_json::to_value(query.compile(&disabled)?)?,
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn grouped_summaries_are_shared_while_a_promoted_input_remains_live()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        "p=elementof(cartpow(reals,3))\ng=[2,1,2,3,2,1]\ny=[true,false,true,false,false,true]\n\
         prior_data=load_data(\"missing-fixed-data.json\",cartpow(interval(-10,10),6))\n\
         z=logdensityof(iid(Normal(0.0,1.0),6),prior_data)\nraw=z\ninputs=(p,raw)\n\
         lp=logdensityof(Bernoulli.(p[g]),y)+logdensityof(Normal(p[1],1.0),z)\noutputs=(lp,z)",
        None,
        Some("/model/query.flatppl"),
    )?;
    let options = EmitOptions::default();
    let scalar = query.compile_prepared(&options)?;
    let setup = scalar.preparation.as_deref().unwrap();
    let count = summary_count(setup);
    let mut batch = BatchSpec {
        shape: vec![4],
        input_axes: vec![vec![Some(0)], vec![None]],
    };
    batch
        .input_axes
        .extend(std::iter::repeat_n(vec![None], count));
    let score = query.compile_batched_prepared(&options, &batch)?;
    assert_eq!(
        score.preparation.as_ref().unwrap().stablehlo,
        setup.stablehlo
    );
    assert_eq!(main_arguments(&score.stablehlo), 2 + count);
    assert_eq!(
        score
            .inputs
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["p", "raw"]
    );
    assert_eq!(score.output_names, [Some("lp".into()), Some("z".into())]);
    batch.input_axes[2][0] = Some(0);
    assert!(query.compile_batched_prepared(&options, &batch).is_err());
    Ok(())
}

#[test]
fn promoted_observations_without_fixed_summaries_use_the_ordinary_export()
-> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "mu=elementof(reals)\ny=[1.0,2.0,4.0]\nalias=y\ninputs=(mu,alias)\n\
         outputs=logdensityof(iid(Normal(mu,1.0),3),y)",
        None,
        None,
    )?;
    let options = EmitOptions::default();
    let ordinary = serde_json::to_value(query.compile(&options)?)?;
    assert!(ordinary.get("preparation").is_none());
    assert_eq!(
        serde_json::to_value(query.compile_prepared(&options)?)?,
        ordinary
    );
    Ok(())
}

#[test]
fn a_missing_output_keeps_the_ordinary_compiler_diagnostic() {
    let query = Context::default()
        .parse("x=elementof(reals)\ninputs=x", None, None)
        .unwrap();
    let options = EmitOptions::default();
    let ordinary = query.compile(&options).err().unwrap();
    let prepared = query.compile_prepared(&options).err().unwrap();
    assert_eq!(
        serde_json::to_value(prepared).unwrap(),
        serde_json::to_value(ordinary).unwrap()
    );
}
