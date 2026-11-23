use flatppl_host::{BatchSpec, Context, Dtype, EmitOptions};

#[test]
fn repeated_predictors_prepare_without_runtime_observations()
-> Result<(), Box<dyn std::error::Error>> {
    for (family, link, support, features) in [
        (
            "Bernoulli",
            "invlogit",
            "booleans",
            "[-1.0,-1.0,0.0,0.0,1.0,1.0]",
        ),
        ("Poisson", "exp", "nonnegintegers", "[-1,-1,0,0,1,1]"),
        (
            "Bernoulli",
            "invlogit",
            "booleans",
            "[2.0,2.0,2.0,2.0,2.0,2.0]",
        ),
    ] {
        let mut context = Context::default();
        let query=context.parse(&format!("a=elementof(reals)\nb=elementof(reals)\nx={features}\ny=external(cartpow({support},6))\ninputs=(a,b,y)\noutputs=logdensityof({family}.({link}.(a .+ b .* x)),y)"),None,None)?;
        let prepared = context.prepare_likelihood(&query, &["y".into()])?;
        assert!(!prepared.summaries.is_empty());
        for dtype in [Dtype::F32, Dtype::F64] {
            let options = EmitOptions {
                dtype,
                ..Default::default()
            };
            prepared.preparation.compile(&options)?;
            let score = prepared.score.compile(&options)?;
            assert!(!score.inputs.iter().any(|field| field.name == "y"));
        }
    }
    Ok(())
}

#[test]
fn promoted_predictor_columns_keep_their_runtime_values() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    let query=context.parse("a=elementof(reals)\nb=elementof(reals)\nx=[-1.0,-1.0,0.0,0.0,1.0,1.0]\ny=external(cartpow(booleans,6))\ninputs=(a,b,x,y)\noutputs=logdensityof(Bernoulli.(invlogit.(a .+ b .* x)),y)",None,None)?;
    assert!(
        context
            .prepare_likelihood(&query, &["x".into(), "y".into()])
            .is_err()
    );
    assert_eq!(
        query.compile(&EmitOptions::default())?.stablehlo,
        query
            .compile(&EmitOptions {
                sufficient_statistics: false,
                ..Default::default()
            })?
            .stablehlo
    );
    Ok(())
}

#[test]
fn prepared_binary_groups_keep_logits_across_host_batches() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    for (group, support) in [
        ("Bernoulli.(p)", "booleans"),
        ("Binomial.([5,5,5,5,5,5],p)", "nonnegintegers"),
    ] {
        let source = format!(
            "a=elementof(reals)\nx=[-1.0,-1.0,0.0,0.0,1.0,1.0]\n\
            p=invlogit.(a .+ x)\ny=external(cartpow({support},6))\n\
            inputs=(a,y)\noutputs=logdensityof({group},y)"
        );
        let query = context.parse(&source, None, None)?;
        let prepared = context.prepare_likelihood(&query, &["y".into()])?;
        assert!(!prepared.summaries.is_empty());
        let options = EmitOptions::default();
        let scalar = prepared.score.compile(&options)?;
        let axes = scalar
            .inputs
            .iter()
            .map(|f| vec![(f.name == "a").then_some(0)])
            .collect();
        let batch = prepared.score.compile_batched(
            &options,
            &BatchSpec {
                shape: vec![4],
                input_axes: axes,
            },
        )?;
        for export in [scalar, batch] {
            assert!(
                !export.stablehlo.contains("stablehlo.log "),
                "{}",
                export.stablehlo
            );
        }
    }
    Ok(())
}

#[test]
fn fixed_group_predictors_prepare_with_both_index_origins() -> Result<(), Box<dyn std::error::Error>>
{
    for predictor in [
        "a[[1,1,2,2,3,3]]",
        "get0(a,[0,0,1,1,2,2])",
        "get.([a],[1,1,2,2,3,3])",
    ] {
        let mut context = Context::default();
        let query=context.parse(&format!("a=elementof(cartpow(reals,3))\nb=elementof(reals)\nx=[-1,-1,0,0,1,1]\ny=external(cartpow(booleans,6))\ninputs=(a,b,y)\noutputs=logdensityof(Bernoulli.(invlogit.({predictor} .+ b .* x)),y)"),None,None)?;
        let prepared = context.prepare_likelihood(&query, &["y".into()])?;
        assert!(!prepared.summaries.is_empty());
        prepared.score.compile(&EmitOptions::default())?;
    }
    Ok(())
}

#[test]
fn unsupported_predictors_leave_the_original_score_unchanged()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query=context.parse("a=elementof(reals)\nb=elementof(reals)\nx=[1.0,1.0,2.0,2.0,3.0,3.0]\ny=[0.1,0.1,1.0,2.0,3.0,4.0]\ninputs=(a,b)\noutputs=logdensityof(StudentT.(a .+ b .* x),y)",None,None)?;
    assert_eq!(
        query.compile(&EmitOptions::default())?.stablehlo,
        query
            .compile(&EmitOptions {
                sufficient_statistics: false,
                ..Default::default()
            })?
            .stablehlo
    );
    Ok(())
}
