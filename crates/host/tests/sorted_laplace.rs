use flatppl_host::{Context, Dtype, EmitOptions};

#[test]
fn sorted_laplace_is_an_explicit_portable_preparation() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query=context.parse("mu=elementof(interval(-10,10))\nb=elementof(interval(.5,4))\ny=external(cartpow(interval(-20,20),5))\ninputs=(mu,b,y)\noutputs=(logdensityof(iid(Laplace(mu,b),5),y),y)",None,None)?;
    assert_eq!(
        query.compile(&EmitOptions::default())?.stablehlo,
        query
            .compile(&EmitOptions {
                sufficient_statistics: false,
                ..Default::default()
            })?
            .stablehlo
    );
    let prepared = context.prepare_likelihood(&query, &["y".into()])?;
    for dtype in [Dtype::F32, Dtype::F64] {
        let options = EmitOptions {
            dtype,
            ..Default::default()
        };
        prepared.preparation.compile(&options)?;
        let score = prepared.score.compile(&options)?;
        assert!(score.inputs.iter().any(|input| input.name == "y"));
    }
    Ok(())
}

#[test]
fn unproved_laplace_ranges_keep_ordinary_scoring() -> Result<(), Box<dyn std::error::Error>> {
    for (mu, b, y, n) in [
        ("reals", "interval(.5,4)", "interval(-20,20)", 5),
        ("interval(-10,10)", "posreals", "interval(-20,20)", 5),
        ("interval(-10,10)", "interval(.5,4)", "reals", 5),
        (
            "interval(-10,10)",
            "interval(.5,4)",
            "interval(-20,20)",
            16385,
        ),
        (
            "interval(-10,10)",
            "interval(1e-30,2e-30)",
            "interval(-20,20)",
            5,
        ),
    ] {
        let mut context = Context::default();
        let query=context.parse(&format!("mu=elementof({mu})\nb=elementof({b})\ny=external(cartpow({y},{n}))\ninputs=(mu,b,y)\noutputs=logdensityof(iid(Laplace(mu,b),{n}),y)"),None,None)?;
        assert!(context.prepare_likelihood(&query, &["y".into()]).is_err());
    }
    Ok(())
}
