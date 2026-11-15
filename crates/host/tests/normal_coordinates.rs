use flatppl_host::{Context, Dtype, EmitOptions};

#[test]
fn selected_coordinate_keeps_other_slots_and_original_query() {
    let mut context = Context::default();
    let module = context.parse("mu=elementof(reals)\ntheta=elementof(reals)\ns=elementof(posreals)\ny=external(reals)\ninputs=(mu,theta,s,y)\noutputs=logdensityof(Normal(mu,s),theta)+logdensityof(Normal(theta,3.0),y)", None, None).unwrap();
    let options = EmitOptions {
        dtype: Dtype::F32,
        ..Default::default()
    };
    let before = module.compile(&options).unwrap().stablehlo;
    let companions = context
        .prepare_normal_coordinate(&module, &["y".into()], "theta", true, &options)
        .unwrap();
    assert_eq!(module.compile(&options).unwrap().stablehlo, before);
    for query in [
        &companions.score,
        &companions.forward,
        &companions.log_jacobian,
    ] {
        assert_eq!(
            query
                .inputs
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>(),
            ["mu", companions.coordinate_input.as_str(), "s", "y"]
        );
    }
    assert_eq!(
        companions
            .inverse
            .inputs
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["mu", "theta", "s", "y"]
    );
}

#[test]
fn one_conditional_map_does_not_compose_cyclic_candidates() {
    let mut context = Context::default();
    let module = context.parse("a=elementof(reals)\nb=elementof(reals)\ninputs=(a,b)\noutputs=logdensityof(Normal(b,1.0),a)+logdensityof(Normal(a,2.0),b)", None, None).unwrap();
    let (_, report) = context.analyze_model(&module, &[]).unwrap();
    assert!(report.normal_coordinates.iter().all(|c| c.cyclic));
    let companions = context
        .prepare_normal_coordinate(&module, &[], "a", true, &EmitOptions::default())
        .unwrap();
    assert_eq!(companions.score.inputs.len(), 2);
    assert_eq!(companions.score.inputs[1].name, "b");
}

#[test]
fn promoted_fixed_input_keeps_its_authored_boundary_after_substitution() {
    let mut context = Context::default();
    let module = context.parse("mu=elementof(reals)\ntheta=external(reals)\ns=elementof(posreals)\ny=theta*theta\ninputs=(mu,theta,s,y)\noutputs=logdensityof(Normal(mu,s),theta)+logdensityof(Normal(theta,3.0),y)", None, None).unwrap();
    let options = EmitOptions::default();
    let before = module.compile(&options).unwrap().stablehlo;
    let companions = context
        .prepare_normal_coordinate(&module, &["y".into()], "theta", true, &options)
        .unwrap();
    assert_eq!(companions.score.inputs[3].name, "y");
    assert_eq!(module.compile(&options).unwrap().stablehlo, before);
}

#[test]
fn fixed_capture_remains_fixed_when_the_map_uses_parameter_inputs() {
    let mut context = Context::default();
    let module = context.parse("mu=elementof(reals)\ntheta=external(reals)\ns=elementof(posreals)\nr=external(cartpow(reals,2))\nf(x)=x*x+theta\ninputs=(mu,theta,s,r)\noutputs=logdensityof(Normal(mu,s),theta)+sum(f.(r))", None, None).unwrap();
    let options = EmitOptions::default();
    let before = module.compile(&options).unwrap().stablehlo;
    let companions = context
        .prepare_normal_coordinate(&module, &["r".into()], "theta", true, &options)
        .unwrap();
    assert_eq!(
        companions
            .score
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        ["mu", companions.coordinate_input.as_str(), "s", "r"]
    );
    assert_eq!(module.compile(&options).unwrap().stablehlo, before);
}
