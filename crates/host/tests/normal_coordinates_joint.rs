use flatppl_host::{Context, Dtype, EmitOptions, InputPath};

fn path(root: &str, fields: &[&str]) -> InputPath {
    InputPath {
        input: root.into(),
        fields: fields.iter().map(|s| (*s).into()).collect(),
    }
}

#[test]
fn joint_maps_keep_record_siblings_abi_order_and_topological_paths() {
    let mut context = Context::default();
    let module=context.parse("p=elementof(cartprod(x=reals,y=reals,w=reals,q=reals,s=posreals))\ninputs=p\noutputs=logdensityof(Normal(1.,2.),p.x)+logdensityof(Normal(3.*p.x+p.q,1.+p.x*p.x),p.y)+logdensityof(Normal(p.x-p.y,2.+p.y*p.y),p.w)",None,None).unwrap();
    let before = module.compile(&EmitOptions::default()).unwrap().stablehlo;
    let paths = [path("p", &["w"]), path("p", &["x"]), path("p", &["y"])];
    let plan = context
        .prepare_normal_coordinates_queries(&module, &[], &paths, true, Dtype::F64)
        .unwrap();
    assert_eq!(plan.coordinate_inputs, ["p"]);
    assert_eq!(
        plan.paths,
        [path("p", &["x"]), path("p", &["y"]), path("p", &["w"])]
    );
    let options = EmitOptions {
        dtype: Dtype::F64,
        ..Default::default()
    };
    for query in [
        &plan.score,
        &plan.forward,
        &plan.inverse,
        &plan.log_jacobian,
    ] {
        let export = query.compile(&options).unwrap();
        assert_eq!(export.inputs.len(), 1);
        assert_eq!(export.inputs[0].name, "p");
    }
    assert_eq!(
        module.compile(&EmitOptions::default()).unwrap().stablehlo,
        before
    );
}

#[test]
fn selected_cycles_refuse_but_unselected_cycles_can_be_conditioned_on() {
    let mut context = Context::default();
    let module=context.parse("a=elementof(reals)\nb=elementof(reals)\ninputs=(a,b)\noutputs=logdensityof(Normal(b,1.),a)+logdensityof(Normal(a,2.),b)",None,None).unwrap();
    assert!(
        context
            .prepare_normal_coordinates_queries(
                &module,
                &[],
                &[path("a", &[]), path("b", &[])],
                true,
                Dtype::F64
            )
            .is_err()
    );
    let single = context
        .prepare_normal_coordinates_queries(&module, &[], &[path("a", &[])], true, Dtype::F64)
        .unwrap();
    assert_eq!(single.coordinate_inputs, ["a"]);
    for paths in [vec![], vec![path("a", &[]), path("a", &[])]] {
        assert!(
            context
                .prepare_normal_coordinates_queries(&module, &[], &paths, true, Dtype::F64)
                .is_err()
        );
    }
}

#[test]
fn generated_captures_refuse_crossed_graph_cuts_without_freezing_authored_inputs() {
    let mut context = Context::default();
    for mean in ["u", "3.*x"] {
        let source = format!(
            "x=external(reals)\nalias=x\nu=x*x\ny=external(reals)\nr=external(cartpow(reals,2))\nf=functionof(x+y,t=x)\ninputs=(y,alias,u,r)\noutputs=logdensityof(Normal(1.,2.),x)+logdensityof(Normal({mean},3.),y)+sum(f.(r))"
        );
        let module = context.parse(&source, None, None).unwrap();
        let before = module.compile(&EmitOptions::default()).unwrap().stablehlo;
        assert!(
            context
                .prepare_normal_coordinates_queries(
                    &module,
                    &[],
                    &[path("y", &[]), path("alias", &[])],
                    true,
                    Dtype::F64
                )
                .is_err()
        );
        assert!(
            context
                .prepare_normal_coordinate_queries(&module, &[], "y", true, Dtype::F64)
                .is_err()
        );
        assert_eq!(
            module.compile(&EmitOptions::default()).unwrap().stablehlo,
            before
        );
    }
    let module=context.parse("x=external(reals)\ny=external(reals)\nr=external(cartpow(reals,2))\nf=functionof(x+y,t=x)\ninputs=(x,y,r)\noutputs=logdensityof(Normal(5.,3.),y)+sum(f.(r))",None,None).unwrap();
    let plan = context
        .prepare_normal_coordinates_queries(&module, &[], &[path("y", &[])], true, Dtype::F64)
        .unwrap();
    plan.score
        .compile(&EmitOptions {
            dtype: Dtype::F64,
            ..Default::default()
        })
        .unwrap();
}
