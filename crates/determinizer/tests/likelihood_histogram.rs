use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Scalar};
use flatppl_determinizer::{LoweringOptions, determinize_with_options, sufficient_statistics};

fn lower(source: &str, enabled: bool) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors = flatppl_infer::infer(&mut module);
    assert!(
        !errors
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error),
        "{errors:?}"
    );
    let roots = [module.intern("inputs"), module.intern("outputs")];
    determinize_with_options(
        &module,
        &Default::default(),
        Some(&roots),
        &LoweringOptions {
            disable_sufficient_statistics: !enabled,
            ..Default::default()
        },
    )
    .unwrap()
}

fn points(module: &Module) -> Vec<NodeId> {
    let mut pending = vec![
        module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == "outputs")
            .unwrap()
            .1
            .rhs,
    ];
    let mut seen = std::collections::HashSet::new();
    let mut points = Vec::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Node::Call(call) = module.node(node)
            && matches!(call.head, CallHead::Builtin(h) if module.resolve(h) == "builtin_logdensityof")
        {
            points.push(call.args[2]);
        }
        if let Node::Ref(reference) = module.node(node)
            && reference.ns == RefNs::SelfMod
            && let Some(binding) = module.binding_by_name(reference.name)
        {
            pending.push(module.binding(binding).rhs);
        }
        pending.extend(module.node(node).children());
    }
    points
}

#[test]
fn repeated_scalar_observations_share_original_density_calls_across_families() {
    for (parameters, inputs, law, data) in [
        (
            "nu=elementof(posreals)",
            "nu",
            "StudentT(nu)",
            "[-2.0,-2.0,1.0,1.0]",
        ),
        (
            "p=elementof(stdsimplex(3))",
            "p",
            "Categorical(p)",
            "[1,1,3,3]",
        ),
        (
            "p=elementof(stdsimplex(3))",
            "p",
            "Categorical0(p)",
            "[0,0,2,2]",
        ),
        (
            "r=elementof(nonnegreals)",
            "r",
            "Poisson(r)",
            "[100,100,101,101]",
        ),
        (
            "n=elementof(posintegers)\np=elementof(unitinterval)",
            "(n,p)",
            "Binomial(n,p)",
            "[0,0,2,2]",
        ),
    ] {
        let source = format!(
            "{parameters}\ny={data}\ninputs={inputs}\noutputs=logdensityof(iid({law},4),y)"
        );
        assert_eq!(
            flatppl_flatpir::write(&lower(&source, false))
                .matches("builtin_logdensityof")
                .count(),
            1
        );
        let mut reduced = lower(&source, true);
        assert_eq!(
            points(&reduced).len(),
            2,
            "{law}: {}",
            flatppl_syntax::print(&reduced)
        );
        assert!(flatppl_determinizer::is_flatpdl(&reduced).is_ok());
        let before = flatppl_flatpir::try_write(&reduced).unwrap();
        assert_eq!(before.matches("(mul 2.0 ").count(), 2, "{law}: {before}");
        assert_eq!(sufficient_statistics(&mut reduced), 0);
        assert_eq!(flatppl_flatpir::try_write(&reduced).unwrap(), before);
        let mut roundtrip = flatppl_syntax::parse(&flatppl_syntax::print(&reduced)).unwrap();
        let errors = flatppl_infer::infer(&mut roundtrip);
        assert!(
            !errors
                .iter()
                .any(|d| d.severity == flatppl_infer::Severity::Error),
            "{law}: {errors:?}"
        );
        assert!(flatppl_determinizer::is_flatpdl(&roundtrip).is_ok());
        assert_eq!(points(&roundtrip).len(), 2);
    }
}

#[test]
fn histogram_points_keep_array_promotion_and_exact_integer_arrays() {
    let module = lower(
        "nu=elementof(posreals)\ninputs=nu\noutputs=logdensityof(iid(StudentT(nu),4),[1,1,1,1.0])",
        true,
    );
    let observed = points(&module);
    assert_eq!(observed.len(), 2);
    assert!(
        observed
            .iter()
            .any(|&id| matches!(module.node(id), Node::Call(call)
        if matches!(call.head, CallHead::Builtin(h) if module.resolve(h) == "real")
            && matches!(module.node(call.args[0]), Node::Lit(Scalar::Int(1)))))
    );
    assert!(
        observed
            .iter()
            .any(|&id| matches!(module.node(id), Node::Lit(Scalar::Real(x)) if *x == 1.0))
    );
    let pir = flatppl_flatpir::write(&module);
    assert!(
        pir.contains("(mul 3.0 ") && pir.contains("(mul 1.0 "),
        "{pir}"
    );
    for integer in [50000, 10000000000_i64] {
        let module = lower(
            &format!(
                "nu=elementof(posreals)\ninputs=nu\noutputs=logdensityof(iid(StudentT(nu),4),[{integer},{integer},1.0,1.0])"
            ),
            true,
        );
        assert!(points(&module).iter().any(|&id| matches!(module.node(id), Node::Call(call)
            if matches!(call.head, CallHead::Builtin(h) if module.resolve(h) == "real")
                && matches!(module.node(call.args[0]), Node::Lit(Scalar::Int(x)) if *x == integer))));
    }

    let module = lower(
        "r=elementof(nonnegreals)\ninputs=r\noutputs=logdensityof(iid(Poisson(r),4),[9007199254740992,9007199254740992,9007199254740993,9007199254740993])",
        true,
    );
    let observed = points(&module);
    assert_eq!(observed.len(), 2);
    for expected in [9007199254740992, 9007199254740993] {
        assert!(
            observed
                .iter()
                .any(|&id| matches!(module.node(id), Node::Lit(Scalar::Int(x)) if *x == expected))
        );
    }
}

#[test]
fn runtime_observations_and_varying_parameters_keep_cellwise_scoring() {
    for source in [
        "nu=elementof(posreals)\ny=[1.0,1.0,2.0,2.0]\ninputs=(nu,y)\noutputs=logdensityof(iid(StudentT(nu),4),y)",
        "nu=external(cartpow(posreals,4))\ninputs=nu\noutputs=logdensityof(StudentT.(nu),[1.0,1.0,2.0,2.0])",
    ] {
        let module = lower(source, true);
        let pir = flatppl_flatpir::write(&module);
        assert_eq!(pir.matches("builtin_logdensityof").count(), 1);
        assert!(
            pir.contains("(broadcast ") && pir.contains("(sum "),
            "{pir}"
        );
    }
}

#[test]
fn histogram_requires_material_compression_and_a_bounded_number_of_points() {
    for data in [vec![0, 1, 2, 3], (0..65).flat_map(|x| [x, x]).collect()] {
        let values = data
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            "nu=elementof(posreals)\ninputs=nu\noutputs=logdensityof(iid(StudentT(nu),{}),[{values}])",
            data.len()
        );
        let module = lower(&source, true);
        assert_eq!(
            flatppl_flatpir::write(&module)
                .matches("builtin_logdensityof")
                .count(),
            1
        );
    }
}
