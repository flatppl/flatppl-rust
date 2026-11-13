use flatppl_core::{Module, Node, RefNs};
use flatppl_determinizer::{LoweringOptions, determinize_with_options, prepare_likelihood};

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

fn summary_count(module: &Module) -> usize {
    module
        .bindings()
        .filter(|(_, b)| module.resolve(b.name).starts_with("__likelihood_summary_"))
        .count()
}

fn inputs(module: &Module) -> Vec<String> {
    let binding = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "inputs")
        .unwrap()
        .1;
    let values = match module.node(binding.rhs) {
        Node::Call(c) => c.args.to_vec(),
        _ => vec![binding.rhs],
    };
    values
        .iter()
        .map(|id| match module.node(*id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => module.resolve(r.name).to_string(),
            node => panic!("{node:?}"),
        })
        .collect()
}

#[test]
fn family_reductions_are_portable_and_the_switch_preserves_the_query_abi() {
    // Each distinct family rule must produce ordinary, serializable FlatPDL.
    // Values and gradients have independent execution oracles outside this crate.
    for (law, data) in [
        ("Normal(a,b)", "[-0.5,0.2,1.25]"),
        ("LogNormal(a,b)", "[0.1,1.0,2.0]"),
        ("Gamma(a,b)", "[0.1,1.0,2.0]"),
        ("Beta(a,b)", "[0.1,0.4,0.8]"),
        ("Exponential(a)", "[0.0,0.2,3.0]"),
        ("Poisson(a)", "[0,2,3]"),
        ("Binomial(6,a)", "[0,2,6]"),
        ("Bernoulli(a)", "[true,false,true]"),
        ("Geometric(a)", "[0,1,3]"),
        ("Uniform(interval(a,b))", "[0.1,0.2,0.3]"),
    ] {
        let source = format!(
            "a=elementof(reals)\nb=elementof(reals)\ny={data}\ninputs=(a,b)\noutputs=logdensityof(iid({law},3),y)"
        );
        let mut reduced = lower(&source, true);
        assert!(
            summary_count(&reduced) > 0,
            "{law}: {}",
            flatppl_syntax::print(&reduced)
        );
        assert_eq!(inputs(&reduced), ["a", "b"]);
        let output = reduced
            .bindings()
            .find(|(_, b)| reduced.resolve(b.name) == "outputs")
            .unwrap()
            .1
            .rhs;
        assert_eq!(
            reduced.type_of(output),
            Some(&flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real)),
            "{law}"
        );
        assert_eq!(summary_count(&lower(&source, false)), 0);
        assert_eq!(flatppl_determinizer::sufficient_statistics(&mut reduced), 0);
        let serialized = flatppl_flatpir::try_write(&reduced).unwrap();
        let mut roundtrip = flatppl_flatpir::read(&serialized).unwrap();
        flatppl_infer::infer(&mut roundtrip);
        assert!(
            flatppl_determinizer::is_flatpdl(&roundtrip).is_ok(),
            "{law}"
        );
        let mut surface = flatppl_syntax::parse(&flatppl_syntax::print(&reduced)).unwrap();
        flatppl_infer::infer(&mut surface);
        assert!(flatppl_determinizer::is_flatpdl(&surface).is_ok(), "{law}");
    }
}

#[test]
fn multivariate_guards_keep_unbounded_data_and_captured_scoring_inputs_live() {
    for (parameters, law, event) in [
        (
            "mu=elementof(cartpow(reals,2))\ncov=elementof(cartpow(reals,[2,2]))\ninputs=(mu,cov,y)",
            "MvNormal(mu,cov)",
            "cartpow(reals,2)",
        ),
        (
            "nu=elementof(posreals)\nscale=elementof(cartpow(reals,[2,2]))\ninputs=(nu,scale,y)",
            "Wishart(nu,scale)",
            "cartpow(reals,[2,2])",
        ),
    ] {
        let source = format!(
            "{parameters}\ny=external(cartpow({event},4))\noutputs=logdensityof(iid({law},4),y)"
        );
        assert_eq!(summary_count(&lower(&source, true)), 0, "{law}");
    }
    let source = "alpha=elementof(cartpow(posreals,2))\nshift=elementof(cartpow(reals,2))\ny=external(cartpow(cartpow(reals,2),4))\nf(v,s)=softmax(v+s)\nz=f.(y,[shift])\ninputs=(alpha,shift,y)\noutputs=logdensityof(iid(Dirichlet(alpha),4),z)";
    assert_eq!(summary_count(&lower(source, true)), 0);
    let pure = source.replace(
        "f(v,s)=softmax(v+s)\nz=f.(y,[shift])",
        "f(v)=softmax(v)\nz=f.(y)",
    );
    assert!(summary_count(&lower(&pure, true)) > 0);
}

#[test]
fn repeated_factors_share_preparation_and_parameter_dependent_observations_stay_live() {
    let prefix = "a=elementof(reals)\ny=external(cartpow(reals,3))\ninputs=(a,y)\n";
    let one = lower(
        &format!("{prefix}outputs=logdensityof(iid(Normal(a,1.0),3),y)"),
        false,
    );
    let two = lower(
        &format!(
            "{prefix}outputs=(logdensityof(iid(Normal(a,1.0),3),y),logdensityof(iid(Normal(2.0*a,2.0),3),y))"
        ),
        false,
    );
    let one = prepare_likelihood(&one, &["y".into()]).unwrap();
    let two = prepare_likelihood(&two, &["y".into()]).unwrap();
    assert_eq!(one.summaries.len(), two.summaries.len());
    assert_eq!(inputs(&two.preparation), ["y"]);
    assert_eq!(inputs(&two.score)[0], "a");
    assert!(!inputs(&two.score).iter().any(|name| name == "y"));
    let dependent = lower(
        &format!(
            "{prefix}outputs=(logdensityof(iid(Normal(a,1.0),3),y),logdensityof(iid(Normal(a,1.0),3),y .+ a))"
        ),
        false,
    );
    let dependent = prepare_likelihood(&dependent, &["y".into()]).unwrap();
    assert!(inputs(&dependent.score).iter().any(|name| name == "y"));
    assert!(!flatppl_syntax::print(&dependent.preparation).contains("a ="));
}

#[test]
fn a_promoted_fixed_value_never_becomes_a_baked_statistic() {
    let source = "a=elementof(reals)\ny=[0.1,0.2,0.3]\ninputs=(a,y)\noutputs=logdensityof(iid(Normal(a,1.0),3),y)";
    let lowered = lower(source, true);
    let prepared = prepare_likelihood(&lowered, &["y".into()]).unwrap();
    let text = flatppl_syntax::print(&prepared.preparation);
    assert!(text.contains("y"));
    assert_eq!(inputs(&prepared.preparation), ["y"]);
    assert!(!inputs(&prepared.score).iter().any(|name| name == "y"));
}
