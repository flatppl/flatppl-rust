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
        ("InverseGamma(a,b)", "[0.1,1.0,2.0]"),
        ("Pareto(a,b)", "[0.1,1.0,2.0]"),
        ("Weibull(2.0,b)", "[0.1,1.0,2.0]"),
        ("NegativeBinomial(2.0,b)", "[0,1,3]"),
        ("NegativeBinomial2(a,2.0)", "[0,1,3]"),
        ("NegativeBinomial(1e8,b)", "[0,1,3]"),
        ("NegativeBinomial2(a,1e8)", "[0,1,3]"),
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
fn large_dispersion_with_unbounded_counts_keeps_ordinary_scoring() {
    for law in ["NegativeBinomial(1e6,a)", "NegativeBinomial2(a,1e6)"] {
        let source = format!(
            "a=elementof(posreals)\ny=external(cartpow(nonnegintegers,3))\ninputs=(a,y)\noutputs=logdensityof(iid({law},3),y)"
        );
        assert_eq!(
            flatppl_syntax::print(&lower(&source, true)),
            flatppl_syntax::print(&lower(&source, false))
        );
        let raw = lower(&source, false);
        assert!(prepare_likelihood(&raw, &["y".into()]).is_err());
    }
}

#[test]
fn fixed_shape_preparation_tracks_each_shape_and_keeps_live_shape_data() {
    for (law, event) in [
        ("Weibull(k,s)", "interval(0.1,10)"),
        ("NegativeBinomial(k,s)", "nonnegintegers"),
        ("NegativeBinomial2(s,k)", "nonnegintegers"),
    ] {
        let source = format!(
            "k=elementof(interval(0.1,10))\ns=elementof(posreals)\ny=external(cartpow({event},4))\ninputs=(s,k,y)\noutputs=logdensityof(iid({law},4),y)"
        );
        let raw = lower(&source, false);
        let prepared = prepare_likelihood(&raw, &["k".into(), "y".into()]).unwrap();
        assert_eq!(inputs(&prepared.preparation), ["k", "y"]);
        assert!(!inputs(&prepared.score).iter().any(|name| name == "y"));
        let live = source.replace(
            "outputs=",
            "outputs=(logdensityof(iid(Normal(s,1.0),4),real.(y)),",
        );
        let live = format!("{live})");
        let live = prepare_likelihood(&lower(&live, false), &["y".into()]).unwrap();
        assert!(inputs(&live.score).iter().any(|name| name == "y"));
    }
    let source = "s=elementof(posreals)\ny=external(cartpow(interval(0.1,10),4))\ninputs=(s,y)\noutputs=(logdensityof(iid(Weibull(2.0,s),4),y),logdensityof(iid(Weibull(3.0,s),4),y))";
    let two = prepare_likelihood(&lower(source, false), &["y".into()]).unwrap();
    assert_eq!(two.summaries.len(), 9);
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
fn runtime_setup_requires_explicit_preparation() {
    for (source, data) in [
        (
            "beta=elementof(cartpow(reals,2))\nsigma=elementof(posreals)\nX=external(cartpow(interval(-100,100),[5,2]))\ny=external(cartpow(interval(-100,100),5))\ninputs=(beta,sigma,X,y)\noutputs=logdensityof(Normal.(X*beta,sigma),y)",
            vec!["X".into(), "y".into()],
        ),
        (
            "nu=elementof(posreals)\nscale=elementof(cartpow(reals,[2,2]))\ny=external(cartpow(cartpow(interval(-100,100),[2,2]),5))\ninputs=(nu,scale,y)\noutputs=logdensityof(iid(InverseWishart(nu,scale),5),y)",
            vec!["y".into()],
        ),
        (
            "a=elementof(posreals)\nb=elementof(posreals)\ny=external(cartpow(interval(0.1,10),5))\ninputs=(a,b,y)\noutputs=logdensityof(iid(InverseGamma(a,b),5),y)",
            vec!["y".into()],
        ),
        (
            "a=elementof(posreals)\nb=elementof(posreals)\ny=external(cartpow(interval(0.1,10),5))\ninputs=(a,b,y)\noutputs=logdensityof(iid(Pareto(a,b),5),y)",
            vec!["y".into()],
        ),
        (
            "a=elementof(posreals)\ny=external(cartpow(interval(0.1,10),5))\ninputs=(a,y)\noutputs=logdensityof(iid(Weibull(2.0,a),5),y)",
            vec!["y".into()],
        ),
        (
            "a=elementof(posreals)\ny=external(cartpow(nonnegintegers,5))\ninputs=(a,y)\noutputs=logdensityof(iid(NegativeBinomial(2.0,a),5),y)",
            vec!["y".into()],
        ),
        (
            "a=elementof(posreals)\ny=external(cartpow(nonnegintegers,5))\ninputs=(a,y)\noutputs=logdensityof(iid(NegativeBinomial2(a,2.0),5),y)",
            vec!["y".into()],
        ),
    ] {
        let ordinary = lower(source, true);
        assert_eq!(summary_count(&ordinary), 0);
        let prepared = prepare_likelihood(&ordinary, &data).unwrap();
        assert!(
            !inputs(&prepared.score)
                .iter()
                .any(|name| data.contains(name))
        );
    }
}

#[test]
fn distinct_regression_designs_keep_distinct_preparation() {
    let source = "a=elementof(reals)\nb=elementof(reals)\nx=external(cartpow(interval(-100,100),5))\nz=external(cartpow(interval(-100,100),5))\ny=external(cartpow(interval(-100,100),5))\ninputs=(a,b,x,z,y)\noutputs=(logdensityof(Normal.(a .* x,1.0),y),logdensityof(Normal.(b .* z,1.0),y))";
    let prepared =
        prepare_likelihood(&lower(source, true), &["x".into(), "z".into(), "y".into()]).unwrap();
    assert_eq!(prepared.summaries.len(), 6);
    assert_eq!(inputs(&prepared.score).len(), 8);
}

#[test]
fn integer_array_arithmetic_does_not_become_real_regression() {
    let source = "beta=elementof(integers)\nsigma=elementof(posreals)\nx=external(cartpow(integers,5))\ny=external(cartpow(interval(-100,100),5))\ninputs=(beta,sigma,x,y)\noutputs=logdensityof(Normal.(beta .* x .+ 0.0,sigma),y)";
    assert!(prepare_likelihood(&lower(source, true), &["x".into(), "y".into()]).is_err());
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
