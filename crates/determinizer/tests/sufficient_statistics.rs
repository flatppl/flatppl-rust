use flatppl_core::{Module, Node, RefNs};
use flatppl_determinizer::{LoweringOptions, determinize_with_options, prepare_likelihood};

fn prepare_unpacked(module: &Module, data: &[String]) -> flatppl_determinizer::PreparedLikelihood {
    flatppl_determinizer::prepare_likelihood_with_options(
        module,
        data,
        &flatppl_determinizer::PreparationOptions {
            pack_summaries: false,
        },
        None,
    )
    .unwrap()
}

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
        ("ChiSquared(a)", "[0.1,1.0,2.0]"),
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
fn promoted_group_indices_and_circular_points_never_become_static_guards() {
    for source in [
        "mu=elementof(cartpow(reals,2))\ng=[1,2,1,2]\ny=[1.0,2.0,3.0,4.0]\ninputs=(mu,g)\noutputs=logdensityof(Normal.(mu[g],1.0),y)",
        "mu=0.3\nk=2.0\ny=[0.1,0.2,0.4]\ninputs=(mu,k)\noutputs=logdensityof(iid(VonMises(mu,k),3),y)",
        "p=elementof(cartpow(reals,3))\ng=external(cartpow(integers,5))\ny=external(cartpow(booleans,5))\ninputs=(p,g,y)\noutputs=logdensityof(Bernoulli.(p[g]),y)",
    ] {
        assert_eq!(
            flatppl_flatpir::write(&lower(source, true)),
            flatppl_flatpir::write(&lower(source, false))
        );
    }
}

#[test]
fn runtime_predictor_guards_keep_unsafe_adjoints_and_nonpointwise_values_ordinary() {
    for (set, expression, extra_inputs) in [
        ("interval(-87,-86)", "invlogit.(a[g])", ""),
        ("interval(-100,-99)", "invlogit.(a[g])", ""),
        ("interval(-1,1)", "invlogit.(a[g] .+ a[h])", ""),
        ("interval(-1,1)", "invlogit.(a[g] .+ get0(a,g))", ""),
        ("interval(-1,1)", "softmax(a[g])", ""),
        ("interval(-1,1)", "q", ",q"),
    ] {
        let source = format!(
            "a=elementof(cartpow({set},3))\ng=external(cartpow(integers,8))\nh=external(cartpow(integers,8))\ny=external(cartpow(booleans,8))\nq=invlogit.(a[g])\ninputs=(a,g,h,y{extra_inputs})\noutputs=logdensityof(Bernoulli.({expression}),y)"
        );
        let raw = lower(&source, false);
        assert!(prepare_likelihood(&raw, &["g".into(), "h".into(), "y".into()]).is_err());
        assert_eq!(
            flatppl_flatpir::write(&raw),
            flatppl_flatpir::write(&lower(&source, true))
        );
    }
    for expression in ["1e38 .* exp.(a[g])", "1e38 .* invlogit.(a[g])"] {
        let source = format!(
            "a=elementof(cartpow(interval(-100,-99),3))\ng=external(cartpow(integers,8))\ny=external(cartpow(integers,8))\ninputs=(a,g,y)\noutputs=logdensityof(Poisson.({expression}),y)"
        );
        assert!(prepare_likelihood(&lower(&source, false), &["g".into(), "y".into()]).is_err());
    }
    let quotient = "a=elementof(cartpow(interval(1e37,1e37),2))\nb=elementof(interval(1e10,1e10))\ng=external(cartpow(integers,64))\ny=external(cartpow(interval(1,1),64))\ninputs=(a,b,g,y)\noutputs=logdensityof(Exponential.(a[g] ./ b),y)";
    assert!(prepare_likelihood(&lower(quotient, false), &["g".into(), "y".into()]).is_err());
    let mut deep = "a=elementof(cartpow(interval(-1,1),3))\ng=external(cartpow(integers,8))\ny=external(cartpow(booleans,8))\nt0=a[g]\n".to_owned();
    for index in 1..=80 {
        deep.push_str(&format!("t{index}=invlogit.(t{})\n", index - 1));
    }
    deep.push_str("inputs=(a,g,y)\noutputs=logdensityof(Bernoulli.(t80),y)");
    assert!(prepare_likelihood(&lower(&deep, false), &["g".into(), "y".into()]).is_err());
}

#[test]
fn runtime_predictors_reuse_family_buffers_and_keep_scoring_parameters_live() {
    for (law, event, buffers) in [
        ("Bernoulli.(invlogit.(a[g] .+ b))", "booleans", 1),
        ("Binomial.(6,invlogit.(a[g] .+ b))", "integers", 1),
        ("Poisson.(exp.(a[g] .+ b))", "integers", 4),
        ("Geometric.(invlogit.(a[g] .+ b))", "integers", 3),
        ("Exponential.(exp.(a[g] .+ b))", "interval(.1,10)", 2),
        ("Gamma.(exp.(a[g]),exp.(a[g] .+ b))", "interval(.1,10)", 2),
        ("Beta.(exp.(a[g]),exp.(a[g] .+ b))", "interval(.1,.9)", 2),
        ("Normal.(exp.(a[g] .+ b),1.0)", "interval(-10,10)", 3),
    ] {
        let source = format!(
            "a=elementof(cartpow(interval(-1,1),3))\nb=elementof(interval(-1,1))\ng=external(cartpow(integers,8))\ny=external(cartpow({event},8))\ninputs=(a,b,g,y)\noutputs=logdensityof({law},y)"
        );
        let prepared = prepare_unpacked(&lower(&source, false), &["g".into(), "y".into()]);
        assert_eq!(prepared.summaries.len(), buffers, "{law}");
        assert_eq!(&inputs(&prepared.score)[..2], ["a", "b"]);
        assert!(
            !inputs(&prepared.score)
                .iter()
                .any(|name| name == "g" || name == "y")
        );
        for module in [&prepared.preparation, &prepared.score] {
            let mut surface = flatppl_syntax::parse(&flatppl_syntax::print(module)).unwrap();
            let errors = flatppl_infer::infer(&mut surface);
            assert!(
                !errors
                    .iter()
                    .any(|d| d.severity == flatppl_infer::Severity::Error),
                "{law}: {errors:?}"
            );
            assert!(flatppl_determinizer::is_flatpdl(&surface).is_ok());
        }
    }
}

#[test]
fn runtime_positive_families_prepare_portable_empty_group_statistics() {
    for (law, event) in [
        ("Exponential.(a[g])", "interval(0,10)"),
        ("Gamma.(a[g],b[g])", "interval(.01,10)"),
        ("Beta.(a[g],b[g])", "interval(.01,.99)"),
    ] {
        let source = format!(
            "a=elementof(cartpow(interval(.1,8),6))\nb=elementof(cartpow(interval(.1,8),6))\ng=external(cartpow(integers,4))\ny=external(cartpow({event},4))\ninputs=(a,b,g,y)\noutputs=logdensityof({law},y)"
        );
        let raw = lower(&source, false);
        let prepared = prepare_unpacked(&raw, &["g".into(), "y".into()]);
        assert!(
            !inputs(&prepared.score)
                .iter()
                .any(|name| name == "y" || name == "g")
        );
        for module in [&prepared.preparation, &prepared.score] {
            let mut printed = flatppl_syntax::parse(&flatppl_syntax::print(module)).unwrap();
            let errors = flatppl_infer::infer(&mut printed);
            assert!(
                !errors
                    .iter()
                    .any(|d| d.severity == flatppl_infer::Severity::Error),
                "{law}: {errors:?}"
            );
            assert!(flatppl_determinizer::is_flatpdl(&printed).is_ok());
            assert!(flatppl_flatpir::read(&flatppl_flatpir::try_write(module).unwrap()).is_ok());
        }
        assert_eq!(
            flatppl_flatpir::write(&raw),
            flatppl_flatpir::write(&lower(&source, true))
        );
    }
}

#[test]
fn runtime_positive_ranges_and_partition_identity_preserve_ordinary_fallbacks() {
    for (family, parameter, event, second, groups, count) in [
        ("Gamma", "interval(.1,65)", "interval(.01,10)", "g", 3, 4),
        ("Beta", "interval(.1,65)", "interval(.01,.99)", "g", 3, 4),
        ("Gamma", "interval(.1,8)", "interval(0,10)", "g", 3, 4),
        ("Beta", "interval(.1,8)", "unitinterval", "g", 3, 4),
        ("Gamma", "interval(.1,8)", "interval(.01,10)", "h", 3, 4),
        ("Gamma", "reals", "interval(.01,10)", "g", 3, 4),
        ("Gamma", "interval(.1,8)", "interval(.01,10)", "g", 65, 4),
        (
            "Gamma",
            "interval(.1,8)",
            "interval(.01,10)",
            "g",
            64,
            131072,
        ),
    ] {
        let source = format!(
            "a=elementof(cartpow({parameter},{groups}))\nb=elementof(cartpow(interval(.1,8),{groups}))\ng=external(cartpow(integers,{count}))\nh=external(cartpow(integers,{count}))\ny=external(cartpow({event},{count}))\ninputs=(a,b,g,h,y)\noutputs=logdensityof({family}.(a[g],b[{second}]),y)"
        );
        let raw = lower(&source, false);
        assert!(prepare_likelihood(&raw, &["g".into(), "h".into(), "y".into()]).is_err());
        assert_eq!(
            flatppl_flatpir::write(&raw),
            flatppl_flatpir::write(&lower(&source, true))
        );
    }
}

#[test]
fn runtime_count_moments_keep_integer_centers_and_boundary_fallbacks() {
    for (family, parameter) in [
        ("Poisson", "interval(.1,1e20)"),
        ("Geometric", "interval(.01,.99)"),
    ] {
        let source = format!(
            "p=elementof(cartpow({parameter},6))\ng=external(cartpow(integers,4))\ny=external(cartpow(integers,4))\ninputs=(p,g,y)\noutputs=logdensityof({family}.(p[g]),y)"
        );
        let prepared = prepare_unpacked(&lower(&source, false), &["g".into(), "y".into()]);
        assert!(
            !inputs(&prepared.score)
                .iter()
                .any(|name| name == "g" || name == "y")
        );
        for module in [&prepared.preparation, &prepared.score] {
            let mut printed = flatppl_syntax::parse(&flatppl_syntax::print(module)).unwrap();
            let errors = flatppl_infer::infer(&mut printed);
            assert!(
                !errors
                    .iter()
                    .any(|d| d.severity == flatppl_infer::Severity::Error),
                "{family}: {errors:?}"
            );
            assert!(flatppl_determinizer::is_flatpdl(&printed).is_ok());
        }
    }
    for (family, parameter) in [("Poisson", "interval(0,10)"), ("Geometric", "unitinterval")] {
        let source = format!(
            "p=elementof(cartpow({parameter},3))\ng=external(cartpow(integers,4))\ny=external(cartpow(integers,4))\ninputs=(p,g,y)\noutputs=logdensityof({family}.(p[g]),y)"
        );
        assert!(prepare_likelihood(&lower(&source, false), &["g".into(), "y".into()]).is_err());
    }
}

#[test]
fn runtime_binomial_histograms_keep_endpoints_and_invalid_outcomes_native() {
    let source = "p=elementof(cartpow(unitinterval,3))\ng=external(cartpow(integers,4))\ny=external(cartpow(integers,4))\ninputs=(p,g,y)\noutputs=logdensityof(Binomial.(6,p[g]),y)";
    let prepared = prepare_unpacked(&lower(source, false), &["g".into(), "y".into()]);
    assert_eq!(prepared.summaries.len(), 1);
    assert_eq!(inputs(&prepared.score).len(), 2);
    for module in [&prepared.preparation, &prepared.score] {
        let mut printed = flatppl_syntax::parse(&flatppl_syntax::print(module)).unwrap();
        let errors = flatppl_infer::infer(&mut printed);
        assert!(
            !errors
                .iter()
                .any(|d| d.severity == flatppl_infer::Severity::Error),
            "{errors:?}"
        );
        assert!(flatppl_determinizer::is_flatpdl(&printed).is_ok());
    }
}

#[test]
fn runtime_group_count_cache_tracks_extents_and_index_origins() {
    for (groups, second, buffers) in [(3, "q[g]", 1), (2, "q[g]", 2), (3, "get0(q,g)", 2)] {
        let source = format!(
            "p=elementof(cartpow(reals,3))\nq=elementof(cartpow(reals,{groups}))\ng=external(cartpow(integers,5))\ny=external(cartpow(booleans,5))\ninputs=(p,q,g,y)\noutputs=(logdensityof(Bernoulli.(p[g]),y),logdensityof(Bernoulli.({second}),y),g,y)"
        );
        let prepared =
            prepare_likelihood(&lower(&source, false), &["g".into(), "y".into()]).unwrap();
        assert_eq!(prepared.summaries.len(), buffers);
        assert!(inputs(&prepared.score).iter().any(|name| name == "g"));
        assert!(inputs(&prepared.score).iter().any(|name| name == "y"));
        let serialized = flatppl_flatpir::try_write(&prepared.score).unwrap();
        let mut roundtrip = flatppl_flatpir::read(&serialized).unwrap();
        flatppl_infer::infer(&mut roundtrip);
        assert!(flatppl_determinizer::is_flatpdl(&roundtrip).is_ok());
    }
}

#[test]
fn runtime_grouping_keeps_small_or_unproved_partitions_live() {
    for (groups, count, data) in [
        (64, 2, vec!["g".into(), "y".into()]),
        (65, 128, vec!["g".into(), "y".into()]),
        (3, 5, vec!["y".into()]),
    ] {
        let source = format!(
            "p=elementof(cartpow(reals,{groups}))\ng=external(cartpow(integers,{count}))\ny=external(cartpow(booleans,{count}))\ninputs=(p,g,y)\noutputs=logdensityof(Bernoulli.(p[g]),y)"
        );
        assert!(prepare_likelihood(&lower(&source, false), &data).is_err());
    }
}

#[test]
fn runtime_gaussian_grouping_retains_unproved_or_overflowing_ranges() {
    for (mu, sigma, y, groups, count, second_index) in [
        ("reals", "interval(0.5,4)", "interval(-20,20)", 3, 8, "g"),
        (
            "interval(0,0)",
            "interval(1e10,2e10)",
            "interval(1e24,2e24)",
            3,
            8,
            "g",
        ),
        (
            "interval(0,0)",
            "interval(1e-30,2e-30)",
            "interval(0,0)",
            3,
            8,
            "g",
        ),
        (
            "interval(0,0)",
            "interval(1e20,2e20)",
            "interval(1e20,2e20)",
            3,
            8,
            "g",
        ),
        (
            "interval(-1,1)",
            "interval(1e38,3e38)",
            "interval(-3e38,3e38)",
            3,
            8,
            "g",
        ),
        (
            "interval(1e20,2e20)",
            "interval(1e-20,2e-20)",
            "interval(1,2)",
            3,
            8,
            "g",
        ),
        (
            "interval(-10,10)",
            "interval(0.5,4)",
            "interval(-20,20)",
            64,
            131072,
            "g",
        ),
        (
            "interval(-10,10)",
            "interval(0.5,4)",
            "interval(-20,20)",
            3,
            8,
            "h",
        ),
    ] {
        let source = format!(
            "mu=elementof(cartpow({mu},{groups}))\nsigma=elementof(cartpow({sigma},{groups}))\ng=external(cartpow(integers,{count}))\nh=external(cartpow(integers,{count}))\ny=external(cartpow({y},{count}))\ninputs=(mu,sigma,g,h,y)\noutputs=logdensityof(Normal.(mu[g],sigma[{second_index}]),y)"
        );
        let ordinary = lower(&source, false);
        assert!(prepare_likelihood(&ordinary, &["g".into(), "h".into(), "y".into()]).is_err());
        assert_eq!(
            flatppl_flatpir::write(&ordinary),
            flatppl_flatpir::write(&lower(&source, true))
        );
    }
}

#[test]
fn runtime_gaussian_moment_cache_tracks_family_extent_and_origin() {
    for (family, groups, second, buffers) in [
        ("Normal", 3, "b[g]", 3),
        ("Normal", 2, "b[g]", 5),
        ("LogNormal", 3, "b[g]", 5),
        ("Normal", 3, "get0(b,g)", 5),
    ] {
        let second_scale = if second.starts_with("get0") {
            "get0(t,g)"
        } else {
            "t[g]"
        };
        let source = format!(
            "a=elementof(cartpow(interval(-10,10),3))\ns=elementof(cartpow(interval(.5,4),3))\nb=elementof(cartpow(interval(-10,10),{groups}))\nt=elementof(cartpow(interval(.5,4),{groups}))\ng=external(cartpow(integers,8))\ny=external(cartpow(interval(.01,100),8))\ninputs=(a,s,b,t,g,y)\noutputs=(logdensityof(Normal.(a[g],s[g]),y),logdensityof({family}.({second},{second_scale}),y))"
        );
        let prepared = prepare_unpacked(&lower(&source, false), &["g".into(), "y".into()]);
        assert_eq!(prepared.summaries.len(), buffers);
        let serialized = flatppl_flatpir::try_write(&prepared.score).unwrap();
        let mut roundtrip = flatppl_flatpir::read(&serialized).unwrap();
        flatppl_infer::infer(&mut roundtrip);
        assert!(flatppl_determinizer::is_flatpdl(&roundtrip).is_ok());
    }
}

#[test]
fn grouped_gathers_prepare_only_occupied_partitions() {
    for source in [
        "mu=elementof(cartpow(reals,4))\nsigma=elementof(posreals)\ng=[1,2,1,2,3]\ny=external(cartpow(reals,5))\ninputs=(mu,sigma,y)\noutputs=logdensityof(Normal.(mu[g],sigma),y)",
        "mu=elementof(cartpow(reals,4))\nsigma=elementof(cartpow(posreals,3))\ng=[1,2,1,2,3]\nh=[1,1,1,1,3]\ny=external(cartpow(reals,5))\ninputs=(mu,sigma,y)\noutputs=logdensityof(Normal.(get.([mu],g),sigma[h]),y)",
        "rate=elementof(cartpow(nonnegreals,4))\ng=[1,2,1,2,3]\ny=external(cartpow(nonnegintegers,5))\ninputs=(rate,y)\noutputs=logdensityof(Poisson.(rate[g]),y)",
    ] {
        let raw = lower(source, false);
        let prepared =
            prepare_likelihood(&raw, &["y".into()]).unwrap_or_else(|e| panic!("{source}: {e:?}"));
        assert_eq!(inputs(&prepared.preparation), ["y"]);
        assert!(!inputs(&prepared.score).iter().any(|s| s == "y"));
        assert!(flatppl_determinizer::is_flatpdl(&prepared.score).is_ok());
        let serialized = flatppl_flatpir::try_write(&prepared.score).unwrap();
        assert!(flatppl_flatpir::read(&serialized).is_ok());
    }
}

#[test]
fn grouped_scalar_families_reuse_their_ordinary_guards() {
    for (law, event) in [
        ("LogNormal.(a[g],b[g])", "interval(0.1,10)"),
        ("Gamma.(a[g],b[g])", "interval(0.1,10)"),
        ("Beta.(a[g],b[g])", "interval(0.1,0.9)"),
        ("Exponential.(a[g])", "interval(0,10)"),
        ("Bernoulli.(a[g])", "booleans"),
        ("Geometric.(a[g])", "nonnegintegers"),
        ("ChiSquared.(a[g])", "interval(0.1,10)"),
        ("InverseGamma.(a[g],b[g])", "interval(0.1,10)"),
        ("Pareto.(a[g],b[g])", "interval(0.1,10)"),
        ("Weibull.([2.0,3.0,4.0,5.0][g],b[g])", "interval(0.1,10)"),
        (
            "NegativeBinomial.([2.0,3.0,4.0,5.0][g],b[g])",
            "nonnegintegers",
        ),
        (
            "NegativeBinomial2.(a[g],[2.0,3.0,4.0,5.0][g])",
            "nonnegintegers",
        ),
    ] {
        let source = format!(
            "a=elementof(cartpow(posreals,4))\nb=elementof(cartpow(posreals,4))\ng=[1,2,1,2,3]\ny=external(cartpow({event},5))\ninputs=(a,b,y)\noutputs=logdensityof({law},y)"
        );
        let prepared = prepare_likelihood(&lower(&source, false), &["y".into()])
            .unwrap_or_else(|e| panic!("{law}: {e:?}"));
        assert!(
            !inputs(&prepared.score).iter().any(|name| name == "y"),
            "{law}"
        );
    }
    let fixed_binomial = "p=elementof(cartpow(interval(0.01,0.99),4))\ng=[1,2,1,2,3]\ninputs=p\noutputs=logdensityof(Binomial.(6,p[g]),[0,1,2,3,4])";
    assert!(summary_count(&lower(fixed_binomial, true)) > 0);
}

#[test]
fn grouped_indexed_lambdas_and_derived_points_prepare() {
    for expression in [
        "Normal.(mu[g],1.0)",
        "Normal.(get0.([mu],[0,1,0,1,2,0]),1.0)",
        "f.(g,[mu])",
        "h.([3,1,6,4,2,5],[mu])",
    ] {
        let source = format!(
            "mu=elementof(cartpow(reals,4))\ng=[1,2,1,2,3,1]\nf(i,m)=Normal(m[i],1.0)\nh(i,m)=Normal(m[g[i]],1.0)\ny=external(cartpow(reals,6))\nz=external(cartpow(reals,6))\nshift(v,w)=v+w\npoint=shift.(y,z)\ninputs=(mu,y,z)\noutputs=(logdensityof({expression},point),y)"
        );
        let prepared = prepare_likelihood(&lower(&source, false), &["y".into(), "z".into()])
            .unwrap_or_else(|e| panic!("{expression}: {e:?}"));
        assert_eq!(inputs(&prepared.preparation), ["y", "z"]);
        assert!(inputs(&prepared.score).iter().any(|name| name == "y"));
        assert!(!inputs(&prepared.score).iter().any(|name| name == "z"));
        let scoring = source
            .replace("shift(v,w)=v+w", "shift(v,w,p)=v+w+p")
            .replace("shift.(y,z)", "shift.(y,z,mu[1])");
        assert!(prepare_likelihood(&lower(&scoring, false), &["y".into(), "z".into()]).is_err());
    }
    let zero = "mu=elementof(cartpow(reals,4))\ny=external(cartpow(reals,6))\nf(i,m)=Normal(get0(m,i),1.0)\ninputs=(mu,y)\noutputs=logdensityof(f.([0,1,0,1,2,0],[mu]),y)";
    assert!(prepare_likelihood(&lower(zero, false), &["y".into()]).is_ok());
    let source = "mu=elementof(cartpow(reals,4))\ng=[1,2,1,2,3,1]\ny=external(cartpow(reals,6))\nz=external(cartpow(reals,6))\nscore(i,m,v,w)=logdensityof(Normal(m[i],1.0),v+w)\ninputs=(mu,y,z)\noutputs=sum(score.(g,[mu],y,z))";
    let prepared = prepare_likelihood(&lower(source, false), &["y".into(), "z".into()]).unwrap();
    assert_eq!(inputs(&prepared.preparation), ["y", "z"]);
    assert!(
        !inputs(&prepared.score)
            .iter()
            .any(|name| name == "y" || name == "z")
    );
    // A promoted literal selector and a data-only dynamic selector both stay live.
    let promoted = source.replace("inputs=(mu,y,z)", "inputs=(mu,y,z,g)");
    for data in [
        vec!["y".into(), "z".into()],
        vec!["y".into(), "z".into(), "g".into()],
    ] {
        assert!(prepare_likelihood(&lower(&promoted, false), &data).is_err());
    }
}

#[test]
fn reduction_reports_trace_committed_rules_without_changing_the_module() {
    use flatppl_determinizer::{
        ReductionOutcome, ReductionReport, prepare_likelihood_with_report,
        sufficient_statistics_with_report,
    };
    let source = "mu=elementof(cartpow(reals,4))\ng=[1,2,1,2,3]\ny=[1.0,2.0,3.0,4.0,5.0]\ninputs=mu\noutputs=logdensityof(Normal.(mu[g],1.0),y)";
    let raw = lower(source, false);
    let mut traced = raw.clone();
    let mut report = ReductionReport::default();
    assert_eq!(
        sufficient_statistics_with_report(&mut traced, &mut report),
        3
    );
    assert_eq!(report.decisions.len(), 1);
    assert!(
        matches!(&report.decisions[0].outcome, ReductionOutcome::Reduced { partition: Some(p), .. } if (p.observations,p.groups,p.singletons)==(5,3,1))
    );
    assert_eq!(
        flatppl_flatpir::write(&traced),
        flatppl_flatpir::write(&lower(source, true))
    );
    let unique = source
        .replace("g=[1,2,1,2,3]", "g=[1,2,3,4]")
        .replace("y=[1.0,2.0,3.0,4.0,5.0]", "y=[1.0,2.0,3.0,4.0]");
    let mut module = lower(&unique, false);
    let mut retained = ReductionReport::default();
    assert_eq!(
        sufficient_statistics_with_report(&mut module, &mut retained),
        0
    );
    assert!(
        retained.to_string().contains("occurs only once"),
        "{retained}"
    );
    assert_eq!(summary_count(&module), 0);
    let mut disabled = ReductionReport::default();
    disabled.record_disabled(&raw);
    assert!(matches!(
        disabled.decisions[0].outcome,
        ReductionOutcome::Disabled
    ));

    for (replacement, reason) in [
        ("g=[1,2,3,4,1]", ""),
        ("g=[1,2,1,2,9]", "in-range gather"),
        ("g=[1,2,3,4,5]", "in-range gather"),
    ] {
        let changed = source.replace("g=[1,2,1,2,3]", replacement);
        let mut module = lower(&changed, false);
        let mut report = ReductionReport::default();
        sufficient_statistics_with_report(&mut module, &mut report);
        if reason.is_empty() {
            assert!(matches!(
                report.decisions[0].outcome,
                ReductionOutcome::Reduced { .. }
            ));
        } else {
            assert!(report.to_string().contains(reason), "{report}");
        }
    }
    // A late bucket fails the NB range guard. No bucket or summary commits.
    let source = "b=elementof(cartpow(posreals,3))\ng=[1,2,1,2,3,3]\nk=[2.0,3.0,1e6]\ny=external(cartpow(nonnegintegers,6))\ninputs=(b,y)\noutputs=logdensityof(NegativeBinomial.(k[g],b[g]),y)";
    let raw = lower(source, false);
    let mut report = ReductionReport::default();
    assert!(prepare_likelihood_with_report(&raw, &["y".into()], &mut report).is_err());
    assert_eq!(report.decisions.len(), 1);
    assert!(
        report
            .to_string()
            .contains("dispersion <= 64 or fixed counts < 64"),
        "{report}"
    );
    let mut ordinary = raw.clone();
    sufficient_statistics_with_report(&mut ordinary, &mut ReductionReport::default());
    assert_eq!(summary_count(&ordinary), 0);
    assert_eq!(
        flatppl_flatpir::write(&ordinary),
        flatppl_flatpir::write(&raw)
    );
}

#[test]
fn bounded_circular_and_correlation_families_prepare() {
    for source in [
        "mu=elementof(interval(-3.14,3.14))\nkappa=elementof(interval(0.1,64))\ny=external(cartpow(interval(-4,4),3))\ninputs=(mu,kappa,y)\noutputs=logdensityof(iid(VonMises(mu,kappa),3),y)",
        "eta=elementof(posreals)\ny=external(cartpow(cartpow(reals,[3,3]),4))\ninputs=(eta,y)\noutputs=logdensityof(iid(LKJ(3,eta),4),y)",
        "eta=elementof(posreals)\ny=external(cartpow(cartpow(reals,[3,3]),4))\ninputs=(eta,y)\noutputs=logdensityof(iid(LKJCholesky(3,eta),4),y)",
    ] {
        let prepared = prepare_likelihood(&lower(source, false), &["y".into()]).unwrap();
        assert!(!inputs(&prepared.score).iter().any(|name| name == "y"));
        assert!(flatppl_determinizer::is_flatpdl(&prepared.score).is_ok());
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
    let two = prepare_unpacked(&lower(source, false), &["y".into()]);
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
fn weighted_regression_keeps_unproved_scale_products_and_ranges_unchanged() {
    for (sigma, scales, data) in [
        ("posreals", "interval(0.1,10)", "interval(-100,100)"),
        ("interval(0.1,10)", "interval(-1,10)", "interval(-100,100)"),
        (
            "interval(1e30,1e31)",
            "interval(1e10,1e11)",
            "interval(-100,100)",
        ),
        (
            "interval(1e-30,1e-29)",
            "interval(1e-20,1e-19)",
            "interval(-100,100)",
        ),
        (
            "interval(1e20,2e20)",
            "interval(1e-40,2e-40)",
            "interval(-100,100)",
        ),
        (
            "interval(1e29,1e30)",
            "interval(1e-30,2e-30)",
            "interval(-100,100)",
        ),
        (
            "interval(1e-20,2e-20)",
            "interval(1e20,2e20)",
            "interval(-100,100)",
        ),
        ("interval(0.5,2)", "interval(1,3)", "interval(-2e8,2e8)"),
        (
            "interval(0.1,10)",
            "interval(1e-20,1e20)",
            "interval(-100,100)",
        ),
        (
            "interval(0.1,10)",
            "interval(1e-20,1)",
            "interval(-1e30,1e30)",
        ),
    ] {
        let source = format!(
            "beta=elementof(cartpow(reals,2))\nsigma=elementof({sigma})\nX=external(cartpow({data},[5,2]))\ny=external(cartpow({data},5))\ns=external(cartpow({scales},5))\ninputs=(beta,sigma,X,y,s)\noutputs=logdensityof(Normal.(X*beta,sigma .* s),y)"
        );
        let ordinary = lower(&source, false);
        assert!(prepare_likelihood(&ordinary, &["X".into(), "y".into(), "s".into()]).is_err());
        assert_eq!(
            flatppl_flatpir::write(&lower(&source, true)),
            flatppl_flatpir::write(&ordinary)
        );
    }
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
    let one = prepare_unpacked(&one, &["y".into()]);
    let two = prepare_unpacked(&two, &["y".into()]);
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
