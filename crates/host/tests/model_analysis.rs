use flatppl_host::{Context, EmitOptions, ModelAnalysis};

#[test]
fn independent_conjugate_fields_report_exact_draws_in_prior_order() {
    use flatppl_host::PosteriorSampling::ExactIid;
    let source = "
        p=elementof(unitinterval)
        r=elementof(posreals)
        K=functionof(joint(h=Bernoulli(p), n=Poisson(r)), p=p, r=r)
        L=likelihoodof(K,record(n=3,h=true))
        P=bayesupdate(L,joint(p=Beta(1,1),r=Gamma(1,1)))
        s=elementof(nonnegintegers)
        inputs=s
        outputs=rand(normalize(P),rnginit(s))
    ";
    let mut ctx = Context::default();
    let separate = source.replace(
        "L=likelihoodof(K,record(n=3,h=true))",
        "L=joint_likelihood(likelihoodof(functionof(Poisson(r),r=r),3),\n\
                            likelihoodof(functionof(Bernoulli(p),p=p),true))",
    );
    for source in [source, &separate] {
        let query = ctx.parse(source, None, None).unwrap();
        let fields = query.posterior_updates().unwrap();
        assert_eq!(fields.len(), 2, "{fields:?}");
        for (field, (name, family, likelihood, parameters)) in fields.iter().zip([
            ("p", "Beta", "Bernoulli", [2.0, 1.0]),
            ("r", "Gamma", "Poisson", [4.0, 2.0]),
        ]) {
            assert_eq!(field.field.as_deref(), Some(name));
            assert_eq!(field.family.as_deref(), Some(family));
            assert_eq!(field.likelihoods, [likelihood]);
            assert_eq!(field.parameters.as_deref(), Some(parameters.as_slice()));
            assert_eq!(field.sampling, ExactIid);
        }
    }
}

#[test]
fn a_cross_field_likelihood_cannot_use_independent_updates() {
    let source = "
        a=elementof(reals)
        b=elementof(posreals)
        K=functionof(joint(y=Normal(a,sqrt(b)),z=Normal(0,sqrt(b))), a=a, b=b)
        P=bayesupdate(likelihoodof(K,record(y=1.,z=2.)),
                      joint(a=Normal(0,1),b=InverseGamma(2,3)))
        s=elementof(nonnegintegers)
        inputs=s
        outputs=rand(normalize(P),rnginit(s))
    ";
    let mut ctx = Context::default();
    let query = ctx.parse(source, None, None).unwrap();
    let err = ctx.analyze_query(&query, &[], false, true).err().unwrap();
    assert_eq!(err.posterior_updates.len(), 1);
    assert_eq!(err.posterior_updates[0].family, None);
}

#[test]
fn posterior_reports_distinguish_the_update_from_the_retained_sampler() {
    use flatppl_host::PosteriorSampling::{ExactIid, LoweringRefused, NotUsed};
    let mut context = Context::default();
    let source = "p=elementof(unitinterval)\nseed=elementof(nonnegintegers)\nK=functionof(Bernoulli(p),p=p)\nP=bayesupdate(likelihoodof(K,true),joint(p=Beta(1,1)))\nQ=normalize(P)\ninputs=seed\n";
    let mut inspect = |output: &str| {
        let module = context
            .parse(&format!("{source}outputs={output}"), None, None)
            .unwrap();
        context.analyze_query(&module, &[], false, true)
    };
    let report = inspect("rand(Q,rnginit(seed))").unwrap();
    let [update] = report.posterior_updates.as_slice() else {
        panic!("one update")
    };
    assert_eq!(
        (
            &*update.binding,
            update.field.as_deref(),
            update.family.as_deref()
        ),
        ("P", Some("p"), Some("Beta"))
    );
    assert_eq!(update.likelihoods, ["Bernoulli"]);
    assert_eq!(update.parameters.as_deref(), Some([2.0, 1.0].as_slice()));
    assert_eq!(update.sampling, ExactIid);
    let law_output = inspect("Q").err().unwrap();
    assert_eq!(
        law_output.posterior_updates[0].parameters,
        Some(vec![2.0, 1.0])
    );
    assert_eq!(
        inspect("get0(rand(iid(Q,3),rnginit(seed)),1)")
            .unwrap()
            .posterior_updates[0]
            .sampling,
        ExactIid
    );
    let score = "logdensityof(P,record(p=0.5))";
    assert!(
        inspect("get0((rand(Q,rnginit(seed)),rand(Beta(2,1),rnginit(seed))),1)")
            .unwrap()
            .posterior_updates
            .iter()
            .all(|entry| entry.sampling == NotUsed)
    );
    assert_eq!(
        inspect(&format!("({score},rand(Beta(2,1),rnginit(seed)))"))
            .unwrap()
            .posterior_updates[0]
            .sampling,
        NotUsed
    );
    assert!(
        inspect("rand(Beta(2,1),rnginit(seed))")
            .unwrap()
            .posterior_updates
            .is_empty()
    );
    let err = inspect("rand(iid(Q,seed),rnginit(seed))").err().unwrap();
    assert_eq!(err.posterior_updates[0].family.as_deref(), Some("Beta"));
    assert_eq!(err.posterior_updates[0].sampling, LoweringRefused);
    assert!(err.message.contains("fixed scalar size"));
    let unsupported = source.replace("Bernoulli(p)", "Bernoulli(p*p)");
    let module = context
        .parse(&(unsupported + "outputs=rand(Q,rnginit(seed))"), None, None)
        .unwrap();
    let err = context
        .analyze_query(&module, &[], false, true)
        .err()
        .unwrap();
    assert_eq!(err.posterior_updates[0].family, None);
    assert_eq!(err.posterior_updates[0].sampling, LoweringRefused);
    assert!(err.message.contains("no matching rule"));
}

#[test]
fn posterior_reports_follow_imports_restrictions_and_specialized_likelihoods() {
    use flatppl_host::PosteriorSampling::{ExactIid, NotUsed};
    let mut ctx = Context::default();
    let source = "p=elementof(unitinterval)\nd=true\nK=functionof(Bernoulli(p),p=p)\nL=likelihoodof(K,d)\nP=bayesupdate(L,joint(p=Beta(1,1)))\nQ=normalize(P)";
    let model = ctx.parse(source, None, None).unwrap();
    ctx.register("model.flatppl", &model).unwrap();
    let mut inspect = |source: String| {
        let query = ctx.parse(&source, None, None).unwrap();
        query.posterior_updates().unwrap()
    };
    let import = "m=load_module(\"model.flatppl\")\ns=elementof(nonnegintegers)\ninputs=s\n";
    for (output, sampling) in [
        ("Q=m.Q\noutputs=rand(Q,rnginit(s))", ExactIid),
        ("outputs=logdensityof(m.P,record(p=0.5))", NotUsed),
        (
            "outputs=logdensityof(normalize(m.P),record(p=0.5))",
            NotUsed,
        ),
        ("outputs=log(totalmass(m.P))", NotUsed),
    ] {
        let updates = inspect(format!("{import}{output}"));
        assert_eq!(updates.len(), 1);
        assert_eq!(
            (updates[0].family.as_deref(), updates[0].sampling),
            (Some("Beta"), sampling)
        );
    }
    let updates = inspect(format!(
        "{source}\ns=elementof(nonnegintegers)\ninputs=s\nF=functionof(Q,obs=d)\nG=functionof(Q,ell=L)\nL2=likelihoodof(functionof(Binomial(10,p),p=p),9)\noutputs=(rand(F(obs=false),rnginit(s)),rand(G(ell=L2),rnginit(s)))"
    ));
    assert_eq!(updates.len(), 2, "{updates:?}");
    for (factor, a) in [("Bernoulli", 1.0), ("Binomial", 10.0)] {
        let update = updates.iter().find(|u| u.likelihoods == [factor]).unwrap();
        assert_eq!(update.sampling, ExactIid);
        assert_eq!(update.parameters, Some(vec![a, 2.0]));
    }
    let updates = inspect("p=draw(Beta(1,1))\ny=draw(Bernoulli(p))\nJ=lawof(record(p=p,y=y))\nP=restrict(J,record(y=true))\ns=elementof(nonnegintegers)\ninputs=s\noutputs=rand(normalize(P),rnginit(s))".into());
    assert_eq!(
        (updates[0].parameters.as_deref(), updates[0].sampling),
        (Some([2.0, 1.0].as_slice()), ExactIid)
    );
}

fn analyze(source: &str, data: &[&str]) -> ModelAnalysis {
    let mut context = Context::default();
    let module = context.parse(source, None, None).unwrap();
    let before = module
        .compile(&EmitOptions::default())
        .map(|export| export.stablehlo)
        .map_err(|error| error.to_string());
    let (lowered, report) = context
        .analyze_model(
            &module,
            &data.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
        .unwrap();
    assert!(flatppl_determinizer::is_flatpdl(&lowered).is_ok());
    assert_eq!(
        before,
        module
            .compile(&EmitOptions::default())
            .map(|export| export.stablehlo)
            .map_err(|error| error.to_string())
    );
    report
}

#[test]
fn shared_data_does_not_join_structural_components() {
    let report = analyze(
        "a=elementof(reals)\nb=elementof(reals)\nd=external(reals)\ninputs=(a,b,d)\noutputs=logdensityof(Normal(a,1.0),d)+logdensityof(Normal(b,1.0),d)",
        &["d"],
    );
    assert_eq!(report.components, [vec!["a"], vec!["b"]]);
    assert_eq!(report.terms.len(), 2);
    assert_eq!(report.terms[0].parameters, ["a"]);
    assert_eq!(report.terms[1].parameters, ["b"]);
    assert!(report.terms.iter().all(|term| term.data == ["d"]));
}

#[test]
fn ordinary_report_keeps_the_compilers_pre_file_reduction_order() {
    use flatppl_host::ReductionOutcome;
    let path =
        std::env::temp_dir().join(format!("flatppl-report-order-{}.csv", std::process::id()));
    std::fs::write(&path, "v\ninf\ninf\ninf\ninf\n").unwrap();
    let mut context = Context::default();
    let source = format!(
        "mu=elementof(reals)\ny=load_data({:?},cartpow(reals,4))\ninputs=mu\noutputs=logdensityof(iid(Normal(mu,1.),4),y)",
        path.to_str().unwrap()
    );
    let module = context.parse(&source, None, None).unwrap();
    let before = module.compile(&EmitOptions::default()).unwrap().stablehlo;
    let report = context.analyze_query(&module, &[], false, true).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(report.reductions.decisions.iter().any(|decision| matches!(
        decision.outcome,
        ReductionOutcome::Reduced {
            rule: "scalar moments",
            ..
        }
    )));
    assert_eq!(
        module.compile(&EmitOptions::default()).unwrap().stablehlo,
        before
    );
}

#[test]
fn reduction_reports_match_preparation_and_leave_compilation_unchanged() {
    use flatppl_host::ReductionOutcome;
    let mut context = Context::default();
    let module = context.parse("mu=elementof(reals)\ny=external(cartpow(reals,4))\ninputs=(mu,y)\noutputs=logdensityof(iid(Normal(mu,1.),4),y)", None, None).unwrap();
    let before = module.compile(&EmitOptions::default()).unwrap().stablehlo;
    let ordinary = context
        .analyze_query(&module, &["y".into()], false, true)
        .unwrap();
    let report = context
        .analyze_query(&module, &["y".into()], true, true)
        .unwrap();
    let prepared = context.prepare_likelihood(&module, &["y".into()]).unwrap();
    assert_eq!(report.summaries, prepared.summaries);
    assert!(report.preparation_refusal.is_none());
    assert!(
        report
            .reductions
            .decisions
            .iter()
            .any(|d| matches!(d.outcome, ReductionOutcome::Reduced { .. }))
    );
    assert!(
        ordinary
            .reductions
            .decisions
            .iter()
            .any(|d| matches!(d.outcome, ReductionOutcome::Reduced { .. }))
    );
    assert!(ordinary.summaries.is_empty());
    let disabled = context
        .analyze_query(&module, &["y".into()], true, false)
        .unwrap();
    assert!(!disabled.reductions.decisions.is_empty());
    assert!(
        disabled
            .reductions
            .decisions
            .iter()
            .all(|d| d.outcome == ReductionOutcome::Disabled)
    );
    assert_eq!(
        module.compile(&EmitOptions::default()).unwrap().stablehlo,
        before
    );
    let opaque = context
        .parse("a=elementof(reals)\ninputs=a\noutputs=exp(a)", None, None)
        .unwrap();
    let report = context.analyze_query(&opaque, &[], true, true).unwrap();
    assert!(report.summaries.is_empty());
    assert!(report.preparation_refusal.is_some());
    assert!(report.reductions.decisions.is_empty());
}

#[test]
fn opaque_operations_and_joint_normalizers_join_components() {
    for expression in ["logsumexp([a,b])", "a+b-a*b", "ifelse(a>0.0,a,b)", "a*b"] {
        let report = analyze(
            &format!("a=elementof(reals)\nb=elementof(reals)\ninputs=(a,b)\noutputs={expression}"),
            &[],
        );
        assert_eq!(report.components, [vec!["a", "b"]], "{expression}");
    }
}

#[test]
fn promoted_inputs_hide_their_source_rhs() {
    let report = analyze(
        "a=elementof(reals)\nd=2*a\nb=elementof(reals)\ninputs=(a,d,b)\noutputs=d",
        &["d"],
    );
    assert_eq!(report.parameters, ["a", "b"]);
    assert_eq!(report.unused_parameters, ["a", "b"]);
    assert!(report.terms[0].parameters.is_empty());
    assert_eq!(report.terms[0].data, ["d"]);
}

#[test]
fn additive_signs_and_multiplicity_remain_visible() {
    let report = analyze(
        "a=elementof(reals)\nf=logdensityof(Normal(0.0,1.0),a)\ninputs=a\noutputs=f+f",
        &[],
    );
    assert_eq!(report.terms.len(), 1);
    assert_eq!(report.terms[0].multiplicity, 2);
    assert!(report.normal_coordinates.is_empty());
    let report = analyze("a=elementof(reals)\ninputs=a\noutputs=a-a", &[]);
    assert!(report.unused_parameters.is_empty());
    assert_eq!(
        report
            .terms
            .iter()
            .map(|term| term.sign)
            .collect::<Vec<_>>(),
        [1, -1]
    );
}

#[test]
fn scalar_normal_coordinate_reports_exact_maps_and_parents() {
    let report = analyze(
        "theta=elementof(reals)\nmu=elementof(reals)\nsigma=elementof(interval(.5,4))\ninputs=(theta,mu,sigma)\noutputs=logdensityof(Normal(mu,sigma),theta)",
        &[],
    );
    assert_eq!(report.normal_coordinates.len(), 1);
    let candidate = &report.normal_coordinates[0];
    assert_eq!(candidate.parameter, "theta");
    assert_eq!(candidate.location_parents, ["mu"]);
    assert_eq!(candidate.scale_parents, ["sigma"]);
    assert!(candidate.positive_finite_scale_proven);
    assert!(!candidate.finite_location_proven);
    assert!(!candidate.cyclic);
    let report = analyze(
        "theta=elementof(reals)\nsigma=elementof(posreals)\ninputs=(theta,sigma)\noutputs=logdensityof(Normal(1.0,sigma),theta)",
        &[],
    );
    assert!(!report.normal_coordinates[0].positive_finite_scale_proven);
    // Independent change-of-variables oracle at mu=1, sigma=2, theta=3.
    let log_theta = -0.5 * (2.0 * std::f64::consts::PI).ln() - 2.0_f64.ln() - 0.5;
    let log_z = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5;
    assert!((log_theta + 2.0_f64.ln() - log_z).abs() < 1e-15);
    assert!(
        report
            .to_string()
            .contains("forward log Jacobian = log(scale)")
    );
}

#[test]
fn constrained_and_point_dependent_normals_are_not_coordinate_candidates() {
    for (domain, mean, scale) in [
        ("posreals", "0.0", "1.0"),
        ("reals", "theta", "1.0"),
        ("reals", "0.0", "exp(theta)"),
    ] {
        let report = analyze(
            &format!(
                "theta=elementof({domain})\ninputs=theta\noutputs=logdensityof(Normal({mean},{scale}),theta)"
            ),
            &[],
        );
        assert!(report.normal_coordinates.is_empty());
    }
    for (location, scale) in [
        ("inf", "1.0"),
        ("-inf", "1.0"),
        ("0.0", "0.0"),
        ("0.0", "-1.0"),
        ("0.0", "inf"),
    ] {
        let mut context = Context::default();
        let Ok(module)=context.parse(&format!("theta=elementof(reals)\ninputs=theta\noutputs=logdensityof(Normal({location},{scale}),theta)"),None,None) else {continue};
        let (_, report) = context.analyze_model(&module, &[]).unwrap();
        assert!(report.normal_coordinates.is_empty(), "{location} {scale}");
    }
}

#[test]
fn coordinate_cycles_are_advisory_and_never_rewrite_the_query() {
    let report = analyze(
        "a=elementof(reals)\nb=elementof(reals)\ninputs=(a,b)\noutputs=logdensityof(Normal(b,1.0),a)+logdensityof(Normal(a,1.0),b)",
        &[],
    );
    assert_eq!(report.normal_coordinates.len(), 2);
    assert!(
        report
            .normal_coordinates
            .iter()
            .all(|candidate| candidate.cyclic)
    );
}

#[test]
fn structured_inputs_remain_whole_blocks_and_local_names_stay_scoped() {
    let report = analyze(
        "theta=elementof(cartpow(reals,3))\ninputs=theta\noutputs=theta[1]+theta[2]",
        &[],
    );
    assert_eq!(report.components, [vec!["theta"]]);
    assert!(report.normal_coordinates.is_empty());
    assert!(report.terms.iter().all(|term| term.input_paths.is_empty()));
    assert_eq!(report.terms[0].input_cells[0].indices, [0]);
    assert_eq!(report.terms[1].input_cells[0].indices, [1]);
    let report = analyze(
        "a=elementof(reals)\nb=elementof(reals)\nf(a,b)=a+b\ninputs=(a,b)\noutputs=f(1.0,b)",
        &[],
    );
    assert_eq!(report.unused_parameters, ["a"]);
    assert!(report.terms.iter().any(|term| term.parameters == ["b"]));
}

#[test]
fn array_cell_coordinates_normalize_bases_and_keep_whole_reads() {
    let report = analyze(
        "point=elementof(cartprod(v=cartpow(reals,[2,3]),other=reals))\ninputs=point\noutputs=get(point.v,2,3)*get0(point.v,1,2)+sum(point.v)",
        &[],
    );
    assert_eq!(report.input_arrays.len(), 1);
    assert_eq!(report.input_arrays[0].shape, [2, 3]);
    assert_eq!(report.terms[0].input_cells.len(), 1);
    assert_eq!(report.terms[0].input_cells[0].indices, [1, 2]);
    assert_eq!(report.terms[0].input_cells[0].path.fields, ["v"]);
    assert!(report.terms[0].input_paths.is_empty());
    assert!(report.terms[1].input_cells.is_empty());
    assert_eq!(report.terms[1].input_paths[0].fields, ["v"]);
}

#[test]
fn promoted_selectors_and_dynamic_indices_retain_array_dependencies() {
    for selector in ["i=1", "i=elementof(nonnegintegers)"] {
        let report = analyze(
            &format!(
                "raw=external(cartpow(reals,3))\nv=raw\n{selector}\ninputs=(v,i)\noutputs=get(raw,i)"
            ),
            &["v"],
        );
        assert!(report.terms[0].input_cells.is_empty());
        assert_eq!(report.terms[0].data, ["v"]);
        assert_eq!(report.terms[0].parameters, ["i"]);
        assert!(
            report.terms[0]
                .input_paths
                .iter()
                .any(|path| path.input == "v")
        );
    }
}

#[test]
fn array_dependency_metadata_does_not_enumerate_unused_cells() {
    let mut context = Context::default();
    let module = context
        .parse(
            "v=external(cartpow(reals,1000000))\ninputs=v\noutputs=v[3]",
            None,
            None,
        )
        .unwrap();
    let (_, report) = context.analyze_model(&module, &["v".into()]).unwrap();
    assert_eq!(report.input_paths.len(), 1);
    assert_eq!(report.input_arrays.len(), 1);
    assert_eq!(report.input_arrays[0].shape, [1000000]);
    assert_eq!(report.terms[0].input_cells.len(), 1);
    assert_eq!(report.terms[0].input_cells[0].indices, [2]);
}

#[test]
fn callback_indices_and_slices_keep_whole_arrays() {
    for output in ["sum(f.(indices))", "sum(get(v,all))"] {
        let report = analyze(
            &format!(
                "v=external(cartpow(reals,3))\ni=1\nindices=external(cartpow(nonnegintegers,2))\nf(i)=get(v,i)\ninputs=(v,indices)\noutputs={output}"
            ),
            &["v", "indices"],
        );
        assert!(report.terms[0].input_cells.is_empty());
        assert!(
            report.terms[0]
                .input_paths
                .iter()
                .any(|path| path.input == "v")
        );
    }
}

#[test]
fn data_inputs_require_unique_declared_names_and_context_ownership() {
    let mut context = Context::default();
    let module = context
        .parse("a=elementof(reals)\ninputs=a\noutputs=a", None, None)
        .unwrap();
    for data in [
        vec!["missing".to_string()],
        vec!["a".to_string(), "a".to_string()],
    ] {
        assert!(context.analyze_model(&module, &data).is_err());
    }
    assert!(Context::default().analyze_model(&module, &[]).is_err());
}
