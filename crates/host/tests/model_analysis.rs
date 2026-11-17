use flatppl_host::{Context, EmitOptions, ModelAnalysis};

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
