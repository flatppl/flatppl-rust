use flatppl_core::{Module, NodeId, ScalarType, Symbol, Type};
use flatppl_model_analysis::{IidNormalPlan, plan_iid_normal};

fn typed(source: &str) -> Module {
    let mut m = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    m
}

fn name(m: &Module, key: &str) -> Symbol {
    m.bindings()
        .find(|(_, b)| m.resolve(b.name) == key)
        .unwrap()
        .1
        .name
}

fn rhs(m: &Module, key: &str) -> NodeId {
    m.binding(m.binding_by_name(name(m, key)).unwrap()).rhs
}

fn plan(m: &Module, scoring: &[&str], data: &[&str]) -> Option<IidNormalPlan> {
    let names = |keys: &[&str]| keys.iter().map(|k| name(m, k)).collect::<Vec<_>>();
    plan_iid_normal(
        m,
        rhs(m, "plate"),
        rhs(m, "observed"),
        &names(scoring),
        &names(data),
    )
}

#[test]
fn structural_normal_matches_the_lowered_factor() {
    let m = typed(
        "mu=elementof(reals)\nsigma=elementof(posreals)\ny=external(cartpow(reals,3))\ng=Normal(sigma=sigma,mu=mu)\nplate=iid(g,3)\nobserved=y\ninputs=(mu,sigma,y)\noutputs=logdensityof(plate,observed)",
    );
    let plan = plan(&m, &["mu", "sigma"], &["y"]).unwrap();
    assert_eq!(plan.observation_count(), 3);
    assert_eq!(
        plan.module().type_of(plan.score()),
        Some(&Type::Scalar(ScalarType::Real))
    );
    assert_eq!(
        plan.module().type_of(plan.observations_are_finite()),
        Some(&Type::Scalar(ScalarType::Boolean))
    );
    let mut available = std::collections::HashSet::from([name(&m, "y")]);
    assert!(plan.summaries().len() > 0);
    for summary in plan.summaries() {
        assert!(summary.synthetic && plan.module().type_of(summary.rhs).is_some());
        let mut dependencies = std::collections::HashSet::new();
        plan.module()
            .collect_referenced_names(summary.rhs, &mut dependencies);
        assert!(dependencies.is_subset(&available));
        available.insert(summary.name);
    }

    let mut structural = plan.module().clone();
    structural.set_binding_rhs(
        structural.binding_by_name(name(&m, "outputs")).unwrap(),
        plan.score(),
    );
    let roots = [name(&m, "inputs"), name(&m, "outputs")];
    let lower = |query| {
        flatppl_determinizer::determinize_with_roots(query, &Default::default(), Some(&roots))
            .unwrap()
    };
    // The lowered pipeline applies the moment rule after canonicalization.
    // Canonicalize its generated arithmetic once, as for the structural score.
    assert_eq!(
        flatppl_syntax::print(&lower(&structural)),
        flatppl_syntax::print(&lower(&lower(&m)))
    );
}

#[test]
fn target_inputs_override_source_values_and_the_authored_query() {
    let source = "theta=0.5\ny=[1.,2.,3.]\na=y\nn=3\ng=Normal(mu=theta,sigma=1.)\nplate=iid(g,n)\nobserved=a\ninputs=y\noutputs=theta";
    let m = typed(source);
    assert!(plan(&m, &["theta"], &["y"]).is_some());
    assert!(plan(&m, &["y"], &[]).is_none());
    assert!(plan(&m, &["a"], &["y"]).is_none());
    assert!(plan(&m, &["theta"], &["y", "n"]).is_none());
    let m = typed(&source.replace("observed=a", "observed=[theta,2.,3.]"));
    assert!(plan(&m, &["theta"], &[]).is_none());
}

#[test]
fn unsupported_factors_and_extents_retain_the_original_measure() {
    let source = "theta=elementof(reals)\ny=[1.,2.,3.]\nplate=iid(Normal(theta,1.),3)\nobserved=y";
    for (from, to) in [
        ("Normal(theta,1.)", "LogNormal(theta,1.)"),
        ("Normal(theta,1.)", "Exponential(1.)"),
        ("1.),3)", "1.),2)"),
        ("[1.,2.,3.]", "[1,2,3]"),
    ] {
        let m = typed(&source.replace(from, to));
        assert!(plan(&m, &["theta"], &[]).is_none(), "{to}");
    }
}

#[test]
fn a_callable_data_input_cannot_hide_a_promoted_capture() {
    let m = typed("mu=0.0\nf(x)=[mu+x,mu+x,mu+x]\nplate=iid(Normal(mu,1.),3)\nobserved=f(0.0)");
    assert!(plan(&m, &["mu"], &["f"]).is_none());
    assert!(plan(&m, &["mu"], &[]).is_none());
}

#[test]
fn a_fixed_nonfinite_literal_retains_native_scoring() {
    let mut m = typed("mu=0.0\ny=[1.,2.,3.]\nplate=iid(Normal(mu,1.),3)\nobserved=y");
    let nonfinite = m.alloc(flatppl_core::Node::Lit(flatppl_core::Scalar::Real(
        f64::NAN,
    )));
    let rhs = rhs(&m, "y");
    let flatppl_core::Node::Call(mut vector) = m.node(rhs).clone() else {
        unreachable!()
    };
    vector.args[0] = nonfinite;
    let rhs = m.alloc(flatppl_core::Node::Call(vector));
    m.set_binding_rhs(m.binding_by_name(name(&m, "y")).unwrap(), rhs);
    flatppl_infer::infer(&mut m);
    assert!(plan(&m, &["mu"], &[]).is_none());
    assert!(plan(&m, &["mu"], &["y"]).is_some());
}
