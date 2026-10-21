use flatppl_core::{ScalarType, Type};
use flatppl_infer::{Level, ModuleBundle, Severity, infer, infer_module};
use std::sync::Arc;

fn check(source: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(source).unwrap();
    let diagnostics = infer(&mut m);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}\n{source}"
    );
    m
}

#[test]
fn whole_records_and_named_spreads_follow_interfaces() {
    let m = check(
        "whole(r) = r.x\nchoose(f) = f\na = choose(whole)\nx = a((x = 2))\nmulti(a,b) = a+b\ny = multi((b=3,a=4))\nz = checked((condition=true,value=8))\npoint = Dirac((value=9))",
    );
    for name in ["x", "y", "z"] {
        let binding = m
            .bindings()
            .find(|(_, b)| m.resolve(b.name) == name)
            .unwrap()
            .1;
        assert_eq!(
            m.type_of(binding.rhs),
            Some(&Type::Scalar(ScalarType::Integer))
        );
    }
    let point = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == "point")
        .unwrap()
        .1;
    assert!(
        matches!(m.type_of(point.rhs), Some(Type::Measure { domain, .. }) if matches!(domain.as_ref(), Type::Record(_)))
    );
}

#[test]
fn one_argument_admission_precedes_domain_checks_and_inferred_inputs_are_named() {
    check(
        "x=elementof(reals)\nf=functionof(x+1)\nchoose(f)=f\ng=choose(f)\na=g((x=2))\noptional=diag\ny=optional(rowstack([[1.0,2.0],[3.0,4.0]]))",
    );
    for source in [
        "x=elementof(reals)\nf=functionof(x+1)\ny=f(2)",
        "choose(f)=f\nx=elementof(reals)\nf=functionof(x+1)\ng=choose(f)\ny=g(2)",
        "y=checked(2, condition=true)",
        "y=Normal(0,sigma=1)",
        "y=Normal((mu=0,wrong=1))",
    ] {
        let mut m = flatppl_syntax::parse(source).unwrap();
        assert!(
            infer(&mut m).iter().any(|d| d.severity == Severity::Error),
            "{source}"
        );
    }
}

#[test]
fn imports_preserve_ordered_and_keyword_only_interfaces() {
    let dep = flatppl_syntax::parse(
        "whole(r)=r.x\ntheta=elementof(reals)\nnamed=functionof(theta+1)\noptional=diag",
    )
    .unwrap();
    let mut bundle = ModuleBundle::new();
    bundle.insert("dep.flatppl", Arc::new(dep));
    let mut m = flatppl_syntax::parse("d=load_module(\"dep.flatppl\")\nalias=d.whole\nx=alias((x=2))\ny=d.named((theta=3))\nbad=d.named(3)\noptional=d.optional\nv=optional(rowstack([[1.0,2.0],[3.0,4.0]]))").unwrap();
    let diagnostics = infer_module(&mut m, &bundle, Level::Shape);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "{diagnostics:?}");
    assert!(errors[0].message.contains("keyword-only"));
}

#[test]
fn unresolved_broadcast_axes_stay_open_until_application() {
    check(
        "rating_probs(cuts,location)=cat(invlogit.(cuts .- location),[1.0]) .- cat([0.0],invlogit.(cuts .- location))\ncuts=elementof(ordered(8))\np=rating_probs(cuts,0.0)",
    );
}
