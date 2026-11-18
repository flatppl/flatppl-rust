//! Exact posterior samplers consume one native draw and preserve the public variate.
use flatppl_core::{CallHead, Module, Node, Type};
use flatppl_determinizer::{determinize_with_options, is_flatpdl};
use flatppl_infer::{ModuleBundle, Severity};
use flatppl_stablehlo::{Dtype, EmitOptions, LOWERING_OPTIONS, Mode, Target};

fn model(body: &str, measure: &str) -> String {
    format!(
        "{body}\nseed = elementof(nonnegintegers)\nstate = rnginit(seed)\n\
             result, next = rand({measure}, state)\ninputs = seed\noutputs = (result, next)"
    )
}

fn inferred(source: &str) -> Module {
    let mut m = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{source}");
    m
}

fn binding_type<'a>(m: &'a Module, name: &str) -> &'a Type {
    let (_, binding) = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap();
    m.type_of(binding.rhs).unwrap()
}

fn check_sample(source: &str, family: &str) {
    let mut m = inferred(source);
    let roots = [m.intern("inputs"), m.intern("outputs")];
    let lowered =
        determinize_with_options(&m, &ModuleBundle::new(), Some(&roots), &LOWERING_OPTIONS)
            .unwrap();
    assert!(is_flatpdl(&lowered).is_ok());
    for name in ["result", "next"] {
        assert_eq!(
            binding_type(&lowered, name),
            binding_type(&m, name),
            "{source}"
        );
    }
    let mut pending: Vec<_> = lowered.bindings().map(|(_, b)| b.rhs).collect();
    let mut seen = std::collections::HashSet::new();
    let mut draws = Vec::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = lowered.node(id);
        if let Node::Call(c) = node
            && matches!(c.head, CallHead::Builtin(h) if lowered.resolve(h) == "builtin_sample")
            && let Node::Const(ctor) = lowered.node(c.args[1])
        {
            draws.push(lowered.resolve(*ctor));
        }
        pending.extend(node.children());
    }
    assert_eq!(draws, [family], "{source}");
    for target in [Target::Cpu, Target::Gpu] {
        flatppl_stablehlo::emit(
            &lowered,
            Mode::Sample,
            &EmitOptions {
                dtype: Dtype::F64,
                target,
                restrict_enzyme_compatible: false,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn fixed_batches_use_the_native_updated_law() {
    for (domain, prior, likelihood, obs, ctor) in [
        (
            "unitinterval",
            "Beta(2,3)",
            "Bernoulli(t)",
            "[true,false]",
            "Beta",
        ),
        (
            "unitinterval",
            "Beta(2,3)",
            "Binomial(5,t)",
            "[2,1]",
            "Beta",
        ),
        ("posreals", "Gamma(2,3)", "Poisson(t)", "[1,4]", "Gamma"),
        ("reals", "Normal(1,2)", "Normal(t,3)", "[1.0,2.0]", "Normal"),
        (
            "posreals",
            "InverseGamma(2,3)",
            "Normal(1,sqrt(t))",
            "[2.0,4.0]",
            "InverseGamma",
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([2,3,5])",
            "Categorical(t)",
            "[1,3]",
            "Dirichlet",
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([2,3,5])",
            "Categorical0(t)",
            "[0,2]",
            "Dirichlet",
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([2,3,5])",
            "Multinomial(3,t)",
            "[[1,1,1],[2,1,0]]",
            "Dirichlet",
        ),
    ] {
        let body = format!(
            "t = elementof({domain})\nK = functionof(iid({likelihood},2), theta=t)\n\
                            L = likelihoodof(K,{obs})\nprior = joint(theta={prior})\n\
                            posterior = bayesupdate(L,prior)\nP = normalize(posterior)"
        );
        check_sample(&model(&body, "P"), ctor);
        check_sample(&model(&body, "iid(P,3)"), ctor);
    }
}

#[test]
fn joined_factors_update_the_declared_prior_after_a_score() {
    let body = "v ~ Beta(2,3)\nprior = lawof(record(p=v))\nprior_score=logdensityof(prior,record(p=0.5))\n\
                p=elementof(unitinterval)\nx ~ Bernoulli(p)\nK=kernelof(record(x=x),p=p)\n\
                A=likelihoodof(K,record(x=true))\nB=likelihoodof(functionof(Binomial(4,p),p=p),2)\n\
                P=normalize(bayesupdate(joint_likelihood(A,B),prior))";
    let src = model(body, "P").replace(
        "outputs = (result, next)",
        "outputs = (prior_score, result, next)",
    );
    check_sample(&src, "Beta");
}

#[test]
fn every_factor_must_match_and_rand_requires_a_normalized_measure() {
    let body = "t=elementof(reals)\nprior=joint(t=Normal(0,1))\n\
                A=likelihoodof(functionof(Normal(t,1),t=t),2.0)\n\
                B=likelihoodof(functionof(Normal(t,exp(t)),t=t),1.0)\n\
                P=bayesupdate(joint_likelihood(A,B),prior)";
    let mut m = inferred(&model(body, "normalize(P)"));
    let roots = [m.intern("outputs")];
    assert!(
        determinize_with_options(&m, &ModuleBundle::new(), Some(&roots), &LOWERING_OPTIONS)
            .is_err()
    );
    let mut raw = flatppl_syntax::parse(&model(body, "P")).unwrap();
    assert!(
        flatppl_infer::infer(&mut raw)
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("rand"))
    );
}
