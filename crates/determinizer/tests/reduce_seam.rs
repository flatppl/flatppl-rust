//! The public measure-level reduction seam: `desugar_restrict_and_disintegrate`
//! (spec §06 "Structural disintegration", "Measure restriction") and
//! `match_conjugate_marginal` (spec §06 "Density of composed measures", the
//! `kchain` row; rows in `src/marginal.md`). Both leave the module at the
//! measure level.

use flatppl_core::{CallHead, Module, Node, NodeId};
use flatppl_determinizer::{
    ConjugateRowId, desugar_restrict_and_disintegrate, match_conjugate_marginal,
};

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn rhs(m: &Module, name: &str) -> NodeId {
    let (_, b) = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap_or_else(|| panic!("no binding {name}"));
    b.rhs
}

fn head(m: &Module, id: NodeId) -> Option<&str> {
    match m.node(id) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(h) => Some(m.resolve(h)),
            _ => None,
        },
        _ => None,
    }
}

/// The constructor under `name = draw(D)`.
fn drawn(m: &Module, name: &str) -> NodeId {
    let id = rhs(m, name);
    assert_eq!(head(m, id), Some("draw"));
    let Node::Call(c) = m.node(id) else {
        unreachable!()
    };
    c.args[0]
}

fn heads(m: &Module) -> Vec<String> {
    let mut out = Vec::new();
    for (_, b) in m.bindings() {
        let mut stack = vec![b.rhs];
        while let Some(id) = stack.pop() {
            if let Some(h) = head(m, id) {
                out.push(h.to_string());
            }
            m.for_each_child(id, |c| stack.push(c));
        }
    }
    out
}

fn row(src: &str) -> Option<ConjugateRowId> {
    let mut m = parse_infer(src);
    let latent = m.intern("z");
    let prior = drawn(&m, "z");
    let lik = drawn(&m, "y");
    match_conjugate_marginal(&mut m, prior, lik, latent).map(|r| r.unwrap().row)
}

#[test]
fn matches_every_conjugate_row() {
    let cases = [
        (
            "z = draw(Normal(mu = 0.0, sigma = 1.0))\ny = draw(Normal(mu = z, sigma = 2.0))",
            ConjugateRowId::NormalMean,
        ),
        (
            "z = draw(Gamma(shape = 2.0, rate = 3.0))\ny = draw(Poisson(rate = z))",
            ConjugateRowId::GammaPoissonRate,
        ),
        (
            "z = draw(Beta(alpha = 2.0, beta = 3.0))\ny = draw(Bernoulli(p = z))",
            ConjugateRowId::BetaBernoulli,
        ),
        (
            "z = draw(Beta(alpha = 2.0, beta = 3.0))\ny = draw(Binomial(n = 7, p = z))",
            ConjugateRowId::BetaBinomial,
        ),
        (
            "z = draw(Exponential(rate = 1.3))\ny = draw(Normal(mu = 0.3, sigma = sqrt(z)))",
            ConjugateRowId::ExponentialVariance,
        ),
        (
            "z = draw(InverseGamma(shape = 2.5, scale = 1.7))\ny = draw(Normal(mu = 0.3, sigma = sqrt(z)))",
            ConjugateRowId::InverseGammaVariance,
        ),
        (
            "z = draw(Exponential(rate = 1.3))\ny = draw(Normal(mu = z, sigma = 0.8))",
            ConjugateRowId::ExponentialMean,
        ),
    ];
    for (src, want) in cases {
        assert_eq!(row(src), Some(want), "{src}");
    }
}

#[test]
fn match_keeps_the_original_parameter_nodes() {
    let mut m =
        parse_infer("z = draw(Gamma(shape = 2.0, rate = 3.0))\ny = draw(Poisson(rate = z))");
    let latent = m.intern("z");
    let prior = drawn(&m, "z");
    let lik = drawn(&m, "y");
    let before = m.node_count();
    let matched = match_conjugate_marginal(&mut m, prior, lik, latent)
        .unwrap()
        .unwrap();
    assert_eq!(
        m.node_count(),
        before,
        "keyword constructors allocate nothing"
    );
    assert_eq!(matched.latent, latent);
    let names: Vec<_> = matched.prior.iter().map(|(s, _)| m.resolve(*s)).collect();
    assert_eq!(names, ["shape", "rate"]);
    let Node::Call(c) = m.node(prior) else {
        unreachable!()
    };
    assert_eq!(matched.prior[0].1, c.named[0].value);
}

#[test]
fn rejects_shapes_outside_the_table() {
    // The latent feeds `sigma` directly, not through `sqrt`.
    assert_eq!(
        row("z = draw(Exponential(rate = 1.3))\ny = draw(Normal(mu = 0.3, sigma = z))"),
        None
    );
    // Both likelihood parameters reference the latent.
    assert_eq!(
        row("z = draw(Normal(mu = 0.0, sigma = 1.0))\ny = draw(Normal(mu = z, sigma = exp(z)))"),
        None
    );
    // No row pairs these families.
    assert_eq!(
        row("z = draw(Gamma(shape = 2.0, rate = 3.0))\ny = draw(Normal(mu = z, sigma = 1.0))"),
        None
    );
}

#[test]
fn non_scalar_parameter_refuses() {
    let mut m = parse_infer(
        "z = draw(Normal(mu = 0.0, sigma = 1.0))\ny = draw(Normal(mu = z, sigma = [1.0, 2.0]))",
    );
    let latent = m.intern("z");
    let prior = drawn(&m, "z");
    let lik = drawn(&m, "y");
    assert!(matches!(
        match_conjugate_marginal(&mut m, prior, lik, latent),
        Some(Err(_))
    ));
}

const JOINT: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Gamma(shape = 2, rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 3)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
observed_data = [1.2, 3.4, 5.1]
";

#[test]
fn desugars_restrict_to_a_measure_level_posterior() {
    let mut m = parse_infer(&format!(
        "{JOINT}posterior = restrict(joint_model, record(obs = observed_data))\n"
    ));
    assert!(desugar_restrict_and_disintegrate(&mut m).unwrap());
    let post = rhs(&m, "posterior");
    assert_eq!(head(&m, post), Some("bayesupdate"));
    let hs = heads(&m);
    assert!(hs.iter().any(|h| h == "likelihoodof"));
    assert!(hs.iter().any(|h| h == "draw"), "draws stay: {hs:?}");
    assert!(
        !hs.iter()
            .any(|h| h == "restrict" || h.starts_with("builtin_"))
    );
    assert!(m.type_of(post).is_some(), "the result is re-inferred");
    // The unreferenced joint survives: no binding is swept.
    assert_eq!(head(&m, rhs(&m, "joint_model")), Some("lawof"));
}

#[test]
fn desugars_disintegrate_projections() {
    let mut m = parse_infer(&format!(
        "{JOINT}forward_kernel, prior = disintegrate([\"obs\"], joint_model)\n\
         L = likelihoodof(forward_kernel, record(obs = observed_data))\n\
         posterior = bayesupdate(L, prior)\n"
    ));
    assert!(desugar_restrict_and_disintegrate(&mut m).unwrap());
    let hs = heads(&m);
    assert!(!hs.iter().any(|h| h == "get"), "{hs:?}");
    assert_eq!(head(&m, rhs(&m, "forward_kernel")), Some("kernelof"));
    assert_eq!(head(&m, rhs(&m, "prior")), Some("lawof"));
    // The unreferenced target survives: no binding is swept.
    assert_eq!(head(&m, rhs(&m, "posterior")), Some("bayesupdate"));
}

#[test]
fn desugar_is_a_no_op_without_restrict_or_disintegrate() {
    let mut m = parse_infer(JOINT);
    assert!(!desugar_restrict_and_disintegrate(&mut m).unwrap());
}

#[test]
fn desugar_refuses_a_restrict_outside_the_structural_case() {
    let mut m = parse_infer(&format!(
        "{JOINT}posterior = restrict(joint_model, record(missing = observed_data))\n"
    ));
    let err = desugar_restrict_and_disintegrate(&mut m).unwrap_err();
    assert_eq!(err.construct, "restrict");
}
