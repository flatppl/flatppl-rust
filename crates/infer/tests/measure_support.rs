//! Spec §11: "for a measure-valued node [the value-set slot] is the measure's
//! support". Draws, `lawof` over draws and records of draws, and the posterior
//! of a record prior carry the §08 supports of the distributions they draw from.

use flatppl_core::Module;

fn infer(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    module
}

fn support(module: &mut Module, name: &str) -> String {
    let symbol = module.intern(name);
    let rhs = module.binding(module.binding_by_name(symbol).unwrap()).rhs;
    let set = module.valueset_of(rhs).expect("a value set").clone();
    module.display_valueset(&set)
}

const MODEL: &str = r#"
theta ~ Normal(mu = 0.0, sigma = 1.0)
s ~ Exponential(rate = 1.0)
p ~ Beta(2.0, 3.0)
w ~ Dirichlet(alpha = [1.0, 1.0, 1.0])
y ~ Normal(mu = theta, sigma = s)
scale = lawof(s)
weights = lawof(w)
prior = lawof(record(theta = theta, s = s, p = p, w = w))
literal = lawof((theta = theta, s = s))
forward = kernelof(record(y = y), theta = theta, s = s, p = p, w = w)
L = likelihoodof(forward, record(y = 1.0))
posterior = bayesupdate(L, prior)
"#;

#[test]
fn draws_carry_their_distribution_supports() {
    let mut module = infer(MODEL);
    assert_eq!(support(&mut module, "theta"), "reals");
    assert_eq!(support(&mut module, "s"), "nonnegreals");
    assert_eq!(support(&mut module, "p"), "unitinterval");
    assert_eq!(support(&mut module, "w"), "stdsimplex(3)");
}

#[test]
fn lawof_a_draw_is_its_support() {
    let mut module = infer(MODEL);
    assert_eq!(support(&mut module, "scale"), "nonnegreals");
    assert_eq!(support(&mut module, "weights"), "stdsimplex(3)");
}

#[test]
fn a_record_prior_is_the_product_of_its_field_supports() {
    let mut module = infer(MODEL);
    assert_eq!(
        support(&mut module, "prior"),
        "cartprod(theta = reals, s = nonnegreals, p = unitinterval, w = stdsimplex(3))"
    );
    assert_eq!(
        support(&mut module, "literal"),
        "cartprod(theta = reals, s = nonnegreals)"
    );
}

#[test]
fn a_posterior_keeps_its_prior_support() {
    let mut module = infer(MODEL);
    assert_eq!(
        support(&mut module, "posterior"),
        "cartprod(theta = reals, s = nonnegreals, p = unitinterval, w = stdsimplex(3))"
    );
}
