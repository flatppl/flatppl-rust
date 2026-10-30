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

const COMBINATORS: &str = r#"
a = Exponential(rate = 1.0)
b = Beta(2.0, 3.0)
named = joint(x = a, y = b)
positional = joint(a, b)
records = joint(named, joint(z = Dirichlet(alpha = [1.0, 1.0])))
sum = superpose(a, LogNormal(mu = 0.0, sigma = 1.0))
labelled = relabel(iid(a, 2), ["p", "q"])
point = Dirac(value = 2.0)
"#;

#[test]
fn products_and_sums_combine_their_component_supports() {
    let mut module = infer(COMBINATORS);
    assert_eq!(
        support(&mut module, "named"),
        "cartprod(x = nonnegreals, y = unitinterval)"
    );
    assert_eq!(
        support(&mut module, "positional"),
        "cartprod(nonnegreals, unitinterval)"
    );
    assert_eq!(
        support(&mut module, "records"),
        "cartprod(x = nonnegreals, y = unitinterval, z = stdsimplex(2))"
    );
    // nonnegreals ∪ posreals.
    assert_eq!(support(&mut module, "sum"), "nonnegreals");
}

#[test]
fn relabel_and_dirac_keep_their_argument_sets() {
    let mut module = infer(COMBINATORS);
    assert_eq!(support(&mut module, "labelled"), "cartpow(nonnegreals, 2)");
    assert_eq!(support(&mut module, "point"), "interval(2, 2)");
}

const CHAINS: &str = r#"
K(t) = LogNormal(mu = t, sigma = 1.0)
chain = jointchain(Normal(mu = 0.0, sigma = 1.0), K)
R(r) = Exponential(rate = exp(r.a))
named = jointchain(a = Normal(mu = 0.0, sigma = 1.0), b = R)
step(s) = Gamma(shape = s + 1.0, rate = 1.0)
trajectory = markovchain(step, 1.0, 4)
mixture = ksuperpose(Exponential, [0.3, 0.7], rate = [1.0, 2.0])
"#;

#[test]
fn jointchain_composes_its_kernels_output_supports() {
    let mut module = infer(CHAINS);
    assert_eq!(support(&mut module, "chain"), "cartprod(reals, posreals)");
    assert_eq!(
        support(&mut module, "named"),
        "cartprod(a = reals, b = nonnegreals)"
    );
}

#[test]
fn markovchain_repeats_its_kernels_output_support() {
    let mut module = infer(CHAINS);
    assert_eq!(
        support(&mut module, "trajectory"),
        "cartpow(nonnegreals, 4)"
    );
}

#[test]
fn ksuperpose_keeps_its_kernels_output_support() {
    let mut module = infer(CHAINS);
    assert_eq!(support(&mut module, "mixture"), "nonnegreals");
}

const TRUNCATIONS: &str = r#"
below = truncate(Exponential(rate = 1.0), interval(-1.0, 2.0))
above = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf))
vector = truncate(iid(Exponential(rate = 1.0), 2), cartpow(interval(-1.0, 3.0), 2))
"#;

#[test]
fn truncate_intersects_interval_supports() {
    let mut module = infer(TRUNCATIONS);
    assert_eq!(support(&mut module, "below"), "interval(0, 2)");
    assert_eq!(support(&mut module, "above"), "interval(0, inf)");
    assert_eq!(support(&mut module, "vector"), "cartpow(interval(0, 3), 2)");
}
