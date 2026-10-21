//! Applied kernel fanout forms an independent product, with each component's
//! boundary substitution scoped to that component.
use flatppl_determinizer::{determinize, determinize_with_roots};
use flatppl_infer::ModuleBundle;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "fixture must infer: {diags:?}");
    m
}

fn lower(src: &str) -> String {
    let mut m = parse_infer(src);
    let lp = m.intern("lp");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[lp])).expect("must lower");
    flatppl_syntax::print(&out)
}

fn lp_line(out: &str) -> String {
    let at = out
        .find("lp = ")
        .unwrap_or_else(|| panic!("no `lp` binding in:\n{out}"));
    out[at..].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Compare the emitted score with an independently derived density expression.
fn assert_density(src: &str, oracle: &str) {
    assert_eq!(lp_line(&lower(src)), lp_line(&lower(oracle)));
}

const PROBE: &str = "\
z  = elementof(reals)
u  ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z)
";

#[test]
fn an_applied_keyword_fan_out_is_the_product_of_marginals() {
    // Each component integrates u separately: variance 1 + 1, covariance zero.
    assert_density(
        &format!(
            "{PROBE}KJ = joint(p = K1, q = K2)\n\
             lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))"
        ),
        "lp = logdensityof(Normal(0.0, sqrt(2.0)), 1.0) + \
              logdensityof(Normal(0.0, sqrt(2.0)), -1.0)",
    );
}

#[test]
fn an_applied_positional_fan_out_is_the_same_product() {
    assert_density(
        &format!(
            "{PROBE}KJ = joint(K1, K2)\n\
             lp = logdensityof(KJ(z = 0.0), [1.0, -1.0])"
        ),
        "lp = logdensityof(Normal(0.0, sqrt(2.0)), 1.0) + \
              logdensityof(Normal(0.0, sqrt(2.0)), -1.0)",
    );
}

#[test]
fn an_applied_fan_out_moves_with_its_input() {
    assert_density(
        &format!(
            "{PROBE}KJ = joint(p = K1, q = K2)\n\
             lp = logdensityof(KJ(z = 5.0), record(p = 1.0, q = -1.0))"
        ),
        "lp = logdensityof(Normal(5.0, sqrt(2.0)), 1.0) + \
              logdensityof(Normal(5.0, sqrt(2.0)), -1.0)",
    );
}

#[test]
fn a_trace_disjoint_applied_fan_out_is_the_per_component_product() {
    assert_density(
        "\
z = elementof(reals)
w = elementof(reals)
b1 ~ Normal(z, 1.0)
b2 ~ Normal(w, 1.0)
K1 = kernelof(b1, z = z)
K2 = kernelof(b2, w = w)
KJ = joint(p = K1, q = K2)
lp = logdensityof(KJ(z = 0.0, w = 2.0), record(p = 1.0, q = -1.0))",
        "lp = logdensityof(Normal(0.0, 1.0), 1.0) + \
              logdensityof(Normal(2.0, 1.0), -1.0)",
    );
}

#[test]
fn a_repeated_applied_kernel_is_an_independent_product() {
    assert_density(
        &format!(
            "{PROBE}KJ = joint(p = K1, q = K1)\n\
             lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))"
        ),
        "lp = logdensityof(Normal(0.0, sqrt(2.0)), 1.0) + \
              logdensityof(Normal(0.0, sqrt(2.0)), -1.0)",
    );
}

#[test]
fn a_measure_component_of_an_applied_fan_out_is_the_nullary_case() {
    assert_density(
        "\
z = elementof(reals)
b1 ~ Normal(z, 1.0)
K1 = kernelof(b1, z = z)
KJ = joint(p = K1, q = Normal(0.0, 2.0))
lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))",
        "lp = logdensityof(Normal(0.0, 1.0), 1.0) + \
              logdensityof(Normal(0.0, 2.0), -1.0)",
    );
}

#[test]
fn a_nullary_kernel_remains_a_measure_in_an_applied_fan_out() {
    assert_density(
        "\
z = elementof(reals)
x ~ Normal(z, 1.0)
K = kernelof(x, z = z)
C = functionof(Normal(2.0, 1.0))
KJ = joint(a = K, b = C)
lp = logdensityof(KJ(z = 0.0), record(a = 0.3, b = -0.4))",
        "lp = logdensityof(Normal(0.0, 1.0), 0.3) + \
              logdensityof(Normal(2.0, 1.0), -0.4)",
    );
}

#[test]
fn an_unapplied_kernel_joint_density_query_stays_refused() {
    let src = format!(
        "{PROBE}KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ, record(p = 1.0, q = -1.0))"
    );
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        !diags.is_empty() || determinize(&m).is_err(),
        "an unapplied fanout is not a closed measure"
    );
}

#[test]
fn an_applied_fan_out_refuses_an_unsupported_ambient_query_point() {
    // The generic applied-reification path still refuses a point naming its
    // boundary. Retain that guard until point substitution is separately supported.
    let src = format!(
        "{PROBE}KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ(z = 0.0), record(p = z, q = -1.0))"
    );
    let err = determinize(&parse_infer(&src)).expect_err("same-name point scope is unsupported");
    assert!(err.reason.contains("boundary input"), "{}", err.reason);
}

#[test]
fn an_applied_fan_out_preserves_a_raw_sibling_parameter() {
    assert_density(
        "\
mu = elementof(reals)
x ~ Normal(mu, 1.0)
K = kernelof(x, input = mu)
KJ = joint(a = K, b = Normal(mu, 1.0))
score = logdensityof(KJ(input = 3.0), record(a = 3.5, b = mu - 0.25))
F = functionof(score, mu = mu)
lp = F(7.0)",
        "lp = logdensityof(Normal(3.0, 1.0), 3.5) + \
              logdensityof(Normal(7.0, 1.0), 6.75)",
    );
}

#[test]
fn an_applied_fan_out_splats_a_sole_positional_table() {
    assert_density(
        "\
z = elementof(reals)
b1 ~ Normal(z, 1.0)
b2 ~ Normal(z, 1.0)
K1 = kernelof(b1, z = z)
K2 = kernelof(b2, z = z)
KJ = joint(p = K1, q = K2)
lp = logdensityof(KJ(table(z = 0.0)), record(p = 1.0, q = -1.0))",
        "lp = logdensityof(Normal(0.0, 1.0), 1.0) + \
              logdensityof(Normal(0.0, 1.0), -1.0)",
    );
}

#[test]
fn an_applied_fan_out_keeps_the_ambient_cross_named_sibling() {
    assert_density(
        "\
z = elementof(reals)
w = elementof(reals)
b1 ~ Normal(z + w, 1.0)
b2 ~ Normal(z + w, 1.0)
K1 = kernelof(b1, z = z, w = w)
K2 = kernelof(b2, z = z, w = w)
KJ = joint(p = K1, q = K2)
lp = logdensityof(KJ(z = w, w = 0.5), record(p = 1.0, q = -1.0))",
        "w = elementof(reals)\n\
         lp = logdensityof(Normal(w + 0.5, 1.0), 1.0) + \
              logdensityof(Normal(w + 0.5, 1.0), -1.0)",
    );
}
