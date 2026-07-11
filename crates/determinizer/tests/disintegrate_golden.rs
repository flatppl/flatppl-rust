//! Structural disintegration of a joint law (spec §06 "Structural
//! disintegration"), the bi3 shape: `disintegrate(["obs"], lawof(record(…)))`
//! splits into a forward kernel + prior, then `likelihoodof` / `bayesupdate` /
//! `logdensityof` recover the posterior density.
//!
//! Task 3 delivers the structural split (`split_disintegrate`, unit-tested in
//! isolation inside `src/disintegrate.rs`, which can reach the `pub(crate)`
//! function). Task 4 wires it into the driver: `get(disintegrate(…), 1)` /
//! `get(disintegrate(…), 2)` are eliminated into the split kernel/marginal, so
//! the downstream `likelihoodof` / `bayesupdate` / `logdensityof` lower via the
//! existing paths. This integration file pins the END-TO-END driver behavior:
//! - `lowers_bi3_posterior_to_builtin_logdensityof` — the bi3 posterior lowers to
//!   `builtin_logdensityof` (the prior over {theta1, theta2} + the obs-likelihood).
//! - `refuses_disintegrate_over_non_lawof_record` /
//!   `refuses_get_disintegrate_out_of_range` — refuse-don't-mislower guards.

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// A self-contained bi3-shape posterior built via structural disintegration.
const BI3_POSTERIOR: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Gamma(alpha = 2, beta = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
forward_kernel, prior = disintegrate([\"obs\"], joint_model)
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn lowers_bi3_posterior_to_builtin_logdensityof() {
    // With `get(disintegrate, i)` eliminated via the structural split, the
    // posterior density lowers to the prior over {theta1, theta2} plus the
    // obs-likelihood — a FlatPDL module carrying `builtin_logdensityof`, and with
    // no residual `disintegrate` / `get`-on-a-disintegrate scaffold.
    let pir = flatppl_flatpir::write(
        &determinize(&parse_infer(BI3_POSTERIOR)).expect("bi3 posterior must lower once wired"),
    );
    assert!(pir.contains("builtin_logdensityof"), "got:\n{pir}");
    assert!(
        !pir.contains("disintegrate"),
        "the disintegrate scaffold must be eliminated; got:\n{pir}"
    );
}

#[test]
fn refuses_disintegrate_over_non_lawof_record() {
    // `split_disintegrate` only handles the explicit `lawof(record(…))` DAG case;
    // a `disintegrate` over a bare measure (here a plain `Normal`) yields `None`,
    // so the driver refuses the `get(disintegrate, i)` rather than mislowering.
    let src = "\
d = Normal(mu = 0, sigma = 1)
fk, pr = disintegrate([\"obs\"], d)";
    let err = determinize(&parse_infer(src))
        .expect_err("disintegrate over a non-lawof(record) measure must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("disintegrate") || msg.contains("get"),
        "refuse must name the non-explicit disintegrate; got: {msg}"
    );
}

#[test]
fn refuses_get_disintegrate_out_of_range() {
    // A `get(disintegrate(…), i)` with `i` outside the 1-based (kernel, marginal)
    // pair is out of range for the 2-tuple; the driver refuses rather than
    // mislowering. `a, b, c = disintegrate(…)` desugars a third `get(_, 3)`.
    let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
obs ~ Normal(mu = theta1, sigma = 1)
joint_model = lawof(record(theta1 = theta1, obs = obs))
a, b, c = disintegrate([\"obs\"], joint_model)";
    let err = determinize(&parse_infer(src))
        .expect_err("get(disintegrate, 3) is out of range and must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("out of range"),
        "refuse must report the out-of-range get index; got: {msg}"
    );
}
