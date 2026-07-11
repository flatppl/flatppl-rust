//! Structural disintegration of a joint law (spec §06 "Structural
//! disintegration"), the bi3 shape: `disintegrate(["obs"], lawof(record(…)))`
//! splits into a forward kernel + prior, then `likelihoodof` / `bayesupdate` /
//! `logdensityof` recover the posterior density.
//!
//! Task 3 delivers the structural split (`split_disintegrate`, unit-tested in
//! isolation inside `src/disintegrate.rs`, which can reach the `pub(crate)`
//! function). This integration file pins the END-TO-END driver behavior:
//! - `refuses_until_get_disintegrate_is_eliminated` is GREEN now — it
//!   characterizes the pre-wiring state (the driver has no rule for
//!   `get(disintegrate, i)`, so it refuses at that `get`).
//! - `lowers_bi3_posterior_to_builtin_logdensityof` is the Task-4 target
//!   (`#[ignore]`d until the driver eliminates `get(disintegrate, i)` via
//!   `split_disintegrate`); un-ignore it and delete the characterization test
//!   when Task 4 lands.

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
fn refuses_until_get_disintegrate_is_eliminated() {
    // Pre-Task-4 characterization: the driver has no rule to eliminate
    // `get(disintegrate, i)` (the `forward_kernel = get(__0xN, 1)` /
    // `prior = get(__0xN, 2)` bindings), so it refuses at that `get` node
    // rather than mislowering. Task 4 wires `split_disintegrate` into the
    // driver to eliminate the `get`s; this assertion then flips.
    let err = determinize(&parse_infer(BI3_POSTERIOR))
        .expect_err("disintegrate/get elimination is not wired until Task 4");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("get") || msg.contains("primitive measure") || msg.contains("disintegrate"),
        "refuse reason must name the unhandled `get(disintegrate, …)`; got: {msg}"
    );
}

#[test]
#[ignore = "Task 4 wires split_disintegrate into the driver to eliminate get(disintegrate, i)"]
fn lowers_bi3_posterior_to_builtin_logdensityof() {
    // Task-4 target: with `get(disintegrate, i)` eliminated via the structural
    // split, the posterior density lowers to the prior over {theta1, theta2}
    // plus the obs-likelihood — a FlatPDL module carrying `builtin_logdensityof`.
    let pir = flatppl_flatpir::write(
        &determinize(&parse_infer(BI3_POSTERIOR)).expect("bi3 posterior must lower once wired"),
    );
    assert!(pir.contains("builtin_logdensityof"), "got:\n{pir}");
}
