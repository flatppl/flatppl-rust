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
    // Pin the term structure: 10 obs-likelihood terms (iid(Normal, 10)) + 2 prior
    // terms (theta1 Normal, theta2 Gamma) = 12 — the SAME density bi1
    // (explicit joint) and bi2 (lawof prior) produce for this model. A dropped
    // prior term, a kernel/marginal swap, or a wrong distribution would change this
    // count.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms; got:\n{pir}"
    );
    assert!(
        !pir.contains("disintegrate"),
        "the disintegrate scaffold must be eliminated; got:\n{pir}"
    );
}

/// A self-contained bi4-shape posterior built via `restrict` (spec §06 "Measure
/// restriction"): `restrict(joint, record(obs = data))` is the non-normalized
/// conditional of the joint given the observed `obs`. It desugars into
/// `bayesupdate(likelihoodof(kernel, record(obs = data)), marginal)` over the
/// disintegration on `x`'s field names — the SAME (kernel, marginal), and so the
/// SAME posterior density, as the bi3 explicit `disintegrate` case above.
const BI4_RESTRICT_POSTERIOR: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
post = restrict(joint_model, record(obs = observed_data))
lp = logdensityof(post, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn restrict_lowers_same_as_bi3_disintegrate() {
    // `restrict(M, x)` desugars into `bayesupdate(likelihoodof(kernel, x),
    // marginal)` over `disintegrate([field-names of x], M)`, so the bi4 posterior
    // lowers to the SAME deterministic density as the bi3 explicit-disintegrate
    // case — a FlatPDL module carrying `builtin_logdensityof`, with no residual
    // `restrict` node.
    let pir = flatppl_flatpir::write(
        &determinize(&parse_infer(BI4_RESTRICT_POSTERIOR))
            .expect("bi4 restrict posterior must lower via the restrict desugaring"),
    );
    // The SAME 12 terms as the bi3 disintegrate case: 10 obs-likelihood terms
    // (iid(Normal, 10)) + 2 prior terms (theta1 Normal, theta2 Exponential). This
    // pins `restrict ≡ bayesupdate(likelihoodof(kernel, x), marginal)` for the
    // model — a dropped term, a kernel/marginal swap, or a wrong desugaring would
    // change the count.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms; got:\n{pir}"
    );
    assert!(
        !pir.contains("restrict"),
        "the restrict node must be desugared away; got:\n{pir}"
    );
}

/// The SAME bi4 shape as `BI4_RESTRICT_POSTERIOR` above, but with `x` given as
/// the spec's idiomatic keyword-splat (spec §06 "Measure restriction":
/// `restrict(M, a = …, b = …)` is auto-splat-equivalent to `restrict(M, record(a
/// = …, b = …))`) instead of an explicit `record(...)` argument.
const BI4_RESTRICT_KEYWORD_SPLAT: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
post = restrict(joint_model, obs = observed_data)
lp = logdensityof(post, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn restrict_keyword_splat_lowers() {
    // `restrict(joint_model, obs = observed_data)` — the keyword-splat form, no
    // explicit `record(...)` — must desugar and lower identically to the
    // explicit-record form above: the SAME 12 `builtin_logdensityof` terms (10
    // obs-likelihood + 2 prior), proving the two forms are equivalent.
    let pir = flatppl_flatpir::write(
        &determinize(&parse_infer(BI4_RESTRICT_KEYWORD_SPLAT))
            .expect("bi4 restrict keyword-splat posterior must lower via the restrict desugaring"),
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms (same as the explicit-record form); got:\n{pir}"
    );
    assert!(
        !pir.contains("restrict"),
        "the restrict node must be desugared away; got:\n{pir}"
    );
}

#[test]
fn refuses_restrict_with_field_not_in_variate() {
    // `restrict(M, record(nonexistent = …))` names a field that is not a variate
    // of `M`; the disintegration selector would name a non-field, so the
    // structural split returns `None` and the driver refuses rather than
    // mislowering (refuse-don't-mislower).
    let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
obs ~ Normal(mu = theta1, sigma = 1)
joint_model = lawof(record(theta1 = theta1, obs = obs))
observed_data = 2.5
post = restrict(joint_model, record(nonexistent = observed_data))
lp = logdensityof(post, record(theta1 = 0.5))";
    let err = determinize(&parse_infer(src))
        .expect_err("restrict naming a non-variate field must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("restrict"),
        "refuse must name the restrict; got: {msg}"
    );
}

/// The CAUSALLY-REVERSED disintegrate: selecting the UPSTREAM roots
/// (`["theta1", "theta2"]`) leaves the DOWNSTREAM `obs` as the non-selected
/// marginal. `obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)` depends on
/// theta1/theta2, so the marginal `lawof(record(obs = …))` is NOT closed (it
/// references the external theta1/theta2 draws not in the marginal), and
/// `jointchain(marginal, kernel) ≢ joint` (§06 "Structural disintegration").
/// The structural split must REFUSE (fail-closed) rather than emit the
/// vacuous-boundary kernel + non-closed marginal that would silently score a
/// WRONG density. The reverse-direction disintegrate (§06 "two formulations") is
/// out of scope. Consumed like bi3 (`likelihoodof` / `bayesupdate` /
/// `logdensityof`) so the driver reaches the split; before the closed-marginal
/// guard this lowered to 12 `builtin_logdensityof` terms (silently wrong).
const REVERSED_DISINTEGRATE: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
forward_kernel, prior = disintegrate([\"theta1\", \"theta2\"], joint_model)
L = likelihoodof(forward_kernel, record(theta1 = 0.5, theta2 = 1.0))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(obs = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]))";

#[test]
fn disintegrate_reversed_selector_refuses() {
    // Selecting the upstream roots {theta1, theta2} leaves a non-closed marginal
    // over the downstream {obs}: the split is measure-theoretically invalid and
    // must refuse (refuse-don't-mislower). Before the closed-marginal guard this
    // returned Ok with a silently wrong split (12 terms).
    let err = determinize(&parse_infer(REVERSED_DISINTEGRATE))
        .expect_err("a causally-reversed disintegrate (non-closed marginal) must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("disintegrate") || msg.contains("get"),
        "refuse must name the reversed disintegrate; got: {msg}"
    );
}

/// The `restrict` mirror of the reversed direction: conditioning on the UPSTREAM
/// params (`record(theta1 = …, theta2 = …)`) instead of the downstream `obs`.
/// This is exactly `bayesian_inference_4.flatppl`'s `pars_predictive =
/// restrict(joint_model, default_pars)` landmine (default_pars = the theta1/theta2
/// params, here bound by name to mirror the fixture). The disintegration on
/// {theta1, theta2} leaves a non-closed marginal over {obs} → `rewrite_restrict`
/// must refuse. Before the guard this lowered silently.
const REVERSED_RESTRICT_PARS_PREDICTIVE: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
default_pars = record(theta1 = 0.5, theta2 = 1.0)
pars_predictive = restrict(joint_model, default_pars)
lp = logdensityof(pars_predictive, record(obs = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]))";

#[test]
fn restrict_on_upstream_params_refuses() {
    // `restrict(joint, record(theta1 = …, theta2 = …))` conditions on the upstream
    // params — the reversed direction. The marginal over the downstream {obs} is
    // not closed (references the external theta1/theta2), so the restrict
    // desugaring must refuse rather than mislower. This pins the
    // `bayesian_inference_4.flatppl` `pars_predictive` query as fail-closed.
    let err = determinize(&parse_infer(REVERSED_RESTRICT_PARS_PREDICTIVE)).expect_err(
        "restrict conditioning on the upstream params (non-closed marginal) must refuse",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("restrict"),
        "refuse must name the restrict; got: {msg}"
    );
}

/// The reversed selector defeating the guard via FIELD ALIASING: the record
/// labels the upstream roots with names (`mu_param`, `sigma_param`) that differ
/// from their bindings (`theta1`, `theta2`), and the selector names those
/// LABELS. A guard that intersected the non-selected closure (binding names)
/// against the selector's surface labels would see an empty intersection and
/// wrongly LOWER — a reopened silent-wrong-density. The fix compares resolved
/// binding names on both sides: selected {mu_param→theta1, sigma_param→theta2}
/// ⇒ {theta1, theta2}; non-selected `obs` closure {obs, theta1, theta2};
/// intersection {theta1, theta2} ≠ ∅ → REFUSE.
const REVERSED_DISINTEGRATE_ALIASED: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(mu_param = theta1, sigma_param = theta2, obs = obs))
forward_kernel, prior = disintegrate([\"mu_param\", \"sigma_param\"], joint_model)
L = likelihoodof(forward_kernel, record(mu_param = 0.5, sigma_param = 1.0))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(obs = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]))";

#[test]
fn disintegrate_reversed_selector_with_aliased_fields_refuses() {
    // Field aliasing (`mu_param = theta1`) must NOT defeat the closed-marginal
    // guard: selecting the upstream roots under aliased labels still leaves a
    // non-closed marginal over the downstream {obs} and must refuse. Before the
    // namespace fix (comparing surface labels, not resolved binding names) this
    // returned Ok with a silently wrong split.
    let err = determinize(&parse_infer(REVERSED_DISINTEGRATE_ALIASED)).expect_err(
        "a causally-reversed disintegrate under field aliasing (non-closed marginal) must refuse",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("disintegrate") || msg.contains("get"),
        "refuse must name the reversed disintegrate; got: {msg}"
    );
}

/// The `restrict` mirror of the aliased reversed direction: conditioning on the
/// upstream params under aliased labels (`record(mu_param = …, sigma_param = …)`
/// where those fields resolve to `theta1`/`theta2`). The disintegration on
/// {mu_param, sigma_param} leaves a non-closed marginal over {obs} →
/// `rewrite_restrict` (via the shared `split_law_record`) must refuse.
const REVERSED_RESTRICT_ALIASED: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(mu_param = theta1, sigma_param = theta2, obs = obs))
post = restrict(joint_model, record(mu_param = 0.5, sigma_param = 1.0))
lp = logdensityof(post, record(obs = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]))";

#[test]
fn restrict_upstream_params_aliased_refuses() {
    // `restrict` conditioning on the upstream params under aliased field labels
    // is the reversed direction; the marginal over the downstream {obs} is not
    // closed, so the restrict desugaring must refuse rather than mislower.
    let err = determinize(&parse_infer(REVERSED_RESTRICT_ALIASED)).expect_err(
        "restrict on the upstream params under field aliasing (non-closed marginal) must refuse",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("restrict"),
        "refuse must name the restrict; got: {msg}"
    );
}

/// The VALID direction WITH field aliasing: the downstream `obs` is labeled
/// `data` in the record, and the selector names that label. This is a genuine
/// (closed-marginal) disintegration — selected field `data` RESOLVES to binding
/// `obs`; non-selected {theta1, theta2} closure {theta1, theta2}; intersection
/// ∅ → LOWER. It guards against the namespace fix OVER-refusing aliased fields:
/// the fix must compare resolved binding names, not merely "refuse whenever a
/// field label differs from its binding". (The split-level proof for the shape
/// where the MARGINAL fields are also aliased — `record(mu_param = theta1,
/// sigma_param = theta2, data = obs)` — lives in `src/disintegrate.rs`'s unit
/// tests as `splits_valid_aliased_marginal_fields`: that shape splits correctly
/// but its posterior θ point cannot lower through the likelihood path, which
/// still requires θ field names to name module bindings — a separate,
/// out-of-scope likelihood-path limitation, not a disintegration defect.)
const VALID_DISINTEGRATE_ALIASED: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, data = obs))
forward_kernel, prior = disintegrate([\"data\"], joint_model)
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(data = observed_data))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn disintegrate_valid_aliased_fields_lowers() {
    // The valid direction must still lower when the SELECTED field is aliased
    // (`data = obs`): 10 obs-likelihood terms + 2 prior terms = 12, the SAME
    // density as the non-aliased bi3 case. The fix resolves the selected field
    // `data` to its binding `obs` (not the label `data`), so the closed-marginal
    // guard sees selected {obs} ∩ non-selected {theta1, theta2} = ∅ and does not
    // over-refuse. Proves the namespace fix is binding-name-correct, not
    // "refuse-more".
    let pir = flatppl_flatpir::write(
        &determinize(&parse_infer(VALID_DISINTEGRATE_ALIASED))
            .expect("the valid direction with aliased fields must still lower"),
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms; got:\n{pir}"
    );
    assert!(
        !pir.contains("disintegrate"),
        "the disintegrate scaffold must be eliminated; got:\n{pir}"
    );
}

/// A TRANSITIVE reversed dependency: the downstream `obs` depends on an
/// intermediate latent `a`, and `a` depends on the selected root `theta1`. The
/// closed-marginal walk must follow the closure transitively (obs → a → theta1)
/// and catch that the non-selected marginal reaches a selected binding →
/// REFUSE. Confirms the generative-closure walk is transitive, not one-level.
const REVERSED_DISINTEGRATE_TRANSITIVE: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
a ~ Normal(mu = theta1, sigma = 1)
obs ~ iid(Normal(mu = a, sigma = theta2), 10)
joint_model = lawof(record(mu_param = theta1, sigma_param = theta2, obs = obs))
forward_kernel, prior = disintegrate([\"mu_param\", \"sigma_param\"], joint_model)
L = likelihoodof(forward_kernel, record(mu_param = 0.5, sigma_param = 1.0))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(obs = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]))";

#[test]
fn disintegrate_reversed_transitive_dep_refuses() {
    // obs depends on `a`, `a` depends on the selected `theta1`: the non-selected
    // closure {obs, a, theta1, theta2} reaches the selected {theta1, theta2}
    // transitively → refuse. Guards that the walk is transitive over deep deps.
    let err = determinize(&parse_infer(REVERSED_DISINTEGRATE_TRANSITIVE)).expect_err(
        "a reversed disintegrate whose non-selected closure reaches a selected root transitively must refuse",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("disintegrate") || msg.contains("get"),
        "refuse must name the reversed disintegrate; got: {msg}"
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
