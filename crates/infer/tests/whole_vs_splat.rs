//! Spec §04 "Calling conventions": a callable that takes one positional input
//! receives a sole record whole, one that does not splats it. The choice is made
//! from the input interface, before the argument's domain is checked. The parameter
//! point of a likelihood is the exception: spec §06 `likelihoodof` binds it, a record,
//! to every input by name, one input included.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(src).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn assert_clean(src: &str) {
    let found = errors(src);
    assert!(found.is_empty(), "unexpected errors: {found:?}\n{src}");
}

fn assert_refused(src: &str, needles: &[&str]) {
    let found = errors(src);
    assert_eq!(found.len(), 1, "expected one error, got {found:?}\n{src}");
    for needle in needles {
        assert!(found[0].contains(needle), "`{needle}` not in: {}", found[0]);
    }
}

const ONE_BOUNDARY: &str = "\
theta = elementof(reals)
K = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
";

const KEYWORD_ONLY: &str = "\
a = elementof(reals)
b = elementof(reals)
K = functionof(Normal(mu = a, sigma = b))
";

#[test]
fn a_record_point_is_bound_by_name_to_a_one_input_likelihood() {
    // Spec §06 `likelihoodof`: the point is a record with one field per input, bound
    // by name for any number of inputs, so a kernel with the single input `theta` is
    // evaluated at `record(theta = 0.3)`, not at `0.3`.
    assert_clean(&format!(
        "{ONE_BOUNDARY}L = likelihoodof(K, 0.5)\nlp = logdensityof(L, record(theta = 0.3))\n"
    ));
    assert_refused(
        &format!("{ONE_BOUNDARY}L = likelihoodof(K, 0.5)\nlp = logdensityof(L, 0.3)\n"),
        &[
            "logdensityof",
            "the parameter point of a likelihood is a record",
            "`theta`",
            "record(theta = ..)",
        ],
    );
    assert_refused(
        &format!(
            "{ONE_BOUNDARY}L = likelihoodof(K, 0.5)\nlp = logdensityof(L, record(phi = 0.3))\n"
        ),
        &["no input named `phi`", "no field for the input `theta`"],
    );
}

#[test]
fn a_record_point_is_bound_whole_to_a_one_boundary_call() {
    assert_refused(
        &format!("{ONE_BOUNDARY}m = K(record(theta = 0.3))\n"),
        &["`K`", "bound whole", "`theta`", "expects real"],
    );
    assert_clean(&format!("{ONE_BOUNDARY}m = K(0.3)\nn = K(theta = 0.3)\n"));
}

#[test]
fn a_keyword_only_callable_splats_a_record() {
    assert_clean(&format!(
        "{KEYWORD_ONLY}d = K(record(a = 0.3, b = 1.0))\n\
         L = likelihoodof(K, 0.5)\nlp = logdensityof(L, record(a = 0.3, b = 1.0))\n"
    ));
}

#[test]
fn a_likelihood_point_names_only_inputs() {
    assert_refused(
        &format!(
            "{KEYWORD_ONLY}L = likelihoodof(K, 0.5)\n\
             lp = logdensityof(L, record(a = 0.3, b = 1.0, c = 2.0))\n"
        ),
        &[
            "bound to the likelihood inputs",
            "`a`, `b`",
            "no input named `c`",
        ],
    );
    assert_refused(
        &format!("{KEYWORD_ONLY}L = likelihoodof(K, 0.5)\nlp = logdensityof(L, record(a = 0.3))\n"),
        &["no field for the input `b`"],
    );
}

#[test]
fn a_one_step_kernel_receives_the_chain_record_whole() {
    assert_refused(
        "sigma = draw(Exponential(rate = 1.0))\n\
         s = elementof(reals)\n\
         reference = Normal(mu = 0.0, sigma = s)\n\
         jc = jointchain(lawof(record(sigma = sigma)), functionof(reference, sigma = s))\n",
        &[
            "jointchain",
            "step 1",
            "cannot unify",
            "record(sigma: real)",
            "expects real",
        ],
    );
}

#[test]
fn chain_steps_that_bind_correctly_are_admitted() {
    // One left variate is fed as it is, to a one-boundary step.
    assert_clean(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         k = kernelof(draw(Normal(mu = a, sigma = 0.5)), a = a)\n\
         j = jointchain(lawof(a), k)\n",
    );
    // A record splats over a keyword-only step's inputs.
    assert_clean(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = elementof(reals)\n\
         k = functionof(Normal(mu = b, sigma = 1.0))\n\
         j = jointchain(lawof(record(b = a)), k)\n",
    );
}

#[test]
fn a_chain_record_splat_names_exactly_the_step_inputs() {
    assert_refused(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = elementof(reals)\n\
         k = functionof(Normal(mu = b, sigma = 1.0))\n\
         j = jointchain(lawof(record(c = a)), k)\n",
        &[
            "jointchain",
            "step 1",
            "no input for field `c`",
            "lacks field `b`",
        ],
    );
}

#[test]
fn a_record_prior_variate_is_bound_by_name_to_a_one_input_likelihood() {
    assert_clean(&format!(
        "{ONE_BOUNDARY}L = likelihoodof(K, 0.5)\n\
         prior = joint(theta = Normal(mu = 0.0, sigma = 1.0))\n\
         post = bayesupdate(L, prior)\n"
    ));
    assert_refused(
        &format!(
            "{ONE_BOUNDARY}L = likelihoodof(K, 0.5)\n\
             prior = Normal(mu = 0.0, sigma = 1.0)\n\
             post = bayesupdate(L, prior)\n"
        ),
        &[
            "bayesupdate",
            "prior's variate",
            "the parameter point of a likelihood is a record",
        ],
    );
}

#[test]
fn a_prior_variate_names_only_inputs() {
    assert_refused(
        &format!(
            "{KEYWORD_ONLY}L = likelihoodof(K, 0.5)\n\
             prior = joint(a = Normal(mu = 0.0, sigma = 1.0), c = Normal(mu = 0.0, sigma = 1.0))\n\
             post = bayesupdate(L, prior)\n"
        ),
        &[
            "bayesupdate",
            "prior's variate",
            "bound to the likelihood inputs",
            "no input named `c`",
        ],
    );
}

#[test]
fn the_keyword_form_chain_applies_the_same_rule() {
    // The previous component's record variate is bound whole to a one-input step.
    assert_refused(
        "sigma = draw(Exponential(rate = 1.0))\n\
         s = elementof(reals)\n\
         reference = Normal(mu = 0.0, sigma = s)\n\
         jc = jointchain(prior = lawof(record(sigma = sigma)), fwd = functionof(reference, sigma = s))\n",
        &[
            "jointchain",
            "step 1",
            "cannot unify",
            "record(sigma: real)",
            "expects real",
        ],
    );
    // A scalar variate feeds a one-input step as it is.
    assert_clean(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         k = kernelof(draw(Normal(mu = a, sigma = 0.5)), a = a)\n\
         j = jointchain(first = lawof(a), second = k)\n",
    );
    // A record splats over a keyword-only step's inputs, and every field names one.
    assert_refused(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = elementof(reals)\n\
         k = functionof(Normal(mu = b, sigma = 1.0))\n\
         j = jointchain(first = lawof(record(c = a)), second = k)\n",
        &["jointchain", "step 1", "no input for field `c`"],
    );
}

const JOINT_LIKELIHOOD: &str = "\
mu = elementof(reals)
model1 = functionof(Normal(mu = mu, sigma = 1.0))
model2 = functionof(Normal(mu = 2.0 * mu, sigma = 0.5))
L = joint_likelihood(likelihoodof(model1, 1.5), likelihoodof(model2, 3.2))
";

#[test]
fn a_joint_likelihood_binds_its_point_by_name_over_the_union_of_inputs() {
    // §06: `joint_likelihood` is `likelihoodof(joint(K1, K2), obs)`, and a `joint`
    // of kernels takes the union of the component inputs by name.
    assert_clean(&format!(
        "{JOINT_LIKELIHOOD}lp = logdensityof(L, record(mu = 0.3))\n"
    ));
    assert_refused(
        &format!("{JOINT_LIKELIHOOD}lp = logdensityof(L, record(mu = 0.3, c = 1.0))\n"),
        &[
            "logdensityof",
            "bound to the likelihood inputs",
            "`mu`",
            "no input named `c`",
        ],
    );
}

#[test]
fn a_joint_likelihood_of_one_input_kernels_takes_the_one_named_point() {
    // Every term takes the one input `mu`, so the point is `record(mu = ..)` and a
    // field that names no input is an error.
    let model = "\
mu = elementof(reals)
a = draw(Normal(mu = mu, sigma = 1.0))
b = draw(Normal(mu = mu, sigma = 2.0))
K1 = kernelof(record(a = a), mu = mu)
K2 = kernelof(record(b = b), mu = mu)
L = joint_likelihood(likelihoodof(K1, record(a = 0.1)), likelihoodof(K2, record(b = 0.2)))
";
    assert_clean(&format!("{model}lp = logdensityof(L, record(mu = 0.3))\n"));
    assert_refused(
        &format!("{model}lp = logdensityof(L, record(mu = 0.3, nu = 1.0))\n"),
        &["no input named `nu`"],
    );
}
