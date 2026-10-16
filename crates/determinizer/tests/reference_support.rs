use flatppl_determinizer::{determinize, is_flatpdl};

fn lower(source: &str) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let module = determinize(&module).unwrap();
    is_flatpdl(&module).unwrap();
    module
}

#[test]
fn dependent_reference_bounds_survive_prior_and_likelihood_reification() {
    lower(
        "bound ~ Exponential(1)\n\
         x = elementof(cartpow(reals, 2))\n\
         y ~ Normal(sum(x), 1)\n\
         prior = jointchain(lawof(record(bound = bound)), functionof(joint(x = iid(Lebesgue(interval(0, bound)), 2)), bound = bound))\n\
         observation = kernelof(y, bound = bound, x = x)\n\
         posterior = bayesupdate(likelihoodof(observation, 2), prior)\n\
         point = elementof(cartprod(bound = posreals, x = cartpow(reals, 2)))\n\
         lp = logdensityof(posterior, point)",
    );
}

#[test]
fn reference_record_and_constrained_vector_supports_lower_together() {
    lower(
        "support = cartprod(a = unitinterval, b = ordered(3), c = zerosum(3))\n\
         prior = Lebesgue(support = support)\n\
         point = elementof(cartprod(a = reals, b = cartpow(reals, 3), c = cartpow(reals, 3)))\n\
         lp = logdensityof(prior, point)",
    );
}

#[test]
fn constrained_reference_priors_use_registered_change_of_variable_rules() {
    lower(
        "z = elementof(reals)\n\
         positive = logdensityof(pushfwd(log, Lebesgue(posreals)), z)\n\
         bounded = logdensityof(pushfwd(logit, Lebesgue(unitinterval)), z)",
    );
}
