use flatppl_determinizer::{determinize, is_flatpdl};

fn lower(source: &str) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(&format!(
        "reg = standard_module(\"regression\", \"0.1\")\n{source}"
    ))
    .unwrap();
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
fn dependent_flat_bounds_survive_prior_and_likelihood_reification() {
    lower(
        "bound ~ Exponential(1)\n\
         x ~ iid(reg.Flat(interval(0, bound)), 2)\n\
         y ~ Normal(sum(x), 1)\n\
         prior = lawof(record(bound = bound, x = x))\n\
         observation = kernelof(y, bound = bound, x = x)\n\
         posterior = bayesupdate(likelihoodof(observation, 2), prior)\n\
         b = elementof(posreals)\n\
         point = elementof(cartpow(reals, 2))\n\
         lp = logdensityof(posterior, record(bound = b, x = point))",
    );
}

#[test]
fn flat_record_and_constrained_vector_supports_lower_together() {
    lower(
        "support = cartprod(a = unitinterval, b = ordered(3), c = zerosum(3))\n\
         x ~ reg.Flat(support = support)\n\
         point = elementof(cartprod(a = reals, b = cartpow(reals, 3), c = cartpow(reals, 3)))\n\
         lp = logdensityof(lawof(x), point)",
    );
}

#[test]
fn constrained_flat_priors_use_registered_change_of_variable_rules() {
    lower(
        "z = elementof(reals)\n\
         positive = logdensityof(pushfwd(log, reg.Flat(posreals)), z)\n\
         bounded = logdensityof(pushfwd(logit, reg.Flat(unitinterval)), z)",
    );
}
