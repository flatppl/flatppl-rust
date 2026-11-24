//! `locscale(iid(D, J), loc, s)` with a vector `loc` and a scalar `s` over a
//! location-scale family D (§06 `locscale`, §07 `mul`): per cell the law
//! D(loc + s·a, |s|·b). A latent draw with a known-sign scale is declared
//! `<offset, multiplier>`, the non-centred parameterization of the same
//! parameter and density. The emitted text does not depend on the egglog rules.

use flatppl_stan::{EmitOptions, RulePolicy};

fn emit(source: &str, rules: &RulePolicy) -> String {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let _ = flatppl_infer::infer(&mut module);
    flatppl_stan::emit_with_rules(&module, &EmitOptions::default(), rules)
        .unwrap_or_else(|r| panic!("emit: {r:?}"))
        .code
}

/// The program under the default rules, checked equal with every rule disabled.
fn program(source: &str) -> String {
    let on = emit(source, &RulePolicy::default());
    assert_eq!(
        on,
        emit(source, &RulePolicy::all_disabled()),
        "rules on and off"
    );
    on
}

fn model(prior: &str, fields: &str, observed: &str) -> String {
    format!(
        "J = 3\nys = [1.5, -0.5, 2.0]\nmu ~ Normal(0, 5)\n{prior}\n\
         Pi_prior = lawof(record({fields}))\n\
         K_model = kernelof(record(y = y), {fields})\n\
         L = likelihoodof(K_model, record(y = ys))\nPi_post = bayesupdate(L, Pi_prior)\n{observed}"
    )
}

#[test]
fn a_latent_normal_draw_is_non_centred_after_its_location_and_scale() {
    let code = program(&model(
        "tau ~ Exponential(1)\ntheta ~ locscale(iid(Normal(0, 1), J), fill(mu, J), tau)\n\
         y ~ Normal.(theta, 1.0)",
        "theta = theta, mu = mu, tau = tau",
        "",
    ));
    let decl = "vector<offset=rep_vector(mu, J), multiplier=tau>[3] theta;";
    let (at_mu, at_tau, at_theta) = (
        code.find("real mu;").expect("mu"),
        code.find("real<lower=0> tau;").expect("tau"),
        code.find(decl).unwrap_or_else(|| panic!("{code}")),
    );
    assert!(at_mu < at_theta && at_tau < at_theta, "{code}");
    assert!(
        code.contains("theta ~ normal(rep_vector(mu, J), tau);"),
        "{code}"
    );
}

#[test]
fn a_latent_student_t_draw_takes_its_locscale_row() {
    let code = program(&model(
        "tau ~ Exponential(1)\ntheta ~ locscale(iid(StudentT(4), J), fill(mu, J), tau)\n\
         y ~ Normal.(theta, 1.0)",
        "theta = theta, mu = mu, tau = tau",
        "",
    ));
    assert!(
        code.contains("vector<offset=rep_vector(mu, J), multiplier=tau>[3] theta;"),
        "{code}"
    );
    assert!(
        code.contains("theta ~ student_t(4, rep_vector(mu, J), tau);"),
        "{code}"
    );
}

#[test]
fn an_observed_draw_is_the_vectorized_family_law() {
    // Cauchy(1, 2) under loc + s·x is Cauchy(loc + s, 2·s).
    let code = program(&model(
        "tau ~ Exponential(1)\ny ~ locscale(iid(Cauchy(1.0, 2.0), J), fill(mu, J), tau)",
        "mu = mu, tau = tau",
        "",
    ));
    assert!(
        code.contains("ys ~ cauchy(rep_vector(mu, J) + tau, tau * 2);"),
        "{code}"
    );
}

#[test]
fn a_scale_of_unknown_sign_stays_centred_with_its_absolute_spread() {
    let code = program(&model(
        "s ~ Normal(0, 1)\ntheta ~ locscale(iid(Normal(0, 1), J), fill(mu, J), s)\n\
         y ~ Normal.(theta, 1.0)",
        "theta = theta, mu = mu, s = s",
        "",
    ));
    assert!(code.contains("vector[3] theta;"), "{code}");
    assert!(
        code.contains("theta ~ normal(rep_vector(mu, J), abs(s));"),
        "{code}"
    );
}

#[test]
fn a_scalar_locscale_of_unknown_sign_takes_the_absolute_spread() {
    // Was a refusal: Cauchy(0, 1) under 0 + m·x is Cauchy(0, |m|).
    let code = program(
        "m ~ Normal(0, 1)\nsigma ~ locscale(Cauchy(0, 1), 0, m)\ny ~ Normal(sigma, 1)\n\
         prior = lawof(record(m = m, sigma = sigma))\n\
         L = likelihoodof(kernelof(record(y = y), m = m, sigma = sigma), record(y = 0.5))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert!(code.contains("sigma ~ cauchy(0, abs(m));"), "{code}");
}
