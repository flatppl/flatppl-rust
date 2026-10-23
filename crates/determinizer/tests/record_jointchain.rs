use flatppl_determinizer::{determinize, determinize_with_roots, is_flatpdl};
use flatppl_infer::{ModuleBundle, Severity};

fn lower(source: &str) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    let output = module.intern("lp");
    let lowered = determinize_with_roots(&module, &ModuleBundle::new(), Some(&[output]))
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
    is_flatpdl(&lowered).unwrap();
    // Generated density-lambda names contain allocation IDs, which differ
    // between the chain and its direct factorization. Preserve alpha identity.
    let printed = flatppl_syntax::print(&lowered);
    let mut names = Vec::new();
    printed
        .split_inclusive(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .map(|token| {
            let word = token.trim_end_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
            if !word.starts_with("density_point_") {
                return token.to_owned();
            }
            let index = match names.iter().position(|name| *name == word) {
                Some(index) => index,
                None => {
                    names.push(word);
                    names.len() - 1
                }
            };
            format!("density_arg_{index}{}", &token[word.len()..])
        })
        .collect()
}

/// At sigma=2 and b=[0.5,1.5], the density is log(2)-4. Restricted
/// Lebesgue contributes support gates, not the -2*log(sigma) of a Uniform law.
#[test]
fn proper_scale_then_reference_coefficients_matches_factorization() {
    let point = "p = elementof(cartprod(b = cartpow(reals, 2), sigma = posreals))\n";
    let chain = lower(&format!(
        "sigma ~ Exponential(2)\n\
         k(r) = iid(Lebesgue(interval(0, r.sigma)), 2)\n\
         prior = jointchain(sigma = lawof(sigma), b = k)\n\
         {point}lp = logdensityof(prior, p)"
    ));
    let direct = lower(&format!(
        "{point}lp = logdensityof(Exponential(2), p.sigma) \
         + logdensityof(iid(Lebesgue(interval(0, p.sigma)), 2), p.b)"
    ));
    assert_eq!(chain, direct);
}

/// A unit reference density for sigma leaves the ordinary Normal score. At
/// sigma=2, b=1 this is -log(2*pi)/2-log(2)-1/8, with no scale prior factor.
#[test]
fn reference_scale_then_proper_law_accepts_both_reifiers() {
    let point = "p = elementof(cartprod(b = reals, sigma = reals))\n";
    let direct = lower(&format!(
        "{point}lp = logdensityof(Lebesgue(posreals), p.sigma) \
         + logdensityof(Normal(0, p.sigma), p.b)"
    ));
    // The reified step has the keyword-only input `sigma`, so the fed record splats.
    for reified in ["kernelof(b)", "functionof(lawof(b))"] {
        let chain = lower(&format!(
            "sigma = elementof(posreals)\n\
             b ~ Normal(0, sigma)\n\
             prior = jointchain(sigma = Lebesgue(posreals), b = {reified})\n\
             {point}lp = logdensityof(prior, p)"
        ));
        assert_eq!(chain, direct, "{reified}");
    }
}

/// Each component can contain several fields. A shared parent within the
/// conditional group contributes once, and the scale in the base is not rescored.
#[test]
fn grouped_dependent_fields_score_each_factor_once() {
    let point = "p = elementof(cartprod(c = reals, sigma = posreals, b = reals, mu = reals))\n";
    let chain = lower(&format!(
        "mu ~ Normal(0, 2)\n\
         sigma ~ Exponential(2)\n\
         b ~ Normal(mu, sigma)\n\
         c ~ Normal(b, sigma)\n\
         prior = jointchain(lawof((mu = mu, sigma = sigma)), kernelof((b = b, c = c), mu, sigma))\n\
         {point}lp = logdensityof(prior, p)"
    ));
    let direct = lower(&format!(
        "{point}lp = (logdensityof(Normal(0, 2), p.mu) + logdensityof(Exponential(2), p.sigma)) \
         + (logdensityof(Normal(p.mu, p.sigma), p.b) + logdensityof(Normal(p.b, p.sigma), p.c))"
    ));
    assert_eq!(chain, direct);
}

#[test]
fn composed_record_sources_keep_nested_runtime_point_types() {
    let point =
        "p = elementof(cartprod(r = cartprod(a = reals, b = cartpow(reals, 2)), sigma = reals))\n";
    let chain = lower(&format!(
        "sigma = elementof(posreals)\n\
         r ~ joint(a = Normal(0, sigma), b = iid(Exponential(1), 2))\n\
         prior = jointchain(sigma = Lebesgue(posreals), r = kernelof(r))\n\
         {point}lp = logdensityof(prior, p)"
    ));
    let direct = lower(&format!(
        "{point}lp = logdensityof(Lebesgue(posreals), p.sigma) \
         + logdensityof(joint(a = Normal(0, p.sigma), b = iid(Exponential(1), 2)), p.r)"
    ));
    assert_eq!(chain, direct);
}

#[test]
fn typed_point_vector_length_mismatch_still_refuses() {
    let mut module = flatppl_syntax::parse(
        "sigma = elementof(posreals)\n\
         b ~ iid(Normal(0, sigma), 2)\n\
         prior = jointchain(sigma = Lebesgue(posreals), b = kernelof(b))\n\
         p = elementof(cartprod(sigma = posreals, b = cartpow(reals, 3)))\n\
         lp = logdensityof(prior, p)",
    )
    .unwrap();
    flatppl_infer::infer(&mut module);
    let error = determinize(&module).expect_err("a 2-vector density cannot score a 3-vector");
    assert!(error.reason.contains("shape"), "{error:?}");
}

#[test]
fn applied_boundaries_do_not_capture_other_point_fields() {
    let inputs = "a = elementof(reals)\nb = elementof(reals)\n";
    let chain = lower(&format!(
        "{inputs}k = functionof(joint(c = Normal(2*a + b, 1)), a, b)\n\
         prior = jointchain(joint(a = Lebesgue(reals), b = Lebesgue(reals)), k)\n\
         lp = logdensityof(prior, (a = b, b = a, c = 0.4))"
    ));
    let direct = lower(&format!(
        "{inputs}lp = (logdensityof(Lebesgue(reals), b) + logdensityof(Lebesgue(reals), a)) \
         + logdensityof(Normal(2*b + a, 1), 0.4)"
    ));
    assert_eq!(chain, direct);
}

#[test]
fn a_value_function_is_not_a_conditional_law() {
    let mut module = flatppl_syntax::parse(
        "sigma = elementof(posreals)\n\
         b ~ Normal(0, sigma)\n\
         prior = jointchain(joint(sigma = Lebesgue(posreals)), functionof((b = b), sigma))\n\
         lp = logdensityof(prior, (sigma = 2, b = 1))",
    )
    .unwrap();
    flatppl_infer::infer(&mut module);
    let error = determinize(&module).expect_err("functionof of a value does not reify its law");
    assert_eq!(error.construct, "jointchain");
}

#[test]
fn repeated_scores_keep_their_own_conditioning_points() {
    let chain = lower(
        "sigma ~ Exponential(2)\n\
         tau ~ Exponential(3)\n\
         b ~ Normal(tau, sigma)\n\
         prior = jointchain(lawof(record(sigma = sigma, tau = tau)), \
                            kernelof(record(b = b), sigma = sigma, tau = tau))\n\
         first = logdensityof(prior, (sigma = 2, tau = 1, b = 1))\n\
         second = logdensityof(prior, (sigma = 3, tau = 5, b = 4))\n\
         lp = first + second",
    );
    let direct = lower(
        "first = (logdensityof(Exponential(2), 2) + logdensityof(Exponential(3), 1)) \
                + logdensityof(Normal(1, 2), 1)\n\
         second = (logdensityof(Exponential(2), 3) + logdensityof(Exponential(3), 5)) \
                + logdensityof(Normal(5, 3), 4)\n\
         lp = first + second",
    );
    assert_eq!(chain, direct);
}

#[test]
fn duplicate_component_fields_and_missing_point_fields_refuse() {
    for field in ["sigma", "b"] {
        let mut module = flatppl_syntax::parse(&format!(
            "sigma = elementof(posreals)\n\
             k = functionof(joint({field} = Normal(0, sigma)), sigma)\n\
             prior = jointchain(joint(sigma = Lebesgue(posreals)), k)\n\
             lp = logdensityof(prior, (sigma = 2))"
        ))
        .unwrap();
        flatppl_infer::infer(&mut module);
        let error = determinize(&module).expect_err("the record partition must be unambiguous");
        assert_eq!(error.construct, "jointchain", "{field}");
    }
}
