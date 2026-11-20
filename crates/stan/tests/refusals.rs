//! One refusal per never-kind, plus the not-yet and target cases. Each message
//! must name the user's construct.

use flatppl_stan::{EmitOptions, Refusal, RefusalKind};

fn refuse(source: &str) -> Refusal {
    refuse_with(source, None)
}

fn refuse_with(source: &str, target: Option<&str>) -> Refusal {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let opts = EmitOptions {
        target: target.map(str::to_string),
        ..Default::default()
    };
    flatppl_stan::emit(&module, &opts).expect_err("refuses")
}

fn posterior(model: &str, prior: &str, kernel: &str) -> String {
    format!(
        "{model}\nprior = {prior}\nL = likelihoodof({kernel}, record(y = 0.5))\n\
         post = bayesupdate(L, prior)\n"
    )
}

#[test]
fn continuous_latent_is_never() {
    let r = refuse(&posterior(
        "z ~ Normal(0, 1)\nmu ~ Normal(0, 1)\ny ~ Normal(mu + z, 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::LatentMarginalized, "{r}");
    assert!(r.kind.is_never());
    assert!(r.reason.contains("`z`"), "{r}");
}

#[test]
fn discrete_parameter_is_never() {
    let r = refuse(&posterior(
        "k ~ Poisson(3.0)\ny ~ Normal(k, 1)",
        "lawof(record(k = k))",
        "kernelof(record(y = y), k = k)",
    ));
    assert_eq!(r.kind, RefusalKind::DiscreteParameter, "{r}");
    assert!(r.reason.contains("`k`"), "{r}");
}

#[test]
fn continuous_kchain_is_never() {
    let r = refuse_with(
        "rate = elementof(posreals)\n\
         z = draw(Gamma(shape = 2.0, rate = rate))\n\
         k = kernelof(record(y = draw(Poisson(z))), z = z)\n\
         L = likelihoodof(functionof(kchain(lawof(z), k)), record(y = 4))\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::KchainMarginal, "{r}");
}

#[test]
fn multi_dimensional_normalizer_is_never() {
    let r = refuse_with(
        "a = elementof(reals)\n\
         m = normalize(logweighted((p, a) -> logdensityof(Normal(a * p.y, 1.0), p.x),\n    \
             Lebesgue(support = cartprod(x = interval(-5.0, 5.0), y = interval(-5.0, 5.0))), a))\n\
         L = likelihoodof(functionof(m), record(x = 0.1, y = 0.2))\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NumericNormalizer, "{r}");
}

#[test]
fn projection_pushforward_is_never() {
    let r = refuse_with(
        "m = relabel(iid(Normal(mu = 0.0, sigma = 1.0), 3), [\"a\", \"b\", \"c\"])\n\
         L = likelihoodof(functionof(pushfwd(fn(get(_, [\"a\", \"c\"])), m)), record(a = 0.1, c = 0.3))\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NonBijectivePushfwd, "{r}");
}

#[test]
fn law_with_no_stan_density_is_never() {
    let r = refuse(&posterior(
        "hep = standard_module(\"particle-physics\", \"0.1\")\n\
         mu ~ Normal(0, 1)\ny ~ hep.Voigtian(mean = mu, width = 1.0, sigma = 0.5)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NoStanFunction, "{r}");
    assert!(r.reason.contains("Voigtian"), "{r}");
}

#[test]
fn unverified_mapping_is_not_yet() {
    let r = refuse(&posterior(
        "sf = standard_module(\"special-functions\", \"0.1\")\n\
         mu ~ Normal(0, 1)\ny ~ Normal(sf.polygamma(1, exp(mu)), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("polygamma"), "{r}");
}

#[test]
fn ordered_nonidentical_normalize_is_never() {
    let r = refuse(&posterior(
        "mu ~ normalize(truncate(joint(Normal(0, 1), Normal(3, 1)), ordered(2)))\ny ~ Normal(mu[1], 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NumericNormalizer, "{r}");
    assert!(r.reason.contains("ordered"), "{r}");
}

#[test]
fn dirac_with_continuous_component_parameter_is_never() {
    // Over data the hurdle split scores it (Ruling 2); a parameter has no
    // density with respect to Lebesgue measure.
    let r = refuse_with(
        "w ~ Beta(2, 2)\nm = normalize(superpose(weighted(w, Normal(0, 1)), weighted(1 - w, Dirac(0.0))))\n\
         z ~ iid(m, 3)\ny ~ Normal(sum(z), 1)\nprior = lawof(record(w = w, z = z))\n\
         L = likelihoodof(kernelof(record(y = y), w = w, z = z), record(y = 0.5))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::MixedReference, "{r}");
    assert!(r.kind.is_never());
}

#[test]
fn dirac_with_continuous_component_other_weights_is_not_yet() {
    let r = refuse_with(
        "w ~ Beta(2, 2)\nm = normalize(superpose(weighted(w, Normal(0, 1)), weighted(0.5, Dirac(0.0))))\n\
         y ~ iid(m, 3)\nprior = lawof(record(w = w))\n\
         L = likelihoodof(kernelof(record(y = y), w = w), record(y = [0.0, 0.5, 1.0]))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("these weights"), "{r}");
}

#[test]
fn truncated_component_without_cdf_is_not_yet() {
    // `Poisson(exp(a))` is Stan's `poisson_log`, which has no CDF; its mass
    // has a closed form all the same, so the refusal is not a never.
    let r = refuse(&posterior(
        "w ~ Beta(2, 2)\na ~ Normal(0, 1)\n\
         y ~ normalize(superpose(weighted(w, truncate(Poisson(exp(a)), interval(1, 4))), weighted(1 - w, Poisson(2.0))))",
        "lawof(record(w = w, a = a))",
        "kernelof(record(y = y), w = w, a = a)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("no Stan CDF"), "{r}");
}

#[test]
fn two_sided_truncation_beside_dirac_per_element_is_not_yet() {
    let r = refuse_with(
        "x = [0.2, -0.5, 1.0]\nhu ~ Beta(2, 2)\nb ~ Normal(0, 1)\n\
         f = (mu, hu) -> normalize(superpose(weighted(hu, Dirac(0)), weighted(1.0 - hu, normalize(truncate(Poisson(mu), interval(1, 6))))))\n\
         y ~ f.(exp.(b .* x), hu)\nprior = lawof(record(hu = hu, b = b))\n\
         L = likelihoodof(kernelof(record(y = y), hu = hu, b = b), record(y = [0, 2, 1]))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("two-sided truncation"), "{r}");
}

#[test]
fn no_and_ambiguous_target() {
    let r = refuse("mu = elementof(reals)\nL = likelihoodof(functionof(Normal(mu, 1)), 0.5)\n");
    assert_eq!(r.kind, RefusalKind::NoTarget, "{r}");
    let r = refuse(
        "mu ~ Normal(0, 1)\ny ~ Normal(mu, 1)\nprior = lawof(record(mu = mu))\n\
         L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = 0.5))\n\
         a = bayesupdate(L, prior)\nb = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::AmbiguousTarget, "{r}");
    assert!(r.reason.contains("a, b"), "{r}");
}

#[test]
fn point_process_is_not_yet() {
    let r = refuse(
        "rate ~ Gamma(shape = 2.0, rate = 1.0)\n\
         y ~ BinnedPoissonProcess([0.0, 1.0, 2.0], weighted(rate, Lebesgue(support = interval(0.0, 2.0))))\n\
         prior = lawof(record(rate = rate))\n\
         L = likelihoodof(kernelof(record(y = y), rate = rate), record(y = [1, 2]))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("BinnedPoissonProcess"), "{r}");
}

#[test]
fn reduction_of_a_looped_body_is_not_yet() {
    let r = refuse(&posterior(
        "X = rowstack([[1.0, -2.0], [0.5, 3.0]])\n\
         positives = aggregate(sum, [.i], ifelse(X[.i, .j] > 0.0, 1.0, 0.0))\n\
         mu ~ Normal(0, 1)\ny ~ Normal(mu + positives[1], 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("aggregate"), "{r}");
}

#[test]
fn diagonal_axis_is_not_yet() {
    let r = refuse(&posterior(
        "X = rowstack([[1.0, -2.0], [0.5, 3.0]])\n\
         trace = aggregate(sum, [], X[.i, .i])\n\
         mu ~ Normal(0, 1)\ny ~ Normal(mu + trace, 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("repeated axis"), "{r}");
}

#[test]
fn user_broadcast_over_a_table_is_not_yet() {
    let r = refuse(&posterior(
        "rows = table(a = [1.0, 2.0], b = [3.0, 4.0])\n\
         total(row) = row.a + row.b\n\
         mu ~ Normal(0, 1)\ny ~ Normal(mu + sum(total.(rows)), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("table"), "{r}");
}

#[test]
fn joint_of_one_draw_twice_is_never() {
    let r = refuse_with(
        "y = draw(Normal(0.0, 1.0))\nlp = logdensityof(lawof(record(a = y, b = y)), record(a = 0.5, b = 0.5))\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::SingularJoint, "{r}");
    assert!(r.kind.is_never());
    assert!(r.reason.contains("`y`"), "{r}");
}

#[test]
fn observation_at_infinity_is_not_yet() {
    let r = refuse_with(
        "s = elementof(posreals)\nlp = logdensityof(Normal(0.0, s), inf)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("infinity"), "{r}");
}

#[test]
fn logweighted_callback_parameter_is_not_yet() {
    let r = refuse(&posterior(
        "mu = elementof(reals)\ny ~ Normal(mu, 1)\nw(x, s) = -0.5 * (x.mu / s)^2",
        "logweighted(w, Lebesgue(cartprod(mu = reals)), 2.0)",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("callback parameter"), "{r}");
}

#[test]
fn logdensityof_of_none_row_is_never() {
    let r = refuse(&posterior(
        "hep = standard_module(\"particle-physics\", \"0.1\")\n\
         mu = elementof(reals)\ny ~ Normal(mu, 1)\n\
         w(x) = logdensityof(hep.Voigtian(mean = x.mu, width = 1.0, sigma = 0.5), 0.2)",
        "logweighted(w, Lebesgue(cartprod(mu = reals)))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NoStanFunction, "{r}");
    assert!(r.reason.contains("Voigtian"), "{r}");
}

#[test]
fn bare_logdensityof_broadcast_is_not_yet() {
    let r = refuse(&posterior(
        "b = elementof(cartpow(reals, 3))\ny ~ Normal(b[1], 1)\n\
         w(x) = sum(exp.(logdensityof.(Normal(0, 1), x.b)))",
        "logweighted(w, Lebesgue(cartprod(b = cartpow(reals, 3))))",
        "kernelof(record(y = y), b = b)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("logdensityof"), "{r}");
}

#[test]
fn ordered_with_interval_bound_is_not_yet() {
    let r = refuse(&posterior(
        "s = elementof(ordered(2))\ny ~ Normal(s[1], s[2])",
        "truncate(Lebesgue(cartprod(s = ordered(2))), cartprod(s = cartpow(interval(0.0, 5.0), 2)))",
        "kernelof(record(y = y), s = s)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("ordered"), "{r}");
}

#[test]
fn positive_set_with_interval_lower_bound_is_not_yet() {
    let r = refuse(&posterior(
        "s = elementof(reals)\ny ~ Normal(s, 1)",
        "truncate(Lebesgue(cartprod(s = interval(-1.0, 5.0))), cartprod(s = posreals))",
        "kernelof(record(y = y), s = s)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("positive set with an interval lower bound"),
        "{r}"
    );
}

#[test]
fn ordered_restriction_of_data_is_not_yet() {
    let r = refuse_with(
        "mu ~ Normal(0, 1)\ny ~ normalize(truncate(iid(Normal(mu, 1), 2), ordered(2)))\n\
         prior = lawof(record(mu = mu))\n\
         L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = [0.1, 2.5]))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("ordered restriction"), "{r}");
}

#[test]
fn logdensityof_of_ordered_restriction_is_not_yet() {
    let r = refuse(&posterior(
        "b = elementof(cartpow(reals, 2))\ny ~ Normal(b[1], 1)\n\
         w(x) = logdensityof(normalize(truncate(iid(Normal(0, 1), 2), ordered(2))), x.b)",
        "logweighted(w, Lebesgue(cartprod(b = cartpow(reals, 2))))",
        "kernelof(record(y = y), b = b)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("ordered restriction"), "{r}");
}

#[test]
fn scan_over_table_is_not_yet() {
    let r = refuse(
        "xs = [0.1, 0.2, 0.3]\nmu ~ Normal(0, 1)\n\
         step(s, x, p) = record(a = s.a + x * p.mu, b = s.a)\n\
         states = scan(step, record(a = 0.0, b = 0.0), xs, record(mu = mu))\n\
         step2(acc, row) = acc + row.a\ntotal = scan(step2, 0.0, states)\n\
         y ~ Normal.(total, 1)\nprior = lawof(record(mu = mu))\n\
         L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = [0.1, 0.2, 0.3]))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("scan over a table"), "{r}");
}

#[test]
fn record_scan_used_whole_is_not_yet() {
    let r = refuse(
        "xs = [0.1, 0.2, 0.3]\nmu ~ Normal(0, 1)\n\
         step(s, x, p) = record(a = s.a + x * p.mu, b = s.a)\n\
         states = scan(step, record(a = 0.0, b = 0.0), xs, record(mu = mu))\n\
         first(t) = t[1].a\ny ~ Normal(first(states), 1)\nprior = lawof(record(mu = mu))\n\
         L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = 0.5))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a record scan used whole"), "{r}");
}

#[test]
fn user_function_over_aggregate_frame_is_not_yet() {
    let r = refuse(&posterior(
        "x_obs = [1.0, 2.0, 3.0]\nb ~ iid(Normal(0, 1), 2)\ng(x, c) = c * x + 1\n\
         m = aggregate(sum, [.i], g(x_obs[.i], b[.k]))\ny ~ Normal(m[1], 1)",
        "lawof(record(b = b))",
        "kernelof(record(y = y), b = b)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("aggregate reducing over this body"),
        "{r}"
    );
}

#[test]
fn vector_scan_read_whole_is_not_yet() {
    let r = refuse(&posterior(
        "K = 2\nx_obs = [0.5, -1.0, 2.0]\nw ~ iid(Normal(0, 1), K)\n\
         step(acc, x, p) = 0.5 .* acc .+ p.w .* x\n\
         run = scan(step, zeros(K), x_obs, record(w = w))\ny ~ Normal(sum(sum(run)), 1)",
        "lawof(record(w = w))",
        "kernelof(record(y = y), w = w)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a scan of vectors read whole"), "{r}");
}

#[test]
fn boundary_input_inside_nested_reification_is_not_yet() {
    let r = refuse(&posterior(
        "a0 = elementof(reals)\nzd ~ Normal(a0, 1)\ninner = lawof(zd)\n\
         v = logdensityof(inner, 0.3)\nf = functionof(v, acc = a0)\nmu ~ Normal(0, 1)\n\
         y ~ Normal(mu + f(1.0), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("nested `lawof`"), "{r}");
}

#[test]
fn markovchain_prior_is_not_yet() {
    let r = refuse(
        "T = 3\np ~ Uniform(unitinterval)\nstep(previous, q) = Normal(q * previous, 1)\n\
         z ~ iid(markovchain(step, 0.0, T, p), 2)\ny ~ Normal(sum(z[1]), 1)\n\
         prior = lawof(record(p = p, z = z))\n\
         L = likelihoodof(kernelof(record(y = y), p = p, z = z), record(y = 0.5))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("markovchain as a prior"), "{r}");
}

#[test]
fn markovchain_one_trajectory_is_not_yet() {
    let r = refuse(
        "T = 3\np ~ Uniform(unitinterval)\nstep(previous, q) = Bernoulli(q)\n\
         y ~ markovchain(step, 0, T, p)\nprior = lawof(record(p = p))\n\
         L = likelihoodof(kernelof(record(y = y), p = p), record(y = [0, 1, 1]))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("one trajectory"), "{r}");
}

#[test]
fn markovchain_kernel_argument_of_other_form_is_not_yet() {
    let r = refuse(
        "M = 3\nT = 3\ndata = rowstack([[0, 1, 1], [1, 1, 0], [0, 0, 1]])\n\
         history(i) = data[i, :]\nhistories = history.(indicesof(zeros(M)))\n\
         a ~ Normal(0, 1)\nstep(previous, q) = Bernoulli(invlogit(q + previous))\n\
         y ~ iid(markovchain(step, 0, T, a), M)\nprior = lawof(record(a = a))\n\
         L = likelihoodof(kernelof(record(y = y), a = a), record(y = histories))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("markovchain kernel argument"), "{r}");
}

#[test]
fn splitblocks_other_shape_is_not_yet() {
    let r = refuse(
        "M = 4\nT = 2\ny_obs = rowstack([[0, 1], [1, 1], [0, 0], [1, 0]])\n\
         p ~ Uniform(unitinterval)\nblock(b, q) = Bernoulli.(fill(q, [2, T]))\n\
         y ~ block.(splitblocks(y_obs, [2, T]), p)\nprior = lawof(record(p = p))\n\
         L = likelihoodof(kernelof(record(y = y), p = p), record(y = splitblocks(y_obs, [2, T])))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
}

#[test]
fn splitblocks_partial_row_is_not_yet() {
    let r = refuse(
        "M = 2\ny_obs = rowstack([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1]])\n\
         p ~ Uniform(unitinterval)\nblock(b, q) = Bernoulli.(fill(q, [1, 3]))\n\
         y ~ block.(splitblocks(y_obs, [1, 3]), p)\nprior = lawof(record(p = p))\n\
         L = likelihoodof(kernelof(record(y = y), p = p), record(y = splitblocks(y_obs, [1, 3])))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("splitblocks"), "{r}");
}

/// Review case: blocks of one column are not rows. The spec value at
/// omega = 0.5, lp = 0 is 3·log 0.75 + log 0.25 = −2.249340; reading the
/// blocks as rows gave −2.549446.
#[test]
fn splitblocks_single_column_blocks_is_not_yet() {
    let r = refuse(
        "M = 2\nT = 2\ny_obs = rowstack([[0, 1], [0, 0]])\nomega ~ Uniform(unitinterval)\n\
         e ~ iid(Normal(0, 1), M)\nlp = addaxes(e, 0, 1) .+ addaxes(zeros(T), 1, 0)\n\
         capture(b, w) = normalize(superpose(weighted(w, Bernoulli.(invlogit.(b))), \
         weighted(1 - w, Dirac(fill(false, [1, 1])))))\n\
         y ~ capture.(splitblocks(lp, [1, 1]), omega)\nprior = lawof(record(omega = omega, e = e))\n\
         L = likelihoodof(kernelof(record(y = y), omega = omega, e = e), \
         record(y = splitblocks(y_obs, [1, 1])))\npost = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
}

#[test]
fn row_mixture_weight_per_row_is_not_yet() {
    let r = refuse(
        "M = 2\nT = 2\ny_obs = rowstack([[0, 1], [0, 0]])\nw ~ iid(Uniform(unitinterval), M)\n\
         e ~ iid(Normal(0, 1), M)\nlp = addaxes(e, 0, 1) .+ addaxes(zeros(T), 1, 0)\n\
         capture(b, q) = normalize(superpose(weighted(q, Bernoulli.(invlogit.(b))), \
         weighted(1 - q, Dirac(fill(false, [1, T])))))\n\
         y ~ capture.(splitblocks(lp, [1, T]), w)\nprior = lawof(record(w = w, e = e))\n\
         L = likelihoodof(kernelof(record(y = y), w = w, e = e), \
         record(y = splitblocks(y_obs, [1, T])))\npost = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("weight per row"), "{r}");
}

#[test]
fn dirac_atom_parameter_is_not_yet() {
    let r = refuse(
        "M = 2\nT = 2\ny_obs = rowstack([[0, 1], [0, 0]])\nhistory(i) = y_obs[i, :]\n\
         histories = history.(indicesof(zeros(M)))\nomega ~ Uniform(unitinterval)\n\
         p ~ iid(Uniform(unitinterval), T)\ntheta ~ Uniform(unitinterval)\n\
         ch = normalize(superpose(weighted(omega, Bernoulli.(p)), \
         weighted(1 - omega, Dirac(fill(ifelse(theta > 0.5, 1, 0), T)))))\n\
         y ~ iid(ch, M)\nprior = lawof(record(omega = omega, p = p, theta = theta))\n\
         L = likelihoodof(kernelof(record(y = y), omega = omega, p = p, theta = theta), \
         record(y = histories))\npost = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("atom whose value is not data"), "{r}");
}

#[test]
fn row_law_rewritten_vector_is_not_yet() {
    let r = refuse(
        "M = 2\nT = 2\ny_obs = rowstack([[0, 1], [0, 0]])\nhistory(i) = y_obs[i, :]\n\
         histories = history.(indicesof(zeros(M)))\nomega ~ Uniform(unitinterval)\n\
         v ~ iid(Normal(0, 1), T)\n\
         ch = normalize(superpose(weighted(omega, Bernoulli.(invlogit.(v))), weighted(1 - omega, Dirac(fill(0, T)))))\n\
         y ~ iid(ch, M)\nprior = lawof(record(omega = omega, v = v))\n\
         L = likelihoodof(kernelof(record(y = y), omega = omega, v = v), record(y = histories))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("rewrites a vector argument"), "{r}");
}

#[test]
fn filter_of_parameter_is_never() {
    let r = refuse(&posterior(
        "y_all = [0.3, -1.0, 2.0]\nmu ~ Normal(0, 1)\nvals = y_all .+ mu\n\
         keep(x) = x > 0\nkept = filter(keep, vals)\ny ~ Normal(sum(kept), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::ParameterDependentExtent, "{r}");
    assert!(r.kind.is_never());
}

#[test]
fn filter_with_callback_parameter_is_not_yet() {
    let r = refuse(&posterior(
        "y_all = [0.3, -1.0, 2.0]\nkeep(i, p) = y_all[i] > p\n\
         idx = filter(keep, indicesof(y_all), 0.0)\nshift(i) = 0.5 * i\n\
         mu ~ Normal(0, 1)\ny ~ Normal(mu + sum(shift.(idx)), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("callback parameter"), "{r}");
}

#[test]
fn filter_over_a_table_is_not_yet() {
    let r = refuse(&posterior(
        "rows = table(a = [1.0, 2.0], b = [3.0, -4.0])\n\
         keep(row) = row.a + row.b > 0\nkept = filter(keep, rows)\n\
         g(t) = 0.5\nmu ~ Normal(0, 1)\ny ~ Normal(mu + g(kept), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("filter over a table"), "{r}");
}

#[test]
fn broadcast_over_a_boolean_mask_is_no_gather() {
    let r = refuse(&posterior(
        "y_all = [0.3, -1.0, 2.0]\nmask = [true, false, true]\npick(i) = y_all[i]\n\
         ys = pick.(mask)\nmu ~ Normal(0, 1)\ny ~ Normal(mu + sum(ys), 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
}

#[test]
fn pushfwd_exp_is_not_yet() {
    let r = refuse(&posterior(
        "rho ~ pushfwd(fn(exp(_)), Normal(0, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("not affine"), "{r}");
}

#[test]
fn pushfwd_matrix_affine_is_not_yet() {
    let r = refuse(&posterior(
        "A = rowstack([[2.0, 0.0], [1.0, 3.0]])\nx ~ pushfwd(fn(A * _), iid(Normal(0, 1), 2))\n\
         y ~ Normal(sum(x), 1)",
        "lawof(record(x = x))",
        "kernelof(record(y = y), x = x)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("array coefficient"), "{r}");
}

#[test]
fn pushfwd_bounded_law_unknown_sign_is_not_yet() {
    let r = refuse(&posterior(
        "cs = [0.5, -2.0]\nrho ~ pushfwd(fn(cs[2] * _), Beta(2, 2))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("unknown sign"), "{r}");
}

#[test]
fn pushfwd_zero_scale_is_never() {
    let r = refuse(&posterior(
        "rho ~ pushfwd(fn(0 * _ + 1), Normal(0, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NonBijectivePushfwd, "{r}");
}

#[test]
fn pushfwd_array_variate_is_not_yet() {
    let r = refuse(&posterior(
        "x ~ pushfwd(fn(2 * _), iid(Normal(0, 1), 3))\ny ~ Normal(sum(x), 1)",
        "lawof(record(x = x))",
        "kernelof(record(y = y), x = x)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("array variate"), "{r}");
}

#[test]
fn pushfwd_inside_iid_is_not_yet() {
    let r = refuse(&posterior(
        "x ~ iid(pushfwd(fn(2 * _), Normal(0, 1)), 3)\ny ~ Normal(sum(x), 1)",
        "lawof(record(x = x))",
        "kernelof(record(y = y), x = x)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("pushfwd"), "{r}");
}

#[test]
fn pushfwd_of_discrete_law_is_not_yet() {
    let r = refuse(
        "lam ~ Exponential(1)\ny ~ pushfwd(fn(_ + 1), Poisson(lam))\n\
         prior = lawof(record(lam = lam))\n\
         L = likelihoodof(kernelof(record(y = y), lam = lam), record(y = 4))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("pushfwd of this law"), "{r}");
}

#[test]
fn nested_iid_three_levels_is_not_yet() {
    let r = refuse(&posterior(
        "w ~ iid(iid(iid(Normal(0, 1), 2), 2), 2)\ny ~ Normal(sum(w[1][1]), 1)",
        "lawof(record(w = w))",
        "kernelof(record(y = y), w = w)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("nested more than twice"), "{r}");
}

#[test]
fn nested_iid_of_vector_law_is_not_yet() {
    let r = refuse(&posterior(
        "w ~ iid(iid(Dirichlet(ones(3)), 2), 2)\ny ~ Normal(sum(w[1][1]), 1)",
        "lawof(record(w = w))",
        "kernelof(record(y = y), w = w)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("nested iid"), "{r}");
}

/// A zero-inflated row of zero-inflated sites, observed at `counts`.
fn nested_rows(site: &str, count: &str) -> String {
    format!(
        "J = 3\ncounts = [[0, 0, 0], [1, 0, 2]]\nw ~ Beta(2, 2)\nq ~ Beta(1, 3)\n\
         lam ~ Exponential(1)\nsite(q, lam) = {site}\n\
         y ~ iid(normalize(superpose(weighted(w, iid(site(q, lam), {count})), \
         weighted(1 - w, Dirac(fill(0, J))))), 2)\n\
         prior = lawof(record(w = w, q = q, lam = lam))\n\
         L = likelihoodof(kernelof(record(y = y), w = w, q = q, lam = lam), record(y = counts))\n\
         post = bayesupdate(L, prior)\n"
    )
}

#[test]
fn nested_dirac_split_other_atom_is_not_yet() {
    let r = refuse(&nested_rows(
        "normalize(superpose(weighted(q, Poisson(lam)), weighted(1 - q, Dirac(1))))",
        "J",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("differs from the row atom"), "{r}");
}

#[test]
fn nested_dirac_split_other_weights_is_not_yet() {
    let r = refuse(&nested_rows(
        "normalize(superpose(weighted(q, Poisson(lam)), weighted(0.5, Dirac(0))))",
        "J",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("these weights"), "{r}");
}

#[test]
fn nested_dirac_split_continuous_site_is_not_yet() {
    let r = refuse(&nested_rows(
        "normalize(superpose(weighted(q, Normal(lam, 1)), weighted(1 - q, Dirac(0))))",
        "J",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("continuous law"), "{r}");
}

#[test]
fn nested_dirac_split_truncated_site_is_not_yet() {
    let r = refuse(&nested_rows(
        "normalize(superpose(weighted(q, normalize(truncate(Poisson(lam), interval(0, 5)))), \
         weighted(1 - q, Dirac(0))))",
        "J",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("truncated mixture component"), "{r}");
}

#[test]
fn nested_dirac_split_unnormalized_site_is_not_yet() {
    let r = refuse(&nested_rows(
        "normalize(superpose(weighted(q, truncate(Poisson(lam), interval(0, 5))), \
         weighted(1 - q, Dirac(0))))",
        "J",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("truncated mixture component"), "{r}");
}

#[test]
fn nested_dirac_split_shifted_site_is_not_yet() {
    let r = refuse(&nested_rows(
        "normalize(superpose(weighted(q, Categorical0([0.2, 0.3, 0.5])), weighted(1 - q, Dirac(0))))",
        "J",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("iid mixture with this law"), "{r}");
}

#[test]
fn dirac_atom_beside_argument_dependent_support_is_not_yet() {
    let r = refuse(
        "w ~ Beta(2, 2)\ny ~ iid(normalize(superpose(weighted(w, Categorical([0.2, 0.3, 0.5])), \
         weighted(1 - w, Dirac(2)))), 3)\nprior = lawof(record(w = w))\n\
         L = likelihoodof(kernelof(record(y = y), w = w), record(y = [2, 1, 3]))\n\
         post = bayesupdate(L, prior)\n",
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("support depends on its arguments"), "{r}");
}

#[test]
fn locscale_truncated_scale_of_unknown_sign_is_not_yet() {
    let r = refuse(&posterior(
        "cs = [2.0, -1.0]\nsigma ~ locscale(normalize(truncate(StudentT(3), interval(0, inf))), 0, cs[1])\n\
         y ~ Normal(0, sigma)",
        "lawof(record(sigma = sigma))",
        "kernelof(record(y = y), sigma = sigma)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("scale of unknown sign"), "{r}");
}

#[test]
fn locscale_scale_of_unknown_sign_is_not_yet() {
    let r = refuse(&posterior(
        "m ~ Normal(0, 1)\nsigma ~ locscale(Cauchy(0, 1), 0, m)\ny ~ Normal(sigma, 1)",
        "lawof(record(m = m, sigma = sigma))",
        "kernelof(record(y = y), m = m, sigma = sigma)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("scale of unknown sign"), "{r}");
}

#[test]
fn query_point_binding_another_input_is_not_yet() {
    let r = refuse_with(
        "mu = elementof(reals)\nnu = elementof(reals)\n\
         L = likelihoodof(functionof(Normal(mu, 1), mu = mu), 0.4)\n\
         inputs = (mu, nu)\noutputs = logdensityof(L, record(mu = nu))\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("while it is the input"), "{r}");
}

#[test]
fn discrete_truncation_at_a_real_end_is_not_yet() {
    // stanc 2.39 rejects `T[lo, hi]`, `T[lo, ]` and `T[, hi]` of a discrete
    // law with a real end.
    for set in ["interval(0.5, 6)", "interval(0.5, inf)", "interval(0, 5.5)"] {
        let r = refuse_with(
            &format!(
                "lam = elementof(posreals)\nk = 3\n\
                 m = normalize(truncate(Poisson(lam), {set}))\n\
                 inputs = (lam, k)\nlp = logdensityof(m, k)"
            ),
            Some("lp"),
        );
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{set}: {r}");
        assert!(
            r.reason
                .contains("discrete law truncated at a non-integer end"),
            "{set}: {r}"
        );
    }
}

#[test]
fn normalized_continuous_truncation_to_integers_is_never() {
    let r = refuse(&posterior(
        "mu ~ Normal(0, 1)\ny ~ normalize(truncate(Normal(mu, 1), posintegers))",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::ZeroMass, "{r}");
    assert!(r.kind.is_never());
    assert!(r.reason.contains("total mass 0"), "{r}");
}

#[test]
fn continuous_truncation_to_integers_is_not_yet() {
    let r = refuse(&posterior(
        "mu = elementof(reals)\ny ~ Normal(mu, 1)\n\
         w(x) = logdensityof(truncate(Normal(0, 1), nonnegintegers), x.mu)",
        "logweighted(w, Lebesgue(cartprod(mu = reals)))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("zero measure"), "{r}");
}

fn weighted_model(log_lik: &str, call: &str) -> String {
    format!(
        "x = [0.5, -1.0, 2.0]\ny_data = [1.1, -0.4, 2.7]\nw_data = [1.0, 0.5, 2.0]\n\
         a ~ Normal(0, 2)\nsigma ~ Exponential(1)\nmu = a .+ x\n\
         prior = lawof(record(a = a, sigma = sigma))\nlog_lik = {log_lik}\n\
         post = logweighted(point -> sum(w_data .* {call}), prior)\n"
    )
}

#[test]
fn keyword_call_with_other_fields_is_not_yet() {
    let r = refuse_with(
        &weighted_model(
            "functionof(((m, v, s) -> logdensityof(Normal(m, s), v)).(mu, y_data, sigma), a = a, sigma = sigma)",
            "log_lik(a = point.sigma, sigma = point.a)",
        ),
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("matching fields"), "{r}");
}

#[test]
fn weighted_law_reading_the_value_is_not_yet() {
    let r = refuse_with(
        &weighted_model(
            "functionof(((m, v, s) -> logdensityof(Normal(m + v, s), v)).(mu, y_data, sigma), a = a, sigma = sigma)",
            "log_lik(a = point.a, sigma = point.sigma)",
        ),
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("reads the observed value"), "{r}");
}

#[test]
fn normalize_weighted_other_weight_is_not_yet() {
    let r = refuse(&posterior(
        "mu ~ Normal(0, 1)\ny ~ normalize(weighted(2.0, Normal(mu, 1)))",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("weighted measure with this weight"),
        "{r}"
    );
}

#[test]
fn normalize_weighted_unbounded_parameter_is_not_yet() {
    // `s` may be negative: R1 needs a declared `lower=0` or a positive set.
    let r = refuse(&posterior(
        "mu ~ Normal(0, 1)\ns ~ Normal(1, 1)\n\
         y ~ normalize(weighted(ifelse(s > 0.0 && isfinite(s), 1.0, 0.0), Normal(mu, ifelse(s > 0.0 && isfinite(s), s, 1.0))))",
        "lawof(record(mu = mu, s = s))",
        "kernelof(record(y = y), mu = mu, s = s)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("weighted measure with this weight"),
        "{r}"
    );
}

#[test]
fn keyword_call_of_a_valued_input_is_not_yet() {
    // `b = 3.0` has its own value: the body would read it, not `point.b`.
    let r = refuse_with(
        "x = [0.5, -1.0, 2.0]\ny_data = [1.1, -0.4, 2.7]\nw_data = [1.0, 0.5, 2.0]\nb = 3.0\n\
         prior = joint(b = Normal(0, 1))\n\
         log_lik = functionof(((m, v) -> logdensityof(Normal(m, 1), v)).(b .+ x, y_data), b = b)\n\
         post = logweighted(point -> sum(w_data .* log_lik(b = point.b)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("matching fields"), "{r}");
}

#[test]
fn keyword_call_of_a_derived_input_is_not_yet() {
    // `a = 2 * alpha` is not the prior field `a` (which is alpha's draw):
    // the body would read 2·alpha, not `point.a`.
    let r = refuse_with(
        "x = [0.5, -1.0, 2.0]\ny_data = [1.1, -0.4, 2.7]\nw_data = [1.0, 0.5, 2.0]\n\
         alpha ~ Normal(0, 2)\nsigma ~ Exponential(1)\na = 2 * alpha\nmu = a .+ x\n\
         prior = lawof(record(a = alpha, sigma = sigma))\n\
         log_lik = functionof(((m, v, s) -> logdensityof(Normal(m, s), v)).(mu, y_data, sigma), a = a, sigma = sigma)\n\
         post = logweighted(point -> sum(w_data .* log_lik(a = point.a, sigma = point.sigma)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("matching fields"), "{r}");
}

#[test]
fn unnormalized_discrete_component_at_a_real_end_is_not_yet() {
    // `interval(0.0, inf)` would reach `poisson_lccdf(0.0 - 1 | …)`, which
    // stanc rejects.
    let r = refuse_with(
        "k = 2\nm = normalize(superpose(weighted(0.3, truncate(Poisson(3.0), interval(0.0, inf))), \
         weighted(0.7, Poisson(1.5))))\ninputs = (k)\nlp = logdensityof(m, k)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("non-integer end"), "{r}");
}

#[test]
fn normalize_weighted_root_of_unbounded_parameter_is_not_yet() {
    // `sqrt(se^2 + p^2)` is positive only with p > 0: p here may be 0.
    let r = refuse_with(
        "x = [0.5, -1.0]\nse = [0.5, 1.2]\nmu ~ Normal(0, 1)\np ~ Normal(1, 1)\n\
         f = (m, s) -> normalize(weighted(ifelse(s > 0.0 && isfinite(s), 1.0, 0.0), Normal(m, ifelse(s > 0.0 && isfinite(s), s, 1.0))))\n\
         y ~ f.(mu .+ x, sqrt.(se .* se .+ p .* p))\nprior = lawof(record(mu = mu, p = p))\n\
         L = likelihoodof(kernelof(record(y = y), mu = mu, p = p), record(y = [0.4, -1.2]))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("weighted measure with this weight"),
        "{r}"
    );
}

#[test]
fn landau_is_never() {
    let r = refuse(&posterior(
        "hep = standard_module(\"particle-physics\", \"0.1\")\n\
         mu ~ Normal(0, 1)\ny ~ hep.Landau(loc = mu, scale = 1.0)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NoStanFunction, "{r}");
    assert!(r.kind.is_never());
    assert!(r.reason.contains("Landau"), "{r}");
}

#[test]
fn draw_from_continued_poisson_is_rejected() {
    // §09: ContinuedPoisson is not normalized, so there is no draw from it
    // (no `rand`); the module is rejected before any Stan emission.
    let mut module = flatppl_syntax::parse(
        "hep = standard_module(\"particle-physics\", \"0.1\")\nx ~ hep.ContinuedPoisson(rate = 3.0)\n",
    )
    .expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
}

#[test]
fn custom_density_with_array_parameters_is_not_yet() {
    let r = refuse_with(
        "hep = standard_module(\"particle-physics\", \"0.1\")\n\
         m0s = [5.27, 5.28, 5.29]\ns ~ Exponential(rate = 200.0)\n\
         y ~ hep.CrystalBall.(m0s, s, 1.5, 3.0)\nprior = lawof(record(s = s))\n\
         L = likelihoodof(kernelof(record(y = y), s = s), record(y = [5.27, 5.28, 5.29]))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("array parameters"), "{r}");
}

#[test]
fn unnormalized_mixture_component_law_is_not_yet() {
    let r = refuse_with(
        "hep = standard_module(\"particle-physics\", \"0.1\")\n\
         w ~ Beta(2, 2)\ny ~ normalize(superpose(weighted(w, hep.ContinuedPoisson(rate = 3.0)), weighted(1 - w, Normal(3, 1))))\n\
         prior = lawof(record(w = w))\n\
         L = likelihoodof(kernelof(record(y = y), w = w), record(y = 2.0))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("unnormalized law"), "{r}");
}

#[test]
fn continued_poisson_outside_a_plain_statement_is_not_yet() {
    // normalize, truncate and weighted would need its mass, which no row gives.
    for (law, reason) in [
        ("normalize(hep.ContinuedPoisson(rate = r))", "normalize"),
        (
            "truncate(hep.ContinuedPoisson(rate = r), interval(0.0, 5.0))",
            "truncate",
        ),
        ("weighted(2.0, hep.ContinuedPoisson(rate = r))", "weighted"),
    ] {
        let r = refuse_with(
            &format!(
                "hep = standard_module(\"particle-physics\", \"0.1\")\nr = elementof(posreals)\n\
                 L = likelihoodof(functionof({law}), 3.0)\n"
            ),
            Some("L"),
        );
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{law}: {r}");
        assert!(r.reason.contains(reason), "{law}: {r}");
    }
}

#[test]
fn real_typed_data_under_a_discrete_law_is_not_yet() {
    let r = refuse_with(
        "obs_data = [51.0, 48.0]\nmu = elementof(posreals)\n\
         L = likelihoodof(functionof(Poisson.(mu .* [50.0, 40.0])), obs_data)\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("real-typed data under the discrete law `Poisson`"),
        "{r}"
    );
}

#[test]
fn interpolator_broadcast_with_matrix_arguments_is_not_yet() {
    // The interpolators have scalar and vector forms only.
    let r = refuse_with(
        "hep = standard_module(\"particle-physics\", \"0.1\")\nalpha = elementof(reals)\n\
         lo = rowstack([[0.9, 0.95], [0.9, 0.95]])\nc = rowstack([[1.0, 1.0], [1.0, 1.0]])\n\
         hi = rowstack([[1.1, 1.2], [1.1, 1.2]])\nf = hep.interp_poly6_exp.(lo, c, hi, alpha)\n\
         L = likelihoodof(functionof(Normal.(f[1, :], 1.0)), [1.0, 1.0])\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("argument shapes"), "{r}");
}

#[test]
fn histosys_broadcast_with_a_matrix_alpha_is_not_yet() {
    // The matrix form takes one alpha per row (`addaxes(u, 0, 1)`), not a
    // full matrix.
    let r = refuse_with(
        "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\nalpha = elementof(cartpow(reals, [2, 2]))\n\
         lo = rowstack([[0.9, 0.95], [0.9, 0.95]])\nnom = rowstack([[1.0, 1.0], [1.0, 1.0]])\n\
         hi = rowstack([[1.1, 1.2], [1.1, 1.2]])\nf = pyhf.histosys_shift.(lo, nom, hi, alpha)\n\
         L = likelihoodof(functionof(Normal.(f[1, :], 1.0)), [1.0, 1.0])\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("argument shapes"), "{r}");
}

#[test]
fn rank3_array_forms_without_a_stan_type_are_not_yet() {
    for (array, reason) in [
        (
            "array(fill(mu, 4), [s, 1, 2], [1, 2, 3])",
            "non-literal size",
        ),
        (
            "array(fill(mu, 4), [1, 2, 2], [3, 2, 1])",
            "another dimorder",
        ),
    ] {
        let r = refuse_with(
            &format!(
                "mu = elementof(reals)\ns = sum([1, 1])\nx = {array}\n\
                 L = likelihoodof(functionof(Normal.(x[1, 1, :], 1.0)), [1.0, 1.0])\n"
            ),
            Some("L"),
        );
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
        assert!(r.reason.contains(reason), "{array}: {r}");
    }
}

#[test]
fn arithmetic_on_a_rank3_value_is_not_yet() {
    // Stan has no arithmetic on an array of matrices, and element loops over a
    // broadcast are not an answer.
    for (x2, reason) in [
        ("x .* 2.0", "arithmetic on a rank-3 value"),
        ("exp.(x)", "arithmetic on a rank-3 value"),
        (
            "array([1.0, 2.0], [1, 1, 2], [1, 2, 3]) .+ x",
            "arithmetic on a rank-3 value",
        ),
        (
            "aggregate(sum, [.s, .k, .b], x[.s, .k, .b] * 2.0)",
            "aggregate with a rank-3 result",
        ),
        ("f.(x)", "broadcast with a rank-3 result"),
    ] {
        let r = refuse_with(
            &format!(
                "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\nmu = elementof(reals)\n\
                 x = fill(mu, [1, 1, 2])\nn = rowstack([[10.0, 20.0]])\ns0 = fill(0.0, [1, 1, 2])\n\
                 f(t) = t * 2.0\nx2 = {x2}\ny = pyhf.sample_yields(n, s0, x2)\n\
                 L = likelihoodof(functionof(Normal.(y[1, :], 1.0)), [1.0, 2.0])\n"
            ),
            Some("L"),
        );
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{x2}: {r}");
        assert!(r.reason.contains(reason), "{x2}: {r}");
    }
}

#[test]
fn rank3_slice_across_the_array_axis_with_the_last_axis_fixed_is_not_yet() {
    // `x[:, :, k]` of an array of matrices is an `array[] vector`, and Stan
    // has no `to_matrix` of one.
    let r = refuse_with(
        "mu = elementof(reals)\nx = fill(mu, [2, 1, 2])\ny = x[:, :, 1]\n\
         L = likelihoodof(functionof(Normal.(y[:, 1], 1.0)), [1.0, 2.0])\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("x[:, :, k]"), "{r}");
}

#[test]
fn law_over_a_rank3_variate_is_not_yet() {
    // Stan flattens a matrix or a rank-2 array in one statement, not an array
    // of matrices.
    let r = refuse_with(
        "mu = elementof(reals)\nx = array(fill(mu, 4), [1, 2, 2], [1, 2, 3])\n\
         L = likelihoodof(functionof(Normal.(x, 1.0)), fill(1.0, [1, 2, 2]))\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("rank-3 variate"), "{r}");
}

#[test]
fn iid_of_a_broadcast_law_over_an_array_of_vectors_is_not_yet() {
    let r = refuse_with(
        "mu = elementof(reals)\ny = [[1.0, 2.0], [3.0, 4.0]]\n\
         L = likelihoodof(functionof(iid(Normal.(fill(mu, 2), 1.0), 2)), y)\n",
        Some("L"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("broadcast"), "{r}");
}

#[test]
fn normalize_holding_continued_poisson_is_not_yet() {
    // ContinuedPoisson has mass Z(rate) != 1; a normalize around it would
    // print its density as if Z were 1.
    let hep = "hep = standard_module(\"particle-physics\", \"0.1\")\n";
    for model in [
        // The hurdle split beside a Dirac atom.
        "y_obs = [0.0, 1.3, 0.0, 0.4, 2.2]\nhu ~ Beta(2, 2)\nmu ~ Gamma(2, 1)\n\
         m = normalize(superpose(weighted(hu, Dirac(0.0)), weighted(1 - hu, hep.ContinuedPoisson(rate = mu))))\n\
         y ~ iid(m, 5)\nprior = lawof(record(hu = hu, mu = mu))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), hu = hu, mu = mu), record(y = y_obs)), prior)\n",
        // R1's `normalize(weighted(c, M))`.
        "y_obs = [1.3, 0.4, 2.2]\ns = 2.0\nmu ~ Gamma(2, 1)\n\
         y ~ iid(normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), hep.ContinuedPoisson(rate = mu))), 3)\n\
         prior = lawof(record(mu = mu))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), mu = mu), record(y = y_obs)), prior)\n",
        // A broadcast user law whose body is the hurdle.
        "x = [0.2, -0.5, 1.0, 0.0, 0.7]\ny_obs = [0.0, 1.3, 0.0, 0.4, 2.2]\nhu ~ Beta(2, 2)\nb ~ Normal(0, 1)\n\
         f = (r, h) -> normalize(superpose(weighted(h, Dirac(0.0)), weighted(1.0 - h, hep.ContinuedPoisson(rate = r))))\n\
         y ~ f.(exp.(b .* x), hu)\nprior = lawof(record(hu = hu, b = b))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), hu = hu, b = b), record(y = y_obs)), prior)\n",
    ] {
        let r = refuse_with(&format!("{hep}{model}"), None);
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
        assert!(
            r.reason.contains("unnormalized law `ContinuedPoisson`"),
            "{r}"
        );
    }
}

#[test]
fn real_typed_matrix_data_under_a_discrete_law_is_not_yet() {
    let r = refuse_with(
        "y_obs = [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]\nmu ~ Gamma(2, 1)\ny ~ iid(iid(Poisson(mu), 3), 2)\n\
         prior = lawof(record(mu = mu))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), mu = mu), record(y = y_obs)), prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("real-typed data under the discrete law `Poisson`"),
        "{r}"
    );
}

#[test]
fn custom_prior_bounded_by_a_derived_value_is_not_yet() {
    // `upper=m` would read a model-block value in the parameters block.
    let r = refuse_with(
        "hep = standard_module(\"particle-physics\", \"0.1\")\nd ~ Normal(0, 0.1)\nm = 5.3 + d\n\
         x ~ hep.Argus(resonance = m, slope = -20.0, power = 0.5)\ny ~ Normal(x, 1)\n\
         prior = lawof(record(d = d, x = x))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), d = d, x = x), record(y = 5.0)), prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("derived value"), "{r}");
}

#[test]
fn an_output_that_is_not_a_log_density_has_no_stan_form() {
    for (out, reason) in [
        ("outputs = (lp, x)", "several values"),
        ("outputs = ifelse(x > 0.0, x, 0.0)", "`ifelse`"),
        ("outputs = densityof(Normal(0.0, 1.0), x)", "`densityof`"),
    ] {
        let r = refuse_with(
            &format!(
                "x = elementof(reals)\nlp = logdensityof(Normal(0.0, 1.0), x)\ninputs = (x)\n{out}\n"
            ),
            Some("outputs"),
        );
        assert_eq!(r.kind, RefusalKind::NotADensity, "{out}: {r}");
        assert!(r.reason.contains(reason), "{out}: {r}");
    }
}

#[test]
fn log_density_sums_this_emitter_does_not_write_are_not_yet() {
    for (out, reason) in [
        (
            "outputs = logdensityof(Normal(0.0, 1.0), x) - logdensityof(Normal(1.0, 1.0), x)",
            "difference of log densities",
        ),
        (
            "outputs = sum([logdensityof(Normal(0.0, 1.0), x)] .* 2.0)",
            "computed list of log densities",
        ),
        (
            "outputs = logdensityof(post, record(m = x)) + logdensityof(Normal(0.0, 1.0), x)",
            "posterior term",
        ),
    ] {
        let r = refuse_with(
            &format!(
                "m ~ Normal(0.0, 1.0)\ny ~ Normal(m, 1.0)\n\
                 L = likelihoodof(kernelof(record(y = y), m = m), record(y = 0.5))\n\
                 post = bayesupdate(L, lawof(record(m = m)))\nx = elementof(reals)\ninputs = (x)\n{out}\n"
            ),
            Some("outputs"),
        );
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{out}: {r}");
        assert!(r.reason.contains(reason), "{out}: {r}");
    }
}

#[test]
fn an_ordered_prior_field_from_an_unordered_input_is_not_yet() {
    // The `ordered` field's restriction lives in a declaration; an input
    // declared on `reals` carries none of it.
    let r = refuse_with(
        "v ~ tilt(iid(Normal(0.0, 2.0), 2), Lebesgue(ordered(2)))\ny ~ Normal(v[1], 1.0)\n\
         L = likelihoodof(kernelof(record(y = y), v = v), record(y = 0.5))\n\
         post = bayesupdate(L, lawof(record(v = v)))\nx = elementof(cartpow(reals, 2))\ninputs = (x)\n\
         outputs = logdensityof(post, record(v = x))\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("declared set"), "{r}");
}

#[test]
fn a_real_input_under_a_discrete_law_is_not_yet() {
    // Stan has no integer parameters, and a real value is not cast.
    let r = refuse_with(
        "t_y = elementof(cartpow(reals, 3))\ninputs = (t_y)\n\
         outputs = logdensityof(broadcast(Poisson, [2.0, 3.5, 1.0]), t_y)\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("real-typed input under the discrete law"),
        "{r}"
    );
}

fn emit_code(source: &str, target: &str) -> String {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let opts = EmitOptions {
        target: Some(target.to_string()),
        ..Default::default()
    };
    flatppl_stan::emit(&module, &opts).expect("emits").code
}

/// The `data` and `parameters` blocks of a program.
fn declarations(code: &str) -> String {
    code.split("model {").next().unwrap_or_default().to_string()
}

#[test]
fn query_inputs_do_not_depend_on_term_order() {
    // `t` is read by a measure in one term: a parameter in either order.
    let a = emit_code(
        "t = elementof(reals)\nlp = logdensityof(Normal(0.0, 1.0), t) + logdensityof(Normal(t, 1.0), 0.5)\n",
        "lp",
    );
    let b = emit_code(
        "t = elementof(reals)\nlp = logdensityof(Normal(t, 1.0), 0.5) + logdensityof(Normal(0.0, 1.0), t)\n",
        "lp",
    );
    assert_eq!(declarations(&a), declarations(&b), "{a}\n{b}");
    assert!(declarations(&a).contains("parameters {\n  real t;"), "{a}");
}

#[test]
fn an_input_outside_the_declared_signature_is_ill_formed() {
    // §13: `inputs` lists every `elementof` an output reaches.
    for out in [
        "outputs = logdensityof(Normal(a, 1.0), b)",
        "outputs = logdensityof(Normal(b, 1.0), a)",
    ] {
        let r = refuse_with(
            &format!("a = elementof(reals)\nb = elementof(reals)\ninputs = (a)\n{out}\n"),
            Some("outputs"),
        );
        assert_eq!(r.kind, RefusalKind::IllFormedModule, "{out}: {r}");
        assert!(r.kind.is_never(), "{out}: {r}");
        assert!(r.reason.contains("`inputs` does not list"), "{out}: {r}");
    }
}

#[test]
fn an_input_behind_a_reification_boundary_is_not_free() {
    // `functionof(b)` makes `a` an input of `f`; the call sets it, so the
    // output does not read the leaf `a` (§13 does not apply).
    let r = refuse_with(
        "a = elementof(reals)\nb = 2.0 * a\nf = functionof(b)\nmu = f(a = 0.5)\n\
         x = elementof(reals)\ninputs = (x)\noutputs = logdensityof(Normal(mu, 1.0), x)\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(!r.reason.contains("§13"), "{r}");
    assert!(
        r.reason
            .contains("a user function called with keyword arguments"),
        "{r}"
    );
}

#[test]
fn a_positional_function_called_by_keyword_is_not_yet_supported() {
    // Infer rejects a true arity mismatch, so the arity text never reaches
    // a well-typed module; a keyword call of a positional boundary names
    // the keyword form.
    let r = refuse_with(
        "a = elementof(reals)\nb = 2.0 * a\nfe = functionof(b, a)\n\
         lp = logdensityof(Normal(fe(a = 0.5), 1.0), 0.3)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a user function called with keyword arguments"),
        "{r}"
    );
}

#[test]
fn an_input_read_beside_its_bound_use_is_still_free() {
    // `f(0.5)` sets the boundary input `a`, but `+ a` reads the leaf itself.
    let r = refuse_with(
        "a = elementof(reals)\nb = 2.0 * a + 1.0\nf = functionof(b, a)\n\
         x = elementof(reals)\ninputs = (x)\noutputs = logdensityof(Normal(f(0.5) + a, 1.0), x)\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason.contains("`a`, which `inputs` does not list"),
        "{r}"
    );
}

#[test]
fn sum_and_density_targets_name_what_the_user_wrote() {
    let r = refuse_with(
        "x = elementof(reals)\ninputs = (x)\noutputs = logdensityof(Normal(0.0, 1.0), x) + 1.0\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("a term that is not a `logdensityof`"),
        "{r}"
    );
    let r = refuse_with(
        "x = elementof(reals)\ninputs = (x)\noutputs = exp(logdensityof(Normal(0.0, 1.0), x))\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotADensity, "{r}");
    let r = refuse_with(
        "x = elementof(reals)\ninputs = (x)\noutputs = 2.0 * logdensityof(Normal(0.0, 1.0), x)\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
}

#[test]
fn a_nested_record_observation_is_not_yet_supported() {
    let r = refuse_with(
        "t1 = elementof(reals)\nt2 = elementof(posreals)\n\
         prior = joint(t1 = Normal(0, 1), t2 = Exponential(1))\n\
         K = functionof(joint(a = Normal(t1, 1.0), b = joint(c = Normal(t1, t2), d = Normal(0.0, t2))))\n\
         post = bayesupdate(likelihoodof(K, record(a = 0.4, b = record(c = 0.1, d = -0.2))), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a nested record observation"), "{r}");
}

#[test]
fn a_not_a_density_refusal_names_the_call_as_written() {
    let r = refuse_with(
        "mu = elementof(reals)\nx ~ Normal(mu, 1.0)\nkernel = kernelof(x, mu = mu)\n\
         dist = kernel(mu = 1.5)\n",
        Some("dist"),
    );
    assert_eq!(r.kind, RefusalKind::NotADensity, "{r}");
    assert!(r.reason.contains("`kernel(…)`"), "{r}");
    assert!(!r.to_string().contains('%'), "{r}");
}

#[test]
fn a_restricted_joint_law_has_a_density_not_yet_written() {
    // `restrict` conditions the joint on observed fields: a density, so not
    // `NotADensity`.
    let src = "theta ~ Normal(0.0, 1.0)\nobs ~ iid(Normal(theta, 1.0), 2)\n\
               joint_model = lawof(record(theta = theta, obs = obs))\n\
               post = restrict(joint_model, record(obs = [0.3, -0.2]))\n";
    let r = refuse_with(src, Some("post"));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("`restrict` of a joint law"), "{r}");
    let r = refuse_with(
        &format!(
            "{src}t = elementof(reals)\ninputs = (t)\noutputs = logdensityof(post, record(theta = t))\n"
        ),
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("`restrict` of a joint law"), "{r}");
}

#[test]
fn a_log_density_written_through_densityof_is_not_yet_supported() {
    let r = refuse_with(
        "x = elementof(reals)\ninputs = (x)\noutputs = log(densityof(Normal(0.0, 1.0), x))\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("write `logdensityof(…)`"), "{r}");
    let r = refuse_with(
        "x = elementof(reals)\ninputs = (x)\n\
         outputs = log(densityof(Normal(0.0, 1.0), x)) + log(densityof(Normal(1.0, 1.0), x))\n",
        Some("outputs"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
}

#[test]
fn a_likelihood_point_with_an_unknown_key_is_ill_formed() {
    let r = refuse_with(
        "mu = elementof(reals)\nL = likelihoodof(Normal(mu, 1.0), 0.5)\n\
         lp = logdensityof(L, record(zz = 0.3))\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason.contains(
            "the point sets `zz`, which is not an input of the likelihood (inputs: `mu`)"
        ),
        "{r}"
    );
}

#[test]
fn a_likelihood_point_with_a_missing_key_is_ill_formed() {
    let r = refuse_with(
        "mu = elementof(reals)\nsg = elementof(posreals)\nL = likelihoodof(Normal(mu, sg), 0.5)\n\
         lp = logdensityof(L, record(mu = 0.3))\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason
            .contains("the point does not set the likelihood input `sg` (inputs: `mu`, `sg`)"),
        "{r}"
    );
}

#[test]
fn a_point_value_that_reads_the_input_it_sets_is_refused() {
    let r = refuse_with(
        "mu = elementof(reals)\nL = likelihoodof(Normal(mu, 1.0), 0.5)\n\
         lp = logdensityof(L, record(mu = mu + 1.0))\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("the point value of `mu` reads `mu`, the input it sets"),
        "{r}"
    );
}
