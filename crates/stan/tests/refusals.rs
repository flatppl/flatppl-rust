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
fn dirac_with_continuous_component_is_never() {
    let r = refuse_with(
        "w ~ Beta(2, 2)\nm = normalize(superpose(weighted(w, Normal(0, 1)), weighted(1 - w, Dirac(0.0))))\n\
         y ~ iid(m, 3)\nprior = lawof(record(w = w))\n\
         L = likelihoodof(kernelof(record(y = y), w = w), record(y = [0.0, 0.5, 1.0]))\n\
         post = bayesupdate(L, prior)\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::MixedReference, "{r}");
    assert!(r.kind.is_never());
}

#[test]
fn unnormalized_mixture_component_is_never() {
    let r = refuse(&posterior(
        "w ~ Beta(2, 2)\ny ~ normalize(superpose(weighted(w, truncate(Normal(0, 1), interval(0.0, 2.0))), weighted(1 - w, Normal(3, 1))))",
        "lawof(record(w = w))",
        "kernelof(record(y = y), w = w)",
    ));
    assert_eq!(r.kind, RefusalKind::NumericNormalizer, "{r}");
    assert!(r.reason.contains("mixture component"), "{r}");
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
fn logdensityof_of_positive_truncation_is_not_yet() {
    let r = refuse(&posterior(
        "mu = elementof(reals)\ny ~ Normal(mu, 1)\n\
         w(x) = logdensityof(truncate(Normal(0, 1), posreals), x.mu)",
        "logweighted(w, Lebesgue(cartprod(mu = reals)))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("truncated"), "{r}");
}

#[test]
fn positive_truncation_mixture_component_is_never() {
    let r = refuse(&posterior(
        "w ~ Beta(2, 2)\ny ~ normalize(superpose(weighted(w, truncate(Normal(0, 1), posreals)), weighted(1 - w, Normal(3, 1))))",
        "lawof(record(w = w))",
        "kernelof(record(y = y), w = w)",
    ));
    assert_eq!(r.kind, RefusalKind::NumericNormalizer, "{r}");
    assert!(r.reason.contains("mixture component"), "{r}");
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
