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
fn mixture_is_not_yet() {
    let r = refuse(&posterior(
        "w ~ Beta(2, 2)\ny ~ normalize(superpose(weighted(w, Normal(0, 1)), weighted(1 - w, Normal(3, 1))))",
        "lawof(record(w = w))",
        "kernelof(record(y = y), w = w)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(!r.kind.is_never());
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
fn general_scan_is_not_yet() {
    let r = refuse(&posterior(
        "xs = [1.0, 2.0, 3.0]\n\
         run = scan((acc, x) -> 0.5 * acc + x, 0.0, xs)\n\
         mu ~ Normal(0, 1)\ny ~ Normal(mu + run[3], 1)",
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("scan"), "{r}");
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
