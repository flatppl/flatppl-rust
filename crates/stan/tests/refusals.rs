//! One refusal per never-kind, plus the not-yet and target cases. Each message
//! must name the user's construct.

use flatppl_stan::{EmitOptions, Refusal, RefusalKind};

fn refuse(source: &str) -> Refusal {
    refuse_with(source, None)
}

fn refuse_with(source: &str, target: Option<&str>) -> Refusal {
    refuse_located(source, target).0
}

/// The refusal and the binding it is located in. A name the parser made up
/// (`__0x1` for `K, prior = …`) means nothing to the user.
fn refuse_located(source: &str, target: Option<&str>) -> (Refusal, Option<String>) {
    refuse_in(source, &flatppl_infer::ModuleBundle::new(), target)
}

/// [`refuse_located`] for a model that loads `mod.flatppl`, whose source is
/// `dependency`: inference resolves the module as the CLI does.
fn refuse_with_module(
    source: &str,
    dependency: &str,
    target: Option<&str>,
) -> (Refusal, Option<String>) {
    refuse_in(source, &module_bundle(dependency), target)
}

/// A bundle holding `dependency` as `mod.flatppl`.
fn module_bundle(dependency: &str) -> flatppl_infer::ModuleBundle {
    let mut bundle = flatppl_infer::ModuleBundle::new();
    bundle.insert(
        "mod.flatppl",
        std::sync::Arc::new(flatppl_syntax::parse(dependency).expect("parse dependency")),
    );
    bundle
}

fn refuse_in(
    source: &str,
    bundle: &flatppl_infer::ModuleBundle,
    target: Option<&str>,
) -> (Refusal, Option<String>) {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer_module(&mut module, bundle, flatppl_infer::Level::Shape);
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
    // As `flatppl stan` emits: `load_module` members are imported first.
    let r = flatppl_stan::emit_with_bundle(&module, bundle, &opts).expect_err("refuses");
    let located = r.node.and_then(|node| enclosing_binding(&module, node));
    if let Some(name) = &located {
        assert!(!name.starts_with("__"), "located in `{name}`: {r}");
    }
    (r, located)
}

fn enclosing_binding(module: &flatppl_core::Module, node: flatppl_core::NodeId) -> Option<String> {
    for (_, binding) in module.bindings() {
        let mut queue = vec![binding.rhs];
        while let Some(id) = queue.pop() {
            if id == node {
                return Some(module.resolve(binding.name).to_string());
            }
            module.for_each_child(id, |child| queue.push(child));
        }
    }
    None
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

/// A Poisson process query `lp = logdensityof(<law>, <x>)` over `inputs`.
fn process_query(decls: &str, law: &str, x: &str, inputs: &str) -> String {
    format!("{decls}\nlp = logdensityof({law}, {x})\ninputs = {inputs}\noutputs = lp\n")
}

/// A function-weighted intensity has no closed mass here; Stan's
/// `integrate_1d` and a polynomial's closed mass are later forms.
#[test]
fn point_process_is_not_yet() {
    let r = refuse_with(
        &process_query(
            "a = elementof(nonnegreals)\n\
             I = weighted((x, p) -> p * exp(x), Lebesgue(interval(0.0, 1.0)), a)",
            "BinnedPoissonProcess([0.0, 0.5, 1.0], I)",
            "[0, 1]",
            "a",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("an intensity weighted by a function of the variate"),
        "{r}"
    );
}

/// A truncation to an interval outside the support has no mass, and
/// `normalize` of it has no density.
#[test]
fn normalized_truncation_outside_the_support_is_ill_formed() {
    let r = refuse_with(
        &process_query(
            "r = elementof(posreals)",
            "normalize(truncate(Exponential(r), interval(-2.0, -1.0)))",
            "0.5",
            "r",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason
            .contains("a truncation to an interval outside the support has zero mass"),
        "{r}"
    );
}

/// The same for a normalized intensity.
#[test]
fn normalized_intensity_of_zero_mass_is_ill_formed() {
    let r = refuse_with(
        &process_query(
            "r = elementof(posreals)",
            "PoissonProcess(normalize(truncate(Exponential(r), interval(-2.0, -1.0))))",
            "[0.5]",
            "r",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason.contains("§06") && r.reason.contains("has zero mass"),
        "{r}"
    );
}

#[test]
fn discrete_intensity_is_not_yet() {
    let r = refuse_with(
        &process_query(
            "r = elementof(posreals)",
            "BinnedPoissonProcess([0.5, 1.5, 2.5], weighted(3.0, Poisson(r)))",
            "[1, 2]",
            "r",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a discrete intensity (`Poisson`)"), "{r}");
}

#[test]
fn atom_intensity_is_not_yet() {
    let r = refuse_with(
        &process_query(
            "s = elementof(posreals)",
            "PoissonProcess(weighted(s, Dirac(1.0)))",
            "[1.0]",
            "s",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("an intensity with an atom (`Dirac`)"),
        "{r}"
    );
}

/// §06 takes a finite-mass intensity: `Lebesgue(reals)` has none.
#[test]
fn infinite_mass_intensity_is_ill_formed() {
    let r = refuse_with(
        &process_query(
            "b = elementof(posreals)",
            "PoissonProcess(weighted(b, Lebesgue(support = reals)))",
            "[1.0]",
            "b",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason.contains("§06") && r.reason.contains("finite mass"),
        "{r}"
    );
}

/// The binned law takes a finite-mass intensity too (§06), even where each
/// bin's mass is finite.
#[test]
fn infinite_mass_binned_intensity_is_ill_formed() {
    let r = refuse_with(
        &process_query(
            "s = elementof(posreals)",
            "BinnedPoissonProcess([0.0, 1.0], weighted(s, Lebesgue(support = reals)))",
            "[1]",
            "s",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason.contains("§06") && r.reason.contains("finite mass"),
        "{r}"
    );
}

/// A negative literal weight is ill-formed (§06 `weighted`), as in a mixture.
#[test]
fn negative_intensity_weight_is_ill_formed() {
    let r = refuse_with(
        &process_query(
            "s = elementof(posreals)",
            "PoissonProcess(weighted(-2.0, Normal(0.0, 1.0)))",
            "[0.5]",
            "s",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(r.reason.contains("the weight -2 is negative"), "{r}");
}

#[test]
fn kernel_intensity_is_not_yet() {
    let r = refuse_with(
        &process_query(
            "mu = elementof(reals)\nK = functionof(weighted(2.0, Normal(mu, 1.0)), mu = mu)",
            "PoissonProcess(K)",
            "[0.5]",
            "mu",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a kernel intensity"), "{r}");
}

#[test]
fn record_bin_edges_are_not_yet() {
    let r = refuse_with(
        &process_query(
            "s = elementof(posreals)\n\
             I = weighted(s, joint(x = Uniform(interval(0, 2)), y = Uniform(interval(0, 3))))",
            "BinnedPoissonProcess(record(x = [0, 0.5, 2], y = [0, 1, 2, 3]), I)",
            "rowstack([[1, 1, 1], [3, 3, 3]])",
            "s",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a record of edge vectors"), "{r}");
}

/// Record-valued points are a table, and a table value refuses before the
/// process does: the record-point refusal waits for tables.
#[test]
fn record_valued_points_are_not_yet() {
    let r = refuse_with(
        &process_query(
            "s = elementof(posreals)\npts = elementof(cartpow(cartprod(x = reals, y = reals), 2))",
            "PoissonProcess(weighted(s, joint(x = Normal(0.0, 1.0), y = Normal(0.0, 1.0))))",
            "pts",
            "(s, pts)",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a record value"), "{r}");
}

#[test]
fn intensity_without_lcdf_is_not_yet() {
    let r = refuse_with(
        &process_query(
            "hep = standard_module(\"particle-physics\", \"0.1\")\ns = elementof(posreals)",
            "BinnedPoissonProcess([5.0, 5.2, 5.4], weighted(s, hep.CrystalBall(5.28, 0.05, 1.5, 3.0)))",
            "[1, 2]",
            "s",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("the intensity law `CrystalBall`, which has no `_lcdf` in Stan"),
        "{r}"
    );
}

#[test]
fn dynamic_bin_edge_count_is_not_yet() {
    let r = refuse_with(
        &process_query(
            "n = elementof(posintegers)",
            "BinnedPoissonProcess(linspace(0.0, 2.0, n), weighted(5.0, Normal(1.0, 0.5)))",
            "[1, 3]",
            "n",
        ),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("a bin-edge vector of non-static length"),
        "{r}"
    );
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
fn markovchain_with_a_state_free_step_is_one_statement_without_a_lag() {
    // A step that does not read the state: the chain is the step law per
    // element, and no lag is built.
    let code = emit_code(
        "T = 3\np ~ Uniform(unitinterval)\nstep(previous, q) = Bernoulli(q)\n\
         y ~ markovchain(step, 0, T, p)\nprior = lawof(record(p = p))\n\
         L = likelihoodof(kernelof(record(y = y), p = p), record(y = [0, 1, 1]))\n\
         post = bayesupdate(L, prior)\n",
        "post",
    );
    assert!(code.contains("y ~ bernoulli(p);"), "{code}");
    assert!(!code.contains("_lag"), "{code}");
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
fn pushfwd_outside_the_bijection_registry_is_not_yet() {
    let r = refuse(&posterior(
        "rho ~ pushfwd(fn(sin(_)), Normal(0, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("pushfwd through `sin`, which is not in the §06 bijection registry"),
        "{r}"
    );
}

#[test]
fn pushfwd_sqrt_of_a_normal_is_outside_its_domain() {
    let r = refuse(&posterior(
        "rho ~ pushfwd(fn(sqrt(_)), Normal(0, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "pushfwd through `sqrt` of a law whose support is not within its domain (§06)"
        ),
        "{r}"
    );
}

#[test]
fn pushfwd_pow_with_a_computed_exponent_is_not_yet() {
    let r = refuse(&posterior(
        "ks = [2.0, 3.0]\nrho ~ pushfwd(fn(pow(_, ks[1])), Gamma(2, 2))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("not a finite nonzero literal"), "{r}");
}

#[test]
fn pushfwd_through_a_map_that_reads_its_input_twice_is_not_yet() {
    let r = refuse(&posterior(
        "f(t) = t + exp(t)\nrho ~ pushfwd(f, Normal(0, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("reads its input more than once"), "{r}");
}

#[test]
fn pushfwd_negative_power_of_a_law_around_zero_is_not_yet() {
    for map in ["fn(pow(_, -1))", "fn(pow(_, -3))", "fn(1 / _)"] {
        let r = refuse(&posterior(
            &format!("rho ~ pushfwd({map}, Normal(0, 1))\ny ~ Normal(rho, 1)"),
            "lawof(record(rho = rho))",
            "kernelof(record(y = y), rho = rho)",
        ));
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{map}: {r}");
        assert!(
            r.reason.contains(
                "pushfwd through `pow` with a negative exponent of a law whose support contains 0, where it is not monotone"
            ),
            "{map}: {r}"
        );
    }
}

#[test]
fn pushfwd_of_a_support_with_a_symbolic_end_names_the_end() {
    let r = refuse(&posterior(
        "lo_s = [0.5, 1.0]\nrho ~ pushfwd(fn(log(_)), Uniform(interval(lo_s[1], 2.0)))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "whose support end `lo_s[1]` is not a literal, so its domain cannot be checked"
        ),
        "{r}"
    );
}

#[test]
fn pushfwd_through_a_bare_name_outside_the_registry_is_not_yet() {
    let r = refuse(&posterior(
        "rho ~ pushfwd(sin, Normal(0, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("pushfwd through `sin`, which is not in the §06 bijection registry"),
        "{r}"
    );
}

/// The emitted program of `post` for a model with a `pushfwd` map spelled
/// `map`, as a prior field and as an observed law.
fn bare_and_written(map: &str, base: &str, observed_base: &str) -> [flatppl_stan::StanProgram; 2] {
    let opts = EmitOptions {
        target: Some("post".to_string()),
        ..Default::default()
    };
    let field = posterior(
        &format!("rho ~ pushfwd({map}, {base})\ny ~ Normal(rho, 1)"),
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    );
    let observed = posterior(
        &format!("mu ~ Exponential(1)\ny ~ pushfwd({map}, {observed_base})"),
        "lawof(record(mu = mu))",
        "kernelof(record(y = y), mu = mu)",
    );
    [field, observed].map(|source| {
        let mut module = flatppl_syntax::parse(&source).expect("parse");
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(
            diagnostics
                .iter()
                .all(|d| d.severity != flatppl_infer::Severity::Error),
            "{source}: {diagnostics:?}"
        );
        flatppl_stan::emit(&module, &opts).unwrap_or_else(|r| panic!("{source}: {r}"))
    })
}

#[test]
fn a_bare_registry_name_prints_as_its_written_map() {
    // §06 recognizes the registry maps by name: `pushfwd(log, M)` is
    // `pushfwd(fn(log(_)), M)`, as a prior field and as an observed law.
    let unbounded = ("Normal(0, 1)", "Normal(0, mu)");
    let positive = ("Gamma(2, 1)", "Gamma(2, mu)");
    let unit = ("Beta(2, 3)", "Beta(2, mu)");
    for (name, (base, observed)) in [
        ("exp", unbounded),
        ("log", positive),
        ("log10", positive),
        ("log1p", positive),
        ("expm1", unbounded),
        ("logit", unit),
        ("invlogit", unbounded),
        ("probit", unit),
        ("invprobit", unbounded),
        ("atan", unbounded),
        ("sinh", unbounded),
        ("asinh", unbounded),
        ("tanh", unbounded),
        ("sqrt", positive),
        ("neg", ("Exponential(2)", "Exponential(mu)")),
    ] {
        let bare = bare_and_written(name, base, observed);
        let written = bare_and_written(&format!("fn({name}(_))"), base, observed);
        for (b, w) in bare.iter().zip(&written) {
            assert_eq!(b.code, w.code, "{name}");
            assert_eq!(b.data, w.data, "{name}");
        }
    }
}

#[test]
fn a_bare_name_outside_its_domain_refuses_as_its_written_map() {
    let source = |map: &str| {
        posterior(
            &format!("rho ~ pushfwd({map}, Normal(0, 1))\ny ~ Normal(rho, 1)"),
            "lawof(record(rho = rho))",
            "kernelof(record(y = y), rho = rho)",
        )
    };
    let (bare, written) = (refuse(&source("log")), refuse(&source("fn(log(_))")));
    assert_eq!(
        (&bare.kind, &bare.reason),
        (&written.kind, &written.reason),
        "{bare}"
    );
}

#[test]
fn a_bare_name_over_a_vector_law_is_an_inference_error_and_its_written_form_not_yet() {
    // A scalar-domain name is no map of an array variate: inference reports
    // it, naming `fn(exp.(_))`, which the emitter does not write yet.
    let source = |map: &str| {
        posterior(
            &format!(
                "mu ~ Normal(0, 1)\nv ~ pushfwd({map}, iid(Normal(mu, 1.0), 3))\ny ~ Normal(mu, 1)"
            ),
            "lawof(record(mu = mu))",
            "kernelof(record(v = v, y = y), mu = mu)",
        )
        .replace("record(y = 0.5)", "record(v = [1.2, 0.5, 2.0], y = 0.5)")
    };
    let mut module = flatppl_syntax::parse(&source("exp")).expect("parse");
    let diagnostics = flatppl_infer::infer_module(
        &mut module,
        &flatppl_infer::ModuleBundle::new(),
        flatppl_infer::Level::Shape,
    );
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error
                && d.message
                    .contains("Write the elementwise map `fn(exp.(_))`")),
        "{diagnostics:?}"
    );
    for written in ["fn(broadcast(exp, _))", "fn(exp.(_))"] {
        let written = refuse(&source(written));
        assert_eq!(written.kind, RefusalKind::NotYetSupported, "{written}");
    }
}

#[test]
fn a_map_of_a_lower_dimensional_law_has_no_stan_form_but_a_uniform_scale() {
    // §03/§06 fix the image's reference measure only for a uniform scale and
    // shift of a stdsimplex or zerosum support; the rest has no density.
    let source = |map: &str, base: &str, v: &str| {
        posterior(
            &format!("mu ~ Normal(0, 1)\nv ~ pushfwd({map}, {base})\ny ~ Normal(mu, 1)"),
            "lawof(record(mu = mu))",
            "kernelof(record(v = v, y = y), mu = mu)",
        )
        .replace("record(y = 0.5)", &format!("record(v = {v}, y = 0.5)"))
    };
    let dirichlet = "Dirichlet([2.0, 3.0, 4.0])";
    let point = "[1.4, 1.6, 2.0]";
    let lkj = "rowstack([[2.0, 0.0], [0.0, 2.0]])";
    for (map, base, v) in [
        ("fn(exp.(_))", dirichlet, point),
        ("fn([2.0, 3.0, 4.0] .* _)", dirichlet, point),
        ("fn(2.0 * _)", "LKJ(2, 2.0)", lkj),
        ("fn(2.0 * _)", "LKJCholesky(2, 2.0)", lkj),
    ] {
        let r = refuse(&source(map, base, v));
        assert_eq!(
            r.kind,
            RefusalKind::NonBijectivePushfwd,
            "{map} {base}: {r}"
        );
        assert!(r.reason.contains("lower-dimensional support"), "{r}");
    }
    for map in [
        "fn(2.0 .* _ .+ 1.0)",
        "fn(2.0 * _ + [1.0, 1.0, 1.0])",
        "neg",
    ] {
        let r = refuse(&source(map, dirichlet, point));
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{map}: {r}");
    }
    // A measure that is no law reaches the check through a log-density target,
    // and a wrapper that keeps the support does not hide it.
    let target = |m: &str| {
        format!(
            "y = elementof(cartpow(reals, 3))\nM = {m}\nlp = logdensityof(M, y)\n\
             inputs = (y)\noutputs = (lp)\n"
        )
    };
    for m in [
        "pushfwd(fn(exp.(_)), Lebesgue(zerosum(3)))",
        "pushfwd(fn(exp.(_)), Lebesgue(stdsimplex(3)))",
        "pushfwd(fn(exp.(_)), normalize(Lebesgue(stdsimplex(3))))",
        "pushfwd(fn(exp.(_)), logweighted(fn(-0.5 * sum(_ .^ 2)), Lebesgue(zerosum(3))))",
    ] {
        let r = refuse_with(&target(m), Some("lp"));
        assert_eq!(r.kind, RefusalKind::NonBijectivePushfwd, "{m}: {r}");
        assert!(r.reason.contains("lower-dimensional support"), "{r}");
    }
    let r = refuse_with(
        &target("pushfwd(fn(2.0 .* _), Lebesgue(zerosum(3)))"),
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    // The support is the inferred value set: through `iid` and a binding too.
    let nested = |m: &str| {
        format!(
            "y = elementof(cartpow(cartpow(reals, 3), 2))\nM = {m}\n\
             lp = logdensityof(M, y)\ninputs = (y)\noutputs = (lp)\n"
        )
    };
    for (source, never) in [
        (
            nested("pushfwd(fn(broadcast(fn(exp.(_)), _)), iid(Dirichlet([2.0, 3.0, 4.0]), 2))"),
            true,
        ),
        (
            nested("pushfwd(fn(2.0 .* _), iid(Dirichlet([2.0, 3.0, 4.0]), 2))"),
            false,
        ),
        (
            target("pushfwd(fn(exp.(_)), D)").replace("y =", "D = Dirichlet([2.0, 3.0, 4.0])\ny ="),
            true,
        ),
    ] {
        let r = refuse_with(&source, Some("lp"));
        if never {
            assert_eq!(r.kind, RefusalKind::NonBijectivePushfwd, "{source}: {r}");
        } else {
            assert_eq!(r.kind, RefusalKind::NotYetSupported, "{source}: {r}");
        }
    }
}

#[test]
fn a_map_of_a_mapped_simplex_law_keeps_the_determinizer_class() {
    // A bijection keeps the dimension of its support, and the uniform affine set
    // is the determinizer's (`flatppl_infer::uniform_affine_steps`).
    let target = |y: &str, m: &str| {
        format!(
            "D = Dirichlet([2.0, 3.0, 4.0])\ny = elementof({y})\nM = {m}\n\
             lp = logdensityof(M, y)\ninputs = (y)\noutputs = (lp)\n"
        )
    };
    let v3 = "cartpow(reals, 3)";
    for (m, kind) in [
        (
            "pushfwd(fn(3.0 .* _), pushfwd(fn(2.0 .* _), D))",
            RefusalKind::NotYetSupported,
        ),
        ("pushfwd(fn(_ ./ 2.0), D)", RefusalKind::NotYetSupported),
        (
            "pushfwd(fn(exp.(_)), pushfwd(fn(2.0 .* _), D))",
            RefusalKind::NonBijectivePushfwd,
        ),
        (
            "pushfwd(fn(exp.(_)), pushfwd(neg, D))",
            RefusalKind::NonBijectivePushfwd,
        ),
    ] {
        let r = refuse_with(&target(v3, m), Some("lp"));
        assert_eq!(r.kind, kind, "{m}: {r}");
    }
    let dims =
        "superpose(weighted(0.5, iid(Normal(0.0, 1.0), 2)), weighted(0.5, Dirichlet([1.0, 1.0])))";
    let r = refuse_with(&target("cartpow(reals, 2)", dims), Some("lp"));
    assert_eq!(r.kind, RefusalKind::MixedReference, "{r}");
    assert!(r.reason.contains("different dimension"), "{r}");
}

#[test]
fn a_bare_array_name_over_an_array_law_refuses_as_its_written_map() {
    // `neg` takes scalars or arrays and `inv` a matrix: over an array law each
    // bare name takes the path of `fn(name(_))`, with its refusal.
    for (map, written, law, point) in [
        (
            "neg",
            "fn(neg(_))",
            "iid(Normal(mu, 1.0), 3)",
            "[-1.2, -0.5, -2.0]",
        ),
        (
            "inv",
            "fn(inv(_))",
            "Wishart(4.0, rowstack([[1.0, 0.2], [0.2, 1.0]]))",
            "rowstack([[1.0, 0.1], [0.1, 1.0]])",
        ),
    ] {
        let source = |m: &str| {
            posterior(
                &format!("mu ~ Normal(0, 1)\nv ~ pushfwd({m}, {law})\ny ~ Normal(mu, 1)"),
                "lawof(record(mu = mu))",
                "kernelof(record(v = v, y = y), mu = mu)",
            )
            .replace("record(y = 0.5)", &format!("record(v = {point}, y = 0.5)"))
        };
        let (bare, written) = (refuse(&source(map)), refuse(&source(written)));
        assert_eq!(
            (&bare.kind, &bare.reason),
            (&written.kind, &written.reason),
            "{map}: {bare}"
        );
    }
}

#[test]
fn a_bare_name_over_a_discrete_base_refuses_as_its_written_map() {
    let source = |exp: &str, neg: &str| {
        posterior(
            &format!("mu ~ Normal(0, 1)\ny ~ pushfwd({exp}, pushfwd({neg}, Poisson(3.0)))"),
            "lawof(record(mu = mu))",
            "kernelof(record(y = y), mu = mu)",
        )
    };
    let bare = refuse(&source("exp", "neg"));
    let written = refuse(&source("fn(exp(_))", "fn(neg(_))"));
    assert_eq!(
        (&bare.kind, &bare.reason),
        (&written.kind, &written.reason),
        "{bare}"
    );
}

#[test]
fn pushfwd_through_the_input_over_itself_is_not_yet() {
    let r = refuse(&posterior(
        "f(t) = t / (1.0 + t)\nrho ~ pushfwd(f, Gamma(2, 1))\ny ~ Normal(rho, 1)",
        "lawof(record(rho = rho))",
        "kernelof(record(y = y), rho = rho)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("reads its input more than once"), "{r}");
}

#[test]
fn pushfwd_of_a_discrete_base_refuses() {
    // An atom at the boundary of `log`'s domain (Poisson at 0) never reaches
    // the domain check's null-set reading.
    let r = refuse(&posterior(
        "v ~ pushfwd(fn(log(_)), Poisson(2.0))\ny ~ Normal(v, 1)",
        "lawof(record(v = v))",
        "kernelof(record(y = y), v = v)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("pushfwd of this law"), "{r}");
}

#[test]
fn pushfwd_of_a_mixture_base_with_an_atom_refuses() {
    let r = refuse(&posterior(
        "v ~ pushfwd(fn(log(_)), normalize(superpose(weighted(0.3, Dirac(0.0)), \
         weighted(0.7, Exponential(1.0)))))\ny ~ Normal(v, 1)",
        "lawof(record(v = v))",
        "kernelof(record(y = y), v = v)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("normalize of this measure"), "{r}");
}

#[test]
fn a_derived_prior_field_of_two_draws_is_not_yet() {
    let r = refuse(&posterior(
        "a ~ Exponential(1)\nb ~ Exponential(2)\ns = sqrt(a + b)\ny ~ Normal(0, s)",
        "lawof(record(s = s))",
        "kernelof(record(y = y), s = s)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a derived prior field of more than one draw"),
        "{r}"
    );
}

#[test]
fn a_derived_prior_field_of_a_prior_field_is_not_yet() {
    let r = refuse(&posterior(
        "v ~ Exponential(1)\ns = sqrt(v)\ny ~ Normal(0, s)",
        "lawof(record(s = s, v = v))",
        "kernelof(record(y = y), s = s, v = v)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a derived prior field of `v`, which is itself a prior field"),
        "{r}"
    );
}

#[test]
fn two_derived_prior_fields_of_one_draw_are_refused() {
    let r = refuse(&posterior(
        "x ~ Exponential(1)\na = sqrt(x)\nb = exp(x)\ny ~ Normal(a, b)",
        "lawof(record(a = a, b = b))",
        "kernelof(record(y = y), a = a, b = b)",
    ));
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("two derived prior fields of the one draw `x`"),
        "{r}"
    );
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
    assert!(
        r.reason
            .contains("the law `CrystalBall` broadcast over `m0`"),
        "{r}"
    );
}

/// A GLM custom row has no overload for an argument above its declared rank.
#[test]
fn custom_glm_broadcast_is_not_yet() {
    let r = refuse_with(
        "reg = standard_module(\"regression\", \"0.1\")\n\
         x = rowstack([[1.0, 0.5], [0.2, -0.3], [0.7, 1.1]])\n\
         a ~ iid(Normal(0, 1), 2)\nbeta ~ iid(Normal(0, 1), 2)\n\
         y ~ reg.GammaLogGLM.([x, x], a, [beta, beta], 2.0)\n\
         L = likelihoodof(kernelof(record(y = y), a = a, beta = beta), \
         record(y = [[0.5, 1.3, 2.2], [0.5, 1.3, 2.2]]))\n\
         post = bayesupdate(L, lawof(record(a = a, beta = beta)))\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("the law `GammaLogGLM` broadcast over `x`"),
        "{r}"
    );
}

/// An iid of a GLM would flatten the response matrix into one GLM call.
#[test]
fn iid_of_a_glm_is_not_yet() {
    let r = refuse_with(
        "reg = standard_module(\"regression\", \"0.1\")\n\
         x = rowstack([[1.0, 0.5], [0.2, -0.3], [0.7, 1.1]])\n\
         beta ~ iid(Normal(0, 1), 2)\n\
         y ~ iid(reg.NormalGLM(x, 0.0, beta, 1.0), 2)\n\
         L = likelihoodof(kernelof(record(y = y), beta = beta), \
         record(y = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]]))\n\
         post = bayesupdate(L, lawof(record(beta = beta)))\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("an iid of the regression law `NormalGLM`"),
        "{r}"
    );
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
fn a_splat_of_a_loaded_table_is_not_yet_supported() {
    let r = refuse_with(
        "m0 ~ Normal(0.0, 1.0)\nT = load_data(\"t.csv\", anything)\ny ~ Normal.(T)\n\
         prior = lawof(record(m0 = m0))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), m0 = m0), record(y = [0.1, 0.9])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a splat of an argument that is not a record or a `table(…)` literal"),
        "{r}"
    );
}

#[test]
fn a_negative_literal_mixture_weight_is_ill_formed() {
    // §06 `weighted` takes a non-negative weight; `log(-0.5)` would be NaN.
    let r = refuse_with(
        "mu ~ Normal(0.0, 1.0)\n\
         y ~ normalize(superpose(weighted(-0.5, Normal(mu, 1.0)), weighted(1.5, Normal(2.0, 1.0))))\n\
         prior = lawof(record(mu = mu))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), mu = mu), record(y = 0.5)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(r.reason.contains("non-negative weight"), "{r}");
}

#[test]
fn a_chain_step_whose_argument_does_not_lift_is_not_yet_supported() {
    // The state inside `ifelse` has no elementwise form over the lag.
    let r = refuse_with(
        "x_data = [0.4, 0.9, 1.1]\nd ~ Normal(0.0, 1.0)\n\
         step(prev, d) = Normal(ifelse(prev > 0.5, d, 0.0), 1.0)\n\
         x ~ markovchain(step, 0.0, 3, d)\nprior = lawof(record(d = d))\n\
         post = bayesupdate(likelihoodof(kernelof(record(x = x), d = d), record(x = x_data)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a chain step"), "{r}");
}

#[test]
fn a_chain_step_whose_support_reads_the_state_is_not_yet_supported() {
    let r = refuse_with(
        "d ~ Normal(0.0, 1.0)\nstep(prev, d) = Uniform(interval(prev, prev + 1.0))\n\
         x ~ markovchain(step, 0.0, 3, d)\ny ~ Normal.(x, 0.5)\n\
         prior = lawof(record(d = d, x = x))\n\
         post = bayesupdate(likelihoodof(kernelof(record(y = y), d = d, x = x), record(y = [0.3, 0.9, 1.4])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a chain step whose support depends on the state"),
        "{r}"
    );
}

#[test]
fn a_chain_step_whose_support_reads_the_state_refuses_on_data_too() {
    let r = refuse_with(
        "d ~ Normal(0.0, 1.0)\nstep(prev, d) = Uniform(interval(prev, prev + 1.0 + d * d))\n\
         x ~ markovchain(step, 0.0, 3, d)\n\
         prior = lawof(record(d = d))\n\
         post = bayesupdate(likelihoodof(kernelof(record(x = x), d = d), record(x = [0.3, 0.9, 1.4])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a chain step whose support depends on the state"),
        "{r}"
    );
}

#[test]
fn a_chain_step_whose_integer_argument_reads_the_state_is_not_yet_supported() {
    let r = refuse_with(
        "b ~ Normal(0.0, 1.0)\nstep(prev, b) = Binomial(prev + 2, invlogit(b))\n\
         k ~ markovchain(step, 3, 4, b)\n\
         prior = lawof(record(b = b))\n\
         post = bayesupdate(likelihoodof(kernelof(record(k = k), b = b), record(k = [2, 3, 1, 2])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "a chain step whose integer argument is computed from the state or the per-step input"
        ),
        "{r}"
    );
}

#[test]
fn a_kscan_step_whose_integer_argument_is_computed_from_the_input_is_not_yet_supported() {
    let r = refuse_with(
        "ns = [5, 8, 10, 6]\nb ~ Normal(0.0, 1.0)\nf(prev, n, b) = Binomial(n + 1, invlogit(b))\n\
         k ~ kscan(f, 1, ns, b)\n\
         prior = lawof(record(b = b))\n\
         post = bayesupdate(likelihoodof(kernelof(record(k = k), b = b), record(k = [2, 3, 6, 4])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "a chain step whose integer argument is computed from the state or the per-step input"
        ),
        "{r}"
    );
}

#[test]
fn a_chain_step_of_a_multivariate_law_names_the_law() {
    let r = refuse_with(
        "d ~ Normal(0.0, 1.0)\n\
         step(prev, d) = MvNormal([prev, d], rowstack([[1.0, 0.0], [0.0, 1.0]]))\n\
         x ~ markovchain(step, 0.0, 4, d)\n\
         prior = lawof(record(d = d))\n\
         post = bayesupdate(likelihoodof(kernelof(record(x = x), d = d), record(x = [0.4, 0.9, 1.1, 1.8])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a chain step of the multivariate law `MvNormal`"),
        "{r}"
    );
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

#[test]
fn a_disintegrate_along_fields_others_depend_on_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\nJ = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\nK, prior = disintegrate([\"theta1\", \"theta2\"], J)\npost = bayesupdate(likelihoodof(K, record(theta1 = 0.5, theta2 = 1.0)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a `disintegrate` along fields that the field `obs` depends on (the marginal integrates them out)"), "{r}");
}

#[test]
fn a_disintegrate_refusal_is_located_in_the_name_the_user_wrote() {
    let (r, located) = refuse_located(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\nJ = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\nK, prior = disintegrate([\"theta1\", \"theta2\"], J)\npost = bayesupdate(likelihoodof(K, record(theta1 = 0.5, theta2 = 1.0)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert_eq!(located.as_deref(), Some("prior"), "{r}");
}

#[test]
fn a_disintegrate_with_a_latent_draw_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\na ~ Normal(theta1, 1)\nobs ~ iid(Normal(a, 1.0), 3)\nJ = lawof(record(theta1 = theta1, obs = obs))\nK, prior = disintegrate([\"obs\"], J)\npost = bayesupdate(likelihoodof(K, record(obs = [0.1, 0.2, 0.3])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "a `disintegrate` of a joint with the latent draw `a` (the marginal integrates it out)"
        ),
        "{r}"
    );
}

#[test]
fn a_disintegrate_of_a_jointchain_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nprior = lawof(record(theta1 = theta1, theta2 = theta2))\nobs_dist = joint(obs = iid(Normal(theta1, theta2), 3))\nk = functionof(obs_dist, theta1 = theta1, theta2 = theta2)\nJ = jointchain(prior, k)\nK, prior2 = disintegrate([\"obs\"], J)\npost = bayesupdate(likelihoodof(K, record(obs = [0.1, 0.2, 0.3])), prior2)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("a `disintegrate` of a `jointchain`"),
        "{r}"
    );
}

#[test]
fn a_bare_string_disintegrate_selector_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\nJ = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\nK, prior = disintegrate(\"obs\", J)\npost = bayesupdate(likelihoodof(K, record(obs = [0.1, 0.2, 0.3])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a bare-string `disintegrate` selector (a bare kernel variate)"),
        "{r}"
    );
}

#[test]
fn a_positional_disintegrate_selector_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\nJ = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\nK, prior = disintegrate([3], J)\npost = bayesupdate(likelihoodof(K, record(obs = [0.1, 0.2, 0.3])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains("a positional `disintegrate` selector"),
        "{r}"
    );
}

#[test]
fn a_restrict_that_pins_fields_others_depend_on_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\nJ = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\npost = restrict(J, record(theta1 = 0.5, theta2 = 1.0))\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "a `restrict` to fields that the field `obs` depends on (the complement form)"
        ),
        "{r}"
    );
}

#[test]
fn a_restrict_to_a_table_point_is_not_yet() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\nJ = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\npost = restrict(J, table(obs = [0.1, 0.2, 0.3]))\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a `restrict` to a table point"), "{r}");
}

#[test]
fn a_restrict_of_the_builtin_joint_names_restrict() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\nobs ~ iid(Normal(theta1, theta2), 3)\npost = restrict(joint, record(theta1 = 0.5, theta2 = 1.0))\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a `restrict` of a measure that is not `lawof(record(…))`"),
        "{r}"
    );
    assert!(!r.reason.contains("disintegrate"), "{r}");
}

#[test]
fn a_disintegrate_of_two_fields_of_one_draw_is_never() {
    let r = refuse_with(
        "theta1 ~ Normal(0, 1)\nobs ~ iid(Normal(theta1, 1.0), 3)\nJ = lawof(record(theta1 = theta1, obs = obs, copy = obs))\nK, prior = disintegrate([\"obs\"], J)\npost = bayesupdate(likelihoodof(K, record(obs = [0.1, 0.2, 0.3])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::SingularJoint, "{r}");
    assert!(
        r.reason.contains("holds the draw `obs` in two fields"),
        "{r}"
    );
}

#[test]
fn a_bayesupdate_and_an_observed_leaf_restrict_are_ambiguous() {
    let r = refuse(
        "theta ~ Normal(0, 1)\nobs ~ iid(Normal(theta, 1.0), 2)\nJ = lawof(record(theta = theta, obs = obs))\n\
         prior = lawof(record(theta = theta))\n\
         L = likelihoodof(kernelof(record(obs = obs), theta = theta), record(obs = [0.3, -0.2]))\n\
         a = bayesupdate(L, prior)\nb = restrict(J, record(obs = [0.3, -0.2]))\n",
    );
    assert_eq!(r.kind, RefusalKind::AmbiguousTarget, "{r}");
    assert!(r.reason.contains("a, b"), "{r}");
}

#[test]
fn restricts_that_pin_parameters_are_no_default_target() {
    // A lone restrict that does not split gives its own refusal.
    let model = "theta ~ Normal(0, 1)\nobs ~ iid(Normal(theta, 1.0), 2)\nJ = lawof(record(theta = theta, obs = obs))\n\
                 pred = restrict(J, record(theta = 0.5))\n";
    let (r, located) = refuse_located(model, None);
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason.contains(
            "a `restrict` to fields that the field `obs` depends on (the complement form)"
        ),
        "{r}"
    );
    assert_eq!(located.as_deref(), Some("pred"), "{r}");
    // Two of them: no single one is meant.
    let r = refuse(&format!(
        "{model}pred2 = restrict(J, record(theta = 0.7))\n"
    ));
    assert_eq!(r.kind, RefusalKind::NoTarget, "{r}");
    assert!(
        r.reason
            .contains("no `bayesupdate` and no `restrict` to observed fields"),
        "{r}"
    );
}

#[test]
fn a_lone_restrict_of_two_fields_of_one_draw_is_its_own_refusal() {
    let (r, located) = refuse_located(
        "a ~ Normal(0, 1)\nb ~ Normal(a, 1)\nJ = lawof(record(a = a, b = b, c = b))\n\
         post = restrict(J, record(c = 1.5))\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::SingularJoint, "{r}");
    assert!(r.reason.contains("holds the draw `b` in two fields"), "{r}");
    assert_eq!(located.as_deref(), Some("post"), "{r}");
}

#[test]
fn a_lone_restrict_to_a_field_the_joint_lacks_is_its_own_refusal() {
    let (r, located) = refuse_located(
        "a ~ Normal(0, 1)\nb ~ Normal(a, 1)\nJ = lawof(record(a = a, b = b))\n\
         post = restrict(J, record(b = 0.9, z = 2.0))\n",
        None,
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a `restrict` point field `z` that is not a field of the joint"),
        "{r}"
    );
    assert_eq!(located.as_deref(), Some("post"), "{r}");
}

#[test]
fn a_restrict_point_of_another_shape_is_ill_formed() {
    let scalar = refuse_with(
        "a ~ Normal(0, 1)\nb ~ Normal(a, 1)\nJ = lawof(record(a = a, b = b))\n\
         post = restrict(J, record(b = [0.9, 1.1]))\n",
        Some("post"),
    );
    assert_eq!(scalar.kind, RefusalKind::IllFormedModule, "{scalar}");
    assert!(
        scalar.reason.contains(
            "the `restrict` point sets `b` to array of real [2], but the field `b` of the joint has variates of type real"
        ),
        "{scalar}"
    );
    let length = refuse_with(
        "a ~ Normal(0, 1)\nb ~ iid(Normal(a, 1), 3)\nJ = lawof(record(a = a, b = b))\n\
         post = restrict(J, record(b = [0.9, 1.1]))\n",
        Some("post"),
    );
    assert_eq!(length.kind, RefusalKind::IllFormedModule, "{length}");
    assert!(
        length.reason.contains(
            "the `restrict` point sets `b` to array of real [2], but the field `b` of the joint has variates of type array of real [3]"
        ),
        "{length}"
    );
}

#[test]
fn observed_data_that_reads_a_parameter_is_not_yet() {
    let head = "a ~ Normal(0, 1)\ny ~ Normal(a, 1)\n";
    let forms = [
        "prior = lawof(record(a = a))\nK = kernelof(record(y = y), a = a)\n\
         post = bayesupdate(likelihoodof(K, record(y = 2 * a)), prior)\n",
        "J = lawof(record(a = a, y = y))\nK, prior = disintegrate([\"y\"], J)\n\
         post = bayesupdate(likelihoodof(K, record(y = 2 * a)), prior)\n",
        "J = lawof(record(a = a, y = y))\npost = restrict(J, record(y = 2 * a))\n",
    ];
    for form in forms {
        let r = refuse_with(&format!("{head}{form}"), Some("post"));
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{form}: {r}");
        assert!(
            r.reason
                .contains("observed data `y` that reads `a`, a parameter of the target"),
            "{form}: {r}"
        );
    }
}

#[test]
fn a_disintegrate_selector_that_names_a_field_twice_is_ill_formed() {
    let r = refuse_with(
        "a ~ Normal(0, 1)\ny ~ Normal(a, 1)\nJ = lawof(record(a = a, y = y))\n\
         K, prior = disintegrate([\"y\", \"y\"], J)\n\
         post = bayesupdate(likelihoodof(K, record(y = 0.5)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason
            .contains("the `disintegrate` selector names the field `y` twice"),
        "{r}"
    );
}

#[test]
fn observed_data_sampled_from_a_closed_law_reads_no_parameter() {
    // `lawof(obs)` absorbs the draw `theta1` (§04), so the sample reads no
    // parameter. The refusal is `rand`'s, in the binding the user wrote,
    // not the tuple's parser-made one.
    let (r, located) = refuse_located(
        "theta1 = draw(Normal(mu = 0, sigma = 1))\na = 5.0 * theta1\n\
         obs = draw(iid(Normal(mu = a, sigma = 1), 4))\nrstate = rnginit([1, 2, 3, 4])\n\
         observed_data, _ = rand(lawof(obs), rstate)\n\
         forward_kernel = kernelof(record(obs = obs), theta1 = theta1)\n\
         prior = lawof(record(theta1 = theta1))\n\
         L = likelihoodof(forward_kernel, record(obs = observed_data))\n\
         posterior = bayesupdate(L, prior)\n",
        Some("posterior"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("`rand`"), "{r}");
    assert!(!r.reason.contains("observed data"), "{r}");
    assert_eq!(located.as_deref(), Some("observed_data"), "{r}");
}

#[test]
fn observed_data_that_reads_an_absorbed_draw_reads_no_parameter() {
    let r = refuse_with(
        "z ~ Normal(0, 1)\nM = lawof(z)\na ~ Normal(0, 1)\ny ~ Normal(a, 1)\n\
         J = lawof(record(a = a, y = y))\npost = restrict(J, record(y = mean(M)))\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("`M` used as a value"), "{r}");
    assert!(!r.reason.contains("observed data"), "{r}");
}

#[test]
fn a_field_law_that_reifies_a_draw_is_no_latent_draw() {
    // `y ~ M` with `M = lawof(zz)`: the law absorbs `zz` (§04), so the split
    // holds and the refusal is the law's, as in the explicit form.
    let split = refuse_with(
        "zz ~ Normal(0, 1)\nM = lawof(zz)\na ~ Normal(0, 1)\ny ~ M\n\
         J = lawof(record(a = a, y = y))\npost = restrict(J, record(y = 0.4))\n",
        Some("post"),
    );
    let explicit = refuse_with(
        "zz ~ Normal(0, 1)\nM = lawof(zz)\na ~ Normal(0, 1)\ny ~ M\n\
         prior = lawof(record(a = a))\nK = kernelof(record(y = y), a = a)\n\
         post = bayesupdate(likelihoodof(K, record(y = 0.4)), prior)\n",
        Some("post"),
    );
    assert!(!split.reason.contains("latent draw"), "{split}");
    assert_eq!(split.kind, explicit.kind, "{split} / {explicit}");
    assert_eq!(split.reason, explicit.reason, "{split} / {explicit}");
}

#[test]
fn a_field_law_from_a_kernel_reads_no_latent_draw() {
    // `kernelof(z, a = a)` absorbs the draw `z` as `lawof` does (§04), so the
    // split holds and the refusal is the law's, as in the explicit form.
    let head = "a ~ Normal(0, 1)\nz ~ Normal(a, 1)\nkk = kernelof(z, a = a)\ny ~ kk(a = a)\n";
    let split = refuse_with(
        &format!("{head}J = lawof(record(a = a, y = y))\npost = restrict(J, record(y = 0.4))\n"),
        Some("post"),
    );
    let explicit = refuse_with(
        &format!(
            "{head}prior = lawof(record(a = a))\nK = kernelof(record(y = y), a = a)\n\
             post = bayesupdate(likelihoodof(K, record(y = 0.4)), prior)\n"
        ),
        Some("post"),
    );
    assert!(!split.reason.contains("latent draw"), "{split}");
    assert_eq!(split.kind, explicit.kind, "{split} / {explicit}");
    assert_eq!(split.reason, explicit.reason, "{split} / {explicit}");
}

#[test]
fn a_restrict_point_field_that_is_a_table_is_not_yet() {
    let r = refuse_with(
        "a ~ Normal(0, 1)\nt ~ iid(joint(u = Normal(a, 1)), 3)\nJ = lawof(record(a = a, t = t))\n\
         post = restrict(J, record(t = table(u = [0.1, 0.2, 0.3])))\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a `restrict` point field `t` that is a table"),
        "{r}"
    );
}

/// The explicit, `disintegrate` and `restrict` forms of one observation of `b`
/// with data `obs`, after the lines in `head`.
fn observation_forms(head: &str, obs: &str) -> [String; 3] {
    [
        format!(
            "{head}prior = lawof(record(a = a))\nK = kernelof(record(b = b), a = a)\n\
             post = bayesupdate(likelihoodof(K, record(b = {obs})), prior)\n"
        ),
        format!(
            "{head}J = lawof(record(a = a, b = b))\nK, prior = disintegrate([\"b\"], J)\n\
             post = bayesupdate(likelihoodof(K, record(b = {obs})), prior)\n"
        ),
        format!("{head}J = lawof(record(a = a, b = b))\npost = restrict(J, record(b = {obs}))\n"),
    ]
}

/// The module the module-member tests load as `mod.flatppl`.
const MODULE_LEN: &str = "flatppl_compat = \"0.1\"\nlen = 3\n";

/// The Stan text of `source`, with `dependency` loaded as `mod.flatppl`.
fn emit_with_module(source: &str, dependency: &str, target: Option<&str>) -> String {
    let bundle = module_bundle(dependency);
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics =
        flatppl_infer::infer_module(&mut module, &bundle, flatppl_infer::Level::Shape);
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
    flatppl_stan::emit_with_bundle(&module, &bundle, &opts)
        .unwrap_or_else(|r| panic!("{source}: {r}"))
        .code
}

#[test]
fn an_iid_count_from_a_module_member_sizes_the_data() {
    // The member is imported, so the count is the program's own integer.
    let head = "m = load_module(\"mod.flatppl\")\na ~ Normal(0, 1)\nb ~ iid(Normal(a, 1), m.len)\n";
    for form in observation_forms(head, "[0.9, 1.1, 0.3]") {
        let code = emit_with_module(&form, MODULE_LEN, Some("post"));
        assert!(code.contains("vector[len] b;"), "{form}:\n{code}");
    }
    // Observed data of another length than the imported count is a static
    // mismatch, which inference sees once the member is imported.
    for form in observation_forms(head, "[0.9, 1.1]") {
        let (r, _) = refuse_with_module(&form, MODULE_LEN, Some("post"));
        assert_eq!(r.kind, RefusalKind::IllFormedModule, "{form}: {r}");
        assert!(r.reason.contains("array of real [2]"), "{form}: {r}");
    }
}

#[test]
fn an_iid_count_from_a_clashing_module_member_keeps_its_bound() {
    // The host's own `len` renames the import; the bound comes from the
    // member's name in its module, and the import prints as `m_len`.
    let head = "m = load_module(\"mod.flatppl\")\nlen = 0.5\na ~ Normal(len, 1)\n\
                b ~ iid(Normal(a, 1), m.len)\n";
    for form in observation_forms(head, "[0.9, 1.1, 0.3]") {
        let code = emit_with_module(&form, MODULE_LEN, Some("post"));
        assert!(code.contains("int<lower=0> m_len;"), "{form}:\n{code}");
        assert!(code.contains("vector[m_len] b;"), "{form}:\n{code}");
        assert!(code.contains("real len;"), "{form}:\n{code}");
    }
}

#[test]
fn a_member_behind_an_unprinted_alias_keeps_its_name() {
    // `f_b = common.f_b` in the loaded module is an alias the program never
    // prints, so the function it names keeps `f_b`: no prefix and no suffix.
    let mut bundle = module_bundle(
        "flatppl_compat = \"0.1\"\ncommon = load_module(\"common.flatppl\", c = c_scaling)\n\
         c_scaling = 5\nf_b = common.f_b\ntheta1 ~ Normal(0, 1)\ntheta2 ~ Exponential(1)\n\
         a = common.f_a(theta2)\nb = f_b(theta1, theta2)\nobs ~ iid(Normal(mu = a, sigma = b), 3)\n\
         joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))\n\
         posterior = restrict(joint_model, record(obs = [1.2, 3.4, 5.1]))\n",
    );
    bundle.insert(
        "common.flatppl",
        std::sync::Arc::new(
            flatppl_syntax::parse(
                "flatppl_compat = \"0.1\"\nc = external(reals)\nf_a = par -> c * par\n\
                 f_b = fn(abs(_) * _)\n",
            )
            .expect("parse common"),
        ),
    );
    let mut module = flatppl_syntax::parse(
        "flatppl_compat = \"0.1\"\nm = load_module(\"mod.flatppl\")\n\
         lp = logdensityof(m.posterior, record(theta1 = 0.5, theta2 = 1.0))\n",
    )
    .expect("parse");
    let diagnostics =
        flatppl_infer::infer_module(&mut module, &bundle, flatppl_infer::Level::Shape);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let opts = EmitOptions {
        target: Some("lp".to_string()),
        ..Default::default()
    };
    let code = flatppl_stan::emit_with_bundle(&module, &bundle, &opts)
        .unwrap_or_else(|r| panic!("{r}"))
        .code;
    assert!(code.contains("real f_a(real par, int c_scaling)"), "{code}");
    assert!(code.contains("real f_b(real arg1, real arg2)"), "{code}");
    assert!(code.contains("real b = f_b(theta1, theta2);"), "{code}");
    assert!(!code.contains("m_f_b"), "{code}");
}

#[test]
fn a_member_from_several_instances_prints_with_each_alias() {
    // One member name from three instances: every one is prefixed, so the
    // text does not depend on which instance was imported first.
    let code = emit_with_module(
        "flatppl_compat = \"0.1\"\nm1 = load_module(\"mod.flatppl\", c = 1.0)\nm2 = load_module(\"mod.flatppl\", c = 2.0)\n\
         m3 = load_module(\"mod.flatppl\", c = 3.0)\nx = elementof(reals)\n\
         lp = logdensityof(m1.M, x) + logdensityof(m2.M, x) + logdensityof(m3.M, x)\n\
         inputs = x\noutputs = lp\n",
        "flatppl_compat = \"0.1\"\nc = external(reals)\nM = Normal(c, 1.0)\n",
        Some("lp"),
    );
    for k in 1..=3 {
        assert!(code.contains(&format!("real m{k}_c;")), "{code}");
        assert!(
            code.contains(&format!("x ~ normal(m{k}_c, 1.0);")),
            "{code}"
        );
    }
    assert!(!code.contains("real c;"), "{code}");
}

#[test]
fn an_iid_count_that_is_the_length_of_a_module_member_emits() {
    // `lengthof(m.v)` has the static extent inference gives the member: the
    // data length is declared from it.
    let bundle = module_bundle("flatppl_compat = \"0.1\"\nv = [1.0, 2.0, 3.0]\n");
    let mut module = flatppl_syntax::parse(
        "flatppl_compat = \"0.1\"\nm = load_module(\"mod.flatppl\")\na ~ Normal(0, 1)\n\
         b ~ iid(Normal(a, 1), lengthof(m.v))\nprior = lawof(record(a = a))\n\
         K = kernelof(record(b = b), a = a)\n\
         post = bayesupdate(likelihoodof(K, record(b = [0.9, 1.1, 0.3])), prior)\n",
    )
    .expect("parse");
    let diagnostics =
        flatppl_infer::infer_module(&mut module, &bundle, flatppl_infer::Level::Shape);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let opts = EmitOptions {
        target: Some("post".to_string()),
        ..Default::default()
    };
    let program = flatppl_stan::emit(&module, &opts).expect("emits");
    assert!(
        program.code.contains("data {\n  vector[3] b;\n}"),
        "{}",
        program.code
    );
    assert!(!program.code.contains("reject"), "{}", program.code);
}

#[test]
fn lengthof_a_matrix_is_ill_formed() {
    // §07 defines `lengthof` for vectors and tables: a matrix's size is
    // `sizeof`, as a value and as an `iid` count.
    for (body, target) in [
        (
            "mu = elementof(reals)\nlp = logdensityof(Normal(lengthof(M) + mu, 1.0), 0.4)\n\
             inputs = (mu)\noutputs = (lp)\n",
            "lp",
        ),
        (
            "a ~ Normal(0, 1)\nb ~ iid(Normal(a, 1), lengthof(M))\nprior = lawof(record(a = a))\n\
             post = bayesupdate(likelihoodof(kernelof(record(b = b), a = a), record(b = [0.9, 1.1])), prior)\n",
            "post",
        ),
    ] {
        let source = format!("M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n{body}");
        let r = refuse_with(&source, Some(target));
        assert_eq!(r.kind, RefusalKind::IllFormedModule, "{source}: {r}");
        assert!(
            r.reason
                .contains("§07 defines `lengthof` for vectors and tables, not for a matrix"),
            "{source}: {r}"
        );
    }
}

#[test]
fn an_index_past_sizeof_of_a_vector_of_vectors_is_ill_formed() {
    // §03: a vector of vectors is not a matrix, so its `sizeof` has one
    // entry per outer axis; a literal index past them is known here.
    for (source, target) in [
        (
            "q = [0.1, 0.5, 0.9]\nmu = elementof(reals)\n\
             lp = logdensityof(Normal(sizeof(fill(q, 2))[2] + mu, 1.0), 0.4)\n\
             inputs = (mu)\noutputs = (lp)\n",
            "lp",
        ),
        (
            "eta ~ Exponential(1)\n\
             Ro = [rowstack([[1.0, 0.2], [0.2, 1.0]]), rowstack([[1.0, -0.3], [-0.3, 1.0]])]\n\
             R ~ iid(LKJ(sizeof(Ro)[2], eta), 2)\nprior = lawof(record(eta = eta))\n\
             K = kernelof(record(R = R), eta = eta)\n\
             post = bayesupdate(likelihoodof(K, record(R = Ro)), prior)\n",
            "post",
        ),
    ] {
        let r = refuse_with(source, Some(target));
        assert_eq!(r.kind, RefusalKind::IllFormedModule, "{source}: {r}");
        assert!(
            r.reason
                .contains("index 2 into `sizeof` of a vector of vectors, which has 1 entry"),
            "{source}: {r}"
        );
    }
}

#[test]
fn an_index_past_sizeof_of_a_vector_is_ill_formed() {
    // §07 `sizeof` of a vector has one entry; Stan's `dims` of a `vector[n]`
    // has two, so the index must not reach it.
    let r = refuse_with(
        "X = [1.0, 2.0, 3.0]\nmu = elementof(reals)\n\
         lp = logdensityof(Normal(sizeof(X)[2] + mu, 1.0), 0.4)\ninputs = (mu)\noutputs = (lp)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason
            .contains("index 2 into `sizeof` of a vector, which has 1 entry"),
        "{r}"
    );
}

#[test]
fn an_index_past_sizeof_of_any_array_is_ill_formed() {
    // The rank is static for each kind, so a literal index past it refuses
    // as for a vector, where Stan's `dims` would throw at run time.
    for (value, index, entries) in [
        ("[1, 2, 3]", 2, "a vector, which has 1 entry"),
        (
            "rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])",
            3,
            "a rank-2 array, which has 2 entries",
        ),
        (
            "fill(0.5, [2, 3])",
            3,
            "a rank-2 array, which has 2 entries",
        ),
    ] {
        let source = format!(
            "X = {value}\nmu = elementof(reals)\n\
             lp = logdensityof(Normal(sizeof(X)[{index}] + mu, 1.0), 0.4)\n\
             inputs = (mu)\noutputs = (lp)\n"
        );
        let r = refuse_with(&source, Some("lp"));
        assert_eq!(r.kind, RefusalKind::IllFormedModule, "{source}: {r}");
        assert!(
            r.reason
                .contains(&format!("index {index} into `sizeof` of {entries}")),
            "{source}: {r}"
        );
    }
    // An input matrix.
    let r = refuse_with(
        "X = elementof(cartpow(reals, [2, 3]))\n\
         lp = logdensityof(Normal(sizeof(X)[3] + X[1, 2], 1.0), 0.4)\n\
         inputs = (X)\noutputs = (lp)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason
            .contains("index 3 into `sizeof` of a rank-2 array, which has 2 entries"),
        "{r}"
    );
}

#[test]
fn a_fill_of_a_row_vector_is_not_yet() {
    let r = refuse_with(
        "q = transpose([0.1, 0.5, 0.9])\nF = fill(q, 2)\nmu = elementof(reals)\n\
         lp = logdensityof(Normal(F[2][3] + mu, 1.0), 0.4)\ninputs = (mu)\noutputs = (lp)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a `fill` of a row vector"), "{r}");
}

#[test]
fn a_dirac_row_atom_of_vectors_is_not_yet() {
    // `row_fill` reads the size form `[1, T]`; the row test compares each
    // element with the atom value, so the value must be a scalar.
    let r = refuse_with(
        "M = 4\nT = 3\ny_obs = rowstack([[0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0]])\n\
         omega ~ Uniform(unitinterval)\nmean_p ~ iid(Uniform(unitinterval), T)\n\
         sigma ~ Uniform(interval(0, 5))\neps_raw ~ iid(Normal(0, 1), M)\n\
         lp = addaxes(sigma .* eps_raw, 0, 1) .+ addaxes(logit.(mean_p), 1, 0)\n\
         capture(row, omega) = normalize(superpose(\n    weighted(omega, Bernoulli.(invlogit.(row))), \
         weighted(1 - omega, Dirac(fill([false, false, false], [1, T])))))\n\
         y ~ capture.(splitblocks(lp, [1, T]), omega)\n\
         prior = lawof(record(omega = omega, mean_p = mean_p, sigma = sigma, eps_raw = eps_raw))\n\
         L = likelihoodof(kernelof(record(y = y), omega = omega, mean_p = mean_p, sigma = sigma, \
         eps_raw = eps_raw), record(y = splitblocks(y_obs, [1, T])))\npost = bayesupdate(L, prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a `Dirac` row atom whose value is not a scalar"),
        "{r}"
    );
}

#[test]
fn an_iid_count_from_an_untyped_load_is_not_yet() {
    let head = "n = load_data(\"n.json\", anything)\na ~ Normal(0, 1)\nb ~ iid(Normal(a, 1), n)\n";
    for form in observation_forms(head, "[0.9, 1.1]") {
        let r = refuse_with(&form, Some("post"));
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{form}: {r}");
        assert!(
            r.reason
                .contains("an `iid` count `n` that the Stan program cannot read as an integer"),
            "{form}: {r}"
        );
    }
}

#[test]
fn an_iid_shape_with_a_module_member_sizes_the_data() {
    let head =
        "m = load_module(\"mod.flatppl\")\na ~ Normal(0, 1)\nb ~ iid(Normal(a, 1), [2, m.len])\n";
    let [explicit, ..] = observation_forms(head, "rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])");
    let code = emit_with_module(&explicit, MODULE_LEN, Some("post"));
    assert!(code.contains("matrix[2, len] b;"), "{code}");
}

#[test]
fn a_refusal_inside_a_module_member_is_located_at_its_reader() {
    // The member is imported under its own name; the refusal names the host
    // binding that reads it, never a binding the user did not write here.
    let (r, located) = refuse_with_module(
        "m = load_module(\"mod.flatppl\")\nx = elementof(reals)\n\
         lp = logdensityof(m.W, x)\ninputs = x\noutputs = lp\n",
        "flatppl_compat = \"0.1\"\nW = Dirac(0.5)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("the law `Dirac` in this form"), "{r}");
    assert_eq!(located.as_deref(), Some("lp"), "{r}");
}

#[test]
fn an_applied_module_kernel_is_not_yet() {
    // The imported `functionof` law applied to an argument is the local form
    // the emitter does not write yet.
    let (r, located) = refuse_with_module(
        "m = load_module(\"mod.flatppl\")\nlp = logdensityof(m.obs_kernel(0.0), 0.5)\n",
        "flatppl_compat = \"0.1\"\ncenter = elementof(reals)\n\
         obs_kernel = functionof(Normal(mu = center, sigma = 1.0), center = center)\n",
        Some("lp"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a law from a user callable"), "{r}");
    assert_eq!(located.as_deref(), Some("lp"), "{r}");
}

#[test]
fn an_lkj_dimension_from_an_untyped_load_is_not_yet() {
    for law in ["LKJ", "LKJCholesky"] {
        let r = refuse_with(
            &format!(
                "dim = load_data(\"n.json\", anything)\neta ~ Exponential(1)\nR ~ {law}(dim, eta)\n\
                 prior = lawof(record(eta = eta))\nK = kernelof(record(R = R), eta = eta)\n\
                 post = bayesupdate(likelihoodof(K, record(R = rowstack([[1.0, 0.0], [0.0, 1.0]]))), prior)\n"
            ),
            Some("post"),
        );
        assert_eq!(r.kind, RefusalKind::NotYetSupported, "{law}: {r}");
        assert!(
            r.reason.contains(&format!(
                "an `{law}` dimension `dim` that the Stan program cannot read as an integer"
            )),
            "{law}: {r}"
        );
    }
}

#[test]
fn a_multinomial_count_the_program_cannot_read_is_not_yet() {
    // The guard `sum(y) != n` needs n: a count with no Stan form refuses by
    // name, never drops the guard.
    let r = refuse_with(
        "k = load_data(\"n.json\", anything)\np ~ Dirichlet([1.0, 1.0, 1.0])\n\
         c ~ Multinomial(k, p)\nprior = lawof(record(p = p))\n\
         K = kernelof(record(c = c), p = p)\n\
         post = bayesupdate(likelihoodof(K, record(c = [1, 0, 2])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("a `Multinomial` count `k` that the Stan program cannot read as an integer"),
        "{r}"
    );
}

#[test]
fn a_multinomial_count_from_a_module_member_is_guarded() {
    let code = emit_with_module(
        "m = load_module(\"mod.flatppl\")\np ~ Dirichlet([1.0, 1.0, 1.0])\n\
         c ~ Multinomial(m.len, p)\nprior = lawof(record(p = p))\n\
         K = kernelof(record(c = c), p = p)\n\
         post = bayesupdate(likelihoodof(K, record(c = [1, 0, 2])), prior)\n",
        MODULE_LEN,
        Some("post"),
    );
    assert!(code.contains("if (sum(c) != len)"), "{code}");
}

#[test]
fn an_iid_count_that_is_a_draw_is_the_latent_draw() {
    // The count `k` is a discrete draw the target integrates out; the
    // refusal names it, not the count.
    let (r, located) = refuse_located(
        "k ~ Poisson(3.0)\nmu ~ Normal(0, 1)\nb ~ iid(Normal(mu, 1.0), k)\n\
         prior = lawof(record(mu = mu))\nK = kernelof(record(b = b), mu = mu)\n\
         post = bayesupdate(likelihoodof(K, record(b = [0.9, 1.1, 0.3])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(
        r.reason
            .contains("`k` is a discrete draw the target marginalizes"),
        "{r}"
    );
    assert!(!r.reason.contains("cannot read as an integer"), "{r}");
    assert_eq!(located.as_deref(), Some("k"), "{r}");
}

#[test]
fn an_iid_count_that_is_the_observed_length_plus_one_is_ill_formed() {
    // The data's length is not known to inference here, so the emitter sees
    // the count `lengthof(yd) + 1` against yd itself: no data fits it.
    let (r, located) = refuse_located(
        "n = load_data(\"n.json\", posintegers)\nyd = load_data(\"y.json\", cartpow(reals, n))\n\
         a ~ Normal(0, 1)\nb ~ iid(Normal(a, 1), lengthof(yd) + 1)\nprior = lawof(record(a = a))\n\
         post = bayesupdate(likelihoodof(kernelof(record(b = b), a = a), record(b = yd)), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::IllFormedModule, "{r}");
    assert!(
        r.reason
            .contains("the law gives `yd` its own extent on axis 1 plus 1"),
        "{r}"
    );
    assert_eq!(located.as_deref(), Some("b"), "{r}");
}

#[test]
fn an_iid_of_an_iid_of_a_multinomial_is_not_yet() {
    let r = refuse_with(
        "p ~ Dirichlet([1.0, 2.0, 3.0])\nc ~ iid(iid(Multinomial(10, p), 2), 3)\n\
         prior = lawof(record(p = p))\nK = kernelof(record(c = c), p = p)\n\
         post = bayesupdate(likelihoodof(K, record(c = [[[2, 3, 5], [1, 4, 5]], [[2, 3, 5], [1, 4, 5]], \
         [[2, 3, 5], [1, 4, 5]]])), prior)\n",
        Some("post"),
    );
    assert_eq!(r.kind, RefusalKind::NotYetSupported, "{r}");
    assert!(r.reason.contains("a nested iid of this law"), "{r}");
}
