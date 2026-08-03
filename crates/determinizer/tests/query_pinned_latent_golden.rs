//! A latent one query pinned is still a latent for the NEXT query.
//!
//! Lowering `logdensityof(lawof(z), 0.3)` rewrites `z = draw(Normal(0, 1))` to
//! `z = 0.3` (§13 "Output reduction": "`draw` nodes take their values from the
//! explicit `point`, unless marginalized out"). After that rewrite nothing in the
//! binding tells the literal from a model constant, so a second query over
//! `y ~ Normal(mu = z, sigma = 1)` read `mu` as fixed and emitted the CONDITIONAL
//! `p(y | z = 0.3)` — a finished, conformance-passing number — where §04 "Reification
//! to measures" makes `lawof(y)` y's TOTAL law, the marginal `Normal(0, √2)`. Both
//! spellings had it (`lawof(y)` and `lawof(record(y = y))`), in both statement orders.
//!
//! `Module::pin_binding_to_query_point` records the pin, so the provenance outlives the
//! rewrite. What must NOT be caught: a genuinely fixed parameter, an independent latent
//! another query pinned, and a SIBLING field of the same record product — that dependence
//! is the chain rule, and §04's "not boundary inputs" exception keeps a `kernelof` body
//! conditional.
//!
//! The pin also records the `draw(prior)` it replaced (`Module::query_pinned_rhs`), which
//! is what `crate::marginal`'s conjugate table needs in the pinned ordering. So the
//! assertions are that both orderings reach the SAME marginal. This file guards the
//! provenance, not the row (`implicit_marginal_golden.rs`; maths in `src/marginal.md`).
//!
//! Structural only (flatppl-rust is not a density engine): every lowering is asserted
//! on its emitted FlatPDL and passed through `is_flatpdl`.
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::pir_binding;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

/// The FlatPIR of a module that must lower, conformance-checked on the way through.
fn pir(src: &str) -> String {
    let out = determinize_src(src);
    let text = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl failed:\n{text}");
    text
}

fn refusal(src: &str) -> String {
    determinize(&parse_infer(src))
        .expect_err("the conditional density must not escape as a finished number")
        .reason
}

// The bare spelling, in both statement orders. `lp_z` first is the order that needs
// the pin provenance: by the time `lp_y` lowers, `z` is the literal `0.3`, so neither the
// residual-`draw` scan nor `z`'s own binding has anything left to say — the provenance is
// the only thing that still knows `z` was a latent AND what its prior was. `lp_y` first
// reaches the same marginal through the still-present `draw`, and both are asserted so a
// future change cannot fix one order by breaking the other.
//
// `Normal(0, √2)` (sigma folds to 1.4142135623730951) is the marginal; the conditional
// `Normal(0.3, 1)` at the pinned latent is what must NOT appear.
#[test]
fn a_pinned_latent_is_not_a_fixed_parameter_bare_spelling() {
    let mut emitted = Vec::new();
    for src in [
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_z = logdensityof(lawof(z), 0.3)
lp_y = logdensityof(lawof(y), 0.5)",
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)
lp_z = logdensityof(lawof(z), 0.3)",
    ] {
        let lp_y = pir_binding(&pir(src), "lp_y");
        assert!(
            lp_y.contains("(%field sigma 1.4142135623730951)"),
            "the latent is marginalized out, not read as a fixed parameter:\n{lp_y}"
        );
        assert!(
            !lp_y.contains("(%field mu 0.3)"),
            "not the conditional at the pinned latent:\n{lp_y}"
        );
        emitted.push(lp_y);
    }
    assert_eq!(
        emitted[0], emitted[1],
        "both statement orders must reach the same marginal"
    );
}

// The record spelling — the more common way to write the query, and wrong in BOTH
// orders before this: with `lp_y` first, the field's measure still carried
// `(%ref self z)` and `build_density_term` scored it, then `lp_z` pinned `z` and the
// emitted term silently became the conditional.
#[test]
fn a_pinned_latent_is_not_a_fixed_parameter_record_spelling() {
    let mut emitted = Vec::new();
    for src in [
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_z = logdensityof(lawof(record(z = z)), record(z = 0.3))
lp_y = logdensityof(lawof(record(y = y)), record(y = 0.5))",
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(record(y = y)), record(y = 0.5))
lp_z = logdensityof(lawof(record(z = z)), record(z = 0.3))",
    ] {
        let lp_y = pir_binding(&pir(src), "lp_y");
        assert!(
            lp_y.contains("(%field sigma 1.4142135623730951)"),
            "the ancestor outside the product is marginalized out:\n{lp_y}"
        );
        assert!(
            !lp_y.contains("(%field mu 0.3)"),
            "not the conditional at the latent another query pinned:\n{lp_y}"
        );
        emitted.push(lp_y);
    }
    assert_eq!(
        emitted[0], emitted[1],
        "both statement orders must reach the same marginal"
    );
}

// A field's MAP, not only its measure, can carry the outside ancestor: `b = y + z` is
// read as a pushforward of y's law, and `build_forward_map` would take the pinned `z`
// for a constant of the map.
#[test]
fn a_transformed_field_is_checked_through_its_map() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = 0.0, sigma = 1.0))
b = y + z
lp_z = logdensityof(lawof(z), 0.3)
lp_b = logdensityof(lawof(record(b = b)), record(b = 0.5))";
    let reason = refusal(src);
    assert!(
        reason.contains("uncarried draw"),
        "must refuse as a marginal: {reason}"
    );
}

// `rand` rewrites a latent's binding the same way a density query does, and the
// realization it writes is even further from a model constant than a query point is —
// scoring `y` against it would condition on one draw of `z`, where §04 asks for y's law
// over all of them.
#[test]
fn a_sampled_latent_is_not_a_fixed_parameter() {
    let src = "\
s = rngstate(7)
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
zs, s2 = rand(s, lawof(record(z = z)))
lp_y = logdensityof(lawof(y), 0.5)";
    let text = pir(src);
    let lp_y = pir_binding(&text, "lp_y");
    assert_eq!(
        lp_y.matches("builtin_logdensityof").count(),
        1,
        "the marginal is one density term:\n{lp_y}"
    );
    assert!(
        lp_y.contains("(%field mu 0.0)") && lp_y.contains("(%field sigma 1.4142135623730951)"),
        "the sampled latent is marginalized out — prior mu, sqrt(2) sigma:\n{lp_y}"
    );
    // The discriminating half: y's law must carry NEITHER the realization (which the
    // conditional would have as its `mu`) nor the likelihood's own sigma. `builtin_sample`
    // alone is weak — the sample is bound elsewhere and could be reached by a ref.
    assert!(
        !lp_y.contains("(%field sigma 1.0)")
            && !lp_y.contains("builtin_sample")
            && !lp_y.contains("(%ref self "),
        "the marginal does not condition on the drawn realization, by value or by ref:\n{lp_y}"
    );
    // The realization is still drawn for the `rand` query itself, so the assertion above is
    // about y's law specifically, not about the module having no sample in it.
    assert!(
        text.contains("builtin_sample"),
        "the rand query still samples z:\n{text}"
    );
}

// The regression half, and the reason this is a guard rather than a blanket refusal on
// a second query. A FIXED parameter is no stochastic ancestor: §04 integrates out
// "internal stochastic nodes in the traced sub-DAG, not boundary inputs". `mu` must
// survive the lowering AS A REFERENCE — a pinned latent would have been substituted —
// and the query must still emit its one density term, with an unrelated pinning query
// present so the provenance path is actually exercised.
#[test]
fn a_fixed_parameter_still_lowers_alongside_a_pinning_query() {
    let text = pir("\
mu = elementof(reals)
y = draw(Normal(mu = mu, sigma = 1.0))
w = draw(Normal(mu = 2.0, sigma = 1.0))
lp_w = logdensityof(lawof(w), 0.1)
lp_y = logdensityof(lawof(y), 0.5)");
    let lp_y = pir_binding(&text, "lp_y");
    assert!(
        lp_y.contains("(%field mu (%ref self mu))"),
        "the fixed parameter stays a parameter, not a pinned literal:\n{lp_y}"
    );
    assert_eq!(
        lp_y.matches("builtin_logdensityof").count(),
        1,
        "one density term:\n{lp_y}"
    );
    assert!(
        pir_binding(&text, "mu").contains("(elementof reals)"),
        "`mu` is still the declared parameter:\n{text}"
    );

    // A literal parameter is likewise no ancestor.
    let literal = pir("\
y = draw(Normal(mu = 2.0, sigma = 1.0))
w = draw(Normal(mu = 3.0, sigma = 1.0))
lp_w = logdensityof(lawof(w), 0.1)
lp_y = logdensityof(lawof(y), 0.5)");
    assert!(
        pir_binding(&literal, "lp_y").contains("(%field mu 2.0)"),
        "a literal parameter lowers unchanged:\n{literal}"
    );
}

// The provenance is per BINDING, not per module: pinning `a` says nothing about `b`.
// Both spellings, since each reaches the guard by a different route.
#[test]
fn independent_latents_pinned_by_separate_queries_still_lower() {
    for src in [
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp_a = logdensityof(lawof(a), 0.3)
lp_b = logdensityof(lawof(b), 0.5)",
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.3))
lp_b = logdensityof(lawof(record(b = b)), record(b = 0.5))",
    ] {
        let text = pir(src);
        assert!(
            pir_binding(&text, "lp_a").contains("(%field sigma 1.0)")
                && pir_binding(&text, "lp_b").contains("(%field sigma 2.0)"),
            "each query scores its own latent's law:\n{text}"
        );
    }
}

// A SIBLING field's draw is not an outside ancestor: `lawof(record(z = z, y = y))` is
// the chain-rule joint `p(z) p(y | z)`, which this lowering already scores with `z`
// pinned to its OWN field of the query point. The dependent factor is asserted by the
// pinned `mu`, so a guard that refused the joint — or one that scored `y`'s prior
// marginal instead — fails here.
#[test]
fn a_sibling_field_is_not_an_outside_ancestor() {
    let text = pir("\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(record(z = z, y = y)), record(z = 0.3, y = 0.5))");
    let lp = pir_binding(&text, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        2,
        "the joint is a product of two factors:\n{lp}"
    );
    assert!(
        lp.contains("(%field mu 0.3)"),
        "the dependent factor keeps its sibling, pinned to that sibling's own point:\n{lp}"
    );

    // The sibling may be reached through a DERIVED binding — `sigma = sqrt(sigma2)`,
    // whose draw site is `sigma2`'s. Exempting by draw SITE rather than by binding
    // name is what admits this.
    let derived = pir(
        "\
mu = draw(Normal(mu = 0.0, sigma = 1.0))
sigma2 = draw(Exponential(1.0))
sigma = sqrt(sigma2)
y = draw(Normal(mu = mu, sigma = sigma))
lp = logdensityof(lawof(record(mu = mu, sigma = sigma, y = y)), record(mu = 0.1, sigma = 1.0, y = 0.5))",
    );
    let lp = pir_binding(&derived, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        3,
        "three factors, one per field:\n{lp}"
    );
    assert!(
        lp.contains("(%field mu 0.1)") && lp.contains("(%field sigma 1.0)"),
        "y's factor keeps both siblings at their own points:\n{lp}"
    );
}

// §04's exception, and the reason the record check is sited at the `lawof` strip points
// rather than in `lower_record_of_draws`: a reification does not integrate its free
// stochastic params out, it CONDITIONS on them as boundary inputs. `forward_kernel =
// kernelof(record(obs = obs), theta1 = theta1, theta2 = theta2)` over derived `a`, `b`
// is that shape — the record's only field depends on two draws it does not carry, and
// the conditional is exactly what the likelihood wants. A guard inside the record
// lowering refused it. Kept in-crate as well as in the CLI fixture corpus
// (`fixtures/flatppl/queries/bayesian_inference_2_posterior.flatppl`) so the boundary
// is asserted where the guard lives.
#[test]
fn a_reification_body_conditions_on_its_boundary_inputs() {
    let text = pir("\
theta1 ~ Normal(0, 1)
theta2 ~ Exponential(1)
a = 5 * theta2
b = abs(theta1) * theta2
obs ~ iid(Normal(mu = a, sigma = b), 10)
prior = lawof(record(theta1 = theta1, theta2 = theta2))
forward_kernel = kernelof(record(obs = obs), theta1 = theta1, theta2 = theta2)
observed = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(obs = observed))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(theta1 = 0.5, theta2 = 1.0))");
    let lp = pir_binding(&text, "lp");
    // Axis-native: one broadcast density term over the ten observations, plus the two
    // prior factors.
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        3,
        "one broadcast likelihood term plus the two prior factors:\n{lp}"
    );
    assert!(
        lp.contains("(sum ") && lp.contains("(broadcast builtin_logdensityof Normal"),
        "the likelihood is the summed broadcast over the observations:\n{lp}"
    );
    // The kernel's boundary inputs are bound to the query's θ, so `b = abs(theta1) *
    // theta2` reduces at `theta1 = 0.5`, `theta2 = 1.0` — the conditional §04's
    // boundary-input exception licenses.
    assert!(
        lp.contains("(abs 0.5)"),
        "the derived sigma reduces at the θ point:\n{lp}"
    );
}

// The capabilities a bare stochastic value gained in measure position must not be
// collateral: each reaches the density dispatcher's fallthrough, where the pin and its
// provenance now live.
#[test]
fn a_bare_value_law_still_serves_in_every_measure_position() {
    let one = "z = draw(Normal(mu = 0.0, sigma = 1.0))\n";
    let two = "z = draw(Normal(mu = 0.0, sigma = 1.0))\nq = draw(Normal(mu = 1.0, sigma = 1.0))\n";
    for (src, marker) in [
        (
            format!("{one}lp = logdensityof(weighted(0.5, lawof(z)), 0.3)"),
            "(log 0.5)",
        ),
        (
            format!("{one}lp = logdensityof(normalize(lawof(z)), 0.3)"),
            "builtin_logdensityof",
        ),
        (
            format!("{one}lp = logdensityof(iid(lawof(z), 2), [0.3, 0.4])"),
            "(add ",
        ),
        (
            format!("{two}lp = logdensityof(superpose(lawof(z), lawof(q)), 0.3)"),
            "(logsumexp ",
        ),
        (
            format!(
                "{two}lp = logdensityof(joint(a = lawof(z), b = lawof(q)), record(a = 0.3, b = 0.4))"
            ),
            "(add ",
        ),
    ] {
        let lp = pir_binding(&pir(&src), "lp");
        assert!(lp.contains(marker), "expected `{marker}` in:\n{lp}");
    }
}
