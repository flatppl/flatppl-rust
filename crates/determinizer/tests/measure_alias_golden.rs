//! A measure bound to a BARE NAME — `M = g1` — lowers exactly as the expression
//! it aliases.
//!
//! §04 *Name resolution*: an unqualified name "resolves to that binding", and a
//! module is "an unordered set of bindings of names to expressions". So an alias
//! names the very expression its right-hand side names, for every measure shape.
//!
//! Before this, the density dispatcher resolved ONE level of `(%ref self x)`, so
//! an alias left it holding the ref node `g1` — not a measure op and not a
//! constructor call — and every aliased measure refused with "primitive measure
//! must be a built-in constructor call". The HS3 importer emits
//! `__M__ = <pdf_name>` on its prenormalized branch, so the whole generic-pdf
//! corpus hit it.

mod common;

use flatppl_determinizer::{determinize, is_flatpdl};

fn determinize_pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "fixture must infer: {diags:?}");
    let out = determinize(&m).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
    pir
}

/// The right-hand side of binding `name`, as emitted FlatPIR.
fn binding_pir(pir: &str, name: &str) -> String {
    let at = pir
        .find(&format!("(%bind {name} "))
        .unwrap_or_else(|| panic!("no binding `{name}` in:\n{pir}"));
    common::call_arg(&pir[at..], "%bind", 1)
}

/// Rename the generated per-cell inputs `density_point_<n>` by order of appearance: their
/// serial number is the module's node count, which an alias changes.
fn alpha_normalize(pir: &str) -> String {
    let mut seen: Vec<String> = Vec::new();
    let mut out = String::new();
    let mut rest = pir;
    while let Some(at) = rest.find("density_point_") {
        out.push_str(&rest[..at]);
        let tail = &rest[at + "density_point_".len()..];
        let digits = tail.chars().take_while(char::is_ascii_digit).count();
        let name = format!("density_point_{}", &tail[..digits]);
        let index = seen.iter().position(|n| *n == name).unwrap_or_else(|| {
            seen.push(name);
            seen.len() - 1
        });
        out.push_str(&format!("density_arg_{index}"));
        rest = &tail[digits..];
    }
    out.push_str(rest);
    out
}

/// Both spellings must lower the query binding to the same FlatPDL.
fn assert_alias_matches_direct(direct: &str, aliased: &str) {
    let d = alpha_normalize(&binding_pir(&determinize_pir(direct), "lp"));
    let a = alpha_normalize(&binding_pir(&determinize_pir(aliased), "lp"));
    assert_eq!(a, d, "aliased spelling must lower like the direct one");
}

#[test]
fn primitive_constructor_alias() {
    assert_alias_matches_direct(
        "\
g1 = Normal(0.0, 1.0)
lp = logdensityof(g1, 0.0)",
        "\
g1 = Normal(0.0, 1.0)
M = g1
lp = logdensityof(M, 0.0)",
    );
}

#[test]
fn truncate_alias() {
    assert_alias_matches_direct(
        "\
t1 = truncate(Normal(0.0, 1.0), interval(-1.0, 1.0))
lp = logdensityof(t1, 0.25)",
        "\
t1 = truncate(Normal(0.0, 1.0), interval(-1.0, 1.0))
M = t1
lp = logdensityof(M, 0.25)",
    );
}

#[test]
fn superpose_alias() {
    assert_alias_matches_direct(
        "\
s1 = superpose(weighted(0.4, Normal(0.0, 1.0)), weighted(0.6, Normal(1.0, 2.0)))
lp = logdensityof(s1, 0.0)",
        "\
s1 = superpose(weighted(0.4, Normal(0.0, 1.0)), weighted(0.6, Normal(1.0, 2.0)))
M = s1
lp = logdensityof(M, 0.0)",
    );
}

#[test]
fn normalize_alias() {
    assert_alias_matches_direct(
        "\
n1 = normalize(superpose(weighted(0.4, Normal(0.0, 1.0)), weighted(0.6, Normal(1.0, 2.0))))
lp = logdensityof(n1, 0.0)",
        "\
n1 = normalize(superpose(weighted(0.4, Normal(0.0, 1.0)), weighted(0.6, Normal(1.0, 2.0))))
M = n1
lp = logdensityof(M, 0.0)",
    );
}

/// A chain of aliases, which one extra hop would still refuse.
#[test]
fn alias_chain() {
    assert_alias_matches_direct(
        "\
g1 = Normal(0.0, 1.0)
lp = logdensityof(g1, 0.0)",
        "\
g1 = Normal(0.0, 1.0)
M = g1
M2 = M
lp = logdensityof(M2, 0.0)",
    );
}

/// A `lawof` behind an alias chain is stripped at the query entry, as it is when
/// spelled inline (§04 *Kernels and `kernelof`*).
#[test]
fn lawof_alias_chain() {
    assert_alias_matches_direct(
        "\
z = draw(Normal(0.0, 1.0))
lp = logdensityof(lawof(z), 0.0)",
        "\
z = draw(Normal(0.0, 1.0))
L = lawof(z)
L2 = L
lp = logdensityof(L2, 0.0)",
    );
}

// ---------------------------------------------------------------------------
// The LIKELIHOOD layer, which the fix above left behind.
//
// The measure dispatcher followed the ref chain; the likelihood dispatcher still
// took one hop, so an aliased likelihood refused with "expected likelihoodof".
// The pyhf importer emits `likelihood = <channel>_likelihood` for any workspace
// whose likelihood has a single term, which is every workspace with no
// constrained parameter, so nine fixtures of the vendored pyhf corpus hit it.
// ---------------------------------------------------------------------------

/// `top = lik` where `lik = likelihoodof(K, obs)`.
#[test]
fn likelihoodof_alias() {
    assert_alias_matches_direct(
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
lik = likelihoodof(k, [2.0, 3.0])
lp = logdensityof(lik, record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
lik = likelihoodof(k, [2.0, 3.0])
top = lik
lp = logdensityof(top, record(mu = 1.0))",
    );
}

/// Two hops, to pin that it is a chain and not a second single hop.
#[test]
fn likelihoodof_alias_chain() {
    assert_alias_matches_direct(
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
lik = likelihoodof(k, [2.0, 3.0])
lp = logdensityof(lik, record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
lik = likelihoodof(k, [2.0, 3.0])
top = lik
top2 = top
lp = logdensityof(top2, record(mu = 1.0))",
    );
}

/// An aliased `joint_likelihood`.
#[test]
fn joint_likelihood_alias() {
    assert_alias_matches_direct(
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
l1 = likelihoodof(k, [2.0, 3.0])
l2 = likelihoodof(k, [1.0])
j = joint_likelihood(l1, l2)
lp = logdensityof(j, record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
l1 = likelihoodof(k, [2.0, 3.0])
l2 = likelihoodof(k, [1.0])
j = joint_likelihood(l1, l2)
top = j
lp = logdensityof(top, record(mu = 1.0))",
    );
}

/// An aliased COMPONENT inside a `joint_likelihood`, which is the second of the
/// three dispatch sites: the components are resolved separately from the joint.
#[test]
fn joint_likelihood_aliased_component() {
    assert_alias_matches_direct(
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
l1 = likelihoodof(k, [2.0, 3.0])
l2 = likelihoodof(k, [1.0])
lp = logdensityof(joint_likelihood(l1, l2), record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu, 1.0))
l1 = likelihoodof(k, [2.0, 3.0])
l2 = likelihoodof(k, [1.0])
a1 = l1
lp = logdensityof(joint_likelihood(a1, l2), record(mu = 1.0))",
    );
}

/// An aliased likelihood inside `bayesupdate`, the third dispatch site: the
/// posterior's likelihood arm resolves separately from the query entry.
#[test]
fn bayesupdate_aliased_likelihood() {
    assert_alias_matches_direct(
        "\
mu ~ Normal(mu = 0.0, sigma = 10.0)
prior = lawof(record(mu = mu))
k = functionof(iid(Normal(mu, 1.0), 2), mu = mu)
lik = likelihoodof(k, [2.0, 3.0])
post = bayesupdate(lik, prior)
lp = logdensityof(post, record(mu = 1.0))",
        "\
mu ~ Normal(mu = 0.0, sigma = 10.0)
prior = lawof(record(mu = mu))
k = functionof(iid(Normal(mu, 1.0), 2), mu = mu)
lik = likelihoodof(k, [2.0, 3.0])
alias = lik
post = bayesupdate(alias, prior)
lp = logdensityof(post, record(mu = 1.0))",
    );
}

// ---------------------------------------------------------------------------
// VALUE aliases: `M = a` where `a` is a draw, scored through `lawof`.
//
// The query entry followed the chain, but the value-law, record-field and
// shared-latent reads each took one hop. So `lawof(M)` refused as a
// non-constructor, `lawof(record(y = M))` as "field is not a draw", and the
// pin landed on the alias instead of the draw binding. Each lowering is
// compared with the direct spelling, and the direct spelling with the closed
// form (independently computed log densities in each comment).
// ---------------------------------------------------------------------------

/// The query binding `lp` of a module lowered for that one output.
fn lp_pir_for_root(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "fixture must infer: {diags:?}");
    let lp = m.intern("lp");
    let out = flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&[lp]),
    )
    .expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
    alpha_normalize(&binding_pir(&pir, "lp"))
}

/// `logpdf(Normal(0.3, 2), 0.5) = -1.617085713764618`.
#[test]
fn value_alias_lawof() {
    let direct = "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
lp = logdensityof(lawof(a), 0.5)";
    let lp = binding_pir(&determinize_pir(direct), "lp");
    assert!(
        lp.contains("(record (%field mu 0.3) (%field sigma 2.0))) 0.5)"),
        "{lp}"
    );
    assert_alias_matches_direct(
        direct,
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
M = a
lp = logdensityof(lawof(M), 0.5)",
    );
    assert_alias_matches_direct(
        direct,
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
M = a
M2 = M
lp = logdensityof(lawof(M2), 0.5)",
    );
}

#[test]
fn value_alias_record_field() {
    assert_alias_matches_direct(
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
lp = logdensityof(lawof(record(y = a)), record(y = 0.5))",
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
M = a
lp = logdensityof(lawof(record(y = M)), record(y = 0.5))",
    );
}

/// The total law of `y` integrates `z` out: `Normal(0, √2)`, and
/// `logpdf(Normal(0, √2), 0.5) = -1.3280121234846454`.
#[test]
fn value_alias_of_a_dependent_draw_marginalizes() {
    let direct = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(y), 0.5)";
    let lp = binding_pir(&determinize_pir(direct), "lp");
    assert!(
        lp.contains("(record (%field mu 0.0) (%field sigma 1.4142135623730951))) 0.5)"),
        "{lp}"
    );
    assert_alias_matches_direct(
        direct,
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
M = y
lp = logdensityof(lawof(M), 0.5)",
    );
}

/// An alias of the latent a dependent draw reads, in a joint record law.
#[test]
fn value_alias_of_a_latent_in_a_record_law() {
    assert_alias_matches_direct(
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(record(z = z, y = y)), record(z = 0.1, y = 0.5))",
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
Z = z
y = draw(Normal(mu = Z, sigma = 1.0))
lp = logdensityof(lawof(record(z = Z, y = y)), record(z = 0.1, y = 0.5))",
    );
}

/// `2a` with `a ~ Normal(0.3, 2)` is `Normal(0.6, 4)`, and
/// `logpdf(Normal(0.6, 4), 0.5) = logpdf(Normal(0.3, 2), 0.25) - log 2 = -2.305545394324563`.
/// Lowered for `lp` alone: with every binding an output, `M` is itself a
/// stochastic output, which refuses as the direct `c = a + 1.0` does.
#[test]
fn value_alias_under_a_transform() {
    let direct = lp_pir_for_root(
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
b = a * 2.0
lp = logdensityof(lawof(b), 0.5)",
    );
    assert!(
        direct.contains("(record (%field mu 0.3) (%field sigma 2.0))) 0.25)")
            && direct.contains("(abs 2.0)"),
        "{direct}"
    );
    let aliased = lp_pir_for_root(
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
M = a
b = M * 2.0
lp = logdensityof(lawof(b), 0.5)",
    );
    assert_eq!(aliased, direct);
}

/// A closed kernel is a measure (§04), so scoring it through an alias lowers,
/// and `lawof` of it refuses as a measure, not as a conjugate-table miss.
#[test]
fn kernel_alias() {
    assert_alias_matches_direct(
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
K = kernelof(a)
lp = logdensityof(K, 0.5)",
        "\
a = draw(Normal(mu = 0.3, sigma = 2.0))
K = kernelof(a)
K2 = K
lp = logdensityof(K2, 0.5)",
    );
    for query in ["lawof(K)", "lawof(K2)"] {
        let mut m = flatppl_syntax::parse(&format!(
            "a = draw(Normal(mu = 0.3, sigma = 2.0))\nK = kernelof(a)\nK2 = K\n\
             lp = logdensityof({query}, 0.5)"
        ))
        .unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let reason = determinize(&m).expect_err("lawof of a measure").reason;
        assert_eq!(
            reason, "lawof requires a value node, not a measure or callable",
            "{query}"
        );
    }
}
