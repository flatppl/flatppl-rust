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
    let _ = flatppl_infer::infer(&mut m);
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

/// Both spellings must lower the query binding to the same FlatPDL.
fn assert_alias_matches_direct(direct: &str, aliased: &str) {
    let d = binding_pir(&determinize_pir(direct), "lp");
    let a = binding_pir(&determinize_pir(aliased), "lp");
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
k = functionof(Normal.(mu))
lik = likelihoodof(k, [2.0, 3.0])
lp = logdensityof(lik, record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu))
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
k = functionof(Normal.(mu))
lik = likelihoodof(k, [2.0, 3.0])
lp = logdensityof(lik, record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu))
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
k = functionof(Normal.(mu))
l1 = likelihoodof(k, [2.0, 3.0])
l2 = likelihoodof(k, [1.0])
j = joint_likelihood(l1, l2)
lp = logdensityof(j, record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu))
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
k = functionof(Normal.(mu))
l1 = likelihoodof(k, [2.0, 3.0])
l2 = likelihoodof(k, [1.0])
lp = logdensityof(joint_likelihood(l1, l2), record(mu = 1.0))",
        "\
mu = elementof(reals)
k = functionof(Normal.(mu))
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
k = functionof(Normal.(mu))
lik = likelihoodof(k, [2.0, 3.0])
post = bayesupdate(lik, prior)
lp = logdensityof(post, record(mu = 1.0))",
        "\
mu ~ Normal(mu = 0.0, sigma = 10.0)
prior = lawof(record(mu = mu))
k = functionof(Normal.(mu))
lik = likelihoodof(k, [2.0, 3.0])
alias = lik
post = bayesupdate(alias, prior)
lp = logdensityof(post, record(mu = 1.0))",
    );
}
