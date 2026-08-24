//! A set bound to a name types exactly like the same set written inline.
//!
//! §04 "Design" opens: "A FlatPPL module is an unordered set of bindings of names to
//! expressions. Expressions are single or nested calls that bind expressions (literal
//! or by name reference) to inputs of callables." A name reference is admitted
//! wherever the expression itself is, so `elementof(S_name)` denotes the same set as
//! `elementof(<S_name's rhs>)`.
//!
//! §03 "Presets" makes the named form a first-class idiom rather than an accident:
//! "**Preset domains.** Any literal/fixed global binding like `some_name =
//! cartprod(name1=some_set, name2=some_other_set, ...)` can be interpreted as a
//! possibly suitable domain for input/parameter-compatible functions, kernels and
//! likelihoods", with the worked example
//! `L_domain = cartprod(a = interval(0, 5), b = cartpow(interval(-10, 10), 3), c = interval(0, 20))`.
//!
//! §03 "Cartesian power": "`cartpow(S, size)` produces the Cartesian power of `S` with
//! shape `size` ... `size` is a positive integer (1-D) or a vector of positive integers
//! (multi-axis). For example, `cartpow(reals, 3)` represents $\mathbb{R}^3$ and
//! `cartpow(reals, [3, 3])` the set of $3 \times 3$ real matrices."
//!
//! §03 "Sets that govern values": "For `x = elementof(S)`, `valueset(x)` is `S`."
//! §04 "Phase classification" puts an `elementof` node in the **parameterized** phase.
//!
//! Before this, neither the element-type reader nor the value-set reader followed a
//! `%ref`, so EVERY set-expression position lost its set through a name: `elementof`
//! and `external` typed `%deferred`/`%unknown`, `Lebesgue(support = S_name)` got a
//! `%deferred` domain and `%unknown` mass, and `truncate(M, S_name)` kept the base
//! measure's value set. The four-vector spelling
//! `fvs = cartpow(reals, 4)` / `v = elementof(fvs)` was the reported symptom; the
//! StableHLO emitter refused it with "type has no MLIR tensor form: Deferred".

use flatppl_infer::infer;

/// The `(%bind name …)` line of `src`, with inference asserted error-free. Notes are
/// tolerated: `stdsimplex` still has no type rule for the SET BINDING itself, which is
/// a separate gap from the element type read here.
fn bind_line(src: &str, name: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags: Vec<_> = infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(diags.is_empty(), "infer errors: {diags:?}");
    let out = flatppl_flatpir::write(&m);
    out.lines()
        .find(|l| l.contains(&format!("(%bind {name} ")))
        .unwrap_or_else(|| panic!("no `{name}` binding in:\n{out}"))
        .trim()
        .to_string()
}

/// The reported four-vector spelling: `elementof` of a named `cartpow(reals, 4)` is a
/// rank-1 real array of size 4, parameterized, with the power as its value set.
#[test]
fn named_cartpow_scalar_size_is_a_fixed_size_vector() {
    let line = bind_line(
        "four_vector_set = cartpow(reals, 4)\nv1 = elementof(four_vector_set)\n",
        "v1",
    );
    assert!(
        line.contains("(%array 1 (4) (%scalar real)) %parameterized (cartpow reals 4)"),
        "named cartpow(reals, 4) must give a size-4 real vector: {line}"
    );
}

/// The multi-axis form: `cartpow(reals, [2, 3])` is the set of 2x3 real matrices
/// (§03 "Cartesian power"), so its element is rank-2.
#[test]
fn named_cartpow_vector_size_is_multi_axis() {
    let line = bind_line("mats = cartpow(reals, [2, 3])\nm = elementof(mats)\n", "m");
    assert!(
        line.contains("(%array 2 (2 3) (%scalar real)) %parameterized"),
        "named cartpow(reals, [2, 3]) must give a rank-2 2x3 array: {line}"
    );
}

/// The named form and the inline form must produce the identical annotation — the
/// name reference is the only difference between the two sources.
#[test]
fn named_and_inline_set_expressions_agree() {
    for set in [
        "reals",
        "posreals",
        "integers",
        "interval(0.0, 1.0)",
        "cartpow(reals, 4)",
        "cartpow(reals, [2, 3])",
        "cartpow(cartpow(reals, 3), 2)",
        "stdsimplex(3)",
        "cartprod(reals, posreals)",
        "cartprod(a = reals, b = posreals)",
        "cartpow(cartprod(a = reals, b = posreals), 5)",
    ] {
        let inline = bind_line(&format!("x = elementof({set})\n"), "x");
        let named = bind_line(&format!("s = {set}\nx = elementof(s)\n"), "x");
        // Only the argument spelling may differ: compare the `%meta` head, which
        // carries the type, the phase and the value set.
        let head = |l: &str| {
            l.split_once("(%meta ")
                .and_then(|(_, r)| r.split_once(") (elementof"))
                .map(|(h, _)| h.to_string())
                .unwrap_or_else(|| panic!("unexpected bind line: {l}"))
        };
        assert_eq!(
            head(&inline),
            head(&named),
            "`elementof({set})` and its named form disagree:\n  inline: {inline}\n  named:  {named}"
        );
    }
}

/// The same reader serves every other set-expression position: `external`'s set,
/// `Lebesgue`'s `support`, and `truncate`'s set argument.
#[test]
fn every_set_expression_position_follows_a_name() {
    let src = "fvs = cartpow(reals, 4)\niv = interval(0.0, 1.0)\n\
               e = external(fvs)\nleb = Lebesgue(support = fvs)\n\
               nrm = Normal(mu = 0.0, sigma = 1.0)\ntr = truncate(nrm, iv)\n";
    let e = bind_line(src, "e");
    assert!(
        e.contains("(%array 1 (4) (%scalar real)) %fixed (cartpow reals 4)"),
        "external of a named cartpow must type as the power's element: {e}"
    );
    let leb = bind_line(src, "leb");
    assert!(
        leb.contains("(%domain (%array 1 (4) (%scalar real))) (%mass %locallyfinite)"),
        "Lebesgue over a named cartpow must carry its domain and mass: {leb}"
    );
    let tr = bind_line(src, "tr");
    assert!(
        tr.contains("%fixed (interval 0.0 1.0)"),
        "truncate to a named interval must adopt that interval as its value set: {tr}"
    );
}

/// A name that does not denote a set must keep deferring rather than getting an
/// invented element type. `nrm` is a measure, so no element type follows.
#[test]
fn a_name_that_is_not_a_set_stays_deferred() {
    let line = bind_line(
        "nrm = Normal(mu = 0.0, sigma = 1.0)\nx = elementof(nrm)\n",
        "x",
    );
    assert!(
        line.contains("(%deferred %parameterized %unknown)"),
        "`elementof` of a non-set name must stay deferred: {line}"
    );
}

/// `s_k = cartprod(s_{k-1}, s_{k-1})` shares one subexpression through a name, so the
/// traversal is $2^k$ over paths only $k$ deep — the blowup a depth bound cannot see.
/// Following refs made it reachable from a tiny file: measured on the un-budgeted fix,
/// k=20 in a 501-byte source cost 5.0s and 847MB, quadrupling per level. The node
/// budget is what the exponent runs out of.
///
/// Terminating quickly is the property under test, so the assertion is on the ANSWER
/// (both slots deferred, no half-resolved shape) rather than a wall-clock threshold: a
/// timing assert would be flaky on a loaded machine, while an exponential regression
/// cannot produce this answer at all.
#[test]
fn a_shared_subexpression_cannot_blow_up_the_traversal() {
    let mut src = String::from("s0 = reals\n");
    for k in 1..=20 {
        src.push_str(&format!("s{k} = cartprod(s{}, s{})\n", k - 1, k - 1));
    }
    src.push_str("x = elementof(s20)\n");
    let line = bind_line(&src, "x");
    assert!(
        line.contains("(%deferred %parameterized %unknown)"),
        "an over-budget set expression must read out deferred in BOTH slots: {line}"
    );
}

/// The budget is a work bound, not a stinginess bound: an 11-level shared product is a
/// 2048-component set and still resolves, which is far past anything a real model
/// writes. This pins the headroom, so a later tightening of the budget cannot quietly
/// start refusing ordinary sets.
#[test]
fn a_deeply_shared_but_affordable_set_still_resolves() {
    let mut src = String::from("s0 = reals\n");
    for k in 1..=11 {
        src.push_str(&format!("s{k} = cartprod(s{}, s{})\n", k - 1, k - 1));
    }
    src.push_str("x = elementof(s11)\n");
    let line = bind_line(&src, "x");
    assert!(
        !line.contains("%deferred") && !line.contains("%unknown"),
        "an 11-level shared product is well within budget and must resolve: {line}"
    );
}

/// The two slots must agree. When the element set is unreadable, the POWER is
/// unreadable: `cartpow` must defer whole rather than wrap `%deferred` in a shape,
/// because a `(%array 1 (4) %deferred)` type beside a `%unknown` value set describes no
/// set at all. `nrm` is a measure, so it gives `cartpow` an unreadable element.
#[test]
fn cartpow_over_an_unreadable_set_defers_whole() {
    let line = bind_line(
        "nrm = Normal(mu = 0.0, sigma = 1.0)\nbad = cartpow(nrm, 4)\nx = elementof(bad)\n",
        "x",
    );
    assert!(
        line.contains("(%deferred %parameterized %unknown)"),
        "cartpow over an unreadable element must defer whole, not wrap it: {line}"
    );
    assert!(
        !line.contains("%array"),
        "no half-resolved shape may survive: {line}"
    );
}
