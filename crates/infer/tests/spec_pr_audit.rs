//! Conformance rows from the audit of flatppl-design PRs #209 and #210 (and the
//! spec text they restate), each a spec sentence with a probe that fails on a
//! checker that does not enforce it.
//!
//! - §04 "Calling conventions" (#209): "The measure combinators likewise take their
//!   non-variadic inputs positionally: a keyword spelling such as `normalize(M = mu)`
//!   is a static error." The keyword form of `joint` stays legal.
//! - §07 `rnginit` (#209): "A byte-vector seed must contain integers in {0, …, 255}."
//! - §04 "Broadcasting" (#210): "Records and tuples are not allowed as operands of
//!   `broadcast`", and the arguments of `broadcasted(f)(args...)` are operands of
//!   `broadcast`, never auto-splatted.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn assert_clean(src: &str) {
    let found = errors(src);
    assert!(found.is_empty(), "unexpected errors {found:?}\n{src}");
}

fn assert_refused(src: &str, needle: &str) {
    let found = errors(src);
    assert!(
        found.iter().any(|m| m.contains(needle)),
        "expected an error containing `{needle}`, got {found:?}\n{src}"
    );
}

const MODEL: &str = "\
mu = Normal(mu = 0.0, sigma = 1.0)
x = elementof(reals)
k = functionof(Normal(mu = x, sigma = 1.0), x = x)
L = likelihoodof(k, 0.5)
";

#[test]
fn a_keyword_spelling_of_a_measure_combinators_input_is_refused() {
    for call in [
        "normalize(M = mu)",
        "totalmass(M = mu)",
        "iid(M = mu, size = 3)",
        "truncate(M = mu, S = interval(0.0, 1.0))",
        "bayesupdate(L = L, prior = mu)",
        "likelihoodof(K = k, obs = 0.5)",
        "restrict(M = mu, x = 0.0)",
        "locscale(m = mu, shift = 1.0, scale = 2.0)",
        "bijection(f = fn(exp(_)), f_inv = fn(log(_)), logvolume = fn(_))",
    ] {
        assert_refused(&format!("{MODEL}n = {call}\n"), "positionally");
    }
    assert_refused(
        "j = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 1.0, sigma = 1.0))\n\
         k1, m1 = disintegrate(selector = \"a\", joint_measure = j)\n",
        "positionally",
    );
}

#[test]
fn the_positional_spellings_and_the_keyword_joint_stay_legal() {
    assert_clean(&format!(
        "{MODEL}a = normalize(mu)\nb = iid(mu, 3)\nc = truncate(mu, interval(0.0, 1.0))\n\
         d = bayesupdate(L, mu)\n\
         e = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 1.0, sigma = 1.0))\n"
    ));
}

#[test]
fn unnamed_variadic_cat_and_get_refuse_a_keyword_spelling() {
    assert_refused("n = cat(a = [1.0], b = [2.0])\n", "positionally");
    assert_refused(
        "r = record(a = 1.0)\nn = get(x = r, i = \"a\")\n",
        "positionally",
    );
}

#[test]
fn flatpir_applies_the_same_rule() {
    let source = "(%module\n  (%public z)\n  (%bind mu (Normal (%kwarg mu 0.0) (%kwarg sigma 1.0)))\n  \
                  (%bind z (normalize (%kwarg M (%ref self mu)))))\n";
    let mut m = flatppl_flatpir::read(source).unwrap();
    let found: Vec<String> = infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        found.iter().any(|m| m.contains("positionally")),
        "{found:?}"
    );
}

#[test]
fn a_written_byte_vector_seed_stays_within_a_byte() {
    assert_clean("a = rnginit(5)\nb = rnginit([0, 1, 255])\n");
    assert_refused("a = rnginit([1, 2, 256])\n", "outside 0..255");
    assert_refused("a = rnginit([1, -2])\n", "outside 0..255");
}

#[test]
fn record_and_tuple_operands_of_broadcast_are_refused() {
    let f = "f(a, b) = a + b\n";
    assert_refused(
        &format!("{f}r = record(a = [1.0, 2.0], b = [3.0, 4.0])\nz = broadcast(f, r)\n"),
        "Records and tuples are not allowed as operands",
    );
    assert_refused(
        &format!("{f}t = ([1.0, 2.0], [3.0, 4.0])\nz = broadcast(f, t)\n"),
        "a tuple",
    );
}

#[test]
fn the_arguments_of_broadcasted_are_operands_and_are_never_splatted() {
    let f = "f(a, b) = a + b\n";
    assert_refused(
        &format!("{f}r = record(a = [1.0, 2.0], b = [3.0, 4.0])\nz = broadcasted(f)(r)\n"),
        "a record",
    );
    // A table operand is traversed row-wise, and positional or keyword arrays bind as usual.
    assert_clean(&format!(
        "{f}t = table(a = [1.0, 2.0], b = [3.0, 4.0])\nz = broadcasted(f)(t)\n\
         w = broadcasted(f)([1.0, 2.0], [3.0, 4.0])\nv = broadcasted(f)(a = [1.0], b = [2.0])\n"
    ));
}
