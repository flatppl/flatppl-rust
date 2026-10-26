//! §06 "Known-bijection registry": `pow` with a finite nonzero literal exponent,
//! "of which `sqrt` = `pow(_, 1/2)` is a case". A quotient of numeric literals,
//! also under `neg`, is therefore a literal exponent. It folds to its value, so it
//! lowers byte-identically to its decimal spelling. It is a real (§03), never an odd
//! integer, so the non-negative support rule holds. Structural only (flatppl-rust is
//! not a density engine): the numbers are checked through Enzyme-JAX in the PR.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn lowered(forward: &str, base: &str, y: &str) -> String {
    let m = parse_infer(&format!(
        "d = pushfwd(fn({forward}), {base})\nlp = logdensityof(d, {y})"
    ));
    let out = determinize(&m).unwrap_or_else(|e| panic!("{forward} must lower: {e:?}"));
    flatppl_flatpir::write(&out)
}

fn refusal(forward: &str, base: &str) -> String {
    let m = parse_infer(&format!(
        "k = 0.5\nd = pushfwd(fn({forward}), {base})\nlp = logdensityof(d, 0.5)"
    ));
    determinize(&m).expect_err("must refuse").reason
}

const EXP: &str = "Exponential(rate = 1.0)";

#[test]
fn a_quotient_exponent_lowers_as_its_decimal_spelling() {
    for (quotient, decimal, y) in [
        ("_^(1/2)", "_^0.5", "0.5"),
        ("pow(_, 1/3)", "pow(_, 0.3333333333333333)", "0.5"),
        ("_^(-1/2)", "_^(-0.5)", "2.0"),
        ("_^(-(1/2))", "_^(-0.5)", "2.0"),
        ("_^(1.0/(-4))", "_^(-0.25)", "2.0"),
    ] {
        assert_eq!(
            lowered(quotient, EXP, y),
            lowered(decimal, EXP, y),
            "{quotient} vs {decimal}"
        );
    }
    let iid = "iid(Exponential(rate = 1.0), 3)";
    assert_eq!(
        lowered("_ .^ (1/2)", iid, "[0.25, 1.0, 4.0]"),
        lowered("_ .^ 0.5", iid, "[0.25, 1.0, 4.0]"),
        "elementwise"
    );
}

#[test]
fn a_quotient_exponent_keeps_the_non_negative_domain() {
    let p = lowered("_^(1/3)", EXP, "0.5");
    assert!(
        p.contains("nonnegreals") && p.contains(" 3.0)"),
        "gated on y >= 0, inverse y^3:\n{p}"
    );
    let e = refusal("_^(1/3)", "Normal(mu = 0.0, sigma = 1.0)");
    assert!(
        e.contains("pow(_, 0.3333333333333333) requires M's support to lie within its domain"),
        "{e}"
    );
}

#[test]
fn only_literals_and_their_quotients_fold() {
    for (forward, reason) in [
        (
            "_^k",
            "only when k is a numeric literal or a quotient of literals",
        ),
        (
            "_^(k/2)",
            "only when k is a numeric literal or a quotient of literals",
        ),
        (
            "_^(1 + 1)",
            "only when k is a numeric literal or a quotient of literals",
        ),
        ("_^(1/0)", "pow with exponent inf is not invertible"),
        ("_^(0/2)", "pow with exponent 0 is not invertible"),
    ] {
        let e = refusal(forward, EXP);
        assert!(e.contains(reason), "{forward}: {e}");
    }
}
