//! An integer or boolean domain is checked by the argument's TYPE, never by its value
//! (user ruling 2026-10-26: "eye(2.0) is nonsense, as is Multinomial(3.0, p). If it is
//! an integer in reality, it needs to be an integer."). §03 "Integer" and "Bool" make
//! `2.0` a real. Before this check the constant-value test admitted `Binomial(1e308, 0.5)`,
//! `eye(2.0)` and `Multinomial(3.0, p)`, because each value lies in its set.
//!
//! §08 also declares its vector parameters "array of reals", and §03 "Arrays" makes a
//! transposed vector a distinct type, so `MvNormal(transpose(v), C)` is refused.
//!
//! An index is an integer too, so a real `get` selector is refused alongside the
//! separate 1-based rule.
//!
//! Every refusal must carry a location.

use flatppl_infer::{Diagnostic, Severity, infer};

const PRELUDE: &str = "v = [1.0, 2.0]\np = [0.5, 0.5]\nC = rowstack([[1.0, 0.0], [0.0, 1.0]])\n";

fn errors(src: &str) -> Vec<Diagnostic> {
    let mut m = flatppl_syntax::parse(&format!("{PRELUDE}{src}\n")).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn assert_refuses(src: &str, needle: &str) {
    let errs = errors(src);
    assert!(
        errs.len() == 1 && errs[0].message.contains(needle) && errs[0].node.is_some(),
        "`{src}` must refuse once, located, with {needle:?}; got: {errs:?}"
    );
}

#[test]
fn a_real_typed_integer_parameter_is_refused_whatever_its_value() {
    for (src, needle) in [
        (
            "x = Binomial(1e308, 0.5)",
            "`Binomial`'s `n` is a real, but spec §08 declares it an integer",
        ),
        ("x = Binomial(10.0, 0.5)", "`Binomial`'s `n` is a real"),
        (
            "x = Binomial(n = 10.0, p = 0.5)",
            "`Binomial`'s `n` is a real",
        ),
        ("x = Multinomial(3.0, p)", "`Multinomial`'s `n` is a real"),
        ("x = LKJ(2.0, 1.0)", "`LKJ`'s `n` is a real"),
        ("x = eye(2.0)", "`eye`'s `n` is a real, but spec §07"),
        ("x = linspace(0.0, 1.0, 5.0)", "`linspace`'s `n` is a real"),
        ("x = onehot(1.0, 3)", "`onehot`'s `i` is a real"),
        (
            "x = addaxes(v, 1.0, 0)",
            "`addaxes`'s `n_leading` is a real",
        ),
        (
            "x = checked(value = 1.0, condition = 2.0)",
            "declares it a boolean",
        ),
    ] {
        assert_refuses(src, needle);
    }
}

#[test]
fn a_real_typed_size_is_refused_whatever_its_value() {
    for (src, needle) in [
        ("x = fill(0.0, 2.0)", "`fill`'s `size` is a real"),
        ("x = zeros([2.0, 3.0])", "`zeros`'s `size` vector is a real"),
        (
            "x = iid(Normal(0.0, 1.0), 3.0)",
            "`iid`'s `size` is a real, but spec §06 `iid`",
        ),
        ("x = cartpow(reals, 2.0)", "`cartpow`'s `size` is a real"),
    ] {
        assert_refuses(src, needle);
    }
}

#[test]
fn integers_and_explicit_conversions_type() {
    for src in [
        "x = Binomial(10, 0.5)",
        "x = Binomial(integer(1e308), 0.5)",
        "x = Multinomial(3, p)",
        "x = eye(2)",
        "x = fill(0.0, [2, 3])",
        "x = iid(Normal(0.0, 1.0), 3)",
        "x = cartpow(reals, 2)",
        "x = addaxes(v, 1, 0)",
        "x = checked(value = 1.0, condition = true)",
        "x = MvNormal(v, C)",
    ] {
        let errs = errors(src);
        assert!(errs.is_empty(), "`{src}` must type; got: {errs:?}");
    }
}

#[test]
fn a_transposed_vector_is_not_an_array_parameter() {
    for (src, param) in [
        ("x = MvNormal(transpose(v), C)", "`MvNormal`'s `mu`"),
        ("x = Dirichlet(transpose(v))", "`Dirichlet`'s `alpha`"),
        ("x = Categorical(transpose(p))", "`Categorical`'s `p`"),
        ("x = Multinomial(3, transpose(p))", "`Multinomial`'s `p`"),
    ] {
        assert_refuses(
            src,
            &format!("{param} is a transposed vector, but spec §08 declares it an array"),
        );
    }
}

#[test]
fn a_real_typed_index_is_refused_whatever_its_value() {
    for (src, needle) in [
        (
            "x = get(v, 1.0)",
            "`get`'s selector 1 is a real, but spec §07 `get` declares it an integer",
        ),
        ("x = v[1.0]", "`get`'s selector 1 is a real"),
        ("x = C[1, 2.0]", "`get`'s selector 2 is a real"),
        ("x = get0(v, 0.0)", "`get0`'s selector 1 is a real"),
        ("x = v[[1.0, 2.0]]", "`get`'s selector 1 vector is a real"),
    ] {
        assert_refuses(src, needle);
    }
    for src in [
        "x = v[1]",
        "x = v[[1, 2]]",
        "x = C[all, 1]",
        "x = record(a = 1.0).a",
    ] {
        let errs = errors(src);
        assert!(errs.is_empty(), "`{src}` must type; got: {errs:?}");
    }
}
