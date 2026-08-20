//! §07 gives the comparisons a SCALAR domain, and the dotted spelling is the
//! elementwise route.
//!
//! §07 "Operator-equivalent functions", Comparison functions: `lt`/`le`/`gt`/`ge`
//! have Domains `reals`, and `equal`/`unequal` have "`integers`, `booleans`,
//! strings" — all scalar value-sets, where the `add`/`sub` rows in the same table
//! read "scalars or arrays of same shape". §05 "Excluded constructs" states the
//! general rule: "**No implicit operator broadcasting.**"
//!
//! Before this, an array operand was ACCEPTED and typed `Scalar(Boolean)` while the
//! StableHLO emitter broadcast the same node to `tensor<nxi1>` — the declared type
//! and the emitted ABI disagreed on the result's shape, with no diagnostic on either
//! side. Both halves are pinned below: the bare form refuses, and the dotted form
//! types the boolean array the emitter actually produces.

use flatppl_infer::infer;

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn call_line(src: &str, head: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    let out = flatppl_flatpir::write(&m);
    out.lines()
        .find(|l| l.contains(&format!("({head} ")))
        .unwrap_or_else(|| panic!("no `{head}` call in:\n{out}"))
        .trim()
        .to_string()
}

/// Every comparison head, every spelling that reaches it, over a vector operand.
#[test]
fn a_bare_comparison_over_an_array_is_refused() {
    for head in ["equal", "unequal", "lt", "le", "gt", "ge"] {
        let src = format!("v = elementof(cartpow(reals, [3]))\nr = {head}(v, 1.0)\n");
        let errs = errors(&src);
        assert_eq!(errs.len(), 1, "`{head}` must refuse exactly once: {errs:?}");
        assert!(
            errs[0].contains(&format!("`{head}` expects scalar operands"))
                && errs[0].contains("Operator-equivalent functions"),
            "`{head}` must cite §07's Domains column: {}",
            errs[0]
        );
    }
}

/// The refusal must fire on EITHER operand, not just the first — `1.0 > v` is the
/// same shape violation as `v > 1.0`.
#[test]
fn the_refusal_fires_on_either_operand() {
    for src in [
        "v = elementof(cartpow(reals, [3]))\nr = gt(v, 1.0)\n",
        "v = elementof(cartpow(reals, [3]))\nr = gt(1.0, v)\n",
        "v = elementof(cartpow(reals, [3]))\nw = elementof(cartpow(reals, [3]))\nr = gt(v, w)\n",
    ] {
        assert_eq!(errors(src).len(), 1, "must refuse: {src}");
    }
}

/// The INFIX spelling lowers to the same named head (§07: "they lower to the
/// following named function equivalents"), so it must refuse identically. This is
/// the form a model is most likely to be written in.
#[test]
fn the_infix_spelling_refuses_too() {
    let errs = errors("v = elementof(cartpow(reals, [3]))\nr = v > 1.0\n");
    assert_eq!(errs.len(), 1, "the infix form must refuse: {errs:?}");
    assert!(
        errs[0].contains("`gt` expects scalar operands"),
        "{}",
        errs[0]
    );
}

/// A matrix operand, and a TRANSPOSED vector — §03 keeps the latter a distinct type,
/// and it is no more a scalar than an array is.
#[test]
fn the_refusal_covers_every_non_scalar_shape() {
    for (label, src) in [
        (
            "matrix",
            "m = elementof(cartpow(reals, [2, 3]))\nr = lt(m, 1.0)\n",
        ),
        (
            "transposed vector",
            "v = elementof(cartpow(reals, [3]))\nr = lt(transpose(v), 1.0)\n",
        ),
    ] {
        let errs = errors(src);
        assert_eq!(errs.len(), 1, "{label} must refuse: {errs:?}");
    }
}

/// The diagnostic must name the route that WORKS, or it sends the reader nowhere.
#[test]
fn the_diagnostic_names_the_dotted_form() {
    let errs = errors("v = elementof(cartpow(reals, [3]))\nr = gt(v, 1.0)\n");
    assert!(
        errs[0].contains("`gt.(a, b)`"),
        "the message must name the dotted spelling: {}",
        errs[0]
    );
}

/// The dotted route, all three spellings — `broadcast(gt, …)`, `gt.(…)` and the
/// dotted operator. Each is §07's elementwise application of a scalar function, so
/// each types a boolean ARRAY of the operand's shape. Before this they all typed
/// `%deferred`, which left the refusal above pointing at a route with no type.
#[test]
fn the_dotted_route_types_a_boolean_array() {
    for expr in ["broadcast(gt, v, 1.0)", "gt.(v, 1.0)", "v .> 1.0"] {
        let src = format!("v = elementof(cartpow(reals, [3]))\nr = {expr}\n");
        let line = call_line(&src, "broadcast");
        assert!(
            line.contains("(%array 1 (3) (%scalar boolean))"),
            "`{expr}` must type a [3] boolean array; got: {line}"
        );
    }
}

/// The dotted logical connectives too, so two masks can be combined. §07 gives
/// `land`/`lor`/`lxor`/`lnot` the scalar domain `booleans`, and `broadcast` is what
/// applies a scalar function elementwise.
#[test]
fn the_dotted_connectives_type_a_boolean_array() {
    let src = "\
v = elementof(cartpow(reals, [3]))
m1 = v .> 1.0
m2 = v .< 2.0
r = land.(m1, m2)
";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    let out = flatppl_flatpir::write(&m);
    let r_line = out
        .lines()
        .find(|l| l.contains("(%bind r") || l.trim_start().starts_with("(r "))
        .unwrap_or_else(|| panic!("no `r` binding in:\n{out}"));
    assert!(
        r_line.contains("(%array 1 (3) (%scalar boolean))"),
        "a dotted `land` over two masks is a [3] boolean array; got: {r_line}"
    );
}

/// The mask is what §07 "Boolean reductions" hands `lany`/`lall`, so the whole route
/// must type end to end. This is the reason the two halves above belong in one
/// change: without the dotted typing, the new boolean reductions would have had no
/// well-typed source-level input at all.
#[test]
fn a_dotted_mask_feeds_the_boolean_reductions() {
    for head in ["lany", "lall"] {
        let src = format!("v = elementof(cartpow(reals, [3]))\nr = {head}(v .> 1.0)\n");
        let line = call_line(&src, head);
        assert!(
            line.contains("(%scalar boolean)"),
            "`{head}` over a dotted mask is a boolean scalar; got: {line}"
        );
    }
}

/// The controls the refusal must NOT catch: two scalars, and the `add`/`sub` family,
/// whose §07 Domains cell explicitly reads "scalars or arrays of same shape".
#[test]
fn scalar_comparisons_and_array_arithmetic_are_untouched() {
    for src in [
        "a = elementof(reals)\nb = elementof(reals)\nr = gt(a, b)\n",
        "a = elementof(reals)\nr = a > 1.0\n",
        "v = elementof(cartpow(reals, [3]))\nw = elementof(cartpow(reals, [3]))\nr = add(v, w)\n",
        "v = elementof(cartpow(reals, [3]))\nw = elementof(cartpow(reals, [3]))\nr = v + w\n",
        "v = elementof(cartpow(reals, [3]))\nr = neg(v)\n",
    ] {
        assert!(errors(src).is_empty(), "must stay legal: {src}");
    }
}

/// A scalar comparison still types a boolean scalar — the refusal must not have
/// changed the ordinary case's type.
#[test]
fn a_scalar_comparison_still_types_boolean() {
    let line = call_line("a = elementof(reals)\nr = gt(a, 1.0)\n", "gt");
    assert!(line.contains("(%scalar boolean)"), "got: {line}");
}

/// The `in` membership test is NOT a comparison row: §07 lists it separately, and
/// its own Arguments are `x`, `S` — a value against a SET, where the set argument is
/// not an array operand to be broadcast. Left alone deliberately.
#[test]
fn the_membership_test_is_not_covered() {
    let src = "v = elementof(cartpow(reals, [3]))\nr = in(1.0, reals)\n";
    assert!(errors(src).is_empty(), "`in` is a separate §07 row: {src}");
}
