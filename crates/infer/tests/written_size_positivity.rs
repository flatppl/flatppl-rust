//! A size WRITTEN in source must be positive; a size DERIVED from data may
//! resolve to 0.
//!
//! The split is the owner's 2026-08-20 ruling on zero-size arrays
//! (`flatppl-dev/empty-arrays-ruling.md`, shape C): "every *written* size is
//! floored positive, no *derived* size is … an author may not write a
//! degenerate shape, data may turn out empty". §06's `iid` entry now carries
//! the derived half in the spec itself — "a `size` derived from data rather
//! than written in source may resolve to 0, giving the empty product measure".
//!
//! Two properties are pinned here:
//!  - every argument position the spec declares a positive size refuses a
//!    non-positive literal, whichever section declares it;
//!  - a derived size stays legal and types `%dynamic`, never `Dim::Static(0)`
//!    (sub-ruling 3: "legality must not depend on optimizer strength"), so
//!    §11's "a positive integer dimension size, or `%dynamic`" holds for every
//!    type this crate produces.

use flatppl_infer::{Diagnostic, Level, Severity, infer_with};

fn diags(src: &str) -> Vec<Diagnostic> {
    let mut m = flatppl_syntax::parse(src).expect("fixture must parse");
    infer_with(&mut m, Level::Shape)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).expect("fixture must parse");
    let _ = infer_with(&mut m, Level::Shape);
    flatppl_flatpir::write(&m)
}

fn refuses(src: &str, needle: &str) -> bool {
    diags(src).iter().any(|d| d.message.contains(needle))
}

// ---- the written half: one row per spec section that declares a size ----

/// Every size position the spec floors positive, at a written `0`. The rows are
/// grouped by the section that declares the constraint, and each cites it: §07
/// "Array and table generation" ("`size`: a vector of positive integers", `eye`
/// / `onehot` / `linspace` / `extlinspace` "positive integer"), §07 "Array and
/// table operations" (`tile` / `splitblocks` / `partition` / `bandedmat`), §03
/// "Cartesian power" and "Standard simplex", §06 `iid` and `markovchain`, and
/// §08's two matrix-dimension parameters (`n = elementof(posintegers)`).
#[test]
fn every_written_size_position_refuses_a_zero() {
    let rows: &[(&str, &str, &str)] = &[
        (
            "array",
            "x = array(data = [1.0, 2.0], size = 0, dimorder = [1])",
            "§07 \"Array and table generation\"",
        ),
        (
            "fill",
            "x = fill(1.0, 0)",
            "§07 \"Array and table generation\"",
        ),
        (
            "zeros",
            "x = zeros(0)",
            "§07 \"Array and table generation\"",
        ),
        ("ones", "x = ones(0)", "§07 \"Array and table generation\""),
        ("eye", "x = eye(0)", "§07 \"Array and table generation\""),
        (
            "onehot",
            "x = onehot(1, 0)",
            "§07 \"Array and table generation\"",
        ),
        (
            "linspace",
            "x = linspace(0.0, 1.0, 0)",
            "§07 \"Array and table generation\"",
        ),
        (
            "extlinspace",
            "x = extlinspace(0.0, 1.0, 0)",
            "§07 \"Array and table generation\"",
        ),
        (
            "tile",
            "v = [1.0, 2.0]\nx = tile(v, 0)",
            "§07 \"Array and table operations\"",
        ),
        (
            "splitblocks",
            "v = [1.0, 2.0]\nx = splitblocks(v, 0)",
            "§07 \"Array and table operations\"",
        ),
        (
            "partition",
            "v = [1.0, 2.0]\nx = partition(v, 0)",
            "§07 \"Array and table operations\"",
        ),
        (
            "bandedmat",
            "v = [1.0, 2.0]\nx = bandedmat(v, 0)",
            "§07 \"Array and table operations\"",
        ),
        (
            "cartpow",
            "x = elementof(cartpow(reals, 0))",
            "§03 \"Cartesian power\"",
        ),
        (
            "stdsimplex",
            "x = elementof(stdsimplex(0))",
            "§03 \"Standard simplex\"",
        ),
        (
            "iid",
            "x = iid(Normal(mu = 0.0, sigma = 1.0), 0)",
            "§06 `iid`",
        ),
        (
            "markovchain",
            "f = s -> Normal(s, 1.0)\n\
             x = markovchain(f, 0.0, 0)",
            "§06 `markovchain`",
        ),
        ("LKJ", "x = LKJ(n = 0, eta = 1.0)", "§08 `LKJ`"),
        (
            "LKJCholesky",
            "x = LKJCholesky(n = 0, eta = 1.0)",
            "§08 `LKJCholesky`",
        ),
    ];
    for (op, src, section) in rows {
        assert!(
            refuses(src, "requires a positive size") && refuses(src, section),
            "`{op}` must refuse a written zero size citing spec {section}: {:?}",
            diags(src)
        );
    }
}

/// A NEGATIVE written size is refused for the same reason and by the same
/// message. Worth its own row because the lexer spells `-3` as `neg(3)`, a
/// call rather than a literal — before this rule a negative size was caught
/// only incidentally, where a downstream domain check happened to reject the
/// degraded shape.
#[test]
fn a_negative_written_size_is_refused() {
    assert!(
        refuses("x = zeros(-3)", "is written as `-3`"),
        "a written negative size must be refused: {:?}",
        diags("x = zeros(-3)")
    );
}

/// A size vector names the offending AXIS, and the other axes are not blamed:
/// `[2, 0, 5]` is one error, at axis 2.
#[test]
fn a_size_vector_names_the_offending_axis() {
    let src = "x = fill(1.0, [2, 0, 5])";
    let msgs: Vec<String> = diags(src).into_iter().map(|d| d.message).collect();
    assert_eq!(msgs.len(), 1, "one error, one axis: {msgs:?}");
    assert!(
        msgs[0].contains("axis 2 of `fill`'s `size` is written as `0`"),
        "the error must name the axis: {msgs:?}"
    );
}

/// The keyword spelling is the same call (§04 "Calling conventions": built-in
/// callables "accept both positional and keyword arguments"), so it is refused
/// too — the check runs on the written argument, whichever way it is bound.
#[test]
fn the_keyword_spelling_is_refused_too() {
    assert!(
        refuses("x = zeros(size = 0)", "`zeros`'s `size` is written as `0`"),
        "a keyword-bound written size must be refused: {:?}",
        diags("x = zeros(size = 0)")
    );
}

// ---- the derived half: legal, and typed `%dynamic` ----

/// A size derived from data is legal at 0. `filter` gives a dynamic length, so
/// nothing resolves and nothing is refused — the §06 "Region-restricted
/// likelihoods" idiom (`filter` → `lengthof` → `iid`) the ruling protects.
#[test]
fn a_derived_size_is_legal() {
    let src = "xs = [1.0, 2.0, 3.0]\n\
               keep = filter(x -> x < 0.0, xs)\n\
               n = lengthof(keep)\n\
               d = iid(Normal(mu = 0.0, sigma = 1.0), n)";
    assert!(
        diags(src).is_empty(),
        "a derived size must not be refused: {:?}",
        diags(src)
    );
}

/// A derived size the const-eval table CAN fold to 0 types `%dynamic`, not
/// `(%array 1 (0) …)`. This is sub-ruling 3 — "legality must not depend on
/// optimizer strength … a derived dimension is typed `%dynamic` even when its
/// value is known, including known-0" — and it is what closes the reachable
/// ill-formed-FlatPIR hole: §11 admits "a positive integer dimension size, or
/// … `%dynamic`", and 0 is neither.
#[test]
fn a_derived_zero_size_types_dynamic_not_a_zero_extent() {
    let src = "xs = [1.0, 2.0, 3.0]\n\
               n = sub(lengthof(xs), lengthof(xs))\n\
               z = zeros(n)";
    assert!(
        diags(src).is_empty(),
        "a derived zero must not be refused: {:?}",
        diags(src)
    );
    let out = ir(src);
    assert!(
        out.contains("(%bind z (%meta ((%array 1 (%dynamic) (%scalar real))"),
        "a derived zero size must type %dynamic:\n{out}"
    );
    assert!(
        !out.contains("(%array 1 (0)"),
        "no zero extent may reach FlatPIR (§11):\n{out}"
    );
}

/// A size that resolves NEGATIVE is refused wherever it is folded — no size is
/// negative at any phase, so the derived exemption does not cover it. It used
/// to become a silent `%dynamic`.
#[test]
fn a_derived_negative_size_is_refused() {
    let src = "xs = [1.0, 2.0]\n\
               n = sub(lengthof(xs), 5)\n\
               z = zeros(n)";
    assert!(
        refuses(src, "resolves to -3"),
        "a derived negative size must be refused: {:?}",
        diags(src)
    );
}

/// A positive written size is untouched, whichever section declares it — the
/// control that fails if the check is over-eager. (`eye(3)` is absent on
/// purpose: it has no shape rule off its size yet and types
/// `(%dynamic %dynamic)` either side of this change.)
#[test]
fn positive_written_sizes_still_type_exactly() {
    let src = "a = zeros([2, 5])\n\
               c = elementof(cartpow(reals, 4))\n\
               d = iid(Normal(mu = 0.0, sigma = 1.0), 7)";
    assert!(diags(src).is_empty(), "{:?}", diags(src));
    let out = ir(src);
    for expect in [
        "(%array 2 (2 5) (%scalar real))",
        "(%array 1 (4) (%scalar real))",
        "(%array 1 (7) (%scalar real))",
    ] {
        assert!(out.contains(expect), "missing {expect}:\n{out}");
    }
}

/// `addaxes`'s counts are NOT sizes: §07 declares them "non-negative integer,
/// non-negative integer", so a written `0` is a legal no-op and must keep
/// resolving the shape exactly. The size flooring would leave the count
/// unresolved and the whole result `%deferred`.
#[test]
fn addaxes_counts_are_non_negative_not_positive() {
    let src = "A = zeros([2, 5])\n\
               B = addaxes(A, 0, 2)";
    assert!(diags(src).is_empty(), "{:?}", diags(src));
    let out = ir(src);
    assert!(
        out.contains("(%array 4 (2 5 1 1) (%scalar real))"),
        "addaxes with a zero leading count must still resolve:\n{out}"
    );
}
