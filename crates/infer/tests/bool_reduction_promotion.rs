//! §03's boolean promotion under the §07 reductions.
//!
//! §03 "Bool": "In arithmetic contexts, `false` is promoted to zero and `true` to one,
//! permitting expressions such as `true + true`, `3 * false`, and `sum(mask)` to count
//! true entries." §03 "Scalar value categories and sets" fixes the canonical inclusions
//! `booleans` $\subset$ `integers` $\subset$ `reals`, so zero and one land in
//! `integers` — the narrowest set the promotion reaches, and what a COUNT is.
//!
//! Each test reads the rendered `%meta` slot of the reduction's own line, so the type
//! and the value-set are both pinned and cannot drift apart.

use flatppl_infer::infer;

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    flatppl_flatpir::write(&m)
}

/// The rendered line carrying the call to `head`.
fn call_line(src: &str, head: &str) -> String {
    let out = ir(src);
    out.lines()
        .find(|l| l.contains(&format!("({head} ")))
        .unwrap_or_else(|| panic!("no `{head}` call in:\n{out}"))
        .trim()
        .to_string()
}

/// `head(b)` over the boolean vector `b`, as one rendered line.
fn over_bools(head: &str) -> String {
    call_line(&format!("b = [true, true, false]\nr = {head}(b)\n"), head)
}

// ---- the fix: `sum`/`prod` count ----

/// §03 "Bool" makes `sum(mask)` "count true entries", and a count is an `Integer`
/// (§03 "Scalar types"; `booleans` $\subset$ `integers` by §03 "Scalar value categories
/// and sets"). Before this rule `sum` kept the boolean element type, and the StableHLO
/// emitter honoured it with a 1-bit `stablehlo.add` — parity. `sum([true, true, false])`
/// IREE-executed to `false` where the count is 2.
#[test]
fn sum_over_booleans_types_as_integer() {
    let line = over_bools("sum");
    assert!(
        line.contains("(%scalar integer)"),
        "sum over booleans must be integer; got: {line}"
    );
    assert!(
        line.contains("%fixed integers"),
        "and its value-set must be integers, not booleans; got: {line}"
    );
}

/// §03's promotion sentence gives `3 * false` as one of its own examples, so
/// multiplication is an arithmetic context: the product of the promoted zeros and ones
/// is an `Integer`, on the same authority as `sum`.
#[test]
fn prod_over_booleans_types_as_integer() {
    let line = over_bools("prod");
    assert!(
        line.contains("(%scalar integer)") && line.contains("%fixed integers"),
        "prod over booleans must be integer/integers; got: {line}"
    );
}

/// The cumulative pair reduces nothing away but performs the same arithmetic:
/// `cumsum([true, true, false])` is `[1, 2, 2]`, and `2` is not a boolean. The
/// catalogue's `SameAsArg(0)` row typed it `cartpow(booleans, 3)` — a value-set that
/// does not contain the value it describes.
#[test]
fn cumulative_over_booleans_types_as_integer_array() {
    for head in ["cumsum", "cumprod"] {
        let line = over_bools(head);
        assert!(
            line.contains("(%array 1 (3) (%scalar integer))"),
            "{head} over booleans must be an integer array; got: {line}"
        );
        assert!(
            line.contains("(cartpow integers 3)"),
            "{head}'s value-set must be cartpow integers 3; got: {line}"
        );
    }
}

// ---- the siblings, each settled from the same sentence ----

/// §07 "Reductions" defines `mean` as $\bar{x} = \frac{1}{n}\sum_i x_i$. The promoted
/// sum is an integer, but the division by `n` is not integral — `mean([true, false])` is
/// `0.5` — so the result is `Real`. Already the behaviour; pinned so the promotion
/// cannot later drag it down to `integers`.
#[test]
fn mean_over_booleans_stays_real() {
    let line = over_bools("mean");
    assert!(
        line.contains("(%scalar real)"),
        "mean over booleans must be real; got: {line}"
    );
}

/// §07 gives `var` $\frac{1}{n-1}\sum_i (x_i - \bar{x})^2$ and `std` $\sqrt{var}$, both
/// with domain "real arrays" — satisfied for a boolean array through §03's inclusion
/// `booleans` $\subset$ `reals`. A square root is not integral, so both stay `Real`.
#[test]
fn var_and_std_over_booleans_stay_real() {
    for head in ["var", "std"] {
        let line = over_bools(head);
        assert!(
            line.contains("(%scalar real)"),
            "{head} over booleans must be real; got: {line}"
        );
    }
}

/// `maximum`/`minimum` are deliberately NOT promoted. §03 scopes its promotion to
/// "arithmetic contexts", and §07 defines these as $\max_i x_i$ / $\min_i x_i$ — a
/// SELECTION of one element of the input, performing no arithmetic on it. A boolean
/// array's maximum is a boolean, which is also what their catalogue row's
/// `ElemScalarKind` result says. The numeric value is the same under either reading, so
/// nothing is silently wrong either way; this pins the narrower one.
#[test]
fn extrema_over_booleans_stay_boolean() {
    for head in ["maximum", "minimum"] {
        let line = over_bools(head);
        assert!(
            line.contains("(%scalar boolean)") && line.contains("%fixed booleans"),
            "{head} over booleans must stay boolean; got: {line}"
        );
    }
}

// ---- non-boolean elements are untouched ----

/// The promotion arm is guarded on a boolean element, so every other element kind keeps
/// the rule it had: an integer array's `sum` is an integer, a real array's is real, a
/// complex array's is complex (§07's "real/complex arrays" domain), and `cumsum` keeps
/// the catalogue's shape-and-kind-preserving `SameAsArg(0)` row.
#[test]
fn non_boolean_reductions_are_unchanged() {
    let cases = [
        ("v = [1, 2, 3]\nr = sum(v)\n", "sum", "(%scalar integer)"),
        ("v = [1.5, 2.5]\nr = sum(v)\n", "sum", "(%scalar real)"),
        (
            "v = [complex(1.0, 2.0), complex(3.0, 4.0)]\nr = sum(v)\n",
            "sum",
            "(%scalar complex)",
        ),
        (
            "v = [1, 2, 3]\nr = cumsum(v)\n",
            "cumsum",
            "(%array 1 (3) (%scalar integer))",
        ),
        (
            "v = [1.5, 2.5]\nr = cumsum(v)\n",
            "cumsum",
            "(%array 1 (2) (%scalar real))",
        ),
        (
            "v = [1, 2, 3]\nr = maximum(v)\n",
            "maximum",
            "(%scalar integer)",
        ),
    ];
    for (src, head, want) in cases {
        let line = call_line(src, head);
        assert!(
            line.contains(want),
            "{head} must still render {want}; got: {line}"
        );
    }
}
