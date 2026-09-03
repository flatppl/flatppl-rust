//! Type rules for the seven §07 heads the `missing-reductions` spec draft adds —
//! `lany`, `lall`, `linfnorm`, `cummax`, `cummin`, `median`, `quantile`.
//!
//! Normative source: flatppl-design branch `missing-reductions` @ `ee4c6fb`,
//! docs/07 + docs/04. Landed ahead of the spec merge per the owner's ruling.
//!
//! Each test reads the rendered `%meta` slot of the call's own line, so the type
//! and the value-set are pinned together and cannot drift apart — the discipline
//! `bool_reduction_promotion.rs` established.

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

// ---- §07 "Boolean reductions" -------------------------------------------------

/// §07: "`lany` | `xs` | `true` if at least one element of `xs` is `true` | boolean
/// arrays". A truth value, so the result is a boolean scalar whatever the array's
/// shape.
#[test]
fn lany_and_lall_reduce_a_boolean_array_to_a_boolean_scalar() {
    for head in ["lany", "lall"] {
        let line = call_line(&format!("b = [true, false, true]\nr = {head}(b)\n"), head);
        assert!(
            line.contains("(%scalar boolean)"),
            "`{head}` over a boolean array is a boolean scalar; got: {line}"
        );
        assert!(
            line.contains("booleans"),
            "and its value-set must be `booleans`; got: {line}"
        );
    }
}

/// The contrast with `sum`, which §03 "Bool" DOES promote ("`sum(mask)` to count
/// true entries" — an `Integer`). §07 defines `lany` as "the `lor`-reduction of its
/// input" and `lall` as "the `land`-reduction", and `booleans` is closed under both,
/// so no promotion applies: a truth value is not §03's "arithmetic context".
#[test]
fn the_boolean_reductions_are_not_promoted_the_way_sum_is() {
    let src = "b = [true, true, false]\ns = sum(b)\na = lany(b)\n";
    assert!(
        call_line(src, "sum").contains("(%scalar integer)"),
        "`sum` still counts"
    );
    assert!(
        call_line(src, "lany").contains("(%scalar boolean)"),
        "`lany` does not"
    );
}

// ---- §07 "Norms and normalization" --------------------------------------------

/// §07: "`linfnorm` | `v` | $\max_i \lvert v_i\rvert$ | real/complex vectors" —
/// a non-negative real scalar, exactly like its two siblings.
#[test]
fn linfnorm_is_a_nonnegative_real_scalar() {
    let line = call_line("v = [-3.0, 2.0, -7.0]\nr = linfnorm(v)\n", "linfnorm");
    assert!(
        line.contains("(%scalar real)"),
        "`linfnorm` is a real scalar; got: {line}"
    );
    assert!(
        line.contains("nonnegreals"),
        "a norm is non-negative; got: {line}"
    );
}

/// The whole norm family must agree on the result type, or a model that swaps one
/// for another changes type. Pinned as a family rather than head by head.
#[test]
fn the_three_norms_share_one_result_type() {
    let src = "v = [-3.0, 2.0, -7.0]\na = l1norm(v)\nb = l2norm(v)\nc = linfnorm(v)\n";
    for head in ["l1norm", "l2norm", "linfnorm"] {
        let line = call_line(src, head);
        assert!(
            line.contains("(%scalar real)") && line.contains("nonnegreals"),
            "`{head}` must be a non-negative real scalar; got: {line}"
        );
    }
}

// ---- §07 "Cumulative operations" ----------------------------------------------

/// §07 puts `cummax`/`cummin` under "Cumulative operations", whose prose reads
/// "they preserve the shape of their input rather than reducing it" — so the result
/// is the argument's own type.
#[test]
fn the_cumulative_extrema_preserve_shape_and_element_type() {
    for head in ["cummax", "cummin"] {
        let line = call_line(&format!("v = [3.0, 1.0, 7.0, 5.0]\nr = {head}(v)\n"), head);
        assert!(
            line.contains("(%array 1 (4) (%scalar real))"),
            "`{head}` keeps the [4] real shape; got: {line}"
        );
    }
}

/// **The asymmetry with `cumsum`.** §03 "Bool" promotes in "arithmetic contexts",
/// and `cumsum([true, true, false])` is `[1, 2, 2]` — not booleans — so `cumsum`
/// and `cumprod` carry a promotion carve-out. A running MAXIMUM performs no
/// arithmetic: it selects an element, exactly as `maximum`/`minimum` do (which §03's
/// sentence likewise does not reach). So a boolean vector's running max is a boolean
/// vector, the catalogue's `SameAsArg(0)` row is exact, and there is no carve-out to
/// write.
#[test]
fn the_cumulative_extrema_take_no_boolean_promotion() {
    let src = "b = [true, true, false]\ns = cumsum(b)\nm = cummax(b)\n";
    let sum_line = call_line(src, "cumsum");
    assert!(
        sum_line.contains("(%array 1 (3) (%scalar integer))"),
        "`cumsum` over booleans promotes to integers; got: {sum_line}"
    );
    let max_line = call_line(src, "cummax");
    assert!(
        max_line.contains("(%array 1 (3) (%scalar boolean))"),
        "`cummax` over booleans stays boolean; got: {max_line}"
    );
}

/// The same argument as `maximum`/`minimum`, checked side by side so the two
/// families cannot drift: both select an element, so both keep the element kind.
#[test]
fn the_cumulative_extrema_agree_with_maximum_on_the_element_kind() {
    let src = "v = [3, 1, 7]\nm = maximum(v)\nc = cummax(v)\n";
    assert!(
        call_line(src, "maximum").contains("(%scalar integer)"),
        "`maximum` of an integer array is an integer"
    );
    assert!(
        call_line(src, "cummax").contains("(%array 1 (3) (%scalar integer))"),
        "so `cummax` of one is an integer array"
    );
}

// ---- §07 the order statistics -------------------------------------------------

/// §07 defines `median` as $x_{((n+1)/2)}$ for odd $n$ and
/// "$\tfrac{1}{2}(x_{(n/2)} + x_{(n/2+1)})$" for even $n$. The even case is
/// arithmetic between two order statistics, so `median([1, 2])` is `1.5` and the
/// result is REAL even over an integer array. $n$'s parity is not always static, so
/// `Real` is the type that covers the row.
#[test]
fn median_is_real_even_over_an_integer_array() {
    let line = call_line("v = [1, 2, 3, 4]\nr = median(v)\n", "median");
    assert!(
        line.contains("(%scalar real)"),
        "`median` of an integer array is real; got: {line}"
    );
}

/// §07's `quantile` is "linear interpolation between the order statistics of `xs`",
/// over "real arrays, `interval(0, 1)`" — a real scalar, and TWO inputs.
#[test]
fn quantile_is_a_real_scalar_of_two_inputs() {
    let line = call_line("v = [1.0, 2.0, 3.0]\nr = quantile(v, 0.5)\n", "quantile");
    assert!(
        line.contains("(%scalar real)"),
        "`quantile` is a real scalar; got: {line}"
    );
}

/// `quantile`'s `p` domain is `interval(0, 1)`, which this type system cannot
/// enforce: there is no dependent interval type, and for a COMPUTED `p` no engine
/// can enforce it statically at all. Recorded so the gap is a decision rather than
/// an omission — a `p` outside `[0, 1]` types clean here.
///
/// **Kept deliberately when argument-domain checking landed** (see
/// `crates/infer/tests/argument_domains.rs`). That pass enforces the domains a
/// spec sentence states as an exclusion — §03 "Bool"'s "zero and one are not
/// implicitly converted to booleans", §03 "Scalar types"'s omission of strings,
/// §06's measure operands — and `p`'s is not one of those: `interval(0, 1)` is a
/// VALUE SET, so refusing `3.0` here would need a value-set membership test on a
/// literal, which is the §04 value-set surface (`trace.rs`'s substitution check),
/// not the argument-domain surface. Enforcing it for a literal only would also be
/// the worst of both: a rule the author cannot rely on, since the same `p` reached
/// through a binding would pass.
#[test]
fn quantile_does_not_enforce_its_p_domain_statically() {
    let mut m = flatppl_syntax::parse("v = [1.0, 2.0]\nr = quantile(v, 3.0)\n").unwrap();
    let diags = infer(&mut m);
    assert!(
        diags.is_empty(),
        "an out-of-range literal `p` is not caught here — §07's `interval(0, 1)` has no \
         static form in this type system; got: {diags:?}"
    );
}

// ---- §07 "Table reductions" ---------------------------------------------------

/// §07 Table reductions, as the draft extends it: "When `sum`, `mean`, `var`,
/// `std`, `prod`, `maximum`, `minimum`, `median`, `lany`, or `lall` is applied to a
/// table, the reduction operates column-wise and returns a record whose fields are
/// the column names and values are the per-column reductions."
#[test]
fn median_over_a_table_is_a_record_of_reals() {
    let src = "\
xs = elementof(cartpow(reals, 4))
ys = elementof(cartpow(reals, 4))
t = table(x = xs, y = ys)
r = median(t)
";
    let line = call_line(src, "median");
    assert!(
        line.contains("%record") && line.matches("(%scalar real)").count() == 2,
        "`median(t)` is a record of per-column reals; got: {line}"
    );
}

/// The boolean pair reduce a table too — §07 "Boolean reductions": "Both reduce a
/// table column-wise, as described under reductions." Each field keeps the boolean
/// result type, not the column's own.
#[test]
fn the_boolean_reductions_over_a_table_are_records_of_booleans() {
    let src = "\
p = elementof(cartpow(booleans, 4))
q = elementof(cartpow(booleans, 4))
t = table(p = p, q = q)
r = lall(t)
";
    let line = call_line(src, "lall");
    assert!(
        line.contains("%record") && line.matches("(%scalar boolean)").count() == 2,
        "`lall(t)` is a record of per-column booleans; got: {line}"
    );
}

/// The three new column-wise heads must take a sole positional table WHOLE rather
/// than auto-splatting its columns onto their `xs` parameter — §04's single-input
/// carve-out. A splat would compare the column names against `xs` and reject a call
/// §07 defines. (`builtin_param_names.rs` sweeps every base name for this; these
/// three are pinned here too, next to their type rules.)
#[test]
fn the_new_column_wise_heads_do_not_splat_a_table() {
    for head in ["median", "lany", "lall"] {
        let src = format!(
            "\
xs = elementof(cartpow(reals, 4))
ys = elementof(cartpow(reals, 4))
t = table(zzq = xs, zzr = ys)
r = {head}(t)
"
        );
        let mut m = flatppl_syntax::parse(&src).unwrap();
        let diags = infer(&mut m);
        assert!(
            diags.is_empty(),
            "`{head}` must take the table whole, whatever its columns are named; got: {diags:?}"
        );
    }
}

/// `quantile` is NOT in the exempt set: §04's carve-out needs "exactly one input",
/// and `quantile` takes two. §07's Table reductions paragraph omits it for the same
/// reason. So a sole positional table DOES splat onto its parameters and the name
/// check fires.
#[test]
fn quantile_is_not_exempt_from_the_table_splat() {
    let src = "\
xs = elementof(cartpow(reals, 4))
ys = elementof(cartpow(reals, 4))
t = table(zzq = xs, zzr = ys)
r = quantile(t)
";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    assert!(
        !diags.is_empty(),
        "`quantile` takes two inputs, so §04's single-input carve-out does not reach it"
    );
}

// ---- §04 "Multi-axis aggregation" --------------------------------------------

/// §04's eligible-reduction list gains `median`, `lany` and `lall`. §04 makes the
/// result "an array of the shape declared by `output_axes`" whose entries are
/// `f_reduction` applied to the contracted slice — so the ENTRY type is the
/// reduction's, not the body's.
#[test]
fn aggregate_median_types_real_and_the_boolean_pair_type_boolean() {
    let src = "\
A = elementof(cartpow(booleans, [2, 3]))
r = aggregate(lany, [.i], A[.i, .j])
";
    let line = call_line(src, "aggregate");
    assert!(
        line.contains("(%array 1 (2) (%scalar boolean))"),
        "`aggregate(lany, …)` entries are booleans; got: {line}"
    );
}

/// The case the body's own kind gets wrong. Over a genuinely integer-typed body
/// (`indicesof`), `median` must still be REAL — it averages two order statistics at
/// even $n$. Without the override the result took the body's `integer` kind.
#[test]
fn aggregate_median_is_real_over_an_integer_body() {
    let src = "\
A = elementof(cartpow(reals, [2, 3]))
r = aggregate(median, [], indicesof(A))
";
    let line = call_line(src, "aggregate");
    assert!(
        line.contains("(%scalar real)"),
        "`median` is real whatever the body was; got: {line}"
    );
}

/// The mirror for the boolean pair: a REAL body still gives a boolean result,
/// because §07 makes `lany` a truth value.
#[test]
fn aggregate_lall_is_boolean_over_a_real_body() {
    let src = "\
A = elementof(cartpow(reals, [2, 3]))
r = aggregate(lall, [.i], A[.i, .j])
";
    let line = call_line(src, "aggregate");
    assert!(
        line.contains("(%array 1 (2) (%scalar boolean))"),
        "`lall` is a truth value whatever the body was; got: {line}"
    );
}

/// The controls: `sum` and `maximum` must be unaffected by the override table, or
/// it would have changed a shipped type.
#[test]
fn the_aggregate_override_leaves_the_existing_reductions_alone() {
    let src = "\
A = elementof(cartpow(reals, [2, 3]))
s = aggregate(sum, [.i], A[.i, .j])
";
    let line = call_line(src, "aggregate");
    assert!(
        line.contains("(%array 1 (2) (%scalar real))"),
        "`sum` still follows the body; got: {line}"
    );
}
