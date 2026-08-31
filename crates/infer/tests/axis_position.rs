//! Spec §05 axis positions.
//!
//! §05 "Axis names and aggregation": "an axis name is legal only as an entry in
//! `aggregate`'s `output_axes` axis list, as an index inside `[...]` within the
//! body, or as a binder on the left-hand side of `:=`. Used anywhere else it is
//! a static error."
//!
//! §05 "Note on axis names": "The grammar likewise admits `AxisList` as a
//! `Primary`, but it is legal only as the `output_axes` argument of an
//! `aggregate` or `metricsum` call and as the axis-list binder of an
//! `AggregateBinding` or `MetricsumBinding`; anywhere else it is a static
//! error. Unlike `ArrayLiteral`, `AxisList` may be empty".
//!
//! Every spelling below parsed and inferred clean before these checks landed:
//! the grammar admits `Axis` and `AxisList` as a `Primary`, so a stray one is
//! not a parse error and nothing else looked at where it sat.

use flatppl_infer::{Severity, infer};

const SETUP: &str = "g = eye(2)\nA = rowstack([[1.0, 0.0], [0.0, 1.0]])\nv = [1.0, 2.0]\n";

fn messages(src: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(src).expect("fixture must parse");
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn rejects(src: &str, expected: &str) {
    let msgs = messages(src);
    assert!(
        msgs.iter().any(|m| m.contains(expected)),
        "no error containing `{expected}` for:\n{src}\ngot {msgs:?}"
    );
}

fn accepts(src: &str) {
    let msgs = messages(src);
    assert!(
        msgs.is_empty(),
        "unexpected errors for:\n{src}\ngot {msgs:?}"
    );
}

const LIST: &str = "an axis list is legal only as `output_axes`";
const AXIS: &str = "is out of position";

// --- `AxisList` outside `output_axes` -------------------------------------

#[test]
fn axis_list_on_a_binding_rhs_is_refused() {
    rejects("x = [.i, .j]\n", LIST);
}

#[test]
fn axis_list_under_a_tilde_is_refused() {
    rejects("y ~ [.i]\n", LIST);
}

#[test]
fn axis_list_under_unary_negation_is_refused() {
    rejects("x = -[.i]\n", LIST);
}

#[test]
fn axis_list_as_a_call_argument_is_refused() {
    rejects("x = sum([.i, .j])\n", LIST);
}

#[test]
fn axis_list_as_a_record_field_is_refused() {
    rejects("x = record(a = [.i, .j])\n", LIST);
}

/// The `expr` slot of the very call whose arg 1 is the one legal position.
#[test]
fn axis_list_in_the_aggregate_expr_slot_is_refused() {
    rejects("x = aggregate(sum, [.i], [.i])\n", LIST);
}

/// `metricsum`'s metric slot. Before this check the metric TYPE rule caught it
/// with a rank complaint, which sent the reader after the wrong problem.
#[test]
fn axis_list_in_the_metricsum_metric_slot_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum([.i], [.mu^], A[.mu^, .nu_] * v[.nu^])\n"),
        LIST,
    );
}

/// The binder desugars to `output_axes`, so the body is an ordinary `expr`
/// position — the list on the right of `:=` is not a second binder.
#[test]
fn axis_list_as_an_aggregate_binding_body_is_refused() {
    rejects(&format!("{SETUP}C[.i] := [.i]\n"), LIST);
}

#[test]
fn indexing_into_an_axis_list_is_refused() {
    rejects("x = [.i, .j][1]\n", LIST);
}

#[test]
fn axis_list_nested_in_an_array_literal_is_refused() {
    rejects("x = [[.i], 2]\n", LIST);
}

#[test]
fn axis_list_in_a_tuple_is_refused() {
    rejects("x = ([.i], 1)\n", LIST);
}

#[test]
fn axis_list_in_a_lambda_body_is_refused() {
    rejects("f = z -> [.i]\n", LIST);
}

#[test]
fn axis_list_in_a_function_definition_body_is_refused() {
    rejects("f(a) = [.i]\n", LIST);
}

#[test]
fn axis_list_in_a_reification_body_is_refused() {
    rejects("x = functionof([.i], y = _y_)\n", LIST);
}

#[test]
fn axis_list_under_a_broadcast_is_refused() {
    rejects("x = sum.([.i])\n", LIST);
}

#[test]
fn axis_list_as_an_iid_size_is_refused() {
    rejects("m = iid(Normal(0.0, 1.0), [.i])\n", LIST);
}

#[test]
fn axis_list_as_an_index_is_refused() {
    rejects(&format!("{SETUP}x = A[[.i]]\n"), LIST);
}

#[test]
fn axis_list_on_a_decomposition_rhs_is_refused() {
    rejects("a, b = [.i]\n", LIST);
}

// --- the empty axis list -------------------------------------------------

/// `flatppl-dev/empty-arrays-ruling.md` (2026-08-20): `[]` has no legal reading
/// outside the two `output_axes` slots, so an empty vector has no literal
/// spelling. §05's `ArrayLiteral ::= "[" Expression ("," Expression)* ","? "]"`
/// admits no empty form, and only `AxisList` "may be empty".
#[test]
fn empty_brackets_on_a_binding_rhs_are_refused() {
    rejects("x = []\n", LIST);
}

#[test]
fn empty_brackets_as_a_call_argument_are_refused() {
    rejects("x = sum([])\n", LIST);
}

#[test]
fn empty_brackets_under_a_tilde_are_refused() {
    rejects("y ~ []\n", LIST);
}

/// The refusal says why `[]` cannot be read as an empty vector, and where an
/// empty array does come from. Without that it reads as a bare prohibition on
/// empty arrays, which the ruling does not impose.
#[test]
fn the_empty_bracket_refusal_explains_the_reading() {
    let msgs = messages("x = []\n");
    let m = msgs.first().expect("one error");
    assert!(
        m.contains("`ArrayLiteral` admits no empty form") && m.contains("derived size"),
        "message must name the grammar reason and the derived-size route; got {m}"
    );
}

// --- bare axis names out of position -------------------------------------

#[test]
fn bare_axis_in_arithmetic_is_refused() {
    rejects("x = .i + 1\n", AXIS);
}

#[test]
fn bare_axis_on_a_binding_rhs_is_refused() {
    rejects("x = .i\n", AXIS);
}

#[test]
fn bare_axis_as_a_call_argument_is_refused() {
    rejects("x = sum(.i)\n", AXIS);
}

#[test]
fn bare_axis_as_a_record_field_is_refused() {
    rejects("x = record(a = .i)\n", AXIS);
}

#[test]
fn bare_axis_under_a_tilde_is_refused() {
    rejects("y ~ .i\n", AXIS);
}

/// An index is a legal axis position only "within the body" of an aggregation:
/// §05 scopes axis names "to the enclosing aggregation", and this index has no
/// enclosing aggregation to be scoped to.
#[test]
fn bare_axis_index_outside_any_aggregation_is_refused() {
    rejects(&format!("{SETUP}x = A[.i]\n"), AXIS);
}

/// Same rule through the `get` spelling §04 gives for the same construct.
#[test]
fn bare_axis_get_index_outside_any_aggregation_is_refused() {
    rejects(&format!("{SETUP}x = get(A, .i)\n"), AXIS);
}

/// The indexed OBJECT is not an index position, so an axis there is out of
/// position even inside a body.
#[test]
fn bare_axis_as_the_indexed_object_is_refused() {
    rejects(&format!("{SETUP}x = aggregate(sum, [.i], .i[1])\n"), AXIS);
}

// --- one error per refused bracket ---------------------------------------

/// A refused bracket raises one located error, not one per entry plus one for
/// the list. `[.i, .j] == 1` also has a type complaint waiting behind it (the
/// comparisons are scalar-domain), which the `Failed` seed keeps quiet.
#[test]
fn a_refused_bracket_raises_exactly_one_error() {
    let msgs = messages("x = [.i, .j] == 1\n");
    assert_eq!(
        msgs.len(),
        1,
        "one error for the bracket, none for its entries; got {msgs:?}"
    );
    assert!(msgs[0].contains(LIST));
}

// --- legal positions -----------------------------------------------------

#[test]
fn aggregate_output_axes_and_body_indices_are_accepted() {
    accepts(&format!("{SETUP}x = aggregate(sum, [.i], A[.i, .j])\n"));
}

#[test]
fn empty_aggregate_output_axes_are_accepted() {
    accepts(&format!("{SETUP}x = aggregate(sum, [], A[.i, .j])\n"));
}

#[test]
fn the_aggregate_binding_binder_is_accepted() {
    accepts(&format!("{SETUP}C[.i] := A[.i, .j]\n"));
}

#[test]
fn the_empty_aggregate_binding_binder_is_accepted() {
    accepts(&format!("{SETUP}s[] := A[.i, .j]\n"));
}

#[test]
fn the_metricsum_binding_binder_is_accepted() {
    accepts(&format!("{SETUP}g: T[.mu^] := A[.mu^, .nu_] * v[.nu^]\n"));
}

/// The empty metricsum binder the density sweep writes
/// (`flatppl-testsuite/src/flatppl_testsuite/sweep/space.py`).
#[test]
fn the_empty_metricsum_binding_binder_is_accepted() {
    accepts("g = eye(2)\np = [3.0, 2.0]\ng: s[] := p[.i^] * p[.i_]\n");
}

/// §04 spells both index forms out: "array indexing may contain axis names,
/// like `A[.i, 1, .j]` or `get(A, .i, 1, .j)`".
#[test]
fn the_get_index_spelling_is_accepted() {
    accepts(&format!(
        "{SETUP}x = aggregate(sum, [.i], get(A, .i, .j))\n"
    ));
}

/// The all-keyword spelling of the same call, keyed on §04's parameter names.
#[test]
fn the_keyword_aggregate_spelling_is_accepted() {
    accepts(&format!(
        "{SETUP}x = aggregate(f_reduction = sum, output_axes = [.i], expr = A[.i, .j])\n"
    ));
}

#[test]
fn a_nested_aggregation_is_accepted() {
    accepts(&format!(
        "{SETUP}x = aggregate(sum, [.i], aggregate(sum, [], A[.i, .j]))\n"
    ));
}

// --- false-positive guards -----------------------------------------------

/// `[.5]` is a one-element vector of the REAL `0.5`, not an axis list: §05
/// resolves "the trailing-dot real literal against a dotted operator" by
/// maximal munch, and the same munch takes `.5` as a `Number`.
#[test]
fn a_leading_dot_real_literal_is_not_an_axis() {
    accepts("x = [.5]\n");
    accepts("x = [.5e2]\n");
}

/// A `.name` after an expression is `FieldAccess`, not `Axis` (§05 "Note on
/// parser disambiguation"), so a list of field reads is an ordinary
/// `ArrayLiteral`.
#[test]
fn a_list_of_field_accesses_is_not_an_axis_list() {
    accepts("r = record(a = 1.0, b = 2.0)\nx = [r.a, r.b]\n");
}

/// A non-empty array literal in a value position is untouched — the check keys
/// on `Axis` entries, not on the `vector` head every literal shares.
#[test]
fn an_ordinary_array_literal_is_accepted() {
    accepts("x = [1.0, 2.0]\ny = sum([1, 2, 3])\n");
}
