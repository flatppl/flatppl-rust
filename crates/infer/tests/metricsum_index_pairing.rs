//! Spec §04 "Metric-aware Einstein summation" / "Static checks": "Every
//! repeated non-output index in `expr` must occur exactly twice — once upper
//! and once lower; every output index must occur in `expr` with the same
//! variance and may not also be contracted".
//!
//! Both halves used to be absorbed silently: an unpaired repeated index summed
//! stored components with no metric factor, because "Lowering to `aggregate`"
//! inserts `inv(metric)` for a `_` axis only.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<flatppl_infer::Diagnostic> {
    let mut module = flatppl_syntax::parse(src).expect("fixture must parse");
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn messages(src: &str) -> Vec<String> {
    errors(src).into_iter().map(|d| d.message).collect()
}

fn rejects(src: &str, expected: &str) {
    let msgs = messages(src);
    assert!(
        msgs.iter().any(|m| m.contains(expected)),
        "no error containing `{expected}`; got {msgs:?}"
    );
}

fn accepts(src: &str) {
    let msgs = messages(src);
    assert!(msgs.is_empty(), "unexpected errors: {msgs:?}");
}

const SETUP: &str = "g = eye(2)\nA = rowstack([[1.0, 0.0], [0.0, 1.0]])\n\
                     B = rowstack([[1.0, 0.0], [0.0, 1.0]])\nv = [1.0, 2.0]\n";

// --- "Every repeated non-output index ... exactly twice, once upper, once lower"

#[test]
fn a_twice_upper_index_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu^] * v[.nu^])\n"),
        "contracted axis `.nu` must occur exactly twice",
    );
}

#[test]
fn a_twice_lower_index_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu_] * v[.nu_])\n"),
        "contracted axis `.nu` must occur exactly twice",
    );
}

#[test]
fn a_three_times_repeated_index_is_refused() {
    rejects(
        &format!("{SETUP}s = metricsum(g, [], v[.nu^] * v[.nu_] * v[.nu_])\n"),
        "contracted axis `.nu` must occur exactly twice",
    );
}

/// The counts, so the author can see which end to change.
#[test]
fn the_refusal_reports_the_counts() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu^] * v[.nu^])\n"),
        "occurs 2 upper and 0 lower",
    );
}

/// A located refusal, so the CLI and the LSP can point at the axis itself.
#[test]
fn the_pairing_refusal_carries_its_node() {
    let errs = errors(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu^] * v[.nu^])\n"
    ));
    let d = errs
        .iter()
        .find(|d| d.message.contains("must occur exactly twice"))
        .expect("expected the pairing refusal");
    assert!(
        d.node.is_some(),
        "the diagnostic must carry its node: {d:?}"
    );
}

#[test]
fn a_paired_contraction_stays_clean() {
    accepts(&format!(
        "{SETUP}T = metricsum(g, [.mu^, .beta_], A[.mu^, .nu_] * B[.nu^, .beta_])\n"
    ));
}

/// The clause governs REPEATED indices only. "Equivalence to `aggregate` under
/// identity metric" makes `metricsum(eye(n), ...)` an `aggregate(sum, ...)`, and
/// a row sum is exactly one unpaired non-output index, so refusing it would
/// refuse a construct the equivalence clause requires.
#[test]
fn a_single_unpaired_non_output_index_is_allowed() {
    accepts(&format!("{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu_])\n"));
}

/// Axis names are "lexically scoped to the enclosing `metricsum`", so a nested
/// `aggregate`'s own axes are not counted here.
#[test]
fn a_nested_aggregate_does_not_feed_the_count() {
    accepts(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu_] \
         * aggregate(sum, [.nu], B[.nu, .k] * v[.k])[.nu^])\n"
    ));
}

// --- "every output index must occur in `expr` with the same variance and may
// --- not also be contracted"

#[test]
fn an_output_index_absent_from_expr_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.rho^], A[.mu^, .nu_] * B[.nu^, .mu_])\n"),
        "output axis `.rho^` does not occur in `expr`",
    );
}

#[test]
fn an_output_index_with_the_opposite_variance_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu^], A[.mu_, .nu_] * B[.nu^, .mu^])\n"),
        "output axis `.mu^` also occurs in `expr` as `.mu_`",
    );
}

/// A lower output index gets the same treatment, with the markers the other way
/// round.
#[test]
fn a_lower_output_index_is_checked_too() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu_], A[.mu^, .nu_] * B[.nu^, .mu^])\n"),
        "output axis `.mu_` also occurs in `expr` as `.mu^`",
    );
}

#[test]
fn the_output_refusal_names_the_contraction_rule() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu^], A[.mu_, .nu_] * B[.nu^, .mu^])\n"),
        "may not also be contracted",
    );
}

/// A located refusal, pointing at the offending body occurrence.
#[test]
fn the_output_refusal_carries_its_node() {
    let errs = errors(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu_, .nu_] * B[.nu^, .mu^])\n"
    ));
    let d = errs
        .iter()
        .find(|d| d.message.contains("also occurs in `expr`"))
        .expect("expected the output-index refusal");
    assert!(
        d.node.is_some(),
        "the diagnostic must carry its node: {d:?}"
    );
}

/// An output index used twice with its declared variance is not a contraction —
/// nothing in the clause forbids it.
#[test]
fn an_output_index_repeated_with_its_own_variance_is_allowed() {
    accepts(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu_] * B[.nu^, .mu^])\n"
    ));
}

/// A plain `aggregate` has neither clause: its axes carry no variance.
#[test]
fn a_plain_aggregate_is_untouched() {
    accepts(&format!("{SETUP}s = aggregate(sum, [], v[.i] * v[.i])\n"));
}

/// Without a literal axis list an output index cannot be told from a contracted
/// one, so both checks stand down rather than refuse valid code.
#[test]
fn a_non_literal_axis_list_defers() {
    accepts(&format!(
        "{SETUP}axes = [.mu^]\nT = metricsum(g, axes, A[.mu^, .nu^])\n"
    ));
}

/// The `:=` shorthand lowers to `metricsum`, so it is gated the same way.
#[test]
fn the_shorthand_is_gated_too() {
    rejects(
        &format!("{SETUP}g: T[.mu^] := A[.mu^, .nu^] * v[.nu^]\n"),
        "contracted axis `.nu` must occur exactly twice",
    );
}
