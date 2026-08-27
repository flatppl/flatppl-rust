//! Spec §04 "Metric-aware Einstein summation" / "Static checks": "bare neutral
//! aggregate axes (`.i` without a variance marker) are not allowed inside
//! `metricsum`."
//!
//! Variance-marked axis names are "required inside `metricsum`" (same section),
//! so a neutral axis has no lowering there: each `_` axis becomes an
//! `inv(metric)` contraction and each `^` axis reads storage directly, and a
//! bare `.i` names neither. Before this gate a neutral axis was absorbed
//! silently and the determiniser lowered it as a plain `aggregate` axis.

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

#[test]
fn a_neutral_axis_in_the_body_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu] * v[.nu])\n"),
        "bare neutral axis `.nu`",
    );
}

#[test]
fn a_neutral_output_axis_is_refused() {
    rejects(
        &format!("{SETUP}T = metricsum(g, [.mu], A[.mu^, .nu_] * v[.nu^])\n"),
        "bare neutral axis `.mu`",
    );
}

#[test]
fn the_refusal_names_both_variance_spellings() {
    let msgs = messages(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu] * v[.nu])\n"
    ));
    assert!(
        msgs.iter()
            .any(|m| m.contains(".nu^") && m.contains(".nu_")),
        "the refusal must offer both markers; got {msgs:?}"
    );
}

/// A located refusal, so the CLI and the LSP can point at the axis itself.
#[test]
fn the_refusal_carries_its_node() {
    let errs = errors(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu] * v[.nu])\n"
    ));
    let d = errs
        .iter()
        .find(|d| d.message.contains("bare neutral axis"))
        .expect("expected the neutral-axis refusal");
    assert!(
        d.node.is_some(),
        "the diagnostic must carry its node: {d:?}"
    );
}

#[test]
fn a_fully_marked_metricsum_stays_clean() {
    accepts(&format!(
        "{SETUP}T = metricsum(g, [.mu^, .beta_], A[.mu^, .nu_] * B[.nu^, .beta_])\n"
    ));
}

/// Axis names are "lexically scoped to the enclosing `metricsum`", so a nested
/// `aggregate` keeps its own neutral axes.
#[test]
fn a_nested_aggregate_keeps_its_neutral_axes() {
    accepts(&format!(
        "{SETUP}T = metricsum(g, [.mu^], A[.mu^, .nu_] \
         * aggregate(sum, [.nu], B[.nu, .k] * v[.k])[.nu^])\n"
    ));
}

/// A plain `aggregate` is untouched: neutral axes are its normal spelling.
#[test]
fn a_plain_aggregate_is_untouched() {
    accepts(&format!("{SETUP}s = aggregate(sum, [], v[.i] * v[.i])\n"));
}
