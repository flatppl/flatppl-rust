//! The `metric` argument of `metricsum` — spec §04 "Metric-aware Einstein
//! summation": "It must be a square, symmetric, and invertible rank-2 array",
//! and "`metric` itself and all arrays indexed with co-/contravariant axis
//! names in `expr` must be arrays of scalars."
//!
//! §03 "Arrays" keeps a nested literal out of that type: "Vectors of vectors
//! are not interpreted as matrices implicitly, but can be turned into matrices
//! explicitly using `rowstack` or `colstack`." So `[[1.0, 0.0], [0.0, -1.0]]`
//! is a vector of two vectors, and only the explicit lift makes it a metric.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(src).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn rejects(src: &str, expected: &str) {
    let messages = errors(src);
    assert!(
        messages.iter().any(|m| m.contains(expected)),
        "no error containing `{expected}`; got {messages:?}"
    );
}

fn accepts(src: &str) {
    let messages = errors(src);
    assert!(messages.is_empty(), "unexpected errors: {messages:?}");
}

#[test]
fn a_nested_literal_metric_is_a_vector_of_vectors() {
    rejects(
        "v = [1.0, 2.0]\nL = metricsum([[1.0, 0.0], [0.0, -1.0]], [.mu^], v[.mu^])\n",
        "vector of vectors",
    );
}

/// The diagnostic must name the lift, since the fix is one call away.
#[test]
fn the_refusal_offers_rowstack_and_colstack() {
    let messages =
        errors("v = [1.0, 2.0]\nL = metricsum([[1.0, 0.0], [0.0, -1.0]], [.mu^], v[.mu^])\n");
    assert!(
        messages
            .iter()
            .any(|m| m.contains("rowstack") && m.contains("colstack")),
        "the refusal must offer both lifts; got {messages:?}"
    );
}

#[test]
fn the_explicit_lift_is_accepted() {
    accepts(
        "v = [1.0, 2.0]\nL = metricsum(rowstack([[1.0, 0.0], [0.0, -1.0]]), [.mu^], v[.mu^])\n",
    );
}

#[test]
fn eye_is_accepted() {
    accepts("v = [1.0, 2.0, 3.0, 4.0]\nL = metricsum(eye(4), [.mu^], v[.mu^])\n");
}

/// A rank-1 metric cannot be square, so it fails the same sentence.
#[test]
fn a_vector_metric_is_refused() {
    rejects(
        "v = [1.0, 2.0]\nL = metricsum([1.0, 2.0], [.mu^], v[.mu^])\n",
        "rank-2 array of scalars",
    );
}

#[test]
fn a_scalar_metric_is_refused() {
    rejects(
        "v = [1.0, 2.0]\nL = metricsum(1.0, [.mu^], v[.mu^])\n",
        "rank-2 array of scalars",
    );
}

/// `aggregate` takes a reduction head, not a metric, so the check must not
/// reach it.
#[test]
fn aggregate_is_untouched() {
    accepts("A = rowstack([[1.0, 2.0], [3.0, 4.0]])\nt = aggregate(sum, [], A[.i, .i])\n");
}
