//! The `expr` half of spec §04 "Metric-aware Einstein summation" /
//! "Expression restrictions": "`metric` itself and all arrays indexed with
//! co-/contravariant axis names in `expr` must be arrays of scalars."
//!
//! §03 "Arrays" keeps a nested literal out of that type: "Vectors of vectors
//! are not interpreted as matrices implicitly, but can be turned into matrices
//! explicitly using `rowstack` or `colstack`." The restriction names
//! co-/contravariant axis names only, so a neutral `aggregate` axis over a
//! nested array is a different question and stays clean here.

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

const NESTED: &str = "g = eye(2)\nL1 = [[1.0, 0.0], [0.0, 1.0]]\nL2 = [[1.0, 0.0], [0.0, 1.0]]\n";

#[test]
fn a_vector_of_vectors_indexed_by_variance_axes_is_refused() {
    rejects(
        &format!("{NESTED}T = metricsum(g, [.mu^, .beta_], L1[.mu^, .nu_] * L2[.nu^, .beta_])\n"),
        "vector of vectors",
    );
}

/// The diagnostic must name the lift, since the fix is one call away, and both
/// lifts, since the storage order is the user's call.
#[test]
fn the_refusal_offers_rowstack_and_colstack() {
    let messages = errors(&format!(
        "{NESTED}T = metricsum(g, [.mu^, .beta_], L1[.mu^, .nu_] * L2[.nu^, .beta_])\n"
    ));
    assert!(
        messages
            .iter()
            .any(|m| m.contains("rowstack") && m.contains("colstack")),
        "the refusal must offer both lifts; got {messages:?}"
    );
}

/// The refusal points at the offending container, not at the whole `metricsum`.
#[test]
fn the_refusal_is_located_on_the_container() {
    let src =
        format!("{NESTED}T = metricsum(g, [.mu^, .beta_], L1[.mu^, .nu_] * L2[.nu^, .beta_])\n");
    let mut module = flatppl_syntax::parse(&src).unwrap();
    let diags = infer(&mut module);
    let nodes: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.node)
        .collect();
    assert!(
        nodes.iter().all(Option::is_some),
        "the refusal must carry a node; got {nodes:?}"
    );
    let named: Vec<_> = nodes
        .iter()
        .flatten()
        .filter_map(|&n| match module.node(n) {
            flatppl_core::Node::Ref(r) => Some(module.resolve(r.name).to_string()),
            _ => None,
        })
        .collect();
    assert!(
        named.iter().any(|n| n == "L1" || n == "L2"),
        "the refusal must point at the container binding; got {named:?}"
    );
}

#[test]
fn the_explicit_lift_is_accepted() {
    accepts(
        "g = eye(2)\nA = rowstack([[1.0, 0.0], [0.0, 1.0]])\n\
         T = metricsum(g, [.mu^, .beta_], A[.mu^, .nu_] * A[.nu^, .beta_])\n",
    );
}

/// A flat vector is an array of scalars, so a rank-1 tensor access is fine.
#[test]
fn a_flat_vector_is_accepted() {
    accepts("g = eye(2)\nv = [1.0, 2.0]\nL = metricsum(g, [.mu^], v[.mu^])\n");
}

/// A container carrying one variance-marked selector is "indexed with
/// co-/contravariant axis names", whatever the other selectors are.
#[test]
fn a_mixed_index_still_refuses_the_nested_container() {
    rejects(
        "g = eye(2)\nL1 = [[1.0, 0.0], [0.0, 1.0]]\n\
         T = metricsum(g, [.mu^], L1[.mu^, 1])\n",
        "vector of vectors",
    );
}

/// A vector of transposed vectors is not a vector of vectors — §03 keeps the
/// two types apart — but it is still not an array of scalars.
#[test]
fn a_vector_of_transposed_vectors_is_refused() {
    rejects(
        "A = [transpose([1.0, 2.0]), transpose([3.0, 4.0])]\n\
         L = metricsum(eye(2), [.mu^], A[.mu^])\n",
        "must be an array of scalars",
    );
}

/// The restriction covers co-/contravariant axis names only. A neutral
/// `aggregate` axis over a nested array is outside it.
#[test]
fn a_neutral_aggregate_axis_over_a_nested_array_is_untouched() {
    accepts("L1 = [[1.0, 0.0], [0.0, 1.0]]\nt = aggregate(sum, [], L1[.i, .j])\n");
}

/// `aggregate` bodies are not `metricsum` bodies; the check must not reach them.
#[test]
fn aggregate_is_untouched() {
    accepts("A = rowstack([[1.0, 2.0], [3.0, 4.0]])\nt = aggregate(sum, [], A[.i, .i])\n");
}
