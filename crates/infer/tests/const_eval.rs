//! The value domain of the inference trace (engine-concepts §17 / §17.1): the
//! demand-driven const-eval of fixed-phase expressions at shape positions. This
//! is the "(mini-)interpreter" that resolves shapes the spec lets depend on
//! fixed-phase values (`zeros(sizeof(M))`, `iid(M, prod(sizeof(M)))`).
//!
//! Two properties are pinned here:
//!  - **generalization beyond integers** — `sizeof` yields a fixed *vector*, and
//!    reductions over it (`prod`/`sum`) feed a shape;
//!  - **the op-gap / `%dynamic` distinction** (§17.1 "the fixed-value boundary"):
//!    a genuinely-unknowable value stays `%dynamic` (no error), but a fixed op
//!    the evaluator cannot fold is a LOUD diagnostic, never a silent `%dynamic`.
//!
//! Per the testing conventions (ARCHITECTURE): distinct, non-coincidental magic
//! dims per axis (2 vs 5 — a swapped/mis-derived axis must show), and assertions
//! encode the spec rule, not merely today's output.

use flatppl_infer::{Diagnostic, Level, infer_with};

fn ir_at(src: &str, level: Level) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer_with(&mut m, level);
    flatppl_flatpir::write(&m)
}

fn diags_at(src: &str, level: Level) -> Vec<Diagnostic> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer_with(&mut m, level)
}

// ---- generalization: sizeof → fixed vector, feeding a multi-dim shape ----

/// `sizeof(M)` of a static 2×5 matrix is the fixed vector `[2, 5]`;
/// `zeros(sizeof(M))` must recover the 2×5 shape (the value flows through
/// `count_dims`, which now consumes a const-evaluated size vector, not just a
/// syntactic `[…]` literal). The first slice (integers only) left this rank-1
/// `%dynamic`.
#[test]
fn sizeof_value_resolves_a_multidim_shape() {
    let out = ir_at("M = fill(0.0, [2, 5])\nz = zeros(sizeof(M))", Level::Shape);
    assert!(
        out.contains("(%bind z (%meta ((%array 2 (2 5) (%scalar real))"),
        "zeros(sizeof(2x5)) should resolve to a 2x5 real matrix, got:\n{out}"
    );
}

/// A reduction over a fixed shape vector resolves a scalar dim: `prod(sizeof(M))`
/// = 2·5 = 10 (the total element count), driving an `iid` length. Exercises the
/// value-domain past scalars — a `prod` folding a `FixedValue::Vec`.
#[test]
fn prod_of_sizeof_resolves_iid_count() {
    let out = ir_at(
        "M = fill(0.0, [2, 5])\nx ~ iid(Normal(0.0, 1.0), prod(sizeof(M)))",
        Level::Shape,
    );
    assert!(
        out.contains("(%array 1 (10) (%scalar real))"),
        "iid(_, prod(sizeof(2x5))) should resolve to a length-10 array, got:\n{out}"
    );
}

// ---- the op-gap vs %dynamic distinction (§17.1) ----

/// A fixed op the evaluator cannot fold at a shape position is a LOUD diagnostic
/// (never a silent `%dynamic` that hides the gap). `div` is fixed-phase and
/// computable in principle but not (yet) in the const-eval table, so it must
/// report — mentioning the op and that a shape needs it.
///
/// (When `div` joins the const-eval table this flips to assert resolution — a
/// deliberate gap-documenting test, per the testing conventions.)
#[test]
fn op_gap_at_shape_position_is_a_loud_diagnostic() {
    let ds = diags_at("x ~ iid(Normal(0.0, 1.0), div(6, 2))", Level::Shape);
    assert!(
        ds.iter()
            .any(|d| d.message.contains("div") && d.message.contains("shape")),
        "an unfoldable fixed op at a shape position should emit a loud diagnostic, got: {:?}",
        ds.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

/// A genuinely-unknowable size stays `%dynamic` with NO diagnostic — the
/// op-gap error must not fire for a parameterized (non-fixed) ancestor. This is
/// the other side of the §17.1 boundary: `%dynamic` is legitimate here.
#[test]
fn parameterized_size_stays_dynamic_without_a_gap_error() {
    let src = "n = elementof(nonnegintegers)\nx ~ iid(Normal(0.0, 1.0), n)";
    let out = ir_at(src, Level::Shape);
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))"),
        "parameterized n should stay %dynamic, got:\n{out}"
    );
    let ds = diags_at(src, Level::Shape);
    assert!(
        !ds.iter().any(|d| d.message.contains("shape")),
        "a legitimately-dynamic size must NOT raise an op-gap error, got: {:?}",
        ds.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

/// `%dynamic` wins over a gap: if a size expression mixes a genuinely-dynamic
/// operand with an unfoldable op, the result is `%dynamic` with NO error — we
/// only nag about an op-gap when the value would otherwise be fully known. Here
/// `add(n, div(6, 2))` has a parameterized `n` (dynamic) and an unfoldable
/// `div` (gap); dynamic dominates, so no diagnostic fires.
#[test]
fn dynamic_dominates_a_gap_no_error() {
    let src = "n = elementof(nonnegintegers)\nx ~ iid(Normal(0.0, 1.0), add(n, div(6, 2)))";
    let ds = diags_at(src, Level::Shape);
    assert!(
        !ds.iter().any(|d| d.message.contains("shape")),
        "a dynamic operand must suppress the op-gap error, got: {:?}",
        ds.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let out = ir_at(src, Level::Shape);
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))"),
        "add(dynamic, gap) should stay %dynamic, got:\n{out}"
    );
}
