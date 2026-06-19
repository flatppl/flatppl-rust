//! Shape-resolution and inference-level coverage tests — spec §17.1 / §17.3.
//!
//! All assertions use exact substrings observed from the engine (discovery
//! phase ran before any assertion was written). Tests marked `#[ignore]` name
//! a candidate-bug: the engine's actual output diverges from what the spec
//! requires.

use flatppl_infer::{Level, infer, infer_with};

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    flatppl_flatpir::write(&m)
}

fn ir_at(src: &str, level: Level) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer_with(&mut m, level);
    flatppl_flatpir::write(&m)
}

// ---- Shape resolution §17.1 ----

/// Multiply, subtract, and negate of a fixed integer ref are all resolved
/// to concrete dims at `Level::Shape`.
#[test]
fn shape_resolver_sub_mul_neg() {
    // J * J: 4 × 4 = 16
    let out = ir("J = 4\nx ~ iid(Normal(0.0,1.0), J * J)");
    assert!(
        out.contains("(%array 1 (16) (%scalar real))"),
        "J*J should resolve to dim 16, got:\n{out}"
    );

    // J - 2: 5 − 2 = 3
    let out = ir("J = 5\nx ~ iid(Normal(0.0,1.0), J - 2)");
    assert!(
        out.contains("(%array 1 (3) (%scalar real))"),
        "J-2 should resolve to dim 3, got:\n{out}"
    );

    // neg(J)+10: −4 + 10 = 6
    let out = ir("J = 4\nx ~ iid(Normal(0.0,1.0), neg(J)+10)");
    assert!(
        out.contains("(%array 1 (6) (%scalar real))"),
        "neg(J)+10 should resolve to dim 6, got:\n{out}"
    );
}

/// A two-hop fixed ref (A → B → iid count) resolves transitively.
#[test]
fn shape_resolver_multi_hop_ref() {
    let out = ir("A = 4\nB = A + 1\nx ~ iid(Normal(0.0,1.0), B)");
    assert!(
        out.contains("(%array 1 (5) (%scalar real))"),
        "A=4, B=A+1 should resolve dim to 5, got:\n{out}"
    );
}

/// A parameterized (non-fixed) integer ref cannot be resolved — the dim stays
/// `%dynamic`.
#[test]
fn shape_resolver_parameterized_ref_stays_dynamic() {
    let out = ir("n = elementof(nonnegintegers)\nx ~ iid(Normal(0.0,1.0), n)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))"),
        "parameterized n should stay %dynamic, got:\n{out}"
    );
}

/// A large-integer overflow (3e9 × 3e9 overflows i64 shape arithmetic) must
/// not panic — the engine falls back to `%dynamic`.
#[test]
fn shape_resolver_overflow_is_dynamic() {
    let out = ir("A = 3000000000\nB = A * A\nx ~ iid(Normal(0.0,1.0), B)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))"),
        "overflowed product should fall back to %dynamic (no panic), got:\n{out}"
    );
}

/// `cartpow(reals, N)` where `N` is a fixed literal resolves the element type
/// and dim at `Level::Shape`.
#[test]
fn cartpow_size_from_fixed_ref() {
    let out = ir_at("N = 3\nx = elementof(cartpow(reals, N))", Level::Shape);
    assert!(
        out.contains("(%array 1 (3) (%scalar real))"),
        "cartpow(reals, N=3) should infer a (3)-shaped real array, got:\n{out}"
    );
}

/// `stdsimplex(N)` where `N` is a fixed binding ref: at `Level::Shape` the
/// resolver fills the dim, and `elementof(stdsimplex(N))` is a length-N real
/// vector — the (N-1)-simplex embedded in ℝᴺ (§03 "Standard simplex"). The
/// ≥0 / sum-to-1 constraint lives in the value-set slot (`(stdsimplex 3)`), so
/// the element TYPE is a rank-1 real array, mirroring `cartpow(reals, N)`.
#[test]
fn stdsimplex_size_from_fixed_ref() {
    let out = ir_at("N = 3\nx = elementof(stdsimplex(N))", Level::Shape);
    assert!(
        out.contains("(%array 1 (3) (%scalar real))"),
        "elementof(stdsimplex(3)) should be a length-3 real array, got:\n{out}"
    );
    // The simplex constraint rides in the value-set slot.
    assert!(
        out.contains("(stdsimplex 3)"),
        "value-set slot should be (stdsimplex 3), got:\n{out}"
    );
}

// ---- Inference levels §17.3 ----

/// At `Level::Shape` (the maximal level), a draw from `Normal` carries both
/// a type annotation and a normalized measure — the level is additive.
#[test]
fn level_shape_is_additive() {
    let out = ir_at("x ~ Normal(0.0,1.0)", Level::Shape);
    assert!(
        out.contains("%normalized"),
        "Level::Shape must include mass annotation, got:\n{out}"
    );
    assert!(
        out.contains("reals"),
        "Level::Shape must include value-set annotation, got:\n{out}"
    );
    assert!(
        out.contains("(%scalar real)"),
        "Level::Shape must annotate scalar-real type, got:\n{out}"
    );
}

/// Shape resolution is skipped at `Level::Normalization` (dim stays `%dynamic`)
/// but active at `Level::Shape` (dim resolves to the literal).
#[test]
fn level_normalization_does_not_resolve_dims() {
    let src = "J = 8\nx ~ iid(Normal(0.0,1.0), J)";

    let norm_out = ir_at(src, Level::Normalization);
    assert!(
        norm_out.contains("(%array 1 (%dynamic) (%scalar real))"),
        "Level::Normalization must leave J's dim as %dynamic, got:\n{norm_out}"
    );

    let shape_out = ir_at(src, Level::Shape);
    assert!(
        shape_out.contains("(%array 1 (8) (%scalar real))"),
        "Level::Shape must resolve J=8 into dim (8), got:\n{shape_out}"
    );
}

// ---- Gap documentation: measure ops ----

/// All of `restrict`, `pushfwd`, `totalmass`, `superpose`, `Dirac`, `kchain`
/// parse without error and are honestly deferred (no type rule yet). Each
/// binding gets `%deferred` rather than a wrong type or a panic.
#[test]
fn unimplemented_measure_ops_are_deferred() {
    // restrict
    let out = ir("x = restrict(Normal(0.0,1.0), interval(0.0,1.0))");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (restrict"),
        "restrict: expected %deferred gap, got:\n{out}"
    );

    // pushfwd
    let out = ir("f = fn(_ * 2.0)\nx = pushfwd(f, Normal(0.0,1.0))");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (pushfwd"),
        "pushfwd: expected %deferred gap, got:\n{out}"
    );

    // totalmass
    let out = ir("x = totalmass(Normal(0.0,1.0))");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (totalmass"),
        "totalmass: expected %deferred gap, got:\n{out}"
    );

    // superpose
    let out = ir("x = superpose(Normal(0.0,1.0), Normal(1.0,1.0))");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (superpose"),
        "superpose: expected %deferred gap, got:\n{out}"
    );

    // Dirac
    let out = ir("x = Dirac(0.0)");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (Dirac"),
        "Dirac: expected %deferred gap, got:\n{out}"
    );

    // kchain
    let out = ir("x = kchain(Normal(0.0,1.0), Normal(0.0,1.0))");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (kchain"),
        "kchain: expected %deferred gap, got:\n{out}"
    );
}

// ---- Gap documentation: linear-algebra functions ----

/// `det`, `transpose`, `eye`, and `linspace` all parse and are honestly
/// deferred — no type rule yet, each binding gets `%deferred` without panic.
#[test]
fn unimplemented_functions_are_deferred() {
    // det
    let out = ir("x = det([[1.0, 0.0], [0.0, 1.0]])");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (det"),
        "det: expected %deferred gap, got:\n{out}"
    );

    // transpose
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\nx = transpose(A)");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (transpose"),
        "transpose: expected %deferred gap, got:\n{out}"
    );

    // eye
    let out = ir("x = eye(3)");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (eye"),
        "eye: expected %deferred gap, got:\n{out}"
    );

    // linspace
    let out = ir("x = linspace(0.0, 1.0, 5)");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (linspace"),
        "linspace: expected %deferred gap, got:\n{out}"
    );
}
