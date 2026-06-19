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

/// `restrict`, `pushfwd`, `superpose`, `Dirac`, `kchain` parse without error
/// and are honestly deferred (no type rule yet — measure-algebra transforms the
/// engine has no evaluator for). Each binding gets `%deferred` rather than a
/// wrong type or a panic. (`totalmass` is no longer here — it now infers a real
/// scalar; see `totalmass_infers_a_real_scalar`.)
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

/// `transpose` still parses and is honestly deferred — no type rule yet (its
/// result rank depends on whether the argument is a vector or a matrix, which
/// the catalogue cannot yet branch on). (`det`/`eye`/`linspace` are no longer
/// here — they now infer via the catalogue; see the tests below.)
#[test]
fn unimplemented_functions_are_deferred() {
    // transpose
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\nx = transpose(A)");
    assert!(
        out.contains("(%meta (%deferred %fixed %unknown) (transpose"),
        "transpose: expected %deferred gap, got:\n{out}"
    );
}

// ---- Previously-deferred §07 functions now inferred via the catalogue ----
// These were %deferred gaps; each is now a catalogue row (no structural arm),
// so the binding carries a real type instead of the %deferred placeholder.

/// `identity(x)` returns its argument unchanged — the full type (array shape +
/// element) threads through via `ResultSig::SameAsArg`.
#[test]
fn identity_threads_the_argument_type() {
    let out = ir("a = [1.0, 2.0, 3.0]\nx = identity(a)");
    assert!(
        out.contains("(%array 1 (3) (%scalar real)) %fixed") && out.contains("(identity"),
        "identity should preserve the (3)-real-array type, got:\n{out}"
    );
}

/// `reverse(xs)` preserves the input vector's shape and element type.
#[test]
fn reverse_preserves_shape_and_element() {
    let out = ir("a = [1.0, 2.0]\nx = reverse(a)");
    assert!(
        out.contains("(%array 1 (2) (%scalar real))") && out.contains("(reverse"),
        "reverse should preserve the (2)-real-array type, got:\n{out}"
    );
}

/// `ifelse(cond, a, b)` is the common type of its two branches — `int`/`real`
/// promote to `real` (`ResultSig::CommonOf`).
#[test]
fn ifelse_is_the_common_type_of_its_branches() {
    let out = ir("c = true\nx = ifelse(c, 1, 2.0)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (ifelse"),
        "ifelse(c, 1, 2.0) should infer a real scalar, got:\n{out}"
    );
}

/// `real(x)` / `imag(x)` are real-valued regardless of input kind
/// (`ResultSig::RealOfArgShape`).
#[test]
fn real_imag_are_real_valued() {
    let out = ir("x = real(complex(1.0, 2.0))");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (real"),
        "real(complex) should infer a real scalar, got:\n{out}"
    );
    let out = ir("x = imag(complex(1.0, 2.0))");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (imag"),
        "imag(complex) should infer a real scalar, got:\n{out}"
    );
}

/// `det` / `trace` thread the matrix's element kind: a real matrix yields a
/// real scalar, a complex matrix a complex scalar (`ResultSig::ElemScalarKind`).
#[test]
fn det_infers_element_scalar_kind() {
    let out = ir("x = det([[1.0, 0.0], [0.0, 1.0]])");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (det"),
        "det(real matrix) should infer a real scalar, got:\n{out}"
    );
    let out = ir("A = [[complex(1.0, 2.0)]]\nx = det(A)");
    assert!(
        out.contains("(%scalar complex)") && out.contains("(det"),
        "det(complex matrix) should infer a complex scalar, got:\n{out}"
    );
    let out = ir("A = [[1.0, 0.0], [0.0, 1.0]]\nx = trace(A)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (trace"),
        "trace(real matrix) should infer a real scalar, got:\n{out}"
    );
}

/// Reductions infer a scalar: `var`/`std`/`logabsdet` are always real; over a
/// real array `maximum`/`minimum` are real too.
#[test]
fn real_scalar_reductions_infer() {
    for op in ["var", "std", "maximum", "minimum"] {
        let out = ir(&format!("a = [1.0, 2.0, 3.0]\nx = {op}(a)"));
        assert!(
            out.contains(&format!("(%meta ((%scalar real) %fixed reals) ({op}")),
            "{op} should infer a real scalar over a real array, got:\n{out}"
        );
    }
    let out = ir("x = logabsdet([[2.0, 0.0], [0.0, 2.0]])");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (logabsdet"),
        "logabsdet should infer a real scalar, got:\n{out}"
    );
}

/// `maximum`/`minimum` thread the element kind: an integer array yields an
/// integer scalar (`ResultSig::ElemScalarKind`), not a widened real.
#[test]
fn maximum_minimum_thread_element_kind() {
    for op in ["maximum", "minimum"] {
        let out = ir(&format!("a = [1, 2, 3]\nx = {op}(a)"));
        assert!(
            out.contains("(%scalar integer)") && out.contains(&format!("({op}")),
            "{op} over an integer array should infer an integer scalar, got:\n{out}"
        );
    }
}

// ---- Round 3: linear-algebra matrix/vector results via the catalogue ----

/// `eye(n)` infers a real (dynamic-dim) matrix.
#[test]
fn eye_infers_a_real_matrix() {
    let out = ir("x = eye(3)");
    assert!(
        out.contains("(%array 2 (%dynamic %dynamic) (%scalar real))") && out.contains("(eye"),
        "eye should infer a rank-2 real array, got:\n{out}"
    );
}

/// `inv` / `lower_cholesky` / `diagmat` infer a matrix whose element kind is
/// preserved from the argument — a complex matrix inverts to a complex matrix.
#[test]
fn matrix_maps_preserve_element_kind() {
    let out = ir("A = [[1.0, 0.0], [0.0, 1.0]]\nx = inv(A)");
    assert!(
        out.contains("(%array 2 (%dynamic %dynamic) (%scalar real))") && out.contains("(inv"),
        "inv(real matrix) should be a real matrix, got:\n{out}"
    );
    let out = ir("A = [[complex(1.0, 0.0)]]\nx = inv(A)");
    assert!(
        out.contains("(%array 2 (%dynamic %dynamic) (%scalar complex))") && out.contains("(inv"),
        "inv(complex matrix) should be a complex matrix, got:\n{out}"
    );
    let out = ir("A = [[4.0, 0.0], [0.0, 9.0]]\nx = lower_cholesky(A)");
    assert!(
        out.contains("(%array 2 (%dynamic %dynamic) (%scalar real))")
            && out.contains("(lower_cholesky"),
        "lower_cholesky(real PD) should be a real matrix, got:\n{out}"
    );
    let out = ir("v = [1.0, 2.0]\nx = diagmat(v)");
    assert!(
        out.contains("(%array 2 (%dynamic %dynamic) (%scalar real))") && out.contains("(diagmat"),
        "diagmat(real vector) should be a real matrix, got:\n{out}"
    );
}

/// Vector-result functions infer a rank-1 array with the right element kind:
/// `linspace` → real, `sizeof` → integer, `diag` → the matrix's element kind.
#[test]
fn vector_result_functions_infer() {
    let out = ir("x = linspace(0.0, 1.0, 5)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(linspace"),
        "linspace should infer a real vector, got:\n{out}"
    );
    let out = ir("v = [1.0, 2.0, 3.0]\nx = sizeof(v)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar integer))") && out.contains("(sizeof"),
        "sizeof should infer an integer vector, got:\n{out}"
    );
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\nx = diag(A)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(diag"),
        "diag(real matrix) should infer a real vector, got:\n{out}"
    );
}

/// `totalmass(M)` is the total mass as a real scalar (spec §06).
#[test]
fn totalmass_infers_a_real_scalar() {
    let out = ir("m = Normal(0.0, 1.0)\nx = totalmass(m)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (totalmass"),
        "totalmass should infer a real scalar, got:\n{out}"
    );
}
