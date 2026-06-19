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

// ---- measure ops ----

/// `Dirac(value)` is the point-mass probability measure at `value` (spec §06):
/// a normalized measure over `value`'s type (works for any variate type — a
/// scalar gives a real-domain measure, a record gives a record-domain measure).
/// `value` may be the named kwarg or positional.
#[test]
fn dirac_infers_a_normalized_point_mass() {
    let out = ir("d = Dirac(value = 3.0)");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized))")
            && out.contains("(Dirac"),
        "Dirac(value=3.0) should be a normalized real measure, got:\n{out}"
    );
    // Positional form too.
    let out = ir("d = Dirac(0.0)");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized))"),
        "Dirac(0.0) should be a normalized real measure, got:\n{out}"
    );
    // Record variate → record-domain measure.
    let out = ir("d = Dirac(value = record(a = 1.0))");
    assert!(
        out.contains("(%measure (%domain (%record (a (%scalar real)))) (%mass %normalized))"),
        "Dirac over a record should be a record-domain measure, got:\n{out}"
    );
}

/// The deterministic composition / structural-disintegration ops infer, reusing
/// existing types: `scan` → a value `array[lengthof(xs)]` of the accumulator
/// type; `fchain` → a `function` with f1's input signature; `disintegrate` → a
/// `(forward_kernel, marginal)` tuple, with mass classes following the joint
/// (a probability joint → Markov kernel + probability marginal).
#[test]
fn scan_fchain_disintegrate_infer() {
    let out = ir("xs = [1.0, 2.0, 3.0]\nf = (acc, x) -> acc + x\ns = scan(f, 0.0, xs)");
    assert!(
        out.contains("(%array 1 (3) (%scalar real))") && out.contains("(scan"),
        "scan should infer a length-3 real value array, got:\n{out}"
    );
    let out = ir("f1 = x -> x + 1.0\nf2 = y -> y * 2.0\nc = fchain(f1, f2)");
    assert!(
        out.contains("(%function (%inputs x))") && out.contains("(fchain"),
        "fchain should infer a function with f1's inputs, got:\n{out}"
    );
    let out = ir("a ~ Normal(0.0, 2.0)\n\
                  b ~ Normal(a, 1.0)\n\
                  jm = lawof(record(a = a, b = b))\n\
                  fk = disintegrate([\"b\"], jm)");
    assert!(
        out.contains("(%tuple (%kernel (%inputs ) (%mass %normalized)) (%measure (%domain %deferred) (%mass %normalized)))")
            && out.contains("(disintegrate"),
        "disintegrate of a probability joint should be a (normalized kernel, normalized marginal) tuple, got:\n{out}"
    );
}

/// The Kleisli / trajectory ops infer a `(%measure …)` type (spec §06), reusing
/// the existing measure type — no new type kind. `markovchain`/`kscan` give a
/// length-resolved trajectory domain (`array[n]` / `array[lengthof(xs)]` of the
/// state type) and stay normalized when the step kernel is a Markov kernel;
/// `kchain` is a measure whose output variate isn't statically extractable
/// (deferred domain) but is normalized when its components are.
#[test]
fn kernel_chain_ops_infer_measures() {
    // markovchain: n=100 folds, state is real → array[100] real, normalized.
    let out = ir("f = x -> Normal(x, 1.0)\ntraj = markovchain(f, 0.0, 100)");
    assert!(
        out.contains("(%measure (%domain (%array 1 (100) (%scalar real))) (%mass %normalized))")
            && out.contains("(markovchain"),
        "markovchain should be a normalized measure over array[100] real, got:\n{out}"
    );
    // kscan: trajectory length = lengthof(xs) = 3.
    let out =
        ir("dts = [0.01, 0.02, 0.015]\ng = (x, dt) -> Normal(x, dt)\ntr = kscan(g, 0.0, dts)");
    assert!(
        out.contains("(%measure (%domain (%array 1 (3) (%scalar real))) (%mass %normalized))")
            && out.contains("(kscan"),
        "kscan should be a normalized measure over array[3] real, got:\n{out}"
    );
    // kchain: a measure (not %deferred), normalized when components are; domain
    // is deferred (last variate not statically extractable).
    let out = ir("lambda ~ Gamma(2.0, 1.0)\n\
                  prior = lawof(record(lambda = lambda))\n\
                  fk = kernelof(record(y = lambda), lambda = lambda)\n\
                  pp = kchain(prior, fk)");
    assert!(
        out.contains("(%measure (%domain %deferred) (%mass %normalized))")
            && out.contains("(kchain"),
        "kchain should be a normalized measure with a deferred domain, got:\n{out}"
    );
    // jointchain: previously %deferred-typed (so its existing mass arm was dead);
    // typing it as a measure activates that arm — a joint chain of a base measure
    // and Markov kernels is a probability measure.
    let out = ir("lambda ~ Gamma(2.0, 1.0)\n\
                  m0 = lawof(record(a = lambda))\n\
                  k = kernelof(record(b = lambda), a = lambda)\n\
                  j = jointchain(m0, k)");
    assert!(
        out.contains("(%measure (%domain %deferred) (%mass %normalized))")
            && out.contains("(jointchain"),
        "jointchain should be a normalized measure with a deferred domain, got:\n{out}"
    );
}

/// Domain-preserving measure-algebra ops infer a `(%measure …)` type with the
/// spec-§06 mass class: `restrict`/`superpose` are sub-/sum-measures (finite,
/// not normalized); `locscale`/`pushfwd` preserve total mass (a probability
/// measure stays normalized).
#[test]
fn domain_preserving_measure_ops_infer() {
    let out = ir("x = restrict(Normal(0.0, 1.0), interval(0.0, 1.0))");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %finite))")
            && out.contains("(restrict"),
        "restrict should infer a finite real measure, got:\n{out}"
    );
    let out = ir("x = superpose(Normal(0.0, 1.0), Normal(1.0, 1.0))");
    assert!(
        out.contains("(%mass %finite)") && out.contains("(superpose"),
        "superpose of two probability measures should be finite (mass 2), got:\n{out}"
    );
    let out = ir("x = locscale(Normal(0.0, 1.0), 2.0, 3.0)");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized))")
            && out.contains("(locscale"),
        "locscale of a probability measure stays normalized, got:\n{out}"
    );
    let out = ir("f = fn(_ * 2.0)\nx = pushfwd(f, Normal(0.0, 1.0))");
    assert!(
        out.contains("(%mass %normalized)") && out.contains("(pushfwd"),
        "pushfwd preserves total mass → normalized, got:\n{out}"
    );
}

// ---- linear-algebra functions now inferred via the catalogue ----

/// `transpose`/`adjoint` preserve rank and element kind: a matrix's two dims
/// swap; a vector's transpose stays a rank-1 transposed vector (spec §07: "the
/// transpose of a vector is a transposed vector, not a single-row matrix").
#[test]
fn transpose_preserves_rank_and_element_kind() {
    // matrix (nested vectors) → matrix, element kind preserved
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\nx = transpose(A)");
    assert!(
        out.contains("(%array 1 (2) (%array 1 (2) (%scalar real)))") && out.contains("(transpose"),
        "transpose(2x2 matrix) should be a 2x2 real matrix, got:\n{out}"
    );
    // vector → rank-1 (a transposed vector is NOT a single-row matrix)
    let out = ir("v = [1.0, 2.0, 3.0]\nx = adjoint(v)");
    assert!(
        out.contains("(%array 1 (") && out.contains("(adjoint"),
        "adjoint(vector) should stay a rank-1 array, got:\n{out}"
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

// ---- Round 4: more linear-algebra / vector ops via the catalogue ----

/// `boolean(x)` → boolean scalar; `quadform(A, x)` → scalar of A's element kind.
#[test]
fn boolean_and_quadform_infer_scalars() {
    let out = ir("b = true\nx = boolean(b)");
    assert!(
        out.contains("(%meta ((%scalar boolean) %fixed booleans) (boolean"),
        "boolean should infer a boolean scalar, got:\n{out}"
    );
    let out = ir("A = [[1.0, 0.0], [0.0, 1.0]]\nv = [1.0, 2.0]\nx = quadform(A, v)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (quadform"),
        "quadform(real) should infer a real scalar, got:\n{out}"
    );
}

/// Gram / outer / block matrix constructors infer a matrix with the element
/// kind preserved from the argument.
#[test]
fn matrix_constructors_infer_matrices() {
    for (op, src) in [
        ("row_gram", "A = [[1.0, 2.0], [3.0, 4.0]]\nx = row_gram(A)"),
        ("col_gram", "A = [[1.0, 2.0], [3.0, 4.0]]\nx = col_gram(A)"),
        ("self_outer", "v = [1.0, 2.0]\nx = self_outer(v)"),
        (
            "colstack",
            "vs = [[1.0, 2.0], [3.0, 4.0]]\nx = colstack(vs)",
        ),
    ] {
        let out = ir(src);
        assert!(
            out.contains("(%array 2 (%dynamic %dynamic) (%scalar real))")
                && out.contains(&format!("({op}")),
            "{op} should infer a real matrix, got:\n{out}"
        );
    }
}

/// Vector-result constructors: `onehot`/`conv` → real, `bincounts` → integer.
#[test]
fn more_vector_results_infer() {
    let out = ir("x = onehot(1, 3)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(onehot"),
        "onehot should infer a real vector, got:\n{out}"
    );
    let out = ir("b = [0.0, 1.0, 2.0]\nd = [0.5, 1.5]\nx = bincounts(b, d)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar integer))") && out.contains("(bincounts"),
        "bincounts should infer an integer vector, got:\n{out}"
    );
}

/// `lxor(a, b)` is a boolean op like `land`/`lor` (spec §07).
#[test]
fn lxor_infers_boolean() {
    let out = ir("a = true\nb = false\nx = lxor(a, b)");
    assert!(
        out.contains("(%meta ((%scalar boolean) %fixed booleans) (lxor"),
        "lxor should infer a boolean scalar, got:\n{out}"
    );
}

/// `linsolve(A, b)` → x with b's type; `polynomial`/`bernstein` evaluated at a
/// scalar x → real scalar (shaped like the eval point, spec §07).
#[test]
fn linsolve_and_basis_evals_infer() {
    let out = ir("A = [[1.0, 0.0], [0.0, 1.0]]\nb = [1.0, 2.0]\nx = linsolve(A, b)");
    assert!(
        out.contains("(%array 1 (2) (%scalar real))") && out.contains("(linsolve"),
        "linsolve should infer b's vector type, got:\n{out}"
    );
    let out = ir("c = [1.0, 2.0, 3.0]\nx = polynomial(c, 0.5)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (polynomial"),
        "polynomial at scalar x should infer a real scalar, got:\n{out}"
    );
}

// ---- Value-shaped array constructors (spec §07, §17.1 shape resolution) ----

/// `zeros`/`ones` are real arrays whose RANK comes from the size argument's
/// value: a scalar size → vector, a vector size → matrix. Dims resolve at
/// `Level::Shape`.
#[test]
fn zeros_ones_rank_from_size_value() {
    let out = ir_at("x = zeros(3)", Level::Shape);
    assert!(
        out.contains("(%array 1 (3) (%scalar real))") && out.contains("(zeros"),
        "zeros(3) should be a length-3 real vector, got:\n{out}"
    );
    let out = ir_at("x = zeros([2, 3])", Level::Shape);
    assert!(
        out.contains("(%array 2 (2 3) (%scalar real))") && out.contains("(zeros"),
        "zeros([2,3]) should be a 2x3 real matrix, got:\n{out}"
    );
    let out = ir_at("x = ones(4)", Level::Shape);
    assert!(
        out.contains("(%array 1 (4) (%scalar real))") && out.contains("(ones"),
        "ones(4) should be a length-4 real vector, got:\n{out}"
    );
}

/// `fill(x, size)` takes the element kind from the fill value; `array(data,
/// size, …)` from the data — both shaped by `size`.
#[test]
fn fill_and_array_element_kind_and_shape() {
    let out = ir_at("x = fill(2, 3)", Level::Shape);
    assert!(
        out.contains("(%array 1 (3) (%scalar integer))") && out.contains("(fill"),
        "fill(2, 3) should be a length-3 integer vector, got:\n{out}"
    );
    let out = ir_at(
        "d = [1.0, 2.0, 3.0, 4.0]\nx = array(d, [2, 2])",
        Level::Shape,
    );
    assert!(
        out.contains("(%array 2 (2 2) (%scalar real))") && out.contains("(array"),
        "array(d, [2,2]) should be a 2x2 real matrix, got:\n{out}"
    );
}

/// `cat` of scalars → a rank-1 vector of that kind; `tile` preserves the
/// argument's rank and element kind (only sizes become dynamic).
#[test]
fn cat_and_tile_preserve_kind_and_rank() {
    let out = ir("x = cat(1.0, 2.0, 3.0)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(cat"),
        "cat(scalars) should be a rank-1 real vector, got:\n{out}"
    );
    let out = ir("a = [1.0, 2.0]\nx = tile(a, 3)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(tile"),
        "tile(vector) should stay a rank-1 real vector, got:\n{out}"
    );
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\nx = tile(A, [2, 2])");
    assert!(
        out.contains("(%array 1 (%dynamic) (%array 1 (%dynamic) (%scalar real)))")
            && out.contains("(tile"),
        "tile(matrix) should stay a rank-2 (nested) real matrix, got:\n{out}"
    );
}

/// `reduce(f, xs)` folds to xs's element type (spec §07: f returns the element
/// type); `filter(pred, data)` keeps data's type/rank with a dynamic length.
#[test]
fn reduce_and_filter_infer() {
    let out = ir("xs = [1.0, 2.0, 3.0]\nx = reduce(fn(_ + 1.0), xs)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (reduce"),
        "reduce over a real vector should infer a real scalar, got:\n{out}"
    );
    let out = ir("d = [1.0, 2.0, 3.0]\ny = filter(fn(_ in interval(0.0, 2.0)), d)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(filter"),
        "filter of a real vector should stay a real vector (dynamic length), got:\n{out}"
    );
}

/// `qr(A)` (spec §07) returns `record(Q, R)` — both matrices with A's element
/// kind, via the RON catalogue's `ResultSig::Record` (field names interned
/// through the lowering context). The reusable record-valued result path.
#[test]
fn qr_infers_a_record_of_matrices() {
    let out = ir("A = [[1.0, 0.0], [0.0, 1.0]]\nd = qr(A)");
    assert!(
        out.contains("(%record (Q (%array 2 (%dynamic %dynamic) (%scalar real))) (R (%array 2 (%dynamic %dynamic) (%scalar real))))")
            && out.contains("(qr"),
        "qr should infer record(Q: matrix, R: matrix), got:\n{out}"
    );
}

/// `aggregate(f, output_axes, expr)` / `metricsum` (spec §04) are einsum-style
/// reductions: the result rank is the number of output axes, the element kind
/// comes from the reduced expr, and empty output axes give a scalar.
#[test]
fn aggregate_rank_from_output_axes() {
    // Two output axes → rank-2 real array (matrix product).
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\n\
                  B = [[5.0, 6.0], [7.0, 8.0]]\n\
                  C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])");
    assert!(
        out.contains("(%array 2 (%dynamic %dynamic) (%scalar real))") && out.contains("(aggregate"),
        "aggregate over 2 axes should be a rank-2 real array, got:\n{out}"
    );
    // Empty output axes → scalar (full contraction).
    let out = ir("A = [1.0, 2.0]\nB = [3.0, 4.0]\ns = aggregate(sum, [], A[.i] * B[.i])");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (aggregate"),
        "aggregate over no axes should be a real scalar, got:\n{out}"
    );
    // One output axis → rank-1; metricsum shares the rule.
    let out = ir("A = [[1.0, 2.0], [3.0, 4.0]]\nV = aggregate(var, [.j], A[.i, .j])");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(aggregate"),
        "aggregate over 1 axis should be a rank-1 real array, got:\n{out}"
    );
}
