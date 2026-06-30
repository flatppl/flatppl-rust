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
/// A lambda whose body BROADCASTS a distribution over its (placeholder) params
/// — `(nr, pr) -> Binomial.(nr, pr)` — classifies as a KERNEL, not a plain
/// function: broadcasting a distribution is a measure (an independent product)
/// even before the shape is known, so the reified body is a measure. Stochastic
/// broadcast-application `r ~ K.(rows, …)` then resolves to the per-row
/// trajectory (the call-site substitution supplies the concrete shape).
#[test]
fn broadcast_distribution_lambda_is_a_kernel() {
    let out = ir("n_data = [[5, 6], [7, 8]]\n\
                  p = [[0.1, 0.2], [0.3, 0.4]]\n\
                  K = (nr, pr) -> Binomial.(nr, pr)\n\
                  r ~ K.(n_data, p)");
    assert!(
        out.contains("(%bind K (%meta ((%kernel (%inputs nr pr)"),
        "lambda with a broadcast-distribution body should be a kernel, got:\n{out}"
    );
    assert!(
        out.contains("(%bind r (%meta ((%array 1 (2) (%array 1 (2) (%scalar integer)))"),
        "r ~ K.(n_data, p) should be a nested 2x2 integer trajectory, got:\n{out}"
    );
}

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
        out.contains("(%tuple (%kernel (%inputs a) (%mass %normalized)) (%measure (%domain (%record (a (%scalar real)))) (%mass %normalized)))")
            && out.contains("(disintegrate"),
        "disintegrate of a probability joint should be a (normalized kernel w/ inputs=a, marginal over record{{a}}) tuple, got:\n{out}"
    );
}

/// The Kleisli / trajectory ops infer a `(%measure …)` type (spec §06), reusing
/// the existing measure type — no new type kind. `markovchain`/`kscan` give a
/// length-resolved trajectory domain; `kchain` carries the LAST component's
/// concrete variate; `jointchain` carries ALL variates as a merged record (or a
/// named record in keyword form); record-state `markovchain`/`kscan` stay deferred.
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
    // kchain keeps the LAST component's variate (spec §06 Kleisli bind): fk's
    // output is record{y: real}.
    let out = ir("lambda ~ Gamma(2.0, 1.0)\n\
                  prior = lawof(record(lambda = lambda))\n\
                  fk = kernelof(record(y = lambda), lambda = lambda)\n\
                  pp = kchain(prior, fk)");
    assert!(
        out.contains("(%measure (%domain (%record (y (%scalar real)))) (%mass %normalized))")
            && out.contains("(kchain"),
        "kchain domain should be the last component's variate record{{y}}, got:\n{out}"
    );
    // jointchain keeps ALL variates: cat of m0's record{a} and k's record{b}.
    let out = ir("lambda ~ Gamma(2.0, 1.0)\n\
                  m0 = lawof(record(a = lambda))\n\
                  k = kernelof(record(b = lambda), a = lambda)\n\
                  j = jointchain(m0, k)");
    assert!(
        out.contains("(%measure (%domain (%record (a (%scalar real)) (b (%scalar real)))) (%mass %normalized))")
            && out.contains("(jointchain"),
        "jointchain domain should be cat of all variates record{{a, b}}, got:\n{out}"
    );

    // Keyword form names each component's variate (spec §06 keyword form).
    let out = ir("lambda ~ Gamma(2.0, 1.0)\n\
                  m0 = lawof(record(a = lambda))\n\
                  k = kernelof(record(b = lambda), a = lambda)\n\
                  j = jointchain(prior = m0, fwd = k)");
    assert!(
        out.contains("(%measure (%domain (%record (prior (%record (a (%scalar real)))) (fwd (%record (b (%scalar real)))))) (%mass %normalized))")
            && out.contains("(jointchain"),
        "jointchain keyword form should name each component's variate, got:\n{out}"
    );

    // Finite-base + Normalized-kernel ⇒ Finite result. The prior arm required
    // ALL components (including the base) to be Normalized, so a Finite base
    // with Normalized kernels yielded Unknown. The real bug was that the 2-arg
    // case happened to produce the correct answer for fully-Normalized inputs
    // only, and ignored the base-mass distinction — any kernel at index ≥ 1 with
    // a non-Normalized base would silently give Unknown. The fix separates
    // base_mass (component 0) from the kernel-Normalized check (components 1..n)
    // and generalises to any arity, adding keyword-form support in the same pass.
    // `superpose` of two probability measures is finite (non-normalized);
    // a Normalized kernel leaves the total mass class unchanged.
    let out = ir("lambda ~ Gamma(2.0, 1.0)\n\
                  m0 = lawof(record(a = lambda))\n\
                  finite_m = superpose(m0, m0)\n\
                  k = kernelof(record(b = lambda), a = lambda)\n\
                  j = jointchain(finite_m, k)");
    assert!(
        out.contains("(%mass %finite)") && out.contains("(jointchain"),
        "jointchain with Finite base + Normalized kernel should yield %finite, got:\n{out}"
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
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized))")
            && out.contains("(pushfwd"),
        "pushfwd domain = f's codomain (real), mass preserved → normalized, got:\n{out}"
    );
}

/// `pushfwd(f, M)` (spec §06) infers its domain as the CODOMAIN of `f` — found by
/// binding f's input to M's variate and reading f's body type — instead of the
/// old `%any` fallback. The codomain flows downstream: `iid(pareto, G)` is a real
/// array, not `any[]`.
#[test]
fn pushfwd_domain_is_f_codomain() {
    let out = ir("pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))\n\
                  a ~ iid(pareto, 2)");
    assert!(
        out.contains("(%bind pareto (%meta ((%measure (%domain (%scalar real))"),
        "pushfwd(exp reparam) should have a real domain, not %any, got:\n{out}"
    );
    assert!(
        out.contains("(%bind a (%meta ((%array 1 (2) (%scalar real))"),
        "iid of the pushforward should be a real vector, not any[], got:\n{out}"
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

/// `mul` (`*`) over matrices/vectors (spec §07 line 448): flat rank-2 matrix ×
/// vector → vector; matrix × matrix → matrix; a static inner-dimension mismatch
/// is a shape error (`%failed`). Scalar/scalar-array forms are covered elsewhere.
#[test]
fn matmul_matrix_vector_and_matrix_matrix() {
    // matrix[3,3] × vector[3] → vector[3] (the matrix-vector product)
    let out = ir(
        "X = rowstack([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])\n\
                  v = [1.0, 2.0, 3.0]\n\
                  m = X * v",
    );
    assert!(
        out.contains("(%bind m (%meta ((%array 1 (3) (%scalar real))"),
        "matrix·vector should be a real vector[3], got:\n{out}"
    );
    // matrix[2,3] × matrix[3,2] → matrix[2,2]
    let out = ir("A = rowstack([[1.0,2.0,3.0],[4.0,5.0,6.0]])\n\
                  B = rowstack([[1.0,2.0],[3.0,4.0],[5.0,6.0]])\n\
                  C = A * B");
    assert!(
        out.contains("(%bind C (%meta ((%array 2 (2 2) (%scalar real))"),
        "matrix·matrix should be a real matrix[2,2], got:\n{out}"
    );
    // inner-dimension mismatch ([2,3] × [2]) is a static shape error
    let out = ir("A = rowstack([[1.0,2.0,3.0],[4.0,5.0,6.0]])\n\
                  v = [1.0, 2.0]\n\
                  m = A * v");
    assert!(
        out.contains("(%bind m (%meta ((%failed"),
        "matrix·vector inner-dim mismatch should be %failed, got:\n{out}"
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
/// real array `maximum`/`minimum` are real too. `var`/`std` are non-negative
/// (a variance/standard deviation ≥ 0 — catalogue `result_set: NonNegReals`),
/// while `maximum`/`minimum` of a real array range over all reals.
#[test]
fn real_scalar_reductions_infer() {
    for op in ["var", "std"] {
        let out = ir(&format!("a = [1.0, 2.0, 3.0]\nx = {op}(a)"));
        assert!(
            out.contains(&format!("(%meta ((%scalar real) %fixed nonnegreals) ({op}")),
            "{op} should infer a non-negative real scalar, got:\n{out}"
        );
    }
    for op in ["maximum", "minimum"] {
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

/// `totalmass(M)` is the total mass as a real scalar (spec §06) — a mass is
/// non-negative, so its value-set is `nonnegreals` (catalogue `result_set`).
#[test]
fn totalmass_infers_a_real_scalar() {
    let out = ir("m = Normal(0.0, 1.0)\nx = totalmass(m)");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed nonnegreals) (totalmass"),
        "totalmass should infer a non-negative real scalar, got:\n{out}"
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

/// `cat` of n scalars → a rank-1 length-`n` vector of that kind (the count is
/// statically known); `tile` preserves the argument's rank and element kind
/// (only sizes become dynamic).
#[test]
fn cat_and_tile_preserve_kind_and_rank() {
    let out = ir("x = cat(1.0, 2.0, 3.0)");
    assert!(
        out.contains("(%array 1 (3) (%scalar real))") && out.contains("(cat"),
        "cat(3 scalars) should be a static length-3 real vector, got:\n{out}"
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
/// reductions: element from the reduced expr, empty axes → scalar, and the
/// result dims are the EXACT extents — each output axis is traced to the input
/// dimension it indexes in the body (`A[.i, .j]` → `.i` is A's flat dim 0).
#[test]
fn aggregate_resolves_exact_einsum_dims() {
    // Matrix product A:(2,3) · B:(3,4) → C[.i,.k] is (2,4): .i ← A dim0,
    // .k ← B dim1 (the contracted .j is gone).
    let out = ir("A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n\
                  B = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 8.0, 7.0, 6.0]]\n\
                  C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])");
    assert!(
        out.contains("(%array 2 (2 4) (%scalar real))") && out.contains("(aggregate"),
        "matmul aggregate should resolve to exact (2,4), got:\n{out}"
    );
    // Empty output axes → scalar (full contraction).
    let out = ir("A = [1.0, 2.0]\nB = [3.0, 4.0]\ns = aggregate(sum, [], A[.i] * B[.i])");
    assert!(
        out.contains("(%meta ((%scalar real) %fixed reals) (aggregate"),
        "aggregate over no axes should be a real scalar, got:\n{out}"
    );
    // var over axis .j of A:(2,3) → length-3 vector (.j ← A dim1); metricsum
    // shares the rule.
    let out = ir("A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\nV = aggregate(var, [.j], A[.i, .j])");
    assert!(
        out.contains("(%array 1 (3) (%scalar real))") && out.contains("(aggregate"),
        "var over .j should resolve to exact length 3, got:\n{out}"
    );
}

/// `partition(xs, spec)` → a vector of sub-vectors (spec §07); `selectbins(…,
/// counts)` → a shorter array of counts' type.
#[test]
fn partition_and_selectbins_infer() {
    let out = ir("xs = [1.0, 2.0, 3.0, 4.0]\np = partition(xs, 2)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%array 1 (%dynamic) (%scalar real)))")
            && out.contains("(partition"),
        "partition should infer a vector of real sub-vectors, got:\n{out}"
    );
    let out = ir("e = [0.0, 1.0, 2.0]\nc = [5.0, 7.0]\nr = selectbins(e, interval(0.0, 1.0), c)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%scalar real))") && out.contains("(selectbins"),
        "selectbins should infer a shorter real count array, got:\n{out}"
    );
}

/// `table(col = vector, …)` (spec §03 "Tables") → a `%table` whose stored
/// column types are the vectors' ELEMENT types, with `%nrows` the shared
/// column length (FlatPIR §11 `(%table (%columns (name elem) …) (%nrows N))`).
#[test]
fn table_constructor_infers() {
    let out = ir("t = table(mass = [1.1, 1.2, 1.3], pt = [4.5, 3.2, 6.7])");
    assert!(
        out.contains("(%table (%columns (mass (%scalar real)) (pt (%scalar real))) (%nrows 3))"),
        "table(...) should infer a 3-row table of real columns, got:\n{out}"
    );
}

/// `addaxes(A, nl, nt)` (spec §07) inserts size-1 axes around A — exact dims
/// when the counts are fixed; `splitblocks(v, bs)` nests a 1-D vector into a
/// vector of sub-vectors.
#[test]
fn addaxes_and_splitblocks_infer() {
    let out = ir("v = [1.0, 2.0, 3.0]\nx = addaxes(v, 1, 0)");
    assert!(
        out.contains("(%array 2 (1 3) (%scalar real))") && out.contains("(addaxes"),
        "addaxes(v,1,0) should be (1,3), got:\n{out}"
    );
    let out = ir("v = [1.0, 2.0, 3.0]\nx = addaxes(v, 0, 1)");
    assert!(
        out.contains("(%array 2 (3 1) (%scalar real))"),
        "addaxes(v,0,1) should be (3,1), got:\n{out}"
    );
    let out = ir("v = [1.0, 2.0, 3.0, 4.0]\nx = splitblocks(v, 2)");
    assert!(
        out.contains("(%array 1 (%dynamic) (%array 1 (%dynamic) (%scalar real)))")
            && out.contains("(splitblocks"),
        "splitblocks(1-D) should be a vector of real sub-vectors, got:\n{out}"
    );
}

/// ext-linear-algebra `lu`/`svd`/`eigen` now infer proper records (via the new
/// ResultSig::Record), no longer the degraded Matrix placeholder; `matexp`
/// passes its shape through and `lstsq` is a vector.
#[test]
fn ext_linalg_record_results_infer() {
    let pre =
        "e = standard_module(\"ext-linear-algebra\", \"0.1\")\nA = [[4.0, 0.0], [0.0, 9.0]]\n";
    let out = ir(&format!("{pre}d = e.lu(A)"));
    assert!(
        out.contains("(%record (P (%array 2")
            && out.contains("(L (%array 2")
            && out.contains("(U (%array 2"),
        "lu should infer record(P, L, U) of matrices, got:\n{out}"
    );
    let out = ir(&format!("{pre}d = e.svd(A)"));
    assert!(
        out.contains("(%record (U (%array 2")
            && out.contains("(S (%array 1 (%dynamic) (%scalar real)))")
            && out.contains("(V (%array 2"),
        "svd should infer record(U: matrix, S: real vector, V: matrix), got:\n{out}"
    );
    let out = ir(&format!("{pre}d = e.eigen(A)"));
    assert!(
        out.contains("(%record (values (%array 1") && out.contains("(vectors (%array 2"),
        "eigen should infer record(values: vector, vectors: matrix), got:\n{out}"
    );
    // matexp passes A's shape through; lstsq is a vector.
    let out = ir(&format!("{pre}d = e.matexp(A)"));
    assert!(
        out.contains("(%array 1 (2) (%array 1 (2) (%scalar real)))") && out.contains("matexp"),
        "matexp should preserve A's shape, got:\n{out}"
    );
}

/// Indexing an array by an integer ARRAY is a gather (`a[group_data]`, spec §07
/// "array of indices subset selection"): the result has the index's shape and
/// the container's element type — so a hierarchical `eta = a[g] .+ b .* x`
/// traces as a real array, not %deferred.
#[test]
fn gather_by_index_array_traces_real() {
    let src = "G = 3\n\
               x_data = [-1.2, 0.4, 1.1]\n\
               group_data = [1, 2, 3]\n\
               a ~ iid(Normal(0.0, 1.0), G)\n\
               b ~ Normal(0.0, 1.0)\n\
               gath = a[group_data]\n\
               eta = a[group_data] .+ b .* x_data\n";
    let out = ir(src);
    let line = |n: &str| {
        out.lines()
            .find(|l| l.contains(&format!("(%bind {n} ")))
            .unwrap_or("NONE")
    };
    assert!(
        line("gath").contains("(%array 1 (3) (%scalar real))"),
        "a[group_data] should gather to a length-3 real array, got:\n{}",
        line("gath")
    );
    assert!(
        line("eta").contains("(%array 1 (3) (%scalar real))"),
        "eta should be a real array (not %deferred element), got:\n{}",
        line("eta")
    );
}

/// distances `pairwise_distance`/`cross_distance` now infer EXACT result dims
/// from their input lengths (N×N and N×M) via DimExpr::OfParam, not the prior
/// Matrix(Dyn, Dyn) degraded placeholder.
#[test]
fn distance_matrix_dims_resolve() {
    let pre = "d = standard_module(\"distances\", \"0.1\")\n\
               f = (u, v) -> euclidean(u, v)\n\
               X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]\n\
               Y = [[0.0, 0.0], [1.0, 1.0]]\n";
    let out = ir(&format!("{pre}r = d.pairwise_distance(f, X)"));
    assert!(
        out.contains("(%array 2 (3 3) (%scalar real))") && out.contains("pairwise_distance"),
        "pairwise_distance over 3 points should be 3x3, got:\n{out}"
    );
    let out = ir(&format!("{pre}r = d.cross_distance(f, X, Y)"));
    assert!(
        out.contains("(%array 2 (3 2) (%scalar real))") && out.contains("cross_distance"),
        "cross_distance (3 x, 2 y) should be 3x2, got:\n{out}"
    );
}

/// `kron(A, B)` resolves EXACT Kronecker dims (rows A · rows B) × (cols A ·
/// cols B) via the new axis-aware DimExpr (Axis + Mul) — e.g. 2×3 ⊗ 2×2 → 4×6.
#[test]
fn kron_resolves_kronecker_dims() {
    let out = ir("e = standard_module(\"ext-linear-algebra\", \"0.1\")\n\
                  A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n\
                  B = [[1.0, 0.0], [0.0, 1.0]]\n\
                  k = e.kron(A, B)");
    assert!(
        out.contains("(%array 2 (4 6) (%scalar real))") && out.contains("kron"),
        "kron(2x3, 2x2) should be 4x6, got:\n{out}"
    );
}

// ---- User-callable application: per-call argument substitution (§04/§11) ----

/// A user function `f(a, b, x) = a + b * x` lowers to a reification whose
/// parameters are unconstrained `%local` placeholders. Applying it must bind the
/// concrete call-arg types to those parameters and re-infer the body — so
/// `predict(1.0, 2.0, 3.0)` is a `real`, NOT `any`. Before the substitution path
/// the body typed as `any` and every application inherited it.
#[test]
fn user_call_substitutes_arg_types_into_body() {
    let out = ir("predict(a, b, x) = a + b * x\n\
         z = predict(1.0, 2.0, 3.0)\n");
    let z = out.lines().find(|l| l.contains("%bind z")).unwrap_or("");
    assert!(
        z.contains("(%scalar real)") && !z.contains("%any"),
        "predict(reals) should yield a real, not any; got:\n{out}"
    );
}

/// The same substitution must flow through `broadcast`: a deterministic
/// user-callable head mapped over a `real[5]` data input yields `real[5]`, not
/// `any[5]`. The per-cell argument types (element of each array input) are bound
/// to the callable's parameters.
#[test]
fn broadcast_user_callable_substitutes_cell_types() {
    let out = ir("predict(a, b, x) = a + b * x\n\
         intercept ~ Normal(0.0, 1.0)\n\
         slope ~ Normal(0.0, 1.0)\n\
         x_data = [1.0, 2.0, 3.0, 4.0, 5.0]\n\
         eta = broadcast(predict, a = intercept, b = slope, x = x_data)\n");
    let eta = out.lines().find(|l| l.contains("%bind eta")).unwrap_or("");
    assert!(
        eta.contains("(%array 1 (5) (%scalar real))") && !eta.contains("%any"),
        "broadcast(predict, x = real[5]) should yield real[5], not any[5]; got:\n{out}"
    );
}

// ---- Function result value-sets via the catalogue `result_set` tag (§07) ----

/// Range-constrained scalar functions carry a value-set tighter than `reals`,
/// driven by the catalogue `result_set` tag rather than hardcoded inference
/// arms: `exp → posreals`, `sqrt`/`abs → nonnegreals`, `invlogit →
/// unitinterval`, `tanh → interval(-1, 1)`, `lengthof → nonnegintegers`.
#[test]
fn function_result_sets_are_tightened() {
    let cases = [
        ("y = exp(x)\nx = elementof(reals)", "posreals"),
        ("y = sqrt(x)\nx = elementof(reals)", "nonnegreals"),
        ("y = abs(x)\nx = elementof(reals)", "nonnegreals"),
        ("y = invlogit(x)\nx = elementof(reals)", "unitinterval"),
        ("y = tanh(x)\nx = elementof(reals)", "(interval -1.0 1.0)"),
        ("y = lengthof(v)\nv = [1.0, 2.0, 3.0]", "nonnegintegers"),
    ];
    for (src, want) in cases {
        let out = ir(src);
        let y = out.lines().find(|l| l.contains("%bind y")).unwrap_or("");
        assert!(
            y.contains(want),
            "expected `y`'s value-set to contain `{want}`; got:\n{out}"
        );
    }
}

/// A real-range `result_set` tag must NOT be claimed for a complex result: `exp`
/// of a complex value is complex-valued and not positive-real, so the value-set
/// falls back to the natural extent `complexes` (the `im` constant is complex).
#[test]
fn function_result_set_falls_back_for_complex() {
    let out = ir("y = exp(im)");
    let y = out.lines().find(|l| l.contains("%bind y")).unwrap_or("");
    assert!(
        y.contains("(%scalar complex)") && y.contains("complexes") && !y.contains("posreals"),
        "exp of a complex should be complexes, not posreals; got:\n{out}"
    );
}

// ---- load_data: dynamic-length vector of the declared valueset (§07) ----

/// `load_data(source, valueset)` is a vector of the declared `valueset`'s
/// element type with a `%dynamic` length (the row count is not statically
/// known). The value-set is `cartpow(valueset, %dynamic)`. Keyword and
/// positional spellings agree; a `cartpow` valueset gives a vector-of-vectors.
#[test]
fn load_data_is_a_dynamic_vector_of_the_valueset() {
    let out = ir("w = load_data(source = \"w.csv\", valueset = reals)");
    let w = out.lines().find(|l| l.contains("%bind w")).unwrap_or("");
    assert!(
        w.contains("(%array 1 (%dynamic) (%scalar real))")
            && w.contains("(cartpow reals %dynamic)"),
        "load_data(valueset=reals) should be a dynamic real vector; got:\n{out}"
    );
    // Positional form agrees.
    let out = ir("w = load_data(\"w.csv\", nonnegintegers)");
    let w = out.lines().find(|l| l.contains("%bind w")).unwrap_or("");
    assert!(
        w.contains("(%array 1 (%dynamic) (%scalar integer))")
            && w.contains("(cartpow nonnegintegers %dynamic)"),
        "positional load_data(.., nonnegintegers) should be a dynamic integer vector; got:\n{out}"
    );
    // A cartpow valueset → a dynamic vector of fixed-width vectors.
    let out = ir("w = load_data(source = \"w.csv\", valueset = cartpow(reals, 3))");
    let w = out.lines().find(|l| l.contains("%bind w")).unwrap_or("");
    assert!(
        w.contains("(%array 1 (%dynamic) (%array 1 (3) (%scalar real)))"),
        "load_data(valueset=cartpow(reals,3)) should be a dynamic vector of 3-vectors; got:\n{out}"
    );
}

// ---- User-callable results carry the substituted body's value-set (§04/§07) --

/// A callable whose body tightens its range carries that value-set to the call
/// site, direct and under `broadcast`: `f(x) = sqrt(x)` applied to a real gives
/// `nonnegreals`; `broadcast(f, v)` over a real vector gives
/// `cartpow(nonnegreals, n)`. Before, only the result TYPE was substituted and
/// the value-set fell back to the natural `reals`.
#[test]
fn user_call_carries_substituted_value_set() {
    let out = ir("f(x) = sqrt(x)\nr = f(2.0)");
    let r = out.lines().find(|l| l.contains("%bind r")).unwrap_or("");
    assert!(
        r.contains("(%scalar real) %fixed nonnegreals"),
        "f(x)=sqrt(x) applied should carry nonnegreals; got:\n{out}"
    );
    let out = ir("f(x) = sqrt(x)\nv = [1.0, 2.0, 3.0]\nr = broadcast(f, v)");
    let r = out.lines().find(|l| l.contains("%bind r")).unwrap_or("");
    assert!(
        r.contains("(cartpow nonnegreals 3)"),
        "broadcast(f, real[3]) should carry cartpow(nonnegreals, 3); got:\n{out}"
    );
}

// ---- Array/table value-sets: multi-axis cartpow, cartprod, load_data ----
// Discovery observations (2026-06-20, before implementation):
//   cartpow_multiaxis: already shows (cartpow (cartpow reals 3) 2) via
//     natural_of fallback (Task 1); set_expr_valueset is updated for
//     directness / correctness.
//   cartprod_positional: shows %unknown — needs set_expr_valueset arm.
//   cartprod_record:    shows %unknown — needs set_expr_valueset arm.
//   load_data_table:    shows %unknown — needs set_expr_valueset cartprod arm
//     which feeds into the load_data CartPow wrapping.

/// A multi-axis `cartpow` carries a nested value-set (gap A/B), not `%unknown`.
#[test]
fn cartpow_multiaxis_valueset() {
    let out = ir("m = elementof(cartpow(reals, [2, 3]))");
    assert!(
        out.contains("(cartpow (cartpow reals 3) 2)"),
        "multi-axis cartpow should carry a nested value-set; got:\n{out}"
    );
}

/// Positional `cartprod` carries a heterogeneous product value-set.
#[test]
fn cartprod_positional_valueset() {
    let out = ir("p = elementof(cartprod(reals, posreals))");
    assert!(
        out.contains("(cartprod reals posreals)"),
        "positional cartprod should carry a CartProd value-set; got:\n{out}"
    );
}

/// Positional `cartprod` is a set of ARRAYS, not a tuple (spec §03): the
/// `elementof` result types as an array of the unified element type, with the
/// per-position membership kept in the value-set slot. A tuple must NOT leak
/// into the type (tuples are objects — no value-set, not an `elementof`
/// result, not a measure domain — spec §04). A scalar+vector mix defers, since
/// §06/§07 `cat` forbid that concatenation.
#[test]
fn cartprod_positional_is_array_not_tuple() {
    let out = ir("p = elementof(cartprod(reals, integers))");
    assert!(
        out.contains("(%array 1 (2) (%scalar real))"),
        "positional cartprod should type as a 2-element real array; got:\n{out}"
    );
    assert!(
        out.contains("(cartprod reals integers)"),
        "per-position membership stays in the value-set; got:\n{out}"
    );
    assert!(
        !out.contains("%tuple"),
        "no tuple may leak from positional cartprod into an elementof type; got:\n{out}"
    );
}

/// Keyword `cartprod` carries a named record value-set.
#[test]
fn cartprod_record_valueset() {
    let out = ir("r = elementof(cartprod(a = reals, b = unitinterval))");
    assert!(
        out.contains("(record (a reals) (b unitinterval))"),
        "keyword cartprod should carry a RecordSet value-set; got:\n{out}"
    );
}

/// `load_data` over a record cartprod is a dynamic-row table value-set.
#[test]
fn load_data_table_valueset() {
    let out =
        ir("t = load_data(source = \"d.csv\", valueset = cartprod(a = reals, b = unitinterval))");
    assert!(
        out.contains("(cartpow (record (a reals) (b unitinterval)) %dynamic)"),
        "load_data(cartprod record) should be a dynamic vector of records; got:\n{out}"
    );
}

/// `disintegrate(selector, joint)` (spec §06): the marginal carries the
/// COMPLEMENT of the selected variates, and the forward kernel's inputs are those
/// complement (conditioning) variates. Selector reads like `get` (`Scalar::Str`).
/// Discovery (2026-06-20): `["b"]` lowers to `(vector "b")` — a vector call of
/// string literals. `mu` types as a record-domain measure over {a, b}. Selecting
/// `b` → complement is `{a}` → kernel inputs = a, marginal domain = record{a}.
#[test]
fn disintegrate_splits_record_joint() {
    let out = ir("a ~ Normal(0.0, 1.0)\n\
                  b ~ Normal(a, 1.0)\n\
                  mu = lawof(record(a = a, b = b))\n\
                  parts = disintegrate([\"b\"], mu)");
    assert!(
        out.contains("(%tuple (%kernel (%inputs a) (%mass %normalized)) (%measure (%domain (%record (a (%scalar real)))) (%mass %normalized)))"),
        "disintegrate(['b'], mu) → (kernel inputs=a, marginal over record{{a}}), got:\n{out}"
    );
}

/// `disintegrate` falls back to empty-inputs kernel + deferred marginal domain
/// when the selector is a non-literal (a binding reference, not a `["b"]` vector
/// literal): spec §06 honesty — no domain is fabricated.
#[test]
fn disintegrate_defers_for_non_literal_selector() {
    // `sel` is a binding ref — `selector_field_names` returns None (it only
    // handles literal strings / vector-of-literals), so both the kernel inputs
    // and the marginal domain fall back to the deferred form.
    // Discovery (2026-06-20): `["b"]` in source lowers to `(vector "b")` as a
    // binding; `disintegrate` receives `(%ref self sel)` — a Ref node — so the
    // selector is not statically resolvable.
    let out = ir("a ~ Normal(0.0, 1.0)\n\
                  b ~ Normal(a, 1.0)\n\
                  mu = lawof(record(a = a, b = b))\n\
                  sel = [\"b\"]\n\
                  parts = disintegrate(sel, mu)");
    assert!(
        out.contains("(%kernel (%inputs ) (%mass %normalized))"),
        "disintegrate with non-literal selector should have empty kernel inputs, got:\n{out}"
    );
    assert!(
        out.contains("(%measure (%domain %deferred) (%mass %normalized))"),
        "disintegrate with non-literal selector should have deferred marginal domain, got:\n{out}"
    );
}

/// `disintegrate([], joint)` — an empty selector `[]` — must yield the deferred
/// fallback, not a fabricated result.
///
/// Discovery (2026-06-20): `[]` lowers to `(vector)` — a `vector` call with zero
/// args, which matches the `vector`-literal branch of `selector_field_names` and
/// collects to an empty vec. Without the guard, that empty vec was treated as
/// `Some([])` — a resolved-but-empty selector — causing `disintegrate` to walk the
/// record fields and produce non-empty kernel inputs and a full marginal domain.
/// The fix returns `None` (→ deferred) whenever the collected name list is empty.
#[test]
fn disintegrate_defers_for_empty_selector() {
    // `[]` lowers to `(vector)` with zero args. `selector_field_names` must
    // return `None` for an empty name list, so `disintegrate_type` falls back
    // to the deferred path: empty kernel inputs + deferred marginal domain.
    let out = ir("a ~ Normal(0.0, 1.0)\n\
                  b ~ Normal(a, 1.0)\n\
                  mu = lawof(record(a = a, b = b))\n\
                  parts = disintegrate([], mu)");
    assert!(
        out.contains("(%inputs )"),
        "disintegrate([], mu) should have empty kernel inputs, got:\n{out}"
    );
    assert!(
        out.contains("(%domain %deferred)"),
        "disintegrate([], mu) should have a deferred marginal domain, got:\n{out}"
    );
}

/// `likelihoodof(fk, obs)` where `fk` is a `disintegrate` forward-kernel recovers
/// the obstype from the joint's selected-variate record (spec §06 "Structural
/// disintegration"). The kernel comes from `fk, prior = disintegrate(sel, joint)`
/// which desugars to `fk = get(__synth, 1)`; `likelihood_type` follows the
/// `get` → ref → `disintegrate` chain and keeps the SELECTED fields.
/// Discovery (2026-06-20): before the fix `L` was `%deferred`; after, it is
/// `(%likelihood (%inputs a) (%obstype (%record (b (%scalar real)))))`.
#[test]
fn likelihoodof_of_disintegrate_kernel_recovers_obstype() {
    let out = ir("a ~ Normal(0.0, 1.0)\n\
                  b ~ Normal(a, 1.0)\n\
                  joint_model = lawof(record(a = a, b = b))\n\
                  fk, prior = disintegrate([\"b\"], joint_model)\n\
                  L = likelihoodof(fk, record(b = 1.0))");
    assert!(
        out.contains(
            "(%bind L (%meta ((%likelihood (%inputs a) (%obstype (%record (b (%scalar real)))))"
        ),
        "likelihoodof of a disintegrate kernel should recover the selected-variate obstype; got:\n{out}"
    );
}

/// A `cartprod`/`cartpow`/`interval` PRESET binding (spec §03 "Presets") denotes
/// a set; its value-set slot carries that set (the PR-#34 vocabulary), not
/// `anything`. (Its TYPE stays `%any` — a set is not a value type, §03 "Sets".)
#[test]
fn set_constructor_presets_carry_their_set() {
    let out = ir("default_domain = cartprod(a = interval(0.0, 5.0), b = unitinterval)");
    assert!(
        out.contains("(record (a (interval 0.0 5.0)) (b unitinterval))"),
        "cartprod preset value-set should be the denoted record set, got:\n{out}"
    );
    let out = ir("grid = cartpow(interval(-10.0, 10.0), 3)");
    assert!(
        out.contains("(cartpow (interval -10.0 10.0) 3)"),
        "cartpow preset value-set should be the denoted power set, got:\n{out}"
    );
    let out = ir("r = interval(0.0, 1.0)");
    assert!(
        out.contains("(interval 0.0 1.0)"),
        "interval preset value-set should be the denoted interval, got:\n{out}"
    );
}
