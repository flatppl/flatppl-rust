//! Golden and refusal tests for `crate::norms` — spec §07 "Reductions" and
//! "Norms and normalization", the twelve bare heads that refused at
//! `origin/main` 50194b2 through `lower_builtin`'s catch-all.
//!
//! A separate file from `tests/golden.rs`, `golden_stack.rs` and
//! `golden_wire.rs`, for the reason `golden_stack.rs` gives: several emitter
//! waves append tests concurrently, and one file per wave keeps the textual
//! conflict surface at zero. The helpers below are local copies for the same
//! reason.
//!
//! **Every frozen `.mlir` golden here was EXECUTED**, not merely pinned:
//! compiled with `iree-base-compiler` 3.11 (llvm-cpu), run under
//! `iree-base-runtime` (local-task), and every output matched against a numpy
//! oracle at `rtol = atol = 2e-6` (`Dtype::F32`). The goldens are read off disk
//! by these tests, so the pinned text and the verified numbers cannot diverge.
//! `var`/`std` were matched against `numpy` with `ddof=1`, which is the $n-1$
//! denominator §07 defines. Every operand arrives as a runtime `elementof`
//! argument rather than a literal, so nothing is constant-folded away — and
//! IREE 3.11 miscompiles a splat-constant-fed reduction, which a literal
//! operand would have walked into. See
//! `.superpowers/sdd/2026-08-05-joint-constructs-the-joint/wave-hlonorm-report.md`.

use flatppl_core::{Dim, Module, Type, ValueSet};

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    m
}

/// Determinize with the `inputs`/`outputs` ABI bindings as the roots — the shape
/// every model here takes, since each head is exercised over a runtime
/// (`elementof`) operand.
fn determinize_abi(src: &str) -> Module {
    let mut m = parse_infer(src);
    let syms: Vec<flatppl_core::Symbol> =
        ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse")
}

fn emit(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_abi(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

fn emit_err(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_abi(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse in the emitter")
    .msg
}

/// One `head(v)` over a runtime vector of `len` elements drawn from `set`.
fn vec_src(head: &str, set: &str, len: usize) -> String {
    format!("v = elementof(cartpow({set}, [{len}]))\ny = {head}(v)\ninputs = (v)\noutputs = (y)\n")
}

/// The sentinel length [`zero_the_extents`] rewrites away. Distinctive on
/// purpose: a `1` would collide with an incidental unit extent and silently
/// rewrite it.
const LEN0_SENTINEL: u32 = 7;

/// Rewrite every [`LEN0_SENTINEL`] extent in a module's inferred types and value
/// sets to 0, in place.
///
/// The written-size positivity rule makes a length-0 extent unspellable — a
/// written `0` size is a static error, and a derived 0 types `%dynamic`
/// (`flatppl-dev/empty-arrays-ruling.md`, 2026-08-20). The emitter's length-0
/// behavior is still reachable at runtime and still needs its regression
/// coverage, so it is exercised by CONSTRUCTING the shape instead of spelling it.
///
/// Applied to the DETERMINIZED module, not the parsed one: the determiniser
/// re-runs inference on its working copy (`driver.rs`, "Re-run inference
/// (idempotent) so type / phase tables are fresh"), which would recompute the
/// sentinel straight back. The emitter re-runs nothing, and its length-0 special
/// cases read the operand extent off these types, so this is the input it used
/// to receive from `cartpow(reals, [0])`. Sound for these models because the
/// FlatPDL graph is length-independent — every head here lowers in the emitter,
/// so the determiniser neither unrolls nor sizes anything.
fn zero_the_extents(m: &mut Module) {
    for i in 0..m.node_count() {
        let id = <flatppl_core::NodeId as flatppl_core::Idx>::from_usize(i);
        if let Some(ty) = m.type_of(id).cloned() {
            m.set_type(id, zero_extents_ty(&ty));
        }
        if let Some(vs) = m.valueset_of(id).cloned() {
            m.set_valueset(id, zero_extents_vs(&vs));
        }
    }
}

fn zero_dim(d: Dim) -> Dim {
    match d {
        Dim::Static(LEN0_SENTINEL) => Dim::Static(0),
        other => other,
    }
}

fn zero_extents_ty(ty: &Type) -> Type {
    match ty {
        Type::Array { shape, elem } => Type::Array {
            shape: shape.iter().map(|d| zero_dim(*d)).collect(),
            elem: Box::new(zero_extents_ty(elem)),
        },
        Type::TVector { len, elem } => Type::TVector {
            len: zero_dim(*len),
            elem: elem.clone(),
        },
        Type::Tuple(elems) => Type::Tuple(elems.iter().map(zero_extents_ty).collect()),
        Type::Measure { domain, mass } => Type::Measure {
            domain: Box::new(zero_extents_ty(domain)),
            mass: *mass,
        },
        other => other.clone(),
    }
}

fn zero_extents_vs(vs: &ValueSet) -> ValueSet {
    match vs {
        ValueSet::CartPow(inner, len) => {
            ValueSet::CartPow(Box::new(zero_extents_vs(inner)), zero_dim(*len))
        }
        ValueSet::StdSimplex(d) => ValueSet::StdSimplex(zero_dim(*d)),
        other => other.clone(),
    }
}

/// One `head(v)` over a length-0 vector, built rather than spelled: the
/// sentinel-length source through the determiniser, then [`zero_the_extents`].
fn len0_module(head: &str, set: &str) -> Module {
    let mut det = determinize_abi(&vec_src(head, set, LEN0_SENTINEL as usize));
    zero_the_extents(&mut det);
    det
}

/// The length-0 counterpart of [`emit`].
fn emit_len0(head: &str, set: &str) -> String {
    flatppl_stablehlo::emit(
        &len0_module(head, set),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

/// The length-0 counterpart of [`emit_err`].
fn emit_err_len0(head: &str, set: &str) -> String {
    flatppl_stablehlo::emit(
        &len0_module(head, set),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse in the emitter")
    .msg
}

// ---- the twelve heads all lower ----------------------------------------------

/// Every head this wave wires emits, over the §07 domain it declares. The point
/// is the map entry: a missing arm is `lower_builtin`'s catch-all
/// "unsupported builtin head", which is what all twelve did at 50194b2.
#[test]
fn every_wired_head_lowers_over_a_real_vector() {
    for head in [
        "prod",
        "mean",
        "var",
        "std",
        "cumsum",
        "cumprod",
        "l1norm",
        "l2norm",
        "l1unit",
        "l2unit",
        "softmax",
        "logsoftmax",
    ] {
        let out = emit(&vec_src(head, "reals", 4));
        assert!(
            out.contains("func.func @logdensity"),
            "`{head}` must lower:\n{out}"
        );
    }
}

// ---- §07 "Reductions" --------------------------------------------------------

/// The frozen `prod`/`mean`/`var`/`std` module — the EXECUTED artifact, read off
/// disk. At `v = [1.5, -2.0, 3.25, 0.5]` it computes `-4.875`, `0.8125`,
/// `4.8072915` and `2.1925538`, matched against `np.prod`, `np.mean`,
/// `np.var(ddof=1)` and `np.std(ddof=1)`.
#[test]
fn reductions_module_matches_frozen_golden() {
    let src = "\
v = elementof(cartpow(reals, [4]))
o1 = prod(v)
o2 = mean(v)
o3 = var(v)
o4 = std(v)
inputs = (v)
outputs = (o1, o2, o3, o4)
";
    assert_eq!(
        emit(src),
        include_str!("goldens/norms_reductions.mlir"),
        "emitted @logdensity drifted from tests/goldens/norms_reductions.mlir"
    );
}

/// `prod` contracts with the MULTIPLICATIVE identity, and `sum` with the
/// additive one. An additive identity under a `multiply` combine would make
/// every product zero, which no structural test on the op name alone would
/// catch.
#[test]
fn prod_uses_the_multiplicative_reduce_identity() {
    let out = emit(&vec_src("prod", "reals", 4));
    assert!(
        out.contains("applies stablehlo.multiply across dimensions = [0]"),
        "prod must reduce with multiply:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<1.000000e+00>"),
        "prod's reduce identity must be 1, not 0:\n{out}"
    );
}

/// §07 defines `var` as $\frac{1}{n-1}\sum_i(x_i-\bar{x})^2$ — the SAMPLE
/// convention. Over four elements the mean divides by `4` and the variance by
/// `3`; a population variance would divide by `4` twice. §04's own column-wise
/// example pins the same convention (it prints `[32, 2, 8]` for a 2-row matrix,
/// where the population variance would print `[16, 1, 4]`), and the executed
/// golden matches `numpy` at `ddof=1`.
#[test]
fn var_uses_the_n_minus_one_denominator() {
    let out = emit(&vec_src("var", "reals", 4));
    assert!(
        out.contains("stablehlo.constant dense<4.0>"),
        "the mean must divide by n = 4:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<3.0>"),
        "the variance must divide by n - 1 = 3:\n{out}"
    );
}

/// `std` is `sqrt(var)`, so it carries `var`'s whole body plus one
/// `stablehlo.sqrt` — including both denominators.
#[test]
fn std_is_the_square_root_of_var() {
    let out = emit(&vec_src("std", "reals", 4));
    assert!(
        out.contains("stablehlo.sqrt"),
        "std must take a root:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<3.0>"),
        "std must inherit var's n - 1 denominator:\n{out}"
    );
}

/// §07 gives the reductions the domain "arrays" of any rank, and writes each as
/// a single index-free aggregate — so a bare `var(A)` over a matrix reduces
/// EVERY element, not one axis. §07 itself routes per-axis contraction
/// elsewhere: "For multi-axis array contraction using these reductions, see
/// multi-axis aggregation."
///
/// EXECUTED over `A = [[1.5, -2.0, 3.25], [0.5, 4.0, -1.25]]`: `prod` `24.375`,
/// `mean` `1.0`, `var` `5.725`, `std` `2.392697`, matching numpy's whole-array
/// reductions (`ddof=1` for the last two).
#[test]
fn a_bare_reduction_over_a_matrix_reduces_every_axis() {
    for head in ["prod", "mean", "var", "std"] {
        let src = format!(
            "A = elementof(cartpow(reals, [2, 3]))\ny = {head}(A)\ninputs = (A)\noutputs = (y)\n"
        );
        let out = emit(&src);
        assert!(
            out.contains("-> tensor<f32>"),
            "`{head}` over a matrix must reduce to a SCALAR:\n{out}"
        );
        assert_eq!(
            out.matches("dimensions = [1]").count() + out.matches("dimensions = [0]").count(),
            if head == "prod" || head == "mean" {
                2
            } else {
                4
            },
            "`{head}` must reduce both axes (twice over, for the two sums \
             var/std take):\n{out}"
        );
    }
}

/// §03 "Bool": "`false` is promoted to zero and `true` to one, permitting
/// expressions such as `true + true`, `3 * false`, and `sum(mask)` to count true
/// entries". So `prod` over a boolean array is the product of 0s and 1s — one
/// iff every entry is true — and `infer::ops::reduced_scalar` types it
/// `integers`. The operand is CONVERTED to `i32` before the reduce, because over
/// `i1` `stablehlo.multiply` is a conjunction rather than §07's product.
///
/// EXECUTED: `prod([true, true, false, true])` returns `0` at `tensor<i32>`.
#[test]
fn prod_over_booleans_promotes_to_integers_per_section_03() {
    let out = emit(&vec_src("prod", "booleans", 4));
    assert!(
        out.contains("stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi32>"),
        "a boolean operand must be widened to i32 first:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<1> : tensor<i32>"),
        "the identity must be the INTEGER 1:\n{out}"
    );
    assert!(
        out.contains("-> tensor<i32>"),
        "the result type must be the integers infer types it:\n{out}"
    );
}

/// An INTEGER array keeps its kind end to end — `prod` returns an element-typed
/// aggregate per §07, and `infer::ops::reduced_scalar` maps `("prod", e)` to `e`.
/// EXECUTED: `prod([2, -3, 4, 5])` returns `-120` at `tensor<i32>`.
#[test]
fn prod_over_integers_stays_integer() {
    let out = emit(&vec_src("prod", "integers", 4));
    assert!(
        out.contains("stablehlo.constant dense<1> : tensor<i32>") && out.contains("-> tensor<i32>"),
        "prod over an integer array must reduce and return at i32:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.convert"),
        "no conversion is needed — the kind already matches:\n{out}"
    );
}

// ---- §07 cumulative pair -----------------------------------------------------

/// The frozen `cumsum`/`cumprod` module — the EXECUTED artifact. At
/// `v = [1.5, -2.0, 3.25, 0.5]` it returns `[1.5, -0.5, 2.75, 3.25]` and
/// `[1.5, -3.0, -9.75, -4.875]`, matching `np.cumsum`/`np.cumprod` exactly.
#[test]
fn cumulative_module_matches_frozen_golden() {
    let src = "\
v = elementof(cartpow(reals, [4]))
o1 = cumsum(v)
o2 = cumprod(v)
inputs = (v)
outputs = (o1, o2)
";
    assert_eq!(
        emit(src),
        include_str!("goldens/norms_cumulative.mlir"),
        "emitted @logdensity drifted from tests/goldens/norms_cumulative.mlir"
    );
}

/// A scan is one `stablehlo.reduce_window` whose window is the whole vector,
/// left-padded by `n - 1`. Both numbers are load-bearing: a shorter window or a
/// different pad would compute a sliding-window reduction rather than §07's
/// prefix, and the padded positions must take the reduce's own identity.
#[test]
fn a_scan_is_a_whole_vector_window_left_padded_by_n_minus_one() {
    let out = emit(&vec_src("cumsum", "reals", 4));
    assert!(
        out.contains("\"stablehlo.reduce_window\""),
        "a scan must use reduce_window:\n{out}"
    );
    assert!(
        out.contains("window_dimensions = array<i64: 4>"),
        "the window must span the whole length-4 vector:\n{out}"
    );
    assert!(
        out.contains("padding = dense<[[3, 0]]> : tensor<1x2xi64>"),
        "the window must be left-padded by n - 1 = 3, and not right-padded:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.while"),
        "the scan must not need a loop:\n{out}"
    );
}

/// `cumprod`'s window pads with the MULTIPLICATIVE identity, `cumsum`'s with the
/// additive one. Padding a product with zeros would make every prefix zero.
#[test]
fn each_scan_pads_with_its_own_reduce_identity() {
    let sum = emit(&vec_src("cumsum", "reals", 4));
    assert!(
        sum.contains("stablehlo.constant dense<0.000000e+00>") && sum.contains("stablehlo.add"),
        "cumsum pads with 0 and combines with add:\n{sum}"
    );
    let prod = emit(&vec_src("cumprod", "reals", 4));
    assert!(
        prod.contains("stablehlo.constant dense<1.000000e+00>")
            && prod.contains("stablehlo.multiply"),
        "cumprod pads with 1 and combines with multiply:\n{prod}"
    );
}

/// §03's promotion reaches the cumulative pair for the same reason it reaches
/// `sum`: `cumsum([true, true, false, true])` is `[1, 2, 2, 3]`, and `2` is not
/// a boolean — `infer`'s own boolean carve-out types the result with an INTEGER
/// element. Reduced in `i1`, `stablehlo.add` would compute parity
/// (`[1, 0, 0, 1]`).
///
/// EXECUTED at `i32`: `[1, 2, 2, 3]` for `cumsum` and `[1, 1, 0, 0]` for
/// `cumprod`.
#[test]
fn the_cumulative_pair_promotes_booleans_rather_than_scanning_in_i1() {
    for head in ["cumsum", "cumprod"] {
        let out = emit(&vec_src(head, "booleans", 4));
        assert!(
            out.contains("stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi32>"),
            "`{head}` must widen a boolean operand:\n{out}"
        );
        assert!(
            !out.contains("tensor<4xi1>) -> tensor<4xi1>") && out.contains("-> tensor<4xi32>"),
            "`{head}` must not scan in i1:\n{out}"
        );
    }
}

// ---- §07 "Norms and normalization" -------------------------------------------

/// The frozen norms module — the EXECUTED artifact, all six heads. At
/// `v = [1.5, -2.0, 3.25, 0.5]`: `l1norm` `7.25`, `l2norm` `4.1306777`, and the
/// four vector-valued heads matched elementwise against numpy.
#[test]
fn vector_norms_module_matches_frozen_golden() {
    let src = "\
v = elementof(cartpow(reals, [4]))
o1 = l1norm(v)
o2 = l2norm(v)
o3 = l1unit(v)
o4 = l2unit(v)
o5 = softmax(v)
o6 = logsoftmax(v)
inputs = (v)
outputs = (o1, o2, o3, o4, o5, o6)
";
    assert_eq!(
        emit(src),
        include_str!("goldens/norms_vector_norms.mlir"),
        "emitted @logdensity drifted from tests/goldens/norms_vector_norms.mlir"
    );
}

/// $\ell^1$ takes an `abs` and $\ell^2$ does not: over the reals this crate
/// emits, `v * v` already discards the sign, so an `abs` before the square would
/// be a wasted op (the reasoning `crate::ops`' `abs2` head records).
#[test]
fn l1norm_takes_an_abs_and_l2norm_squares_instead() {
    let l1 = emit(&vec_src("l1norm", "reals", 4));
    assert!(l1.contains("stablehlo.abs"), "l1norm needs abs:\n{l1}");
    assert!(
        !l1.contains("stablehlo.sqrt"),
        "l1norm takes no root:\n{l1}"
    );
    let l2 = emit(&vec_src("l2norm", "reals", 4));
    assert!(
        l2.contains("stablehlo.multiply %arg0, %arg0"),
        "l2norm squares the operand:\n{l2}"
    );
    assert!(
        !l2.contains("stablehlo.abs"),
        "squaring discards the sign, so l2norm needs no abs:\n{l2}"
    );
    assert!(l2.contains("stablehlo.sqrt"), "l2norm takes a root:\n{l2}");
}

/// `l1unit`/`l2unit` are their norm plus one divide, with the reduced scalar
/// splatted back over the vector's shape (StableHLO has no implicit scalar
/// broadcast).
#[test]
fn the_unit_heads_divide_the_vector_by_its_own_norm() {
    for head in ["l1unit", "l2unit"] {
        let out = emit(&vec_src(head, "reals", 4));
        assert!(
            out.contains("stablehlo.broadcast_in_dim") && out.contains("stablehlo.divide"),
            "`{head}` must splat the norm and divide:\n{out}"
        );
        assert!(
            out.contains("-> tensor<4xf32>"),
            "`{head}` must return a VECTOR, not the norm:\n{out}"
        );
    }
}

/// §07's domain for the norms is "real/complex vectors", and §03's
/// `booleans ⊂ integers ⊂ reals` puts an integer vector inside it. `infer` types
/// `l1norm` `Scalar(Real)`, so the operand is widened rather than reduced at
/// `i32` and returned as an integer. EXECUTED: `l1norm([2, -3, 4, 5])` is `14.0`
/// at `tensor<f32>`.
#[test]
fn a_norm_widens_an_integer_vector_to_reals() {
    let out = emit(&vec_src("l1norm", "integers", 4));
    assert!(
        out.contains("stablehlo.convert %arg0 : (tensor<4xi32>) -> tensor<4xf32>"),
        "an integer operand must widen to f32:\n{out}"
    );
    assert!(
        out.contains("-> tensor<f32>"),
        "the result must be the real infer types it:\n{out}"
    );
}

/// `softmax`/`logsoftmax` are MAX-SHIFTED: both subtract `max(v)` before any
/// `exp`, so every exponent is $\le 0$ and nothing overflows. A naive
/// $e^{v_i}/\sum_j e^{v_j}$ is `NaN` at `v = [1000, 1001, 999, 1000.5]` even in
/// f64; the emitted form returns the correct distribution there (measured under
/// IREE) and sums to one at f32. ("At f32" is the honest qualifier: the four
/// components accumulate to `1.0` in f32 but to `0.9999999702` in f64, and to
/// `1.0000000522` at the mirrored all-negative probe.)
#[test]
fn softmax_and_logsoftmax_shift_by_the_max_before_exponentiating() {
    for head in ["softmax", "logsoftmax"] {
        let out = emit(&vec_src(head, "reals", 4));
        let max_at = out
            .find("applies stablehlo.maximum")
            .unwrap_or_else(|| panic!("`{head}` must reduce a max:\n{out}"));
        let sub_at = out
            .find("stablehlo.subtract")
            .unwrap_or_else(|| panic!("`{head}` must subtract it:\n{out}"));
        let exp_at = out
            .find("stablehlo.exponential")
            .unwrap_or_else(|| panic!("`{head}` must exponentiate:\n{out}"));
        assert!(
            max_at < sub_at && sub_at < exp_at,
            "`{head}` must subtract the max BEFORE the exp, not after:\n{out}"
        );
    }
}

/// `softmax` DIVIDES by the sum of the shifted exponentials rather than taking
/// `exp(v - logsumexp(v))`. The two are equal in the reals, but the `logsumexp`
/// route takes a `log` and then an `exp` of the same quantity, and that round
/// trip loses accuracy the division does not. So `softmax` emits NO `log`, while
/// `logsoftmax` — which needs one by definition — emits exactly one and no
/// second `exp` of it.
#[test]
fn softmax_divides_rather_than_round_tripping_through_a_log() {
    let sm = emit(&vec_src("softmax", "reals", 4));
    assert!(
        !sm.contains("stablehlo.log"),
        "softmax must not go through a log:\n{sm}"
    );
    assert_eq!(
        sm.matches("stablehlo.divide").count(),
        1,
        "softmax is one divide by the shifted sum:\n{sm}"
    );
    let lsm = emit(&vec_src("logsoftmax", "reals", 4));
    assert_eq!(
        lsm.matches("stablehlo.log").count(),
        1,
        "logsoftmax needs exactly one log:\n{lsm}"
    );
    assert_eq!(
        lsm.matches("stablehlo.exponential").count(),
        1,
        "logsoftmax must not exponentiate its own log back:\n{lsm}"
    );
}

// ---- refusals ----------------------------------------------------------------

/// §07 gives the cumulative pair and every norm the domain "vectors"
/// specifically, unlike the reductions' "arrays". A matrix operand therefore has
/// no §07 meaning here, and is refused rather than answered along a guessed
/// axis.
#[test]
fn the_vector_only_heads_refuse_a_matrix() {
    for head in [
        "cumsum",
        "cumprod",
        "l1norm",
        "l2norm",
        "l1unit",
        "l2unit",
        "softmax",
        "logsoftmax",
    ] {
        let src = format!(
            "A = elementof(cartpow(reals, [2, 3]))\ny = {head}(A)\ninputs = (A)\noutputs = (y)\n"
        );
        let err = emit_err(&src);
        assert!(
            err.contains(head) && err.contains("the domain \"vectors\""),
            "`{head}` must refuse a matrix citing §07's vectors domain, got: {err}"
        );
    }
}

/// **§07 pins NO scanned axis for `cumsum`/`cumprod`, so neither multi-axis
/// spelling is answered.** The §07 row gives the domain "vectors" and describes
/// the flat sequence $(x_1, x_1+x_2, \dots)$ — one running index, no axis
/// argument and no default axis. Nothing anywhere else in the spec adds one:
/// outside its own §07 row the pair appears only in §08's Dirichlet prose
/// (descriptive) and §12's Stan profile mapping table (verified by grepping all
/// of `flatppl-design/docs/`). So there is no rule to implement here, and the
/// two ways to write a multi-axis scan both refuse — each citing the rule that
/// actually applies to it, which is NOT the same rule:
///
/// - `aggregate(cumsum, [.i], A[.i, .j])` names `cumsum` as the `f_reduction`,
///   and §04 requires "an order-invariant vector-to-scalar reduction", listing
///   the eligible built-ins explicitly. A scan is neither order-invariant nor
///   vector-to-scalar, so `crate::aggregate` refuses it. Pre-existing (the
///   `hlo-aggregate` wave), and locked in `tests/golden.rs`; asserted here only
///   so wiring the bare head cannot have quietly made `cumsum` eligible.
/// - `y[.i] := cumsum(A[.i, .j])` does NOT name `cumsum` as the reduction. The
///   `:=` sugar is a **sum**-aggregate (`parser.rs`: "`C[.i, .k] := expr` —
///   sum-aggregate"), so this is `aggregate(sum, [.i], cumsum(A[.i, .j]))` with
///   `cumsum` as a BODY function — confirmed by reading the FlatPIR. The body is
///   applied to a frame-shaped rank-2 value, so §04's eligible-reduction rule is
///   irrelevant and §07's vectors domain is the applicable one.
#[test]
fn no_multi_axis_scan_spelling_is_answered_since_section_07_pins_no_axis() {
    let matrix = "A = elementof(cartpow(reals, [2, 3]))\n";
    for head in ["cumsum", "cumprod"] {
        // `cumsum` AS the f_reduction — §04's eligibility rule.
        let err = emit_err(&format!(
            "{matrix}y = aggregate({head}, [.i], get(A, .i, .j))\ninputs = (A)\noutputs = (y)\n"
        ));
        assert!(
            err.contains("is not an eligible reduction")
                && err.contains("order-invariant vector-to-scalar reduction"),
            "`aggregate({head}, …)` must refuse on §04's eligibility rule, got: {err}"
        );

        // `cumsum` in the BODY of the `:=` sum-aggregate — §07's vectors domain.
        let err = emit_err(&format!(
            "{matrix}y[.i] := {head}(A[.i, .j])\ninputs = (A)\noutputs = (y)\n"
        ));
        assert!(
            err.contains(head) && err.contains("the domain \"vectors\""),
            "`y[.i] := {head}(…)` must refuse on §07's vectors domain, got: {err}"
        );
    }
}

/// **The complete length-0 behavior of all eight vector heads**, in one place, so
/// none can regress independently. Every answer is FIXED by the owner's
/// zero-size-arrays ruling of 2026-08-20 (`flatppl-dev/empty-arrays-ruling.md`,
/// sub-ruling 2), each matching numpy and the js engine's recorded position, and
/// each verified by EXECUTION on a `tensor<0x…>` runtime argument:
///
/// | head | empty result | why |
/// |---|---|---|
/// | `cumsum`/`cumprod` | the empty vector | the prefix sequence of an empty sequence is empty (`np.cumsum([])` is `[]`) |
/// | `l1norm`/`l2norm` | `0.0` | the empty sum, and `sqrt(0) == 0` |
/// | `l1unit`/`l2unit` | the empty vector | no element to divide, so the division is vacuous |
/// | `softmax`/`logsoftmax` | the empty vector | likewise — and no `NaN`, since there is no element to produce one |
///
/// Only `mean`/`var`/`std` refuse over an empty array, and only because they
/// divide by the element count ($0/0$); nothing here does.
///
/// **The scans are a REGRESSION TEST.** They used to reach
/// `Emitter::prefix_scan`, whose `n - 1` padding term underflowed: debug panicked
/// with "attempt to subtract with overflow", and release emitted
/// `window_dimensions = 0` with `padding = 2^64 - 1` from a CLI that exited 0 —
/// text IREE rejects ("expects window to have positive value for 0-th window
/// dimension"). `stablehlo.reduce_window` cannot express a length-0 scan at all,
/// so `lower_cumulative` answers it directly by returning the operand.
///
/// **The operand is now CONSTRUCTED, not spelled** — see [`zero_the_extents`].
/// `cartpow(reals, [0])` is a static error since the written-size positivity
/// rule, so this coverage would otherwise have been lost with its fixture;
/// [`a_written_zero_size_is_refused`] guards that fixture change itself.
#[test]
fn every_vector_head_has_a_defined_length_zero_result() {
    // The scans and the three vector-valued norm families return a length-0
    // vector; nothing is emitted for the scans, so the module is a bare return.
    for head in ["cumsum", "cumprod"] {
        let out = emit_len0(head, "reals");
        assert!(
            out.contains("-> tensor<0xf32>"),
            "`{head}` over an empty vector must return an empty vector:\n{out}"
        );
        assert!(
            !out.contains("reduce_window"),
            "a length-0 scan has no reduce_window form, so none may be emitted:\n{out}"
        );
        assert!(
            !out.contains("18446744073709551615"),
            "`{head}` must not emit an underflowed padding term:\n{out}"
        );
    }
    for head in ["l1unit", "l2unit", "softmax", "logsoftmax"] {
        let out = emit_len0(head, "reals");
        assert!(
            out.contains("-> tensor<0xf32>"),
            "`{head}` over an empty vector must return an empty vector:\n{out}"
        );
    }
    // The two scalar-valued norms reduce to a scalar, not to an empty vector.
    for head in ["l1norm", "l2norm"] {
        let out = emit_len0(head, "reals");
        assert!(
            out.contains("-> tensor<f32>"),
            "`{head}` over an empty vector must return the empty sum, a scalar:\n{out}"
        );
    }
}

/// §03's promotion is applied BEFORE the empty-scan shortcut, so an empty
/// BOOLEAN scan still returns the integer element type `infer` gives it
/// (`tensor<0xi32>`, not `tensor<0xi1>`). Ordering-sensitive: returning the
/// operand before converting would have disagreed with the inferred type.
#[test]
fn an_empty_boolean_scan_keeps_the_promoted_integer_type() {
    for head in ["cumsum", "cumprod"] {
        let out = emit_len0(head, "booleans");
        assert!(
            out.contains("stablehlo.convert %arg0 : (tensor<0xi1>) -> tensor<0xi32>")
                && out.contains("-> tensor<0xi32>"),
            "`{head}` over an empty boolean vector must still promote to i32:\n{out}"
        );
    }
}

/// The guard on the constructed-operand detour above: the SOURCE spelling these
/// tests used to carry is now a static error, so [`zero_the_extents`] is
/// necessary rather than decorative. If the positivity rule were ever reverted,
/// this fails and the helpers can go back to plain source.
#[test]
fn a_written_zero_size_is_refused() {
    let mut m = flatppl_syntax::parse(&vec_src("cumsum", "reals", 0)).expect("parse");
    let msgs: Vec<String> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        msgs.iter()
            .any(|m| m.contains("`cartpow`'s `size` is written as `0`")
                && m.contains("§03 \"Cartesian power\"")),
        "a written zero size must be refused citing §03's positive size: {msgs:?}"
    );
}

/// §07 lists `prod`/`mean`/`var`/`std` under reductions over ARRAYS, so a scalar
/// operand is refused — the same guard `crate::ops::lower_extremum` puts on
/// `maximum`/`minimum`. (`sum` returns a scalar operand unchanged instead; that
/// asymmetry is pre-existing and untouched here.)
#[test]
fn the_bare_reductions_refuse_a_scalar_operand() {
    for head in ["prod", "mean", "var", "std"] {
        let src = format!("x = elementof(reals)\ny = {head}(x)\ninputs = (x)\noutputs = (y)\n");
        let err = emit_err(&src);
        assert!(
            err.contains(head) && err.contains("must be a statically-shaped array"),
            "`{head}` must refuse a scalar, got: {err}"
        );
    }
}

/// §07 defines `var` with the $n-1$ denominator, so a one-element operand is
/// $0/0$. §04 "Relationship to broadcasting" states the exclusion directly —
/// the broadcast equivalence holds "for every eligible `f_reduction` that is the
/// identity on a one-element input; `var` and `std` are undefined over a single
/// element". `crate::aggregate::reduce` refuses the identical case, and the
/// determiniser passes a length-1 `var(v)` straight through, so the refusal has
/// to land in the emitter.
#[test]
fn var_and_std_refuse_a_single_element_but_mean_does_not() {
    for head in ["var", "std"] {
        let err = emit_err(&vec_src(head, "reals", 1));
        assert!(
            err.contains("over 1 element(s) this is undefined")
                && err.contains("$n-1$ denominator"),
            "`{head}` must refuse one element citing the n-1 denominator, got: {err}"
        );
    }
    // `mean` IS the identity on a one-element input, so it lowers.
    let out = emit(&vec_src("mean", "reals", 1));
    assert!(
        out.contains("stablehlo.divide"),
        "mean over one element must still lower:\n{out}"
    );
}

/// A moment over an EMPTY array divides by zero, so it refuses. `prod` does not:
/// the empty product is the multiplicative identity, which the reduce already
/// emits — EXECUTED, `prod` over a length-0 real vector returns `1.0`.
///
/// The operand is CONSTRUCTED rather than spelled ([`zero_the_extents`]): a
/// written `cartpow(reals, [0])` is a static error since the positivity rule, so
/// the refusal is no longer reachable from surface source. It stays reachable at
/// runtime through a derived length, which is why the emitter still needs it.
#[test]
fn the_moments_refuse_an_empty_array_but_prod_returns_its_identity() {
    for head in ["mean", "var", "std"] {
        let err = emit_err_len0(head, "reals");
        assert!(
            err.contains("over an empty array this is undefined")
                && err.contains("divides by the element count"),
            "`{head}` must refuse an empty array, got: {err}"
        );
    }
    let out = emit_len0("prod", "reals");
    assert!(
        out.contains("stablehlo.constant dense<1.000000e+00>") && out.contains("tensor<0xf32>"),
        "prod over an empty array is the empty product, not a refusal:\n{out}"
    );
}

/// §07 "Table reductions": applied to a table these reduce column-wise "and
/// return a record whose fields are the column names and values are the
/// per-column reductions". This emitter has no record value, so the result is
/// not expressible and every reduction head refuses with that reason — not with
/// whatever its argument's own lowering would have said first.
///
/// `sum` already did this at 50194b2; `prod`/`mean`/`var`/`std` reached
/// "unsupported builtin head" for every argument type and so needed no table
/// guard until this wave wired them.
#[test]
fn every_reduction_head_refuses_a_table_with_the_section_07_reason() {
    for head in ["sum", "prod", "mean", "var", "std"] {
        for (label, decl) in [
            (
                "load_data",
                "data = load_data(\"d.csv\", cartpow(cartprod(x = reals), 4))",
            ),
            (
                "elementof",
                "data = elementof(cartpow(cartprod(x = reals), 4))",
            ),
        ] {
            let src = format!(
                "alpha = elementof(reals)\n{decl}\n\
                 lp = logdensityof(lawof(record(y = draw(Normal(mu = alpha, sigma = 1.0)))), \
                 record(y = get({head}(data), [\"x\"])))\n\
                 inputs = (alpha, data)\noutputs = (lp)\n"
            );
            let err = emit_err(&src);
            assert!(
                err.contains("a table reduction has no tensor form")
                    && err.contains(&format!("{head}(data.x)")),
                "the {label} table must give `{head}` the reduction refusal naming the \
                 column spelling, got: {err}"
            );
        }
    }
}
