//! Golden and refusal tests for the wire-an-existing-helper batch: the §07
//! heads that refused only because `crate::ops`'s builtin map had no entry,
//! while an `Emitter` helper or a one-to-one StableHLO/CHLO op already existed.
//!
//! Deliberately a separate file from `tests/golden.rs` and
//! `tests/golden_stack.rs`, for the reason `golden_stack.rs` gives: several
//! emitter waves append tests concurrently and one file per wave keeps the
//! textual conflict surface at zero. The helpers below are local copies for the
//! same reason.
//!
//! **Every frozen `.mlir` golden here was EXECUTED**, not just pinned:
//! compiled with `iree-base-compiler` 3.11 (llvm-cpu) and run under
//! `iree-base-runtime` (local-task), with the results matched against
//! numpy/scipy oracles. The goldens are read off disk by these tests, so the
//! pinned text and the verified numbers cannot diverge. The one head with no
//! executed number is `lower_cholesky`, and the reason is recorded on its test.
//! See
//! `.superpowers/sdd/2026-08-05-joint-constructs-the-joint/wave-hlowire-report.md`.

use flatppl_core::Module;

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    m
}

/// Determinize with the `inputs`/`outputs` ABI bindings as the roots — the
/// shape every model here takes, since each head is exercised over a runtime
/// (`elementof`) operand rather than a literal that could be constant-folded.
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

/// The emitter's refusal message for a source it declines to lower. `infer` and
/// the determiniser must both be clean, so the refusal is this crate's.
fn emit_err(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_abi(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse in the emitter")
    .msg
}

/// One scalar `elementof(reals)` argument through `head`, as the sole output.
fn unary_real_src(head: &str) -> String {
    format!("x = elementof(reals)\ny = {head}(x)\ninputs = (x)\noutputs = (y)\n")
}

// ---- §07 elementary functions ------------------------------------------------

/// Every §07 "Elementary functions" entry this wave wired emits its intended op.
/// One test over the whole batch rather than eleven: the point is the map entry
/// and the op text, and a table keeps a later addition one line.
///
/// EXECUTED against scipy/numpy per head (`np.sin`, `np.ceil`, `np.log10`,
/// `scipy.special.gammaln`, `scipy.special.gamma`, …) at f32 tolerance; see the
/// wave report's per-head table.
#[test]
fn elementary_heads_emit_their_ops() {
    for (head, op) in [
        ("sin", "stablehlo.sine"),
        ("floor", "stablehlo.floor"),
        ("ceil", "stablehlo.ceil"),
        ("asin", "chlo.asin"),
        ("acos", "chlo.acos"),
        ("acosh", "chlo.acosh"),
        ("loggamma", "chlo.lgamma"),
    ] {
        let out = emit(&unary_real_src(head));
        assert!(out.contains(op), "`{head}` must emit `{op}`:\n{out}");
    }
}

/// §07 `log10`, $\log_{10}(x)$ — `log(x)` DIVIDED by the `ln(10)` constant, not
/// multiplied by its reciprocal (`1/ln(10)` is inexact, so the multiply would
/// add a second rounding). The constant is pinned to full f64 digits: the
/// emitter renders one literal and `Dtype` decides the precision at parse time.
#[test]
fn log10_divides_by_the_ln10_constant() {
    let out = emit(&unary_real_src("log10"));
    assert!(
        out.contains("stablehlo.log %arg0"),
        "must log first:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<2.302585092994046>"),
        "must carry ln(10) as a literal:\n{out}"
    );
    assert!(
        out.contains("stablehlo.divide"),
        "must DIVIDE by ln(10), not multiply by 1/ln(10):\n{out}"
    );
}

/// §07 `abs2`, $\vert x\vert^2$ — `x * x` over the reals this crate emits, so
/// ONE multiply and no `stablehlo.abs` at all (squaring already discards the
/// sign, and `abs` first would be a wasted op).
#[test]
fn abs2_is_one_multiply_with_no_abs() {
    let out = emit(&unary_real_src("abs2"));
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        1,
        "exactly one multiply:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.abs"),
        "squaring discards the sign, so no abs is needed:\n{out}"
    );
}

/// §07 `gamma`, $\Gamma(x)$ over `posreals` — `exp(lgamma(x))`, with NO sign
/// correction. `chlo.lgamma` is $\log\vert\Gamma\vert$, so `exp` of it is
/// $\vert\Gamma\vert$; on §07's stated domain $\Gamma > 0$, which makes the two
/// equal. A sign fix would be code for arguments §07 puts out of domain.
#[test]
fn gamma_is_exp_of_lgamma_with_no_sign_fix() {
    let out = emit(&unary_real_src("gamma"));
    assert!(out.contains("chlo.lgamma"), "must use lgamma:\n{out}");
    assert!(
        out.contains("stablehlo.exponential"),
        "must exponentiate it:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.sign") && !out.contains("stablehlo.select"),
        "no sign correction on §07's `posreals` domain:\n{out}"
    );
}

/// §07 `identity(x)` "returns `x` unchanged", domain "any" — so the lowering is
/// the operand itself and the function body is a bare `return`.
#[test]
fn identity_emits_no_op_at_all() {
    let out = emit(&unary_real_src("identity"));
    assert!(
        out.contains("func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32>"),
        "argument passes straight through:\n{out}"
    );
    assert!(
        !out.contains("stablehlo."),
        "identity emits no StableHLO op:\n{out}"
    );
}

/// §07's BINARY `min`/`max` ($\min(a, b)$ / $\max(a, b)$) and its
/// same-named-family `minimum`/`maximum` REDUCTIONS ($\min_i x_i$) are
/// different functions with different arities, and both lower. One test so they
/// cannot drift into each other: the binary pair takes two scalars, the
/// reductions one array.
#[test]
fn binary_min_max_and_the_reductions_are_distinct_heads() {
    let binary = "a = elementof(reals)\nb = elementof(reals)\ny = min(a, b)\nz = max(a, b)\n\
                  inputs = (a, b)\noutputs = (y, z)\n";
    let out = emit(binary);
    assert!(
        out.contains("stablehlo.minimum %arg0, %arg1")
            && out.contains("stablehlo.maximum %arg0, %arg1"),
        "the binary pair is one op each over two scalars:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.reduce"),
        "the binary pair must NOT reduce:\n{out}"
    );

    let reduction = "v = elementof(cartpow(reals, 3))\ny = minimum(v)\nz = maximum(v)\n\
                     inputs = (v)\noutputs = (y, z)\n";
    let out = emit(reduction);
    assert_eq!(
        out.matches("stablehlo.reduce(").count(),
        2,
        "the reductions are one `stablehlo.reduce` each:\n{out}"
    );
}

/// A `Bool` operand refuses for the binary `min`/`max`: §07's domain there is
/// `reals`, and over booleans `stablehlo.minimum`/`maximum` is a conjunction /
/// disjunction — §07's `land`/`lor`, a different function.
#[test]
fn binary_min_refuses_a_boolean_operand() {
    let src = "a = elementof(booleans)\nb = elementof(booleans)\ny = min(a, b)\n\
               inputs = (a, b)\noutputs = (y)\n";
    let msg = emit_err(src);
    assert!(
        msg.contains("min/max") && msg.contains("conjunction"),
        "unexpected message: {msg}"
    );
}

/// §07 `atan2(y, x)` carries an ORIGIN GATE, and the gate is normative rather
/// than defensive: §07 states "`atan2(0, 0)` returns `0`", and the bare
/// `stablehlo.atan2` does not deliver it — compiled through
/// `iree-base-compiler` 3.11 (llvm-cpu) it returns **NaN** at the origin, since
/// the pipeline lowers it as `atan(y/x)` with a quadrant fixup and `0/0` is NaN
/// before the fixup runs. Measured; every other quadrant and both axes already
/// match `np.arctan2` to f32.
#[test]
fn atan2_gates_the_origin_to_zero() {
    let src = "y = elementof(reals)\nx = elementof(reals)\nr = atan2(y, x)\n\
               inputs = (y, x)\noutputs = (r)\n";
    let out = emit(src);
    assert!(
        out.contains("stablehlo.atan2"),
        "must use the core op:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.compare EQ").count(),
        2,
        "both operands are tested against zero:\n{out}"
    );
    assert!(
        out.contains("stablehlo.and") && out.contains("stablehlo.select"),
        "the two zero tests conjoin and select the §07 value:\n{out}"
    );
}

/// `Emitter::atan` (the §06 change-of-variables head) must NOT pick up
/// `atan2`'s origin gate: its `x` is the constant `1`, so the origin is
/// unreachable and the gate would be three dead ops in every `pushfwd(atan, …)`
/// density. Pins that the shared op helper did not get widened underneath it.
#[test]
fn atan_keeps_its_ungated_single_op_form() {
    let out = emit(&unary_real_src("atan"));
    assert!(
        out.contains("stablehlo.atan2"),
        "atan is atan2(x, 1):\n{out}"
    );
    assert!(
        !out.contains("stablehlo.select"),
        "no origin gate on `atan`:\n{out}"
    );
}

/// The frozen multi-head elementwise module — the EXECUTED artifact, read off
/// disk. At `x = -1.5` it computes `sin(x)`, `ceil(floor(x))`, `log10(abs2(x))`,
/// `gamma(abs2(x))` and `max(min(x, 2), -2)`, matched against
/// numpy/`scipy.special` at f32 tolerance.
#[test]
fn elementwise_module_matches_frozen_golden() {
    let src = "\
x = elementof(reals)
o1 = sin(x)
o2 = ceil(floor(x))
o3 = log10(abs2(x))
o4 = gamma(abs2(x))
o5 = max(min(x, 2.0), -2.0)
inputs = (x)
outputs = (o1, o2, o3, o4, o5)
";
    assert_eq!(
        emit(src),
        include_str!("goldens/wire_elementwise.mlir"),
        "emitted @logdensity drifted from tests/goldens/wire_elementwise.mlir"
    );
}

// ---- §07 comparison functions and logical operators -------------------------

/// §07 `equal`/`unequal` — one `stablehlo.compare EQ`/`NE`, with `SIGNED` on the
/// integer operands §07's domain admits.
#[test]
fn equal_and_unequal_compare_integers() {
    for (head, dir) in [("equal", "EQ"), ("unequal", "NE")] {
        let src = format!(
            "a = elementof(integers)\nb = elementof(integers)\ny = {head}(a, b)\n\
             inputs = (a, b)\noutputs = (y)\n"
        );
        let out = emit(&src);
        assert!(
            out.contains(&format!(
                "stablehlo.compare {dir}, %arg0, %arg1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>"
            )),
            "`{head}` must be a signed integer compare {dir}:\n{out}"
        );
    }
}

/// A REAL operand refuses. §07 gives `equal`/`unequal` the domain "`integers`,
/// `booleans`, strings" and states the reason: exact equality "is restricted to
/// discrete domains to avoid dependence on numerical precision". Emitting a
/// float `compare EQ` would answer a question §07 declines to define, so the
/// refusal is the lowering — and it points at the four discretizing functions
/// §07 names, plus `iszero`, which §07 does define for a non-discrete input.
#[test]
fn equal_refuses_real_operands_and_names_the_alternatives() {
    let src = "a = elementof(reals)\nb = elementof(reals)\ny = equal(a, b)\n\
               inputs = (a, b)\noutputs = (y)\n";
    let msg = emit_err(src);
    assert!(
        msg.contains("discrete domains"),
        "unexpected message: {msg}"
    );
    for alt in ["integer", "floor", "ceil", "round", "iszero"] {
        assert!(
            msg.contains(alt),
            "the refusal must name `{alt}` as the way out: {msg}"
        );
    }
}

/// `iszero` still lowers over the SAME real operand `equal` refuses — §07:
/// "`iszero(x)`, unlike `x == 0`, allows non-discrete inputs". One test so the
/// new `equal` guard cannot be widened into `iszero`'s territory.
#[test]
fn iszero_still_accepts_the_real_operand_equal_refuses() {
    let out = emit(&unary_real_src("iszero"));
    assert!(
        out.contains("stablehlo.compare EQ, %arg0"),
        "iszero is an exact compare against zero, real operand included:\n{out}"
    );
}

/// §07's four logical operators, each one op. `lor`/`lxor` are pinned on
/// OVERLAPPING predicates (`x < 3`, `x > 1`, both true on `(1, 3)`) — on
/// disjoint predicates a disjunction and an exclusive disjunction agree
/// everywhere, so a disjoint pair would not tell the two apart.
#[test]
fn logical_operators_emit_one_op_each() {
    for (expr, op) in [
        ("land(lt(x, 3.0), gt(x, 1.0))", "stablehlo.and"),
        ("lor(lt(x, 3.0), gt(x, 1.0))", "stablehlo.or"),
        ("lxor(lt(x, 3.0), gt(x, 1.0))", "stablehlo.xor"),
        ("lnot(lt(x, 3.0))", "stablehlo.not"),
    ] {
        let src = format!("x = elementof(reals)\ny = {expr}\ninputs = (x)\noutputs = (y)\n");
        let out = emit(&src);
        assert!(out.contains(op), "`{expr}` must emit `{op}`:\n{out}");
    }
}

/// A boolean-typed result reaching the ABI returns `tensor<i1>`, and the
/// function SIGNATURE agrees with it.
///
/// This is a fix, not just a pin: `Emitter::and`/`or`/`not` tagged their result
/// `ElemKind::Real` while rendering the op text through `render_i1`, so
/// `outputs = (land(…))` emitted `return %4 : tensor<f32>` for an `i1` value —
/// ill-typed MLIR, reachable on `main` with nothing but `land` (measured). The
/// three helpers now tag `ElemKind::Bool`, which is what they emit. No
/// pre-existing golden moved: every other caller feeds a `select` predicate or a
/// `while` condition, both of which render through `render_i1` regardless of the
/// tag.
#[test]
fn a_boolean_output_is_typed_i1_end_to_end() {
    for expr in [
        "land(lt(x, 3.0), gt(x, 1.0))",
        "lor(lt(x, 3.0), gt(x, 1.0))",
        "lxor(lt(x, 3.0), gt(x, 1.0))",
        "lnot(lt(x, 3.0))",
        "isnan(x)",
    ] {
        let src = format!("x = elementof(reals)\ny = {expr}\ninputs = (x)\noutputs = (y)\n");
        let out = emit(&src);
        assert!(
            out.contains("func.func @logdensity(%arg0: tensor<f32>) -> tensor<i1>"),
            "`{expr}` must declare an i1 result:\n{out}"
        );
        assert!(
            out.contains("return %") && out.contains(" : tensor<i1>\n"),
            "`{expr}` must return it as i1:\n{out}"
        );
        assert!(
            !out.contains("-> tensor<f32> {"),
            "`{expr}` must not declare an f32 result for an i1 value:\n{out}"
        );
    }
}

/// The connectives keep `land`'s narrow operand rule: the operand must be a
/// boolean-producing CALL node, so a `Bool`-typed ABI input still refuses. That
/// gap (a boolean VALUE gating a conditional) is deliberately left open and
/// documented rather than half-closed here — see
/// `flatppl-dev/stablehlo-feature-matrix.md`'s prioritized gap 6.
#[test]
fn a_boolean_value_still_cannot_gate_a_connective() {
    let src = "p = elementof(booleans)\nq = elementof(booleans)\ny = lor(p, q)\n\
               inputs = (p, q)\noutputs = (y)\n";
    let msg = emit_err(src);
    assert!(
        msg.contains("lor operand must be a boolean predicate"),
        "unexpected message: {msg}"
    );
}

/// The newly wired boolean heads are recognized as `ifelse` conditions and as
/// connective operands, because each lowers to an `i1`. Before this wave the
/// predicate vocabulary was `in`/compare/`land`/`iszero` only, so `equal`,
/// `lor`, `lnot`, `lxor` and the `isfinite` family were all rejected as
/// conditions even once they lowered.
#[test]
fn the_new_boolean_heads_can_gate_an_ifelse() {
    for cond in [
        "equal(a, b)",
        "unequal(a, b)",
        "lor(lt(x, 1.0), gt(x, 2.0))",
        "lxor(lt(x, 1.0), gt(x, 2.0))",
        "lnot(lt(x, 1.0))",
        "isfinite(x)",
        "isinf(x)",
        "isnan(x)",
    ] {
        let src = format!(
            "a = elementof(integers)\nb = elementof(integers)\nx = elementof(reals)\n\
             y = ifelse({cond}, 1.0, 2.0)\ninputs = (a, b, x)\noutputs = (y)\n"
        );
        let out = emit(&src);
        assert!(
            out.contains("stablehlo.select"),
            "`{cond}` must be accepted as an ifelse condition:\n{out}"
        );
    }
}

/// §07's three finiteness predicates, composed out of ops this crate has
/// already validated rather than the `chlo.is_inf` family:
///
/// - `isnan(x)` is `x != x`, the IEEE-754 definition;
/// - `isinf(x)` is `abs(x) == inf`, true for both signs, false for NaN;
/// - `isfinite(x)` is `abs(x) < inf`, false for ±∞ AND false for NaN (an
///   unordered comparison is false) — exactly §07's "not ±∞, not NaN".
///
/// All three EXECUTED over `{0, ±finite, +inf, -inf, NaN}` and matched against
/// `np.isfinite`/`np.isinf`/`np.isnan`.
#[test]
fn finiteness_predicates_use_validated_ops_only() {
    let nan = emit(&unary_real_src("isnan"));
    assert!(
        nan.contains("stablehlo.compare NE, %arg0, %arg0"),
        "isnan is the IEEE self-inequality:\n{nan}"
    );
    assert!(
        !nan.contains("chlo."),
        "no CHLO op is needed for isnan:\n{nan}"
    );

    for (head, dir) in [("isinf", "EQ"), ("isfinite", "LT")] {
        let out = emit(&unary_real_src(head));
        assert!(
            out.contains("stablehlo.abs %arg0"),
            "`{head}` takes |x|:\n{out}"
        );
        assert!(
            out.contains("stablehlo.constant dense<0x7F800000>"),
            "`{head}` compares against the +inf bit pattern:\n{out}"
        );
        assert!(
            out.contains(&format!("stablehlo.compare {dir}")),
            "`{head}` must compare {dir}:\n{out}"
        );
        assert!(
            !out.contains("chlo."),
            "no CHLO op is needed for `{head}`:\n{out}"
        );
    }
}

/// The frozen logic module — the EXECUTED artifact, read off disk. It nests all
/// six new boolean heads under `land` and gates an `ifelse` with the result:
/// `land(lor(equal(a, b), lnot(isnan(x))), lxor(isfinite(x), isinf(x)))`, which
/// reduces to "`x` is not NaN". Run over `x` in `{1.5, inf, NaN}` crossed with
/// `a == b` and `a != b`; all six results match that reduction.
#[test]
fn logic_module_matches_frozen_golden() {
    let src = "\
a = elementof(integers)
b = elementof(integers)
x = elementof(reals)
y = ifelse(land(lor(equal(a, b), lnot(isnan(x))), lxor(isfinite(x), isinf(x))), 10.0, 20.0)
inputs = (a, b, x)
outputs = (y)
";
    assert_eq!(
        emit(src),
        include_str!("goldens/wire_logic.mlir"),
        "emitted @logdensity drifted from tests/goldens/wire_logic.mlir"
    );
}

// ---- §07 linear algebra -------------------------------------------------------

/// A square real matrix and a matching vector as ABI arguments — the operand
/// shape every §07 linear-algebra head below takes.
const MAT_VEC_ABI: &str = "\
A = elementof(cartpow(cartpow(reals, 3), 3))
v = elementof(cartpow(reals, 3))
";

/// §07 `trace(A)` = $\mathrm{tr}(\mathbf{A})$ — the diagonal extraction summed.
/// `Emitter::diag` has no native StableHLO op behind it, so the diagonal is the
/// iota/compare/select/row-sum idiom the multivariate densities already use;
/// `trace` adds the second reduction that collapses the resulting vector.
#[test]
fn trace_sums_the_extracted_diagonal() {
    let out = emit(&format!(
        "{MAT_VEC_ABI}t = trace(A)\ninputs = (A, v)\noutputs = (t)\n"
    ));
    assert_eq!(
        out.matches("stablehlo.iota").count(),
        2,
        "row and column index tensors:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.reduce(").count(),
        2,
        "one reduce for the diagonal, one for its sum:\n{out}"
    );
    assert!(
        out.contains("-> tensor<f32>"),
        "the trace is a scalar:\n{out}"
    );
}

/// §07 `diag(A, k)` "extracts the $k$th diagonal of $\mathbf{A}$ as a vector",
/// and `diag(A)` defaults `k` to `0`. Both spellings of the main diagonal emit
/// the same thing, and the result is a length-`n` vector rather than a matrix.
#[test]
fn diag_lowers_both_spellings_of_the_main_diagonal() {
    let implicit = emit(&format!(
        "{MAT_VEC_ABI}d = diag(A)\ninputs = (A, v)\noutputs = (d)\n"
    ));
    let explicit = emit(&format!(
        "{MAT_VEC_ABI}d = diag(A, 0)\ninputs = (A, v)\noutputs = (d)\n"
    ));
    assert_eq!(
        implicit, explicit,
        "§07 makes `diag(A)` exactly `diag(A, 0)`"
    );
    assert!(
        implicit.contains("-> tensor<3xf32>"),
        "a 3x3 matrix's diagonal is a length-3 vector:\n{implicit}"
    );
}

/// A non-zero `k` refuses. §07 defines super- and sub-diagonals, but
/// `Emitter::diag` masks on `row == col`; a shifted diagonal needs a shifted
/// mask AND a shorter result, which is a different lowering rather than a
/// parameter of this one. PARTIAL, stated, not approximated.
#[test]
fn diag_refuses_a_non_zero_offset() {
    let msg = emit_err(&format!(
        "{MAT_VEC_ABI}d = diag(A, 2)\ninputs = (A, v)\noutputs = (d)\n"
    ));
    assert!(
        msg.contains("only the MAIN diagonal") && msg.contains("k = 2"),
        "unexpected message: {msg}"
    );
}

/// `diag` and `trace` refuse a NON-SQUARE matrix even though §07's `diag` domain
/// is "matrices" (only `trace` says "square matrices"). `Emitter::diag` row-sums
/// an `n`-column mask, so on an `m`x`n` operand with `m > n` it would return `m`
/// entries — zeros for the rows the diagonal never reaches — instead of the
/// `min(m, n)` §07 defines. Refusing keeps that from being silently answered.
#[test]
fn diag_and_trace_refuse_a_non_square_matrix() {
    let abi = "M = elementof(cartpow(cartpow(reals, 2), 3))\n";
    for head in ["diag", "trace"] {
        let msg = emit_err(&format!(
            "{abi}y = {head}(M)\ninputs = (M)\noutputs = (y)\n"
        ));
        assert!(
            msg.contains("square matrices") && msg.contains("3x2"),
            "`{head}` on a 3x2: unexpected message: {msg}"
        );
    }
}

/// `diag`, `trace` and `lower_cholesky` refuse a non-`Real` matrix. This is the
/// Real-hardcode family: `Emitter::diag` renders its iota/mask matrices and its
/// reduction identity as floats unconditionally, and `Emitter::cholesky` renders
/// `stablehlo.cholesky` at the operand's own kind while tagging the result
/// `Real` — so an integer operand would emit a self-contradictory module IREE
/// rejects outright. The head refuses instead of reaching the helper.
#[test]
fn the_real_hardcode_family_refuses_an_integer_matrix() {
    let abi = "N = elementof(cartpow(cartpow(integers, 3), 3))\n";
    for head in ["diag", "trace", "lower_cholesky"] {
        let msg = emit_err(&format!(
            "{abi}y = {head}(N)\ninputs = (N)\noutputs = (y)\n"
        ));
        assert!(
            msg.contains("only a real matrix is supported") && msg.contains("Int"),
            "`{head}` on an integer matrix: unexpected message: {msg}"
        );
    }
}

/// §07 `lower_cholesky(A)` — one `stablehlo.cholesky` with `lower = true`, the
/// factor convention §07 fixes ("lower-triangular $\mathbf{L}$ with
/// $\mathbf{A} = \mathbf{L}\mathbf{L}^\dagger$").
///
/// **NOT numerically executed, and the reason is pre-existing.**
/// `iree-base-compiler` 3.11 refuses to legalize `stablehlo.cholesky` at all
/// ("failed to legalize operation 'stablehlo.cholesky' that was explicitly
/// marked illegal"), so no IREE run of this head is possible. That is not
/// something this head introduces: the already-frozen
/// `goldens/mvnormal_logdensity.mlir` carries the identical op and fails the
/// same way (measured). The module here PARSES and VERIFIES under MLIR, and the
/// op is spec-legal StableHLO; the numeric check stays UN-VERIFIED against an
/// executed oracle until IREE grows the lowering or the emitter composes the
/// factorization itself.
#[test]
fn lower_cholesky_emits_the_lower_factor() {
    let out = emit(&format!(
        "{MAT_VEC_ABI}L = lower_cholesky(A)\ninputs = (A, v)\noutputs = (L)\n"
    ));
    assert!(
        out.contains("stablehlo.cholesky %arg0, lower = true : tensor<3x3xf32>"),
        "§07 fixes the LOWER factor:\n{out}"
    );
    assert!(
        out.contains("-> tensor<3x3xf32>"),
        "the factor is shape-preserving:\n{out}"
    );
}

/// §07 `self_outer(x)` = "$\mathbf{x} \cdot \mathbf{x}^\dagger$", domain
/// "vectors" — the operand against ITSELF, so both `broadcast_in_dim`s spread
/// `%arg`, one along each axis, and the result is square.
#[test]
fn self_outer_spreads_one_operand_along_both_axes() {
    let out =
        emit("v = elementof(cartpow(reals, 3))\nS = self_outer(v)\ninputs = (v)\noutputs = (S)\n");
    assert!(
        out.contains("stablehlo.broadcast_in_dim %arg0, dims = [0]")
            && out.contains("stablehlo.broadcast_in_dim %arg0, dims = [1]"),
        "the same operand spreads along both axes:\n{out}"
    );
    assert!(
        out.contains("-> tensor<3x3xf32>"),
        "a length-3 vector's self outer product is 3x3:\n{out}"
    );
}

/// §07 `row_gram(A)` = $\mathbf{A}\mathbf{A}^\dagger$ and `col_gram(A)` =
/// $\mathbf{A}^\dagger\mathbf{A}$ are DIFFERENT matrices, and on a non-square
/// operand they do not even share a shape: a 3x2 `A` gives a 3x3 row Gram and a
/// 2x2 col Gram. One test so the two cannot drift into each other — a symmetric
/// square operand would not tell them apart.
#[test]
fn row_gram_and_col_gram_have_different_shapes() {
    let out = emit(
        "M = elementof(cartpow(cartpow(reals, 2), 3))\nG = row_gram(M)\nH = col_gram(M)\n\
         inputs = (M)\noutputs = (G, H)\n",
    );
    assert!(
        out.contains("-> (tensor<3x3xf32>, tensor<2x2xf32>)"),
        "row Gram 3x3, col Gram 2x2 for a 3x2 operand:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.dot_general").count(),
        2,
        "one product each:\n{out}"
    );
}

/// §07 `quadform(A, x)` = "$\mathbf{x}^\dagger \mathbf{A} \mathbf{x}$" —
/// associated as $\mathbf{x}^\dagger(\mathbf{A}\mathbf{x})$, so TWO
/// `dot_general`s (a mat-vec then an inner product) rather than the three a
/// left-to-right association would need, and a scalar result.
#[test]
fn quadform_is_two_dot_generals_and_a_scalar() {
    let out = emit(&format!(
        "{MAT_VEC_ABI}q = quadform(A, v)\ninputs = (A, v)\noutputs = (q)\n"
    ));
    assert_eq!(
        out.matches("stablehlo.dot_general").count(),
        2,
        "mat-vec then inner product:\n{out}"
    );
    assert!(
        out.contains("contracting_dims = [1] x [0]")
            && out.contains("contracting_dims = [0] x [0]"),
        "the mat-vec contracts A's trailing axis, the inner product both axis 0:\n{out}"
    );
    assert!(
        out.contains(") -> tensor<f32> {"),
        "a quadratic form is a scalar:\n{out}"
    );
}

/// A `quadform` whose vector length disagrees with `A`'s order refuses rather
/// than reaching `Emitter::matvec`, which PANICS on a contracting-dim mismatch.
#[test]
fn quadform_refuses_a_mismatched_vector_length() {
    let src = "A = elementof(cartpow(cartpow(reals, 3), 3))\nw = elementof(cartpow(reals, 2))\n\
               q = quadform(A, w)\ninputs = (A, w)\noutputs = (q)\n";
    let msg = emit_err(src);
    assert!(
        msg.contains("quadform") && msg.contains("length 3"),
        "unexpected message: {msg}"
    );
}

/// The rank guards, one per head: §07's domains are matrices and vectors, so a
/// vector where a matrix belongs (and the reverse) refuses instead of reaching a
/// helper that would panic on the rank.
#[test]
fn the_linalg_heads_refuse_the_wrong_rank() {
    let vec_abi = "v = elementof(cartpow(reals, 3))\n";
    for head in ["trace", "diag", "lower_cholesky", "row_gram", "col_gram"] {
        let msg = emit_err(&format!(
            "{vec_abi}y = {head}(v)\ninputs = (v)\noutputs = (y)\n"
        ));
        assert!(
            msg.contains("rank-2 operand is required"),
            "`{head}` on a vector: unexpected message: {msg}"
        );
    }
    let mat_abi = "A = elementof(cartpow(cartpow(reals, 3), 3))\n";
    let msg = emit_err(&format!(
        "{mat_abi}y = self_outer(A)\ninputs = (A)\noutputs = (y)\n"
    ));
    assert!(
        msg.contains("self_outer") && msg.contains("rank-1 operand is required"),
        "`self_outer` on a matrix: unexpected message: {msg}"
    );
}

/// The frozen linear-algebra module — the EXECUTED artifact, read off disk. Six
/// outputs from one 3x3 `A` and one length-3 `v`: `trace`, `diag`, `quadform`,
/// `self_outer`, `row_gram`, `col_gram`. Run over a symmetric positive-definite
/// `A` AND a non-symmetric one (which is what separates the two Grams), matched
/// against `np.trace`, `np.diag`, `v @ A @ v`, `np.outer`, `A @ A.T` and
/// `A.T @ A`.
#[test]
fn linalg_module_matches_frozen_golden() {
    let src = "\
A = elementof(cartpow(cartpow(reals, 3), 3))
v = elementof(cartpow(reals, 3))
t = trace(A)
d = diag(A)
q = quadform(A, v)
S = self_outer(v)
G = row_gram(A)
H = col_gram(A)
inputs = (A, v)
outputs = (t, d, q, S, G, H)
";
    assert_eq!(
        emit(src),
        include_str!("goldens/wire_linalg.mlir"),
        "emitted @logdensity drifted from tests/goldens/wire_linalg.mlir"
    );
}
