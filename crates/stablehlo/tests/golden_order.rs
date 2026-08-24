//! Golden and refusal tests for `crate::order` — spec §07 "Boolean reductions",
//! "Cumulative operations", the infinity norm, and the two order statistics.
//!
//! Normative source: flatppl-design branch `missing-reductions` @ `ee4c6fb`,
//! docs/07 + docs/04. Landed ahead of the spec merge per the owner's ruling.
//!
//! A separate file from `golden.rs`/`golden_norms.rs`/`golden_stack.rs`/
//! `golden_wire.rs` for the reason `golden_stack.rs` gives: several emitter waves
//! append tests concurrently, and one file per wave keeps the textual conflict
//! surface at zero. The helpers below are local copies for the same reason.
//!
//! **Every frozen `.mlir` golden here was EXECUTED**, not merely pinned: compiled
//! with `iree-base-compiler` 3.11 (llvm-cpu), run under `iree-base-runtime`
//! (local-task), and every output matched against a numpy oracle. `Dtype::F32`, so
//! the reals are `f32`; every comparison below was EXACT (these heads select or
//! combine elements and introduce no rounding), so no tolerance was needed.
//! Every operand arrives as a runtime `elementof` argument rather than a literal, so
//! nothing is constant-folded away — and IREE 3.11 miscompiles a splat-constant-fed
//! reduction, which a literal operand would have walked into. The goldens are read
//! off disk by these tests, so the pinned text and the verified numbers cannot
//! diverge.
//!
//! Executed oracle table, `v = [1.5, -2.0, 3.25, 0.5]` and
//! `b = [true, false, true, true]`:
//!
//! | call | executed | oracle |
//! |---|---|---|
//! | `cummax(v)` | `[1.5, 1.5, 3.25, 3.25]` | `np.maximum.accumulate` |
//! | `cummin(v)` | `[1.5, -2.0, -2.0, -2.0]` | `np.minimum.accumulate` |
//! | `linfnorm(v)` | `3.25` | `np.linalg.norm(v, inf)` |
//! | `lany(b)` / `lall(b)` | `true` / `false` | `np.any` / `np.all` |
//! | `lany(v .> 3.0)` / `lall(v .> 3.0)` | `true` / `false` | mask `[F, F, T, F]` |
//! | `cummax([3, 1, 7, 5])` (i32) | `[3, 3, 7, 7]` | `np.maximum.accumulate` |
//! | `cummin([-3, -7, -1, -5])` | `[-3, -7, -7, -7]` | `np.minimum.accumulate` |
//! | `cummax([NaN, 1, 2, 3])` | `[NaN, NaN, NaN, NaN]` | `np.maximum.accumulate` |
//! | `cummax([1, NaN, 2, 3])` | `[1, NaN, NaN, NaN]` | `np.maximum.accumulate` |
//! | `linfnorm([NaN, 1, 2, 3])` | `NaN` | `np.linalg.norm` |
//! | `linfnorm([])` | `0.0` | `np.linalg.norm([], inf)` |
//! | `cummax([])` / `cummin([])` | `[]` / `[]`, shape `(0,)` | the empty prefix sequence |
//! | `lany([])` / `lall([])` | `false` / `true` | `np.any([])` / `np.all([])` |
//! | `ifelse(lany(v .> 3.0), 1, 2)` | `1.0` | `np.any(v > 3.0)` is `True` |
//! | `ifelse(lall(v .> 0.0), 1, 2)` | `2.0` | `np.all(v > 0.0)` is `False` |
//! | `ifelse(land(lany(v .> 3.0), lall(v .> 0.0)), 1, 2)` | `2.0` | `True and False` |
//!
//! The predicate module was executed over four operands, not one:
//! `[0.5, 1, 2, 3]` → `2.0`/`1.0`/`2.0`, `[4, 5, 6, 7]` → `1.0`/`1.0`/`1.0`, and
//! `[-1, -2, -3, -4]` → `2.0`/`2.0`/`2.0`, each matching numpy. All four sign
//! combinations of the two conditions are covered, so a swapped `select` operand
//! order or a flipped reduce identity cannot pass.
//!
//! Two of those rows are worth naming. `cummax([-3, -7, -1, -5])` is
//! `[-3, -3, -1, -1]` and `cummax([-1e30, …])` is `[-1e30, …]`, so the `-inf`
//! window seed never surfaces as a value — and `cummax([1, NaN, 2, 3])` propagating
//! NaN is what proves it: `stablehlo.maximum` is NaN-propagating, so
//! `maximum(-inf, NaN)` is NaN rather than `-inf`. The js engine's `>`-comparison
//! scan SKIPS a mid-input NaN (`[1, 1, 1]` where Julia's `accumulate(max, ·)` gives
//! `[1, NaN, NaN]`); this backend matches the maths, and the divergence is recorded
//! in `flatppl-dev/TODO-flatppl-rust.md` for a §07 ruling.

use flatppl_core::Module;

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
    let m = parse_infer(src);
    let mut m = m;
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

// ---- the five heads that lower ------------------------------------------------

/// Every head this wave WIRES emits, over the §07 domain it declares. The point is
/// the map entry: a missing arm is `lower_builtin`'s catch-all "unsupported builtin
/// head", which is what all seven did at `origin/main` 8bd8331.
#[test]
fn every_wired_head_lowers_over_its_own_domain() {
    for (head, set) in [
        ("cummax", "reals"),
        ("cummin", "reals"),
        ("linfnorm", "reals"),
        ("lany", "booleans"),
        ("lall", "booleans"),
    ] {
        let out = emit(&vec_src(head, set, 4));
        assert!(
            out.contains("func.func @logdensity"),
            "`{head}` must lower:\n{out}"
        );
    }
}

// ---- §07 "Cumulative operations" ----------------------------------------------

/// The frozen `cummax`/`cummin` module — the EXECUTED artifact, read off disk.
#[test]
fn cumulative_extrema_match_the_frozen_golden() {
    let out = emit(
        "\
v = elementof(cartpow(reals, [4]))
o1 = cummax(v)
o2 = cummin(v)
inputs = (v)
outputs = (o1, o2)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_cumulative_extrema.mlir"),
        "emitted cummax/cummin drifted from tests/goldens/order_cumulative_extrema.mlir"
    );
}

/// The scan is the SAME `reduce_window` pass `cumsum`/`cumprod` take — only the
/// combine and the seed differ. Pinned structurally so a future rewrite into an
/// unrolled or `while`-based scan is a visible decision.
#[test]
fn the_cumulative_extrema_are_one_reduce_window_each() {
    for (head, op) in [
        ("cummax", "stablehlo.maximum"),
        ("cummin", "stablehlo.minimum"),
    ] {
        let out = emit(&vec_src(head, "reals", 4));
        assert_eq!(
            out.matches("stablehlo.reduce_window").count(),
            1,
            "`{head}` is one window pass:\n{out}"
        );
        assert!(out.contains(op), "`{head}` combines with {op}:\n{out}");
        assert!(
            out.contains("window_dimensions = array<i64: 4>")
                && out.contains("padding = dense<[[3, 0]]>"),
            "`{head}`'s window must be the full length, left-padded by n - 1:\n{out}"
        );
    }
}

/// The seeds are the combines' IDENTITIES, spelled as the dtype-exact ±inf bit
/// patterns rather than a finite stand-in — a `-1e30` seed would win against any
/// input at or below it. Executed at `[-1e30, -1e30, -1e30, -1e30]`, which returns
/// itself, so the seed never surfaces.
#[test]
fn the_cumulative_extrema_seed_the_window_with_their_identity() {
    let max = emit(&vec_src("cummax", "reals", 4));
    assert!(
        max.contains("dense<0xFF800000>"),
        "cummax seeds -inf:\n{max}"
    );
    let min = emit(&vec_src("cummin", "reals", 4));
    assert!(
        min.contains("dense<0x7F800000>"),
        "cummin seeds +inf:\n{min}"
    );
}

/// An INTEGER vector scans in `i32`, not in the float dtype. `infer`'s
/// `SameAsArg(0)` row types `cummax` of an integer vector an INTEGER vector, so
/// widening to reals would make the emitted return type disagree with the inferred
/// one. The seed is the type's extreme value, there being no integer ±inf; executed
/// with `i32::MIN` and `i32::MAX` IN the data, which the seed does not corrupt.
#[test]
fn an_integer_scan_stays_integer_and_matches_the_frozen_golden() {
    let out = emit(
        "\
v = elementof(cartpow(integers, [4]))
o1 = cummax(v)
o2 = cummin(v)
inputs = (v)
outputs = (o1, o2)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_integer_scan.mlir"),
        "emitted integer scan drifted from tests/goldens/order_integer_scan.mlir"
    );
    assert!(
        out.contains("dense<-2147483648> : tensor<i32>")
            && out.contains("dense<2147483647> : tensor<i32>"),
        "the seeds are i32's extremes:\n{out}"
    );
}

/// A BOOLEAN vector refuses. §03's `booleans ⊂ integers ⊂ reals` puts it in §07's
/// "real vectors" and `max`/`min` over booleans are exactly `lor`/`land`, but IREE
/// 3.11 cannot compile ANY `i1` `reduce_window` region — probed at
/// `stablehlo.or`, `stablehlo.and` and `stablehlo.maximum` alike, each failing with
/// "'arith.ori' op requires the same type for all operands and results". Promoting
/// instead would contradict the boolean result type `infer` gives the call, which is
/// the type/ABI disagreement `infer::ops::refuse_nonscalar_operand` exists to
/// prevent.
#[test]
fn a_boolean_cumulative_extremum_refuses_rather_than_promoting() {
    for head in ["cummax", "cummin"] {
        let msg = emit_err(&vec_src(head, "booleans", 4));
        assert!(
            msg.contains("has no Bool form here") && msg.contains("arith.ori"),
            "`{head}` must refuse a boolean operand and name the limit, got: {msg}"
        );
    }
}

/// §07's Domains cell for both is "real vectors", so a MATRIX has no §07 meaning to
/// lower rather than a meaning this backend declines — the same refusal
/// `cumsum`/`cumprod` give, and for the same reason (§07 pins no scanned axis).
#[test]
fn a_matrix_operand_refuses_for_the_cumulative_extrema() {
    for head in ["cummax", "cummin"] {
        let src = format!(
            "m = elementof(cartpow(reals, [2, 3]))\ny = {head}(m)\ninputs = (m)\noutputs = (y)\n"
        );
        let msg = emit_err(&src);
        assert!(
            msg.contains("the domain \"vectors\""),
            "`{head}` over a matrix must refuse, got: {msg}"
        );
    }
}

// ---- §07 "Norms and normalization": linfnorm ----------------------------------

/// The frozen `linfnorm` module — the EXECUTED artifact. At
/// `v = [1.5, -2.0, 3.25, 0.5]` it computes `3.25`, matched against
/// `np.linalg.norm(v, inf)`.
#[test]
fn linfnorm_matches_the_frozen_golden() {
    let out = emit(
        "\
v = elementof(cartpow(reals, [4]))
o1 = linfnorm(v)
inputs = (v)
outputs = (o1)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_linfnorm.mlir"),
        "emitted linfnorm drifted from tests/goldens/order_linfnorm.mlir"
    );
}

/// $\max_i \lvert v_i \rvert$ is an `abs` then a `maximum` reduce — the shape
/// `l1norm`'s $\sum_i \lvert v_i \rvert$ takes with the combine swapped, and the
/// contrast with `l2norm`, which squares instead and needs no `abs` at all.
#[test]
fn linfnorm_is_an_abs_then_a_maximum_reduce() {
    let out = emit(&vec_src("linfnorm", "reals", 4));
    assert!(
        out.contains("stablehlo.abs"),
        "linfnorm takes moduli:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum"),
        "and reduces with maximum, not add:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.sqrt"),
        "an infinity norm takes no root:\n{out}"
    );
}

/// An integer or boolean vector is WIDENED, not refused: §03 admits both inside
/// §07's "real/complex vectors", and `infer` types the call `Scalar(Real)`, so the
/// emitted return type must be the float dtype whatever arrived. Same treatment as
/// `l1norm`/`l2norm`.
#[test]
fn linfnorm_widens_a_narrower_operand_to_the_float_dtype() {
    for set in ["integers", "booleans"] {
        let out = emit(&vec_src("linfnorm", set, 4));
        assert!(
            out.contains("stablehlo.convert") && out.contains("-> tensor<f32>"),
            "`linfnorm` over {set} widens and returns f32:\n{out}"
        );
    }
}

// ---- §07 "Boolean reductions" -------------------------------------------------

/// The frozen `lany`/`lall` module — the EXECUTED artifact. At
/// `b = [true, false, true, true]` it computes `true` and `false`, matched against
/// `np.any` / `np.all`.
#[test]
fn the_boolean_reductions_match_the_frozen_golden() {
    let out = emit(
        "\
b = elementof(cartpow(booleans, [4]))
o1 = lany(b)
o2 = lall(b)
inputs = (b)
outputs = (o1, o2)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_boolean_reductions.mlir"),
        "emitted lany/lall drifted from tests/goldens/order_boolean_reductions.mlir"
    );
}

/// **The BOOLSUM contrast, and the point of this pair.** §03 "Bool" makes a boolean
/// reach a §07 reduction PROMOTED, because a 1-bit `stablehlo.add` is parity rather
/// than a count — that is why `sum` over a boolean array emits an `i32` reduce.
/// `lany`/`lall` are the opposite case: §07 defines them AS the `lor`- and
/// `land`-reduction, `booleans` is closed under both, and a truth value is not §03's
/// "arithmetic context". So they reduce in `i1` with no `stablehlo.convert` at all,
/// and the module returns `tensor<i1>` — which is what `infer` types the call.
#[test]
fn the_boolean_reductions_reduce_in_i1_with_no_promotion() {
    for (head, op, identity) in [
        ("lany", "stablehlo.or", "false"),
        ("lall", "stablehlo.and", "true"),
    ] {
        let out = emit(&vec_src(head, "booleans", 4));
        assert!(
            out.contains(&format!("applies {op}")),
            "`{head}` combines with {op}:\n{out}"
        );
        assert!(
            out.contains(&format!("dense<{identity}> : tensor<i1>")),
            "`{head}`'s identity is `{identity}` at i1:\n{out}"
        );
        assert!(
            !out.contains("stablehlo.convert"),
            "`{head}` must NOT promote — that is `sum`'s rule, not this one:\n{out}"
        );
        assert!(
            out.contains("-> tensor<i1> {"),
            "`{head}` returns a boolean, agreeing with infer's Scalar(Boolean):\n{out}"
        );
    }
}

/// A boolean reduction is a legal `ifelse` CONDITION — the EXECUTED artifact.
///
/// `lany`/`lall` lower to a scalar `tensor<i1>`, which is exactly what
/// `PREDICATE_HEADS` admits ("Every entry lowers to an `i1`-typed `Value`"), and
/// conditioning on one is the most natural consumer a §07 boolean reduction has. Both
/// were missing from that list, so a bare `lany(v .> 3.0)` lowered as a module OUTPUT
/// while `ifelse(lany(v .> 3.0), 1.0, 2.0)` refused. This is not the `Bool`-typed
/// VALUE the list deliberately excludes (gap 6): it is a call node the map already
/// lowers.
///
/// At `v = [1.5, -2.0, 3.25, 0.5]`: `np.any(v > 3.0)` is `True` → `1.0`,
/// `np.all(v > 0.0)` is `False` → `2.0`, and their `land` is `False` → `2.0`.
#[test]
fn a_boolean_reduction_conditions_an_ifelse() {
    let out = emit(
        "\
v = elementof(cartpow(reals, [4]))
o1 = ifelse(lany(v .> 3.0), 1.0, 2.0)
o2 = ifelse(lall(v .> 0.0), 1.0, 2.0)
o3 = ifelse(land(lany(v .> 3.0), lall(v .> 0.0)), 1.0, 2.0)
inputs = (v)
outputs = (o1, o2, o3)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_boolean_predicate.mlir"),
        "emitted predicate module drifted from tests/goldens/order_boolean_predicate.mlir"
    );
    // Each `select` takes a SCALAR `i1` — the reduce's result, not a rank-lifted mask.
    assert_eq!(
        out.matches("stablehlo.select %").count(),
        3,
        "three `ifelse`es, three selects:\n{out}"
    );
    assert!(
        out.contains("(tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>"),
        "the condition must arrive as a scalar i1:\n{out}"
    );
}

/// The carve-out that keeps the gate honest: a broadcast of a boolean reduction is
/// NOT a predicate. `broadcast(P, …)` counts only for the ELEMENTWISE heads, because
/// that arm exists to admit a dotted comparison; `lany` consumes an array and yields
/// one scalar, so a broadcast of it lifts nothing and the map has no lowering for it
/// (`infer`'s broadcast cell table leaves it `%deferred`, no diagnostic).
#[test]
fn a_broadcast_boolean_reduction_is_not_a_predicate() {
    let err = emit_err(
        "\
b = elementof(cartpow(booleans, [4]))
y = ifelse(lany.(b), 1.0, 2.0)
inputs = (b)
outputs = (y)
",
    );
    assert!(
        err.contains("must be a boolean predicate"),
        "a broadcast `lany` must still refuse: {err}"
    );
}

/// The contrast held side by side, so neither head can drift onto the other's rule.
#[test]
fn sum_promotes_where_the_boolean_reductions_do_not() {
    let sum = emit(&vec_src("sum", "booleans", 4));
    assert!(
        sum.contains("stablehlo.convert") && sum.contains("applies stablehlo.add"),
        "`sum` over booleans still widens to count:\n{sum}"
    );
    let any = emit(&vec_src("lany", "booleans", 4));
    assert!(
        !any.contains("stablehlo.convert"),
        "`lany` does not:\n{any}"
    );
}

/// §07's domain is "boolean arrays" of any rank, so a MATRIX reduces (unlike the
/// scans, whose Domains cell reads "vectors"): one `i1` reduce per axis.
#[test]
fn a_boolean_matrix_reduces_axis_by_axis() {
    let out = emit(
        "m = elementof(cartpow(booleans, [2, 3]))\ny = lany(m)\ninputs = (m)\noutputs = (y)\n",
    );
    assert_eq!(
        out.matches("applies stablehlo.or").count(),
        2,
        "a rank-2 operand takes two `i1` reduces:\n{out}"
    );
    assert!(
        out.contains("-> tensor<i1> {"),
        "and lands on a scalar:\n{out}"
    );
}

/// A REAL array refuses. §03's `booleans ⊂ integers ⊂ reals` runs the other way, so
/// a real array is not in §07's "boolean arrays" domain, and reading truthiness off
/// one would be a convention §07 never states. (The js engine does read truthiness
/// there, and its own review flags that as a gap; the two engines answer to the
/// spec, not to each other.)
#[test]
fn a_non_boolean_operand_refuses_rather_than_reading_truthiness() {
    for head in ["lany", "lall"] {
        for set in ["reals", "integers"] {
            let msg = emit_err(&vec_src(head, set, 4));
            assert!(
                msg.contains("the domain \"boolean arrays\""),
                "`{head}` over {set} must refuse, got: {msg}"
            );
        }
    }
}

/// The DOTTED comparison mask, end to end — the input §07 "Boolean reductions"
/// actually gives these heads, and the frozen EXECUTED artifact for it. The bare
/// `gt(v, 3.0)` is not an alternative: `infer` refuses it, because §07 gives the
/// comparisons a scalar domain (`crates/infer/tests/comparison_scalar_domain.rs`).
#[test]
fn a_dotted_mask_reduction_matches_the_frozen_golden() {
    let out = emit(
        "\
v = elementof(cartpow(reals, [4]))
o1 = lany(v .> 3.0)
o2 = lall(v .> 3.0)
inputs = (v)
outputs = (o1, o2)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_mask_reduction.mlir"),
        "emitted mask reduction drifted from tests/goldens/order_mask_reduction.mlir"
    );
    assert_eq!(
        out.matches("stablehlo.compare GT").count(),
        2,
        "each reduction carries its own mask (no CSE here):\n{out}"
    );
}

// ---- §07 "Table reductions" ---------------------------------------------------

/// §07 makes `lany`/`lall` over a table a RECORD of per-column reductions, and this
/// emitter represents every value as a tensor — so the refusal is the honest
/// outcome, and it points at the column-wise spelling that does lower. Checked
/// BEFORE the argument is lowered, so the blame lands on the reduction rather than
/// on the table.
#[test]
fn a_table_boolean_reduction_refuses_and_names_the_column_form() {
    let src = "\
p = elementof(cartpow(booleans, 4))
q = elementof(cartpow(booleans, 4))
t = table(p = p, q = q)
y = lall(t)
inputs = (p, q)
outputs = (y)
";
    let msg = emit_err(src);
    assert!(
        msg.contains("a table reduction has no tensor form") && msg.contains("`lall(data.x)`"),
        "got: {msg}"
    );
}

// ---- §07 the order statistics: refused ----------------------------------------

/// `median`/`quantile` refuse, localized to the call. Both are ORDER statistics and
/// this crate has no sort — it emits no `stablehlo.sort` and no top-k. The refusal
/// says so and names the reason a sort-free rank-select is not substituted here: it
/// fabricates an element whenever the input contains NaN, and a wrong number with no
/// diagnostic is worse than refusing. The NaN guard that would make it safe is
/// unbuilt and out of this wave's scope, not unbuildable — `order.rs` says which ops
/// it would take, so a future wave does not read the route as closed.
#[test]
fn the_order_statistics_refuse_with_a_located_message() {
    for (head, src) in [
        ("median", vec_src("median", "reals", 4)),
        (
            "quantile",
            "v = elementof(cartpow(reals, [4]))\ny = quantile(v, 0.5)\ninputs = (v)\noutputs = (y)\n"
                .to_string(),
        ),
    ] {
        let msg = emit_err(&src);
        assert!(
            msg.contains(&format!("{head} has no lowering here"))
                && msg.contains("this emitter has no sort")
                && msg.contains("NaN"),
            "`{head}` must refuse and say why, got: {msg}"
        );
    }
}

/// §04's eligible-reduction list gains `median`, `lany` and `lall`, so the aggregate
/// refusal must not tell the caller they are INELIGIBLE — that would contradict the
/// spec. Two distinct messages: eligible-but-unlowered, and genuinely ineligible.
#[test]
fn an_aggregate_over_an_eligible_but_unlowered_reduction_says_so() {
    for head in ["median", "lany", "lall"] {
        let src = format!(
            "A = elementof(cartpow(reals, [2, 3]))\ny = aggregate({head}, [.i], A[.i, .j])\n\
             inputs = (A)\noutputs = (y)\n"
        );
        let msg = emit_err(&src);
        assert!(
            msg.contains(&format!("`{head}` IS an eligible reduction")),
            "`{head}` is eligible under §04, so the message must not deny it, got: {msg}"
        );
    }
}

/// The genuinely ineligible names keep the ineligibility message, and it now
/// enumerates §04's TEN eligible built-ins. `quantile` takes two inputs and is not
/// order-invariant in one; `cumsum` is a scan, which §04's "vector-to-scalar" phrase
/// excludes.
#[test]
fn an_ineligible_reduction_still_gets_the_eligibility_message() {
    for head in ["quantile", "cumsum", "cummax", "linfnorm"] {
        let src = format!(
            "A = elementof(cartpow(reals, [2, 3]))\ny = aggregate({head}, [.i], A[.i, .j])\n\
             inputs = (A)\noutputs = (y)\n"
        );
        let msg = emit_err(&src);
        assert!(
            msg.contains(&format!("`{head}` is not an eligible reduction"))
                && msg.contains("`median`, `lany` and `lall`"),
            "`{head}` is ineligible and the list must name all ten, got: {msg}"
        );
    }
}

// ---- empty inputs -------------------------------------------------------------

/// The frozen EMPTY module — the EXECUTED artifact, all five heads at length 0.
///
/// Every answer here is FIXED by the owner's zero-size-arrays ruling of 2026-08-20
/// (`flatppl-dev/empty-arrays-ruling.md`, sub-ruling 2), not chosen by this wave:
/// "`lany([])` = `false`, `lall([])` = `true`, `l1norm`/`l2norm`/`linfnorm` = `0`
/// stand (forced identities)", and the vector-valued heads "all map empty to empty",
/// uniform "across all vector-valued heads" — which `cummax`/`cummin` are. Each was
/// also matched against an oracle: `np.linalg.norm([], inf)` is `0.0`, `np.any([])` /
/// `np.all([])` are `false` / `true`, and Julia's `any(Bool[])` / `all(Bool[])` agree.
///
/// `linfnorm`'s is the one that needs a special case rather than falling out: the
/// reduce's own answer over a zero-length axis is its identity, `-inf` (executed) —
/// a NEGATIVE value for a quantity §07 declares a norm. The scans need one too,
/// because `stablehlo.reduce_window` has no `window_dimensions = 0` form. The
/// boolean pair need none: `stablehlo.reduce` over a zero-length axis returns its
/// init, which is already the right answer.
///
/// **The SOURCE form here is on borrowed time, and so is `golden_norms.rs`'s.** The
/// same ruling requires `infer` to refuse a WRITTEN literal size of 0 — "never
/// produce `Dim::Static(0)` — a derived 0 becomes `Dynamic`" — so
/// `elementof(cartpow(reals, [0]))` becomes a static error, and a length-0 vector can
/// then only reach the emitter with a `%dynamic` extent, which
/// `require_static_vector` refuses. That makes every length-0 answer above
/// unreachable from legal source once the infer half lands. Recorded in
/// `flatppl-dev/TODO-flatppl-rust.md`: the two files' empty-input tests need
/// rewriting in the same wave that closes the positivity hole, and the guard needs a
/// decision about the dynamic-extent path.
#[test]
fn the_empty_cases_match_the_frozen_golden() {
    let out = emit(
        "\
v = elementof(cartpow(reals, [0]))
b = elementof(cartpow(booleans, [0]))
o1 = linfnorm(v)
o2 = cummax(v)
o3 = lany(b)
o4 = lall(b)
o5 = cummin(v)
inputs = (v, b)
outputs = (o1, o2, o3, o4, o5)
",
    );
    assert_eq!(
        out,
        include_str!("goldens/order_empty.mlir"),
        "emitted empty-input module drifted from tests/goldens/order_empty.mlir"
    );
    // `linfnorm([])` is a constant, NOT a reduce that would answer -inf.
    assert!(
        out.contains("dense<0.0> : tensor<f32>") && !out.contains("applies stablehlo.maximum"),
        "linfnorm over an empty vector must be 0.0, not the reduce's -inf identity:\n{out}"
    );
    // Both scans are the operand itself, no `reduce_window` emitted at all — the
    // sibling head shares the special case, so it is executed at length 0 too.
    assert!(
        !out.contains("stablehlo.reduce_window"),
        "an empty scan emits no window op:\n{out}"
    );
    let ret = out
        .lines()
        .find(|l| l.trim_start().starts_with("return "))
        .expect("a return line");
    assert_eq!(
        ret.matches("%arg0").count(),
        2,
        "`cummax([])` and `cummin([])` both return the operand unchanged: {ret}"
    );
    // The boolean pair DO reduce; the zero-length axis returns the init.
    assert_eq!(
        out.matches("stablehlo.reduce(").count(),
        2,
        "lany/lall still reduce, and answer their identity:\n{out}"
    );
}
