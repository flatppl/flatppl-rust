//! The StableHLO emitter core: SSA bookkeeping, the `NodeId` → [`Value`] memo
//! map, and the typed op-helper API every later lowering task builds on.
//!
//! [`Emitter`] accumulates one textual MLIR line per emitted op into an
//! internal buffer; [`Emitter::finish`] wraps that buffer in a `module {
//! func.func @name(...) -> ret_ty { ... return ... } }` skeleton, 2-space
//! indented per nesting level (mirroring `flatppl_flatpir::writer`'s
//! canonical-text formatting style).
//!
//! Every op helper takes already-typed [`Value`]s and returns a fresh one —
//! no `Result`: these are pure text-emission primitives over values the
//! caller (Task 4's `lower_node`, and later tasks) has already type-checked
//! against the FlatPDL side-tables. Refuse-don't-mislower happens one layer
//! up, at the point an untranslatable FlatPDL node is encountered — not
//! here, where a bad shape reaching one of these helpers is an internal
//! invariant violation (hence the `panic!`s on e.g. a non-square `diag`
//! operand, rather than a `Result`).
//!
//! [`MlirTy`] (Task 2) carries shape only, no element dtype: elementwise ops
//! copy the operand's `MlirTy` unchanged (`dtype` only matters at
//! [`MlirTy::render`] time). [`Emitter::compare`]'s result is logically an
//! `i1` tensor of the same shape; since `MlirTy` has no boolean variant,
//! this module renders that one `tensor<...xi1>` text form locally
//! ([`render_i1`]) rather than extending `MlirTy` — [`Emitter::select`] does
//! the same for its predicate operand.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    CallHead, Inputs, Module, NamedKind, Node, NodeId, Ref, RefNs, Scalar, Symbol, ValueSet,
};

use crate::Dtype;
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::refuse::EmitError;

/// The dtype-exact `stablehlo.reduce` identity for `stablehlo.maximum`: real
/// negative infinity, spelled as the raw bit pattern MLIR's float-attribute
/// hex-literal syntax expects (`0xFF800000` / `0xFFF0000000000000`). A finite
/// stand-in like `-1e30` is silently wrong for any input at or below it
/// (e.g. `log(0)`), since it would then compare as the (wrong) max.
fn reduce_max_identity(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "0xFF800000",
        Dtype::F64 => "0xFFF0000000000000",
    }
}

/// The dtype-exact StableHLO literal for **positive** infinity — the mirror
/// of [`reduce_max_identity`] (same magnitude bit pattern, sign bit
/// cleared). See [`Emitter::inf`] for why the decimal-literal path
/// (`render_float_literal`) can't be used instead.
fn pos_inf_literal(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "0x7F800000",
        Dtype::F64 => "0x7FF0000000000000",
    }
}

/// The canonical spec §03 embedding order `booleans ⊂ integers ⊂ reals`, as
/// a rank: [`Emitter::binary`]'s mismatched-operand-kind widening converges
/// on whichever of its two operands' kinds has the HIGHER rank (e.g. a
/// `bool`-vs-`int` mismatch widens to `int`, an `int`-vs-`real` mismatch to
/// `real`), never the other way — the embedding only ever goes "up" the
/// inclusion chain.
fn elem_rank(k: ElemKind) -> u8 {
    match k {
        ElemKind::Bool => 0,
        ElemKind::Int => 1,
        ElemKind::Real => 2,
    }
}

/// Emits textual StableHLO into an internal buffer while assigning fresh SSA
/// names and tracking which FlatPDL [`NodeId`]s have already been lowered.
pub struct Emitter<'m> {
    /// The FlatPDL module being lowered. Read by [`Emitter::lower_node`]'s
    /// dispatch (node structure) and by [`Emitter::node`]/[`Emitter::resolve`]
    /// (narrow accessors `crate::ops::lower_builtin` uses to inspect a call's
    /// structure from outside this module).
    m: &'m Module,
    dtype: Dtype,
    next: u32,
    /// Memoizes `NodeId -> Value` so a shared sub-expression is lowered (and
    /// its op line emitted) once — see [`Emitter::lower_node`]. Also the seed
    /// point for a caller-bound leaf (a function/kernel argument's `NodeId`
    /// pre-bound to its `%argN` `Value` via [`Emitter::bind`]) before the body
    /// graph that references it is walked.
    memo: HashMap<NodeId, Value>,
    body: String,
    /// The threaded rng-state key (spec §07 rng ABI). Set by
    /// [`crate::registry::lower_sample`] from a `builtin_sample`'s rng arg
    /// before the distribution builder draws; each [`Emitter::rng`] call
    /// advances it via `stablehlo.rng_bit_generator`. `None` until the first
    /// sample seeds it — a draw with no key set is an internal invariant
    /// violation (see [`Emitter::cur_key`]).
    cur_key: Option<Value>,
    /// The advanced key each `builtin_sample` node produced, keyed by that
    /// node's [`NodeId`] — the tensor-side realization of spec §07's
    /// `(value, new_rngstate)` pair's second slot. Read by
    /// [`Emitter::lower_node`]'s `get0(sample, 1)`/`get(sample, 2)` arm so a
    /// chained `rand` threads the advanced key onward without re-drawing.
    sample_keys: HashMap<NodeId, Value>,
    /// The fan-out batch shape, set by [`crate::registry::lower_sample`] around
    /// a batched `builtin_sample(rng, ctor, input, n)` (spec §07 size dims).
    /// When `Some`, [`Emitter::rng`] OVERRIDES the per-element `out_ty` the
    /// distribution builder passes and draws one `[n]`-shaped batch with a
    /// single `rng_bit_generator` advance; the builder's scalar params/constants
    /// then broadcast over that batch via [`Emitter::binary`]'s auto-broadcast.
    /// `None` (the scalar case) leaves every draw sized exactly as before.
    batch_shape: Option<Vec<u64>>,
}

impl<'m> Emitter<'m> {
    pub fn new(m: &'m Module, dtype: Dtype) -> Self {
        Emitter {
            m,
            dtype,
            next: 0,
            memo: HashMap::new(),
            body: String::new(),
            cur_key: None,
            sample_keys: HashMap::new(),
            batch_shape: None,
        }
    }

    // ---- rng-key threading (spec §07 rng ABI) -------------------------------

    /// Seed the threaded rng key — [`crate::registry::lower_sample`] calls this
    /// with a `builtin_sample`'s (already-lowered) rng argument before running
    /// the distribution builder, so every [`Emitter::rng`] draw the builder
    /// makes advances from this key.
    pub(crate) fn set_cur_key(&mut self, k: Value) {
        self.cur_key = Some(k);
    }

    /// The current threaded rng key. Panics if no key has been set — a draw
    /// reaching [`Emitter::rng`] outside a `builtin_sample` (which is the only
    /// thing that seeds a key) is an internal invariant violation, mirroring
    /// this module's other panic-on-bad-state discipline.
    pub(crate) fn cur_key(&self) -> Value {
        self.cur_key
            .clone()
            .expect("rng draw with no threaded key (builtin_sample must set_cur_key first)")
    }

    /// Record the advanced key `k` a `builtin_sample` node `id` produced, for
    /// the `get0(sample, 1)`/`get(sample, 2)` projection to read back.
    pub(crate) fn record_sample_key(&mut self, id: NodeId, k: Value) {
        self.sample_keys.insert(id, k);
    }

    /// The advanced key recorded for `builtin_sample` node `id`, or `None` if
    /// that node has not been lowered yet.
    pub(crate) fn sample_key(&self, id: NodeId) -> Option<Value> {
        self.sample_keys.get(&id).cloned()
    }

    /// Set the fan-out batch shape [`Emitter::rng`] draws at — called by
    /// [`crate::registry::lower_sample`] with `[n]` around a batched iid
    /// `builtin_sample`, then [`Emitter::clear_batch_shape`]ed (even on error)
    /// so a later scalar sample in the same module is unaffected.
    pub(crate) fn set_batch_shape(&mut self, dims: Vec<u64>) {
        self.batch_shape = Some(dims);
    }

    /// Clear the fan-out batch shape — see [`Emitter::set_batch_shape`].
    pub(crate) fn clear_batch_shape(&mut self) {
        self.batch_shape = None;
    }

    /// The current fan-out batch shape, if a batched `builtin_sample` set one.
    /// `crate::registry`'s rejection samplers read this to switch a scalar
    /// [`draw_gamma`]-style `while` to its batched `[n]` form (Tier 2 fan-out):
    /// they must ALSO size their pre-drawn candidate batches at `[MAXITER, n]`,
    /// which needs the concrete `n` here, not just the `Emitter::rng` override.
    pub(crate) fn batch_shape(&self) -> Option<Vec<u64>> {
        self.batch_shape.clone()
    }

    /// Allocate a fresh SSA name (`%0`, `%1`, ...).
    fn fresh(&mut self) -> String {
        let name = format!("%{}", self.next);
        self.next += 1;
        name
    }

    /// Append an already-formatted op (one line, or a region's several
    /// lines) to the function body.
    fn push(&mut self, text: &str) {
        for line in text.lines() {
            self.body.push_str(line);
            self.body.push('\n');
        }
    }

    // ---- elementary ops -------------------------------------------------

    /// `%N = stablehlo.constant dense<x> : ty` — a (possibly splat, for a
    /// non-scalar `ty`) constant.
    pub fn constant(&mut self, x: f64, ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let ty_text = ty.render(self.dtype, ElemKind::Real);
        let lit = render_float_literal(x);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{lit}> : {ty_text}"
        ));
        Value {
            ssa,
            ty,
            elem: ElemKind::Real,
        }
    }

    /// A scalar-literal convenience: `constant(x, MlirTy::Scalar)`.
    pub fn scalar(&mut self, x: f64) -> Value {
        self.constant(x, MlirTy::Scalar)
    }

    /// `%N = stablehlo.constant dense<+inf> : ty` — positive infinity (the
    /// `ifelse`/`neg(inf)` "outside the support" log-density floor). Cannot
    /// go through [`Emitter::constant`]: that renders `x` as a *decimal*
    /// literal (`render_float_literal`), and `f64::INFINITY` prints as `inf`,
    /// which — like the bare `-inf` a decimal `f64::NEG_INFINITY` would
    /// produce — is not a valid MLIR float-attribute token (verified against
    /// the real StableHLO parser, jax 0.10.2); only the dtype-exact hex bit
    /// pattern parses. Same reasoning as [`reduce_max_identity`]'s negative
    /// infinity, sign bit cleared.
    pub fn inf(&mut self, ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let ty_text = ty.render(self.dtype, ElemKind::Real);
        let lit = pos_inf_literal(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{lit}> : {ty_text}"
        ));
        Value {
            ssa,
            ty,
            elem: ElemKind::Real,
        }
    }

    /// One elementwise unary op: `%N = {op} %a : ty`. Result type copies the
    /// operand's `MlirTy` — elementwise ops are shape-preserving. The result
    /// `elem` copies `a`'s (kind-polymorphic `neg`/`abs` pass their operand's
    /// own `Int`/`Real` through unchanged; every real-only caller
    /// (`log`/`exp`/`sqrt`/`cos`/`invlogit`/…) only ever reaches this with an
    /// already-`Real` `a` — the caller converted it first — so the pass-
    /// through is equally correct there).
    pub fn unary(&mut self, op: &str, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!("{ssa} = {op} {} : {ty_text}", a.ssa));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: a.elem,
        }
    }

    /// Emit one elementwise binary op at `a`'s shape, with NO broadcasting —
    /// the raw text primitive [`Emitter::binary`] wraps. Both operands are
    /// assumed to already share `a`'s shape (the caller has broadcast a scalar
    /// operand up first, if needed) AND `a`'s `elem` (the caller has
    /// reconciled a kind mismatch first, if needed) — the result `elem`
    /// copies `a`'s.
    fn emit_binary(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!("{ssa} = {op} {}, {} : {ty_text}", a.ssa, b.ssa));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: a.elem,
        }
    }

    /// One elementwise binary op: `%N = {op} %a, %b : ty`. Two operand
    /// mismatches are reconciled before emitting, so every direct caller
    /// (both `crate::ops`'s dispatch table AND `crate::registry`'s
    /// distribution builders, which call `add`/`sub`/`mul`/`neg`/`abs`
    /// directly — those never carry a `NodeId` to coerce against an inferred
    /// result kind) gets well-typed StableHLO with no extra ceremony:
    ///
    /// - **Elem-kind** (spec §03 `booleans ⊂ integers ⊂ reals`): a mismatched
    ///   pair widens the narrower operand up to the wider (via
    ///   [`Emitter::convert`], [`elem_rank`]'s ordering) BEFORE anything else —
    ///   e.g. an `Int` `k` mixed with a `Real` parameter (a discrete
    ///   distribution's logpdf, `k * log(rate)`) converts `k` to `Real`
    ///   first. A matching pair (both `Int`, e.g. Binomial's `n - k`) is left
    ///   alone, so an all-integer expression stays integer end to end.
    /// - **Shape** ([`Emitter::broadcast_pair`], spec §04 "Broadcasting"):
    ///   when one operand is a `Scalar` and the other `Ranked`, the scalar is
    ///   [`Emitter::broadcast_scalar`]d up to the ranked shape (StableHLO's
    ///   elementwise ops require identical operand shapes) — the mechanism a
    ///   fan-out Tier-1 iid draw relies on to mix a batched `[n]` draw with
    ///   the distribution's scalar parameters/constants. When BOTH operands
    ///   are `Ranked` but their shapes differ, each size-1 axis expands to
    ///   the other side's size via [`Emitter::broadcast_in_dim`] — the
    ///   mechanism an `iid(Dist, n)` density's length-1 array-of-records
    ///   parameters need to combine with the length-`n` observation vector.
    ///
    /// When both already match (every `@logdensity` path and every scalar
    /// `@sample` this emitter built before Task A2 — inference has kind- and
    /// shape-unified their operands upstream, plus every same-length batched
    /// pair), neither reconciliation emits anything and the output is
    /// byte-identical to before. A genuinely incompatible Ranked-vs-Ranked
    /// pair (different rank, or an axis neither equal nor size-1) panics — an
    /// internal invariant violation upstream type-checking should have ruled
    /// out, per this module's doc comment.
    pub fn binary(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        let target = if elem_rank(a.elem) >= elem_rank(b.elem) {
            a.elem
        } else {
            b.elem
        };
        let a = self.convert(a, target);
        let b = self.convert(b, target);
        let (a, b) = self.broadcast_pair(&a, &b);
        self.emit_binary(op, &a, &b)
    }

    pub fn add(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.add", a, b)
    }
    pub fn sub(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.subtract", a, b)
    }
    pub fn mul(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.multiply", a, b)
    }
    /// A binary op that ONLY has real semantics (`divide`/`power`): both
    /// operands are [`Emitter::convert`]ed to [`ElemKind::Real`] first,
    /// unconditionally — unlike [`Emitter::binary`]'s kind-polymorphic
    /// widening (which would leave a matching `Int`/`Int` pair alone),
    /// `divide(3, 4)` must still be the real division `0.75`, never an
    /// integer floor division (that's the separate, unimplemented `div`
    /// head). Both callers (`crate::ops`'s dispatch table AND
    /// `crate::registry`'s distribution builders, e.g. a literal-parameter
    /// `Gamma(2, 1)`'s `rate^shape`) get this for free with no extra
    /// ceremony at the call site.
    fn binary_real(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        let a = self.convert(a, ElemKind::Real);
        let b = self.convert(b, ElemKind::Real);
        self.binary(op, &a, &b)
    }

    /// A unary op that ONLY has real semantics (`log`/`exp`/`sqrt`/`cos`/
    /// `invlogit`/`sin`/`floor` — none of these are meaningful StableHLO
    /// integer ops): the operand is [`Emitter::convert`]ed to
    /// [`ElemKind::Real`] first, unconditionally. Unlike [`Emitter::unary`]
    /// (kind-polymorphic `neg`/`abs`, which pass an `Int` operand through),
    /// this fixes e.g. `crate::registry::gamma_logpdf`'s `log(&rate)`/
    /// `lgamma(&shape)` for a literal-parameter `Gamma(2, 1)` — `shape`/
    /// `rate` are `Int` value constants there, with no `NodeId` for the
    /// caller to coerce against an inferred result kind (registry.rs calls
    /// these directly, never through `crate::ops`'s dispatch table).
    fn unary_real(&mut self, op: &str, a: &Value) -> Value {
        let a = self.convert(a, ElemKind::Real);
        self.unary(op, &a)
    }

    pub fn div(&mut self, a: &Value, b: &Value) -> Value {
        self.binary_real("stablehlo.divide", a, b)
    }
    pub fn pow(&mut self, a: &Value, b: &Value) -> Value {
        self.binary_real("stablehlo.power", a, b)
    }

    /// `%N = stablehlo.divide %a, %b : ty` — the raw, KIND-POLYMORPHIC divide
    /// [`Emitter::binary`] wraps, unlike [`Emitter::div`]'s `binary_real`
    /// (which unconditionally forces both operands to `Real`). [`Emitter::
    /// floor_div`]'s correction algorithm needs StableHLO's native INTEGER
    /// `divide` — truncating toward zero — so it must stay off `div`'s
    /// real-forcing path entirely; every caller here already has both
    /// operands `Int` (spec §07 `div`'s domain, from inference), so
    /// `binary`'s kind-polymorphic widening is a no-op and this simply emits
    /// the same op text at `Int`.
    fn trunc_div(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.divide", a, b)
    }

    /// `%N = stablehlo.remainder %a, %b : ty` — StableHLO's native truncated
    /// remainder (sign of the dividend `a`), kind-polymorphic via
    /// [`Emitter::binary`] like [`Emitter::trunc_div`]. Used by [`Emitter::
    /// floor_mod`]'s correction algorithm; unlike `div`, `mod` has no
    /// real-only counterpart to stay off — nothing else in this emitter
    /// needs a real remainder.
    fn rem(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.remainder", a, b)
    }

    /// `div(a, b) = ⌊a/b⌋` (spec §07 integer floor division, `Int` operands,
    /// `b ≠ 0`). StableHLO's integer `divide` ([`Emitter::trunc_div`])
    /// truncates TOWARD ZERO, not down, so the two disagree exactly when the
    /// truncated remainder `r = a - q_t*b` is nonzero AND its sign differs
    /// from the divisor's — the one case corrected here by stepping the
    /// truncated quotient down by one. `signs_differ` is a boolean XOR,
    /// computed as `r_neg != b_neg` via [`Emitter::compare`]'s `"NE"`
    /// direction (valid for `i1` operands, unlike an ordering compare).
    pub fn floor_div(&mut self, a: &Value, b: &Value) -> Value {
        let q_t = self.trunc_div(a, b);
        let prod = self.mul(&q_t, b);
        let r = self.sub(a, &prod);
        let zero = self.int_value_const(0);
        let r_nz = self.compare("NE", &r, &zero);
        let r_neg = self.compare("LT", &r, &zero);
        let b_neg = self.compare("LT", b, &zero);
        let signs_differ = self.compare("NE", &r_neg, &b_neg);
        let need_fix = self.and(&r_nz, &signs_differ);
        let one = self.int_value_const(1);
        let q_minus1 = self.sub(&q_t, &one);
        self.select(&need_fix, &q_minus1, &q_t)
    }

    /// `mod(a, b) = a − b·⌊a/b⌋` (spec §07 floored modulo, `Int` operands,
    /// `b ≠ 0`; the result takes the DIVISOR's sign — Python `%`, not C `%`).
    /// Same sign-correction shape as [`Emitter::floor_div`], applied to
    /// StableHLO's truncated `remainder` ([`Emitter::rem`], sign of the
    /// dividend `a`) instead: nonzero and sign-disagreeing with `b` means the
    /// floored remainder is `r_t + b`.
    pub fn floor_mod(&mut self, a: &Value, b: &Value) -> Value {
        let r_t = self.rem(a, b);
        let zero = self.int_value_const(0);
        let r_nz = self.compare("NE", &r_t, &zero);
        let r_neg = self.compare("LT", &r_t, &zero);
        let b_neg = self.compare("LT", b, &zero);
        let signs_differ = self.compare("NE", &r_neg, &b_neg);
        let need_fix = self.and(&r_nz, &signs_differ);
        let r_plus_b = self.add(&r_t, b);
        self.select(&need_fix, &r_plus_b, &r_t)
    }

    pub fn neg(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.negate", a)
    }
    pub fn log(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.log", a)
    }
    pub fn exp(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.exponential", a)
    }
    pub fn sqrt(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.sqrt", a)
    }
    pub fn abs(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.abs", a)
    }
    pub fn cos(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.cosine", a)
    }
    /// `invlogit(x) = 1/(1+exp(-x))` (the logistic sigmoid, §07) — emitted as the
    /// native `stablehlo.logistic`, which is numerically stable (no `exp`
    /// overflow for large-magnitude `x`) and IREE-supported, rather than the
    /// naive composition. Rank-preserving, so it batches under `broadcast`
    /// (`invlogit.(linear_predictor)`) via the shared unary path.
    pub fn invlogit(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.logistic", a)
    }
    /// `stablehlo.sine` — a NEW op form for this crate (Task 14's Cauchy
    /// `@sample`, which needs `tan(t) = sin(t) / cos(t)`; no `chlo`/
    /// `stablehlo` `tan` op is used, mirroring [`Emitter::cos`]'s existing
    /// `stablehlo.cosine`). Parser-validated against the real StableHLO
    /// parser (jax 0.10.2, `jax._src.interpreters.mlir.make_ir_context`),
    /// same discipline as every other op text this module emits.
    pub fn sin(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.sine", a)
    }
    /// `stablehlo.floor` — a NEW op form for this crate (Task 16's Geometric
    /// `@sample`, `floor(log(U) / log(1 - p))`, the only discrete sampler that
    /// rounds a real-valued inverse-CDF down to an integer count). Elementwise,
    /// shape-preserving, same plain `: ty` form as every other `stablehlo.*`
    /// unary; parser-validated against the real StableHLO parser (jax 0.10.2),
    /// same discipline as [`Emitter::sin`].
    pub fn floor(&mut self, a: &Value) -> Value {
        self.unary_real("stablehlo.floor", a)
    }

    /// `%N = stablehlo.compare {dir}, %a, %b : (lhs, rhs) -> i1-shape`.
    /// `dir` is a StableHLO `comparison_direction` (`"LT"`, `"GE"`, `"EQ"`,
    /// ...). The result is logically an `i1` tensor of the operands' shape —
    /// see the module doc comment for why that is rendered via [`render_i1`]
    /// rather than through `MlirTy`/`Dtype`; the returned `Value`'s `ty` still
    /// carries that shape so a later [`Emitter::select`] can reuse it.
    ///
    /// Shape reconciliation is [`Emitter::broadcast_pair`], the same
    /// mechanism [`Emitter::binary`] uses (StableHLO's `compare` requires
    /// identical operand shapes, no implicit broadcast): a `Scalar`-vs-
    /// `Ranked` pair splats the scalar up first — the mechanism a batched
    /// (Tier-2 fan-out) rejection sampler leans on to test a `[n]` candidate
    /// against a scalar bound — and a `Ranked`-vs-`Ranked` pair with a size-1
    /// axis expands it to the other side's size. When the shapes already
    /// match (every scalar `@sample` / `@logdensity` path, inference-unified
    /// upstream, plus every same-length batched pair), no broadcast is
    /// emitted and the output is byte-identical to before.
    ///
    /// A mismatched-elem-kind operand pair is ALSO reconciled first, same
    /// widening rule as [`Emitter::binary`] ([`elem_rank`]'s order, via
    /// [`Emitter::convert`]) — e.g. `ops::lower_in`'s `compare(int_product,
    /// real_zero)` (an all-integer `in(k, interval(0, 10))`) widens the
    /// product to `Real` before comparing. A matching pair (both operands
    /// already the same kind — every existing caller, all-`Real`) converts
    /// nothing, so the output stays byte-identical to before.
    pub fn compare(&mut self, dir: &str, a: &Value, b: &Value) -> Value {
        let target = if elem_rank(a.elem) >= elem_rank(b.elem) {
            a.elem
        } else {
            b.elem
        };
        let a = self.convert(a, target);
        let b = self.convert(b, target);
        let (a, b) = self.broadcast_pair(&a, &b);
        let ssa = self.fresh();
        let lhs_ty = a.ty.render(self.dtype, a.elem);
        let rhs_ty = b.ty.render(self.dtype, b.elem);
        let result_ty = render_i1(&a.ty);
        // StableHLO requires an explicit `compare_type` only to disambiguate
        // integer signedness (a `Bool` operand must NOT carry one; a `Real`
        // operand pair is left to its FLOAT default) — this emitter's `Int`
        // values are always signed (`i32`/`i64`), so a reconciled `Int` pair
        // appends `SIGNED` (matching `Emitter::int_compare`'s raw form); a
        // `Real`/`Bool` pair emits exactly as before, byte-identical.
        let compare_type = if a.elem == ElemKind::Int {
            ", SIGNED"
        } else {
            ""
        };
        self.push(&format!(
            "{ssa} = stablehlo.compare {dir}, {}, {}{compare_type} : ({lhs_ty}, {rhs_ty}) -> {result_ty}",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: a.ty,
            elem: ElemKind::Bool,
        }
    }

    /// `%N = stablehlo.select %pred, %a, %b : (i1-shape, ty, ty) -> ty`.
    /// `c` is treated as an `i1` tensor of its own `MlirTy` shape (typically
    /// an [`Emitter::compare`] result) regardless of what element type its
    /// `MlirTy` would otherwise render as — see the module doc comment.
    ///
    /// A mixed `Scalar`/`Ranked` operand set auto-broadcasts every scalar VALUE
    /// operand up to the ranked shape (StableHLO's `select` requires
    /// `on_true`/`on_false` to share the result shape) — the mechanism a
    /// batched (Tier-2 fan-out) rejection sampler uses to fold a `[n]`
    /// candidate against a scalar fallback, or pick a per-lane sign. `a`/`b`
    /// are first reconciled to each other via [`Emitter::broadcast_pair`]
    /// (so a `Ranked`-vs-`Ranked` size-1 mismatch between the two VALUE
    /// branches expands, same as [`Emitter::binary`]); a second pass then
    /// picks up the PREDICATE's shape too, for the case broadcast_pair alone
    /// can't see — `a`/`b` both `Scalar` but `c` `Ranked` (`floor_div`/
    /// `floor_mod`'s `need_fix` compare result against scalar-arithmetic
    /// branches). StableHLO accepts a rank-0 `pred` with ranked operands
    /// (parse-validated), so a scalar predicate itself never needs
    /// broadcasting. When all three already share a shape (every scalar
    /// path, inference-unified upstream), no broadcast is emitted and the
    /// output is byte-identical to before.
    ///
    /// `a`/`b` are ALSO reconciled to one elem kind first, same widening rule
    /// as [`Emitter::binary`]/[`Emitter::compare`] ([`elem_rank`]'s order, via
    /// [`Emitter::convert`]) — an `ifelse` over two `Int` branches must return
    /// an `Int`-tagged value whose tag matches the emitted `i32` SSA, not a
    /// hardcoded `Real`. A matching pair (every existing caller, all-`Real`)
    /// converts nothing, so the output stays byte-identical to before.
    pub fn select(&mut self, c: &Value, a: &Value, b: &Value) -> Value {
        let elem_target = if elem_rank(a.elem) >= elem_rank(b.elem) {
            a.elem
        } else {
            b.elem
        };
        let a = self.convert(a, elem_target);
        let b = self.convert(b, elem_target);
        let (a, b) = self.broadcast_pair(&a, &b);
        // Target the ranked shape among {pred, on_true, on_false}, if any —
        // `a`/`b` already share a shape (just above); this second pass only
        // does anything when that shared shape is `Scalar` but `c` is
        // `Ranked` (`broadcast_scalar`'s no-op guard makes it a pure no-op
        // otherwise, since `a`/`b` already equal any ranked shape it'd pick).
        let shape_target = [&c.ty, &a.ty, &b.ty]
            .into_iter()
            .find(|t| matches!(t, MlirTy::Ranked(_)))
            .cloned();
        let (a, b) = match &shape_target {
            Some(shape) => (
                self.broadcast_scalar(&a, shape),
                self.broadcast_scalar(&b, shape),
            ),
            None => (a, b),
        };
        let ssa = self.fresh();
        let pred_ty = render_i1(&c.ty);
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = stablehlo.select {}, {}, {} : ({pred_ty}, {ty_text}, {ty_text}) -> {ty_text}",
            c.ssa, a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: a.ty,
            elem: elem_target,
        }
    }

    /// `%N = stablehlo.convert %a : (from) -> to` — a canonical scalar-kind
    /// embedding (spec §03 `booleans ⊂ integers ⊂ reals`), e.g. widening an
    /// `i32` operand up to `f32` at a real-only op's boundary. Numerically
    /// exact for every embedding this emitter ever performs (an integer or
    /// boolean value has an exact real representation). A no-op — returns
    /// `v` unchanged, emits no line — when `v` is already at `target`, so
    /// callers can convert unconditionally without checking first.
    pub fn convert(&mut self, v: &Value, target: ElemKind) -> Value {
        if v.elem == target {
            return v.clone();
        }
        let ssa = self.fresh();
        let from = v.ty.render(self.dtype, v.elem);
        let to = v.ty.render(self.dtype, target);
        self.push(&format!(
            "{ssa} = stablehlo.convert {} : ({from}) -> {to}",
            v.ssa
        ));
        Value {
            ssa,
            ty: v.ty.clone(),
            elem: target,
        }
    }

    // ---- shape ops (Task 4: `get`/`get0`, `logsumexp`/`in` broadcasting) ---

    /// `%N = stablehlo.slice %a [s0:l0, s1:l1:t1, ...] : (operand_ty) ->
    /// result_ty` — a static per-axis slice (`starts`/`limits`/`strides`,
    /// one triple per `a`'s rank; StableHLO's pretty form omits `:stride`
    /// when it's `1`). Each result dimension is `(limit - start).div_ceil(stride)`.
    /// Shape-only — StableHLO requires a slice's result element type to
    /// match its operand's exactly, so the result `elem` copies `a`'s (an
    /// `Int`-array `get`/`get0` slices out an `Int` scalar, not a `Real` one).
    pub fn slice(&mut self, a: &Value, starts: &[u64], limits: &[u64], strides: &[u64]) -> Value {
        let dims = match &a.ty {
            MlirTy::Ranked(dims) => dims,
            other => panic!("slice expects a ranked operand, got {other:?}"),
        };
        assert_eq!(dims.len(), starts.len(), "slice: starts rank mismatch");
        assert_eq!(dims.len(), limits.len(), "slice: limits rank mismatch");
        assert_eq!(dims.len(), strides.len(), "slice: strides rank mismatch");

        let ranges: Vec<String> = starts
            .iter()
            .zip(limits)
            .zip(strides)
            .map(|((s, l), t)| {
                if *t == 1 {
                    format!("{s}:{l}")
                } else {
                    format!("{s}:{l}:{t}")
                }
            })
            .collect();
        let result_dims: Vec<Option<u64>> = starts
            .iter()
            .zip(limits)
            .zip(strides)
            .map(|((s, l), t)| Some((l - s).div_ceil(*t)))
            .collect();
        let result_ty = MlirTy::Ranked(result_dims);

        let ssa = self.fresh();
        let operand_ty = a.ty.render(self.dtype, a.elem);
        let result_ty_text = result_ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = stablehlo.slice {} [{}] : ({operand_ty}) -> {result_ty_text}",
            a.ssa,
            ranges.join(", ")
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: a.elem,
        }
    }

    /// `%N = stablehlo.reshape %a : (operand_ty) -> result_ty` — reinterprets
    /// `a`'s elements (same element count) under a different static shape,
    /// e.g. dropping `get0`/`get`'s now-length-1 sliced axis down to a
    /// `Scalar`. Shape-only — same element-type-preserving contract as
    /// [`Emitter::slice`], so the result `elem` copies `a`'s.
    pub fn reshape(&mut self, a: &Value, ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let operand_ty = a.ty.render(self.dtype, a.elem);
        let result_ty_text = ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = stablehlo.reshape {} : ({operand_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty,
            elem: a.elem,
        }
    }

    /// `%N = stablehlo.broadcast_in_dim %a, dims = [...] : (operand_ty) ->
    /// ty` — broadcasts `a` up to the (larger) shape `ty`, mapping `a`'s
    /// existing dimensions onto the `dims` positions of the result, in
    /// order. A rank-0 (`Scalar`) operand takes `dims = []`, StableHLO's
    /// documented scalar-broadcast form — the only shape this emitter's
    /// callers need today (`logsumexp`'s reduced max, `in`'s interval bounds,
    /// broadcast back up to the input vector/variate's shape; StableHLO's
    /// elementwise ops require identical operand shapes, no implicit
    /// broadcast). Shape-only — same element-type-preserving contract as
    /// [`Emitter::slice`], so the result `elem` copies `a`'s.
    pub fn broadcast_in_dim(&mut self, a: &Value, dims: &[u64], ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let operand_ty = a.ty.render(self.dtype, a.elem);
        let result_ty_text = ty.render(self.dtype, a.elem);
        let dims_text = dims
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        self.push(&format!(
            "{ssa} = stablehlo.broadcast_in_dim {}, dims = [{dims_text}] : ({operand_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty,
            elem: a.elem,
        }
    }

    /// `get(operand, idx)` / `get0(...)` (spec §07) with a RUNTIME rank-1
    /// `Int` selector `idx` (the `theta[person]`-style vector-index case —
    /// `crate::ops::lower_get`'s fallback once its compile-time
    /// `literal_index` fast path fails) — lowers `operand[idx]` (`operand`
    /// rank-1, length `K`) at every position of `idx` (rank-1, length `N`,
    /// `base`-based) to a rank-1 result of length `N`, via
    /// `stablehlo.gather`. `base` (1 for `get`, 0 for `get0`) is subtracted
    /// from `idx` first ([`Emitter::sub`], kind-polymorphic — stays `Int`,
    /// auto-broadcasting the scalar `base` over `idx`'s shape) to land on
    /// StableHLO's 0-based convention; FlatPPL indices are valid
    /// `posintegers`, so the result is always in range after subtraction
    /// (`stablehlo.gather` also clamps internally, but no explicit clamp is
    /// needed here). The 1-D index vector is then reshaped `[N] -> [N, 1]`
    /// (`index_vector_dim = 1`) before the generic-form `stablehlo.gather`
    /// (no pretty form — same reasoning as [`Emitter::tri_solve`]): one
    /// scalar slice per index (`slice_sizes = [1]`, `collapsed_slice_dims =
    /// [0]`), gathered along `operand`'s only axis (`start_index_map =
    /// [0]`). Dimension numbers are pinned VERBATIM against JAX/XLA's own
    /// emission for `operand[idx]` — do not deviate. Result `elem` copies
    /// `operand`'s (a gather of reals stays real, of ints stays int); both
    /// `operand` and `idx` must already be rank-1 — `crate::ops::lower_get`'s
    /// job to check before calling this.
    pub fn gather(&mut self, operand: &Value, idx: &Value, base: i64) -> Value {
        assert!(
            matches!(&operand.ty, MlirTy::Ranked(dims) if dims.len() == 1),
            "gather expects a rank-1 operand, got {:?}",
            operand.ty
        );
        let n = match &idx.ty {
            MlirTy::Ranked(dims) if dims.len() == 1 => dims[0],
            other => panic!("gather expects a rank-1 index, got {other:?}"),
        };
        assert_eq!(
            idx.elem,
            ElemKind::Int,
            "gather: index must be an Int tensor"
        );

        let base_const = self.int_value_const(base);
        let idx0 = self.sub(idx, &base_const);
        let idx2d = self.reshape(&idx0, MlirTy::Ranked(vec![n, Some(1)]));

        let result_ty = MlirTy::Ranked(vec![n]);
        let ssa = self.fresh();
        let operand_ty = operand.ty.render(self.dtype, operand.elem);
        let idx_ty = idx2d.ty.render(self.dtype, idx2d.elem);
        let result_ty_text = result_ty.render(self.dtype, operand.elem);
        self.push(&format!(
            "{ssa} = \"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}}> : ({operand_ty}, {idx_ty}) -> {result_ty_text}",
            operand.ssa, idx2d.ssa
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: operand.elem,
        }
    }

    /// `%N = stablehlo.concatenate %a, %b, ..., dim = 0 : (op1_ty, op2_ty,
    /// ...) -> result_ty` — packs `elems` into a tensor one rank higher than
    /// each element, of length `elems.len()` along the new leading dim:
    /// every element is first `reshape`d to add a length-1 leading axis
    /// (`tensor<1x...>`, `...` being the element's own shape), then
    /// concatenated along dim 0. Rank-generic because spec §03 arrays may
    /// nest (a `vector(...)` of scalars is the common case — used by
    /// `logsumexp(vector(t1, …, tk))`, superpose/discrete-marginal — but a
    /// `vector(...)` of same-shape ARRAY elements, a legal vector-of-vectors
    /// distinct from a matrix, is equally valid and must lower to a rank-2
    /// tensor, not silently truncate to rank-1 by assuming a scalar
    /// element). Every `elems[i].ty` must be identical — checked by the
    /// caller (`ops::lower_vector`, which has the `NodeId` to blame and
    /// returns a precise refusal for a ragged vector-of-vectors); a shape
    /// mismatch reaching this point is an internal invariant violation, per
    /// this module's doc comment. Parser-validated against the real
    /// StableHLO parser (jax 0.10.2) for both the scalar-element rank-1 case
    /// (`stablehlo.concatenate %a, %b, dim = 0 : (tensor<1xf32>,
    /// tensor<1xf32>) -> tensor<2xf32>`) and the vector-element rank-2 case
    /// (`stablehlo.concatenate %a, %b, dim = 0 : (tensor<1x3xf32>,
    /// tensor<1x3xf32>) -> tensor<2x3xf32>`).
    pub fn vector(&mut self, elems: &[Value]) -> Value {
        assert!(!elems.is_empty(), "vector: expected at least one element");
        let elem_ty = elems[0].ty.clone();
        let elem_kind = elems[0].elem;
        assert!(
            elems.iter().all(|v| v.ty == elem_ty),
            "vector: elements must have identical shape (ragged vector-of-vectors \
             must be refused by the caller before this is reached)"
        );
        assert!(
            elems.iter().all(|v| v.elem == elem_kind),
            "vector: elements must share one elem kind (the caller reconciles a kind \
             mismatch — e.g. via node_kind — before this is reached)"
        );
        let inner_dims: Vec<Option<u64>> = match &elem_ty {
            MlirTy::Scalar => Vec::new(),
            MlirTy::Ranked(dims) => dims.clone(),
            MlirTy::Tuple(_) => panic!("vector: tuple elements have no tensor form"),
            MlirTy::Key => panic!("vector: an rng key has no tensor form to stack"),
        };
        let stacked_elem_ty = {
            let mut dims = Vec::with_capacity(inner_dims.len() + 1);
            dims.push(Some(1));
            dims.extend(inner_dims.iter().copied());
            MlirTy::Ranked(dims)
        };
        let reshaped: Vec<Value> = elems
            .iter()
            .map(|v| self.reshape(v, stacked_elem_ty.clone()))
            .collect();

        let mut result_dims = Vec::with_capacity(inner_dims.len() + 1);
        result_dims.push(Some(reshaped.len() as u64));
        result_dims.extend(inner_dims.iter().copied());
        let result_ty = MlirTy::Ranked(result_dims);

        let operand_ssas = reshaped
            .iter()
            .map(|v| v.ssa.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let operand_tys = reshaped
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>()
            .join(", ");
        let result_ty_text = result_ty.render(self.dtype, elem_kind);

        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.concatenate {operand_ssas}, dim = 0 : ({operand_tys}) -> {result_ty_text}"
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: elem_kind,
        }
    }

    /// `%N = stablehlo.transpose %a, dims = [perm...] : (operand_ty) ->
    /// result_ty` — permutes `a`'s axes so result axis `k` is operand axis
    /// `perm[k]` (result dim sizes reordered to match). Used by the fanned
    /// Dirichlet draw to reorient the axis-0 column stack `[d, m]` (the d
    /// per-component `[m]` Gamma columns [`Emitter::vector`] stacks on dim 0)
    /// into the `[m, d]` batch of simplex rows. Panics on a non-`Ranked`
    /// operand or a permutation whose length differs from the operand rank —
    /// an internal invariant violation, per this module's doc comment.
    /// Parser-validated against the real StableHLO parser (jax 0.10.2) for the
    /// rank-2 `[d, m] -> [m, d]` case (`dims = [1, 0]`). Shape-only — same
    /// element-type-preserving contract as [`Emitter::slice`], so the result
    /// `elem` copies `a`'s.
    pub fn transpose(&mut self, a: &Value, perm: &[u64]) -> Value {
        let in_dims = match &a.ty {
            MlirTy::Ranked(dims) => dims.clone(),
            other => panic!("transpose expects a ranked operand, got {other:?}"),
        };
        assert_eq!(
            perm.len(),
            in_dims.len(),
            "transpose: permutation length must equal operand rank"
        );
        let out_dims: Vec<Option<u64>> = perm.iter().map(|&p| in_dims[p as usize]).collect();
        let result_ty = MlirTy::Ranked(out_dims);
        let operand_ty = a.ty.render(self.dtype, a.elem);
        let result_ty_text = result_ty.render(self.dtype, a.elem);
        let dims_text = perm
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.transpose {}, dims = [{dims_text}] : ({operand_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: a.elem,
        }
    }

    // ---- CHLO special functions ------------------------------------------

    /// `%N = chlo.lgamma %a : in_ty -> out_ty` — the log-gamma function.
    /// Unlike the `stablehlo.*` elementary unary ops, `chlo.lgamma` is a
    /// function-type op (its operand and result types are separated by
    /// `->`, both spelled out) rather than the single-`: ty` form `unary`
    /// emits — elementwise here, so `in_ty == out_ty`, but both must still
    /// be written for the op to parse. Real-only (like [`Emitter::log`]):
    /// `a` is [`Emitter::convert`]ed to [`ElemKind::Real`] first — e.g. a
    /// literal-parameter `Gamma(2, 1)`'s `shape` reaches
    /// `crate::registry::gamma_logpdf`'s `lgamma(&shape)` as a bare `Int`
    /// value constant, with no `NodeId` for the caller to coerce against an
    /// inferred result kind (`crate::registry` calls this directly, never
    /// through `crate::ops`'s dispatch table).
    pub fn lgamma(&mut self, a: &Value) -> Value {
        let a = &self.convert(a, ElemKind::Real);
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = chlo.lgamma {} : {ty_text} -> {ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    // VonMises log-I₀ (Task 10) must inline a polynomial approximation —
    // `chlo.bessel_i0e` is not a real CHLO op (no pretty or generic form
    // parses), so there is no op helper for it here.

    // ---- reductions -------------------------------------------------------

    /// Full reduction (all axes) to a scalar via repeated `stablehlo.add`.
    pub fn reduce_sum(&mut self, a: &Value) -> Value {
        self.reduce_full("stablehlo.add", "0.000000e+00", a)
    }

    /// Full reduction (all axes) to a scalar via repeated `stablehlo.maximum`.
    pub fn reduce_max(&mut self, a: &Value) -> Value {
        let identity = reduce_max_identity(self.dtype);
        self.reduce_full("stablehlo.maximum", identity, a)
    }

    /// Reduce ONLY the innermost (last) axis via `stablehlo.add`, leaving every
    /// outer axis intact: `[m, n] → [m]`, `[m, n, d] → [m, n]`, `[n] → Scalar`.
    /// Unlike [`Emitter::reduce_sum`] (which collapses EVERY axis to a scalar
    /// via [`Emitter::reduce_full`]), this reduces the single last dimension —
    /// the per-lane reduction a fanned discrete draw needs, where the outer
    /// `[m]` fan-out axis must SURVIVE while the distribution's own inner axis
    /// (a Binomial's `n` Bernoulli trials, [`binomial_sample`]) is summed away
    /// to one count per lane. For a rank-1 `[n]` operand the last axis is axis
    /// 0, so this emits the identical `stablehlo.reduce(... dimensions = [0]
    /// ...)` [`Emitter::reduce_sum`] does — the scalar Binomial path is
    /// unchanged whichever it calls. Emits one `stablehlo.reduce` over
    /// `dimensions = [<last>]` (via [`Emitter::reduce_axis`]). Panics on a
    /// rank-0 (`Scalar`) or non-`Ranked` operand — no inner axis to reduce, an
    /// internal invariant violation mirroring [`Emitter::reduce_axis`]'s
    /// panic-on-bad-shape discipline.
    pub fn reduce_sum_last_axis(&mut self, a: &Value) -> Value {
        let rank = match &a.ty {
            MlirTy::Ranked(dims) => dims.len(),
            other => panic!("reduce_sum_last_axis expects a ranked operand, got {other:?}"),
        };
        assert!(
            rank >= 1,
            "reduce_sum_last_axis: operand must have rank >= 1 (a last axis to reduce)"
        );
        self.reduce_axis("stablehlo.add", "0.000000e+00", a, rank - 1)
    }

    /// Shared full-reduction lowering: reduces axis 0 with [`reduce_axis`]
    /// once per rank, which collapses an `n`-D tensor to a scalar (an
    /// already-`Scalar` operand takes the zero-iteration path unchanged).
    fn reduce_full(&mut self, combine_op: &str, identity_lit: &str, a: &Value) -> Value {
        let rank = match &a.ty {
            MlirTy::Scalar => 0,
            MlirTy::Ranked(dims) => dims.len(),
            MlirTy::Tuple(_) => panic!("reduce over a tuple type has no lowering"),
            MlirTy::Key => panic!("reduce over an rng key has no lowering"),
        };
        let mut cur = a.clone();
        for _ in 0..rank {
            cur = self.reduce_axis(combine_op, identity_lit, &cur, 0);
        }
        cur
    }

    /// A single-axis reduction: reduces `axis` of the `n`-D `Ranked` operand
    /// `a`, leaving an `(n-1)`-D tensor (or a `Scalar` when `n == 1`), via
    /// `stablehlo.reduce`'s pretty form: `stablehlo.reduce(%in init: %init)
    /// applies {combine_op} across dimensions = [axis] : (in_ty, init_ty) ->
    /// out_ty` — no region block needed (unlike the generic form).
    ///
    /// Private: used by [`Emitter::reduce_full`] (repeatedly, to reach a
    /// scalar) and by [`Emitter::diag`]'s row-sum. The public reduction API
    /// (`reduce_sum`/`reduce_max`) always fully reduces to a scalar; a
    /// partial per-axis reduction is not yet part of the typed op-helper API.
    ///
    /// The init constant and result carry `a`'s own `elem` (e.g. `sum` over
    /// an `Int`-typed array stays `Int` end to end) rather than a hardcoded
    /// `Real` — StableHLO requires a `stablehlo.reduce`'s operand/init/result
    /// element types to all agree. `identity_lit` is used verbatim only for a
    /// `Real` operand (byte-identical to before this fix — every existing
    /// caller); a non-`Real` operand needs its OWN identity literal in that
    /// kind's own syntax, not `identity_lit`'s float formatting (`"0"`/
    /// `"false"`, never the float-only `"0.000000e+00"` or a dtype-exact
    /// -inf bit pattern). Only the additive (`stablehlo.add`) identity has a
    /// non-`Real` form implemented — `reduce_max` is only ever reached via
    /// `ops::lower_logsumexp`, whose vector argument is always `Real` by
    /// construction (see its own doc comment: every element is a
    /// `logdensityof` term), so a non-`Real` operand reaching the `maximum`
    /// combine is an internal invariant violation, not a case this emitter
    /// has a literal for.
    fn reduce_axis(
        &mut self,
        combine_op: &str,
        identity_lit: &str,
        a: &Value,
        axis: usize,
    ) -> Value {
        let dims = match &a.ty {
            MlirTy::Ranked(dims) => dims.clone(),
            other => panic!("reduce_axis expects a ranked operand, got {other:?}"),
        };
        let mut result_dims = dims;
        result_dims.remove(axis);
        let result_ty = if result_dims.is_empty() {
            MlirTy::Scalar
        } else {
            MlirTy::Ranked(result_dims)
        };

        let elem_ty = MlirTy::Scalar.render(self.dtype, a.elem);
        let operand_ty = a.ty.render(self.dtype, a.elem);
        let result_ty_text = result_ty.render(self.dtype, a.elem);

        let init_lit: &str = match a.elem {
            ElemKind::Real => identity_lit,
            ElemKind::Int => "0",
            ElemKind::Bool => "false",
        };
        assert!(
            a.elem == ElemKind::Real || combine_op == "stablehlo.add",
            "reduce_axis: a non-Real reduction identity is only implemented for the \
             additive (stablehlo.add) combine — reduce_max's dtype-exact -inf bit \
             pattern is only ever reached via logsumexp, whose vector argument is \
             always Real by construction"
        );

        let init_ssa = self.fresh();
        self.push(&format!(
            "{init_ssa} = stablehlo.constant dense<{init_lit}> : {elem_ty}"
        ));

        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.reduce({} init: {init_ssa}) applies {combine_op} across dimensions = [{axis}] : ({operand_ty}, {elem_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: a.elem,
        }
    }

    // ---- matrix helpers -----------------------------------------------------

    /// `%N = stablehlo.cholesky %a, lower = true : ty` — the lower-triangular
    /// Cholesky factor of `a` (shape-preserving: same square-matrix `MlirTy`).
    pub fn cholesky(&mut self, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = stablehlo.cholesky {}, lower = true : {ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// The diagonal of a square matrix `a` (`Ranked([n, n])`) as a length-`n`
    /// vector — used by multivariate-normal-style densities for the
    /// log-determinant of a Cholesky factor (`2 * sum(log(diag(chol)))`).
    ///
    /// StableHLO has no native "extract diagonal" op, so this lowers via the
    /// standard iota/compare/select/reduce idiom: build row- and
    /// column-index tensors, mask everything off the diagonal to zero, then
    /// row-sum (exactly one nonzero survives per row).
    pub fn diag(&mut self, a: &Value) -> Value {
        match &a.ty {
            MlirTy::Ranked(dims) if dims.len() == 2 => {}
            other => panic!("diag expects a rank-2 (square matrix) operand, got {other:?}"),
        }
        let mat_ty = a.ty.clone();
        let mat_ty_text = mat_ty.render(self.dtype, ElemKind::Real);

        let row_ssa = self.fresh();
        self.push(&format!(
            "{row_ssa} = stablehlo.iota dim = 0 : {mat_ty_text}"
        ));
        let row = Value {
            ssa: row_ssa,
            ty: mat_ty.clone(),
            elem: ElemKind::Real,
        };

        let col_ssa = self.fresh();
        self.push(&format!(
            "{col_ssa} = stablehlo.iota dim = 1 : {mat_ty_text}"
        ));
        let col = Value {
            ssa: col_ssa,
            ty: mat_ty.clone(),
            elem: ElemKind::Real,
        };

        let mask = self.compare("EQ", &row, &col);
        let zero = self.constant(0.0, mat_ty);
        let masked = self.select(&mask, a, &zero);

        self.reduce_axis("stablehlo.add", "0.000000e+00", &masked, 1)
    }

    /// Matrix-vector product `a @ b` via `stablehlo.dot_general`'s pretty
    /// form, contracting `a`'s (rank-2, `[m, n]`) last dimension against `b`'s
    /// (rank-1, `[n]`) only dimension: `a, b, contracting_dims = [1] x [0],
    /// precision = [DEFAULT, DEFAULT] : (a_ty, b_ty) -> r_ty`. The result
    /// takes `a`'s leading dimension (`[m]`), *not* `b`'s type — a `[m, n]`
    /// times `[n]` product has shape `[m]`, which only coincides with `b`'s
    /// `[n]` shape in the square (`m == n`) case.
    pub fn matvec(&mut self, a: &Value, b: &Value) -> Value {
        let a_dims = match &a.ty {
            MlirTy::Ranked(dims) if dims.len() == 2 => dims.clone(),
            other => panic!("matvec expects a rank-2 (matrix) lhs operand, got {other:?}"),
        };
        let b_dims = match &b.ty {
            MlirTy::Ranked(dims) if dims.len() == 1 => dims.clone(),
            other => panic!("matvec expects a rank-1 (vector) rhs operand, got {other:?}"),
        };
        if a_dims[1] != b_dims[0] {
            panic!(
                "matvec: lhs trailing dim {:?} does not match rhs length {:?}",
                a_dims[1], b_dims[0]
            );
        }

        let ssa = self.fresh();
        let a_ty = a.ty.render(self.dtype, a.elem);
        let b_ty = b.ty.render(self.dtype, b.elem);
        let result_ty = MlirTy::Ranked(vec![a_dims[0]]);
        let result_ty_text = result_ty.render(self.dtype, ElemKind::Real);
        self.push(&format!(
            "{ssa} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({a_ty}, {b_ty}) -> {result_ty_text}",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: ElemKind::Real,
        }
    }

    /// Batched row-wise mat-vec: apply the shared `[d, d]` matrix `l` to every
    /// row of `z` (`[n, d]`), yielding `[n, d]` whose row `i` is `l @ z_i` —
    /// the fanned MvNormal transform (Task 10b: `mu + L·z` over `n` independent
    /// standard-normal rows at once). Equal to `z @ lᵀ`: `result[i, j] = Σ_k
    /// z[i, k] · l[j, k] = (l @ z_i)[j]`, so it contracts `z`'s trailing dim
    /// against `l`'s TRAILING dim (`lᵀ`) — `stablehlo.dot_general`'s pretty form
    /// with `contracting_dims = [1] x [1]` (cf. [`Emitter::matvec`]'s `[1] x
    /// [0]` for the un-batched `l @ z`). The result takes `z`'s leading dim
    /// (`[n]`) then `l`'s leading dim (`[d]`). Panics on bad ranks / a
    /// contracting-dim mismatch (an internal invariant violation, mirroring
    /// [`Emitter::matvec`]).
    pub fn batched_row_matvec(&mut self, z: &Value, l: &Value) -> Value {
        let z_dims = match &z.ty {
            MlirTy::Ranked(dims) if dims.len() == 2 => dims.clone(),
            other => {
                panic!("batched_row_matvec expects a rank-2 (batch) lhs operand, got {other:?}")
            }
        };
        let l_dims = match &l.ty {
            MlirTy::Ranked(dims) if dims.len() == 2 => dims.clone(),
            other => {
                panic!("batched_row_matvec expects a rank-2 (matrix) rhs operand, got {other:?}")
            }
        };
        if z_dims[1] != l_dims[1] {
            panic!(
                "batched_row_matvec: lhs trailing dim {:?} does not match rhs trailing dim {:?}",
                z_dims[1], l_dims[1]
            );
        }

        let ssa = self.fresh();
        let z_ty = z.ty.render(self.dtype, z.elem);
        let l_ty = l.ty.render(self.dtype, l.elem);
        let result_ty = MlirTy::Ranked(vec![z_dims[0], l_dims[0]]);
        let result_ty_text = result_ty.render(self.dtype, ElemKind::Real);
        self.push(&format!(
            "{ssa} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({z_ty}, {l_ty}) -> {result_ty_text}",
            z.ssa, l.ssa
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: ElemKind::Real,
        }
    }

    /// Solve the lower-triangular system `l @ y = b` for `y`, via
    /// `stablehlo.triangular_solve` (`l: [n, n]`, `b: [n, k]` -> `y: [n,
    /// k]`). `b` must be a rank-2 MATRIX right-hand side — the real
    /// StableHLO parser (jax 0.10.2's `ir.Module.parse`) rejects a rank-1 `b`
    /// outright, unlike genuinely rank-generic ops such as [`Emitter::mul`];
    /// `k = n` solves `l @ Y = B` column-by-column (`registry.rs`'s
    /// `trace_via_frobenius`, Task 13 Wishart/InverseWishart, calls this with
    /// a square matrix `b`), and `k = 1` solves for a single vector reshaped
    /// to a `[n, 1]` column (`registry.rs`'s `mvnormal_logpdf`, Task 12,
    /// reshapes `x-mu` to `[n, 1]` before calling this and reshapes the
    /// `[n, 1]` result back to `[n]` afterwards — this fn does not reshape
    /// for the caller). `y`'s result type is always `b.ty` unchanged.
    /// `triangular_solve` has no pretty form, so this emits its parser-
    /// validated *generic* form verbatim (quoted op name, `<{...}>`
    /// properties dict: `left_side`/`lower`/`unit_diagonal`/`transpose_a`).
    pub fn tri_solve(&mut self, l: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let l_ty = l.ty.render(self.dtype, l.elem);
        let b_ty = b.ty.render(self.dtype, b.elem);
        let result_ty = b.ty.clone();
        let result_ty_text = result_ty.render(self.dtype, ElemKind::Real);
        self.push(&format!(
            "{ssa} = \"stablehlo.triangular_solve\"({}, {}) <{{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}}> : ({l_ty}, {b_ty}) -> {result_ty_text}",
            l.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: result_ty,
            elem: ElemKind::Real,
        }
    }

    // ---- sampling (Task 6) --------------------------------------------------

    /// Draw a standard `out_ty`-shaped variate from the threaded rng key
    /// (spec §07 rng ABI), advancing [`Emitter::cur_key`]. `dist` is the
    /// sampling family (`"NORMAL"`/`"UNIFORM"`), returning a standard normal
    /// or a uniform in `[0, 1)` — every one of the 26 distribution builders
    /// that call this applies its OWN location/scale to the standard draw
    /// (e.g. `normal_sample`'s `mu + sigma * z`, in `crate::registry`), so
    /// there is no affine `a`/`b` here to duplicate that: an earlier revision
    /// threaded `a`/`b` bounds through this call and had every builder pass
    /// the identity `(0, 1)`, making the affine dead ops on every single draw.
    ///
    /// Fan-out (Tier 1): when [`Emitter::set_batch_shape`] has set a `[n]`
    /// batch shape, the draw is sized to `[n]` instead of `out_ty` (one
    /// `rng_bit_generator` advance for the whole iid batch — spec §07 size
    /// dims); the calling straight-line builder's own scalar params broadcast
    /// over it via [`Emitter::binary`]. This is why the builders stay
    /// unchanged for both the scalar and the fanned draw.
    ///
    /// Threaded, not XLA-seeded: raw bits come from
    /// `stablehlo.rng_bit_generator` on `self.cur_key` (which this call then
    /// replaces with the generator's advanced state), mapped to a uniform in
    /// `[0, 1)` and — for `NORMAL` — through the `chlo.erf_inv` probit. Every
    /// op form is the exact text pinned in the rng-threaded-rand plan's Task-1
    /// spike (parse-validated against the real StableHLO parser, jax 0.10.2,
    /// and Enzyme-executed). See [`Emitter::rng_bit_generator_uniform`] /
    /// [`Emitter::uniform_to_normal`].
    ///
    /// Panics (an internal invariant violation, not a user-facing refusal —
    /// mirrors `diag`/`matvec`'s panic-on-bad-shape discipline) if `out_ty`
    /// has a dynamic dimension or is a `Tuple`/`Key` (no static bits-tensor
    /// form), or if `dist` is not a supported family — and, via
    /// [`Emitter::cur_key`], if no key is threaded.
    pub fn rng(&mut self, dist: &str, out_ty: &MlirTy) -> Value {
        // Fan-out override (spec §07 size dims): a batched iid draw sizes the
        // draw by `batch_shape`, ignoring the per-element `out_ty` the builder
        // passed — one `rng_bit_generator` advance yields the whole `[n]` batch.
        // A `None` batch shape (the scalar case) leaves the draw at `out_ty`.
        let draw_ty = match &self.batch_shape {
            Some(dims) => MlirTy::Ranked(dims.iter().map(|d| Some(*d)).collect()),
            None => out_ty.clone(),
        };

        // Draw uniform bits from (and advance) the threaded key.
        let (new_key, u01) = self.rng_bit_generator_uniform(&draw_ty);
        self.cur_key = Some(new_key);

        match dist {
            "UNIFORM" => u01,
            "NORMAL" => self.uniform_to_normal(&u01),
            other => panic!("rng: unsupported distribution family {other:?}"),
        }
    }

    /// Emit a `stablehlo.constant` with the verbatim literal text `lit` (not a
    /// re-formatted `f64`) at `ty`'s shape — for the rng math's pinned
    /// dtype-exact float constants (`2^-23`, `√2`), whose spike-validated
    /// spellings must be reproduced exactly rather than round-tripped through
    /// [`render_float_literal`].
    fn const_lit(&mut self, lit: &str, ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let ty_text = ty.render(self.dtype, ElemKind::Real);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{lit}> : {ty_text}"
        ));
        Value {
            ssa,
            ty,
            elem: ElemKind::Real,
        }
    }

    /// Draw `out_ty`-shaped raw bits from the threaded key and map them to a
    /// uniform in `[0, 1)`, returning `(advanced_key, uniform)`. Emits the
    /// plan's Task-1-pinned op forms: `stablehlo.rng_bit_generator` in its
    /// custom-assembly `THREE_FRY` spelling (the attribute-dict form is
    /// rejected by the parser; the pretty-printer's two spaces after
    /// `algorithm =` are the exact round-tripped text), then a shift-right /
    /// `convert` / multiply-by-scale bits→uniform sequence — DTYPE-AWARE, not
    /// a fixed f32-mantissa pipeline hardwired regardless of the emitter's
    /// configured precision: `Dtype::F32` draws `ui32` bits, shifts right 9
    /// (keeping the top `32 - 9 = 23` bits, an f32 mantissa's width) and
    /// scales by `2^-23`; `Dtype::F64` draws `ui64` bits, shifts right 12
    /// (keeping the top `64 - 12 = 52` bits, an f64 mantissa's width) and
    /// scales by `2^-52`. Using the f32 pipeline unconditionally for an
    /// `F64` emitter would silently quantize every `@sample` draw to ~2^23
    /// levels regardless of `dtype`; matching the shift to the mantissa width
    /// (so the shifted integer's range is exactly `[0, 2^mantissa)`) is also
    /// what keeps the scaled result inside `[0, 1)` — a shift one bit
    /// shallower would let the integer reach `2^mantissa` and the draw touch
    /// `1.0`. The bits tensor's element type is always `ui32`/`ui64` per the
    /// above (the generator's raw output width, never this emitter's float
    /// `dtype`); its shape follows `out_ty` (rank-0 for a scalar draw,
    /// `tensor<N x {ui32,ui64}>` for a length-`N` batch). Panics on a
    /// dynamic/`Tuple`/`Key` `out_ty` ([`render_bits_ty`] has no such form).
    fn rng_bit_generator_uniform(&mut self, out_ty: &MlirTy) -> (Value, Value) {
        let key = self.cur_key();
        let key_ty_text = MlirTy::Key.render(self.dtype, ElemKind::Real);
        let bits_ty_text = render_bits_ty(out_ty, self.dtype);
        let float_ty_text = out_ty.render(self.dtype, ElemKind::Real);

        let state_ssa = self.fresh();
        let bits_ssa = self.fresh();
        self.push(&format!(
            "{state_ssa}, {bits_ssa} = stablehlo.rng_bit_generator {}, algorithm =  THREE_FRY : ({key_ty_text}) -> ({key_ty_text}, {bits_ty_text})",
            key.ssa
        ));
        let new_key = Value {
            ssa: state_ssa,
            ty: MlirTy::Key,
            elem: ElemKind::Real,
        };

        // (shift, scale) per dtype: shift keeps the top `mantissa_bits` of the
        // raw integer (`32 - 9 = 23` for f32, `64 - 12 = 52` for f64); scale
        // is `2^-mantissa_bits`, the pinned exact spellings.
        let (shift, scale_lit) = match self.dtype {
            Dtype::F32 => (9, "1.1920929E-7"),           // 2^-23
            Dtype::F64 => (12, "2.220446049250313E-16"), // 2^-52
        };
        let c_shift_ssa = self.fresh();
        self.push(&format!(
            "{c_shift_ssa} = stablehlo.constant dense<{shift}> : {bits_ty_text}"
        ));
        let hi_ssa = self.fresh();
        self.push(&format!(
            "{hi_ssa} = stablehlo.shift_right_logical {bits_ssa}, {c_shift_ssa} : {bits_ty_text}"
        ));
        let f_ssa = self.fresh();
        self.push(&format!(
            "{f_ssa} = stablehlo.convert {hi_ssa} : ({bits_ty_text}) -> {float_ty_text}"
        ));
        let scale = self.const_lit(scale_lit, out_ty.clone());
        let u_ssa = self.fresh();
        self.push(&format!(
            "{u_ssa} = stablehlo.multiply {f_ssa}, {} : {float_ty_text}",
            scale.ssa
        ));
        let u = Value {
            ssa: u_ssa,
            ty: out_ty.clone(),
            elem: ElemKind::Real,
        };
        (new_key, u)
    }

    /// Map a uniform-in-`[0, 1)` draw `u` to a standard normal via the plan's
    /// Task-1-pinned probit path (Path A, which won over Box–Muller):
    /// `z = √2 · erf_inv(2u − 1)`. Shape-preserving; `chlo.erf_inv` is the
    /// CHLO function-type op (`operand-ty -> result-ty`), same assembly shape
    /// as [`Emitter::lgamma`].
    fn uniform_to_normal(&mut self, u: &Value) -> Value {
        let two = self.constant(2.0, u.ty.clone());
        let one = self.constant(1.0, u.ty.clone());
        let t = self.mul(u, &two);
        let s = self.sub(&t, &one);
        let e = self.erf_inv(&s);
        let sqrt2 = self.const_lit("1.4142135", u.ty.clone());
        self.mul(&e, &sqrt2)
    }

    /// `%N = chlo.erf %a : ty -> ty` — the error function, a CHLO function-type
    /// op like [`Emitter::erf_inv`]/[`Emitter::lgamma`] (parses + Enzyme-
    /// executes; a golden using it must therefore carry the `chlo` dialect).
    /// The Normal CDF's core (spec §07 `builtin_touniform`, see
    /// [`crate::registry::normal_cdf`]).
    pub(crate) fn erf(&mut self, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = chlo.erf {} : {ty_text} -> {ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// `atan(a)` (the arctangent, `(-π/2, π/2)`), via the core StableHLO binary
    /// `stablehlo.atan2(a, 1)` — `atan2(y, x)` is `atan(y/x)` in the correct
    /// quadrant, so `atan2(a, 1) = atan(a)`. Preferred over a unary `chlo.atan`
    /// because `stablehlo.atan2` is a core StableHLO op (guaranteed parse +
    /// Enzyme-differentiable). The Cauchy CDF's core (spec §07
    /// `builtin_touniform`, see [`crate::registry::cauchy_cdf`]); `atan(±inf) =
    /// ±π/2` gives the correct `F(±inf) = {1, 0}` limits.
    pub(crate) fn atan(&mut self, a: &Value) -> Value {
        let one = self.constant(1.0, a.ty.clone());
        self.binary("stablehlo.atan2", a, &one)
    }

    /// `%N = chlo.erf_inv %a : ty -> ty` — the inverse error function (the
    /// probit's core), a CHLO function-type op like [`Emitter::lgamma`]. Pinned
    /// in the plan's Task-1 spike (parses + Enzyme-executes; a golden using it
    /// must therefore carry the `chlo` dialect). Private: only
    /// [`Emitter::uniform_to_normal`] needs it.
    fn erf_inv(&mut self, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype, a.elem);
        self.push(&format!(
            "{ssa} = chlo.erf_inv {} : {ty_text} -> {ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// Broadcast a `Scalar` operand `s` up to `out_ty` (a no-op clone when
    /// `out_ty` is itself scalar), so [`Emitter::rng`]'s affine can lift a
    /// scalar bound onto a shaped (batched) draw — StableHLO's elementwise ops
    /// require identical operand shapes. Delegates to
    /// [`Emitter::broadcast_in_dim`]'s documented scalar form (`dims = []`).
    fn broadcast_scalar(&mut self, s: &Value, out_ty: &MlirTy) -> Value {
        if &s.ty == out_ty {
            s.clone()
        } else {
            self.broadcast_in_dim(s, &[], out_ty.clone())
        }
    }

    /// Reconcile two operands' SHAPES to a common broadcast shape (spec §04
    /// "Broadcasting": collections share RANK — no NumPy-style rank-
    /// prepending — and each axis either already matches or one side is
    /// size-1 and expands by repetition). Element KIND is assumed already
    /// reconciled by the caller — [`Emitter::binary`]/[`Emitter::compare`]/
    /// [`Emitter::select`] each [`Emitter::convert`] both operands to one
    /// [`ElemKind`] before calling this, exactly as they did before this
    /// helper existed. Returns both operands re-expressed at the common
    /// shape:
    ///
    /// - equal shapes (the overwhelming common case — every scalar
    ///   `@sample`/`@logdensity` path, inference-unified upstream, plus a
    ///   same-length batched pair) → both returned unchanged, no op
    ///   emitted: byte-identical to this crate's behavior before this
    ///   helper existed;
    /// - `(Scalar, Ranked)` / `(Ranked, Scalar)` → the scalar side is
    ///   splatted up via [`Emitter::broadcast_scalar`] (the existing
    ///   mechanism, unchanged) — a Tier-1/Tier-2 fan-out mixing a batched
    ///   draw with a scalar parameter/constant/bound;
    /// - `(Ranked(da), Ranked(db))` of equal rank, NOT already equal → the
    ///   axis-wise common size (both concrete and equal → that size; one
    ///   side `Some(1)` → the other; both `None` → `None`), then whichever
    ///   operand's shape differs from that common shape is broadcast up via
    ///   [`Emitter::broadcast_in_dim`] under the IDENTITY dimension map `[0,
    ///   1, …, rank-1]` (StableHLO's `broadcast_in_dim` expands a size-1
    ///   axis to the target size under an identity mapping) — the mechanism
    ///   an `iid(Dist, n)` density's length-1 array-of-records parameters
    ///   need to combine with the length-`n` observation vector
    ///   (`crate::registry`'s `Params::get`, feeding a rank-agnostic logpdf
    ///   builder's `Emitter::sub`/`div`/... calls).
    ///
    /// Panics on a genuinely incompatible pair (different rank, or an axis
    /// pair that is neither equal nor size-1-vs-concrete) rather than
    /// silently emitting a shape-mismatched op — an internal invariant
    /// upstream shape/type inference should have ruled out, matching
    /// [`Emitter::slice`]/[`Emitter::gather`]'s established
    /// refuse(panic)-don't-mislower discipline for this crate's infallible
    /// helpers (`binary`/`compare`/`select` have no `Result` to propagate a
    /// caller-facing [`EmitError`] through — see their own doc comments).
    fn broadcast_pair(&mut self, a: &Value, b: &Value) -> (Value, Value) {
        if a.ty == b.ty {
            return (a.clone(), b.clone());
        }
        match (&a.ty, &b.ty) {
            (MlirTy::Scalar, MlirTy::Ranked(_)) => (self.broadcast_scalar(a, &b.ty), b.clone()),
            (MlirTy::Ranked(_), MlirTy::Scalar) => (a.clone(), self.broadcast_scalar(b, &a.ty)),
            (MlirTy::Ranked(da), MlirTy::Ranked(db)) => {
                assert_eq!(
                    da.len(),
                    db.len(),
                    "broadcast_pair: rank mismatch ({da:?} vs {db:?}) — §04 broadcasting \
                     requires equal rank (addaxes handles rank differences upstream)"
                );
                let common: Vec<Option<u64>> = da
                    .iter()
                    .zip(db.iter())
                    .map(|(&x, &y)| match (x, y) {
                        (Some(m), Some(n)) if m == n => Some(m),
                        (Some(1), Some(n)) => Some(n),
                        (Some(m), Some(1)) => Some(m),
                        (None, None) => None,
                        _ => panic!(
                            "broadcast_pair: incompatible axis sizes ({x:?} vs {y:?}) in \
                             {da:?} vs {db:?} — neither equal nor size-1 (§04 broadcasting \
                             invariant violated upstream)"
                        ),
                    })
                    .collect();
                let common_ty = MlirTy::Ranked(common);
                let dims: Vec<u64> = (0..da.len() as u64).collect();
                let a_out = if a.ty == common_ty {
                    a.clone()
                } else {
                    self.broadcast_in_dim(a, &dims, common_ty.clone())
                };
                let b_out = if b.ty == common_ty {
                    b.clone()
                } else {
                    self.broadcast_in_dim(b, &dims, common_ty)
                };
                (a_out, b_out)
            }
            (ta, tb) => panic!(
                "broadcast_pair: unsupported shape pair ({ta:?}, {tb:?}) — no broadcast form"
            ),
        }
    }

    // ---- rejection sampling (Task 15) ---------------------------------------

    /// The configured floating-point element type — a narrow accessor
    /// (mirroring [`Emitter::node`]/[`Emitter::resolve`]) for
    /// `crate::registry`'s rejection samplers, which must render the
    /// float-typed carried variable of a [`Emitter::while_loop`] (its
    /// `tensor<f32>`/`tensor<f64>` result) as text alongside the fixed
    /// `tensor<i32>`/`tensor<i1>` loop-counter/accept-flag types — neither of
    /// which [`MlirTy`] can express (see [`Emitter::int_const`]/
    /// [`Emitter::bool_const`]).
    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// `%N = stablehlo.constant dense<x> : tensor<i32>` — a rank-0 signed
    /// 32-bit integer constant. StableHLO's `while`-loop counter (and the
    /// [`Emitter::dynamic_slice_scalar`] start index it feeds) is an INTEGER
    /// tensor, never this emitter's `f32`/`f64` element dtype, so — like
    /// [`Emitter::rng`]'s integer shape-constant — it is built as raw text
    /// here rather than through the dtype-parameterized [`Emitter::constant`].
    /// The returned [`Value`]'s `ty` is a placeholder [`MlirTy::Scalar`]: it
    /// must only ever be fed to the integer-typed helpers below
    /// ([`Emitter::int_add`]/[`Emitter::int_compare`]/
    /// [`Emitter::dynamic_slice_scalar`]), never a float op — whose `render`
    /// would (wrongly) spell it `tensor<f32>`.
    pub fn int_const(&mut self, x: i64) -> Value {
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{x}> : tensor<i32>"
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.constant dense<{true|false}> : tensor<i1>` — a rank-0
    /// boolean constant (the accept-flag carried variable's initial `false`).
    /// Same dtype-independent raw-text reasoning as [`Emitter::int_const`];
    /// its `ty` placeholder must only reach the [`render_i1`]-based helpers
    /// ([`Emitter::compare`]/[`Emitter::select`]/[`Emitter::and`]/
    /// [`Emitter::not`]), never a float op.
    pub fn bool_const(&mut self, b: bool) -> Value {
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{b}> : tensor<i1>"
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.constant dense<{i}> : tensor<i32|i64>` — a rank-0
    /// VALUE-path integer literal (`Node::Lit(Scalar::Int(_))`, spec §03),
    /// rendered at [`ElemKind::Int`] via [`MlirTy::render`] (dtype-configurable
    /// `i32`/`i64`, unlike the fixed-`i32` control-flow [`Emitter::int_const`]
    /// this is deliberately distinct from — that one is a loop counter, never
    /// reaching a FlatPDL value; this is the FlatPDL integer VALUE itself).
    pub fn int_value_const(&mut self, i: i64) -> Value {
        let ssa = self.fresh();
        let ty_text = MlirTy::Scalar.render(self.dtype, ElemKind::Int);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{i}> : {ty_text}"
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Int,
        }
    }

    /// `%N = stablehlo.constant dense<{true|false}> : tensor<i1>` — a rank-0
    /// VALUE-path boolean literal (`Node::Lit(Scalar::Bool(_))`, spec §03).
    /// Textually identical to [`Emitter::bool_const`] (`i1` is dtype-
    /// independent either way) but distinct in *kind*: the returned
    /// [`Value`]'s `elem` is [`ElemKind::Bool`], not the control-flow
    /// placeholder's [`ElemKind::Real`] — this is the FlatPDL boolean VALUE
    /// itself, not a loop's accept-flag.
    pub fn bool_value_const(&mut self, b: bool) -> Value {
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{b}> : tensor<i1>"
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Bool,
        }
    }

    /// `%N = stablehlo.add %a, %b : tensor<i32>` — integer add (the loop
    /// counter's `i + 1`). Separate from [`Emitter::add`] because that renders
    /// its operand type via the float [`Dtype`]; both operands here are the
    /// integer counter (see [`Emitter::int_const`]).
    pub fn int_add(&mut self, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.add {}, {} : tensor<i32>",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.compare {dir}, %a, %b, SIGNED : (tensor<i32>,
    /// tensor<i32>) -> tensor<i1>` — integer comparison (the loop counter's
    /// `i < MAXITER`). Unlike [`Emitter::compare`]'s float form, an integer
    /// comparison carries an explicit `SIGNED` `compare_type` (parser-
    /// validated against the real StableHLO parser, jax 0.10.2). The result
    /// is an `i1`, rendered like [`Emitter::compare`]'s.
    pub fn int_compare(&mut self, dir: &str, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.compare {dir}, {}, {}, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.and %a, %b : tensor<i1>` — boolean conjunction of two
    /// `i1` predicates (the rejection test's `v > 0 && log(u) < ...`). Both
    /// operands are [`Emitter::compare`]-shaped `i1`s; rendered via
    /// [`render_i1`], like `compare`/`select`.
    pub fn and(&mut self, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let ty = render_i1(&a.ty);
        self.push(&format!(
            "{ssa} = stablehlo.and {}, {} : {ty}",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.or %a, %b : tensor<i1>` — boolean disjunction of two
    /// `i1` predicates. Added for the batched (Tier-2 fan-out) rejection loop's
    /// per-lane `accepted := accepted || accept_this` carry (a lane latches
    /// once it first accepts). Same [`render_i1`] shape-rendering as
    /// [`Emitter::and`]; both operands share `a`'s shape (`tensor<i1>` scalar
    /// or `tensor<Nxi1>` batch).
    pub fn or(&mut self, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let ty = render_i1(&a.ty);
        self.push(&format!("{ssa} = stablehlo.or {}, {} : {ty}", a.ssa, b.ssa));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.not %a : tensor<i1>` — boolean negation of an `i1`
    /// predicate (the loop condition's `!accepted`). Rendered via
    /// [`render_i1`], like [`Emitter::and`].
    pub fn not(&mut self, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty = render_i1(&a.ty);
        self.push(&format!("{ssa} = stablehlo.not {} : {ty}", a.ssa));
        Value {
            ssa,
            ty: a.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// `%N = stablehlo.constant dense<{b}> : tensor<Nxi1>` — a rank-1 boolean
    /// (splat) constant, the batched (Tier-2 fan-out) rejection loop's initial
    /// per-lane `accepted` flags (all `false`). The `[n]` analogue of
    /// [`Emitter::bool_const`]: same dtype-independent raw-text reasoning (`i1`
    /// is never this emitter's float dtype), but its `ty` carries the `[n]`
    /// shape so the loop's [`Emitter::and`]/[`Emitter::or`]/[`Emitter::not`]
    /// render `tensor<Nxi1>`.
    pub fn bool_batch_const(&mut self, n: u64, b: bool) -> Value {
        let ty = MlirTy::Ranked(vec![Some(n)]);
        let ty_text = render_i1(&ty);
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{b}> : {ty_text}"
        ));
        Value {
            ssa,
            ty,
            elem: ElemKind::Real,
        }
    }

    /// Reduce a rank-1 `[n]` boolean (`i1`) tensor to a scalar `i1` via
    /// `stablehlo.reduce` with a `stablehlo.and` combine and a `true` identity
    /// — the "all lanes accepted" test the batched (Tier-2 fan-out) rejection
    /// loop's condition needs (`!all(accepted)`; `stablehlo` has no scalar
    /// boolean all-reduce op). Mirrors [`Emitter::reduce_axis`]'s pretty
    /// `stablehlo.reduce(... init: ...) applies ... across dimensions = [0]`
    /// form, but over `i1` (rendered via [`render_i1`], since [`MlirTy`] carries
    /// no boolean element type — see [`Emitter::and`]) rather than the float
    /// dtype. Returns a `Scalar`-shaped `i1` placeholder (like
    /// [`Emitter::bool_const`]); panics on a non-rank-1 operand (an internal
    /// invariant violation, mirroring the other shape-typed helpers).
    pub fn reduce_all(&mut self, a: &Value) -> Value {
        match &a.ty {
            MlirTy::Ranked(dims) if dims.len() == 1 => {}
            other => panic!("reduce_all expects a rank-1 (boolean vector) operand, got {other:?}"),
        }
        let operand_ty = render_i1(&a.ty);
        let scalar_i1 = render_i1(&MlirTy::Scalar);
        let init_ssa = self.fresh();
        self.push(&format!(
            "{init_ssa} = stablehlo.constant dense<true> : {scalar_i1}"
        ));
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.reduce({} init: {init_ssa}) applies stablehlo.and across dimensions = [0] : ({operand_ty}, {scalar_i1}) -> {scalar_i1}",
            a.ssa
        ));
        Value {
            ssa,
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        }
    }

    /// Extract element `index` (a runtime `i32` scalar — see
    /// [`Emitter::int_const`]) of the rank-1 tensor `operand` as a `Scalar`,
    /// via `stablehlo.dynamic_slice` + [`Emitter::reshape`] — the
    /// runtime-index analogue of the static-index slice+reshape idiom
    /// [`Emitter::slice`]/`registry::slice_indexed_prob` use.
    /// `stablehlo.dynamic_slice` clamps its start index into
    /// `[0, len - size]`, so an index at (or past) the batch length is safe —
    /// the rejection loop's counter never exceeds its bound while the loop
    /// runs, and even a clamped out-of-range read only re-reads the last batch
    /// element (never out-of-bounds memory). Panics on a non-rank-1 operand
    /// (an internal invariant violation, mirroring [`Emitter::diag`]/
    /// [`Emitter::matvec`]).
    pub fn dynamic_slice_scalar(&mut self, operand: &Value, index: &Value) -> Value {
        match &operand.ty {
            MlirTy::Ranked(dims) if dims.len() == 1 => {}
            other => panic!("dynamic_slice_scalar expects a rank-1 operand, got {other:?}"),
        }
        let operand_ty = operand.ty.render(self.dtype, operand.elem);
        let slice_ty = MlirTy::Ranked(vec![Some(1)]);
        let slice_ty_text = slice_ty.render(self.dtype, ElemKind::Real);
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.dynamic_slice {}, {}, sizes = [1] : ({operand_ty}, tensor<i32>) -> {slice_ty_text}",
            operand.ssa, index.ssa
        ));
        let sliced = Value {
            ssa,
            ty: slice_ty,
            elem: ElemKind::Real,
        };
        self.reshape(&sliced, MlirTy::Scalar)
    }

    /// Extract row `index` (a runtime `i32` scalar — see [`Emitter::int_const`])
    /// of a rank-2 `[m, n]` tensor `operand` as a rank-1 `[n]` vector, via
    /// `stablehlo.dynamic_slice` (`sizes = [1, n]`, a zero start on the trailing
    /// axis) + [`Emitter::reshape`] dropping the length-1 leading axis. The
    /// rank-2 analogue of [`Emitter::dynamic_slice_scalar`]: a batched (Tier-2
    /// fan-out) rejection loop reads its `[MAXITER, n]` pre-drawn candidate
    /// batch one `[n]` row per iteration this way (drawing the whole batch
    /// OUTSIDE the loop keeps the key advance fixed and the draw reproducible).
    /// Like `dynamic_slice`, the leading start index is clamped into range, so a
    /// counter at/past `MAXITER` only re-reads the last row (never out of
    /// bounds). Panics on a non-rank-2 (or dynamic-trailing-dim) operand — an
    /// internal invariant violation, mirroring [`Emitter::dynamic_slice_scalar`].
    pub fn dynamic_slice_row(&mut self, operand: &Value, index: &Value) -> Value {
        let n = match &operand.ty {
            MlirTy::Ranked(dims) if dims.len() == 2 => dims[1]
                .expect("dynamic_slice_row: trailing dim must be static (no dynamic ui32 form)"),
            other => panic!("dynamic_slice_row expects a rank-2 operand, got {other:?}"),
        };
        let operand_ty = operand.ty.render(self.dtype, operand.elem);
        let zero_i = self.int_const(0);
        let slice_ty = MlirTy::Ranked(vec![Some(1), Some(n)]);
        let slice_ty_text = slice_ty.render(self.dtype, ElemKind::Real);
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.dynamic_slice {}, {}, {}, sizes = [1, {n}] : ({operand_ty}, tensor<i32>, tensor<i32>) -> {slice_ty_text}",
            operand.ssa, index.ssa, zero_i.ssa
        ));
        let sliced = Value {
            ssa,
            ty: slice_ty,
            elem: ElemKind::Real,
        };
        self.reshape(&sliced, MlirTy::Ranked(vec![Some(n)]))
    }

    /// Emit a `stablehlo.while` carrying the [`Value`]s `inits` (one per
    /// carried variable), with `carried_tys[k]` the rendered MLIR type text of
    /// `inits[k]`. The types are passed explicitly because the loop counter's
    /// `tensor<i32>` and the accept-flag's `tensor<i1>` are types [`MlirTy`]
    /// cannot express (see [`Emitter::int_const`]/[`Emitter::bool_const`]).
    ///
    /// `cond` builds the loop-condition `i1` predicate from the carried
    /// variables (passed as the regions' block arguments); `body` builds the
    /// next carried-variable values (one per `inits` entry, same order). Both
    /// closures may reference values defined BEFORE the loop —
    /// `stablehlo.while` regions are not isolated-from-above — which is
    /// exactly how the rejection samplers read their pre-drawn candidate
    /// batches inside the loop body without redrawing (an in-loop
    /// [`Emitter::rng`] could repeat values in this XLA-seeded vertical; see
    /// the registry's `draw_gamma` doc comment).
    ///
    /// Returns the loop's results (`%r#0`, `%r#1`, …), each typed from its
    /// `inits` entry. The two region bodies are emitted into a scratch buffer
    /// (via `std::mem::take`/`replace` on `self.body`) so their op lines land
    /// inside the `cond {…}`/`do {…}` blocks rather than the enclosing
    /// function body; the shared `fresh()` counter keeps every SSA name
    /// globally unique across the swap. Parser-validated (the header's
    /// `%r:N = stablehlo.while(%arg = %init, …) : tys` form, the `cond`/`do`
    /// region keywords, region-captured outer operands, and the `%r#k`
    /// multi-result projection) against the real StableHLO parser, jax 0.10.2.
    pub fn while_loop(
        &mut self,
        inits: &[Value],
        carried_tys: &[String],
        cond: impl FnOnce(&mut Self, &[Value]) -> Value,
        body: impl FnOnce(&mut Self, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        assert_eq!(
            inits.len(),
            carried_tys.len(),
            "while_loop: inits/carried_tys length mismatch"
        );
        assert!(
            !inits.is_empty(),
            "while_loop: expected at least one carried variable"
        );

        // Region block-argument names (the iterArgs), shared by cond and body.
        let arg_names: Vec<String> = inits.iter().map(|_| self.fresh()).collect();
        let arg_values: Vec<Value> = arg_names
            .iter()
            .zip(inits)
            .map(|(n, init)| Value {
                ssa: n.clone(),
                ty: init.ty.clone(),
                elem: ElemKind::Real,
            })
            .collect();
        // The multi-result group name (%r:N -> %r#0, %r#1, ...).
        let result_name = self.fresh();

        // cond region, captured into its own buffer.
        let saved = std::mem::take(&mut self.body);
        let pred = cond(&mut *self, &arg_values);
        let cond_body = std::mem::replace(&mut self.body, saved);

        // do region, captured into its own buffer.
        let saved = std::mem::take(&mut self.body);
        let next = body(&mut *self, &arg_values);
        let do_body = std::mem::replace(&mut self.body, saved);
        assert_eq!(
            next.len(),
            inits.len(),
            "while_loop: body must return one value per carried variable"
        );

        let arity = inits.len();
        let bindings = arg_names
            .iter()
            .zip(inits)
            .map(|(n, init)| format!("{n} = {}", init.ssa))
            .collect::<Vec<_>>()
            .join(", ");
        let tys = carried_tys.join(", ");

        let mut text = String::new();
        text.push_str(&format!(
            "{result_name}:{arity} = stablehlo.while({bindings}) : {tys}\n"
        ));
        text.push_str("cond {\n");
        for line in cond_body.lines() {
            text.push_str("  ");
            text.push_str(line);
            text.push('\n');
        }
        text.push_str(&format!("  stablehlo.return {} : tensor<i1>\n", pred.ssa));
        text.push_str("} do {\n");
        for line in do_body.lines() {
            text.push_str("  ");
            text.push_str(line);
            text.push('\n');
        }
        let ret_ssas = next
            .iter()
            .map(|v| v.ssa.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        text.push_str(&format!("  stablehlo.return {ret_ssas} : {tys}\n"));
        text.push('}');
        self.push(&text);

        (0..arity)
            .map(|k| Value {
                ssa: format!("{result_name}#{k}"),
                ty: inits[k].ty.clone(),
                elem: ElemKind::Real,
            })
            .collect()
    }

    /// If `args` is `get0`/`get`'s `[container, index]` pair and `container`
    /// resolves (see [`Emitter::resolves_to_builtin_sample`]) to a
    /// `builtin_sample(...)` call, return the requested slot's ZERO-based
    /// index: `0` is the drawn-value slot — exactly what
    /// [`crate::registry::lower_sample`]'s dispatch already computes for the
    /// `builtin_sample` node itself, so `lower_node_uncached` reads it
    /// straight through rather than trying to tensor-slice a sampled
    /// `(value, new_rngstate)` pair, which has no rank-1-tensor form. `1` is
    /// the advanced rng-state slot, which has no tensor form at all in this
    /// vertical (see [`Emitter::rng`]'s doc comment). `base` distinguishes
    /// `get0` (0-based) from `get` (1-based), mirroring
    /// `ops::lower_builtin`'s own dispatch. `None` when `container` is not a
    /// sampled-tuple projection at all — the caller falls back to
    /// `ops::lower_builtin`'s ordinary rank-1-tensor `get`/`get0`.
    fn sample_tuple_slot(&self, args: &[NodeId], base: i64) -> Option<i64> {
        let [container, index] = <[NodeId; 2]>::try_from(args).ok()?;
        if !self.resolves_to_builtin_sample(container) {
            return None;
        }
        match self.m.node(index) {
            Node::Lit(Scalar::Int(i)) => Some(i - base),
            _ => None,
        }
    }

    /// If `args` is `get`/`get0`'s `[container, index]` pair and `container`
    /// resolves (one `(%ref self x)` hop, [`Emitter::resolve_ref_one`]) to a
    /// literal `tuple(...)` call with a literal-integer `index`, return the
    /// projected element's [`NodeId`]. The determiniser builds a
    /// `tuple(value, advanced_rng)` for a DESTRUCTURED `rand` (spec §07's full
    /// `(value, new_rstate)` contract) and then projects it with the parser's
    /// 1-based `get(_, 1)`/`get(_, 2)` (or a user's 0-based `get0`); this lets
    /// [`Emitter::lower_node`] follow that projection straight to the element
    /// (itself a `get0(builtin_sample, j)`), so a chained `rand` resolves
    /// value/advanced-key through the tuple without a tensor `get`. `None`
    /// when `container` is not a tuple literal (the caller then tries
    /// [`Emitter::sample_tuple_slot`], else the ordinary tensor `get`).
    fn tuple_projection(&self, args: &[NodeId], base: i64) -> Option<NodeId> {
        let [container, index] = <[NodeId; 2]>::try_from(args).ok()?;
        let resolved = self.resolve_ref_one(container);
        let elems = match self.m.node(resolved) {
            Node::Call(c) => match c.head {
                CallHead::Builtin(sym) if self.m.resolve(sym) == "tuple" => &c.args,
                _ => return None,
            },
            _ => return None,
        };
        let selector = match self.m.node(index) {
            Node::Lit(Scalar::Int(i)) => *i,
            _ => return None,
        };
        let idx = selector - base;
        if idx < 0 || idx as usize >= elems.len() {
            return None;
        }
        Some(elems[idx as usize])
    }

    /// If `args` is `get`/`get0`'s `[container, selector]` pair and
    /// `container` resolves (one `(%ref self x)` hop,
    /// [`Emitter::resolve_ref_one`]) to a `table(...)` or `record(...)`
    /// literal with a named entry matching the `selector`, return that
    /// entry's value [`NodeId`]. The parser lowers field access `obj.name`
    /// to `get(obj, "name")` with a string-literal selector
    /// (`flatppl_syntax::parser`); a bare-atom `Node::Const` selector is
    /// accepted too. A `table`/`record` has no monolithic tensor form
    /// (`ops::lower_builtin` refuses the `table`/`record` head), but a
    /// named-field projection selects one column/field — itself lowerable —
    /// so `datasets.exposure` reaches the column node directly instead of
    /// trying to lower the whole aggregate. `None` when `container` is not a
    /// table/record literal, the selector is not a field name, or no field
    /// matches (the caller then tries [`Emitter::tuple_projection`], else the
    /// ordinary tensor `get`).
    fn named_field_projection(&self, args: &[NodeId]) -> Option<NodeId> {
        let [container, selector] = <[NodeId; 2]>::try_from(args).ok()?;
        let resolved = self.resolve_ref_one(container);
        let named = match self.m.node(resolved) {
            Node::Call(c) => match c.head {
                CallHead::Builtin(sym) if matches!(self.m.resolve(sym), "table" | "record") => {
                    &c.named
                }
                _ => return None,
            },
            _ => return None,
        };
        let field = match self.m.node(selector) {
            Node::Lit(Scalar::Str(s)) => s.as_ref(),
            Node::Const(sym) => self.m.resolve(*sym),
            _ => return None,
        };
        named
            .iter()
            .find(|na| self.m.resolve(na.name) == field)
            .map(|na| na.value)
    }

    /// Resolve `id` through at most one level of `(%ref self x)` indirection
    /// (mirroring [`Emitter::lower_ref`]'s `SelfMod` case, and the
    /// determinizer's own `resolve_ref_one`: a shared latent's
    /// `builtin_sample` is bound to a name by
    /// `flatppl_determinizer::sample::lower_shared_record_sample`, an inline
    /// single draw's is not, via that module's `build_sample_term`) — a
    /// narrow accessor shared by [`Emitter::resolves_to_builtin_sample`]
    /// (below) and `crate::registry`'s matrix-distribution builders (Task
    /// 13), which need it to read a FIXED-phase kwarg field (e.g. `LKJ`'s
    /// `n`) down to its literal value: a fixed-phase binding's *use site* is
    /// exactly this one-level `(%ref self n)` indirection to the literal
    /// `(%bind n 3)`, never the literal inlined directly at the call site
    /// (spec §04's phase system). Returns `id` unchanged when it is not this
    /// shape (already a literal, a `Local`/`Module` ref, or any other node).
    pub(crate) fn resolve_ref_one(&self, id: NodeId) -> NodeId {
        match self.m.node(id) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => self
                .m
                .binding_by_name(*name)
                .map(|bid| self.m.binding(bid).rhs)
                .unwrap_or(id),
            _ => id,
        }
    }

    /// Whether `id` — resolved via [`Emitter::resolve_ref_one`] — is a
    /// `builtin_sample(...)` call.
    fn resolves_to_builtin_sample(&self, id: NodeId) -> bool {
        let resolved = self.resolve_ref_one(id);
        matches!(
            self.m.node(resolved),
            Node::Call(c) if matches!(
                c.head,
                CallHead::Builtin(sym) if self.m.resolve(sym) == "builtin_sample"
            )
        )
    }

    // ---- node dispatch (Task 4) ---------------------------------------------

    /// Pre-bind `id` to `value` in the memo map, without emitting any op.
    /// Used by the mode builder (Task 5+) to seed a model input's `NodeId`
    /// with its already-allocated `%argN` value before the body graph that
    /// references it is walked — [`Emitter::lower_node`]'s `Ref{Local, ..}`
    /// case (a `%local` function/kernel argument) refuses precisely because
    /// it expects the caller to have done this first, rather than guessing an
    /// argument's `Value` itself.
    pub fn bind(&mut self, id: NodeId, value: Value) {
        self.memo.insert(id, value);
    }

    /// Read a node from the underlying module. A narrow accessor for
    /// `crate::ops::lower_builtin` (a sibling module, so it cannot reach the
    /// private `m` field directly) to inspect a call's structure — e.g. a
    /// `get`/`get0` selector, which must be a literal, not a general
    /// expression to recursively lower.
    pub(crate) fn node(&self, id: NodeId) -> &Node {
        self.m.node(id)
    }

    /// Resolve an interned name. A narrow accessor mirroring [`Emitter::node`].
    pub(crate) fn resolve(&self, sym: Symbol) -> &str {
        self.m.resolve(sym)
    }

    /// Resolve a node's statically-known [`ValueSet`] (spec §03), read
    /// straight from the FlatPDL module's `Module::valueset_of` side table.
    /// A narrow accessor mirroring [`Emitter::node`]/[`Emitter::resolve`] —
    /// used by `registry::uniform_logpdf` to inspect a `support` set
    /// expression's closed-form Lebesgue measure (e.g. an `interval(lo,
    /// hi)` call's inferred `ValueSet::Interval(lo, hi)`) without lowering
    /// it as a tensor: a set expression has no tensor form of its own (see
    /// `ops::lower_in`'s identical structural, not-a-tensor treatment of
    /// `in`'s second argument).
    pub(crate) fn valueset_of(&self, id: NodeId) -> Option<&ValueSet> {
        self.m.valueset_of(id)
    }

    /// `id`'s inferred scalar kind (spec §03 boolean/integer/real), via
    /// [`crate::types::mlir_type_of`] — the downstream contract a value-
    /// producing op's result `elem` must satisfy (`crate::ops`'s operand-
    /// coercion arms read this to decide the target kind their operands
    /// convert to). Falls back to [`ElemKind::Real`] when `id` has no
    /// inferred type recorded (a hand-built test `Module` that never called
    /// `set_type` — every such existing test predates per-kind tensors and
    /// is built entirely from `Real` operands, so the fallback reproduces
    /// its prior all-`Real` behaviour exactly) or when the type has no
    /// tensor form at all (e.g. a residual measure-layer type): either way,
    /// a bare bool is not a meaningful signal to propagate as an
    /// [`EmitError`] from a `Result`-less accessor.
    pub(crate) fn node_kind(&self, id: NodeId) -> ElemKind {
        crate::types::mlir_type_of(self.m, id, self.dtype)
            .map(|(_, k)| k)
            .unwrap_or(ElemKind::Real)
    }

    /// Lower one FlatPDL node to a [`Value`], memoizing the result so a
    /// shared sub-expression — reached from more than one parent, e.g. a
    /// `Ref`ed top-level binding used twice, or a caller-[`Emitter::bind`]-
    /// bound argument read at several sites — is only ever emitted once:
    /// later calls for the same `id` return the *same* `Value` (same SSA
    /// name) without appending any further op text.
    pub fn lower_node(&mut self, id: NodeId) -> Result<Value, EmitError> {
        if let Some(v) = self.memo.get(&id) {
            return Ok(v.clone());
        }
        let value = self.lower_node_uncached(id)?;
        self.memo.insert(id, value.clone());
        Ok(value)
    }

    /// Lower `broadcast(f, rest...)` (§04 sec:broadcasting): apply the callable
    /// `f` (`args[0]`) elementwise over `rest`, scalars auto-broadcasting.
    ///
    /// `f` is either a bare builtin name (`Const`) — `Emitter::binary` and the
    /// registry logpdf builders are rank-agnostic (scalar↔rank-1 auto-broadcast),
    /// so the `broadcast`+`f` wrapper is stripped and `rest` routed to the SAME
    /// handler the un-broadcast form uses, the batch shape flowing through the
    /// arithmetic — or a **reified user function** (`functionof`, reached via a
    /// `SelfMod` ref to a top-level fn binding or inline). FlatPPL is loop-free,
    /// so a `functionof` under `broadcast` is a deterministic elementwise map:
    /// it is monomorphised by binding each declared input to the (already-
    /// lowered, possibly rank-1) broadcast argument and lowering the body — the
    /// body's own pure-arithmetic ops then auto-broadcast, exactly as the bare-
    /// builtin path relies on (`crate::ops::lower_builtin`). See
    /// [`Emitter::lower_broadcast_userfn`].
    ///
    /// `broadcast(add, s, vec)` → `ops::lower_builtin("add", …)` (a rank-1 add);
    /// `broadcast(predict, a=…, b=…, x=vec)` with `predict(a,b,x)=a+b*x` inlines
    /// to a rank-1 `a + b*x`; the dotted density `broadcast(builtin_logdensityof,
    /// Dist, broadcast(record, …), vec)` → `registry::lower_logdensityof` over the
    /// batched record + vector variate, yielding a rank-1 vector of
    /// log-densities (its `sum` caller reduces it to the iid log-likelihood).
    fn lower_broadcast(&mut self, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
        let f = *args
            .first()
            .ok_or_else(|| EmitError::at(id, "broadcast: missing callable"))?;
        let fname = match self.m.node(f) {
            Node::Const(sym) => self.m.resolve(*sym).to_string(),
            _ => {
                // Not a bare builtin: the only other lowerable callable is a
                // reified `functionof` (a user function passed in higher-order
                // position — the determiniser inlines direct user calls, but a
                // callable under `broadcast` survives as a first-class value).
                // Anything else (a kernel, a `%local`, an unresolved ref) is
                // genuinely un-lowerable and refuses — refuse-don't-mislower.
                if let Some(fn_id) = self.resolve_functionof(f) {
                    return self.lower_broadcast_userfn(id, fn_id, &args[1..]);
                }
                return Err(EmitError::at(
                    f,
                    "broadcast: callable must be a bare builtin name or a reified function",
                ));
            }
        };
        let rest = &args[1..];
        if fname == "builtin_logdensityof" {
            // Batched density: `Params::field_id` reads the batched
            // `broadcast(record, %kwarg…)` kernel input; a rank-agnostic logpdf
            // builder auto-broadcasts over the rank-1 variate → a rank-1
            // log-density vec. GUARD: only a rank-agnostic (pure-arithmetic
            // univariate) distribution is sound here — a structural builder
            // (matrix/gather/reduce/`support`) would drive the batched inputs to
            // shape-inconsistent StableHLO. Refuse a non-batch-safe dist rather
            // than mislower (refuse-don't-mislower). See `registry::is_batch_safe`.
            let dist = rest.first().and_then(|&d| match self.m.node(d) {
                Node::Const(sym) => Some(self.m.resolve(*sym).to_string()),
                _ => None,
            });
            match dist {
                Some(d) if crate::registry::is_batch_safe(&d) => {
                    crate::registry::lower_logdensityof(self, id, rest)
                }
                Some(d) => Err(EmitError::at(
                    id,
                    format!(
                        "broadcast over builtin_logdensityof of '{d}' is unsupported: \
                         its density builder is not rank-agnostic (batched density is \
                         sound only for univariate pure-arithmetic distributions)"
                    ),
                )),
                None => Err(EmitError::at(
                    id,
                    "broadcast(builtin_logdensityof, …): distribution must be a bare constructor",
                )),
            }
        } else {
            // Elementwise arithmetic/unary (`add`/`mul`/… from `.+`/`.*`): the
            // op's `Emitter::binary`/`unary` auto-broadcasts scalar↔rank-1. An op
            // this emitter doesn't lower (e.g. `divide`) refuses there — same
            // message as its non-broadcast form.
            crate::ops::lower_builtin(self, id, &fname, rest)
        }
    }

    /// Resolve a `broadcast` callable node (`args[0]`) to a reified
    /// `functionof`, following a `SelfMod` ref to its top-level binding rhs
    /// (the common `predict = (a,b,x) -> …` case) or accepting an inline
    /// reification directly. Returns the reification node's [`NodeId`], or
    /// `None` for anything that is not a `functionof` (a bare builtin `Const`,
    /// a kernel, a `%local`, …) — the caller handles / refuses those.
    fn resolve_functionof(&self, callable: NodeId) -> Option<NodeId> {
        let mut id = callable;
        // Follow `SelfMod` ref hops (a fn bound to a name); bounded so a
        // pathological self-referential binding can't spin forever.
        for _ in 0..64 {
            match self.m.node(id) {
                Node::Ref(Ref {
                    ns: RefNs::SelfMod,
                    name,
                }) => {
                    id = self.m.binding(self.m.binding_by_name(*name)?).rhs;
                }
                Node::Call(c) => {
                    let is_functionof = matches!(
                        c.head,
                        CallHead::Builtin(s) if self.m.resolve(s) == "functionof"
                    );
                    return (is_functionof && c.inputs.is_some()).then_some(id);
                }
                _ => return None,
            }
        }
        None
    }

    /// Lower `broadcast(f, rest...)` where `f` is a reified `functionof`
    /// (`fn_id`) by monomorphising it elementwise (§04 sec:broadcasting; §05
    /// "Named functions" — a named function is sugar for `functionof`). FlatPPL
    /// is loop-free, so this application is a deterministic map: each declared
    /// input is bound to its broadcast argument (positional in `rest`, or by
    /// keyword name from the call's `%kwarg` entries, per §04 "Keyword arguments
    /// bind inputs by name … positional binding is also permitted"), then the
    /// body is lowered. The body's arithmetic ops auto-broadcast scalar↔rank-1
    /// exactly as the bare-builtin `broadcast` path relies on, so a scalar-and-
    /// vector mix (`a + b*x` with scalar `a`,`b` and rank-1 `x`) yields the
    /// right rank-1 result with no explicit iteration.
    ///
    /// Inputs are bound by seeding each body `%local` ref's `NodeId` in the memo
    /// (via [`Emitter::bind`], the same mechanism the mode builder uses for model
    /// arguments). The body subtree's memo entries are snapshotted and restored
    /// around the lowering, so the SAME `functionof` broadcast at two sites with
    /// different arguments re-lowers freshly rather than reusing a stale memo.
    fn lower_broadcast_userfn(
        &mut self,
        id: NodeId,
        fn_id: NodeId,
        positional_rest: &[NodeId],
    ) -> Result<Value, EmitError> {
        // The reified callable: `functionof(body, %specinputs ((param placeholder)…))`.
        // Read its body + ordered input list out up front (dropping the borrow
        // before the `&mut self` lowering below).
        let (body, entries) = match self.m.node(fn_id) {
            Node::Call(c) => {
                let body = *c.args.first().ok_or_else(|| {
                    EmitError::at(fn_id, "broadcast: reified function has no body")
                })?;
                let entries: Vec<(Symbol, Ref)> = match &c.inputs {
                    Some(Inputs::Spec(es)) => es.to_vec(),
                    Some(Inputs::Auto) => self
                        .m
                        .auto_inputs_of(fn_id)
                        .ok_or_else(|| {
                            EmitError::at(
                                fn_id,
                                "broadcast: reified function has an unresolved (%autoinputs) \
                                 input list",
                            )
                        })?
                        .to_vec(),
                    None => {
                        return Err(EmitError::at(
                            fn_id,
                            "broadcast: callable is not a reified function (no input list)",
                        ));
                    }
                };
                (body, entries)
            }
            _ => {
                return Err(EmitError::at(
                    fn_id,
                    "broadcast: callable did not resolve to a reified function",
                ));
            }
        };

        // The broadcast call's `%kwarg` entries — the by-name argument binding.
        let kwargs: Vec<(Symbol, NodeId)> = match self.m.node(id) {
            Node::Call(c) => c
                .named
                .iter()
                .filter(|n| n.kind == NamedKind::Kwarg)
                .map(|n| (n.name, n.value))
                .collect(),
            _ => Vec::new(),
        };

        // Bind each declared input to its argument, keyed by the body-side
        // `%local` placeholder name (`entry.1.name`), lowering the argument now
        // (it lives outside the body subtree — the caller's own expression).
        let mut local_values: HashMap<Symbol, Value> = HashMap::new();
        for (i, (param, placeholder)) in entries.iter().enumerate() {
            let arg = kwargs
                .iter()
                .find(|(k, _)| k == param)
                .map(|(_, v)| *v)
                .or_else(|| positional_rest.get(i).copied())
                .ok_or_else(|| {
                    EmitError::at(
                        id,
                        format!(
                            "broadcast: no argument for input '{}'",
                            self.m.resolve(*param)
                        ),
                    )
                })?;
            let value = self.lower_node(arg)?;
            local_values.insert(placeholder.name, value);
        }

        // Collect the body subtree's `NodeId`s (the walk stops at ref/lit leaves
        // — `for_each_child` yields nothing for a non-`Call`, so a `SelfMod` ref
        // is a leaf and its target binding is NOT pulled in and stays memoized).
        let mut subtree = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![body];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            subtree.push(n);
            self.m.node(n).for_each_child(|c| stack.push(c));
        }

        // Snapshot the subtree's prior memo state, then seed each body `%local`
        // ref with its bound argument value.
        let snapshot: Vec<(NodeId, Option<Value>)> = subtree
            .iter()
            .map(|&n| (n, self.memo.get(&n).cloned()))
            .collect();
        for &n in &subtree {
            if let Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }) = self.m.node(n)
            {
                if let Some(v) = local_values.get(name) {
                    self.bind(n, v.clone());
                }
                // A `%local` not among the declared inputs is a malformed
                // reification; leaving it unbound lets the body walk hit
                // `lower_ref`'s `Local` refusal — refuse-don't-mislower.
            }
        }

        let result = self.lower_node(body);

        // Restore memo isolation (whatever the outcome) so a second application
        // of the same `functionof` re-lowers against its own arguments.
        for (n, prev) in snapshot {
            match prev {
                Some(v) => {
                    self.memo.insert(n, v);
                }
                None => {
                    self.memo.remove(&n);
                }
            }
        }
        result
    }

    /// The uncached half of [`Emitter::lower_node`]'s dispatch: every FlatPDL
    /// leaf/call kind that can reach this emitter, matched once. `self.m` is
    /// read out as a plain `&'m Module` up front — an ordinary reference
    /// value copied out of the field, not a borrow of `self` — so the match
    /// arms below stay free to call back into `&mut self` (e.g. `self.add`,
    /// `self.lower_node`) while still holding a node/child reference derived
    /// from it.
    fn lower_node_uncached(&mut self, id: NodeId) -> Result<Value, EmitError> {
        let m: &'m Module = self.m;
        match m.node(id) {
            Node::Lit(Scalar::Int(i)) => Ok(self.int_value_const(*i)),
            Node::Lit(Scalar::Real(x)) => Ok(self.constant(*x, MlirTy::Scalar)),
            Node::Lit(Scalar::Bool(b)) => Ok(self.bool_value_const(*b)),
            Node::Lit(Scalar::Str(_)) => {
                Err(EmitError::at(id, "string literal has no tensor form"))
            }
            // A bare built-in constant (`inf`, `pi`, ...) — dispatched through
            // the same builtin-head map as a zero-arg call, so `inf`'s entry
            // there is the single source of truth for both spellings.
            Node::Const(sym) => {
                let name = m.resolve(*sym).to_string();
                crate::ops::lower_builtin(self, id, &name, &[])
            }
            Node::Ref(r) => self.lower_ref(id, *r),
            Node::Hole => Err(EmitError::at(id, "bare hole has no tensor form")),
            Node::Axis(_) => Err(EmitError::at(id, "axis label has no tensor form")),
            Node::Call(call) => match call.head {
                CallHead::Builtin(sym) => {
                    let name = m.resolve(sym).to_string();
                    // The registry gate: `builtin_logdensityof`/`builtin_sample`
                    // dispatch to the distribution registry (`crate::registry`),
                    // never to `ops::lower_builtin`'s deterministic
                    // (non-distribution) map — see that module's doc comment.
                    if name == "builtin_logdensityof" {
                        crate::registry::lower_logdensityof(self, id, &call.args)
                    } else if name == "builtin_sample" {
                        crate::registry::lower_sample(self, id, &call.args)
                    } else if name == "builtin_touniform" {
                        crate::registry::lower_touniform(self, id, &call.args)
                    } else if name == "broadcast" {
                        self.lower_broadcast(id, &call.args)
                    } else if matches!(name.as_str(), "get0" | "get") {
                        // `get0(builtin_sample(...), k)` / `get((%ref self
                        // <shared-latent>), k)`: a projection of a sampled
                        // `(value, new_rngstate)` pair (slot 0 = drawn value,
                        // slot 1 = advanced rng key), or a `get`/`get0` of a
                        // `tuple(value, advanced_rng)` the determiniser built
                        // for a destructured `rand` — neither is a real rank-1
                        // tensor. See `Emitter::sample_tuple_slot` /
                        // `Emitter::tuple_projection`. Anything else (the
                        // ordinary case) falls through to `ops::lower_builtin`'s
                        // generic rank-1-tensor `get`/`get0`.
                        let base = if name == "get0" { 0 } else { 1 };
                        if let Some(field) = self.named_field_projection(&call.args) {
                            return self.lower_node(field);
                        }
                        if let Some(elem) = self.tuple_projection(&call.args, base) {
                            return self.lower_node(elem);
                        }
                        match self.sample_tuple_slot(&call.args, base) {
                            Some(0) => self.lower_node(call.args[0]),
                            Some(1) => {
                                // Advanced rng key: lower the sample first
                                // (populating `sample_keys` — a `get0(_, 1)` may
                                // be visited before its `get0(_, 0)`), then read
                                // the recorded key.
                                let sample_node = self.resolve_ref_one(call.args[0]);
                                self.lower_node(call.args[0])?;
                                self.sample_key(sample_node).ok_or_else(|| {
                                    EmitError::at(
                                        id,
                                        "advanced rng key not recorded for this sample",
                                    )
                                })
                            }
                            Some(_) => Err(EmitError::at(
                                id,
                                "sample tuple has only slots 0 (value) and 1 (rng)",
                            )),
                            None => crate::ops::lower_builtin(self, id, &name, &call.args),
                        }
                    } else {
                        crate::ops::lower_builtin(self, id, &name, &call.args)
                    }
                }
                CallHead::User(_) => Err(EmitError::at(
                    id,
                    "user-callable application has no lowering (expected to be inlined by determinize)",
                )),
            },
        }
    }

    /// Resolve a `Ref` leaf. `SelfMod` dereferences through the module's
    /// top-level binding table and recurses (memoized, so re-`Ref`ing the
    /// same binding from several call sites still emits its RHS only once).
    /// `Local` (a `%local` function/kernel argument) refuses: the caller is
    /// expected to have pre-bound it via [`Emitter::bind`] before this node
    /// is ever visited, so reaching here means it didn't. `Module` (a
    /// standard-module member reference) has no lowering yet.
    fn lower_ref(&mut self, id: NodeId, r: Ref) -> Result<Value, EmitError> {
        match r.ns {
            RefNs::SelfMod => {
                let bid = self.m.binding_by_name(r.name).ok_or_else(|| {
                    EmitError::at(
                        id,
                        format!("unresolved reference '{}'", self.m.resolve(r.name)),
                    )
                })?;
                let rhs = self.m.binding(bid).rhs;
                self.lower_node(rhs)
            }
            RefNs::Local => Err(EmitError::at(
                id,
                "unbound %local reference (expected to be pre-bound by the caller via Emitter::bind)",
            )),
            RefNs::Module(_) => Err(EmitError::at(
                id,
                "module-member reference has no lowering yet",
            )),
        }
    }

    // ---- module assembly ----------------------------------------------------

    /// Wrap the accumulated body in `module { func.func @{name}(<args>) ->
    /// <ret-tys> { <body> return <ret-ssas> : <ret-tys> } }`, 2-space indented
    /// per nesting level (mirroring `flatppl_flatpir::writer`'s canonical-text
    /// formatting style).
    ///
    /// `rets` is a slice so this serves both the single-result `@logdensity`
    /// output and the multi-result `@sample` `(value, new_key)` ABI (and
    /// buffy #107's record-output `@sample` later). A single-element slice
    /// renders `-> T` / `return %x : T` (no parenthesized tuple), byte-for-byte
    /// identical to the previous single-`ret` output; two-or-more render the
    /// parenthesized result-type list and comma-joined return.
    pub fn finish(
        self,
        func_name: &str,
        args: &[(String, MlirTy, ElemKind)],
        rets: &[&Value],
    ) -> String {
        debug_assert!(
            !rets.is_empty(),
            "finish requires at least one return value"
        );
        let dtype = self.dtype;
        let arg_list = args
            .iter()
            .map(|(name, ty, elem)| format!("{name}: {}", ty.render(dtype, *elem)))
            .collect::<Vec<_>>()
            .join(", ");
        let ret_tys: Vec<String> = rets.iter().map(|r| r.ty.render(dtype, r.elem)).collect();
        let ret_tys_joined = ret_tys.join(", ");
        let ret_ty_text = if ret_tys.len() == 1 {
            ret_tys_joined.clone()
        } else {
            format!("({ret_tys_joined})")
        };
        let ret_ssas = rets
            .iter()
            .map(|r| r.ssa.clone())
            .collect::<Vec<_>>()
            .join(", ");

        let mut out = String::from("module {\n");
        out.push_str(&format!(
            "  func.func @{func_name}({arg_list}) -> {ret_ty_text} {{\n"
        ));
        for line in self.body.lines() {
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!("    return {ret_ssas} : {ret_tys_joined}\n"));
        out.push_str("  }\n");
        out.push_str("}\n");
        out
    }
}

/// Format a float as an MLIR-parseable literal, always with a `.` so it
/// reads back as a float attribute rather than an integer one (same
/// reasoning as `flatpir::writer::render_real`, reimplemented locally since
/// that helper is private to the `flatpir` crate).
fn render_float_literal(x: f64) -> String {
    let s = format!("{x}");
    if s.contains(['.', 'e', 'E']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// Render `ty`'s shape as a boolean (`i1`-element) MLIR tensor type text.
/// `MlirTy` (Task 2) carries no element dtype, so this can't go through
/// `MlirTy::render` — see the module doc comment on why `compare`/`select`
/// need this local override instead of a `MlirTy` boolean variant.
fn render_i1(ty: &MlirTy) -> String {
    match ty {
        MlirTy::Scalar => "tensor<i1>".to_string(),
        MlirTy::Ranked(dims) => {
            let mut out = String::from("tensor<");
            for dim in dims {
                match dim {
                    Some(n) => out.push_str(&n.to_string()),
                    None => out.push('?'),
                }
                out.push('x');
            }
            out.push_str("i1");
            out.push('>');
            out
        }
        MlirTy::Tuple(_) => panic!("compare/select over a tuple type has no i1 rendering"),
        MlirTy::Key => panic!("compare/select over an rng key has no i1 rendering"),
    }
}

/// Render `ty`'s shape as a `ui32`- or `ui64`-element MLIR tensor type — the
/// raw-bits tensor `stablehlo.rng_bit_generator` produces (spec §07 rng ABI),
/// `ui32` for `Dtype::F32` and `ui64` for `Dtype::F64` (see
/// [`Emitter::rng_bit_generator_uniform`]'s doc comment on why the bits width
/// tracks `dtype`). `MlirTy` carries no element dtype, and the bits element
/// (`ui32`/`ui64`) is never this emitter's float dtype (`f32`/`f64`) either,
/// so — exactly like [`render_i1`] — this render is done locally rather than
/// through `MlirTy::render`.
fn render_bits_ty(ty: &MlirTy, dtype: Dtype) -> String {
    let elem = match dtype {
        Dtype::F32 => "ui32",
        Dtype::F64 => "ui64",
    };
    match ty {
        MlirTy::Scalar => format!("tensor<{elem}>"),
        MlirTy::Ranked(dims) => {
            let mut out = String::from("tensor<");
            for dim in dims {
                match dim {
                    Some(n) => out.push_str(&n.to_string()),
                    None => panic!("rng bits over a dynamic dimension has no static {elem} form"),
                }
                out.push('x');
            }
            out.push_str(elem);
            out.push('>');
            out
        }
        MlirTy::Tuple(_) => panic!("rng bits over a tuple type have no {elem} rendering"),
        MlirTy::Key => panic!("rng bits over an rng key have no {elem} rendering"),
    }
}
