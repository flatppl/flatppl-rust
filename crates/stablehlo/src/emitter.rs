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

use std::collections::HashMap;

use flatppl_core::{CallHead, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol, ValueSet};

use crate::Dtype;
use crate::mlir::{MlirTy, Value};
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
}

impl<'m> Emitter<'m> {
    pub fn new(m: &'m Module, dtype: Dtype) -> Self {
        Emitter {
            m,
            dtype,
            next: 0,
            memo: HashMap::new(),
            body: String::new(),
        }
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
        let ty_text = ty.render(self.dtype);
        let lit = render_float_literal(x);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{lit}> : {ty_text}"
        ));
        Value { ssa, ty }
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
        let ty_text = ty.render(self.dtype);
        let lit = pos_inf_literal(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{lit}> : {ty_text}"
        ));
        Value { ssa, ty }
    }

    /// One elementwise unary op: `%N = {op} %a : ty`. Result type copies the
    /// operand's `MlirTy` — elementwise ops are shape-preserving.
    pub fn unary(&mut self, op: &str, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype);
        self.push(&format!("{ssa} = {op} {} : {ty_text}", a.ssa));
        Value {
            ssa,
            ty: a.ty.clone(),
        }
    }

    /// One elementwise binary op: `%N = {op} %a, %b : ty`. Result type
    /// copies `a`'s `MlirTy` (operands are assumed already shape-unified by
    /// inference, upstream of this emitter).
    pub fn binary(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype);
        self.push(&format!("{ssa} = {op} {}, {} : {ty_text}", a.ssa, b.ssa));
        Value {
            ssa,
            ty: a.ty.clone(),
        }
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
    pub fn div(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.divide", a, b)
    }
    pub fn pow(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("stablehlo.power", a, b)
    }

    pub fn neg(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.negate", a)
    }
    pub fn log(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.log", a)
    }
    pub fn exp(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.exponential", a)
    }
    pub fn sqrt(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.sqrt", a)
    }
    pub fn abs(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.abs", a)
    }
    pub fn cos(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.cosine", a)
    }
    /// `stablehlo.sine` — a NEW op form for this crate (Task 14's Cauchy
    /// `@sample`, which needs `tan(t) = sin(t) / cos(t)`; no `chlo`/
    /// `stablehlo` `tan` op is used, mirroring [`Emitter::cos`]'s existing
    /// `stablehlo.cosine`). Parser-validated against the real StableHLO
    /// parser (jax 0.10.2, `jax._src.interpreters.mlir.make_ir_context`),
    /// same discipline as every other op text this module emits.
    pub fn sin(&mut self, a: &Value) -> Value {
        self.unary("stablehlo.sine", a)
    }

    /// `%N = stablehlo.compare {dir}, %a, %b : (lhs, rhs) -> i1-shape`.
    /// `dir` is a StableHLO `comparison_direction` (`"LT"`, `"GE"`, `"EQ"`,
    /// ...). The result is logically an `i1` tensor of `a`'s shape — see the
    /// module doc comment for why that is rendered via [`render_i1`] rather
    /// than through `MlirTy`/`Dtype`; the returned `Value`'s `ty` still
    /// carries `a`'s shape so a later [`Emitter::select`] can reuse it.
    pub fn compare(&mut self, dir: &str, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let lhs_ty = a.ty.render(self.dtype);
        let rhs_ty = b.ty.render(self.dtype);
        let result_ty = render_i1(&a.ty);
        self.push(&format!(
            "{ssa} = stablehlo.compare {dir}, {}, {} : ({lhs_ty}, {rhs_ty}) -> {result_ty}",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
        }
    }

    /// `%N = stablehlo.select %pred, %a, %b : (i1-shape, ty, ty) -> ty`.
    /// `c` is treated as an `i1` tensor of its own `MlirTy` shape (typically
    /// an [`Emitter::compare`] result) regardless of what element type its
    /// `MlirTy` would otherwise render as — see the module doc comment.
    pub fn select(&mut self, c: &Value, a: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let pred_ty = render_i1(&c.ty);
        let ty_text = a.ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.select {}, {}, {} : ({pred_ty}, {ty_text}, {ty_text}) -> {ty_text}",
            c.ssa, a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
        }
    }

    // ---- shape ops (Task 4: `get`/`get0`, `logsumexp`/`in` broadcasting) ---

    /// `%N = stablehlo.slice %a [s0:l0, s1:l1:t1, ...] : (operand_ty) ->
    /// result_ty` — a static per-axis slice (`starts`/`limits`/`strides`,
    /// one triple per `a`'s rank; StableHLO's pretty form omits `:stride`
    /// when it's `1`). Each result dimension is `(limit - start).div_ceil(stride)`.
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
        let operand_ty = a.ty.render(self.dtype);
        let result_ty_text = result_ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.slice {} [{}] : ({operand_ty}) -> {result_ty_text}",
            a.ssa,
            ranges.join(", ")
        ));
        Value { ssa, ty: result_ty }
    }

    /// `%N = stablehlo.reshape %a : (operand_ty) -> result_ty` — reinterprets
    /// `a`'s elements (same element count) under a different static shape,
    /// e.g. dropping `get0`/`get`'s now-length-1 sliced axis down to a
    /// `Scalar`.
    pub fn reshape(&mut self, a: &Value, ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let operand_ty = a.ty.render(self.dtype);
        let result_ty_text = ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.reshape {} : ({operand_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value { ssa, ty }
    }

    /// `%N = stablehlo.broadcast_in_dim %a, dims = [...] : (operand_ty) ->
    /// ty` — broadcasts `a` up to the (larger) shape `ty`, mapping `a`'s
    /// existing dimensions onto the `dims` positions of the result, in
    /// order. A rank-0 (`Scalar`) operand takes `dims = []`, StableHLO's
    /// documented scalar-broadcast form — the only shape this emitter's
    /// callers need today (`logsumexp`'s reduced max, `in`'s interval bounds,
    /// broadcast back up to the input vector/variate's shape; StableHLO's
    /// elementwise ops require identical operand shapes, no implicit
    /// broadcast).
    pub fn broadcast_in_dim(&mut self, a: &Value, dims: &[u64], ty: MlirTy) -> Value {
        let ssa = self.fresh();
        let operand_ty = a.ty.render(self.dtype);
        let result_ty_text = ty.render(self.dtype);
        let dims_text = dims
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        self.push(&format!(
            "{ssa} = stablehlo.broadcast_in_dim {}, dims = [{dims_text}] : ({operand_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value { ssa, ty }
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
        assert!(
            elems.iter().all(|v| v.ty == elem_ty),
            "vector: elements must have identical shape (ragged vector-of-vectors \
             must be refused by the caller before this is reached)"
        );
        let inner_dims: Vec<Option<u64>> = match &elem_ty {
            MlirTy::Scalar => Vec::new(),
            MlirTy::Ranked(dims) => dims.clone(),
            MlirTy::Tuple(_) => panic!("vector: tuple elements have no tensor form"),
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
            .map(|v| v.ty.render(self.dtype))
            .collect::<Vec<_>>()
            .join(", ");
        let result_ty_text = result_ty.render(self.dtype);

        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.concatenate {operand_ssas}, dim = 0 : ({operand_tys}) -> {result_ty_text}"
        ));
        Value { ssa, ty: result_ty }
    }

    // ---- CHLO special functions ------------------------------------------

    /// `%N = chlo.lgamma %a : in_ty -> out_ty` — the log-gamma function.
    /// Unlike the `stablehlo.*` elementary unary ops, `chlo.lgamma` is a
    /// function-type op (its operand and result types are separated by
    /// `->`, both spelled out) rather than the single-`: ty` form `unary`
    /// emits — elementwise here, so `in_ty == out_ty`, but both must still
    /// be written for the op to parse.
    pub fn lgamma(&mut self, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = chlo.lgamma {} : {ty_text} -> {ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
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

    /// Shared full-reduction lowering: reduces axis 0 with [`reduce_axis`]
    /// once per rank, which collapses an `n`-D tensor to a scalar (an
    /// already-`Scalar` operand takes the zero-iteration path unchanged).
    fn reduce_full(&mut self, combine_op: &str, identity_lit: &str, a: &Value) -> Value {
        let rank = match &a.ty {
            MlirTy::Scalar => 0,
            MlirTy::Ranked(dims) => dims.len(),
            MlirTy::Tuple(_) => panic!("reduce over a tuple type has no lowering"),
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

        let elem_ty = MlirTy::Scalar.render(self.dtype);
        let operand_ty = a.ty.render(self.dtype);
        let result_ty_text = result_ty.render(self.dtype);

        let init_ssa = self.fresh();
        self.push(&format!(
            "{init_ssa} = stablehlo.constant dense<{identity_lit}> : {elem_ty}"
        ));

        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.reduce({} init: {init_ssa}) applies {combine_op} across dimensions = [{axis}] : ({operand_ty}, {elem_ty}) -> {result_ty_text}",
            a.ssa
        ));
        Value { ssa, ty: result_ty }
    }

    // ---- matrix helpers -----------------------------------------------------

    /// `%N = stablehlo.cholesky %a, lower = true : ty` — the lower-triangular
    /// Cholesky factor of `a` (shape-preserving: same square-matrix `MlirTy`).
    pub fn cholesky(&mut self, a: &Value) -> Value {
        let ssa = self.fresh();
        let ty_text = a.ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.cholesky {}, lower = true : {ty_text}",
            a.ssa
        ));
        Value {
            ssa,
            ty: a.ty.clone(),
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
        let mat_ty_text = mat_ty.render(self.dtype);

        let row_ssa = self.fresh();
        self.push(&format!(
            "{row_ssa} = stablehlo.iota dim = 0 : {mat_ty_text}"
        ));
        let row = Value {
            ssa: row_ssa,
            ty: mat_ty.clone(),
        };

        let col_ssa = self.fresh();
        self.push(&format!(
            "{col_ssa} = stablehlo.iota dim = 1 : {mat_ty_text}"
        ));
        let col = Value {
            ssa: col_ssa,
            ty: mat_ty.clone(),
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
        let a_ty = a.ty.render(self.dtype);
        let b_ty = b.ty.render(self.dtype);
        let result_ty = MlirTy::Ranked(vec![a_dims[0]]);
        let result_ty_text = result_ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({a_ty}, {b_ty}) -> {result_ty_text}",
            a.ssa, b.ssa
        ));
        Value { ssa, ty: result_ty }
    }

    /// Solve the lower-triangular system `l @ y = b` for `y`, via
    /// `stablehlo.triangular_solve` (`l: [n, n]`, `b: [n]` -> `y: [n]`).
    /// Shape-generic in `b`: `stablehlo.triangular_solve` also accepts a
    /// MATRIX right-hand side (`b: [n, n]` -> `y: [n, n]`, solving `l @ Y =
    /// B` column-by-column) — `y`'s result type is always `b.ty` unchanged,
    /// never hardcoded to rank-1, which is what lets `registry.rs`'s
    /// `trace_via_frobenius` (Task 13, Wishart/InverseWishart) call this
    /// with a matrix `b` to compute `W = tri_solve(L_A, L_B)` for `tr(A^-1
    /// B)`. `triangular_solve` has no pretty form, so this emits its parser-
    /// validated *generic* form verbatim (quoted op name, `<{...}>`
    /// properties dict: `left_side`/`lower`/`unit_diagonal`/`transpose_a`).
    pub fn tri_solve(&mut self, l: &Value, b: &Value) -> Value {
        let ssa = self.fresh();
        let l_ty = l.ty.render(self.dtype);
        let b_ty = b.ty.render(self.dtype);
        let result_ty = b.ty.clone();
        let result_ty_text = result_ty.render(self.dtype);
        self.push(&format!(
            "{ssa} = \"stablehlo.triangular_solve\"({}, {}) <{{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}}> : ({l_ty}, {b_ty}) -> {result_ty_text}",
            l.ssa, b.ssa
        ));
        Value { ssa, ty: result_ty }
    }

    // ---- sampling (Task 6) --------------------------------------------------

    /// `%sh = stablehlo.constant dense<...> : tensor<Kxi64>` then `%N =
    /// stablehlo.rng %a, %b, %sh, distribution = {NORMAL|UNIFORM} : (a_ty,
    /// b_ty, tensor<Kxi64>) -> out_ty` — parser-validated verbatim against
    /// the real StableHLO parser (jax 0.10.2). `dist` is a StableHLO
    /// `rng_distribution` (`"NORMAL"`/`"UNIFORM"`); no explicit RNG key is
    /// threaded (this vertical is XLA-seeded — see
    /// `registry::lower_sample`'s doc comment on why `builtin_sample`'s
    /// threaded rng-state argument is simply never lowered).
    ///
    /// `K` is `out_ty`'s rank and the shape constant's `K` elements are its
    /// own per-axis dimension sizes: `K = 0` (`tensor<0xi64>`, `dense<>`) for
    /// a `Scalar` result, `K = 1` (`tensor<1xi64>`, `dense<N>` — a bare
    /// scalar literal, not `dense<[N]>`) for a length-`N` vector result.
    /// `stablehlo.rng`'s shape operand is always an INTEGER tensor (a static
    /// output-shape descriptor), never this emitter's `f32`/`f64` element
    /// dtype — unlike every other constant this emitter builds, it cannot go
    /// through [`Emitter::constant`]/`MlirTy::render` (both
    /// dtype-parameterized), so it is built as raw text here instead
    /// (mirroring [`render_i1`]'s reasoning for the same kind of
    /// dtype-independent local render).
    ///
    /// Panics (an internal invariant violation, not a user-facing refusal —
    /// mirrors `diag`/`matvec`'s panic-on-bad-shape discipline) if `a`/`b`
    /// is not rank-0 (`stablehlo.rng`'s bounds operands must be scalar,
    /// regardless of `out_ty`'s shape — the shape constant, not `a`/`b`,
    /// carries the output rank), or if `out_ty` has a dynamic dimension or
    /// is a `Tuple`: neither has a static shape-constant form.
    pub fn rng(&mut self, dist: &str, a: &Value, b: &Value, out_ty: &MlirTy) -> Value {
        match &a.ty {
            MlirTy::Scalar => {}
            other => panic!("rng expects a rank-0 (scalar) `a` operand, got {other:?}"),
        }
        match &b.ty {
            MlirTy::Scalar => {}
            other => panic!("rng expects a rank-0 (scalar) `b` operand, got {other:?}"),
        }
        let dims: Vec<u64> = match out_ty {
            MlirTy::Scalar => Vec::new(),
            MlirTy::Ranked(dims) => dims
                .iter()
                .map(|d| {
                    d.unwrap_or_else(|| {
                        panic!("rng: dynamic output dimension has no static shape-constant form")
                    })
                })
                .collect(),
            MlirTy::Tuple(_) => panic!("rng: tuple output type has no shape-constant form"),
        };
        let shape_ty_text = format!("tensor<{}xi64>", dims.len());
        let shape_lit = match dims.len() {
            0 => String::new(),
            1 => dims[0].to_string(),
            _ => format!(
                "[{}]",
                dims.iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let shape_ssa = self.fresh();
        self.push(&format!(
            "{shape_ssa} = stablehlo.constant dense<{shape_lit}> : {shape_ty_text}"
        ));

        let a_ty = a.ty.render(self.dtype);
        let b_ty = b.ty.render(self.dtype);
        let out_ty_text = out_ty.render(self.dtype);
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = stablehlo.rng {}, {}, {shape_ssa}, distribution = {dist} : ({a_ty}, {b_ty}, {shape_ty_text}) -> {out_ty_text}",
            a.ssa, b.ssa
        ));
        Value {
            ssa,
            ty: out_ty.clone(),
        }
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
            Node::Lit(Scalar::Int(i)) => Ok(self.constant(*i as f64, MlirTy::Scalar)),
            Node::Lit(Scalar::Real(x)) => Ok(self.constant(*x, MlirTy::Scalar)),
            Node::Lit(Scalar::Bool(b)) => {
                Ok(self.constant(if *b { 1.0 } else { 0.0 }, MlirTy::Scalar))
            }
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
                    } else if matches!(name.as_str(), "get0" | "get") {
                        // `get0(builtin_sample(...), 0)` / `get((%ref self
                        // <shared-latent>), 1)`: a projection of a sampled
                        // `(value, new_rngstate)` tuple, not a real rank-1
                        // tensor — see `Emitter::sample_tuple_slot`'s doc
                        // comment. Anything else (the ordinary case) falls
                        // through to `ops::lower_builtin`'s generic
                        // rank-1-tensor `get`/`get0`.
                        let base = if name == "get0" { 0 } else { 1 };
                        match self.sample_tuple_slot(&call.args, base) {
                            Some(0) => self.lower_node(call.args[0]),
                            Some(_) => Err(EmitError::at(
                                id,
                                "sampled rng state has no tensor form (this vertical is \
                                 XLA-seeded: stablehlo.rng takes no explicit rng key, so the \
                                 threaded rng-state slot of a sampled tuple is never lowered)",
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
    /// {ret.ty} { <body> return {ret.ssa} : {ret.ty} } }`, 2-space indented
    /// per nesting level (mirroring `flatppl_flatpir::writer`'s
    /// canonical-text formatting style).
    pub fn finish(self, func_name: &str, args: &[(String, MlirTy)], ret: &Value) -> String {
        let dtype = self.dtype;
        let arg_list = args
            .iter()
            .map(|(name, ty)| format!("{name}: {}", ty.render(dtype)))
            .collect::<Vec<_>>()
            .join(", ");
        let ret_ty_text = ret.ty.render(dtype);

        let mut out = String::from("module {\n");
        out.push_str(&format!(
            "  func.func @{func_name}({arg_list}) -> {ret_ty_text} {{\n"
        ));
        for line in self.body.lines() {
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!("    return {} : {ret_ty_text}\n", ret.ssa));
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
    }
}
