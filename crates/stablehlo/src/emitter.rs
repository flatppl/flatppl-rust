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

use flatppl_core::{Module, NodeId};

use crate::Dtype;
use crate::mlir::{MlirTy, Value};

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

/// Emits textual StableHLO into an internal buffer while assigning fresh SSA
/// names and tracking which FlatPDL [`NodeId`]s have already been lowered.
pub struct Emitter<'m> {
    /// The FlatPDL module being lowered. Unused by the op helpers in this
    /// file (they operate on already-typed [`Value`]s, not on `Module`
    /// nodes); read by Task 4's `lower_node` dispatch to resolve node
    /// structure/types via `self.m`.
    #[allow(dead_code)]
    m: &'m Module,
    dtype: Dtype,
    next: u32,
    /// Memoizes `NodeId -> Value` so a shared sub-expression is lowered (and
    /// its op line emitted) once. Populated and consulted by Task 4's
    /// `lower_node`, not by this file's op helpers.
    #[allow(dead_code)]
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
    /// `triangular_solve` has no pretty form, so this emits its parser-
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
