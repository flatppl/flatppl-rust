//! §07 "Scalar restrictions and constructors" (`integer`, `boolean`) and
//! "Approximation functions" (`polynomial`, `bernstein`, `stepwise`).
//!
//! `integer` and `boolean` lower only where the value is provably in their
//! domain. §07 makes any other value an error, and an integer or boolean
//! tensor has no NaN to carry one.

use flatppl_core::{CallHead, Node, NodeId, Scalar};

use crate::emitter::Emitter;
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::ops::{args_exact, get_literal_value};
use crate::refuse::EmitError;

/// `integer(x)`: an integer or boolean `x`, a written integral number, or
/// `round`, `floor` or `ceil` of a real, which is integral by construction.
/// Any other real is refused.
pub(crate) fn lower_integer(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    let integral = match e.node(e.resolve_ref_one(a)) {
        Node::Lit(Scalar::Real(v)) => v.fract() == 0.0,
        Node::Call(c) => match c.head {
            CallHead::Builtin(head) => matches!(e.resolve(head), "round" | "floor" | "ceil"),
            CallHead::User(_) => false,
        },
        _ => false,
    };
    let x = e.lower_node(a)?;
    if x.elem != ElemKind::Real || integral {
        return Ok(e.convert(&x, ElemKind::Int));
    }
    Err(EmitError::at(
        id,
        "`integer(x)` of a computed real has no lowering here: §07 makes a \
         non-integral value an error, and an integer tensor cannot carry one. Round \
         explicitly instead, as in `integer(round(x))`",
    ))
}

/// `boolean(x)`: a boolean passes through, and a written 0 or 1 is a constant.
/// A computed number is refused: a value outside {0, 1} is an error §07 states,
/// and a boolean tensor has no NaN to carry it.
pub(crate) fn lower_boolean(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    if let Some(v) = written_number(e, a)
        && (v == 0.0 || v == 1.0)
    {
        let x = e.scalar(v);
        return Ok(e.convert(&x, ElemKind::Bool));
    }
    let x = e.lower_node(a)?;
    if x.elem == ElemKind::Bool {
        return Ok(x);
    }
    Err(EmitError::at(
        id,
        "`boolean(x)` of a computed number has no lowering here: §07 makes a value \
         outside {0, 1} an error, and a boolean tensor cannot carry one. Compare \
         explicitly instead, as in `x == 1`",
    ))
}

/// A numeric literal, possibly negated, as written.
fn written_number(e: &Emitter, id: NodeId) -> Option<f64> {
    match e.node(id) {
        Node::Lit(Scalar::Int(n)) => Some(*n as f64),
        Node::Lit(Scalar::Real(x)) => Some(*x),
        Node::Lit(Scalar::Bool(b)) => Some(f64::from(*b)),
        _ => None,
    }
}

/// The real scalar cells of a static-length vector argument.
fn vector_cells(
    e: &mut Emitter,
    id: NodeId,
    arg: NodeId,
    what: &str,
) -> Result<Vec<Value>, EmitError> {
    let v = e.lower_node(arg)?;
    let v = e.with_typed_axes(arg, v);
    let v = e.convert(&v, ElemKind::Real);
    let len = match e.cell_ty(&v) {
        MlirTy::Ranked(dims) if dims.len() == 1 => dims[0],
        _ => None,
    };
    let Some(len) = len else {
        return Err(EmitError::at(
            id,
            format!("{what} must be a vector of static length"),
        ));
    };
    (0..len)
        .map(|i| get_literal_value(e, id, v.clone(), i))
        .collect()
}

/// The real scalar evaluation point.
fn scalar_point(e: &mut Emitter, id: NodeId, arg: NodeId, head: &str) -> Result<Value, EmitError> {
    let x = e.lower_node(arg)?;
    if e.cell_ty(&x) != MlirTy::Scalar {
        return Err(EmitError::at(
            id,
            format!("`{head}` takes a scalar `x`; apply it elementwise with `{head}.(…)`"),
        ));
    }
    Ok(e.convert(&x, ElemKind::Real))
}

/// `polynomial(c, x) = Σ c_{i+1} x^i`, by Horner's rule. A complex `x` would
/// make a complex value, refused like every other complex result.
pub(crate) fn lower_polynomial(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [c, x] = args_exact(id, args)?;
    if matches!(
        e.type_of(x),
        Some(flatppl_core::Type::Scalar(
            flatppl_core::ScalarType::Complex
        ))
    ) {
        return Err(EmitError::at(
            id,
            "`polynomial` at a complex `x` makes a complex value, which has no tensor form here",
        ));
    }
    let cells = vector_cells(e, id, c, "polynomial coefficients")?;
    let x = scalar_point(e, id, x, "polynomial")?;
    let Some((last, rest)) = cells.split_last() else {
        return Ok(e.scalar(0.0));
    };
    let mut acc = last.clone();
    for c in rest.iter().rev() {
        let scaled = e.mul(&acc, &x);
        acc = e.add(&scaled, c);
    }
    Ok(acc)
}

/// `bernstein(c, x) = Σ_k c_{k+1} C(n, k) x^k (1 − x)^{n−k}`, by de Casteljau's
/// recurrence, which needs no binomial coefficient and stays stable on [0, 1].
/// Outside [0, 1] it continues the same polynomial, as flatppl-js does.
pub(crate) fn lower_bernstein(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [c, x] = args_exact(id, args)?;
    let mut b = vector_cells(e, id, c, "bernstein coefficients")?;
    let x = scalar_point(e, id, x, "bernstein")?;
    if b.is_empty() {
        return Ok(e.scalar(0.0));
    }
    let one = e.scalar(1.0);
    let omx = e.sub(&one, &x);
    while b.len() > 1 {
        let mut next = Vec::with_capacity(b.len() - 1);
        for pair in b.windows(2) {
            let l = e.mul(&omx, &pair[0]);
            let r = e.mul(&x, &pair[1]);
            next.push(e.add(&l, &r));
        }
        b = next;
    }
    Ok(b.remove(0))
}

/// `stepwise(edges, values, x)`: `values[i]` on `[edges[i], edges[i+1])`, the last
/// bin closed on the right; NaN outside `[edges[1], edges[n+1]]`, as flatppl-js
/// gives.
pub(crate) fn lower_stepwise(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [edges, values, x] = args_exact(id, args)?;
    let edges = vector_cells(e, id, edges, "stepwise edges")?;
    let values = vector_cells(e, id, values, "stepwise values")?;
    if edges.len() != values.len() + 1 {
        return Err(EmitError::at(
            id,
            "stepwise: `edges` must be one longer than `values` (spec §07)",
        ));
    }
    let x = scalar_point(e, id, x, "stepwise")?;
    let n = values.len();
    let mut acc = e.constant_like(f64::NAN, &x);
    for i in (0..n).rev() {
        let lo = e.compare("GE", &x, &edges[i]);
        let hi = e.compare(if i + 1 == n { "LE" } else { "LT" }, &x, &edges[i + 1]);
        let inside = e.and(&lo, &hi);
        acc = e.select(&inside, &values[i], &acc);
    }
    Ok(acc)
}
