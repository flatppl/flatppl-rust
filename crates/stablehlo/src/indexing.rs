//! Numeric multi-axis subset selection for §07 `get` / `get0`.
//!
//! The ordinary one-selector vector paths remain in `ops`: their established
//! slice/gather IR is a compatibility surface. This module owns the stricter
//! tensor contract: every cell axis has a selector, literal and singleton axes
//! collapse, `all` survives, and at most one shared integer-vector selector
//! replaces its axis. Callable batch prefixes are carried but never selected.

use flatppl_core::{Node, NodeId, Scalar};

use crate::emitter::Emitter;
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::refuse::EmitError;

enum TensorSelector {
    Scalar(u64),
    All,
    Vector(Value),
}

pub(crate) fn lower_multi_axis(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    base: i64,
) -> Result<Value, EmitError> {
    let Some((&container, selector_nodes)) = args.split_first() else {
        return Err(EmitError::at(
            id,
            "get/get0: expected a container and selectors",
        ));
    };
    if selector_nodes.is_empty() {
        return Err(EmitError::at(
            id,
            "get/get0: expected at least one selector",
        ));
    }

    let operand = e.lower_node(container)?;
    let operand = e.with_typed_axes(container, operand);
    let axes = e.axes_of(&operand);
    let cell_dims = match e.cell_ty(&operand) {
        MlirTy::Ranked(dims) if axes.layers.len() == 1 => dims,
        other => {
            return Err(EmitError::at(
                id,
                format!(
                    "get/get0: multi-axis selection requires one flat numeric tensor cell, got {other:?}"
                ),
            ));
        }
    };
    if !matches!(operand.elem, ElemKind::Real | ElemKind::Int) {
        return Err(EmitError::at(
            id,
            "get/get0: multi-axis selection requires a numeric tensor",
        ));
    }
    if selector_nodes.len() != cell_dims.len() {
        return Err(EmitError::at(
            id,
            format!(
                "get/get0: {} selector(s) for a rank-{} tensor cell; every cell axis must be selected",
                selector_nodes.len(),
                cell_dims.len()
            ),
        ));
    }

    let selectors = lower_selectors(e, id, selector_nodes, &cell_dims, base)?;
    let batch = e.batch_rank(&operand);
    let physical_dims = match &operand.ty {
        MlirTy::Ranked(dims) if dims.iter().all(Option::is_some) => dims,
        _ => {
            return Err(EmitError::at(
                id,
                "get/get0: multi-axis selection needs a static tensor shape",
            ));
        }
    };
    let mut starts = vec![0; physical_dims.len()];
    let mut limits = physical_dims
        .iter()
        .map(|d| d.expect("static shape checked"))
        .collect::<Vec<_>>();
    let mut collapses = false;
    for (axis, selector) in selectors.iter().enumerate() {
        if let TensorSelector::Scalar(index) = selector {
            starts[batch + axis] = *index;
            limits[batch + axis] = index + 1;
            collapses = true;
        }
    }
    let sliced = if collapses {
        e.slice(&operand, &starts, &limits, &vec![1; physical_dims.len()])
    } else {
        operand
    };

    let mut kept_dims = limits[..batch]
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    kept_dims.extend(
        selectors
            .iter()
            .zip(&cell_dims)
            .filter_map(|(selector, &dim)| {
                matches!(selector, TensorSelector::All | TensorSelector::Vector(_)).then_some(dim)
            }),
    );
    let kept_ty = if kept_dims.is_empty() {
        MlirTy::Scalar
    } else {
        MlirTy::Ranked(kept_dims)
    };
    let selected = if sliced.ty == kept_ty {
        sliced
    } else {
        e.reshape(&sliced, kept_ty)
    };

    let Some((cell_axis, index)) = selectors
        .iter()
        .filter(|selector| !matches!(selector, TensorSelector::Scalar(_)))
        .enumerate()
        .find_map(|(axis, selector)| match selector {
            TensorSelector::Vector(index) => Some((axis, index)),
            TensorSelector::All => None,
            TensorSelector::Scalar(_) => unreachable!(),
        })
    else {
        return Ok(selected);
    };
    Ok(e.gather_axis(&selected, index, base, cell_axis))
}

fn lower_selectors(
    e: &mut Emitter,
    id: NodeId,
    nodes: &[NodeId],
    cell_dims: &[Option<u64>],
    base: i64,
) -> Result<Vec<TensorSelector>, EmitError> {
    let mut selectors = Vec::with_capacity(nodes.len());
    let mut vector_count = 0usize;
    for (&node, &extent) in nodes.iter().zip(cell_dims) {
        let selector = match e.node(node) {
            Node::Lit(Scalar::Int(i)) => {
                let zero = *i - base;
                if zero < 0 || extent.is_some_and(|n| zero as u64 >= n) {
                    return Err(EmitError::at(id, "get/get0: index out of range"));
                }
                TensorSelector::Scalar(zero as u64)
            }
            Node::Const(name) if e.resolve(*name) == "all" => TensorSelector::All,
            Node::Const(name) if e.resolve(*name) == "only" => {
                if extent != Some(1) {
                    return Err(EmitError::at(
                        id,
                        "get/get0: `only` requires an axis of length one",
                    ));
                }
                TensorSelector::Scalar(0)
            }
            _ => {
                let index = e.lower_node(node)?;
                if index.elem != ElemKind::Int
                    || e.batch_rank(&index) != 0
                    || !matches!(e.cell_ty(&index), MlirTy::Ranked(dims) if dims.len() == 1)
                    || e.axes_of(&index).layers.len() != 1
                {
                    return Err(EmitError::at(
                        id,
                        "get/get0: a non-literal multi-axis selector must be one shared rank-1 integer vector",
                    ));
                }
                vector_count += 1;
                TensorSelector::Vector(index)
            }
        };
        selectors.push(selector);
    }
    if vector_count > 1 {
        return Err(EmitError::at(
            id,
            "get/get0: multiple integer-vector selectors have no implemented tensor lowering",
        ));
    }
    Ok(selectors)
}
