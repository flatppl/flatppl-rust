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
    DynamicScalar(Value),
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
                (!matches!(selector, TensorSelector::Scalar(_))).then_some(dim)
            }),
    );
    let kept_ty = if kept_dims.is_empty() {
        MlirTy::Scalar
    } else {
        MlirTy::Ranked(kept_dims)
    };
    let mut selected = if sliced.ty == kept_ty {
        sliced
    } else {
        e.reshape(&sliced, kept_ty)
    };

    let mut cell_axis = 0;
    for selector in &selectors {
        match selector {
            TensorSelector::Scalar(_) => {}
            TensorSelector::All => cell_axis += 1,
            TensorSelector::Vector(index) => {
                selected = e.gather_axis(&selected, index, base, cell_axis);
                cell_axis += 1;
            }
            TensorSelector::DynamicScalar(index) => {
                let batch = e.batch_rank(&selected);
                let rank = match e.cell_ty(&selected) {
                    MlirTy::Ranked(dims) => dims.len(),
                    _ => unreachable!(),
                };
                let mut permutation: Vec<_> = (0..batch as u64).collect();
                permutation.push((batch + cell_axis) as u64);
                permutation.extend(
                    (batch..batch + rank)
                        .filter(|&i| i != batch + cell_axis)
                        .map(|i| i as u64),
                );
                let moved = e.transpose(&selected, &permutation);
                let mut axes = e.axes_of(&selected);
                axes.layers = if rank == 1 {
                    vec![1]
                } else {
                    vec![1, rank - 1]
                };
                let moved = e.axes_view(&moved, axes);
                selected = e.gather_cells(&moved, index, base);
            }
        }
    }
    Ok(e.with_typed_axes(id, selected))
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
                let Some(zero) = i.checked_sub(base).filter(|zero| *zero >= 0) else {
                    return Err(EmitError::at(id, "get/get0: index out of range"));
                };
                if extent.is_some_and(|n| zero as u64 >= n) {
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
                if index.elem == ElemKind::Int && e.cell_ty(&index) == MlirTy::Scalar {
                    let size = extent
                        .ok_or_else(|| EmitError::at(id, "get/get0 requires a static axis"))?;
                    let index = e.checked_index(id, &index, size, base)?;
                    selectors.push(TensorSelector::DynamicScalar(index));
                    continue;
                }
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
                let size =
                    extent.ok_or_else(|| EmitError::at(id, "get/get0 requires a static axis"))?;
                let index = e.checked_index(id, &index, size, base)?;
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
