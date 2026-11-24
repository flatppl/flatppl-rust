//! Transposes of linear array permutations, copies, and selections.

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{Dim, Module, Node, NodeId, Type};

use crate::{Emitter, Key, Query, Step, callbacks, contract, tensor};

pub(super) fn operands(
    module: &Module,
    query: &Query<'_>,
    op: &str,
    args: &[NodeId],
    node: NodeId,
) -> Option<(&'static str, Vec<NodeId>)> {
    match (op, args) {
        ("blockdiagmat", &[arg]) => {
            let Key::Node(container) = query.key(module, arg).ok()? else {
                return None;
            };
            let blocks = expect_builtin_call(module, container, "vector")?;
            Some(("blockdiag literal", blocks.args.to_vec()))
        }
        ("filter", &[_, data])
            if tensor::real_shape(module, node)
                .is_some_and(|shape| Some(shape) == tensor::real_shape(module, data)) =>
        {
            Some(("filter all", vec![data]))
        }
        _ => None,
    }
}

pub(super) fn valid(module: &Module, step: &Step<'_>) -> bool {
    let real = |node| tensor::real_shape(module, node);
    let nested = |node| tensor::array_of_arrays(module.type_of(node)?);
    match (step.op, step.args.as_slice()) {
        ("tile" | "bandedmat" | "filter all", &[array, ..]) => {
            real(array).is_some() && real(step.node).is_some()
        }
        ("splitblocks" | "partition", &[array, _]) => {
            real(array).is_some() && nested(step.node).is_some()
        }
        ("joinblocks" | "blockdiagmat", &[arrays]) => {
            nested(arrays).is_some() && real(step.node).is_some()
        }
        ("blockdiag literal", arrays) => {
            !arrays.is_empty()
                && arrays.iter().all(|&array| {
                    tensor::numeric_shape(module, array).is_some_and(|shape| shape.len() == 2)
                })
                && real(step.node).is_some()
        }
        ("selectbins", &[_, _, array]) => real(array).is_some() && real(step.node).is_some(),
        _ => false,
    }
}

pub(super) fn pushforward(
    emitter: &mut Emitter,
    step: &Step<'_>,
    tangent: NodeId,
) -> Option<NodeId> {
    let output = emitter.module.type_of(step.node)?;
    let sizing = match step.op {
        "tile" => tensor::real_shape_of(output)?
            .iter()
            .zip(tensor::real_shape(&emitter.module, step.args[0])?)
            .map(|(&output, &input)| match (output, input) {
                (Dim::Static(full), Dim::Static(size)) if size > 0 && full % size == 0 => {
                    Some(Dim::Static(full / size))
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?,
        "splitblocks" | "partition" => {
            let (_, cell) = tensor::array_of_arrays(output)?;
            tensor::real_shape_of(cell)?.to_vec()
        }
        "bandedmat" => vec![*tensor::real_shape_of(output)?.first()?],
        _ => return None,
    };
    // Size operands are static metadata. Keep them literal in the emitted
    // program, including when the primal and tangent share the same shape.
    let m = &mut emitter.module;
    let sizing = dimension_arg(m, &sizing)?;
    Some(build_call(m, step.op, &[tangent, sizing]))
}

pub(super) fn pullback(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
) -> Option<NodeId> {
    match (step.op, step.args.as_slice()) {
        ("splitblocks" | "partition", _) => {
            Some(build_call(&mut emitter.module, "joinblocks", &[bar]))
        }
        ("joinblocks", &[arg]) => {
            let (_, cell) = tensor::array_of_arrays(emitter.module.type_of(arg)?)?;
            let shape = tensor::real_shape_of(cell)?.to_vec();
            let size = dimension_arg(&mut emitter.module, &shape)?;
            Some(build_call(&mut emitter.module, "splitblocks", &[bar, size]))
        }
        ("tile", &[arg, _]) => {
            let shape = tensor::real_shape(&emitter.module, arg)?.to_vec();
            let m = &mut emitter.module;
            let size = dimension_arg(m, &shape)?;
            let blocks = build_call(m, "splitblocks", &[bar, size]);
            let axes: Vec<_> = (0..2 * shape.len())
                .map(|i| contract::axis(m, &format!("tile_ad_{i}")))
                .collect();
            let value = contract::indexed(m, blocks, &axes);
            let sum = constant(m, "sum");
            Some(contract::aggregate(m, sum, &axes[shape.len()..], value))
        }
        ("bandedmat", &[array, _]) => {
            let [Dim::Static(width)] = *tensor::real_shape(&emitter.module, array)? else {
                return None;
            };
            let [Dim::Static(rows), _] = *tensor::real_shape(&emitter.module, step.node)? else {
                return None;
            };
            Some(banded(&mut emitter.module, bar, rows, width))
        }
        ("blockdiag literal", args) => {
            let mut row = 0_i64;
            let mut col = 0_i64;
            for &arg in &args[..slot] {
                let [Dim::Static(r), Dim::Static(c)] =
                    *tensor::numeric_shape(&emitter.module, arg)?
                else {
                    return None;
                };
                row += i64::from(r);
                col += i64::from(c);
            }
            let [Dim::Static(rows), Dim::Static(cols)] =
                *tensor::numeric_shape(&emitter.module, args[slot])?
            else {
                return None;
            };
            let m = &mut emitter.module;
            let row = tensor::int(m, row);
            let col = tensor::int(m, col);
            Some(block(m, bar, row, col, rows, cols))
        }
        ("blockdiagmat", &[arg]) => {
            let (outer, cell) = tensor::array_of_arrays(emitter.module.type_of(arg)?)?;
            let ([Dim::Static(count)], [Dim::Static(rows), Dim::Static(cols)]) =
                (outer, tensor::real_shape_of(cell)?)
            else {
                return None;
            };
            let (count, rows, cols) = (*count, *rows, *cols);
            Some(blocks(&mut emitter.module, bar, count, rows, cols))
        }
        ("selectbins", &[edges, region, counts]) => {
            let [out] = *tensor::real_shape(&emitter.module, step.node)? else {
                return None;
            };
            let m = &mut emitter.module;
            let indices = build_call(m, "indicesof", &[counts]);
            let indices = build_call(m, "selectbins", &[edges, region, indices]);
            m.set_type(
                indices,
                Type::Array {
                    shape: Box::new([out]),
                    elem: Box::new(Type::Scalar(flatppl_core::ScalarType::Integer)),
                },
            );
            tensor::scatter(emitter, bar, counts, &[indices], 1)
        }
        _ => None,
    }
}

pub(super) fn blockdiag_pushforward(
    emitter: &mut Emitter,
    step: &Step<'_>,
    terms: impl Iterator<Item = Option<NodeId>>,
) -> Result<NodeId, crate::Error> {
    let blocks = terms
        .zip(&step.args)
        .map(|(term, &primal)| match term {
            Some(value) => Ok(value),
            None => {
                let shape = tensor::numeric_shape(&emitter.module, primal)
                    .ok_or_else(|| crate::Error::new(Some(primal), "missing matrix block shape"))?
                    .to_vec();
                let m = &mut emitter.module;
                let size = dimension_arg(m, &shape).ok_or_else(|| {
                    crate::Error::new(Some(primal), "cannot construct an inactive matrix block")
                })?;
                let zero = crate::real(m, 0.0);
                Ok(build_call(m, "fill", &[zero, size]))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let m = &mut emitter.module;
    let vector = build_call(m, "vector", &blocks);
    Ok(build_call(m, "blockdiagmat", &[vector]))
}

fn constant(m: &mut Module, name: &str) -> NodeId {
    let name = m.intern(name);
    m.alloc(Node::Const(name))
}

fn dimension_arg(m: &mut Module, shape: &[Dim]) -> Option<NodeId> {
    let sizes = shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(size) if *size > 0 => Some(tensor::int(m, i64::from(*size))),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(match sizes.as_slice() {
        &[one] => one,
        _ => build_call(m, "vector", &sizes),
    })
}

fn positions(m: &mut Module, size: u32, base: i64) -> NodeId {
    let count = tensor::int(m, i64::from(size));
    let zeros = build_call(m, "zeros", &[count]);
    build_call(
        m,
        if base == 0 { "indicesof0" } else { "indicesof" },
        &[zeros],
    )
}

fn block(m: &mut Module, bar: NodeId, row: NodeId, col: NodeId, rows: u32, cols: u32) -> NodeId {
    let row_index = positions(m, rows, 1);
    let col_index = positions(m, cols, 1);
    let row_index = tensor::broadcast(m, "add", &[row_index, row]);
    let col_index = tensor::broadcast(m, "add", &[col_index, col]);
    build_call(m, "get", &[bar, row_index, col_index])
}

fn blocks(m: &mut Module, bar: NodeId, count: u32, rows: u32, cols: u32) -> NodeId {
    let mut boundary = Vec::new();
    let bar_cell = callbacks::callback_input(m, "source", &mut boundary);
    let index = callbacks::callback_input(m, "block", &mut boundary);
    let r = tensor::int(m, i64::from(rows));
    let c = tensor::int(m, i64::from(cols));
    let r = build_call(m, "mul", &[index, r]);
    let c = build_call(m, "mul", &[index, c]);
    let cell = mapped_block(m, bar_cell, r, c, rows, cols);
    let callback = callbacks::reify(m, cell, boundary);
    let bar = build_call(m, "vector", &[bar]);
    let indices = positions(m, count, 0);
    build_call(m, "broadcast", &[callback, bar, indices])
}

// Scalar indexing within two coordinate axes gives the Cartesian block;
// vector indexing with two computed selectors would be ambiguous.
fn mapped_block(
    m: &mut Module,
    bar: NodeId,
    row: NodeId,
    col: NodeId,
    rows: u32,
    cols: u32,
) -> NodeId {
    let row_positions = positions(m, rows, 1);
    let col_positions = positions(m, cols, 1);
    let row_positions = tensor::broadcast(m, "add", &[row_positions, row]);
    let col_positions = tensor::broadcast(m, "add", &[col_positions, col]);
    let zero = tensor::int(m, 0);
    let one = tensor::int(m, 1);
    let row_positions = build_call(m, "addaxes", &[row_positions, zero, one]);
    let col_positions = build_call(m, "addaxes", &[col_positions, one, zero]);
    let singleton = build_call(m, "vector", &[one, one]);
    let held = build_call(m, "fill", &[bar, singleton]);

    let mut boundary = Vec::new();
    let matrix = callbacks::callback_input(m, "source", &mut boundary);
    let r = callbacks::callback_input(m, "row", &mut boundary);
    let c = callbacks::callback_input(m, "column", &mut boundary);
    let element = build_call(m, "get", &[matrix, r, c]);
    let callback = callbacks::reify(m, element, boundary);
    build_call(
        m,
        "broadcast",
        &[callback, held, row_positions, col_positions],
    )
}

fn banded(m: &mut Module, bar: NodeId, rows: u32, width: u32) -> NodeId {
    let mut outer = Vec::new();
    let matrix = callbacks::callback_input(m, "matrix", &mut outer);
    let diagonal = callbacks::callback_input(m, "diagonal", &mut outer);
    let mut inner = Vec::new();
    let source = callbacks::callback_input(m, "source", &mut inner);
    let row = callbacks::callback_input(m, "row", &mut inner);
    let offset = callbacks::callback_input(m, "offset", &mut inner);
    let column = build_call(m, "add", &[row, offset]);
    let value = build_call(m, "get", &[source, row, column]);
    let extract = callbacks::reify(m, value, inner);
    let held = build_call(m, "vector", &[matrix]);
    let rows = positions(m, rows, 1);
    let values = build_call(m, "broadcast", &[extract, held, rows, diagonal]);
    let sum = build_call(m, "sum", &[values]);
    let reduce = callbacks::reify(m, sum, outer);
    let held = build_call(m, "vector", &[bar]);
    let offsets = positions(m, width, 0);
    build_call(m, "broadcast", &[reduce, held, offsets])
}
