//! Sum aggregates whose body is the product of two fully indexed real arrays.

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{Axis, Module, Node, NodeId, Symbol};

use crate::{Emitter, Step, real, tensor};

pub(super) struct SumProduct {
    sum: NodeId,
    output_axes: Vec<NodeId>,
    operands: [Indexed; 2],
}

struct Indexed {
    container: NodeId,
    axes: Vec<NodeId>,
}

impl SumProduct {
    pub(super) fn read(module: &Module, args: &[NodeId]) -> Option<Self> {
        let [sum, axes, body] = *args else {
            return None;
        };
        let Node::Const(name) = module.node(sum) else {
            return None;
        };
        if module.resolve(*name) != "sum" {
            return None;
        }
        let output_axes = positional(module, axes, "vector")?;
        checked_axes(module, output_axes)?;
        let [left, right] = *positional(module, body, "mul")? else {
            return None;
        };
        let operands = [Indexed::read(module, left)?, Indexed::read(module, right)?];
        Some(Self {
            sum,
            output_axes: output_axes.to_vec(),
            operands,
        })
    }

    pub(super) fn args(&self) -> [NodeId; 2] {
        std::array::from_fn(|i| self.operands[i].container)
    }

    pub(super) fn contribution(
        &self,
        emitter: &mut Emitter,
        seed: NodeId,
        slot: usize,
        forward: bool,
        masked: bool,
    ) -> Option<NodeId> {
        let active = self.operands.get(slot)?;
        let other = &self.operands[1 - slot];
        let other_array = emitter.primal(other.container);
        let other_cell = indexed(&mut emitter.module, other_array, &other.axes);
        if forward {
            let module = &mut emitter.module;
            let seed_cell = indexed(module, seed, &active.axes);
            let term = build_call(module, "mul", &[seed_cell, other_cell]);
            return Some(aggregate(module, self.sum, &self.output_axes, term));
        }
        let module = &mut emitter.module;
        let seed_cell = indexed(module, seed, &self.output_axes);
        let mut term = cotangent_product(module, seed_cell, other_cell, masked);
        if !active.axes.iter().all(|&target| {
            let name = axis_name(module, target);
            self.output_axes
                .iter()
                .chain(&other.axes)
                .any(|&axis| axis_name(module, axis) == name)
        }) {
            // A summed axis appearing only in the active array has a constant
            // cotangent. Attach its extent without reading the primal cells.
            let one = real(module, 1.0);
            let ones = tensor::fill_like(emitter, one, active.container)?;
            let module = &mut emitter.module;
            let one_cell = indexed(module, ones, &active.axes);
            term = build_call(module, "mul", &[term, one_cell]);
        }
        Some(aggregate(&mut emitter.module, self.sum, &active.axes, term))
    }
}

impl Indexed {
    fn read(module: &Module, node: NodeId) -> Option<Self> {
        let args = positional(module, node, "get").or_else(|| positional(module, node, "get0"))?;
        let (&container, axes) = args.split_first()?;
        if axes.is_empty() || tensor::real_shape(module, container)?.len() != axes.len() {
            return None;
        }
        checked_axes(module, axes)?;
        Some(Self {
            container,
            axes: axes.to_vec(),
        })
    }
}

fn positional<'a>(module: &'a Module, node: NodeId, op: &str) -> Option<&'a [NodeId]> {
    let call = expect_builtin_call(module, node, op)?;
    call.named.is_empty().then_some(&call.args)
}

pub(super) fn axis_name(module: &Module, node: NodeId) -> Option<Symbol> {
    match module.node(node) {
        Node::Axis(axis) if axis.variance.is_none() => Some(axis.name),
        _ => None,
    }
}

pub(super) fn checked_axes(module: &Module, axes: &[NodeId]) -> Option<()> {
    let mut seen = std::collections::HashSet::new();
    for &axis in axes {
        if !seen.insert(axis_name(module, axis)?) {
            return None;
        }
    }
    Some(())
}

pub(super) fn indexed(module: &mut Module, array: NodeId, axes: &[NodeId]) -> NodeId {
    if axes.is_empty() {
        return array;
    }
    let args: Vec<_> = std::iter::once(array).chain(axes.iter().copied()).collect();
    build_call(module, "get", &args)
}

pub(super) fn aggregate(
    module: &mut Module,
    sum: NodeId,
    output_axes: &[NodeId],
    body: NodeId,
) -> NodeId {
    let axes = build_call(module, "vector", output_axes);
    build_call(module, "aggregate", &[sum, axes, body])
}

pub(super) fn matrix_pullback(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
) -> Option<NodeId> {
    let [_, right] = *step.args.as_slice() else {
        return None;
    };
    let rank = tensor::numeric_shape(&emitter.module, right)?.len();
    let other = emitter.primal(*step.args.get(1usize.checked_sub(slot)?)?);
    let m = &mut emitter.module;
    let row = axis(m, "adrow");
    let inner = axis(m, "adinner");
    let col = axis(m, "adcol");
    let (right_axes, out_axes) = match rank {
        1 => (vec![inner], vec![row]),
        2 => (vec![inner, col], vec![row, col]),
        _ => return None,
    };
    let left_axes = [row, inner];
    let seed_cell = indexed(m, bar, &out_axes);
    let (target_axes, other_axes): (&[NodeId], &[NodeId]) = if slot == 0 {
        (&left_axes, &right_axes)
    } else {
        (&right_axes, &left_axes)
    };
    let other_cell = indexed(m, other, other_axes);
    let term = cotangent_product(m, seed_cell, other_cell, true);
    let sum = m.intern("sum");
    let sum = m.alloc(Node::Const(sum));
    Some(aggregate(m, sum, target_axes, term))
}

fn axis(module: &mut Module, name: &str) -> NodeId {
    let name = module.intern(name);
    module.alloc(Node::Axis(Axis {
        name,
        variance: None,
    }))
}

pub(super) fn cotangent_product(
    module: &mut Module,
    seed: NodeId,
    other: NodeId,
    masked: bool,
) -> NodeId {
    let term = build_call(module, "mul", &[seed, other]);
    if masked {
        tensor::zero_when_masked(module, seed, term, false)
    } else {
        term
    }
}
