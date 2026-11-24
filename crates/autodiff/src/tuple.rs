//! Positional projections and cotangents for promoted tuples.

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{Module, Node, NodeId, Scalar, Type};

use crate::{Emitter, record, tensor};

pub(super) fn access(module: &Module, args: &[NodeId], base: i64) -> bool {
    let [tuple, selector] = *args else {
        return false;
    };
    matches!(module.type_of(tuple), Some(ty @ Type::Tuple(items))
        if record::is_differentiable(ty)
            && index(module, selector, base).is_some_and(|i| i < items.len()))
}

fn index(module: &Module, selector: NodeId, base: i64) -> Option<usize> {
    let Node::Lit(Scalar::Int(index)) = module.node(selector) else {
        return None;
    };
    usize::try_from(index.checked_sub(base)?).ok()
}

pub(super) fn project(module: &mut Module, tuple: NodeId, index: usize) -> NodeId {
    if let Some(constructor) = expect_builtin_call(module, tuple, "tuple")
        && let Some(&item) = constructor.args.get(index)
    {
        return item;
    }
    let selector = tensor::int(module, index as i64 + 1);
    let get = build_call(module, "get", &[tuple, selector]);
    if let Some(Type::Tuple(items)) = module.type_of(tuple) {
        module.set_type(get, items[index].clone());
    }
    get
}

pub(super) fn fill(
    emitter: &mut Emitter,
    value: NodeId,
    primal: NodeId,
    items: &[Type],
    at: Option<(usize, NodeId)>,
) -> Option<NodeId> {
    let values = items
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            if let Some((selected, value)) = at
                && selected == index
            {
                return Some(value);
            }
            let child = project(&mut emitter.module, primal, index);
            tensor::fill_typed(emitter, value, child, ty)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(build_call(&mut emitter.module, "tuple", &values))
}

pub(super) fn add(emitter: &mut Emitter, items: &[Type], terms: &[NodeId]) -> NodeId {
    let values: Vec<_> = items
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            let children: Vec<_> = terms
                .iter()
                .map(|&term| project(&mut emitter.module, term, index))
                .collect();
            tensor::add_typed(emitter, ty, &children)
        })
        .collect();
    build_call(&mut emitter.module, "tuple", &values)
}

pub(super) fn scatter(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    items: &[Type],
    selectors: &[NodeId],
    base: i64,
) -> Option<NodeId> {
    let [selector] = *selectors else {
        return None;
    };
    let index = index(&emitter.module, selector, base)?;
    let zero = crate::real(&mut emitter.module, 0.0);
    fill(emitter, zero, primal, items, Some((index, bar)))
}
