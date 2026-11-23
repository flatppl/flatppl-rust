//! Shape changes used by tensor pullbacks.

use flatppl_core::graph::{build_call, fold_add};
use flatppl_core::{Axis, Dim, Module, Node, NodeId, Scalar, ScalarType, Type};

use crate::{Emitter, Error, REAL, real};

pub(super) fn real_shape(module: &Module, node: NodeId) -> Option<&[Dim]> {
    real_shape_of(module.type_of(node)?)
}

pub(super) fn real_shape_of(ty: &Type) -> Option<&[Dim]> {
    match ty {
        Type::Scalar(ScalarType::Real) => Some(&[]),
        Type::Array { shape, elem } if **elem == REAL && fixed(shape) => Some(shape),
        _ => None,
    }
}

pub(super) fn nested_vector(module: &Module, node: NodeId) -> Option<(u32, u32)> {
    nested_vector_type(module.type_of(node)?)
}

pub(super) fn nested_vector_type(ty: &Type) -> Option<(u32, u32)> {
    let Type::Array { shape: outer, elem } = ty else {
        return None;
    };
    let Type::Array { shape: inner, elem } = elem.as_ref() else {
        return None;
    };
    match (outer.as_ref(), inner.as_ref()) {
        ([Dim::Static(rows)], [Dim::Static(cols)]) if *rows > 0 && *cols > 0 && **elem == REAL => {
            Some((*rows, *cols))
        }
        _ => None,
    }
}

pub(super) fn add_terms(emitter: &mut Emitter, primal: NodeId, terms: &[NodeId]) -> NodeId {
    if terms.len() == 1 {
        return terms[0];
    }
    if let Some(ty) = emitter.module.type_of(primal)
        && (matches!(ty, Type::Record(_)) || nested_vector_type(ty).is_some())
    {
        return add_typed(emitter, &ty.clone(), terms);
    }
    fold_add(&mut emitter.module, terms)
}

pub(super) fn add_typed(emitter: &mut Emitter, ty: &Type, terms: &[NodeId]) -> NodeId {
    if terms.len() == 1 {
        return terms[0];
    }
    if let Type::Record(fields) = ty {
        crate::record::add(emitter, fields, terms)
    } else if let Some((count, _)) = nested_vector_type(ty) {
        let m = &mut emitter.module;
        let matrices: Vec<_> = terms
            .iter()
            .map(|&term| build_call(m, "rowstack", &[term]))
            .collect();
        let matrix = fold_add(m, &matrices);
        crate::callbacks::pack_rows(m, matrix, count)
    } else {
        fold_add(&mut emitter.module, terms)
    }
}

pub(super) fn numeric_shape(module: &Module, node: NodeId) -> Option<&[Dim]> {
    const INTEGER: Type = Type::Scalar(ScalarType::Integer);
    match module.type_of(node)? {
        Type::Scalar(ScalarType::Integer) => Some(&[]),
        Type::Array { shape, elem } if **elem == INTEGER && fixed(shape) => Some(shape),
        _ => real_shape(module, node),
    }
}

pub(super) fn matmul(module: &Module, args: &[NodeId]) -> bool {
    matches!(
        args,
        [a, b] if matches!(numeric_shape(module, *a), Some([_, _]))
            && matches!(numeric_shape(module, *b), Some([_] | [_, _]))
    )
}

fn fixed(shape: &[Dim]) -> bool {
    shape.iter().all(|dim| matches!(dim, Dim::Static(_)))
}

pub(super) fn boolean_shape(module: &Module, node: NodeId) -> bool {
    const BOOL: Type = Type::Scalar(ScalarType::Boolean);
    match module.type_of(node) {
        Some(ty) if *ty == BOOL => true,
        Some(Type::Array { shape, elem }) => **elem == BOOL && fixed(shape),
        _ => false,
    }
}

pub(super) fn zero_when_masked(
    module: &mut Module,
    bar: NodeId,
    contribution: NodeId,
    pointwise: bool,
) -> NodeId {
    // Select the result so an inactive lane cannot turn 0 * NaN into NaN.
    let zero = real(module, 0.0);
    let call = if pointwise { broadcast } else { build_call };
    let inactive = call(module, "iszero", &[bar]);
    call(module, "ifelse", &[inactive, zero, contribution])
}

pub(super) fn int(module: &mut Module, value: i64) -> NodeId {
    module.alloc(Node::Lit(Scalar::Int(value)))
}

fn constant(module: &mut Module, name: &str) -> NodeId {
    let name = module.intern(name);
    module.alloc(Node::Const(name))
}

pub(super) fn broadcast(module: &mut Module, op: &str, args: &[NodeId]) -> NodeId {
    let head = constant(module, op);
    let args: Vec<_> = std::iter::once(head).chain(args.iter().copied()).collect();
    build_call(module, "broadcast", &args)
}

pub(super) fn fill_like(emitter: &mut Emitter, value: NodeId, primal: NodeId) -> Option<NodeId> {
    let ty = emitter.module.type_of(primal)?.clone();
    fill_typed(emitter, value, primal, &ty)
}

pub(super) fn fill_typed(
    emitter: &mut Emitter,
    value: NodeId,
    primal: NodeId,
    ty: &Type,
) -> Option<NodeId> {
    if let Type::Record(fields) = ty {
        return crate::record::fill(emitter, value, primal, fields);
    }
    if let Some((rows, cols)) = nested_vector_type(ty) {
        let m = &mut emitter.module;
        let cols = int(m, i64::from(cols));
        let row = build_call(m, "fill", &[value, cols]);
        let rows = int(m, i64::from(rows));
        return Some(build_call(m, "fill", &[row, rows]));
    }
    let shape = real_shape_of(ty)?;
    if shape.is_empty() {
        return Some(value);
    }
    let size = shape_arg(emitter, primal, shape)?;
    Some(build_call(&mut emitter.module, "fill", &[value, size]))
}

pub(super) fn seed(emitter: &mut Emitter, name: &str, primal: NodeId) -> Result<NodeId, Error> {
    let fail = || {
        Error::new(
            Some(primal),
            "cannot declare a matching tangent or cotangent input",
        )
    };
    let ty = emitter.module.type_of(primal).ok_or_else(fail)?.clone();
    let set = seed_set(emitter, primal, &ty).ok_or_else(fail)?;
    let input = build_call(&mut emitter.module, "elementof", &[set]);
    Ok(emitter.define(name, input))
}

pub(super) fn seed_set(emitter: &mut Emitter, primal: NodeId, ty: &Type) -> Option<NodeId> {
    if let Type::Record(fields) = ty {
        return crate::record::seed_set(emitter, primal, fields);
    }
    let reals = constant(&mut emitter.module, "reals");
    if let Some((rows, cols)) = nested_vector_type(ty) {
        let m = &mut emitter.module;
        let cols = int(m, i64::from(cols));
        let row_set = build_call(m, "cartpow", &[reals, cols]);
        let rows = int(m, i64::from(rows));
        Some(build_call(m, "cartpow", &[row_set, rows]))
    } else {
        let shape = real_shape_of(ty)?;
        if shape.is_empty() {
            Some(reals)
        } else {
            let size = shape_arg(emitter, primal, shape)?;
            Some(build_call(&mut emitter.module, "cartpow", &[reals, size]))
        }
    }
}

pub(super) fn expand_like(
    emitter: &mut Emitter,
    term: NodeId,
    primal: NodeId,
) -> Result<NodeId, Error> {
    let zero = emitter.zero_like(primal)?;
    let m = &mut emitter.module;
    let yes = m.alloc(Node::Lit(Scalar::Bool(true)));
    Ok(broadcast(m, "ifelse", &[yes, term, zero]))
}

pub(super) fn insert(emitter: &mut Emitter, value: NodeId, index: i64, length: usize) -> NodeId {
    let m = &mut emitter.module;
    if length == 1 {
        return build_call(m, "vector", &[value]);
    }
    let index = int(m, index);
    let length = int(m, length as i64);
    let basis = build_call(m, "onehot", &[index, length]);
    build_call(m, "mul", &[basis, value])
}

fn cat_extent(module: &Module, input: NodeId) -> Option<(i64, bool)> {
    if let Some((count, _)) = nested_vector(module, input) {
        return Some((i64::from(count), true));
    }
    match numeric_shape(module, input)? {
        [] => Some((1, false)),
        [Dim::Static(n)] => Some((i64::from(*n), true)),
        _ => None,
    }
}

pub(super) fn cat_slice(
    emitter: &mut Emitter,
    bar: NodeId,
    args: &[NodeId],
    slot: usize,
) -> Option<NodeId> {
    let &input = args.get(slot)?;
    let offset = args[..slot].iter().try_fold(0i64, |offset, &arg| {
        offset.checked_add(cat_extent(&emitter.module, arg)?.0)
    })?;
    let (_, is_vector) = cat_extent(&emitter.module, input)?;
    let input = emitter.primal(input);
    let m = &mut emitter.module;
    let start = int(m, offset);
    let selector = if is_vector {
        let indices = build_call(m, "indicesof0", &[input]);
        if offset == 0 {
            indices
        } else {
            broadcast(m, "add", &[indices, start])
        }
    } else {
        start
    };
    Some(build_call(m, "get0", &[bar, selector]))
}

pub(super) fn cat_embed(
    emitter: &mut Emitter,
    tangent: NodeId,
    args: &[NodeId],
    slot: usize,
) -> Option<NodeId> {
    let zero = real(&mut emitter.module, 0.0);
    let parts = args
        .iter()
        .enumerate()
        .map(|(index, &primal)| {
            if index == slot {
                return Some(tangent);
            }
            if nested_vector(&emitter.module, primal).is_some() {
                return emitter.zero_like(primal).ok();
            }
            match numeric_shape(&emitter.module, primal)? {
                [] => Some(zero),
                [dim] => {
                    let dim = *dim;
                    let size = shape_arg(emitter, primal, &[dim])?;
                    Some(build_call(&mut emitter.module, "fill", &[zero, size]))
                }
                _ => None,
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some(build_call(&mut emitter.module, "cat", &parts))
}

pub(super) fn stack_pick(module: &mut Module, bar: NodeId, index: i64, columns: bool) -> NodeId {
    let index = int(module, index);
    let all = constant(module, "all");
    let selectors = if columns { [all, index] } else { [index, all] };
    indexed(module, bar, &selectors)
}

pub(super) fn unstack(emitter: &mut Emitter, bar: NodeId, count: u32, columns: bool) -> NodeId {
    let bar = emitter.primal(bar);
    let module = &mut emitter.module;
    let parts: Vec<_> = (1..=i64::from(count))
        .map(|index| stack_pick(module, bar, index, columns))
        .collect();
    build_call(module, "vector", &parts)
}

pub(super) fn insert_nested(
    module: &mut Module,
    tangent: NodeId,
    zero: NodeId,
    slot: usize,
    count: usize,
) -> NodeId {
    let mut parts = vec![zero; count];
    parts[slot] = tangent;
    build_call(module, "vector", &parts)
}

pub(super) fn stack_insert(
    module: &mut Module,
    tangent: NodeId,
    zero: NodeId,
    slot: usize,
    count: usize,
    columns: bool,
) -> NodeId {
    let vectors = insert_nested(module, tangent, zero, slot, count);
    let op = if columns { "colstack" } else { "rowstack" };
    build_call(module, op, &[vectors])
}

fn shape_arg(emitter: &mut Emitter, primal: NodeId, shape: &[Dim]) -> Option<NodeId> {
    let sizes = shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => Some(*n),
            Dim::Dynamic => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if sizes.contains(&0) {
        // A runtime-derived size may contain zero.
        let primal = emitter.primal(primal);
        return Some(build_call(&mut emitter.module, "sizeof", &[primal]));
    }
    let sizes: Vec<_> = sizes
        .into_iter()
        .map(|n| int(&mut emitter.module, i64::from(n)))
        .collect();
    if let [n] = sizes.as_slice() {
        Some(*n)
    } else {
        Some(build_call(&mut emitter.module, "vector", &sizes))
    }
}

pub(super) fn mean_pullback(emitter: &mut Emitter, bar: NodeId, primal: NodeId) -> Option<NodeId> {
    let shape = real_shape(&emitter.module, primal)?;
    let count = shape.iter().try_fold(1u64, |size, dim| match dim {
        Dim::Static(n) => size.checked_mul(u64::from(*n)),
        Dim::Dynamic => None,
    })?;
    let count = real(&mut emitter.module, count as f64);
    let value = build_call(&mut emitter.module, "divide", &[bar, count]);
    fill_like(emitter, value, primal)
}

/// Sum exactly the axes that broadcast repeated. Restore singleton dimensions
/// in their original positions, including singleton axes in the middle.
pub(super) fn sum_to_shape(
    emitter: &mut Emitter,
    bar: NodeId,
    output: NodeId,
    input: NodeId,
) -> Result<NodeId, Error> {
    let fail = || Error::new(Some(output), "pullback cannot reduce to the input shape");
    let full = real_shape(&emitter.module, output).ok_or_else(fail)?;
    let shape = real_shape(&emitter.module, input).ok_or_else(fail)?;
    if full == shape {
        return Ok(bar);
    }
    if shape.is_empty() {
        return Ok(build_call(&mut emitter.module, "sum", &[bar]));
    }
    if full.len() != shape.len() {
        return Err(fail());
    }
    let (kept, reduced): (Vec<_>, Vec<_>) =
        (0..shape.len()).partition(|&axis| shape[axis] == full[axis]);
    if reduced.iter().any(|&axis| shape[axis] != Dim::Static(1)) {
        return Err(fail());
    }
    let rank = shape.len();
    if kept.is_empty() {
        let scalar = build_call(&mut emitter.module, "sum", &[bar]);
        return fill_like(emitter, scalar, input).ok_or_else(fail);
    }
    let m = &mut emitter.module;
    let axes = axes(m, rank);
    let sliced = indexed(m, bar, &axes);
    let kept_axes: Vec<_> = kept.iter().map(|&axis| axes[axis]).collect();
    let sum = aggregate(m, &kept_axes, sliced);

    let zero = int(m, 0);
    let trailing = int(m, reduced.len() as i64);
    let expanded = build_call(m, "addaxes", &[sum, zero, trailing]);
    let order: Vec<_> = kept.into_iter().chain(reduced).collect();
    if order.iter().copied().eq(0..rank) {
        return Ok(expanded);
    }
    let reordered: Vec<_> = order.iter().map(|&axis| axes[axis]).collect();
    let cells = indexed(m, expanded, &reordered);
    Ok(aggregate(m, &axes, cells))
}

pub(super) fn drop_added_axes(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    leading: NodeId,
    output: NodeId,
) -> Option<NodeId> {
    let rank = real_shape(&emitter.module, primal)?.len();
    let output_rank = real_shape(&emitter.module, output)?.len();
    let leading = flatppl_infer::fixed_int_value(&mut emitter.module, leading)?;
    let start = usize::try_from(leading).ok()?;
    let end = start.checked_add(rank)?;
    if rank == 0 || end > output_rank {
        return None;
    }
    let m = &mut emitter.module;
    let one = int(m, 1);
    let all = constant(m, "all");
    let mut selectors = vec![one; output_rank];
    selectors[start..end].fill(all);
    Some(indexed(m, bar, &selectors))
}

pub(super) fn index_axes(module: &Module, selector: NodeId) -> Option<usize> {
    if let Node::Const(name) = module.node(selector) {
        return match module.resolve(*name) {
            "all" => Some(1),
            "only" => Some(0),
            _ => None,
        };
    }
    const INTEGER: Type = Type::Scalar(ScalarType::Integer);
    match module.type_of(selector)? {
        Type::Scalar(ScalarType::Integer) => Some(0),
        Type::Array { shape, elem } if **elem == INTEGER && fixed(shape) => Some(shape.len()),
        _ => None,
    }
}

pub(super) fn scatter(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    selectors: &[NodeId],
    base: i64,
) -> Option<NodeId> {
    if let Some(Type::Record(fields)) = emitter.module.type_of(primal).cloned() {
        return crate::record::scatter(emitter, bar, primal, &fields, selectors);
    }
    if let Some((count, _)) = nested_vector(&emitter.module, primal) {
        let [selector] = *selectors else {
            return None;
        };
        let is_keyword = |keyword| {
            matches!(
                emitter.module.node(selector), Node::Const(name) if emitter.module.resolve(*name) == keyword
            )
        };
        if is_keyword("all") {
            return Some(bar);
        }
        if is_keyword("only") && count != 1 {
            return None;
        }
        let rank = index_axes(&emitter.module, selector)?;
        let primal = emitter.primal(primal);
        let matrix = build_call(&mut emitter.module, "rowstack", &[primal]);
        let bar = match rank {
            0 => bar,
            1 => build_call(&mut emitter.module, "rowstack", &[bar]),
            _ => return None,
        };
        let matrix = scatter_axes(emitter, bar, matrix, selectors, base, 2)?;
        return Some(unstack(emitter, matrix, count, false));
    }
    let rank = real_shape(&emitter.module, primal)?.len();
    scatter_axes(emitter, bar, primal, selectors, base, rank)
}

fn scatter_axes(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    selectors: &[NodeId],
    base: i64,
    rank: usize,
) -> Option<NodeId> {
    if rank == 0 || selectors.len() > rank {
        return None;
    }
    let gather_rank =
        selectors
            .iter()
            .try_fold(0, |rank, &selector| match emitter.module.node(selector) {
                Node::Const(_) => Some(rank),
                _ => Some(rank + index_axes(&emitter.module, selector)?),
            })?;
    let selector_refs: Vec<_> = selectors.iter().map(|&id| emitter.primal(id)).collect();
    let primal = emitter.primal(primal);
    let m = &mut emitter.module;
    let axis_nodes = axes(m, rank + gather_rank);
    let (input_axes, mut gather_axes) = axis_nodes.split_at(rank);
    let mut bar_axes = Vec::new();
    let mut mask = None;
    let indices = build_call(
        m,
        if base == 0 { "indicesof0" } else { "indicesof" },
        &[primal],
    );
    for (axis, &input_axis) in input_axes.iter().enumerate() {
        let Some(&selector) = selector_refs.get(axis) else {
            bar_axes.push(input_axis);
            continue;
        };
        if matches!(m.node(selector), Node::Const(name) if m.resolve(*name) == "all") {
            bar_axes.push(input_axis);
            continue;
        }
        let positions = if rank == 1 {
            indices
        } else {
            let axis = int(m, axis as i64 + 1);
            indexed(m, indices, &[axis])
        };
        let position = indexed(m, positions, &[input_axis]);
        let selected = if matches!(m.node(selector), Node::Const(name) if m.resolve(*name) == "only")
        {
            int(m, base)
        } else {
            let dims = index_axes(m, selectors[axis])?;
            let (selected_axes, rest) = gather_axes.split_at(dims);
            gather_axes = rest;
            bar_axes.extend_from_slice(selected_axes);
            if selected_axes.is_empty() {
                selector
            } else {
                indexed(m, selector, selected_axes)
            }
        };
        let equal = build_call(m, "equal", &[position, selected]);
        mask = Some(match mask {
            Some(previous) => build_call(m, "land", &[previous, equal]),
            None => equal,
        });
    }
    let Some(mask) = mask else {
        return Some(bar);
    };
    let cell = if bar_axes.is_empty() {
        bar
    } else {
        indexed(m, bar, &bar_axes)
    };
    let zero = real(m, 0.0);
    // Sum over all gathered uses so repeated indices accumulate.
    let contribution = build_call(m, "ifelse", &[mask, cell, zero]);
    Some(aggregate(m, input_axes, contribution))
}

fn axes(module: &mut Module, rank: usize) -> Vec<NodeId> {
    (0..rank)
        .map(|axis| {
            let name = module.intern(&format!("ad{axis}"));
            module.alloc(Node::Axis(Axis {
                name,
                variance: None,
            }))
        })
        .collect()
}

fn indexed(module: &mut Module, value: NodeId, selectors: &[NodeId]) -> NodeId {
    let args: Vec<_> = std::iter::once(value)
        .chain(selectors.iter().copied())
        .collect();
    build_call(module, "get", &args)
}

pub(super) fn reduce_rows(module: &mut Module, operation: &str, matrix: NodeId) -> NodeId {
    let axes = axes(module, 2);
    let cell = indexed(module, matrix, &axes);
    let row_axis = build_call(module, "vector", &axes[..1]);
    let reduction = constant(module, operation);
    build_call(module, "aggregate", &[reduction, row_axis, cell])
}

fn aggregate(module: &mut Module, output_axes: &[NodeId], expression: NodeId) -> NodeId {
    let sum = constant(module, "sum");
    let axes = build_call(module, "vector", output_axes);
    build_call(module, "aggregate", &[sum, axes, expression])
}

pub(super) fn cumulative_extreme(
    emitter: &mut Emitter,
    bar: NodeId,
    input: NodeId,
    output: NodeId,
    maximum: bool,
    forward: bool,
) -> Option<NodeId> {
    let [length] = *real_shape(&emitter.module, input)? else {
        return None;
    };
    if length == Dim::Static(0) {
        return Some(bar);
    }
    let source = emitter.primal(input);
    let prefix = emitter.primal(output);
    let m = &mut emitter.module;
    let indices = build_call(m, "indicesof", &[source]);
    let one = int(m, 1);
    let zero = int(m, 0);
    let previous_indices = broadcast(m, "sub", &[indices, one]);
    let previous_indices = broadcast(m, "max", &[previous_indices, one]);
    let previous = build_call(m, "get", &[prefix, previous_indices]);
    let first = broadcast(m, "equal", &[indices, one]);
    // At equal values, keep the previous winner. A finite tie between
    // distinct arguments is outside the smooth domain.
    let compare = if maximum { "gt" } else { "lt" };
    let better = broadcast(m, compare, &[source, previous]);
    let took = broadcast(m, "lor", &[first, better]);
    let changes = broadcast(m, "ifelse", &[took, indices, zero]);
    let winners = build_call(m, "cummax", &[changes]);
    m.set_type(
        winners,
        Type::Array {
            shape: Box::new([length]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        },
    );
    if forward {
        Some(build_call(m, "get", &[bar, winners]))
    } else {
        scatter(emitter, bar, input, &[winners], 1)
    }
}

pub(super) fn logsumexp(
    emitter: &mut Emitter,
    input: NodeId,
    output: NodeId,
    bar: NodeId,
    forward: bool,
) -> Option<NodeId> {
    let [_] = real_shape(&emitter.module, input)? else {
        return None;
    };
    let [] = real_shape(&emitter.module, output)? else {
        return None;
    };
    let input = emitter.primal(input);
    let m = &mut emitter.module;
    // Subtracting the rounded logsumexp loses relative precision when the
    // logarithmic coordinates share a large common offset.
    let maximum = build_call(m, "maximum", &[input]);
    let shifted = broadcast(m, "sub", &[input, maximum]);
    let weights = broadcast(m, "exp", &[shifted]);
    let total = build_call(m, "sum", &[weights]);
    let weights = broadcast(m, "divide", &[weights, total]);
    let term = broadcast(m, "mul", &[weights, bar]);
    let term = zero_when_masked(m, bar, term, true);
    Some(if forward {
        build_call(m, "sum", &[term])
    } else {
        term
    })
}
