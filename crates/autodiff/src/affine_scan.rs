//! Loop-preserving AD for a fixed real scan with scalar affine state.

use std::collections::{HashMap, HashSet};

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{
    BindingId, Call, CallHead, Dim, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs,
    Scalar, Type,
};

use crate::{REAL, callbacks, recurrence, tensor};

pub(super) fn args(
    module: &Module,
    inputs: &HashSet<BindingId>,
    args: &[NodeId],
) -> Option<[NodeId; 3]> {
    let [function, initial, sequence, coefficient] = *args else {
        return None;
    };
    let (body, formals) = recurrence::callback(module, inputs, function)?;
    let [state, cell, scale] = *formals.as_slice() else {
        return None;
    };
    let (weighted, _) = sum(module, body, |id| is_ref(module, id, cell))?;
    let (_, factor) = product(module, weighted, |id| is_ref(module, id, state))?;
    if !is_ref(module, factor, scale)
        || !matches!(tensor::real_shape(module, sequence), Some([Dim::Static(n)]) if *n > 0)
        || module.type_of(initial) != Some(&REAL)
        || module.type_of(coefficient) != Some(&REAL)
    {
        return None;
    }
    Some([initial, sequence, coefficient])
}

pub(super) fn square_scan(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    let original = expect_builtin_call(module, node, "scan")?;
    let [function, initial, sequence, parameters] = *original.args.as_ref() else {
        return None;
    };
    if !matches!(tensor::real_shape(module, sequence), Some([Dim::Static(n)]) if *n > recurrence::MAX_STEPS)
        || module.type_of(initial) != Some(&REAL)
    {
        return None;
    }
    let expected = module.type_of(node)?.clone();
    let (body, formals) = recurrence::callback(module, inputs, function)?;
    let [state, cell, params] = *formals.as_slice() else {
        return None;
    };
    if formals.iter().any(|reference| reference.ns != RefNs::Local) {
        return None;
    }
    let [radicand] = *expect_builtin_call(module, body, "sqrt")?.args.as_ref() else {
        return None;
    };
    let has_square = |id| product(module, id, |id| square_of(module, id, state)).is_some();
    let (base, weighted) = sum(module, radicand, has_square)?;
    let (_, scale) = product(module, weighted, |id| square_of(module, id, state))?;
    if refers_to(module, base, &[state]) || refers_to(module, scale, &[state, cell]) {
        return None;
    }

    let offsets = offsets(module, inputs, base, cell, params, sequence, parameters)?;
    let scale = recurrence::substitute(
        module,
        inputs,
        scale,
        &[(params, parameters)],
        &mut HashMap::new(),
        0,
    )?;
    callbacks::annotate_equivalent(module, offsets, &expected)?;
    let offsets = callbacks::lift_broadcast(module, inputs, offsets).unwrap_or(offsets);
    let initial = build_call(module, "abs2", &[initial]);
    let variance = scan(module, initial, offsets, scale, false);
    let result = tensor::broadcast(module, "sqrt", &[variance]);
    callbacks::annotate_equivalent(module, result, &expected)
}

fn offsets(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    base: NodeId,
    cell: Ref,
    params: Ref,
    sequence: NodeId,
    parameters: NodeId,
) -> Option<NodeId> {
    let (label, new_cell) = placeholder(module, "item");
    let mut formals = vec![(label, new_cell)];
    let mut actual = vec![sequence];
    let replacement = match module.type_of(parameters)?.clone() {
        Type::Record(fields) => {
            let mut named = Vec::with_capacity(fields.len());
            for (i, (name, ty)) in fields.iter().enumerate() {
                if *ty != REAL {
                    return None;
                }
                let (label, reference) = placeholder(module, &format!("parameter{i}"));
                formals.push((label, reference));
                let value = module.alloc(Node::Ref(reference));
                named.push(NamedArg {
                    kind: NamedKind::Field,
                    name: *name,
                    value,
                });
                let key = module.alloc(Node::Lit(Scalar::Str(module.resolve(*name).into())));
                actual.push(build_call(module, "get", &[parameters, key]));
            }
            let head = CallHead::Builtin(module.intern("record"));
            module.alloc(Node::Call(Call {
                head,
                args: Box::new([]),
                named: named.into(),
                inputs: None,
            }))
        }
        ty if ty == REAL => {
            let (label, reference) = placeholder(module, "parameter");
            formals.push((label, reference));
            actual.push(parameters);
            module.alloc(Node::Ref(reference))
        }
        _ => return None,
    };
    let local_cell = module.alloc(Node::Ref(new_cell));
    let bindings = [(params, replacement), (cell, local_cell)];
    let body = recurrence::substitute(module, inputs, base, &bindings, &mut HashMap::new(), 0)?;
    let lifted = function_of(module, body, &formals);
    let args: Vec<_> = std::iter::once(lifted).chain(actual).collect();
    Some(build_call(module, "broadcast", &args))
}

fn placeholder(module: &mut Module, name: &str) -> (flatppl_core::Symbol, Ref) {
    let label = module.intern(name);
    let name = module.intern(&format!("_{name}_"));
    (
        label,
        Ref {
            ns: RefNs::Local,
            name,
        },
    )
}

fn sum(
    module: &Module,
    node: NodeId,
    on_right: impl Fn(NodeId) -> bool,
) -> Option<(NodeId, NodeId)> {
    let [a, b] = *expect_builtin_call(module, node, "add")?.args.as_ref() else {
        return None;
    };
    if on_right(b) {
        Some((a, b))
    } else {
        on_right(a).then_some((b, a))
    }
}

fn product(
    module: &Module,
    node: NodeId,
    on_left: impl Fn(NodeId) -> bool,
) -> Option<(NodeId, NodeId)> {
    let [a, b] = *expect_builtin_call(module, node, "mul")?.args.as_ref() else {
        return None;
    };
    if on_left(a) {
        Some((a, b))
    } else {
        on_left(b).then_some((b, a))
    }
}

fn is_ref(module: &Module, node: NodeId, formal: Ref) -> bool {
    matches!(module.node(node), Node::Ref(reference) if *reference == formal)
}

fn square_of(module: &Module, node: NodeId, formal: Ref) -> bool {
    if let Some(call) = expect_builtin_call(module, node, "abs2") {
        return matches!(call.args.as_ref(), &[base] if is_ref(module, base, formal));
    }
    expect_builtin_call(module, node, "pow").is_some_and(|call| {
        matches!(call.args.as_ref(), &[base, power]
          if is_ref(module, base, formal)
             && matches!(module.node(power), Node::Lit(Scalar::Int(2) | Scalar::Real(2.0))))
    })
}

fn refers_to(module: &Module, node: NodeId, refs: &[Ref]) -> bool {
    let mut pending = vec![node];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if matches!(module.node(node), Node::Ref(reference) if refs.contains(reference)) {
            return true;
        }
        pending.extend(module.node(node).children());
    }
    false
}

pub(super) fn scan(
    module: &mut Module,
    initial: NodeId,
    sequence: NodeId,
    scale: NodeId,
    guarded: bool,
) -> NodeId {
    let formals = ["state", "offset", "scale"].map(|name| placeholder(module, name));
    let [state, offset, coefficient] =
        formals.map(|(_, reference)| module.alloc(Node::Ref(reference)));
    let weight = build_call(module, "mul", &[state, coefficient]);
    let weight = if guarded {
        tensor::zero_when_masked(module, state, weight, false)
    } else {
        weight
    };
    let body = build_call(module, "add", &[weight, offset]);
    let step = function_of(module, body, &formals);
    build_call(module, "scan", &[step, initial, sequence, scale])
}

fn function_of(
    module: &mut Module,
    body: NodeId,
    formals: &[(flatppl_core::Symbol, Ref)],
) -> NodeId {
    let head = CallHead::Builtin(module.intern("functionof"));
    module.alloc(Node::Call(Call {
        head,
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(formals.into())),
    }))
}

pub(super) fn lag(module: &mut Module, initial: NodeId, states: NodeId, length: u32) -> NodeId {
    let first = build_call(module, "vector", &[initial]);
    if length == 1 {
        return first;
    }
    let length = tensor::int(module, i64::from(length - 1));
    let indices = build_call(module, "ones", &[length]);
    let indices = build_call(module, "indicesof", &[indices]);
    let previous = build_call(module, "get", &[states, indices]);
    build_call(module, "cat", &[first, previous])
}

/// Project the prediction of a record scan that copies its current item to
/// the next lag and advances an affine prediction residual.
pub(super) fn residual_projection(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    let [projection, sequence] = positional(module, node, "broadcast")? else {
        return None;
    };
    let (view, formals) = recurrence::callback(module, inputs, *projection)?;
    let [view_state] = *formals.as_slice() else {
        return None;
    };
    let output = field_key(module, view, view_state)?;
    let sequence = callbacks::resolve_unpromoted(module, inputs, *sequence)?;
    let [callback, initial, items, parameters] = *positional(module, sequence, "scan")? else {
        return None;
    };
    let &[Dim::Static(length)] = tensor::real_shape(module, items)? else {
        return None;
    };
    if length == 0 {
        return None;
    }

    let (body, formals) = recurrence::callback(module, inputs, callback)?;
    let [state, item, capture] = *formals.as_slice() else {
        return None;
    };
    if formals.iter().any(|formal| formal.ns != RefNs::Local) {
        return None;
    }
    let record = expect_builtin_call(module, body, "record")?;
    let fields = &record.named;
    if !record.args.is_empty() || fields.len() != 3 {
        return None;
    }
    let prediction = fields
        .iter()
        .find(|field| module.resolve(field.name) == output)?
        .value;
    let lag = fields
        .iter()
        .find(|field| is_ref(module, field.value, item))?
        .name;
    let residual = fields.iter().find_map(|field| {
        let [observation, fitted] = *positional(module, field.value, "sub")? else {
            return None;
        };
        (module.resolve(field.name) != output
            && field.name != lag
            && is_ref(module, observation, item)
            && module.structural_eq(fitted, prediction))
        .then_some(field.name)
    })?;
    if module.resolve(lag) == output {
        return None;
    }

    let (base, residual_weight) = term_and_weight(module, prediction, state, residual)?;
    let (location, lag_weight) = term_and_weight(module, base, state, lag)?;
    let bindings = [(capture, parameters)];
    let parameter = |module: &mut Module, term| {
        if refers_to(module, term, &[state, item]) {
            return None;
        }
        recurrence::substitute(module, inputs, term, &bindings, &mut HashMap::new(), 0)
    };
    let location = parameter(module, location)?;
    let lag_weight = parameter(module, lag_weight)?;
    let residual_weight = parameter(module, residual_weight)?;

    let expected = module.type_of(node)?.clone();
    let first_item = record_field(module, initial, lag);
    let first_residual = record_field(module, initial, residual);
    let lagged_items = self::lag(module, first_item, items, length);
    let lagged_part = tensor::broadcast(module, "mul", &[lag_weight, lagged_items]);
    let drive = tensor::broadcast(module, "add", &[location, lagged_part]);
    let offsets = build_call(module, "sub", &[items, drive]);
    let neg_weight = build_call(module, "neg", &[residual_weight]);
    let errors = scan(module, first_residual, offsets, neg_weight, false);
    let lagged_errors = self::lag(module, first_residual, errors, length);
    let residual_part = tensor::broadcast(module, "mul", &[residual_weight, lagged_errors]);
    let prediction = build_call(module, "add", &[drive, residual_part]);
    callbacks::annotate_equivalent(module, prediction, &expected)
}

fn term_and_weight(
    module: &Module,
    body: NodeId,
    state: Ref,
    field: flatppl_core::Symbol,
) -> Option<(NodeId, NodeId)> {
    let is_projection = |node| field_key(module, node, state) == Some(module.resolve(field));
    let (base, weighted) = sum(module, body, |node| {
        product(module, node, is_projection).is_some()
    })?;
    let (_, weight) = product(module, weighted, is_projection)?;
    Some((base, weight))
}

fn field_key(module: &Module, node: NodeId, state: Ref) -> Option<&str> {
    let [source, selector] = *positional(module, node, "get")? else {
        return None;
    };
    let Node::Lit(Scalar::Str(name)) = module.node(selector) else {
        return None;
    };
    is_ref(module, source, state).then_some(())?;
    Some(name)
}

fn record_field(module: &mut Module, record: NodeId, field: flatppl_core::Symbol) -> NodeId {
    let key = module.alloc(Node::Lit(Scalar::Str(module.resolve(field).into())));
    build_call(module, "get", &[record, key])
}

fn positional<'a>(module: &'a Module, node: NodeId, head: &str) -> Option<&'a [NodeId]> {
    let call = expect_builtin_call(module, node, head)?;
    call.named.is_empty().then_some(&call.args)
}
