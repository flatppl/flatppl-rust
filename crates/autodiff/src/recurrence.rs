//! Expose small, fixed recurrences and record scan projections to the AD graph.

use std::collections::{HashMap, HashSet};

use flatppl_core::calling::reified_inputs;
use flatppl_core::graph::{
    build_call, build_record, expect_builtin_call, rebuild_with_children, resolve_ref_one,
};
use flatppl_core::{
    Binding, BindingId, DEFAULT_MAX_DEPTH, Dim, Module, Node, NodeId, Ref, RefNs, Scalar, Type,
};

use crate::{REAL, tensor};

pub(super) const MAX_STEPS: u32 = 64;

pub(super) fn lower(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    if let Some(projection) = projected_scan(module, inputs, node) {
        return Some(projection);
    }
    let (function, initial, sequence, params) =
        if let Some(call) = expect_builtin_call(module, node, "reduce") {
            let [function, sequence, params @ ..] = call.args.as_ref() else {
                return None;
            };
            (*function, None, *sequence, params)
        } else {
            let call = expect_builtin_call(module, node, "scan")?;
            let [function, initial, sequence, params @ ..] = call.args.as_ref() else {
                return None;
            };
            (*function, Some(*initial), *sequence, params)
        };
    let params = params.to_vec();
    let [Dim::Static(length)] = *tensor::real_shape(module, sequence)? else {
        return None;
    };
    if length == 0 || length > MAX_STEPS || params.len() > 1 {
        return None;
    }
    let expected = module.type_of(node)?.clone();
    if initial.map_or(expected != REAL, |value| {
        module.type_of(value) != Some(&REAL)
    }) {
        return None;
    }
    let (body, formals) = callback(module, inputs, function)?;
    if formals.len() != 2 + params.len() {
        return None;
    }

    let start = if initial.is_some() { 1 } else { 2 };
    let mut state = initial.unwrap_or_else(|| element(module, sequence, 1));
    let mut states = Vec::new();
    for index in start..=length {
        let cell = element(module, sequence, index);
        let actual = [state, cell].into_iter().chain(params.iter().copied());
        let bindings: Vec<_> = formals.iter().copied().zip(actual).collect();
        state = substitute(module, inputs, body, &bindings, &mut HashMap::new(), 0)?;
        states.push(state);
    }
    let result = if initial.is_some() {
        build_call(module, "vector", &states)
    } else {
        state
    };
    infer_result(module, result, &expected, &[])?;

    // A step can read its state multiple times. Bind each new state to keep
    // the source representation linear in the number of iterations.
    let mut named = HashMap::new();
    for state in states {
        if named.contains_key(&state) {
            continue;
        }
        let rhs = replace_states(module, state, &mut named, 0)?;
        let alias = bind_value(module, rhs, "_ad_state")?;
        named.insert(state, alias);
    }
    replace_states(module, result, &mut named, 0)
}

// A fixed scan of scalar records can expose one column without making AD
// differentiate a table. Bind each iteration's scalar fields before using the
// record as the next carry, which keeps the generated source's depth bounded.
fn projected_scan(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    let call = expect_builtin_call(module, node, "broadcast")?;
    let &[project, scan] = call.args.as_ref() else {
        return None;
    };
    if !call.named.is_empty() {
        return None;
    }
    let scan = resolve_unpromoted(module, inputs, scan)?;
    let scan_call = expect_builtin_call(module, scan, "scan")?;
    if !scan_call.named.is_empty() {
        return None;
    }
    let [step, initial, sequence, params @ ..] = scan_call.args.as_ref() else {
        return None;
    };
    let [Dim::Static(length)] = *tensor::numeric_shape(module, *sequence)? else {
        return None;
    };
    if length == 0
        || length > MAX_STEPS
        || tensor::real_shape(module, node)? != [Dim::Static(length)]
    {
        return None;
    }
    let Type::Table { columns, .. } = module.type_of(scan)? else {
        return None;
    };
    if columns.iter().any(|(_, ty)| *ty != REAL) {
        return None;
    }
    let (access, projected) = callback(module, inputs, project)?;
    let [formal] = projected.as_slice() else {
        return None;
    };
    let [receiver, selector] = *expect_builtin_call(module, access, "get")?.args.as_ref() else {
        return None;
    };
    let Node::Ref(receiver) = module.node(receiver) else {
        return None;
    };
    let Node::Lit(Scalar::Str(field)) = module.node(selector) else {
        return None;
    };
    if receiver != formal {
        return None;
    }
    let column = columns
        .iter()
        .position(|(name, _)| module.resolve(*name) == field.as_ref())?;
    let (body, formals) = callback(module, inputs, *step)?;
    if formals.len() != 2 + params.len() {
        return None;
    }
    expect_builtin_call(module, body, "record")?;

    let expected = Type::Record(columns.clone());
    let result_type = module.type_of(node)?.clone();
    let params = params.to_vec();
    // Leave no generated bindings behind if a specialization cannot type-check.
    let mut staged = module.clone();
    let mut state = *initial;
    let mut output = Vec::with_capacity(length as usize);
    let mut seeds = Vec::new();
    for index in 1..=length {
        let cell = element(&mut staged, *sequence, index);
        let actual = [state, cell].into_iter().chain(params.iter().copied());
        let bindings: Vec<_> = formals.iter().copied().zip(actual).collect();
        let next = substitute(&mut staged, inputs, body, &bindings, &mut HashMap::new(), 0)?;
        infer_result(&mut staged, next, &expected, &seeds)?;

        let record = expect_builtin_call(&staged, next, "record")?;
        let fields = columns
            .iter()
            .map(|(name, _)| {
                let value = record.named.iter().find(|field| field.name == *name)?.value;
                Some((*name, value))
            })
            .collect::<Option<Vec<_>>>()?;
        let fields = fields
            .into_iter()
            .map(|(name, value)| Some((name, bind_value(&mut staged, value, "_ad_state")?)))
            .collect::<Option<Vec<_>>>()?;
        seeds = fields.iter().map(|&(_, field)| (field, REAL)).collect();
        state = build_record(&mut staged, &fields);
        output.push(fields[column].1);
    }
    let result = build_call(&mut staged, "vector", &output);
    staged.set_type(result, result_type);
    *module = staged;
    Some(result)
}

fn infer_result(
    module: &mut Module,
    result: NodeId,
    expected: &Type,
    inputs: &[(NodeId, Type)],
) -> Option<()> {
    let (types, diagnostics) = flatppl_infer::infer_expression(module, result, inputs);
    if types.get(&result) != Some(expected)
        || diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return None;
    }
    for (node, ty) in types {
        if module.type_of(node).is_none() {
            module.set_type(node, ty);
        }
    }
    Some(())
}

fn resolve_unpromoted(
    module: &Module,
    inputs: &HashSet<BindingId>,
    mut value: NodeId,
) -> Option<NodeId> {
    for _ in 0..=DEFAULT_MAX_DEPTH {
        let (rhs, Some(binding)) = resolve_ref_one(module, value) else {
            return Some(value);
        };
        if inputs.contains(&binding) {
            return None;
        }
        value = rhs;
    }
    None
}

pub(super) fn callback(
    module: &Module,
    inputs: &HashSet<BindingId>,
    function: NodeId,
) -> Option<(NodeId, Vec<Ref>)> {
    let function = resolve_unpromoted(module, inputs, function)?;
    let callable = expect_builtin_call(module, function, "functionof")?;
    let [body] = *callable.args.as_ref() else {
        return None;
    };
    let entries = reified_inputs(module, function, callable.inputs.as_ref()?)?;
    let params = entries.iter().map(|&(_, reference)| reference).collect();
    Some((body, params))
}

fn element(module: &mut Module, vector: NodeId, index: u32) -> NodeId {
    let index = tensor::int(module, i64::from(index));
    build_call(module, "get", &[vector, index])
}

pub(super) fn substitute(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
    bindings: &[(Ref, NodeId)],
    memo: &mut HashMap<NodeId, NodeId>,
    depth: usize,
) -> Option<NodeId> {
    if depth > DEFAULT_MAX_DEPTH {
        return None;
    }
    if let Some(&previous) = memo.get(&node) {
        return Some(previous);
    }
    let replacement = match module.node(node) {
        Node::Ref(reference) => {
            if let Some((_, actual)) = bindings.iter().find(|(formal, _)| formal == reference) {
                *actual
            } else if reference.ns == RefNs::Local {
                return None;
            } else {
                let (rhs, binding) = resolve_ref_one(module, node);
                if binding.is_some_and(|binding| !inputs.contains(&binding)) {
                    let replaced = substitute(module, inputs, rhs, bindings, memo, depth + 1)?;
                    if replaced == rhs { node } else { replaced }
                } else {
                    node
                }
            }
        }
        Node::Call(call) if call.inputs.is_some() => return None,
        _ => {
            let children = module.node(node).children();
            let args: Vec<_> = children
                .iter()
                .map(|&child| substitute(module, inputs, child, bindings, memo, depth + 1))
                .collect::<Option<_>>()?;
            if expect_builtin_call(module, node, "get").is_some()
                && let [receiver, selector] = *args.as_slice()
                && let Some(field) = record_field(module, inputs, receiver, selector)
            {
                field
            } else {
                rebuild(module, node, children, args)
            }
        }
    };
    memo.insert(node, replacement);
    Some(replacement)
}

fn record_field(
    module: &Module,
    inputs: &HashSet<BindingId>,
    receiver: NodeId,
    selector: NodeId,
) -> Option<NodeId> {
    let Node::Lit(Scalar::Str(field)) = module.node(selector) else {
        return None;
    };
    let receiver = resolve_unpromoted(module, inputs, receiver)?;
    let record = expect_builtin_call(module, receiver, "record")?;
    record
        .named
        .iter()
        .find(|arg| module.resolve(arg.name) == field.as_ref())
        .map(|arg| arg.value)
}
fn replace_states(
    module: &mut Module,
    node: NodeId,
    named: &mut HashMap<NodeId, NodeId>,
    depth: usize,
) -> Option<NodeId> {
    if let Some(&reference) = named.get(&node) {
        return Some(reference);
    }
    if depth > DEFAULT_MAX_DEPTH {
        return None;
    }
    let children = module.node(node).children();
    let args = children
        .iter()
        .map(|&child| replace_states(module, child, named, depth + 1))
        .collect::<Option<_>>()?;
    let result = rebuild(module, node, children, args);
    if result != node {
        let ty = module.type_of(node)?.clone();
        module.set_type(result, ty);
    }
    // Do not insert unchanged nodes, since a future state can equal this node.
    if result != node {
        named.insert(node, result);
    }
    Some(result)
}

pub(super) fn bind_value(module: &mut Module, rhs: NodeId, prefix: &str) -> Option<NodeId> {
    let mut index = module.node_count();
    let name = loop {
        let name = module.intern(&format!("{prefix}_{index}"));
        if module.binding_by_name(name).is_none() {
            break name;
        }
        index += 1;
    };
    let ty = module.type_of(rhs)?.clone();
    module.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public: false,
        synthetic: false,
    });
    let reference = module.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }));
    module.set_type(reference, ty);
    Some(reference)
}

fn rebuild(module: &mut Module, node: NodeId, before: Vec<NodeId>, after: Vec<NodeId>) -> NodeId {
    if before == after {
        node
    } else {
        rebuild_with_children(module, node, &after)
    }
}
