//! Independent observations share one density body. External parameters become
//! explicit singleton arguments, so nested reifications do not capture them.

use super::*;
use flatppl_core::Idx;

pub(crate) fn query_head(m: &Module, id: NodeId) -> Option<Symbol> {
    let call = expect_builtin_call(m, id, "broadcast")?;
    let (head, _) = resolve_ref_chain(m, *call.args.first()?);
    match m.node(head) {
        Node::Const(head) if matches!(m.resolve(*head), "logdensityof" | "densityof") => {
            Some(*head)
        }
        _ => None,
    }
}

/// A measure is held constant, never passed to the numeric broadcast. Reify
/// the scalar query so the ordinary density rules eliminate it before emission.
pub(crate) fn lower_query(m: &mut Module, id: NodeId) -> Option<NodeId> {
    let op = query_head(m, id)?;
    let call = expect_builtin_call(m, id, "broadcast")?;
    let [_, measure, values] = *call.args else {
        return None;
    };
    if !call.named.is_empty() {
        return None;
    }
    let (resolved, _) = resolve_ref_chain(m, values);
    let (cell, rank) = match m.type_of(values).or_else(|| m.type_of(resolved)).cloned() {
        Some(Type::Array { elem, shape }) => (*elem, shape.len()),
        Some(Type::Table { columns, .. }) => (Type::Record(columns), 1),
        Some(ty) => (ty, 0),
        None => (Type::Any, 0),
    };
    if rank == 0 {
        return Some(m.alloc(Node::Call(Call {
            head: CallHead::Builtin(op),
            args: vec![measure, values].into(),
            named: vec![].into(),
            inputs: None,
        })));
    }
    let (label, reference, point) = placeholder(m, cell);
    let query = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(op),
        args: vec![measure, point].into(),
        named: vec![].into(),
        inputs: None,
    }));
    let body = expand_objects(m, query, &mut std::collections::HashMap::new());
    Some(map_body(
        m,
        body,
        vec![(label, reference)],
        vec![values],
        rank,
    ))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Capture {
    Reference(Ref),
    Projection(NodeId),
}

/// Find the root reference of a field path through records.
fn record_projection(m: &Module, id: NodeId) -> Option<Ref> {
    let call = expect_builtin_call(m, id, "get")?;
    let [source, field] = *call.args else {
        return None;
    };
    let (resolved, _) = resolve_ref_chain(m, source);
    if !matches!(m.node(field), Node::Lit(Scalar::Str(_)))
        || !matches!(
            m.type_of(source).or_else(|| m.type_of(resolved)),
            Some(Type::Record(_))
        )
    {
        return None;
    }
    match m.node(source) {
        Node::Ref(
            reference @ Ref {
                ns: RefNs::Local | RefNs::SelfMod,
                ..
            },
        ) => Some(*reference),
        _ => record_projection(m, source),
    }
}

fn captures(m: &Module, root: NodeId, inputs: &[(Symbol, Ref)]) -> Vec<(Capture, NodeId)> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        // Arrays cannot contain records. Pass the used scalar/array field,
        // rather than lifting its entire parameter record into vector(record).
        // Fields of this density's cell inputs must still vary by cell.
        if let Some(reference) = record_projection(m, id)
            && !inputs.iter().any(|(_, r)| *r == reference)
            && !matches!(m.type_of(id), Some(Type::Record(_)))
        {
            result.push((Capture::Projection(id), id));
            continue;
        }
        match m.node(id) {
            Node::Ref(
                reference @ Ref {
                    ns: RefNs::Local | RefNs::SelfMod,
                    ..
                },
            ) => {
                let capture = Capture::Reference(*reference);
                if !inputs.iter().any(|(_, r)| r == reference)
                    && !result.iter().any(|(r, _)| *r == capture)
                {
                    result.push((capture, id));
                }
            }
            Node::Call(c) if c.inputs.is_some() => {}
            node => node.for_each_child(|child| stack.push(child)),
        }
    }
    result
}

pub(crate) fn placeholder(m: &mut Module, cell: Type) -> (Symbol, Ref, NodeId) {
    let used: std::collections::HashSet<_> = (0..m.node_count())
        .filter_map(|i| match m.node(NodeId::from_usize(i)) {
            Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }) => Some(*name),
            _ => None,
        })
        .collect();
    let mut serial = m.node_count();
    let name = loop {
        let name = m.intern(&format!("_density_point_{serial}_"));
        if !used.contains(&name) {
            break name;
        }
        serial += 1;
    };
    let point_ref = Ref {
        ns: RefNs::Local,
        name,
    };
    let point = m.alloc(Node::Ref(point_ref));
    m.set_type(point, cell);
    // Surface boundaries need an ordinary keyword and a placeholder-shaped
    // local target, so serialized density functions retain their scope.
    let label = m.intern(&format!("density_point_{serial}"));
    (label, point_ref, point)
}

fn has_object(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Measure { .. }
            | Type::Kernel { .. }
            | Type::Function { .. }
            | Type::Likelihood { .. }
            | Type::Module
            | Type::Tuple(_)
    )
}

/// Sequence callbacks receive objects at compile time. Specialize those objects
/// into the body, then carry only their live value leaves through the loop.
pub(crate) fn specialize_sequence_parameters(m: &mut Module) -> bool {
    fn walk(
        m: &mut Module,
        id: NodeId,
        memo: &mut std::collections::HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&result) = memo.get(&id) {
            return result;
        }
        let children = m.node(id).children();
        let mapped: Vec<_> = children.iter().map(|&child| walk(m, child, memo)).collect();
        let current = if mapped == children {
            id
        } else {
            crate::driver::rebuild_with_children(m, id, &mapped)
        };
        let result = specialize(m, current).unwrap_or(current);
        memo.insert(id, result);
        result
    }

    fn specialize(m: &mut Module, id: NodeId) -> Option<NodeId> {
        let Node::Call(call) = m.node(id) else {
            return None;
        };
        let CallHead::Builtin(head) = call.head else {
            return None;
        };
        let base_arity = match m.resolve(head) {
            "scan" => 3,
            "reduce" => 2,
            _ => return None,
        };
        if call.args.len() != base_arity + 1 {
            return None;
        }
        let parameter = call.args[base_arity];
        let (resolved, _) = resolve_ref_chain(m, parameter);
        if !m
            .type_of(parameter)
            .or_else(|| m.type_of(resolved))
            .is_some_and(has_object)
        {
            return None;
        }
        let mut function = crate::kernel::resolve_reified(m, call.args[0])?;
        if function.auto || function.inputs.len() != 3 {
            return None;
        }
        let mut args = call.args[..base_arity].to_vec();
        let (_, parameter_ref) = function.inputs.pop()?;
        let body = crate::kernel::substitute_admitted(
            m,
            function.body,
            &[(parameter_ref, parameter)],
            crate::kernel::Substitute::All,
            false,
        );
        let body = expand_objects(m, body, &mut std::collections::HashMap::new());
        let body = crate::canon::inline_expression(m, body);
        // Inlined helpers may expose further object aliases or projections.
        let body = expand_objects(m, body, &mut std::collections::HashMap::new());
        let live: Vec<_> = captures(m, body, &function.inputs)
            .into_iter()
            .filter(|(_, value)| {
                let resolved = resolve_ref_chain(m, *value).0;
                m.phase_of(*value).or_else(|| m.phase_of(resolved))
                    != Some(flatppl_core::Phase::Fixed)
            })
            .collect();
        let mut replacements = Vec::new();
        if !live.is_empty() {
            let types: Vec<_> = live
                .iter()
                .map(|(_, value)| {
                    m.type_of(*value)
                        .or_else(|| m.type_of(resolve_ref_chain(m, *value).0))
                        .cloned()
                        .unwrap_or(Type::Any)
                })
                .collect();
            let ty = if types.len() == 1 {
                types[0].clone()
            } else {
                Type::Tuple(types.clone().into())
            };
            let (label, reference, point) = placeholder(m, ty);
            function.inputs.push((label, reference));
            for (i, (capture, _)) in live.iter().enumerate() {
                let value = if live.len() == 1 {
                    point
                } else {
                    let index = m.alloc(Node::Lit(Scalar::Int((i + 1) as i64)));
                    let value = build_call(m, "get", &[point, index]);
                    m.set_type(value, types[i].clone());
                    value
                };
                replacements.push((*capture, value));
            }
            let values: Vec<_> = live.iter().map(|(_, value)| *value).collect();
            args.push(if values.len() == 1 {
                values[0]
            } else {
                build_call(m, "tuple", &values)
            });
        }
        let body = crate::driver::map_tree(m, body, &mut |m, id| {
            replacements
                .iter()
                .find_map(|(capture, value)| {
                    let matches = match capture {
                        Capture::Reference(reference) => m.node(id) == &Node::Ref(*reference),
                        Capture::Projection(projection) => *projection == id,
                    };
                    matches.then_some(*value)
                })
                .or_else(|| matches!(m.node(id), Node::Call(c) if c.inputs.is_some()).then_some(id))
        });
        let reify = CallHead::Builtin(m.intern("functionof"));
        args[0] = m.alloc(Node::Call(Call {
            head: reify,
            args: vec![body].into(),
            named: Box::new([]),
            inputs: Some(Inputs::Spec(function.inputs.into())),
        }));
        Some(m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: args.into(),
            named: Box::new([]),
            inputs: None,
        })))
    }

    let bindings: Vec<_> = m.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    let mut memo = std::collections::HashMap::new();
    let mut changed = false;
    for (bid, root) in bindings {
        let mapped = walk(m, root, &mut memo);
        if mapped != root {
            m.set_binding_rhs(bid, mapped);
            changed = true;
        }
    }
    changed
}

// Resolve static object structure while preserving value references as the
// identity-bearing leaves of the enclosing computation and its derivatives.
fn expand_objects(
    m: &mut Module,
    id: NodeId,
    memo: &mut std::collections::HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&result) = memo.get(&id) {
        return result;
    }
    memo.insert(id, id);
    if matches!(m.node(id), Node::Call(c) if c.inputs.is_some()) {
        return id;
    }
    let resolved = resolve_ref_chain(m, id).0;
    let object_ref = matches!(m.node(id), Node::Ref(_))
        && m.type_of(id)
            .or_else(|| m.type_of(resolved))
            .is_some_and(has_object);
    if resolved != id && object_ref {
        let result = expand_objects(m, resolved, memo);
        memo.insert(id, result);
        return result;
    }
    let children = m.node(id).children();
    let mapped: Vec<_> = children
        .iter()
        .map(|&child| expand_objects(m, child, memo))
        .collect();
    let result = if children == mapped {
        id
    } else {
        crate::driver::rebuild_with_children(m, id, &mapped)
    };
    let projected = if matches!(builtin_name(m, result), Some("get" | "get0")) {
        let promoted = crate::canon::promoted_input_names(m);
        crate::canon::project_tuple(m, result, &promoted)
            .or_else(|| crate::canon::project(m, result, &promoted))
            .map(|value| expand_objects(m, value, memo))
            .unwrap_or(result)
    } else {
        result
    };
    memo.insert(id, projected);
    projected
}

pub(super) fn sum_density(
    m: &mut Module,
    measure: NodeId,
    values: NodeId,
) -> Result<NodeId, RefuseError> {
    sum_density_with(m, measure, values, Vec::new(), Vec::new())
}

/// Bind varying kernel arguments before collecting invariant captures. Apply
/// the kernel at one cell, so its composed density is lowered only once.
pub(super) fn kernel_density(
    m: &mut Module,
    head: NodeId,
    kernel: &crate::kernel::Kernel,
    positional: &[NodeId],
    named: &[(Symbol, NodeId)],
    values: NodeId,
) -> Result<NodeId, RefuseError> {
    if (kernel.auto && !positional.is_empty())
        || positional.len() + named.len() != kernel.inputs.len()
    {
        return Err(refuse(
            head,
            m,
            "broadcast arguments must bind each kernel input once",
        ));
    }
    // Broadcast does not auto-splat table rows. Normalize to keywords before
    // constructing an ordinary application, which otherwise would auto-splat.
    let mut bound = Vec::new();
    for (i, &(name, _)) in kernel.inputs.iter().enumerate() {
        let mut keywords = named.iter().filter(|(n, _)| *n == name);
        let value = match (positional.get(i), keywords.next(), keywords.next()) {
            (Some(&value), None, None) | (None, Some(&(_, value)), None) => value,
            _ => {
                return Err(refuse(
                    head,
                    m,
                    "broadcast arguments must bind each kernel input once",
                ));
            }
        };
        bound.push((name, value));
    }
    call_density(m, CallHead::User(head), &[], &bound, values)
}

/// Composed measures use the same per-cell lowering as reified kernels. Measure
/// operands stay invariant even when their variates are arrays.
pub(super) fn call_density(
    m: &mut Module,
    head: CallHead,
    positional: &[NodeId],
    named: &[(Symbol, NodeId)],
    values: NodeId,
) -> Result<NodeId, RefuseError> {
    let BoundCall {
        measure,
        inputs,
        args,
    } = bind_call(m, head, positional, named)?;
    if inputs.is_empty() {
        return lower_measure_density(m, measure, values);
    }
    sum_density_with(m, measure, values, inputs, args)
}

/// A mixture shares the whole observation across components. Only the family
/// and weight axes vary; independent coordinates inside a component stay inside.
pub(super) fn mixture_density(
    m: &mut Module,
    head: NodeId,
    positional: &[NodeId],
    named: &[(Symbol, NodeId)],
    weights: NodeId,
    value: NodeId,
) -> Result<NodeId, RefuseError> {
    if positional
        .iter()
        .chain(named.iter().map(|(_, value)| value))
        .any(|&value| {
            let (resolved, _) = resolve_ref_chain(m, value);
            m.type_of(value)
                .or_else(|| m.type_of(resolved))
                .map(array_axes)
                .unwrap_or(0)
                > 1
        })
    {
        return Err(refuse(
            head,
            m,
            "reified ksuperpose needs a one-axis parameter family",
        ));
    }
    let BoundCall {
        measure,
        mut inputs,
        mut args,
    } = bind_call(m, CallHead::User(head), positional, named)?;
    if !matches!(
        superposition_reference(m, measure, &mut Vec::new()),
        Some(SuperpositionReference::Counting | SuperpositionReference::Lebesgue)
    ) {
        return Err(refuse(
            head,
            m,
            "ksuperpose kernel needs a common counting or Lebesgue reference",
        ));
    }
    let weight = cell_input(m, weights, &mut inputs, &mut args)?;
    let weighted = build_call(m, "weighted", &[weight, measure]);
    let body = lower_measure_density(m, weighted, value)?;
    let terms = map_body(m, body, inputs, args, 1);
    Ok(build_call(m, "logsumexp", &[terms]))
}

struct BoundCall {
    measure: NodeId,
    inputs: Vec<(Symbol, Ref)>,
    args: Vec<NodeId>,
}

fn bind_call(
    m: &mut Module,
    head: CallHead,
    positional: &[NodeId],
    named: &[(Symbol, NodeId)],
) -> Result<BoundCall, RefuseError> {
    let mut inputs = Vec::new();
    let mut args = Vec::new();
    let mut bind = |m: &mut Module, value| {
        let (resolved, _) = resolve_ref_chain(m, value);
        match m.type_of(value).or_else(|| m.type_of(resolved)) {
            Some(Type::Array { .. } | Type::Table { .. }) => {
                cell_input(m, value, &mut inputs, &mut args)
            }
            _ => Ok(value),
        }
    };
    let positional = positional
        .iter()
        .map(|&value| bind(m, value))
        .collect::<Result<Vec<_>, RefuseError>>()?;
    let named = named
        .iter()
        .map(|&(name, value)| {
            Ok(NamedArg {
                kind: NamedKind::Kwarg,
                name,
                value: bind(m, value)?,
            })
        })
        .collect::<Result<Vec<_>, RefuseError>>()?;
    let measure = m.alloc(Node::Call(Call {
        head,
        args: positional.into(),
        named: named.into(),
        inputs: None,
    }));
    Ok(BoundCall {
        measure,
        inputs,
        args,
    })
}

pub(crate) fn cell_input(
    m: &mut Module,
    values: NodeId,
    inputs: &mut Vec<(Symbol, Ref)>,
    args: &mut Vec<NodeId>,
) -> Result<NodeId, RefuseError> {
    let (resolved, _) = resolve_ref_one(m, values);
    let ty = m.type_of(values).or_else(|| m.type_of(resolved)).cloned();
    let point = match ty {
        Some(Type::Array { elem, .. }) => {
            let (label, reference, value) = placeholder(m, *elem);
            inputs.push((label, reference));
            args.push(values);
            value
        }
        Some(Type::Table { columns, .. }) => {
            // Broadcast the columns, not a table value. The density receives
            // a syntactic record so record-law lowering can resolve fields.
            let visible = expect_builtin_call(m, resolved, "table").map(|c| c.named.to_vec());
            let mut fields = Vec::new();
            for (name, ty) in columns.iter() {
                let (label, reference, value) = placeholder(m, ty.clone());
                inputs.push((label, reference));
                let column = match visible
                    .as_ref()
                    .and_then(|xs| xs.iter().find(|x| x.name == *name))
                {
                    Some(field) => field.value,
                    None => {
                        let key = m.alloc(Node::Lit(Scalar::Str(m.resolve(*name).into())));
                        build_call(m, "get", &[values, key])
                    }
                };
                args.push(column);
                fields.push(NamedArg {
                    kind: NamedKind::Field,
                    name: *name,
                    value,
                });
            }
            let head = CallHead::Builtin(m.intern("record"));
            m.alloc(Node::Call(Call {
                head,
                args: vec![].into(),
                named: fields.into(),
                inputs: None,
            }))
        }
        _ => {
            return Err(refuse(
                values,
                m,
                "density broadcast needs an array or table",
            ));
        }
    };
    Ok(point)
}

fn sum_density_with(
    m: &mut Module,
    measure: NodeId,
    values: NodeId,
    mut inputs: Vec<(Symbol, Ref)>,
    mut args: Vec<NodeId>,
) -> Result<NodeId, RefuseError> {
    let (resolved, _) = resolve_ref_one(m, values);
    let rank = match m.type_of(values).or_else(|| m.type_of(resolved)) {
        Some(Type::Array { shape, .. }) => shape.len(),
        _ => 1,
    };
    let point = cell_input(m, values, &mut inputs, &mut args)?;
    let body = lower_measure_density(m, measure, point)?;
    let mapped = map_body(m, body, inputs, args, rank);
    Ok(build_call(m, "sum", &[mapped]))
}

/// Map a scalar body, keeping captured arrays outside its cell axes.
pub(crate) fn map_body(
    m: &mut Module,
    body: NodeId,
    mut inputs: Vec<(Symbol, Ref)>,
    mut args: Vec<NodeId>,
    rank: usize,
) -> NodeId {
    let mut replacements = Vec::new();
    for (capture, value) in captures(m, body, &inputs) {
        let (resolved, _) = resolve_ref_chain(m, value);
        // Closed support sets and other fixed module bindings remain global.
        // In particular, a set must not become a runtime tensor argument.
        if matches!(
            capture,
            Capture::Reference(Ref {
                ns: RefNs::SelfMod,
                ..
            })
        ) && m.phase_of(resolved) == Some(flatppl_core::Phase::Fixed)
        {
            continue;
        }
        // A later posterior substitution can turn module references into
        // outer locals. Keep those references in arguments, outside this
        // boundary, and bind fresh placeholders inside the shared body.
        let ty = m
            .type_of(value)
            .or_else(|| m.type_of(resolved))
            .cloned()
            .unwrap_or(Type::Any);
        let (label, local, replacement) = placeholder(m, ty.clone());
        inputs.push((label, local));
        replacements.push((capture, replacement));
        let actual = match capture {
            Capture::Reference(reference) => m.alloc(Node::Ref(reference)),
            Capture::Projection(id) => id,
        };
        m.set_type(actual, ty);
        let mut singleton = build_call(m, "vector", &[actual]);
        if rank > 1 {
            let leading = m.alloc(Node::Lit(Scalar::Int((rank - 1) as i64)));
            let trailing = m.alloc(Node::Lit(Scalar::Int(0)));
            singleton = build_call(m, "addaxes", &[singleton, leading, trailing]);
        }
        args.push(singleton);
    }
    let body = crate::driver::map_tree(m, body, &mut |m, id| {
        replacements
            .iter()
            .find_map(|(capture, value)| {
                let matches = match capture {
                    Capture::Reference(reference) => m.node(id) == &Node::Ref(*reference),
                    Capture::Projection(projection) => *projection == id,
                };
                matches.then_some(*value)
            })
            .or_else(|| matches!(m.node(id), Node::Call(c) if c.inputs.is_some()).then_some(id))
    });
    let head = CallHead::Builtin(m.intern("functionof"));
    let function = m.alloc(Node::Call(Call {
        head,
        args: vec![body].into(),
        named: vec![].into(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    args.insert(0, function);
    build_call(m, "broadcast", &args)
}
