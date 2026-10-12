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
    let cell = match m.type_of(values).or_else(|| m.type_of(resolved)).cloned() {
        Some(Type::Array { elem, .. }) => *elem,
        Some(Type::Table { columns, .. }) => Type::Record(columns),
        Some(ty) => ty,
        None => Type::Any,
    };
    let (label, reference, point) = placeholder(m, cell);
    let query = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(op),
        args: vec![measure, point].into(),
        named: vec![].into(),
        inputs: None,
    }));
    let head = CallHead::Builtin(m.intern("functionof"));
    let function = m.alloc(Node::Call(Call {
        head,
        args: vec![query].into(),
        named: vec![].into(),
        inputs: Some(Inputs::Spec(vec![(label, reference)].into())),
    }));
    Some(build_call(m, "broadcast", &[function, values]))
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
    if inputs.is_empty() {
        return lower_measure_density(m, measure, values);
    }
    sum_density_with(m, measure, values, inputs, args)
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
