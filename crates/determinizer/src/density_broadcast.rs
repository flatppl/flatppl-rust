//! Independent observations share one density body. External parameters become
//! explicit singleton arguments, so nested reifications do not capture them.

use super::*;
use flatppl_core::Idx;

fn captures(m: &Module, root: NodeId) -> Vec<(Ref, NodeId)> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match m.node(id) {
            Node::Ref(
                reference @ Ref {
                    ns: RefNs::Local | RefNs::SelfMod,
                    ..
                },
            ) => {
                if !result.iter().any(|(r, _)| r == reference) {
                    result.push((*reference, id));
                }
            }
            Node::Call(c) if c.inputs.is_some() => {}
            node => node.for_each_child(|child| stack.push(child)),
        }
    }
    result
}

fn placeholder(m: &mut Module, cell: Type) -> (Symbol, Ref, NodeId) {
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
    let named = bound
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
        head: CallHead::User(head),
        args: vec![].into(),
        named: named.into(),
        inputs: None,
    }));
    if inputs.is_empty() {
        return lower_measure_density(m, measure, values);
    }
    sum_density_with(m, measure, values, inputs, args)
}

fn cell_input(
    m: &mut Module,
    values: NodeId,
    inputs: &mut Vec<(Symbol, Ref)>,
    args: &mut Vec<NodeId>,
) -> Result<NodeId, RefuseError> {
    let (resolved, _) = resolve_ref_one(m, values);
    let ty = m.type_of(values).or_else(|| m.type_of(resolved)).cloned();
    let point = match ty {
        Some(Type::Array { shape, elem }) if shape.len() == 1 => {
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
                "density broadcast needs a one-dimensional array or table",
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
    let point = cell_input(m, values, &mut inputs, &mut args)?;
    let body = lower_measure_density(m, measure, point)?;
    let mut replacements = Vec::new();
    for (reference, value) in captures(m, body) {
        if inputs.iter().any(|(_, r)| *r == reference) {
            continue;
        }
        let (resolved, _) = resolve_ref_chain(m, value);
        // Closed support sets and other fixed module bindings remain global.
        // In particular, a set must not become a runtime tensor argument.
        if reference.ns == RefNs::SelfMod
            && m.phase_of(resolved) == Some(flatppl_core::Phase::Fixed)
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
        replacements.push((reference, replacement));
        let actual = m.alloc(Node::Ref(reference));
        m.set_type(actual, ty);
        args.push(build_call(m, "vector", &[actual]));
    }
    let body = crate::driver::map_tree(m, body, &mut |m, id| match m.node(id) {
        Node::Ref(reference) => replacements
            .iter()
            .find_map(|(r, value)| (r == reference).then_some(*value)),
        Node::Call(c) if c.inputs.is_some() => Some(id),
        _ => None,
    });
    let head = CallHead::Builtin(m.intern("functionof"));
    let function = m.alloc(Node::Call(Call {
        head,
        args: vec![body].into(),
        named: vec![].into(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    args.insert(0, function);
    let mapped = build_call(m, "broadcast", &args);
    Ok(build_call(m, "sum", &[mapped]))
}
