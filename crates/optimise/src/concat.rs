//! Identify written scalar reads and singleton stacks of real arrays.

use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{Dim, Module, NamedKind, Node, NodeId, Phase, Scalar, ScalarType, Type};

use crate::engine::{Fact, FactArg};
use crate::types::fact;

pub(super) struct Witness {
    pub(super) cat: NodeId,
    pub(super) parts: Vec<NodeId>,
    head: &'static str,
    index: i64,
    size: i64,
    fixed: [bool; 3],
}

pub(super) fn witness(
    m: &Module,
    id: NodeId,
    resolve: impl Fn(NodeId) -> Option<NodeId>,
) -> Option<Witness> {
    if m.type_of(id) != Some(&Type::Scalar(ScalarType::Real)) {
        return None;
    }
    let (head, base, read) = if let Some(read) = expect_builtin_call(m, id, "get") {
        ("get", 1, read)
    } else {
        ("get0", 0, expect_builtin_call(m, id, "get0")?)
    };
    let [source, index] = *read.args else {
        return None;
    };
    if !read.named.is_empty() || read.inputs.is_some() || resolve(index) != Some(index) {
        return None;
    }
    let Node::Lit(Scalar::Int(index)) = *m.node(index) else {
        return None;
    };
    let size = vector_len(m, source)?;
    if index < base || index >= size + base || size > i64::from(i32::MAX) {
        return None;
    }
    let cat = resolve(source)?;
    let call = expect_builtin_call(m, cat, "cat")?;
    if !call.named.is_empty()
        || call.inputs.is_some()
        || !(2..=3).contains(&call.args.len())
        || vector_len(m, cat)? != size
        || m.is_sample_result(cat)
        || m.sample_origin(cat).is_some()
    {
        return None;
    }
    let mut fixed = [false; 3];
    let mut scalar = 0;
    for (i, &part) in call.args.iter().enumerate() {
        if m.type_of(part) == Some(&Type::Scalar(ScalarType::Real)) {
            scalar += 1;
        } else {
            vector_len(m, part)?;
        }
        fixed[i] = literal_part(m, part, &resolve);
    }
    // Mixed ranks or element kinds can require conversions at the cat site.
    if scalar != 0 && scalar != call.args.len() {
        return None;
    }
    Some(Witness {
        cat,
        parts: call.args.to_vec(),
        head,
        index,
        size,
        fixed,
    })
}

pub(super) fn seed(
    witness: &Witness,
    term: usize,
    args: &[usize],
    uses: &[usize],
    facts: &mut Vec<Fact>,
) {
    use FactArg::{Expr, Int, Str};
    let [get_uses, cat_uses, part_uses @ ..] = uses else {
        return;
    };
    facts.push(fact(
        if args.len() == 2 {
            "concat-pair"
        } else {
            "concat-triple"
        },
        term,
        args.iter().copied().map(Expr),
    ));
    facts.push(fact(
        "concat-read",
        term,
        [Str(witness.head), Int(witness.index), Int(witness.size)],
    ));
    for (i, (&arg, &uses)) in args.iter().zip(part_uses).enumerate() {
        if witness.fixed[i] {
            facts.push(fact("concat-fixed-part", arg, []));
        }
        if *get_uses == 1 && *cat_uses == 1 && uses == 1 {
            facts.push(fact("concat-exclusive-part", term, [Expr(arg)]));
        }
    }
}

fn vector_len(m: &Module, id: NodeId) -> Option<i64> {
    let Type::Array { shape, elem } = m.type_of(id)? else {
        return None;
    };
    let [Dim::Static(n)] = **shape else {
        return None;
    };
    (**elem == Type::Scalar(ScalarType::Real)).then_some(i64::from(n))
}

pub(super) fn fixed_stack_vector(
    m: &Module,
    id: NodeId,
    resolve: impl Fn(NodeId) -> Option<NodeId>,
) -> Option<(&'static str, NodeId)> {
    let (head, stack) = if let Some(stack) = expect_builtin_call(m, id, "rowstack") {
        ("rowstack", stack)
    } else {
        ("colstack", expect_builtin_call(m, id, "colstack")?)
    };
    let [wrapper] = *stack.args else {
        return None;
    };
    if !stack.named.is_empty() || stack.inputs.is_some() || resolve(wrapper) != Some(wrapper) {
        return None;
    }
    let container = expect_builtin_call(m, wrapper, "vector")?;
    let [source] = *container.args else {
        return None;
    };
    if !container.named.is_empty() || container.inputs.is_some() {
        return None;
    }
    vector_len(m, source)?;
    fixed_data_vector(m, source, resolve).then_some((head, source))
}

fn fixed_data_vector(
    m: &Module,
    source: NodeId,
    resolve: impl Fn(NodeId) -> Option<NodeId>,
) -> bool {
    if literal_part(m, source, &resolve) {
        return true;
    }
    let Some(id) = resolve(source) else {
        return false;
    };
    if m.phase_of(id) != Some(Phase::Fixed)
        || m.is_sample_result(id)
        || m.sample_origin(id).is_some()
    {
        return false;
    }
    let Some(access) = expect_builtin_call(m, id, "get") else {
        return false;
    };
    let [record, key] = *access.args else {
        return false;
    };
    let Node::Lit(Scalar::Str(field)) = m.node(key) else {
        return false;
    };
    let Some(record) = resolve(record) else {
        return false;
    };
    if !access.named.is_empty()
        || access.inputs.is_some()
        || resolve(key) != Some(key)
        || m.phase_of(record) != Some(Phase::Fixed)
        || m.is_sample_result(record)
        || m.sample_origin(record).is_some()
    {
        return false;
    }
    // Fixed file data is still read through the same expression. An embedded
    // record can also contain promoted fields, so certify the selected field.
    expect_builtin_call(m, record, "load_data").is_some_and(|call| call.inputs.is_none())
        || expect_builtin_call(m, record, "record").is_some_and(|call| {
            call.inputs.is_none()
                && call.named.iter().any(|arg| {
                    arg.kind == NamedKind::Field
                        && m.resolve(arg.name) == field.as_ref()
                        && literal_part(m, arg.value, &resolve)
                })
        })
}

fn literal_part(m: &Module, node: NodeId, resolve: impl Fn(NodeId) -> Option<NodeId>) -> bool {
    let is_literal = |id| {
        resolve(id) == Some(id)
            && !m.is_sample_result(id)
            && m.sample_origin(id).is_none()
            && matches!(m.node(id), Node::Lit(Scalar::Real(_)))
    };
    let Some(id) = resolve(node) else {
        return false;
    };
    if m.is_sample_result(id) || m.sample_origin(id).is_some() {
        return false;
    }
    is_literal(id)
        || expect_builtin_call(m, id, "vector").is_some_and(|call| {
            call.named.is_empty()
                && call.inputs.is_none()
                && call.args.iter().copied().all(is_literal)
        })
}
