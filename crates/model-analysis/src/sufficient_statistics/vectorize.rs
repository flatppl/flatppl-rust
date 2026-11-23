//! Stack like-shaped fixed-partition scores along an occupied-group axis.

use std::collections::HashMap;
use std::mem::{Discriminant, discriminant};

use flatppl_core::graph::{build_call, fold_add};
use flatppl_core::{
    Call, CallHead, Dim, Idx, Module, NamedKind, Node, NodeId, Scalar, ScalarType, Symbol, Type,
};

/// Only inspect scoring operations created by the partition. User expressions
/// remain whole, and the existing scalar-family proof owns each group's score.
enum Column<'a> {
    Scalar(ScalarType),
    Cell(NodeId, NodeId),
    Apply(Symbol, &'a Call),
    Constant(Symbol),
    Whole(NodeId),
}

fn arguments(call: &Call) -> impl Iterator<Item = NodeId> + '_ {
    call.args
        .iter()
        .copied()
        .chain(call.named.iter().map(|a| a.value))
}

fn column(m: &Module, id: NodeId, original_nodes: usize) -> Column<'_> {
    let ty = m.type_of(id);
    if let Node::Const(name) = m.node(id) {
        return Column::Constant(*name);
    }
    if id.index() >= original_nodes
        && let Node::Call(call) = m.node(id)
        && let CallHead::Builtin(op) = call.head
        && call.inputs.is_none()
    {
        match m.resolve(op) {
            "get"
                if let [array, index] = *call.args
                    && let Node::Lit(Scalar::Int(i)) = m.node(index)
                    && let Some(Type::Array { shape, elem }) = m.type_of(array)
                    && let [Dim::Static(n)] = **shape
                    && (1..=i64::from(n)).contains(i)
                    && ty == Some(elem.as_ref()) =>
            {
                return Column::Cell(array, index);
            }
            "record"
                if call.args.is_empty()
                    && call.named.iter().all(|arg| arg.kind == NamedKind::Field)
                    && matches!(ty, Some(Type::Record(fields)) if fields.iter().all(|(_, ty)| matches!(ty, Type::Scalar(_)))) =>
            {
                return Column::Apply(op, call);
            }
            "get" | "get0" | "broadcast" => {}
            _ if call.named.is_empty() && matches!(ty, Some(Type::Scalar(_))) => {
                return Column::Apply(op, call);
            }
            _ => {}
        }
    }
    match ty {
        Some(Type::Scalar(kind)) => Column::Scalar(*kind),
        _ => Column::Whole(id),
    }
}

#[derive(Eq, Hash, PartialEq)]
enum Shape {
    Scalar(Discriminant<ScalarType>),
    Cells(NodeId),
    Call(Symbol, Vec<usize>, Vec<Symbol>),
    Constant(Symbol),
    Whole(NodeId),
}

#[derive(Default)]
struct Shapes {
    nodes: HashMap<NodeId, usize>,
    distinct: HashMap<Shape, usize>,
}

impl Shapes {
    fn classify(&mut self, m: &Module, id: NodeId, original_nodes: usize) -> usize {
        if let Some(&shape) = self.nodes.get(&id) {
            return shape;
        }
        let shape = match column(m, id, original_nodes) {
            Column::Scalar(kind) => Shape::Scalar(discriminant(&kind)),
            Column::Cell(array, _) => Shape::Cells(array),
            Column::Apply(op, call) => Shape::Call(
                op,
                arguments(call)
                    .map(|arg| self.classify(m, arg, original_nodes))
                    .collect(),
                call.named.iter().map(|a| a.name).collect(),
            ),
            Column::Constant(name) => Shape::Constant(name),
            Column::Whole(id) => Shape::Whole(id),
        };
        let next = self.distinct.len();
        let shape = *self.distinct.entry(shape).or_insert(next);
        self.nodes.insert(id, shape);
        shape
    }
}

fn same_leaf(m: &Module, a: NodeId, b: NodeId) -> bool {
    a == b
        || match (m.node(a), m.node(b)) {
            // Scalar equality identifies the two signed zeros.
            (Node::Lit(Scalar::Real(a)), Node::Lit(Scalar::Real(b))) => a.to_bits() == b.to_bits(),
            (Node::Lit(a), Node::Lit(b)) => a == b,
            (Node::Ref(a), Node::Ref(b)) => a == b,
            (Node::Const(a), Node::Const(b)) => a == b,
            _ => false,
        }
}

/// A varying result must have exactly the original cell type, once per group.
fn has_cells(m: &Module, vector: NodeId, cell: NodeId, groups: Dim) -> bool {
    match (m.type_of(vector), m.type_of(cell)) {
        (Some(Type::Array { shape, elem }), Some(ty @ Type::Scalar(_))) => {
            shape.as_ref() == [groups] && elem.as_ref() == ty
        }
        (Some(Type::Table { nrows, columns }), Some(Type::Record(fields))) => {
            *nrows == groups && columns == fields
        }
        _ => false,
    }
}

#[derive(Default)]
struct Stacks {
    // A shared subexpression stays shared across all of its uses in the score.
    columns: HashMap<Vec<NodeId>, (NodeId, bool)>,
}

impl Stacks {
    /// The nodes in this column all have the same interned structure.
    fn stack(
        &mut self,
        m: &mut Module,
        ids: &[NodeId],
        original_nodes: usize,
    ) -> Option<(NodeId, bool)> {
        if let Some(&result) = self.columns.get(ids) {
            return Some(result);
        }
        let result = self.stack_column(m, ids, original_nodes)?;
        self.columns.insert(ids.to_vec(), result);
        Some(result)
    }

    fn stack_column(
        &mut self,
        m: &mut Module,
        ids: &[NodeId],
        original_nodes: usize,
    ) -> Option<(NodeId, bool)> {
        let first = ids[0];
        if ids.iter().skip(1).all(|&id| same_leaf(m, first, id)) {
            return Some((first, false));
        }
        let value = match column(m, first, original_nodes) {
            Column::Scalar(_) => build_call(m, "vector", ids),
            Column::Cell(array, _) => {
                let indices = ids
                    .iter()
                    .map(|&id| match column(m, id, original_nodes) {
                        Column::Cell(_, index) => Some(index),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                if indices
                    .iter()
                    .skip(1)
                    .all(|&id| same_leaf(m, indices[0], id))
                {
                    return Some((first, false));
                }
                let indices = build_call(m, "vector", &indices);
                build_call(m, "get", &[array, indices])
            }
            Column::Apply(op, call) => {
                let mut broadcast = call.clone();
                let mut operands: Vec<_> = arguments(call)
                    .map(|_| Vec::with_capacity(ids.len()))
                    .collect();
                for &id in ids {
                    let Node::Call(call) = m.node(id) else {
                        return None;
                    };
                    for (column, arg) in operands.iter_mut().zip(arguments(call)) {
                        column.push(arg);
                    }
                }
                let operands = operands
                    .iter()
                    .map(|column| self.stack(m, column, original_nodes))
                    .collect::<Option<Vec<_>>>()?;
                if operands.iter().all(|(_, varies)| !varies) {
                    return Some((first, false));
                }
                let mut args = vec![m.alloc(Node::Const(op))];
                for (index, (value, varies)) in operands.into_iter().enumerate() {
                    if !varies
                        && !(matches!(m.type_of(value), Some(Type::Scalar(_)))
                            || matches!(m.node(value), Node::Const(_)))
                    {
                        // A shared array would be iterated, and a shared record
                        // is not a broadcast operand.
                        return None;
                    }
                    if index < broadcast.args.len() {
                        args.push(value);
                    } else {
                        let arg = &mut broadcast.named[index - broadcast.args.len()];
                        arg.kind = NamedKind::Kwarg;
                        arg.value = value;
                    }
                }
                broadcast.head = CallHead::Builtin(m.intern("broadcast"));
                broadcast.args = args.into();
                m.alloc(Node::Call(broadcast))
            }
            Column::Constant(_) | Column::Whole(_) => return None,
        };
        Some((value, true))
    }
}

pub(super) fn sum_terms(m: &mut Module, terms: &[NodeId], original_nodes: usize) -> NodeId {
    if terms.len() < 2 {
        return fold_add(m, terms);
    }
    // The terms are not bound yet. Infer all their scalar expressions and
    // summary references in one traversal.
    let roots = build_call(m, "tuple", terms);
    flatppl_infer::annotate_rebuilt(m, roots);
    let mut shapes = Shapes::default();
    let mut positions = HashMap::new();
    let mut classes = Vec::<Vec<usize>>::new();
    for (position, &term) in terms.iter().enumerate() {
        let shape = shapes.classify(m, term, original_nodes);
        let index = *positions.entry(shape).or_insert_with(|| {
            classes.push(Vec::new());
            classes.len() - 1
        });
        classes[index].push(position);
    }
    let mut sums = terms.iter().copied().map(Some).collect::<Vec<_>>();
    for slots in classes.into_iter().filter(|slots| slots.len() >= 2) {
        let members = slots.iter().map(|&i| terms[i]).collect::<Vec<_>>();
        let mut stacks = Stacks::default();
        if m.type_of(members[0]) == Some(&Type::Scalar(ScalarType::Real))
            && let Some((vector, true)) = stacks.stack(m, &members, original_nodes)
        {
            let sum = build_call(m, "sum", &[vector]);
            flatppl_infer::annotate_rebuilt(m, sum);
            let groups = Dim::Static(members.len() as u32); // bounded by the observation extent
            if m.type_of(sum) == m.type_of(members[0])
                && stacks
                    .columns
                    .iter()
                    .all(|(ids, &(value, varies))| !varies || has_cells(m, value, ids[0], groups))
            {
                sums[slots[0]] = Some(sum);
                for &slot in &slots[1..] {
                    sums[slot] = None;
                }
            }
        }
    }
    fold_add(m, &sums.into_iter().flatten().collect::<Vec<_>>())
}
