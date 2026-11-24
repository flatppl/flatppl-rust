//! Cotangents for records with fixed-shape real leaves.

use std::collections::HashMap;

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{
    Call, CallHead, Dim, Module, NamedArg, NamedKind, Node, NodeId, Scalar, ScalarType, Symbol,
    Type,
};

use crate::{Emitter, Error, real, table, tensor};

type Fields = [(Symbol, Type)];

pub(super) fn is_differentiable(ty: &Type) -> bool {
    match ty {
        Type::Record(fields) => fields_are_differentiable(fields),
        Type::Tuple(items) => !items.is_empty() && items.iter().all(is_differentiable),
        Type::Table {
            nrows: Dim::Static(_),
            ..
        } => table::columns(ty).is_some_and(|columns| fields_are_differentiable(&columns)),
        _ => {
            tensor::real_shape_of(ty).is_some()
                || tensor::nested_vector_type(ty).is_some()
                || tensor::vector_of_arrays(ty).is_some()
        }
    }
}

fn fields_are_differentiable(fields: &Fields) -> bool {
    let mut has_tangent = false;
    for (_, ty) in fields {
        if is_differentiable(ty) {
            has_tangent = true;
        } else if !is_passive(ty) {
            return false;
        }
    }
    has_tangent
}

/// Integer and Boolean metadata remains in the primal ABI.
pub(super) fn is_passive(ty: &Type) -> bool {
    match ty {
        Type::Scalar(ScalarType::Integer | ScalarType::Boolean) => true,
        Type::Array { elem, .. } => is_passive(elem),
        Type::Record(fields)
        | Type::Table {
            columns: fields, ..
        } => !fields.is_empty() && fields.iter().all(|(_, ty)| is_passive(ty)),
        _ => false,
    }
}

pub(super) fn tangent_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Record(fields) if is_differentiable(ty) => Some(Type::Record(
            active_fields(fields)
                .map(|(name, ty)| (*name, tangent_type(ty).unwrap()))
                .collect(),
        )),
        Type::Table { columns, nrows } if is_differentiable(ty) => Some(Type::Table {
            columns: active_fields(columns)
                .map(|(name, ty)| (*name, tangent_type(ty).unwrap()))
                .collect(),
            nrows: *nrows,
        }),
        Type::Tuple(items) => {
            let mut items: Vec<_> = items.iter().filter_map(tangent_type).collect();
            match items.len() {
                0 => None,
                1 => items.pop(),
                _ => Some(Type::Tuple(items.into())),
            }
        }
        _ => is_differentiable(ty).then(|| ty.clone()),
    }
}

pub(super) fn active_fields(fields: &Fields) -> impl Iterator<Item = &(Symbol, Type)> {
    fields.iter().filter(|(_, ty)| is_differentiable(ty))
}

pub(super) fn is_access(module: &Module, args: &[NodeId]) -> bool {
    let Some((&input, selectors)) = args.split_first() else {
        return false;
    };
    let Some(ty @ (Type::Record(_) | Type::Table { .. })) = module.type_of(input) else {
        return false;
    };
    if !is_differentiable(ty) || selectors.is_empty() {
        return false;
    }
    let mut current = ty.clone();
    for (index, &selector) in selectors.iter().enumerate() {
        current = match &current {
            Type::Record(fields) => {
                let Some((_, field)) = selected_field(module, fields, selector) else {
                    return false;
                };
                field.clone()
            }
            Type::Table { columns, nrows } => {
                if let Some((_, cell)) = selected_field(module, columns, selector) {
                    table::column_type(cell, *nrows)
                } else if tensor::index_axes(module, selector) == Some(0) {
                    Type::Record(columns.clone())
                } else {
                    return false;
                }
            }
            _ => {
                return selectors[index..]
                    .iter()
                    .all(|&index| tensor::index_axes(module, index).is_some());
            }
        };
    }
    true
}

pub(super) fn access_path(module: &Module, args: &[NodeId]) -> Option<Vec<Symbol>> {
    let (&input, selectors) = args.split_first()?;
    let mut current @ Type::Record(_) = module.type_of(input)? else {
        return None;
    };
    if !is_differentiable(current) || selectors.is_empty() {
        return None;
    }
    selectors
        .iter()
        .map(|&selector| {
            let Type::Record(fields) = current else {
                return None;
            };
            let (name, field) = selected_field(module, fields, selector)?;
            current = field;
            Some(name)
        })
        .collect()
}

pub(super) fn fill(
    emitter: &mut Emitter,
    value: NodeId,
    primal: NodeId,
    fields: &Fields,
    head: &str,
) -> Option<NodeId> {
    let fields: Vec<_> = active_fields(fields)
        .map(|(name, ty)| {
            let primal = project(&mut emitter.module, primal, *name);
            Some((*name, tensor::fill_typed(emitter, value, primal, ty)?))
        })
        .collect::<Option<_>>()?;
    Some(build_named(&mut emitter.module, head, &fields))
}

pub(super) fn seed_set(emitter: &mut Emitter, primal: NodeId, fields: &Fields) -> Option<NodeId> {
    let named: Vec<_> = active_fields(fields)
        .map(|(name, ty)| {
            let primal = project(&mut emitter.module, primal, *name);
            Some(NamedArg {
                name: *name,
                kind: NamedKind::Field,
                value: tensor::seed_set(emitter, primal, ty)?,
            })
        })
        .collect::<Option<_>>()?;
    let head = emitter.module.intern("cartprod");
    Some(emitter.module.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([]),
        named: named.into(),
        inputs: None,
    })))
}

pub(super) fn add(
    emitter: &mut Emitter,
    fields: &Fields,
    records: &[NodeId],
    head: &str,
) -> NodeId {
    let terms: Vec<_> = active_fields(fields)
        .map(|(name, ty)| {
            let values: Vec<_> = records
                .iter()
                .map(|&record| project(&mut emitter.module, record, *name))
                .collect();
            (*name, tensor::add_typed(emitter, ty, &values))
        })
        .collect();
    build_named(&mut emitter.module, head, &terms)
}

pub(super) fn pick(
    emitter: &mut Emitter,
    primal: NodeId,
    adjoint: NodeId,
    slot: usize,
    head: &str,
) -> Option<NodeId> {
    let constructor = expect_builtin_call(&emitter.module, primal, head)?;
    let name = constructor.named.get(slot)?.name;
    Some(project(&mut emitter.module, adjoint, name))
}

pub(super) fn construct_tangent(
    emitter: &mut Emitter,
    primal: NodeId,
    members: &[NodeId],
    tangents: impl Iterator<Item = Option<NodeId>>,
    head: &str,
) -> Result<NodeId, Error> {
    let constructor = expect_builtin_call(&emitter.module, primal, head)
        .ok_or_else(|| Error::new(Some(primal), "missing record or table constructor"))?;
    let derivative_fields: Vec<_> = constructor
        .named
        .iter()
        .zip(members)
        .zip(tangents)
        .filter_map(|((field, &member), tangent)| {
            emitter
                .module
                .type_of(member)
                .is_some_and(is_differentiable)
                .then_some((field.name, member, tangent))
        })
        .collect();
    let values: Vec<_> = derivative_fields
        .into_iter()
        .map(|(name, member, tangent)| {
            let value = match tangent {
                Some(tangent) => tangent,
                None => emitter.zero_like(member)?,
            };
            Ok((name, value))
        })
        .collect::<Result<_, Error>>()?;
    Ok(build_named(&mut emitter.module, head, &values))
}

// A projection contributes only to the selected subtree. Materialize zeros
// once, when a pullback or the output ABI needs the complete record.
pub(super) enum Cotangent {
    Whole(NodeId),
    Fields(HashMap<Symbol, Cotangent>),
}

impl Cotangent {
    pub(super) fn through(self, selectors: &[Symbol]) -> Self {
        selectors.iter().rev().fold(self, |term, name| {
            Self::Fields(HashMap::from([(*name, term)]))
        })
    }

    fn value(&self) -> Option<NodeId> {
        match self {
            Self::Whole(value) => Some(*value),
            Self::Fields(_) => None,
        }
    }

    pub(super) fn materialize(
        self,
        emitter: &mut Emitter,
        primal: NodeId,
    ) -> Result<NodeId, Error> {
        if let Self::Whole(value) = self {
            return Ok(value);
        }
        let ty = emitter
            .module
            .type_of(primal)
            .cloned()
            .ok_or_else(|| Error::new(Some(primal), "missing type for a record cotangent"))?;
        self.materialize_typed(emitter, primal, &ty)
    }

    fn materialize_typed(
        self,
        emitter: &mut Emitter,
        primal: NodeId,
        ty: &Type,
    ) -> Result<NodeId, Error> {
        let mut contributions = match self {
            Self::Whole(value) => return Ok(value),
            Self::Fields(contributions) => contributions,
        };
        let Type::Record(fields) = ty else {
            unreachable!("record cotangent paths are validated on the primal");
        };
        let members = active_fields(fields)
            .map(|(name, ty)| {
                let child = project(&mut emitter.module, primal, *name);
                let value = match contributions.remove(name) {
                    Some(term) => term.materialize_typed(emitter, child, ty)?,
                    None => {
                        let zero = real(&mut emitter.module, 0.0);
                        tensor::fill_typed(emitter, zero, child, ty).ok_or_else(|| {
                            Error::new(Some(child), "cannot construct a shape-matched zero")
                        })?
                    }
                };
                Ok((*name, value))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(build_named(&mut emitter.module, "record", &members))
    }
}

pub(super) fn sum(
    emitter: &mut Emitter,
    primal: NodeId,
    mut terms: Vec<Cotangent>,
) -> Result<Cotangent, Error> {
    if terms.len() == 1 {
        return Ok(terms.pop().unwrap());
    }
    if let Some(values) = terms
        .iter()
        .map(Cotangent::value)
        .collect::<Option<Vec<_>>>()
    {
        return Ok(Cotangent::Whole(tensor::add_terms(
            emitter, primal, &values,
        )));
    }
    let ty = emitter
        .module
        .type_of(primal)
        .cloned()
        .ok_or_else(|| Error::new(Some(primal), "missing type for a record cotangent"))?;
    Ok(sum_typed(emitter, &ty, terms))
}

fn sum_typed(emitter: &mut Emitter, ty: &Type, mut terms: Vec<Cotangent>) -> Cotangent {
    if terms.len() == 1 {
        return terms.pop().unwrap();
    }
    if let Some(values) = terms
        .iter()
        .map(Cotangent::value)
        .collect::<Option<Vec<_>>>()
    {
        return Cotangent::Whole(tensor::add_typed(emitter, ty, &values));
    }
    let Type::Record(fields) = ty else {
        unreachable!("record cotangent paths are validated on the primal");
    };
    let mut by_field: HashMap<_, Vec<_>> = HashMap::new();
    // Preserve the order of live terms for every field. A whole-record term
    // contributes to every field.
    for term in terms {
        match term {
            Cotangent::Fields(terms) => {
                for (name, term) in terms {
                    by_field.entry(name).or_default().push(term);
                }
            }
            Cotangent::Whole(value) => {
                for (name, _) in active_fields(fields) {
                    let field = project(&mut emitter.module, value, *name);
                    by_field
                        .entry(*name)
                        .or_default()
                        .push(Cotangent::Whole(field));
                }
            }
        }
    }
    Cotangent::Fields(
        active_fields(fields)
            .filter_map(|(name, ty)| {
                let terms = by_field.remove(name)?;
                Some((*name, sum_typed(emitter, ty, terms)))
            })
            .collect(),
    )
}

pub(super) fn scatter(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    fields: &Fields,
    selectors: &[NodeId],
    base: i64,
    head: &str,
) -> Option<NodeId> {
    let (&selector, rest) = selectors.split_first()?;
    let (chosen, _) = selected_field(&emitter.module, fields, selector)?;
    let zero = real(&mut emitter.module, 0.0);
    let adjoints: Vec<_> = active_fields(fields)
        .map(|(name, ty)| {
            let primal = project(&mut emitter.module, primal, *name);
            let value = if *name == chosen {
                if rest.is_empty() {
                    bar
                } else {
                    tensor::scatter_typed(emitter, bar, primal, rest, base, ty)?
                }
            } else {
                tensor::fill_typed(emitter, zero, primal, ty)?
            };
            Some((*name, value))
        })
        .collect::<Option<_>>()?;
    Some(build_named(&mut emitter.module, head, &adjoints))
}

pub(super) fn selected_field<'a>(
    module: &Module,
    fields: &'a Fields,
    selector: NodeId,
) -> Option<(Symbol, &'a Type)> {
    let Node::Lit(Scalar::Str(name)) = module.node(selector) else {
        return None;
    };
    fields
        .iter()
        .find(|(symbol, _)| module.resolve(*symbol) == &**name)
        .map(|(name, ty)| (*name, ty))
}

pub(super) fn project(module: &mut Module, record: NodeId, name: Symbol) -> NodeId {
    if let Some(constructor) = expect_builtin_call(module, record, "record")
        .or_else(|| expect_builtin_call(module, record, "table"))
        && let Some(field) = constructor.named.iter().find(|field| field.name == name)
    {
        return field.value;
    }
    let key = module.alloc(Node::Lit(Scalar::Str(module.resolve(name).into())));
    build_call(module, "get", &[record, key])
}

pub(super) fn build_named(module: &mut Module, head: &str, fields: &[(Symbol, NodeId)]) -> NodeId {
    let head = module.intern(head);
    let named = fields
        .iter()
        .map(|&(name, value)| NamedArg {
            name,
            kind: NamedKind::Field,
            value,
        })
        .collect();
    module.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([]),
        named,
        inputs: None,
    }))
}
