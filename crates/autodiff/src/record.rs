//! Cotangents for records with fixed-shape real leaves.

use flatppl_core::graph::{build_call, build_record, expect_builtin_call, rebuild_with_children};
use flatppl_core::{
    Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Scalar, Symbol, Type,
};

use crate::{Emitter, real, tensor};

type Fields = [(Symbol, Type)];

pub(super) fn is_differentiable(ty: &Type) -> bool {
    match ty {
        Type::Record(fields) => {
            !fields.is_empty() && fields.iter().all(|(_, field)| is_differentiable(field))
        }
        _ => tensor::real_shape_of(ty).is_some() || tensor::nested_vector_type(ty).is_some(),
    }
}

pub(super) fn is_access(module: &Module, args: &[NodeId]) -> bool {
    let Some((&input, selectors)) = args.split_first() else {
        return false;
    };
    let Some(mut current @ Type::Record(_)) = module.type_of(input) else {
        return false;
    };
    if !is_differentiable(current) || selectors.is_empty() {
        return false;
    }
    for &selector in selectors {
        let Type::Record(fields) = current else {
            return false;
        };
        let Some((_, field)) = selected_field(module, fields, selector) else {
            return false;
        };
        current = field;
    }
    true
}

pub(super) fn fill(
    emitter: &mut Emitter,
    value: NodeId,
    primal: NodeId,
    fields: &Fields,
) -> Option<NodeId> {
    let fields: Vec<_> = fields
        .iter()
        .map(|(name, ty)| {
            let primal = project(&mut emitter.module, primal, *name);
            Some((*name, tensor::fill_typed(emitter, value, primal, ty)?))
        })
        .collect::<Option<_>>()?;
    Some(build_record(&mut emitter.module, &fields))
}

pub(super) fn seed_set(emitter: &mut Emitter, primal: NodeId, fields: &Fields) -> Option<NodeId> {
    let named: Vec<_> = fields
        .iter()
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

pub(super) fn add(emitter: &mut Emitter, fields: &Fields, records: &[NodeId]) -> NodeId {
    let terms: Vec<_> = fields
        .iter()
        .map(|(name, ty)| {
            let values: Vec<_> = records
                .iter()
                .map(|&record| project(&mut emitter.module, record, *name))
                .collect();
            (*name, tensor::add_typed(emitter, ty, &values))
        })
        .collect();
    build_record(&mut emitter.module, &terms)
}

pub(super) fn pick(
    emitter: &mut Emitter,
    primal: NodeId,
    adjoint: NodeId,
    slot: usize,
) -> Option<NodeId> {
    let constructor = expect_builtin_call(&emitter.module, primal, "record")?;
    let name = constructor.named.get(slot)?.name;
    Some(project(&mut emitter.module, adjoint, name))
}

pub(super) fn embed(
    emitter: &mut Emitter,
    primal: NodeId,
    tangent: NodeId,
    slot: usize,
) -> Option<NodeId> {
    let constructor = expect_builtin_call(&emitter.module, primal, "record")?;
    let members: Vec<_> = constructor.named.iter().map(|field| field.value).collect();
    let values: Vec<_> = members
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if index == slot {
                Some(tangent)
            } else {
                emitter.zero_like(value).ok()
            }
        })
        .collect::<Option<_>>()?;
    Some(rebuild_with_children(&mut emitter.module, primal, &values))
}

pub(super) fn scatter(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    fields: &Fields,
    selectors: &[NodeId],
) -> Option<NodeId> {
    let (&selector, rest) = selectors.split_first()?;
    let (chosen, _) = selected_field(&emitter.module, fields, selector)?;
    let zero = real(&mut emitter.module, 0.0);
    let adjoints: Vec<_> = fields
        .iter()
        .map(|(name, ty)| {
            let primal = project(&mut emitter.module, primal, *name);
            let value = if *name == chosen {
                if rest.is_empty() {
                    bar
                } else if let Type::Record(fields) = ty {
                    scatter(emitter, bar, primal, fields, rest)?
                } else {
                    return None;
                }
            } else {
                tensor::fill_typed(emitter, zero, primal, ty)?
            };
            Some((*name, value))
        })
        .collect::<Option<_>>()?;
    Some(build_record(&mut emitter.module, &adjoints))
}

fn selected_field<'a>(
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
        && let Some(field) = constructor.named.iter().find(|field| field.name == name)
    {
        return field.value;
    }
    let key = module.alloc(Node::Lit(Scalar::Str(module.resolve(name).into())));
    build_call(module, "get", &[record, key])
}
