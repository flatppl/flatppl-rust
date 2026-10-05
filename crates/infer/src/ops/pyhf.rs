//! Shape contracts for resolved pyhf helpers, before their portable expansion.

use super::*;
use flatppl_core::Module;

pub(super) fn call_type(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
    callee: NodeId,
    args: &[ArgInfo],
    named: &[NamedInfo],
    broadcast: bool,
) -> Option<Type> {
    let callee = resolve_binding_refs(inf, callee);
    let origin = inf.module_catalogue_ref(callee)?;
    if origin.module != "pyhf_helpers" {
        return None;
    }
    let member = origin.member.clone();
    let sig = origin.sig.clone();
    if let Some(failed) = module_member_arity_check(inf, id, callee, &sig, args, named) {
        return Some(failed);
    }
    let names = crate::catalogue::sig_param_names(&sig)?;
    let Some(mut args) = ordered_args(inf.module, names, args, named) else {
        return Some(Type::Deferred);
    };
    if let Some((_, failed @ Type::Failed(_), _)) =
        args.iter().find(|(_, ty, _)| matches!(ty, Type::Failed(_)))
    {
        return Some(failed.clone());
    }
    let mut batch: Option<Box<[Dim]>> = None;
    if broadcast {
        for (_, ty, _) in &mut args {
            let (shape, cell) = match ty {
                Type::Array { shape, elem } => (shape.clone(), (**elem).clone()),
                Type::Table { columns, nrows } => {
                    (vec![*nrows].into(), Type::Record(columns.clone()))
                }
                _ => continue,
            };
            batch = Some(match batch {
                None => shape,
                Some(previous) => match broadcast_shape_join(&previous, &shape) {
                    Some(shape) => shape,
                    None => return Some(fail(inf, id, &member, "incompatible broadcast shapes")),
                },
            });
            *ty = cell;
        }
    }
    for ((_, ty, _), name) in args.iter().zip(names) {
        if !matches!(
            ty,
            Type::Any
                | Type::Deferred
                | Type::Scalar(ScalarType::Boolean | ScalarType::Integer | ScalarType::Real)
        ) {
            return Some(fail(
                inf,
                id,
                &member,
                &format!("`{name}` requires a real scalar, got {ty:?}"),
            ));
        }
    }
    let cell = catalogue_lower(inf.module, &sig, &args).0;
    Some(match batch {
        Some(shape) => Type::Array {
            shape,
            elem: Box::new(cell),
        },
        None => cell,
    })
}

/// Field types retain their aggregate node for diagnostics, as in builtin
/// keyword normalization. A table contributes columns, not row scalar types.
fn ordered_args(
    module: &Module,
    names: &[String],
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Vec<ArgInfo>> {
    let mut slots = vec![None; names.len()];
    let mut put = |name: Symbol, value: ArgInfo| {
        let index = names.iter().position(|n| n == module.resolve(name))?;
        if slots[index].is_some() {
            return None;
        }
        slots[index] = Some(value);
        Some(())
    };
    if named.is_empty()
        && let [(node, ty, phase)] = args
    {
        match ty {
            Type::Record(fields) => {
                for (name, ty) in fields {
                    put(*name, (*node, ty.clone(), *phase))?;
                }
                return slots.into_iter().collect();
            }
            Type::Table { columns, nrows } => {
                for (name, ty) in columns {
                    put(
                        *name,
                        (
                            *node,
                            Type::Array {
                                shape: vec![*nrows].into(),
                                elem: Box::new(ty.clone()),
                            },
                            *phase,
                        ),
                    )?;
                }
                return slots.into_iter().collect();
            }
            _ => {}
        }
    }
    if args.len() > names.len() {
        return None;
    }
    for (slot, value) in slots.iter_mut().zip(args) {
        *slot = Some(value.clone());
    }
    for (name, node, ty, phase) in named {
        let index = names.iter().position(|n| n == module.resolve(*name))?;
        if slots[index].is_some() {
            return None;
        }
        slots[index] = Some((*node, ty.clone(), *phase));
    }
    slots.into_iter().collect()
}

fn fail(inf: &mut Inferencer<'_, '_>, id: NodeId, member: &str, reason: &str) -> Type {
    let message = format!("pyhf_helpers.{member}: {reason}");
    inf.diags
        .push(crate::Diagnostic::error_at(id, message.clone()));
    Type::Failed(message.into())
}
