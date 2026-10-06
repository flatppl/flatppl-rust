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
    let ranks: &[usize] = match member.as_str() {
        "normsys_factor" => &[0, 0, 0],
        "histosys_shift" => &[0, 0, 0, 0],
        "sample_yields" => &[2, 3, 3],
        "expected_counts" => &[2],
        _ => return None,
    };
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
    let mut shapes = Vec::with_capacity(args.len());
    for ((_, ty, _), (&rank, name)) in args.iter().zip(ranks.iter().zip(names)) {
        let shape = match cell_shape(ty, rank) {
            Ok(shape) => shape,
            Err(()) => {
                return Some(fail(
                    inf,
                    id,
                    &member,
                    &format!("`{name}` requires a real-valued rank-{rank} cell, got {ty:?}"),
                ));
            }
        };
        shapes.push(shape);
    }
    if member == "sample_yields" {
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            if known_mismatch(&shapes[left], 0, &shapes[right], 0) {
                return Some(fail(inf, id, &member, "unequal sample extents"));
            }
        }
        for (left, li, right, ri) in [(0, 1, 1, 2), (0, 1, 2, 2), (1, 2, 2, 2)] {
            if known_mismatch(&shapes[left], li, &shapes[right], ri) {
                return Some(fail(inf, id, &member, "unequal bin extents"));
            }
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

fn cell_shape(ty: &Type, rank: usize) -> Result<Option<&[Dim]>, ()> {
    match ty {
        Type::Any | Type::Deferred | Type::Failed(_) => Ok(None),
        Type::Scalar(ScalarType::Boolean | ScalarType::Integer | ScalarType::Real) if rank == 0 => {
            Ok(Some(&[]))
        }
        Type::Array { shape, elem } if shape.len() == rank => match elem.as_ref() {
            Type::Scalar(ScalarType::Boolean | ScalarType::Integer | ScalarType::Real) => {
                Ok(Some(shape))
            }
            Type::Any | Type::Deferred => Ok(None),
            _ => Err(()),
        },
        _ => Err(()),
    }
}

fn known_mismatch(a: &Option<&[Dim]>, ai: usize, b: &Option<&[Dim]>, bi: usize) -> bool {
    matches!((a.and_then(|s| s.get(ai)), b.and_then(|s| s.get(bi))),
        (Some(Dim::Static(a)), Some(Dim::Static(b))) if a != b)
}

fn fail(inf: &mut Inferencer<'_, '_>, id: NodeId, member: &str, reason: &str) -> Type {
    let message = format!("pyhf_helpers.{member}: {reason}");
    inf.diags
        .push(crate::Diagnostic::error_at(id, message.clone()));
    Type::Failed(message.into())
}
