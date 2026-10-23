//! Spec §04 "Additional arguments" for the lowering rules: the values bound to a
//! callback's remaining inputs, read from the call that carries them.
//!
//! `weighted(f, M, ...)`, `pushfwd(f, M, ...)`, `markovchain(K, init, n, ...)` and
//! `kscan(K, init, ...seqs)` call their callback with leading arguments of their own,
//! then the additional arguments `...`. Inference types the call through
//! [`flatppl_infer::extras::bind`]; this module binds with the same function, so the
//! lowering reads the arguments exactly as inference did.

use flatppl_core::{Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Symbol, Type};
use flatppl_infer::extras::{Error, Remaining, Source, bind};

use crate::density::{build_user_call_n, refuse};
use crate::kernel::{is_splattable, record_field, resolve_reified, splat_fields};
use crate::refuse::RefuseError;

/// A callback's input names and whether they accept positional arguments.
pub(crate) struct Interface {
    pub names: Vec<Symbol>,
    pub ordered: bool,
}

/// The inputs of a reified callback, or of a callback known only by its type.
pub(crate) fn interface(m: &Module, callback: NodeId) -> Option<Interface> {
    if let Some(kernel) = resolve_reified(m, callback) {
        return Some(Interface {
            names: kernel.inputs.iter().map(|(name, _)| *name).collect(),
            ordered: !kernel.auto,
        });
    }
    match m.type_of(callback)? {
        Type::Function { inputs } | Type::Kernel { inputs, .. } => Some(Interface {
            names: inputs.to_vec(),
            ordered: true,
        }),
        _ => None,
    }
}

/// The additional arguments of `node`, which follow its first `lead` positional
/// arguments, bound to the inputs of `callback` after the first `callback_lead`.
/// The result lists one value per remaining input, in declaration order.
///
/// A callback with unknown inputs keeps its positional arguments as written.
pub(crate) fn bind_extras(
    m: &mut Module,
    node: NodeId,
    callback: NodeId,
    lead: usize,
    callback_lead: usize,
) -> Result<Vec<NodeId>, RefuseError> {
    let (positional, named) = {
        let Node::Call(c) = m.node(node) else {
            return Err(refuse(node, m, "expected a call"));
        };
        if c.args.len() < lead {
            return Err(refuse(node, m, "too few arguments"));
        }
        (c.args[lead..].to_vec(), c.named.to_vec())
    };
    if positional.is_empty() && named.is_empty() {
        return Ok(Vec::new());
    }
    let Some(interface) = interface(m, callback) else {
        if named.is_empty() {
            return Ok(positional);
        }
        return Err(refuse(
            node,
            m,
            "the callback's inputs are not known, so keyword additional arguments cannot bind",
        ));
    };
    if interface.names.len() < callback_lead {
        return Err(refuse(
            node,
            m,
            "the callback has fewer inputs than the operation supplies",
        ));
    }
    let remaining_names = &interface.names[callback_lead..];
    let remaining: Vec<String> = remaining_names
        .iter()
        .map(|name| m.resolve(*name).to_string())
        .collect();
    let remaining_view = Remaining {
        names: &remaining,
        ordered: interface.ordered,
    };
    let aggregate: Option<(NodeId, Vec<Symbol>)> = match positional.as_slice() {
        [only] if named.is_empty() && is_splattable(m, *only) => {
            splat_fields(m, *only).map(|fields| (*only, fields))
        }
        _ => None,
    };
    let aggregate_names: Option<Vec<String>> = aggregate.as_ref().map(|(_, fields)| {
        fields
            .iter()
            .map(|name| m.resolve(*name).to_string())
            .collect()
    });
    let named_names: Vec<String> = named
        .iter()
        .map(|a| m.resolve(a.name).to_string())
        .collect();
    let named_refs: Vec<&str> = named_names.iter().map(String::as_str).collect();
    let sources = bind(
        &remaining_view,
        positional.len(),
        &named_refs,
        aggregate_names.as_deref(),
    )
    .map_err(|error| {
        let what = match error {
            Error::KeywordOnly => {
                "the callback has keyword-only inputs, so additional arguments bind by name"
                    .to_string()
            }
            Error::TooMany { given, inputs } => format!(
                "{given} additional argument(s) but the callback has {inputs} remaining input(s)"
            ),
            Error::Unknown(n) => format!("the callback has no remaining input `{n}`"),
            Error::Twice(n) => format!("input `{n}` is bound twice"),
            Error::Unbound(n) => format!("input `{n}` is left unbound"),
        };
        refuse(node, m, &format!("additional arguments: {what} (spec §04)"))
    })?;
    let mut values = Vec::with_capacity(sources.len());
    for (slot, source) in sources.iter().enumerate() {
        values.push(match source {
            Source::Positional(k) => positional[*k],
            Source::Named(j) => named[*j].value,
            Source::Field(_) => {
                let (record, _) = aggregate
                    .as_ref()
                    .expect("a field source comes from the aggregate");
                record_field(m, *record, remaining_names[slot]).ok_or_else(|| {
                    refuse(
                        node,
                        m,
                        "the additional record or table field is not readable",
                    )
                })?
            }
        });
    }
    Ok(values)
}

/// Apply `callback` to `first`, then to the bound `extras`. An ordered callback
/// takes positional arguments. A keyword-only callback takes every argument by
/// name, and so does a unary callback, which keeps a whole record or table
/// variate from auto-splatting.
pub(crate) fn apply_callback(
    m: &mut Module,
    callback: NodeId,
    first: NodeId,
    extras: &[NodeId],
) -> Result<NodeId, RefuseError> {
    let interface = interface(m, callback);
    if let Some(interface) = &interface
        && interface.names.len() != 1 + extras.len()
    {
        return Err(refuse(
            callback,
            m,
            "callback inputs do not match the variate and the additional arguments",
        ));
    }
    let keyword_only = interface.as_ref().is_some_and(|i| !i.ordered);
    if !extras.is_empty() && !keyword_only {
        let mut args = vec![first];
        args.extend_from_slice(extras);
        return Ok(build_user_call_n(m, callback, args));
    }
    if let Some(interface) = interface {
        let named: Vec<NamedArg> = std::iter::once(first)
            .chain(extras.iter().copied())
            .zip(&interface.names)
            .map(|(value, name)| NamedArg {
                kind: NamedKind::Kwarg,
                name: *name,
                value,
            })
            .collect();
        return Ok(m.alloc(Node::Call(Call {
            head: CallHead::User(callback),
            args: Box::new([]),
            named: named.into(),
            inputs: None,
        })));
    }
    let mut args = vec![first];
    args.extend_from_slice(extras);
    Ok(build_user_call_n(m, callback, args))
}
