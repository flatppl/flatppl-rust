//! Bind graph arguments using the §04 calling conventions.

use crate::graph::{build_call, resolve_ref};
use crate::{
    CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, Scalar, Symbol, Type,
};

/// Ordered label/target pairs for an explicit or inferred reification boundary.
pub fn reified_inputs<'a>(
    m: &'a Module,
    node: NodeId,
    inputs: &'a Inputs,
) -> Option<&'a [(Symbol, Ref)]> {
    match inputs {
        Inputs::Spec(entries) => Some(entries),
        Inputs::Auto => m.auto_inputs_of(node),
    }
}

/// §04 calling convention: leading positionals, then keywords, or one record/table
/// splatted by name. Every input must bind exactly once before any substitution.
pub fn bind_call_args(
    m: &mut Module,
    names: &[Symbol],
    args: &[NodeId],
    named: &[NamedArg],
    keyword_only: bool,
) -> Option<Vec<NodeId>> {
    bind_optional_call_args(m, names, names.len(), args, named, keyword_only)
}

/// Bind a catalogue interface with an optional trailing parameter suffix.
pub fn bind_optional_call_args(
    m: &mut Module,
    names: &[Symbol],
    required: usize,
    args: &[NodeId],
    named: &[NamedArg],
    keyword_only: bool,
) -> Option<Vec<NodeId>> {
    // A sole tuple argument of an ordered callable that does not accept a lone
    // positional argument splats into positional arguments (spec §04). A count
    // outside `required..=names.len()` is a static error, refused here as it is
    // for a record.
    if !keyword_only
        && required > 1
        && named.is_empty()
        && args.len() == 1
        && let Some(count) = tuple_arity(m, args[0])
    {
        if count < required || count > names.len() {
            return None;
        }
        return (0..count).map(|i| tuple_element(m, args[0], i)).collect();
    }
    if (keyword_only || required > 1 || names.is_empty())
        && named.is_empty()
        && args.len() == 1
        && is_splattable(m, args[0])
    {
        let fields = splat_fields(m, args[0])?;
        if fields.len() < required || !fields.iter().all(|name| names.contains(name)) {
            return None;
        }
        return names
            .get(..fields.len())?
            .iter()
            .map(|name| record_field(m, args[0], *name))
            .collect();
    }
    let count = args.len() + named.len();
    if count < required || count > names.len() || (keyword_only && !args.is_empty()) {
        return None;
    }
    let mut values = vec![None; names.len()];
    for (slot, arg) in values.iter_mut().zip(args) {
        *slot = Some(*arg);
    }
    for arg in named {
        if arg.kind != NamedKind::Kwarg {
            return None;
        }
        let index = names.iter().position(|name| *name == arg.name)?;
        if values[index].replace(arg.value).is_some() {
            return None;
        }
    }
    values.into_iter().take(count).collect()
}

/// The element count of a tuple value: a written `tuple(…)` literal, or a value
/// whose inferred type is a tuple.
fn tuple_arity(m: &Module, value: NodeId) -> Option<usize> {
    let (resolved, _) = resolve_ref(m, value);
    if matches!(splat_head(m, value), Some("tuple"))
        && let Node::Call(c) = m.node(resolved)
        && c.named.is_empty()
    {
        return Some(c.args.len());
    }
    match m.type_of(value).or_else(|| m.type_of(resolved))? {
        Type::Tuple(parts) => Some(parts.len()),
        _ => None,
    }
}

/// Element `index` (0-based) of a tuple value. A written `tuple(…)` literal gives
/// its element; any other value gives the projection `get(t, index + 1)`, the
/// positional argument spec §11 resolves a splatted tuple to.
fn tuple_element(m: &mut Module, value: NodeId, index: usize) -> Option<NodeId> {
    let (resolved, _) = resolve_ref(m, value);
    if matches!(splat_head(m, value), Some("tuple"))
        && let Node::Call(c) = m.node(resolved)
        && c.named.is_empty()
    {
        return c.args.get(index).copied();
    }
    let position = m.alloc(Node::Lit(Scalar::Int(i64::try_from(index).ok()? + 1)));
    Some(build_call(m, "get", &[value, position]))
}

/// Whether the value exposes record fields or table columns for named binding.
pub fn is_splattable(m: &Module, rec: NodeId) -> bool {
    splat_fields(m, rec).is_some()
}

/// Field names come from the literal or its inferred record/table type. Opaque
/// inputs and loaded tables obey the same splat rule without reading their values.
pub fn splat_fields(m: &Module, rec: NodeId) -> Option<Vec<Symbol>> {
    let (resolved, _) = resolve_ref(m, rec);
    if matches!(splat_head(m, rec), Some("record") | Some("table")) {
        let Node::Call(c) = m.node(resolved) else {
            return None;
        };
        return Some(c.named.iter().map(|arg| arg.name).collect());
    }
    let ty = m.type_of(rec).or_else(|| m.type_of(resolved))?;
    match ty {
        Type::Table { columns, .. } | Type::Record(columns) => {
            Some(columns.iter().map(|(n, _)| *n).collect())
        }
        _ => None,
    }
}

/// The `record`/`table` head of `rec` after its alias chain, or `None`.
fn splat_head(m: &Module, rec: NodeId) -> Option<&str> {
    let (resolved, _) = resolve_ref(m, rec);
    let Node::Call(c) = m.node(resolved) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    Some(m.resolve(sym))
}

/// The value §04's splat binds to `name`: field/column `name` of a `record(%field … )` or
/// `table(%field … )` literal, or — for an OPAQUE table — a synthesized column access.
/// `None` if `rec` is neither, or lacks the name.
///
/// `None` on a missing name is what makes a mismatch REFUSE rather than fall back to
/// whole-value binding.
///
/// **The opaque case.** A `load_data` result has no syntactic columns, so D6 could not
/// destructure it and the reduction refused — which left a §04-legal model
/// (`g(t)` against a `g` whose parameters are `t`'s column names) refusing with the generic
/// "residual user call", while its `table(...)` literal twin lowered. The columns are
/// statically known from the type ([`splat_fields`]), and §03 "Tables" already spells the
/// per-column value: "Column access by name: `t.a` … returns the column with that name as a
/// vector", which the parser lowers to `get(t, "a")`. So the splat IS constructible — it is
/// the same set of column values the literal case binds, reached through `get` instead of
/// through the literal's `%field` list. Synthesizing it makes the two forms agree, which is
/// what §04 requires: "whether its field or column names match the callable's argument names
/// decides only whether the call is valid, never whether the splat occurs."
pub fn record_field(m: &mut Module, rec: NodeId, name: Symbol) -> Option<NodeId> {
    if matches!(splat_head(m, rec), Some("record") | Some("table")) {
        let (resolved, _) = resolve_ref(m, rec);
        let Node::Call(c) = m.node(resolved) else {
            return None;
        };
        return c.named.iter().find(|na| na.name == name).map(|na| na.value);
    }
    // Opaque record/table: bind the field/column access declared by its type.
    //
    // A multi-column splat that refuses on a LATER column leaves the `get` nodes already
    // allocated for the earlier ones unreferenced. Benign: the arena is append-only and never
    // freed per node (`crates/core/src/id.rs`), nothing reaches them, and inference rejects a
    // name mismatch before the determiniser runs anyway — so the partial case is only reachable
    // when some other layer has already failed.
    if !splat_fields(m, rec)?.contains(&name) {
        return None;
    }
    let key = m.alloc(Node::Lit(Scalar::Str(m.resolve(name).into())));
    Some(build_call(m, "get", &[rec, key]))
}
