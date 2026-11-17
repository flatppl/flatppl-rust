//! Resolve primitive constructor names and bind their arguments by catalogue order.

use flatppl_core::calling::{bind_call_args, is_splattable};
use flatppl_core::graph::resolve_ref;
use flatppl_core::{CallHead, Module, NamedKind, Node, NodeId, Ref, RefNs, Scalar, Symbol};

/// Standard-module members are catalogue entries, not file-module graphs.
pub fn std_module_member(host: &Module, id: NodeId) -> Option<(String, String)> {
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }) = *host.node(id)
    else {
        return None;
    };
    let bid = host.alias_target(alias)?;
    let Node::Call(c) = host.node(host.binding(bid).rhs) else {
        return None;
    };
    if !matches!(c.head, CallHead::Builtin(s) if host.resolve(s) == "standard_module") {
        return None;
    }
    let Node::Lit(Scalar::Str(module)) = host.node(*c.args.first()?) else {
        return None;
    };
    Some((module.to_string(), host.resolve(name).to_string()))
}

/// The bare constructor symbol of a §09 standard-module DISTRIBUTION member
/// reached as a call callee: `hep.CrystalBall` → `CrystalBall`. Bare-name aliases
/// are followed, so `sig = hep.CrystalBall; sig(m0 = …)` resolves too.
///
/// `None` for anything else — a `load_module` member (which grafts instead), a
/// local callee, or a §09 member whose catalogue row is a *function* rather than a
/// distribution. §09 "Standard modules" gives no unqualified spelling for a
/// member, so the bare name cannot collide with a base §08 constructor: the
/// catalogue keeps the two namespaces disjoint (`Catalogue::is_module_member`).
pub fn std_module_ctor_sym(m: &Module, callee: NodeId) -> Option<Symbol> {
    let ref_id = if std_module_member(m, callee).is_some() {
        callee
    } else {
        let hop = resolve_ref(m, callee).0;
        std_module_member(m, hop).map(|_| hop)?
    };
    let Node::Ref(Ref { name: member, .. }) = *m.node(ref_id) else {
        return None;
    };
    crate::distribution_param_names(m.resolve(member))?;
    Some(member)
}

/// Return the builtin call head or the bare name of a standard-module distribution.
pub fn constructor_symbol(m: &Module, node: NodeId) -> Option<Symbol> {
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    match c.head {
        CallHead::Builtin(sym) => Some(sym),
        CallHead::User(callee) => std_module_ctor_sym(m, callee),
    }
}

/// Bind positional arguments in catalogue order and keep named arguments in source order.
/// A sole record or table may supply the fields of a multi-parameter constructor.
/// Projecting an opaque argument can allocate field accesses in the module.
/// A keyword-only builtin needs no catalogue lookup. The caller selects its supported heads.
pub fn split_kernel_constructor(
    m: &mut Module,
    node: NodeId,
) -> Option<(Symbol, Vec<(Symbol, NodeId)>)> {
    let Node::Call(c) = m.node(node) else {
        return None;
    };
    let sym = constructor_symbol(m, node)?;
    // Snapshot positional args and keyword args (NodeId/Symbol are Copy) so the
    // `&Call` borrow ends before interning parameter names below needs `&mut`.
    let pos_args: Vec<NodeId> = c.args.to_vec();
    let mut kwargs = Vec::with_capacity(c.named.len());
    for n in c.named.iter() {
        if n.kind != NamedKind::Kwarg {
            return None;
        }
        kwargs.push((n.name, n.value));
    }

    // Keyword-only form (the common case): return the named args verbatim.
    if pos_args.is_empty() {
        return Some((sym, kwargs));
    }

    // Positional / mixed form: bind positional args to the constructor's ordered
    // parameter names. A head that is not a known distribution constructor, or
    // more positional args than it has parameters, is not a well-formed
    // positional constructor call → None.
    let param_names = crate::constructor_param_names(m.resolve(sym))?;

    // Constructor domains do not decide whether the argument is spread.
    if param_names.len() != 1
        && kwargs.is_empty()
        && pos_args.len() == 1
        && is_splattable(m, pos_args[0])
    {
        let names: Vec<_> = param_names.iter().map(|name| m.intern(name)).collect();
        let values = bind_call_args(m, &names, &pos_args, &[], false)?;
        return Some((sym, names.into_iter().zip(values).collect()));
    }

    if pos_args.len() > param_names.len() {
        return None;
    }
    let mut args: Vec<(Symbol, NodeId)> = Vec::with_capacity(pos_args.len() + kwargs.len());
    for (i, &arg) in pos_args.iter().enumerate() {
        args.push((m.intern(&param_names[i]), arg));
    }
    // A parameter bound BOTH positionally and by keyword is a §04 double-bind
    // (static error) — refuse rather than emit a record with duplicate fields.
    if kwargs
        .iter()
        .any(|(kw_name, _)| args.iter().any(|(pos_name, _)| pos_name == kw_name))
    {
        return None;
    }
    args.extend(kwargs);
    Some((sym, args))
}
