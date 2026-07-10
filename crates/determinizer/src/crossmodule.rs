//! Resolve a `(%ref <loaded-module> member)` measure ref into the loaded
//! submodule graph carried by the [`ModuleBundle`], then graft that subtree into
//! the host module so the existing (bundle-free) lowering can proceed on a
//! self-contained node.
//!
//! Spec §04 "Reification and module scope": a *measure* crosses module
//! boundaries freely (`lawof(draw(m)) ≡ m`), so resolving a cross-module measure
//! ref is spec-legal. (Taking a cross-module *parameterized* value AS a
//! reification input is the disallowed case — a static error — but that is a
//! reification built in the host, not a reference to a submodule's own reified
//! kernel, which is what we resolve here.)
//!
//! ## Why graft rather than thread the bundle everywhere
//!
//! The referenced node lives in the *submodule's* arena and interner; the
//! density lowering mutates (and interns into) the *host*. Rather than thread
//! `&ModuleBundle` — and a `&submodule` — through the entire `lower_measure_*`
//! recursion, we deep-copy the referenced subtree into the host once, at the
//! resolution site (the likelihood-kernel resolution in `density.rs`),
//! re-interning every symbol and recursively grafting the submodule bindings it
//! closes over. The result is an ordinary host-local node; everything downstream
//! runs unchanged and needs no bundle.
//!
//! ## Load-time substitution and namespace independence (refuse-don't-mislower)
//!
//! Two host-vs-submodule interactions matter for correctness (§04 "Multi-file
//! models"):
//!
//! * **`load_module` `%assign` parameters.** `load_module("h", center = <expr>)`
//!   substitutes the submodule's input parameter `center` with the host `<expr>`
//!   at the load boundary. The graft honors this: a `(%ref self center)` in the
//!   grafted subtree naming a substituted parameter is replaced by the host
//!   expression (not by a fresh copy of the submodule's own `center` binding).
//!
//! * **Independent namespaces.** A submodule binding and a same-named host
//!   binding are unrelated unless the `%assign` above links them. So a grafted
//!   submodule dependency whose name collides with an unrelated pre-existing host
//!   binding must NOT reuse the host binding (that would silently score against
//!   the wrong value) — the graft **refuses** instead.

use std::collections::HashSet;

use flatppl_core::{
    Axis, Binding, Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs,
    Scalar, Symbol,
};
use flatppl_infer::ModuleBundle;

/// A resolved cross-module kernel reference: the submodule that owns it, the
/// member's node id in that submodule, and the host's load-time `%assign`
/// substitutions (submodule-parameter name → host expression node id).
pub(crate) struct ResolvedRef<'a> {
    pub sub: &'a Module,
    pub member_rhs: NodeId,
    /// `(submodule-parameter-name, host-expr-node)` for each `%assign` on the
    /// `load_module` call. The node ids are HOST nodes (already interned).
    pub assign: Vec<(String, NodeId)>,
}

/// For a `Node::Ref { ns: RefNs::Module(alias), name: member }` at `id` in
/// `host`, follow `alias` → its `load_module("path", …)` binding → `path` →
/// `bundle.get(path)` → the submodule's `member` binding, and return the
/// submodule, the member rhs, and the `load_module` `%assign` substitutions.
///
/// `None` (⇒ the caller refuses, per refuse-don't-mislower) when `id` is not a
/// module ref, the alias is not a `load_module` binding, the path dependency is
/// absent from the bundle, or the submodule has no such (member) binding.
pub(crate) fn resolve_module_ref<'a>(
    bundle: &'a ModuleBundle,
    host: &Module,
    id: NodeId,
) -> Option<ResolvedRef<'a>> {
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name: member,
    }) = *host.node(id)
    else {
        return None;
    };
    // `alias` names a host binding whose rhs is `load_module("path", …)`.
    let bid = host.binding_by_name(alias)?;
    let load = host.binding(bid).rhs;
    let Node::Call(c) = host.node(load) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if host.resolve(sym) != "load_module" {
        return None;
    }
    // First positional arg is the path string literal.
    let path = string_literal(host, *c.args.first()?)?;
    // Load-time `%assign` substitutions: `load_module("p", center = <host expr>)`
    // binds the submodule input parameter `center` to `<host expr>`. Keyed by the
    // submodule-parameter name (string, so it matches across interners); the value
    // is a host node id.
    let assign: Vec<(String, NodeId)> = c
        .named
        .iter()
        .filter(|na| na.kind == NamedKind::Assign)
        .map(|na| (host.resolve(na.name).to_string(), na.value))
        .collect();
    let sub = bundle.get(&path)?;
    // Cross-interner lookup: match the member by string, not by Symbol.
    let member_name = host.resolve(member);
    let member_rhs = sub
        .bindings()
        .find(|(_, b)| sub.resolve(b.name) == member_name)
        .map(|(_, b)| b.rhs)?;
    Some(ResolvedRef {
        sub,
        member_rhs,
        assign,
    })
}

/// The `Box<str>` payload of a `Node::Lit(Scalar::Str(_))`, else `None`.
fn string_literal(m: &Module, id: NodeId) -> Option<String> {
    match m.node(id) {
        Node::Lit(Scalar::Str(s)) => Some(s.to_string()),
        _ => None,
    }
}

/// State threaded through the recursive graft.
struct GraftCtx<'a> {
    /// Load-time `%assign` substitutions (submodule-parameter name → host node).
    assign: &'a [(String, NodeId)],
    /// Names of host bindings that existed BEFORE the graft began. A submodule
    /// dependency whose name collides with one of these (and is not `%assign`-
    /// linked) is an unrelated host binding: refuse rather than reuse it.
    preexisting: HashSet<String>,
    /// Submodule-binding names grafted so far this pass (cycle/repeat guard).
    grafted: HashSet<String>,
}

/// Deep-copy the subtree rooted at `root` from `src` into `host`, re-interning
/// every symbol through the host interner and returning the new host `NodeId`.
///
/// A `(%ref self <name>)` in the copied subtree names a *submodule* binding.
/// Handling depends on what `<name>` is:
///
/// * a load-time `%assign` parameter (`load_module("h", <name> = <host expr>)`) —
///   the reference is replaced by the host `<expr>` (load-time substitution);
/// * an ordinary submodule binding whose name does NOT collide with a
///   pre-existing host binding — that binding is itself grafted into the host
///   (recursively), keeping the grafted node self-contained;
/// * an ordinary submodule binding whose name DOES collide with an unrelated
///   pre-existing host binding — **refuse** (`Err`): modules are independent
///   namespaces, so reusing the host binding would score against the wrong value.
///
/// Placeholder (`%local`) refs and boundary-input entries are re-interned in
/// place; a nested cross-module (`%ref <alias> …`) ref is re-interned but left
/// unresolved (there is no bundle here — an unresolved nested module ref simply
/// refuses downstream, never mislowers).
pub(crate) fn graft_subtree(
    host: &mut Module,
    resolved: &ResolvedRef<'_>,
) -> Result<NodeId, String> {
    let preexisting: HashSet<String> = host
        .bindings()
        .map(|(_, b)| host.resolve(b.name).to_string())
        .collect();
    let mut ctx = GraftCtx {
        assign: &resolved.assign,
        preexisting,
        grafted: HashSet::new(),
    };
    graft_node(host, resolved.sub, resolved.member_rhs, &mut ctx)
}

fn graft_node(
    host: &mut Module,
    src: &Module,
    id: NodeId,
    ctx: &mut GraftCtx<'_>,
) -> Result<NodeId, String> {
    // Clone the source node so the `src` borrow ends before we mutate `host`.
    let node = src.node(id).clone();
    let out = match node {
        Node::Lit(s) => host.alloc(Node::Lit(s)),
        Node::Hole => host.alloc(Node::Hole),
        Node::Const(sym) => {
            let hsym = host.intern(src.resolve(sym));
            host.alloc(Node::Const(hsym))
        }
        Node::Axis(Axis { name, variance }) => {
            let hname = host.intern(src.resolve(name));
            host.alloc(Node::Axis(Axis {
                name: hname,
                variance,
            }))
        }
        Node::Ref(r) => {
            // Load-time `%assign` substitution: a `(%ref self <param>)` naming a
            // substituted submodule parameter is replaced by the host expression
            // it was bound to at `load_module`. The host node is already interned.
            if r.ns == RefNs::SelfMod {
                if let Some((_, host_expr)) = ctx
                    .assign
                    .iter()
                    .find(|(n, _)| n.as_str() == src.resolve(r.name))
                {
                    return Ok(*host_expr);
                }
            }
            let hr = graft_ref(host, src, r, ctx)?;
            host.alloc(Node::Ref(hr))
        }
        Node::Call(c) => {
            let head = match c.head {
                CallHead::Builtin(sym) => CallHead::Builtin(host.intern(src.resolve(sym))),
                CallHead::User(callee) => CallHead::User(graft_node(host, src, callee, ctx)?),
            };
            let mut args: Vec<NodeId> = Vec::with_capacity(c.args.len());
            for &a in c.args.iter() {
                args.push(graft_node(host, src, a, ctx)?);
            }
            let mut named: Vec<NamedArg> = Vec::with_capacity(c.named.len());
            for na in c.named.iter() {
                named.push(NamedArg {
                    kind: na.kind,
                    name: host.intern(src.resolve(na.name)),
                    value: graft_node(host, src, na.value, ctx)?,
                });
            }
            let inputs = match c.inputs.as_ref() {
                Some(Inputs::Spec(entries)) => {
                    let mut out = Vec::with_capacity(entries.len());
                    for (nm, r) in entries.iter() {
                        let hnm = host.intern(src.resolve(*nm));
                        let hr = graft_input_ref(host, src, *r, ctx)?;
                        out.push((hnm, hr));
                    }
                    Some(Inputs::Spec(out.into()))
                }
                Some(Inputs::Auto) => Some(Inputs::Auto),
                None => None,
            };
            host.alloc(Node::Call(Call {
                head,
                args: args.into(),
                named: named.into(),
                inputs,
            }))
        }
    };
    Ok(out)
}

/// Re-intern a [`Ref`] into the host interner. For a `SelfMod` ref, also graft
/// the submodule binding it names so the host is self-contained (refusing on an
/// unrelated-host-binding collision). `%assign`-substituted names are handled by
/// the caller ([`graft_node`]) before this is reached.
fn graft_ref(
    host: &mut Module,
    src: &Module,
    r: Ref,
    ctx: &mut GraftCtx<'_>,
) -> Result<Ref, String> {
    let hname = host.intern(src.resolve(r.name));
    match r.ns {
        RefNs::Local => Ok(Ref {
            ns: RefNs::Local,
            name: hname,
        }),
        RefNs::SelfMod => {
            graft_binding(host, src, r.name, ctx)?;
            Ok(Ref {
                ns: RefNs::SelfMod,
                name: hname,
            })
        }
        RefNs::Module(alias) => {
            let halias = host.intern(src.resolve(alias));
            Ok(Ref {
                ns: RefNs::Module(halias),
                name: hname,
            })
        }
    }
}

/// Re-intern a reification boundary-input SOURCE ref (an [`Inputs::Spec`] entry).
///
/// Unlike a body ref, a boundary-input source naming a `%assign` parameter is NOT
/// replaced by the host expression: an `Inputs::Spec` entry can only hold a
/// [`Ref`], and the host expression may be an arbitrary node. It is re-interned
/// in place (naming the parameter), and NO submodule binding is grafted for it —
/// the actual body references are already substituted by [`graft_node`], and the
/// reification's `Inputs` bucket is consumed (dropped) when the reified measure is
/// reduced in `density::lower_reified_measure`, so this source ref is cosmetic.
fn graft_input_ref(
    host: &mut Module,
    src: &Module,
    r: Ref,
    ctx: &mut GraftCtx<'_>,
) -> Result<Ref, String> {
    if r.ns == RefNs::SelfMod
        && ctx
            .assign
            .iter()
            .any(|(n, _)| n.as_str() == src.resolve(r.name))
    {
        return Ok(Ref {
            ns: RefNs::SelfMod,
            name: host.intern(src.resolve(r.name)),
        });
    }
    graft_ref(host, src, r, ctx)
}

/// Graft the submodule binding named `name_sym` (a `src` symbol) into `host`.
///
/// * Already grafted this pass → nothing to do (cycle/repeat guard).
/// * Name collides with a pre-existing host binding → **refuse**: modules are
///   independent namespaces and this binding is not `%assign`-linked (those are
///   substituted before reaching here), so reusing the host binding would score
///   against an unrelated value.
/// * No submodule binding of that name (a dangling ref, e.g. a bare-builtin
///   name) → nothing to graft; leave the re-interned ref for downstream handling.
/// * Otherwise → deep-copy the binding's rhs and add it to the host.
fn graft_binding(
    host: &mut Module,
    src: &Module,
    name_sym: Symbol,
    ctx: &mut GraftCtx<'_>,
) -> Result<(), String> {
    let name = src.resolve(name_sym).to_string();
    if !ctx.grafted.insert(name.clone()) {
        return Ok(()); // already grafted this pass
    }
    if ctx.preexisting.contains(&name) {
        return Err(format!(
            "grafted submodule kernel depends on a binding `{name}` whose name collides with an \
             unrelated pre-existing host binding (modules are independent namespaces, and this \
             dependency is not linked by a load_module `%assign`); refuse rather than reuse the \
             host binding and mislower"
        ));
    }
    let Some(src_bid) = src.binding_by_name(name_sym) else {
        return Ok(()); // dangling ref (e.g. a bare-builtin name); nothing to graft
    };
    let src_binding = src.binding(src_bid);
    let src_rhs = src_binding.rhs;
    let public = src_binding.public;
    let host_rhs = graft_node(host, src, src_rhs, ctx)?;
    let hname = host.intern(&name);
    host.add_binding(Binding {
        name: hname,
        rhs: host_rhs,
        doc: None,
        public,
        synthetic: false,
    });
    Ok(())
}
