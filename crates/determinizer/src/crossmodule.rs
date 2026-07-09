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

use std::collections::HashSet;

use flatppl_core::{
    Axis, Binding, Call, CallHead, Inputs, Module, NamedArg, Node, NodeId, Ref, RefNs, Scalar,
    Symbol,
};
use flatppl_infer::ModuleBundle;

/// For a `Node::Ref { ns: RefNs::Module(alias), name: member }` at `id` in
/// `host`, follow `alias` → its `load_module("path", …)` binding → `path` →
/// `bundle.get(path)` → the submodule's `member` binding, and return
/// `(submodule, member_rhs_node)`.
///
/// `None` (⇒ the caller refuses, per refuse-don't-mislower) when `id` is not a
/// module ref, the alias is not a `load_module` binding, the path dependency is
/// absent from the bundle, or the submodule has no such (member) binding.
pub(crate) fn resolve_module_ref<'a>(
    bundle: &'a ModuleBundle,
    host: &Module,
    id: NodeId,
) -> Option<(&'a Module, NodeId)> {
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
    let sub = bundle.get(&path)?;
    // Cross-interner lookup: match the member by string, not by Symbol.
    let member_name = host.resolve(member);
    let member_rhs = sub
        .bindings()
        .find(|(_, b)| sub.resolve(b.name) == member_name)
        .map(|(_, b)| b.rhs)?;
    Some((sub, member_rhs))
}

/// The `Box<str>` payload of a `Node::Lit(Scalar::Str(_))`, else `None`.
fn string_literal(m: &Module, id: NodeId) -> Option<String> {
    match m.node(id) {
        Node::Lit(Scalar::Str(s)) => Some(s.to_string()),
        _ => None,
    }
}

/// Deep-copy the subtree rooted at `root` from `src` into `host`, re-interning
/// every symbol through the host interner and returning the new host `NodeId`.
///
/// A `(%ref self <name>)` in the copied subtree names a *submodule* binding; to
/// keep the grafted node self-contained, that binding is itself grafted into the
/// host (recursively), unless the host already carries a binding of the same
/// name (the workspace no-shadowing assumption — reuse it). Placeholder
/// (`%local`) refs and boundary-input entries are re-interned in place; a nested
/// cross-module (`%ref <alias> …`) ref is re-interned but left unresolved (there
/// is no bundle here — an unresolved nested module ref simply refuses downstream,
/// never mislowers).
pub(crate) fn graft_subtree(host: &mut Module, src: &Module, root: NodeId) -> NodeId {
    let mut grafted: HashSet<String> = HashSet::new();
    graft_node(host, src, root, &mut grafted)
}

fn graft_node(
    host: &mut Module,
    src: &Module,
    id: NodeId,
    grafted: &mut HashSet<String>,
) -> NodeId {
    // Clone the source node so the `src` borrow ends before we mutate `host`.
    let node = src.node(id).clone();
    match node {
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
            let hr = graft_ref(host, src, r, grafted);
            host.alloc(Node::Ref(hr))
        }
        Node::Call(c) => {
            let head = match c.head {
                CallHead::Builtin(sym) => CallHead::Builtin(host.intern(src.resolve(sym))),
                CallHead::User(callee) => CallHead::User(graft_node(host, src, callee, grafted)),
            };
            let args: Vec<NodeId> = c
                .args
                .iter()
                .map(|&a| graft_node(host, src, a, grafted))
                .collect();
            let named: Vec<NamedArg> = c
                .named
                .iter()
                .map(|na| NamedArg {
                    kind: na.kind,
                    name: host.intern(src.resolve(na.name)),
                    value: graft_node(host, src, na.value, grafted),
                })
                .collect();
            let inputs = c.inputs.as_ref().map(|inp| match inp {
                Inputs::Spec(entries) => Inputs::Spec(
                    entries
                        .iter()
                        .map(|(nm, r)| {
                            let hnm = host.intern(src.resolve(*nm));
                            let hr = graft_ref(host, src, *r, grafted);
                            (hnm, hr)
                        })
                        .collect(),
                ),
                Inputs::Auto => Inputs::Auto,
            });
            host.alloc(Node::Call(Call {
                head,
                args: args.into(),
                named: named.into(),
                inputs,
            }))
        }
    }
}

/// Re-intern a [`Ref`] into the host interner. For a `SelfMod` ref, also graft
/// the submodule binding it names so the host is self-contained.
fn graft_ref(host: &mut Module, src: &Module, r: Ref, grafted: &mut HashSet<String>) -> Ref {
    let hname = host.intern(src.resolve(r.name));
    match r.ns {
        RefNs::Local => Ref {
            ns: RefNs::Local,
            name: hname,
        },
        RefNs::SelfMod => {
            graft_binding(host, src, r.name, grafted);
            Ref {
                ns: RefNs::SelfMod,
                name: hname,
            }
        }
        RefNs::Module(alias) => {
            let halias = host.intern(src.resolve(alias));
            Ref {
                ns: RefNs::Module(halias),
                name: hname,
            }
        }
    }
}

/// Graft the submodule binding named `name_sym` (a `src` symbol) into `host`,
/// unless it is already present (grafted earlier this pass, or a pre-existing
/// host binding of the same name — the no-shadowing assumption lets us reuse
/// it). The recursion is bounded by `grafted` against reference cycles.
fn graft_binding(host: &mut Module, src: &Module, name_sym: Symbol, grafted: &mut HashSet<String>) {
    let name = src.resolve(name_sym).to_string();
    if !grafted.insert(name.clone()) {
        return; // already visited this pass
    }
    let hname = host.intern(&name);
    if host.binding_by_name(hname).is_some() {
        return; // host already carries this binding — reuse it
    }
    let Some(src_bid) = src.binding_by_name(name_sym) else {
        return; // dangling ref (e.g. a bare-builtin name); nothing to graft
    };
    let src_binding = src.binding(src_bid);
    let src_rhs = src_binding.rhs;
    let public = src_binding.public;
    let host_rhs = graft_node(host, src, src_rhs, grafted);
    host.add_binding(Binding {
        name: hname,
        rhs: host_rhs,
        doc: None,
        public,
        synthetic: false,
    });
}
