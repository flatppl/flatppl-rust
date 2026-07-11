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
//! ## Why graft rather than thread the bundle through the lowering
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
//! The graft *itself* does carry the bundle ([`GraftCtx::bundle`]) — but only so
//! it can resolve a NESTED cross-module ref (a submodule being grafted that has
//! its own `load_module` and references it): that nested member is resolved
//! against the bundle and grafted recursively, so a measure that crosses two or
//! more module boundaries (§04 "measure crosses module boundaries transitively")
//! still collapses to a self-contained host node the downstream lowering needs no
//! bundle for.
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
//!
//! * **Distinct submodules are independent namespaces too.** A single graft
//!   chain can now pull in bindings from more than one submodule (a NESTED
//!   cross-module ref: host → A → B, with A and B distinct modules). Two
//!   independent submodules that each define an unrelated binding of the SAME
//!   bare name (e.g. both define `scale`) are just as unrelated as a
//!   host/submodule pair — deduping the second graft onto the first submodule's
//!   value by bare name alone would silently score against the wrong value. So
//!   [`GraftCtx::grafted`] records not just the name but the ORIGIN (the
//!   bundle-path of the submodule that owns the binding) it was grafted from: a
//!   re-request of the same name from the SAME origin (a diamond — the same
//!   submodule binding reached twice) dedups validly, but a re-request from a
//!   DIFFERENT origin **refuses** rather than mislower.
//!
//!   `preexisting` and `grafted` stay two separate checks: `preexisting` is
//!   host-vs-submodule (a bare `HashSet`, since the host is a single fixed
//!   namespace this pass), while `grafted` is submodule-vs-submodule (needs the
//!   extra origin key because multiple submodule namespaces are in play).

use std::collections::{HashMap, HashSet};

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
    /// The bundle path string `sub` was loaded from — the ORIGIN key for
    /// [`GraftCtx::grafted`]'s submodule-vs-submodule collision guard.
    pub path: String,
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
        path,
    })
}

/// A resolved NESTED cross-module ref: the nested submodule (`sub`) named by the
/// CURRENT submodule's own `load_module(alias)`, the referenced member's node id
/// and visibility in `sub`, and the nested `load_module` `%assign` substitutions
/// as `(param-name, SRC node id)` — where the SRC nodes live in the submodule
/// that owns the `load_module` call and must be grafted into the host by the
/// caller before use.
struct NestedResolved<'a> {
    sub: &'a Module,
    member_rhs: NodeId,
    member_public: bool,
    /// `(submodule-parameter-name, SRC node id of the `%assign` value)`.
    assign_src: Vec<(String, NodeId)>,
    /// Bundle path string of the nested submodule (the cycle-guard key uses it).
    path: String,
}

/// Like [`resolve_module_ref`], but resolves a NESTED cross-module ref against
/// the submodule `src` that is currently being grafted: `alias_sym` names `src`'s
/// OWN `load_module("path", …)` binding, `path` → `bundle.get(path)` → the nested
/// submodule's `member_sym` binding. Returns the nested submodule, the member's
/// rhs + visibility, the nested `%assign` (values as SRC nodes — the caller
/// grafts them), and `path` (for the cycle key).
///
/// `None` (⇒ caller refuses) when `alias_sym` is not a `load_module` binding in
/// `src`, its path is absent from the bundle, or the nested submodule has no such
/// member. The member is matched by STRING (cross-interner), like
/// [`resolve_module_ref`].
fn resolve_src_module_ref<'a>(
    bundle: &'a ModuleBundle,
    src: &Module,
    alias_sym: Symbol,
    member_sym: Symbol,
) -> Option<NestedResolved<'a>> {
    let bid = src.binding_by_name(alias_sym)?;
    let load = src.binding(bid).rhs;
    let Node::Call(c) = src.node(load) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if src.resolve(sym) != "load_module" {
        return None;
    }
    let path = string_literal(src, *c.args.first()?)?;
    let assign_src: Vec<(String, NodeId)> = c
        .named
        .iter()
        .filter(|na| na.kind == NamedKind::Assign)
        .map(|na| (src.resolve(na.name).to_string(), na.value))
        .collect();
    let sub = bundle.get(&path)?;
    let member_name = src.resolve(member_sym);
    let (member_rhs, member_public) = sub
        .bindings()
        .find(|(_, b)| sub.resolve(b.name) == member_name)
        .map(|(_, b)| (b.rhs, b.public))?;
    Some(NestedResolved {
        sub,
        member_rhs,
        member_public,
        assign_src,
        path,
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
    /// The host's dependency bundle (path → parsed submodule). Threaded so the
    /// graft can resolve a NESTED cross-module ref — a `RefNs::Module(alias)` in
    /// the submodule currently being grafted, naming that submodule's OWN
    /// `load_module(alias)` — against the bundle and recursively graft it.
    bundle: &'a ModuleBundle,
    /// Load-time `%assign` substitutions for the CURRENT module level
    /// (submodule-parameter name → host node). Owned so it can be swapped for the
    /// nested level's own `%assign` context while recursing (and restored after).
    assign: Vec<(String, NodeId)>,
    /// Names of host bindings that existed BEFORE the graft began. A submodule
    /// dependency whose name collides with one of these (and is not `%assign`-
    /// linked) is an unrelated host binding: refuse rather than reuse it.
    preexisting: HashSet<String>,
    /// Submodule-binding names grafted so far this pass, mapped to the ORIGIN
    /// (bundle-path string of the submodule that owns the binding) they were
    /// grafted from. A re-request of a name already present with the SAME
    /// origin is a diamond — the same submodule binding reached twice — and
    /// dedups validly (`Ok`, skip re-graft). A re-request with a DIFFERENT
    /// origin means two distinct submodules define the same bare name: refuse
    /// (submodules are independent namespaces, same as the host/submodule
    /// case, but here neither name owns `preexisting` status).
    grafted: HashMap<String, String>,
    /// The bundle-path string of the submodule CURRENTLY being grafted — the
    /// origin key recorded into `grafted` for any binding grafted at this
    /// recursion depth. Swapped (alongside `assign`) when the recursion
    /// descends into a NESTED module via [`graft_nested_module_ref`], and
    /// restored on return so the caller's own bindings are attributed to the
    /// caller's origin, not the callee's.
    origin: String,
    /// `(submodule-path, member)` pairs for the nested cross-module members
    /// currently ON the recursion stack. Re-entering one is a cyclic module graph
    /// (A loads B, B loads A): refuse rather than recurse forever. A completed
    /// member is popped, so a legitimate diamond (the same nested member reached
    /// twice) is not mistaken for a cycle.
    in_progress: HashSet<String>,
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
/// place; a nested cross-module (`%ref <alias> …`) ref — i.e. the submodule
/// being grafted itself contains a `load_module` and the grafted body
/// references that nested alias — is resolved against `bundle` and grafted
/// **recursively** (see the `RefNs::Module` arm of `graft_ref`): the nested
/// member and its transitive submodule dependencies land in the host too, with
/// the nested module's OWN `%assign`, the host-collision refuse, and a cycle
/// guard all applying at that level (a measure crosses module boundaries
/// transitively, §04). A nested ref that cannot be resolved (the alias is not a
/// `load_module`, its path is absent from `bundle`, or the member is absent), or
/// a cyclic module graph, is **refused** (`Err`) rather than mislowered.
pub(crate) fn graft_subtree(
    host: &mut Module,
    resolved: &ResolvedRef<'_>,
    bundle: &ModuleBundle,
) -> Result<NodeId, String> {
    let preexisting: HashSet<String> = host
        .bindings()
        .map(|(_, b)| host.resolve(b.name).to_string())
        .collect();
    let mut ctx = GraftCtx {
        bundle,
        assign: resolved.assign.clone(),
        preexisting,
        grafted: HashMap::new(),
        origin: resolved.path.clone(),
        in_progress: HashSet::new(),
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
///
/// A `Module`-namespace ref is a NESTED cross-module reference: the submodule
/// being grafted itself has its own `load_module`, and the grafted body names
/// that nested alias. It is resolved against the bundle (`ctx.bundle`) via
/// [`resolve_src_module_ref`] — reading the CURRENT submodule's own
/// `load_module(alias)` binding — and the referenced nested member is grafted
/// **recursively** into the host, returning a host-local `SelfMod` ref to it.
/// The nested module's OWN `%assign` context, the host-collision refuse
/// (`preexisting`), and the DAG-dedup (`grafted`) all apply at the nested level,
/// and a cyclic module graph is caught by `in_progress` (refuse, not recurse
/// forever). An unresolvable nested ref is refused (`Err`).
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
        RefNs::Module(alias) => graft_nested_module_ref(host, src, alias, r.name, ctx),
    }
}

/// Resolve and RECURSIVELY graft a NESTED cross-module ref `(%ref <alias> member)`
/// found inside the submodule `src` currently being grafted — where `alias` names
/// `src`'s OWN `load_module(alias)` — returning a host-local `SelfMod` ref to the
/// grafted nested member.
///
/// The measure crosses one more module boundary (§04 "measure crosses module
/// boundaries transitively"): resolve `alias` against `ctx.bundle` via
/// [`resolve_src_module_ref`], graft the nested `load_module`'s own `%assign`
/// values into the host under the CURRENT assign context, then graft the nested
/// member (and its transitive submodule deps) under the NESTED assign context.
///
/// Refuse-don't-mislower, at this level too:
/// * unresolvable nested ref (alias not a `load_module` in `src`, its path absent
///   from the bundle, or the member absent) → `Err`;
/// * a nested grafted binding whose name collides with an unrelated pre-existing
///   host binding → `Err` (via [`graft_module_member`] / [`graft_binding`]);
/// * a cyclic module graph (this `(path, member)` is already on the graft stack)
///   → `Err`, guaranteeing termination.
fn graft_nested_module_ref(
    host: &mut Module,
    src: &Module,
    alias: Symbol,
    member: Symbol,
    ctx: &mut GraftCtx<'_>,
) -> Result<Ref, String> {
    let bundle = ctx.bundle;
    let alias_name = src.resolve(alias).to_string();
    let member_name = src.resolve(member).to_string();
    let Some(resolved) = resolve_src_module_ref(bundle, src, alias, member) else {
        return Err(format!(
            "nested cross-module ref `{alias_name}.{member_name}` is unresolvable (the alias is \
             not a `load_module` in the source submodule, its path is absent from the bundle, or \
             the member is absent); refuse rather than mislower"
        ));
    };
    // Cycle guard: a `(submodule-path, member)` already on the graft stack means
    // the module graph loops (A loads B, B loads A). Refuse rather than recurse.
    let cycle_key = format!("{}\u{0}{}", resolved.path, member_name);
    if !ctx.in_progress.insert(cycle_key.clone()) {
        return Err(format!(
            "cyclic module graph: nested cross-module ref `{alias_name}.{member_name}` re-enters \
             `{}` while it is still being grafted; refuse rather than recurse forever",
            resolved.path
        ));
    }
    // The nested `load_module`'s `%assign` values are expressions in `src`; graft
    // them into the host under the CURRENT assign context so they become host
    // nodes, then use them as the assign context for the nested subtree.
    let mut nested_assign: Vec<(String, NodeId)> = Vec::with_capacity(resolved.assign_src.len());
    for (name, src_val) in resolved.assign_src.iter() {
        match graft_node(host, src, *src_val, ctx) {
            Ok(hval) => nested_assign.push((name.clone(), hval)),
            Err(e) => {
                ctx.in_progress.remove(&cycle_key);
                return Err(e);
            }
        }
    }
    // Graft the nested member (and its transitive submodule deps) with the nested
    // assign context AND the nested origin; #75 safety (host collision /
    // DAG-dedup) still applies, now origin-aware (see `GraftCtx::grafted`).
    let saved_assign = std::mem::replace(&mut ctx.assign, nested_assign);
    let saved_origin = std::mem::replace(&mut ctx.origin, resolved.path.clone());
    let graft_res = graft_module_member(
        host,
        resolved.sub,
        &member_name,
        resolved.member_rhs,
        resolved.member_public,
        ctx,
    );
    ctx.origin = saved_origin;
    ctx.assign = saved_assign;
    ctx.in_progress.remove(&cycle_key);
    graft_res?;
    Ok(Ref {
        ns: RefNs::SelfMod,
        name: host.intern(&member_name),
    })
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
/// * Already grafted this pass from the SAME origin (`ctx.origin`) → nothing to
///   do (diamond/repeat guard — legitimate sharing).
/// * Already grafted this pass from a DIFFERENT origin → **refuse**: two
///   distinct loaded modules define a binding of this name, and cross-module
///   name collision is not yet namespaced (see [`GraftCtx::grafted`]).
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
    if let Some(existing_origin) = ctx.grafted.get(&name) {
        if *existing_origin == ctx.origin {
            return Ok(()); // already grafted this pass, same origin (diamond)
        }
        return Err(format!(
            "two distinct loaded modules define a binding named `{name}`; cross-module name \
             collision is not yet namespaced — refuse rather than mislower"
        ));
    }
    ctx.grafted.insert(name.clone(), ctx.origin.clone());
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

/// Graft a NESTED cross-module member `name` (its rhs `rhs` living in the nested
/// submodule `src`) into the host as a standalone binding, then return via a
/// host-local `SelfMod` ref (built by the caller). Mirrors [`graft_binding`]'s
/// #75 safety — origin-aware DAG-dedup (`grafted`) and host-collision refuse
/// (`preexisting`) — but the rhs is supplied directly (the member is known to
/// exist, so there is no dangling case) and the source is the NESTED submodule.
///
/// The nested cross-module cycle guard is the caller's responsibility
/// ([`graft_nested_module_ref`] pushes/pops `ctx.in_progress`, and swaps
/// `ctx.origin` to this nested submodule's bundle path before calling here);
/// this only guards against re-grafting the same (name, origin) pair (a
/// legitimate diamond), a DIFFERENT origin already having grafted this name
/// (two distinct submodules colliding — refuse), and a name collision with an
/// unrelated host binding.
fn graft_module_member(
    host: &mut Module,
    src: &Module,
    name: &str,
    rhs: NodeId,
    public: bool,
    ctx: &mut GraftCtx<'_>,
) -> Result<(), String> {
    if let Some(existing_origin) = ctx.grafted.get(name) {
        if *existing_origin == ctx.origin {
            return Ok(()); // already grafted this pass, same origin (diamond)
        }
        return Err(format!(
            "two distinct loaded modules define a binding named `{name}`; cross-module name \
             collision is not yet namespaced — refuse rather than mislower"
        ));
    }
    ctx.grafted.insert(name.to_string(), ctx.origin.clone());
    if ctx.preexisting.contains(name) {
        return Err(format!(
            "nested grafted submodule member `{name}` collides with an unrelated pre-existing \
             host binding (modules are independent namespaces, and this member is not linked by a \
             load_module `%assign`); refuse rather than reuse the host binding and mislower"
        ));
    }
    let host_rhs = graft_node(host, src, rhs, ctx)?;
    let hname = host.intern(name);
    host.add_binding(Binding {
        name: hname,
        rhs: host_rhs,
        doc: None,
        public,
        synthetic: false,
    });
    Ok(())
}
