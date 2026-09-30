//! Import module members into the host graph before their existing lowering.
//!
//! §04 Module composition distinguishes a source file from a loaded instance.
//! One load shares its dependencies across uses; separate loads share no nodes.
//! Imported bindings therefore belong to an instance and a source binding, not
//! to a file/name pair. Host names are allocated independently of that identity.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use flatppl_core::{
    Axis, Binding, BindingId, Call, CallHead, Idx, Inputs, Module, NamedArg, NamedKind, Node,
    NodeId, Ref, RefNs, Scalar, Symbol,
};
use flatppl_infer::ModuleBundle;

use crate::refuse::RefuseError;

pub(crate) struct ResolvedRef<'a> {
    sub: &'a Module,
    member: BindingId,
    load: BindingId,
    /// Assignment nodes belong to the module containing the load.
    assign: Vec<(String, NodeId)>,
    path: String,
}

/// Standard-module members are catalogue entries, not file-module graphs.
pub(crate) fn std_module_member(host: &Module, id: NodeId) -> Option<(String, String)> {
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }) = *host.node(id)
    else {
        return None;
    };
    let bid = host.binding_by_name(alias)?;
    let Node::Call(c) = host.node(host.binding(bid).rhs) else {
        return None;
    };
    if !matches!(c.head, CallHead::Builtin(s) if host.resolve(s) == "standard_module") {
        return None;
    }
    Some((
        string_literal(host, *c.args.first()?)?,
        host.resolve(name).to_string(),
    ))
}

pub(crate) fn resolve_module_ref<'a>(
    bundle: &'a ModuleBundle,
    host: &Module,
    id: NodeId,
) -> Option<ResolvedRef<'a>> {
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }) = *host.node(id)
    else {
        return None;
    };
    resolve_member(bundle, bundle.root(), host, alias, name)
}

fn resolve_member<'a>(
    bundle: &'a ModuleBundle,
    importer: &str,
    src: &Module,
    alias: Symbol,
    member: Symbol,
) -> Option<ResolvedRef<'a>> {
    let mut load = src.binding_by_name(alias)?;
    let mut seen = HashSet::new();
    let call = loop {
        if !seen.insert(load) {
            return None;
        }
        match src.node(src.binding(load).rhs) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => load = src.binding_by_name(*name)?,
            Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if src.resolve(s) == "load_module") =>
            {
                break c;
            }
            _ => return None,
        }
    };
    let literal = string_literal(src, *call.args.first()?)?;
    let path = bundle.identity_of(importer, &literal)?.to_string();
    let sub = bundle.get_by_id(&path)?;
    let member = sub
        .bindings()
        .find(|(_, b)| sub.resolve(b.name) == src.resolve(member))?
        .0;
    let assign = call
        .named
        .iter()
        .filter(|arg| arg.kind == NamedKind::Assign)
        .map(|arg| (src.resolve(arg.name).to_string(), arg.value))
        .collect();
    Some(ResolvedRef {
        sub,
        member,
        load,
        assign,
        path,
    })
}

fn string_literal(m: &Module, id: NodeId) -> Option<String> {
    match m.node(id) {
        Node::Lit(Scalar::Str(s)) => Some(s.to_string()),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct InstanceId(usize);

type Member = (InstanceId, BindingId);
type SourceInputs = HashMap<NodeId, Box<[(Symbol, Ref)]>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Materialization {
    Reserved,
    Visiting,
    Ready,
}

#[derive(Clone, Copy)]
struct ImportedBinding {
    host: BindingId,
    state: Materialization,
}

/// One determinization owns this state, including all its lazy graft phases.
#[derive(Default)]
pub(crate) struct GraftState {
    instances: HashMap<(Option<InstanceId>, BindingId), InstanceId>,
    bindings: HashMap<Member, ImportedBinding>,
    owners: HashMap<BindingId, Member>,
    /// Source recursion differs from instance sharing: recursive loads create
    /// ever-new instances, so an instance-only guard cannot detect them.
    expanding: HashSet<(String, BindingId)>,
    preparing: HashSet<InstanceId>,
    inferred_inputs: HashMap<String, Arc<SourceInputs>>,
    next_name: usize,
}

impl GraftState {
    fn live_binding(&self, host: &Module, key: Member) -> Option<ImportedBinding> {
        self.bindings
            .get(&key)
            .copied()
            .filter(|entry| host.binding_by_name(host.binding(entry.host).name) == Some(entry.host))
    }

    fn instance(&mut self, parent: Option<InstanceId>, load: BindingId) -> InstanceId {
        let next = InstanceId(self.instances.len());
        *self.instances.entry((parent, load)).or_insert(next)
    }

    /// The measure sweep may discard a member needed by a later lazy import.
    /// Canonical DCE is checked separately through the live name index.
    pub(crate) fn discard(&mut self, bid: BindingId) {
        if let Some(key) = self.owners.get(&bid)
            && let Some(imported) = self.bindings.get_mut(key)
        {
            imported.state = Materialization::Reserved;
        }
    }

    fn adopt_alias(&mut self, host: &Module, resolved: &ResolvedRef<'_>, bid: BindingId) {
        // Synthetic aliases can be zeroed by canonicalization. Own a separate
        // non-synthetic slot instead, just as for an ordinary imported member.
        if host.binding(bid).synthetic {
            return;
        }
        let instance = self.instance(None, resolved.load);
        let key = (instance, resolved.member);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.bindings.entry(key) {
            entry.insert(ImportedBinding {
                host: bid,
                state: Materialization::Reserved,
            });
            self.owners.insert(bid, key);
        }
    }

    fn fresh_name(&mut self, host: &mut Module, base: &str) -> Symbol {
        let name = host.intern(base);
        if host.binding_by_name(name).is_none() {
            return name;
        }
        loop {
            let name = host.intern(&format!("__import_{base}_{}", self.next_name));
            self.next_name += 1;
            if host.binding_by_name(name).is_none() {
                return name;
            }
        }
    }
}

struct GraftCtx<'a, 's> {
    bundle: &'a ModuleBundle,
    state: &'s mut GraftState,
    instance: InstanceId,
    assign: Vec<(String, NodeId)>,
    path: String,
}

pub(crate) fn graft_subtree(
    host: &mut Module,
    resolved: &ResolvedRef<'_>,
    bundle: &ModuleBundle,
    state: &mut GraftState,
) -> Result<NodeId, String> {
    let instance = state.instance(None, resolved.load);
    if let Some(entry) = state.live_binding(host, (instance, resolved.member))
        && entry.state == Materialization::Ready
    {
        let name = host.binding(entry.host).name;
        return Ok(host.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        })));
    }
    if !state.preparing.insert(instance) {
        return Err("cyclic module load assignments".into());
    }
    let assign = resolved
        .assign
        .iter()
        .map(|(name, value)| {
            graft_host_value(host, *value, bundle, state).map(|value| (name.clone(), value))
        })
        .collect::<Result<_, _>>();
    state.preparing.remove(&instance);
    let mut ctx = GraftCtx {
        bundle,
        state,
        instance,
        assign: assign?,
        path: resolved.path.clone(),
    };
    let name = graft_binding(host, resolved.sub, resolved.member, &mut ctx)?;
    Ok(host.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    })))
}

/// Load assignments belong to the loading module. Resolve file-module values
/// there before entering the loaded member's source expansion stack.
fn graft_host_value(
    host: &mut Module,
    root: NodeId,
    bundle: &ModuleBundle,
    state: &mut GraftState,
) -> Result<NodeId, String> {
    let mut pending = vec![root];
    let mut seen = HashSet::new();
    let mut replacements = HashMap::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(resolved) = resolve_module_ref(bundle, host, id) {
            replacements.insert(id, graft_subtree(host, &resolved, bundle, state)?);
        } else {
            host.for_each_child(id, |child| pending.push(child));
        }
    }
    if replacements.is_empty() {
        Ok(root)
    } else {
        Ok(crate::driver::map_tree(host, root, &mut |_, id| {
            replacements.get(&id).copied()
        }))
    }
}

/// Resolve aliases and draw-measure refs before density destructuring. Other
/// entry points remain lazy so simplification can discard unused arguments.
pub(crate) fn resolve_crossmodule_aliases(
    host: &mut Module,
    bundle: &ModuleBundle,
    state: &mut GraftState,
) -> Result<(), RefuseError> {
    let plans: Vec<_> = host
        .bindings()
        .filter_map(|(bid, b)| {
            resolve_module_ref(bundle, host, b.rhs).map(|resolved| (bid, b.rhs, resolved))
        })
        .collect();
    for (bid, _, resolved) in &plans {
        state.adopt_alias(host, resolved, *bid);
    }
    for (bid, ref_id, resolved) in plans {
        let root = graft_subtree(host, &resolved, bundle, state)
            .map_err(|reason| crate::density::refuse(ref_id, host, &reason))?;
        // Grafting fills an adopted alias in place. Other aliases retain a
        // reference to that slot rather than duplicating its stochastic graph.
        if !matches!(host.node(root), Node::Ref(r) if r.ns == RefNs::SelfMod && r.name == host.binding(bid).name)
        {
            host.set_binding_rhs(bid, root);
        }
    }
    let mut replacements = HashMap::new();
    for ref_id in collect_draw_measure_module_refs(host) {
        if let Some(resolved) = resolve_module_ref(bundle, host, ref_id) {
            let root = graft_subtree(host, &resolved, bundle, state)
                .map_err(|reason| crate::density::refuse(ref_id, host, &reason))?;
            replacements.insert(ref_id, root);
        }
    }
    if !replacements.is_empty() {
        let pairs: Vec<_> = host.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
        for (bid, root) in pairs {
            let new =
                crate::driver::map_tree(host, root, &mut |_, id| replacements.get(&id).copied());
            if new != root {
                host.set_binding_rhs(bid, new);
            }
        }
    }
    Ok(())
}

fn collect_draw_measure_module_refs(host: &Module) -> Vec<NodeId> {
    let mut seen = HashSet::new();
    let mut out = HashSet::new();
    let mut stack: Vec<_> = host.bindings().map(|(_, b)| (b.rhs, false)).collect();
    let mut refs = Vec::new();
    while let Some((id, under_draw)) = stack.pop() {
        if !seen.insert((id, under_draw)) {
            continue;
        }
        let under_draw = under_draw
            || matches!(host.node(id), Node::Call(c)
            if matches!(c.head, CallHead::Builtin(s) if host.resolve(s) == "draw"));
        if under_draw
            && matches!(
                host.node(id),
                Node::Ref(Ref {
                    ns: RefNs::Module(_),
                    ..
                })
            )
        {
            if out.insert(id) {
                refs.push(id);
            }
        } else {
            host.for_each_child(id, |child| stack.push((child, under_draw)));
        }
    }
    refs
}

fn graft_node(
    host: &mut Module,
    src: &Module,
    id: NodeId,
    ctx: &mut GraftCtx<'_, '_>,
) -> Result<NodeId, String> {
    let node = match src.node(id).clone() {
        Node::Lit(s) => Node::Lit(s),
        Node::Hole => Node::Hole,
        Node::Const(s) => Node::Const(host.intern(src.resolve(s))),
        Node::Axis(Axis { name, variance }) => Node::Axis(Axis {
            name: host.intern(src.resolve(name)),
            variance,
        }),
        Node::Ref(r) => Node::Ref(graft_ref(host, src, r, ctx)?),
        Node::Call(c) => {
            let head = match c.head {
                CallHead::Builtin(s) => CallHead::Builtin(host.intern(src.resolve(s))),
                CallHead::User(callee) => CallHead::User(graft_node(host, src, callee, ctx)?),
            };
            let args = c
                .args
                .iter()
                .map(|&arg| graft_node(host, src, arg, ctx))
                .collect::<Result<_, _>>()?;
            let named = c
                .named
                .iter()
                .map(|arg| {
                    Ok(NamedArg {
                        kind: arg.kind,
                        name: host.intern(src.resolve(arg.name)),
                        value: graft_node(host, src, arg.value, ctx)?,
                    })
                })
                .collect::<Result<_, String>>()?;
            let inputs = match &c.inputs {
                Some(Inputs::Spec(entries)) => {
                    Some(Inputs::Spec(graft_inputs(host, src, entries, ctx)?))
                }
                Some(Inputs::Auto) => Some(Inputs::Auto),
                None => None,
            };
            let result = host.alloc(Node::Call(Call {
                head,
                args,
                named,
                inputs,
            }));
            if matches!(c.inputs, Some(Inputs::Auto)) {
                let entries = if let Some(entries) = src
                    .auto_inputs_of(id)
                    .or_else(|| ctx.bundle.auto_inputs_of(&ctx.path, id))
                {
                    graft_inputs(host, src, entries, ctx)?
                } else {
                    let source_inputs = ctx
                        .state
                        .inferred_inputs
                        .entry(ctx.path.clone())
                        .or_insert_with(|| {
                            let mut inferred = src.clone();
                            let mut bundle = ctx.bundle.clone();
                            bundle.set_root(&ctx.path);
                            // Only phases and callable inputs are consumed here;
                            // the host resolves shapes after grafting.
                            let _ = flatppl_infer::infer_module(
                                &mut inferred,
                                &bundle,
                                flatppl_infer::Level::Type,
                            );
                            // Input refs use existing source symbols. Retain
                            // them without retaining the annotated source DAG.
                            let inputs = (0..src.node_count())
                                .map(NodeId::from_usize)
                                .filter_map(|id| {
                                    inferred
                                        .auto_inputs_of(id)
                                        .map(|entries| (id, entries.into()))
                                })
                                .collect();
                            Arc::new(inputs)
                        })
                        .clone();
                    let entries = source_inputs
                        .get(&id)
                        .ok_or("cannot resolve imported callable's automatic inputs")?;
                    graft_inputs(host, src, entries, ctx)?
                };
                host.set_auto_inputs(result, entries);
            }
            return Ok(result);
        }
    };
    Ok(host.alloc(node))
}

fn graft_inputs(
    host: &mut Module,
    src: &Module,
    entries: &[(Symbol, Ref)],
    ctx: &mut GraftCtx<'_, '_>,
) -> Result<Box<[(Symbol, Ref)]>, String> {
    entries
        .iter()
        .map(|(label, r)| {
            Ok((
                host.intern(src.resolve(*label)),
                graft_ref(host, src, *r, ctx)?,
            ))
        })
        .collect()
}

fn graft_ref(
    host: &mut Module,
    src: &Module,
    r: Ref,
    ctx: &mut GraftCtx<'_, '_>,
) -> Result<Ref, String> {
    let name = host.intern(src.resolve(r.name));
    match r.ns {
        RefNs::Local => Ok(Ref {
            ns: RefNs::Local,
            name,
        }),
        RefNs::SelfMod => {
            let name = match src.binding_by_name(r.name) {
                Some(bid) => graft_binding(host, src, bid, ctx)?,
                None => name,
            };
            Ok(Ref {
                ns: RefNs::SelfMod,
                name,
            })
        }
        RefNs::Module(alias) => {
            if let Some(bid) = src.binding_by_name(alias)
                && crate::density::builtin_name(src, src.binding(bid).rhs)
                    == Some("standard_module")
            {
                let alias = graft_binding(host, src, bid, ctx)?;
                return Ok(Ref {
                    ns: RefNs::Module(alias),
                    name,
                });
            }
            let resolved =
                resolve_member(ctx.bundle, &ctx.path, src, alias, r.name).ok_or_else(|| {
                    format!(
                        "nested cross-module ref `{}.{}` is unresolvable",
                        src.resolve(alias),
                        src.resolve(r.name)
                    )
                })?;
            let instance = ctx.state.instance(Some(ctx.instance), resolved.load);
            // Assignments run in the parent, before the child's source-cycle
            // guard: another sibling instance of this same file is not a cycle.
            let mut assign = Vec::with_capacity(resolved.assign.len());
            for (name, value) in &resolved.assign {
                assign.push((name.clone(), graft_node(host, src, *value, ctx)?));
            }
            let mut child = GraftCtx {
                bundle: ctx.bundle,
                state: ctx.state,
                instance,
                assign,
                path: resolved.path,
            };
            let name = graft_binding(host, resolved.sub, resolved.member, &mut child)?;
            Ok(Ref {
                ns: RefNs::SelfMod,
                name,
            })
        }
    }
}

fn graft_binding(
    host: &mut Module,
    src: &Module,
    bid: BindingId,
    ctx: &mut GraftCtx<'_, '_>,
) -> Result<Symbol, String> {
    let key = (ctx.instance, bid);
    let cached = ctx.state.live_binding(host, key);
    if let Some(entry) = cached {
        match entry.state {
            Materialization::Ready => return Ok(host.binding(entry.host).name),
            Materialization::Visiting => return Err("cyclic module binding dependency".into()),
            Materialization::Reserved => {}
        }
    }
    let source_key = (ctx.path.clone(), bid);
    if !ctx.state.expanding.insert(source_key.clone()) {
        return Err(format!(
            "cyclic module graph: re-entered `{}` in `{}`",
            src.resolve(src.binding(bid).name),
            ctx.path
        ));
    }
    let binding = src.binding(bid);
    let host_bid = if let Some(entry) = cached {
        entry.host
    } else {
        let name = ctx.state.fresh_name(host, src.resolve(binding.name));
        let rhs = host.alloc(Node::Lit(Scalar::Real(0.0)));
        // Only explicit host aliases expose imported values in the host's
        // interface. These slots must not enter the synthetic-value sweep.
        host.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: false,
            synthetic: false,
        })
    };
    ctx.state.bindings.insert(
        key,
        ImportedBinding {
            host: host_bid,
            state: Materialization::Visiting,
        },
    );
    ctx.state.owners.insert(host_bid, key);
    // Assigned inputs need the same node identity in bodies and boundaries.
    // An owned slot also accommodates assignments that are not plain refs.
    let rhs = if let Some((_, value)) = ctx
        .assign
        .iter()
        .find(|(name, _)| name == src.resolve(binding.name))
    {
        Ok(*value)
    } else {
        graft_node(host, src, binding.rhs, ctx)
    };
    ctx.state.expanding.remove(&source_key);
    host.set_binding_rhs(host_bid, rhs?);
    ctx.state.bindings.get_mut(&key).unwrap().state = Materialization::Ready;
    Ok(host.binding(host_bid).name)
}
