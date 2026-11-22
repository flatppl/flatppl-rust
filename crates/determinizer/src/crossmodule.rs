//! Import module members into the host graph before their existing lowering.
//!
//! §04 Module composition distinguishes a source file from a loaded instance.
//! Loads of one source with equal substitutions denote one instance, wherever
//! they occur; loads with different substitutions share no nodes. Imported
//! bindings therefore belong to an instance and a source binding, not to a
//! file/name pair. Host names are allocated independently of that identity.

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

pub(crate) use flatppl_infer::constructors::std_module_member;
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
    let load = src.alias_target(alias)?;
    let call = match src.node(src.binding(load).rhs) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if src.resolve(s) == "load_module") => {
            c
        }
        _ => return None,
    };
    let literal = flatppl_infer::module_source(src, call)?;
    let path = bundle.identity_of(importer, literal)?.to_string();
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

/// What makes a load one instance: its source and its substitutions.
struct InstanceKey {
    path: String,
    /// The instance whose module holds the load; `None` for the host.
    parent: Option<InstanceId>,
    /// The first load of this instance, in the parent's module.
    load: BindingId,
    /// Each substitution, by name: its value node in the parent's module, and
    /// its text when the value reads no binding.
    assign: Vec<(String, NodeId, Option<String>)>,
}

/// One determinization owns this state, including all its lazy graft phases.
#[derive(Default)]
pub(crate) struct GraftState {
    /// Indexed by [`InstanceId`].
    instances: Vec<InstanceKey>,
    bindings: HashMap<Member, ImportedBinding>,
    owners: HashMap<BindingId, Member>,
    /// Source recursion differs from instance sharing: recursive loads create
    /// ever-new instances, so an instance-only guard cannot detect them.
    expanding: HashSet<(String, BindingId)>,
    preparing: HashSet<InstanceId>,
    inferred_inputs: HashMap<String, Arc<SourceInputs>>,
    next_name: usize,
    /// Each binding the graft added, by its name in its own module.
    member_names: HashMap<BindingId, String>,
}

impl GraftState {
    fn live_binding(&self, host: &Module, key: Member) -> Option<ImportedBinding> {
        self.bindings
            .get(&key)
            .copied()
            .filter(|entry| host.binding_by_name(host.binding(entry.host).name) == Some(entry.host))
    }

    /// The instance of a load of `resolved.path` in `src`, the module of
    /// instance `parent`. §04: the same source with the same substitutions
    /// denotes the same module. Two substitution values are equal when they
    /// are structurally equal expressions in one loading module (equal
    /// references, not equal values: `Module::structural_eq`), or when
    /// neither reads a binding and their texts are equal. Equal values that
    /// read different bindings (`c = a` and `c = b` with `a = b`) give two
    /// instances.
    fn instance(
        &mut self,
        src: &Module,
        parent: Option<InstanceId>,
        resolved: &ResolvedRef<'_>,
    ) -> InstanceId {
        let mut assign: Vec<(String, NodeId, Option<String>)> = resolved
            .assign
            .iter()
            .map(|(name, value)| (name.clone(), *value, fixed_text(src, *value)))
            .collect();
        assign.sort_by(|a, b| a.0.cmp(&b.0));
        let equal = |key: &InstanceKey| {
            key.path == resolved.path
                && key.assign.len() == assign.len()
                && key.assign.iter().zip(&assign).all(|(a, b)| {
                    a.0 == b.0
                        && ((key.parent == parent && src.structural_eq(a.1, b.1))
                            || (a.2.is_some() && a.2 == b.2))
                })
        };
        if let Some(k) = self.instances.iter().position(equal) {
            return InstanceId(k);
        }
        self.instances.push(InstanceKey {
            path: resolved.path.clone(),
            parent,
            load: resolved.load,
            assign,
        });
        InstanceId(self.instances.len() - 1)
    }

    fn adopt_alias(&mut self, host: &Module, resolved: &ResolvedRef<'_>, bid: BindingId) {
        // Synthetic aliases can be zeroed by canonicalization. Own a separate
        // non-synthetic slot instead, just as for an ordinary imported member.
        if host.binding(bid).synthetic {
            return;
        }
        let instance = self.instance(host, None, resolved);
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
    let instance = state.instance(host, None, resolved);
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

/// The text of an expression that reads no binding (`1.0`, `[1.0, 2.0]`,
/// `exp(2.0)`), comparable across modules; `None` for one that reads a
/// binding, a placeholder or a user callable.
fn fixed_text(m: &Module, id: NodeId) -> Option<String> {
    match m.node(id) {
        Node::Lit(s) => Some(format!("{s:?}")),
        Node::Const(s) => Some(m.resolve(*s).to_string()),
        Node::Call(c) if c.inputs.is_none() => {
            let CallHead::Builtin(head) = c.head else {
                return None;
            };
            let mut out = format!("{}(", m.resolve(head));
            for &a in c.args.iter() {
                out.push_str(&fixed_text(m, a)?);
                out.push(',');
            }
            for na in c.named.iter() {
                out.push_str(&format!("{:?} {}=", na.kind, m.resolve(na.name)));
                out.push_str(&fixed_text(m, na.value)?);
                out.push(',');
            }
            out.push(')');
            Some(out)
        }
        _ => None,
    }
}

/// Load assignments belong to the loading module. Resolve file-module values
/// there before entering the loaded member's source expansion stack.
pub(crate) fn graft_host_value(
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

/// A host module with every `load_module` member it reads imported.
pub struct Grafted {
    /// The host with the members grafted in, not yet re-inferred.
    pub module: Module,
    /// Each grafted root with the host node it replaced (the `m.x` reference,
    /// or the application `m.k(x)`): a node inside the root's import reads
    /// in the host at that node.
    pub readers: Vec<(NodeId, NodeId)>,
    /// The host's node and binding counts before the graft: a lower id is a
    /// host node or binding, unchanged.
    pub host_nodes: usize,
    pub host_bindings: usize,
    /// Each binding the graft added, with where it came from.
    pub imports: Vec<Import>,
}

/// A binding the graft added to the host.
pub struct Import {
    pub binding: BindingId,
    /// The host's `load_module` alias the member was reached through (for a
    /// member of a module that module loads, the host alias at the top).
    pub alias: String,
    /// The member's name in its own module.
    pub member: String,
}

/// Import every `load_module` member that `host` reads into its own graph, as
/// the determinizer does before lowering (§11: tooling may flatten modules
/// internally). One graft state serves the whole call, and loads of one
/// source with equal substitutions are one instance (§04 Module
/// composition): two substitution values are equal when they are the same
/// expression in one loading module (equal references, not equal values), or
/// the same expression that reads no binding. Standard-module members are not
/// imported.
pub fn graft_imports(host: &Module, bundle: &ModuleBundle) -> Result<Grafted, RefuseError> {
    let mut work = host.clone();
    let mut state = GraftState::default();
    let mut readers = Vec::new();
    // Aliases and refs under `draw` first, as the determinizer's pre-pass.
    let aliased: Vec<_> = work
        .bindings()
        .filter(|(_, b)| resolve_module_ref(bundle, &work, b.rhs).is_some())
        .map(|(bid, b)| (bid, b.rhs))
        .collect();
    resolve_crossmodule_aliases(&mut work, bundle, &mut state)?;
    for (bid, reader) in aliased {
        readers.push((work.binding(bid).rhs, reader));
    }
    // Applications of an imported callee (`m.k(x)`).
    let bindings: Vec<_> = work.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    for (bid, rhs) in bindings {
        if let Some(rebuilt) =
            crate::density::graft_kernel_application_callee(&mut work, rhs, bundle, &mut state)?
        {
            work.set_binding_rhs(bid, rebuilt);
            readers.push((rebuilt, rhs));
        }
    }
    // Every other module reference, recorded with its reader.
    loop {
        let bindings: Vec<_> = work.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
        let mut changed = false;
        for (bid, root) in bindings {
            let mut pending = vec![root];
            let mut seen = HashSet::new();
            let mut replacements = HashMap::new();
            while let Some(id) = pending.pop() {
                if !seen.insert(id) {
                    continue;
                }
                if let Some(resolved) = resolve_module_ref(bundle, &work, id) {
                    let grafted = graft_subtree(&mut work, &resolved, bundle, &mut state)
                        .map_err(|reason| crate::density::refuse(id, &work, &reason))?;
                    readers.push((grafted, id));
                    replacements.insert(id, grafted);
                } else {
                    work.for_each_child(id, |child| pending.push(child));
                }
            }
            if !replacements.is_empty() {
                let rhs = crate::driver::map_tree(&mut work, root, &mut |_, id| {
                    replacements.get(&id).copied()
                });
                work.set_binding_rhs(bid, rhs);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // The top-level host load of each instance, through the parents of its
    // first load (a parent is always an older instance).
    let mut imports: Vec<Import> = state
        .member_names
        .iter()
        .filter_map(|(&binding, member)| {
            let (InstanceId(mut instance), _) = *state.owners.get(&binding)?;
            let load = loop {
                let key = state.instances.get(instance)?;
                match key.parent {
                    Some(InstanceId(p)) => instance = p,
                    None => break key.load,
                }
            };
            Some(Import {
                binding,
                alias: work.resolve(work.binding(load).name).to_string(),
                member: member.clone(),
            })
        })
        .collect();
    imports.sort_by_key(|i| i.binding);
    Ok(Grafted {
        module: work,
        readers,
        host_nodes: host.node_count(),
        host_bindings: host.binding_count(),
        imports,
    })
}

/// Resolve retained value expressions after simplification and callee guards.
pub(crate) fn resolve_crossmodule_values(
    host: &mut Module,
    bundle: &ModuleBundle,
    state: &mut GraftState,
) -> Result<bool, RefuseError> {
    let bindings: Vec<_> = host.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    let mut changed = false;
    for (bid, root) in bindings {
        let rhs = graft_host_value(host, root, bundle, state)
            .map_err(|reason| crate::density::refuse(root, host, &reason))?;
        if rhs != root {
            host.set_binding_rhs(bid, rhs);
            changed = true;
        }
    }
    Ok(changed)
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
            if let Some(bid) = src.alias_target(alias)
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
            let instance = ctx.state.instance(src, Some(ctx.instance), &resolved);
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
        let host_bid = host.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: false,
            synthetic: false,
        });
        ctx.state
            .member_names
            .insert(host_bid, src.resolve(binding.name).to_string());
        host_bid
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
