//! Backward demand for components of ordinary value constructors.

use std::collections::{HashMap, HashSet};

use flatppl_core::graph::{rebuild_with_children, resolve_ref_one};
use flatppl_core::{
    Call, CallHead, Dim, Idx, Inputs, Module, NamedKind, Node, NodeId, Phase, RefNs, Scalar,
    ScalarType, Symbol, Type,
};

use crate::project::Cuts;
use crate::{Dtype, Profile};

#[path = "scan_demand.rs"]
mod scan;
pub use scan::{ScanOutputDemand, scan_output_demand, scan_output_demand_with_rules};

const MAX_DEMAND_NODES: usize = 65_536;
const MAX_PROOF_DEPTH: usize = 64;

#[derive(Clone, Copy)]
struct Projection {
    container: NodeId,
    selected: NodeId,
}

pub(super) fn prune(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
) -> Option<Module> {
    if module.node_count() > MAX_DEMAND_NODES
        || !(0..module.node_count()).any(|i| selector(module, NodeId::from_usize(i)).is_some())
    {
        return None;
    }
    let active: Vec<_> = match roots {
        Some(names) => names
            .iter()
            .map(|&name| module.binding_by_name(name))
            .collect::<Option<_>>()?,
        None => module
            .public_bindings()
            .map(|(binding, _)| binding)
            .collect(),
    };
    let entries: Vec<_> = active.iter().map(|&b| module.binding(b).rhs).collect();
    if entries.is_empty() {
        return None;
    }
    let cuts = Cuts::new(module, profile);
    let mut reachable = HashSet::new();
    let mut protected = HashSet::new();
    let mut pending: Vec<_> = module.bindings().map(|(_, b)| (b.rhs, false)).collect();
    while let Some((id, preserve)) = pending.pop() {
        let preserve = preserve || cuts.blocked(module, id) || identity_boundary(module, id);
        if preserve {
            if !protected.insert(id) {
                continue;
            }
        } else if !reachable.insert(id) {
            continue;
        }
        dependencies(module, id, |child| pending.push((child, preserve)));
    }

    // A whole-value consumer makes every component live. Component consumers
    // of the same constructor contribute their selected values to the worklist.
    let mut whole = HashSet::new();
    let mut components: HashMap<NodeId, Vec<(NodeId, NodeId)>> = HashMap::new();
    let mut proof = Proof {
        module,
        protected,
        abi_cuts: (roots.is_some() && matches!(profile, Profile::StableHlo { .. }))
            .then_some(&cuts),
        wide_integers: matches!(
            profile,
            Profile::StableHlo {
                dtype: Dtype::F64,
                ..
            }
        ),
        values: HashMap::new(),
        projections: HashMap::new(),
    };
    let mut pending = entries;
    while let Some(id) = pending.pop() {
        if !whole.insert(id) {
            continue;
        }
        if let Some(projection) = proof.projection(id, 0) {
            components
                .entry(projection.container)
                .or_default()
                .push((id, projection.selected));
            pending.push(projection.selected);
        } else {
            dependencies(module, id, |child| pending.push(child));
        }
    }

    let replace: HashMap<_, _> = components
        .into_iter()
        .filter(|(container, _)| !whole.contains(container))
        .flat_map(|(_, consumers)| consumers)
        .collect();
    if replace.is_empty() {
        return None;
    }

    let mut output = module.clone();
    let mut rebuilt = HashMap::new();
    // Leave binding ids and the ABI declaration in place. Emitters traverse
    // the selected outputs, so unreachable component work is no longer emitted.
    for (binding, value) in module
        .bindings()
        .filter(|(_, value)| whole.contains(&value.rhs))
    {
        let result = rebuild(module, &mut output, value.rhs, &replace, &mut rebuilt)?;
        output.set_binding_rhs(binding, result);
    }
    let diagnostics = flatppl_infer::infer_module(&mut output, bundle, flatppl_infer::Level::Shape);
    if diagnostics
        .iter()
        .any(|d| d.severity == flatppl_infer::Severity::Error)
        || module.bindings().any(|(binding, original)| {
            let rewritten = output.binding(binding).rhs;
            module.type_of(original.rhs) != output.type_of(rewritten)
                || module.phase_of(original.rhs) != output.phase_of(rewritten)
        })
    {
        return None;
    }
    Some(output)
}

fn builtin(module: &Module, node: NodeId) -> Option<(&str, &Call)> {
    let Node::Call(call) = module.node(node) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    Some((module.resolve(head), call))
}

fn identity_boundary(module: &Module, node: NodeId) -> bool {
    module.is_sample_result(node)
        || module.sample_origin(node).is_some()
        || module.sampled_measure(node).is_some()
        || match module.node(node) {
            Node::Call(call) => {
                call.inputs.is_some()
                    || matches!(call.head,
            CallHead::Builtin(head) if matches!(module.resolve(head), "lawof" | "draw" | "rand" | "checked"))
            }
            _ => false,
        }
}

fn dependencies(module: &Module, node: NodeId, mut push: impl FnMut(NodeId)) {
    let (value, binding) = resolve_ref_one(module, node);
    if binding.is_some() {
        push(value);
    }
    module.for_each_child(node, &mut push);
    if let Node::Call(call) = module.node(node) {
        let inputs = match &call.inputs {
            Some(Inputs::Spec(inputs)) => Some(inputs.as_ref()),
            Some(Inputs::Auto) => module.auto_inputs_of(node),
            None => None,
        };
        for (_, reference) in inputs.into_iter().flatten() {
            if reference.ns == RefNs::SelfMod
                && let Some(binding) = module.binding_by_name(reference.name)
            {
                push(module.binding(binding).rhs);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Selector<'a> {
    Field(&'a str),
    Index(usize),
}

fn selector(module: &Module, node: NodeId) -> Option<(NodeId, NodeId, Selector<'_>)> {
    let (head, call) = builtin(module, node)?;
    let [container, key] = *call.args else {
        return None;
    };
    if !call.named.is_empty() || call.inputs.is_some() {
        return None;
    }
    let cell = match (head, module.node(key)) {
        ("get" | "get_field", Node::Lit(Scalar::Str(name))) => Selector::Field(name),
        ("get" | "get0" | "tuple_get", Node::Lit(Scalar::Int(index))) => {
            let offset = i64::from(head == "get");
            Selector::Index(usize::try_from(index.checked_sub(offset)?).ok()?)
        }
        _ => return None,
    };
    Some((container, key, cell))
}

struct Proof<'a> {
    module: &'a Module,
    protected: HashSet<NodeId>,
    abi_cuts: Option<&'a Cuts>,
    wide_integers: bool,
    values: HashMap<(NodeId, ValueUse), bool>,
    projections: HashMap<NodeId, Option<Projection>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ValueUse {
    Keep,
    Discard,
}

impl Proof<'_> {
    fn projection(&mut self, node: NodeId, depth: usize) -> Option<Projection> {
        if depth >= MAX_PROOF_DEPTH || self.protected.contains(&node) {
            return None;
        }
        if let Some(&proof) = self.projections.get(&node) {
            return proof;
        }
        let proof = self.project_cell(node, depth);
        self.projections.insert(node, proof);
        proof
    }

    fn project_cell(&mut self, node: NodeId, depth: usize) -> Option<Projection> {
        let (container, key, cell) = selector(self.module, node)?;
        if self.protected.contains(&key) {
            return None;
        }
        let container = self.source(container, depth + 1)?;
        let (head, call) = builtin(self.module, container)?;
        if call.inputs.is_some() || !self.module.type_of(container)?.is_concrete_value() {
            return None;
        }
        let selected = match (head, self.module.type_of(container)?, cell) {
            ("record", Type::Record(_), Selector::Field(name)) if call.args.is_empty() => {
                let mut names = HashSet::new();
                let mut chosen = None;
                for field in &call.named {
                    if field.kind != NamedKind::Field || !names.insert(field.name) {
                        return None;
                    }
                    if self.module.resolve(field.name) == name {
                        chosen = Some(field.value);
                    } else if !self.discarded_cell(field.value) {
                        return None;
                    }
                }
                chosen?
            }
            ("tuple", Type::Tuple(parts), Selector::Index(index))
                if call.named.is_empty() && parts.len() == call.args.len() =>
            {
                self.positional(call, index)?
            }
            ("vector", Type::Array { shape, .. }, Selector::Index(index))
                if call.named.is_empty()
                    && matches!(shape.as_ref(), [Dim::Static(n)] if *n as usize == call.args.len()) =>
            {
                self.positional(call, index)?
            }
            _ => return None,
        };
        let phase = self.module.phase_of(node)?;
        if self.module.type_of(node)? != self.module.type_of(selected)?
            || phase != self.module.phase_of(selected)?
            || phase == Phase::Stochastic
            || !self.value(selected, ValueUse::Keep, 0)
        {
            return None;
        }
        Some(Projection {
            container,
            selected,
        })
    }

    fn source(&mut self, node: NodeId, depth: usize) -> Option<NodeId> {
        if depth >= MAX_PROOF_DEPTH || self.protected.contains(&node) {
            return None;
        }
        let (next, binding) = resolve_ref_one(self.module, node);
        if let Some(binding) = binding {
            if self.module.query_pinned_rhs(binding).is_some() {
                return None;
            }
            return self.source(next, depth + 1);
        }
        if let Some(projection) = self.projection(node, depth + 1) {
            return self.source(projection.selected, depth + 1);
        }
        Some(node)
    }

    fn positional(&mut self, call: &Call, index: usize) -> Option<NodeId> {
        let selected = *call.args.get(index)?;
        call.args
            .iter()
            .enumerate()
            .all(|(position, &value)| position == index || self.discarded_cell(value))
            .then_some(selected)
    }

    fn discarded_cell(&mut self, node: NodeId) -> bool {
        if let Some(cuts) = self.abi_cuts {
            let m = self.module;
            let input = |id| {
                matches!(m.node(id), Node::Ref(_))
                    && cuts.abi_input(m, id)
                    && m.type_of(id).is_some_and(Type::is_concrete_value)
            };
            if input(node) {
                return true;
            }
            // Both operations only permute real cells. Their pullback only
            // permutes the zero cotangent of a discarded cell, even for NaNs.
            if !self.protected.contains(&node)
                && let Some(("transpose" | "adjoint", call)) = builtin(m, node)
                && call.named.is_empty()
                && call.inputs.is_none()
                && let [x] = *call.args
                && input(x)
                && let Some((rows, cols)) = real_matrix(m, x)
                && real_matrix(m, node) == Some((cols, rows))
            {
                return true;
            }
        }
        self.value(node, ValueUse::Discard, 0)
    }

    fn value(&mut self, node: NodeId, usage: ValueUse, depth: usize) -> bool {
        // Keeping the exact input reference preserves its boundary. Following
        // that reference to prove a constant would cross the boundary.
        if usage == ValueUse::Keep && matches!(self.module.node(node), Node::Ref(_)) {
            return true;
        }
        if depth >= MAX_PROOF_DEPTH
            || self.protected.contains(&node)
            || (usage == ValueUse::Discard && self.module.phase_of(node) != Some(Phase::Fixed))
        {
            return false;
        }
        if let Some(&valid) = self.values.get(&(node, usage)) {
            return valid;
        }
        let valid = match self.module.node(node) {
            Node::Lit(Scalar::Real(_) | Scalar::Bool(_)) => true,
            Node::Lit(Scalar::Int(value)) => self.wide_integers || i32::try_from(*value).is_ok(),
            // Dropping a string may hide a backend's missing value operation.
            Node::Lit(Scalar::Str(_)) => false,
            Node::Ref(_) => {
                let (next, binding) = resolve_ref_one(self.module, node);
                binding.is_some_and(|b| self.module.query_pinned_rhs(b).is_none())
                    && self.value(next, usage, depth + 1)
            }
            Node::Call(_) => self.value_constructor(node, usage, depth),
            _ => false,
        };
        self.values.insert((node, usage), valid);
        valid
    }

    fn value_constructor(&mut self, node: NodeId, usage: ValueUse, depth: usize) -> bool {
        let Some((head, call)) = builtin(self.module, node) else {
            return false;
        };
        if call.inputs.is_some()
            || !self
                .module
                .type_of(node)
                .is_some_and(Type::is_concrete_value)
        {
            return false;
        }
        match head {
            "vector" | "tuple" => {
                call.named.is_empty()
                    && call
                        .args
                        .iter()
                        .all(|&child| self.value(child, usage, depth + 1))
            }
            "record" => {
                call.args.is_empty()
                    && call.named.iter().all(|field| {
                        field.kind == NamedKind::Field && self.value(field.value, usage, depth + 1)
                    })
            }
            "add" | "sub" | "mul" | "neg" | "abs" | "exp" | "tanh" => {
                call.named.is_empty()
                    && self.module.type_of(node) == Some(&Type::Scalar(ScalarType::Real))
                    && call.args.iter().all(|&child| {
                        self.module.type_of(child) == Some(&Type::Scalar(ScalarType::Real))
                            && self.value(child, usage, depth + 1)
                    })
            }
            _ => false,
        }
    }
}

fn real_matrix(m: &Module, node: NodeId) -> Option<(u32, u32)> {
    let Type::Array { shape, elem } = m.type_of(node)? else {
        return None;
    };
    let [Dim::Static(rows), Dim::Static(cols)] = **shape else {
        return None;
    };
    (**elem == Type::Scalar(ScalarType::Real)).then_some((rows, cols))
}

fn rebuild(
    source: &Module,
    output: &mut Module,
    root: NodeId,
    replace: &HashMap<NodeId, NodeId>,
    rebuilt: &mut HashMap<NodeId, NodeId>,
) -> Option<NodeId> {
    let mut pending = vec![(root, false)];
    while let Some((id, ready)) = pending.pop() {
        if pending.len() > MAX_DEMAND_NODES {
            return None;
        }
        if rebuilt.contains_key(&id) {
            continue;
        }
        if let Some(&replacement) = replace.get(&id) {
            if let Some(&value) = rebuilt.get(&replacement) {
                rebuilt.insert(id, value);
            } else {
                pending.extend([(id, true), (replacement, false)]);
            }
        } else if ready {
            let mut changed = false;
            let mut children = Vec::new();
            source.for_each_child(id, |child| {
                let mapped = rebuilt[&child];
                changed |= mapped != child;
                children.push(mapped);
            });
            let new = if changed {
                copy_annotations(source, output, id, &children)
            } else {
                id
            };
            rebuilt.insert(id, new);
        } else {
            pending.push((id, true));
            source.for_each_child(id, |child| pending.push((child, false)));
        }
    }
    Some(rebuilt[&root])
}

fn copy_annotations(
    source: &Module,
    output: &mut Module,
    old: NodeId,
    children: &[NodeId],
) -> NodeId {
    let new = rebuild_with_children(output, old, children);
    if let Some(ty) = source.type_of(old) {
        output.set_type(new, ty.clone());
    }
    if let Some(phase) = source.phase_of(old) {
        output.set_phase(new, phase);
    }
    if let Some(set) = source.valueset_of(old) {
        output.set_valueset(new, set.clone());
    }
    if let Some(span) = source.span_of(old) {
        output.set_span(new, span);
    }
    new
}
