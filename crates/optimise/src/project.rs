//! Project typed expressions, retaining the original arena identity of every boundary.

use crate::engine::{self, Fact, FactArg, Term};
use crate::fixed_poisson::Partition;
use crate::selectors::{self, Selectors};
use crate::types::{TypeFacts, dimensions, fact};
use crate::{Dtype, Profile, RulePolicy, Target, concat, conjugacy, gaussian};
use flatppl_core::graph::{build_call, resolve_ref_one};
use flatppl_core::query::{Component, component};
use flatppl_core::{
    Call, CallHead, DEFAULT_MAX_DEPTH, Dim, Idx, Inputs, Mass, Module, Node, NodeId, Phase, RefNs,
    Scalar, ScalarType, Symbol, Type,
};
use std::collections::{HashMap, HashSet};

fn builtin<'a>(m: &'a Module, c: &Call) -> Option<&'a str> {
    match c.head {
        CallHead::Builtin(s) => Some(m.resolve(s)),
        _ => None,
    }
}

/// Order the original interval endpoints only when both targets represent them exactly.
fn interval_bound(terms: &[Term], id: usize) -> Option<i64> {
    let (term, negative) = match &terms[id] {
        Term::Call(head, args) if head == "neg" && args.len() == 1 => (&terms[args[0]], true),
        term => (term, false),
    };
    let value = match *term {
        Term::Integer(n) => f64::from(i32::try_from(n).ok()?),
        Term::Real(bits) => f64::from_bits(bits as u64),
        _ => return None,
    };
    if !value.is_finite() || f64::from(value as f32) != value {
        return None;
    }
    let value = if negative { -value } else { value };
    if value == 0.0 {
        return Some(0);
    }
    let bits = value.to_bits() as i64;
    Some(if bits < 0 { bits ^ i64::MAX } else { bits })
}

fn trigger(m: &Module, profile: Profile) -> bool {
    (0..m.node_count()).any(|i| {
        let Node::Call(c) = m.node(NodeId::from_usize(i)) else {
            return false;
        };
        match builtin(m, c) {
            Some(
                "identity"
                | "normalize"
                | "truncate"
                | "lengthof"
                | "transpose"
                | "adjoint"
                | "rowstack"
                | "colstack"
                | "reverse"
                | "ifelse"
                | "land"
                | "lor"
                | "lxor"
                | "lnot"
                | "equal"
                | "unequal"
                | "get"
                | "get0"
                | "kchain"
                | "builtin_touniform"
                | "builtin_tonormal"
                | "builtin_fromnormal"
                | "builtin_logdensityof"
                | "totalmass"
                | "iid"
                | "joint",
            ) => true,
            Some(
                "tile" | "addaxes" | "splitblocks" | "joinblocks" | "trace" | "diag" | "diagmat"
                | "logsumexp",
            )
            | None => matches!(profile, Profile::StableHlo { .. }),
            Some("draw") => profile == Profile::Stan,
            _ => false,
        }
    })
}

// Stan names a draw once. A real matrix reader can contract the same named
// draw with its transpose while the draw itself stays at its binding.
fn stan_matrix_reader(m: &Module, id: NodeId, profile: Profile) -> bool {
    profile == Profile::Stan
        && !m.is_sample_result(id)
        && m.sample_origin(id).is_none()
        && matches!(m.type_of(id), Some(Type::Array {shape,elem}) if shape.len()==2 &&
            **elem==Type::Scalar(flatppl_core::ScalarType::Real))
        && match m.node(id) {
            Node::Ref(r) => r.ns == RefNs::SelfMod,
            Node::Call(c) => matches!(builtin(m, c), Some("mul" | "adjoint" | "transpose")),
            _ => false,
        }
}

// Stan names a draw once and scores its law in the model block, also inside
// a kernel. The GLM rules read the law of such a draw and its linear
// predictor, whatever the phase of their parameters.
fn stan_glm_node(m: &Module, id: NodeId, profile: Profile) -> bool {
    if profile != Profile::Stan {
        return false;
    }
    let Node::Call(c) = m.node(id) else {
        return false;
    };
    if !c.named.is_empty() || c.inputs.is_some() || m.is_sample_result(id) {
        return false;
    }
    let operation = |head: NodeId, names: &[&str]| matches!(m.node(head), Node::Const(op) if names.contains(&m.resolve(*op)));
    match (builtin(m, c), c.args.as_ref()) {
        (Some("draw"), [law]) => matches!(m.node(*law), Node::Call(l)
            if builtin(m, l) == Some("broadcast")
                && l.args.first().is_some_and(|&op| operation(op, &["Normal", "Bernoulli", "Poisson"]))),
        (Some("broadcast"), [op, ..]) => operation(
            *op,
            &["Normal", "Bernoulli", "Poisson", "invlogit", "exp", "add"],
        ),
        (Some("mul"), [x, b]) => {
            let rank = |id: NodeId| match m.type_of(id) {
                Some(Type::Array { shape, .. }) => shape.len(),
                _ => 0,
            };
            rank(*x) == 2 && rank(*b) == 1
        }
        _ => false,
    }
}

/// Reified arguments and promoted ABI values replace their original expressions.
/// A literal at one of these arena addresses is a runtime input.
pub(super) struct Cuts {
    names: HashSet<Symbol>,
    nodes: HashSet<NodeId>,
    scoped_nodes: HashSet<NodeId>,
    abi_names: HashSet<Symbol>,
    abi_nodes: HashSet<NodeId>,
}

impl Cuts {
    pub(super) fn new(m: &Module, profile: Profile) -> Self {
        let abi_names = flatppl_core::query::promoted_input_names(m);
        let abi_nodes = m
            .bindings()
            .filter(|(_, b)| abi_names.contains(&b.name))
            .map(|(_, b)| b.rhs)
            .collect();
        let mut names = abi_names.clone();
        for (binding, value) in m.bindings() {
            if m.query_pinned_rhs(binding).is_some() {
                names.extend(m.alias_chain(value.name));
            }
        }
        let mut scoped = Vec::new();
        let mut live: Vec<_> = m
            .bindings()
            .map(|(_, b)| b.rhs)
            .filter(|&n| matches!(m.node(n), Node::Call(_)))
            .collect();
        let mut seen = HashSet::new();
        while let Some(id) = live.pop() {
            if !seen.insert(id) {
                continue;
            }
            let Node::Call(c) = m.node(id) else {
                continue;
            };
            m.for_each_child(id, |n| {
                if matches!(m.node(n), Node::Call(_)) {
                    live.push(n);
                }
            });
            let entries = match &c.inputs {
                Some(Inputs::Spec(args)) => Some(args.as_ref()),
                Some(Inputs::Auto) => m.auto_inputs_of(id),
                None => None,
            };
            if c.inputs.is_some() {
                scoped.extend_from_slice(&c.args);
            }
            for (_, r) in entries.into_iter().flatten() {
                if r.ns == RefNs::SelfMod {
                    names.extend(m.alias_chain(r.name));
                }
            }
        }
        let nodes: HashSet<_> = m
            .bindings()
            .filter(|(_, b)| names.contains(&b.name))
            .map(|(_, b)| b.rhs)
            .collect();
        // The same named expression may also run as a callback with leading
        // batch axes. Its top-level type does not prove the callback lowering.
        let mut visited = HashSet::new();
        let mut scoped_nodes = HashSet::new();
        while let Some(node) = scoped.pop() {
            if !nodes.contains(&node) && visited.insert(node) {
                if !stan_matrix_reader(m, node, profile) && !stan_glm_node(m, node, profile) {
                    scoped_nodes.insert(node);
                }
                m.for_each_child(node, |child| scoped.push(child));
                let (rhs, binding) = resolve_ref_one(m, node);
                if binding.is_some() {
                    scoped.push(rhs);
                }
            }
        }
        Self {
            names,
            nodes,
            scoped_nodes,
            abi_names,
            abi_nodes,
        }
    }

    pub(super) fn blocked(&self, m: &Module, id: NodeId) -> bool {
        self.scoped_nodes.contains(&id) || self.replaced(m, id)
    }

    fn replaced(&self, m: &Module, id: NodeId) -> bool {
        self.nodes.contains(&id)
            || matches!(m.node(id), Node::Ref(r)
            if r.ns == RefNs::SelfMod && self.names.contains(&r.name))
    }

    pub(super) fn abi_input(&self, m: &Module, id: NodeId) -> bool {
        self.abi_nodes.contains(&id)
            || matches!(m.node(id),Node::Ref(r)
            if r.ns==RefNs::SelfMod && self.abi_names.contains(&r.name))
    }

    pub(super) fn resolve(&self, m: &Module, id: NodeId) -> Option<NodeId> {
        self.follow(m, id, true)
    }

    fn follow(&self, m: &Module, mut id: NodeId, respect_scope: bool) -> Option<NodeId> {
        for _ in 0..=m.binding_count() {
            if self.replaced(m, id) || (respect_scope && self.scoped_nodes.contains(&id)) {
                return None;
            }
            let (next, binding) = resolve_ref_one(m, id);
            if binding.is_none() {
                return Some(id);
            }
            id = next;
        }
        None
    }
}

/// The emitter also shares every spelling of a field of the same input.
/// Selecting a field reads that field, whereas passing the whole aggregate
/// to another consumer adds a read to every descendant.
fn storage(m: &Module, cuts: &Cuts, mut node: NodeId) -> Option<(NodeId, Vec<Component>)> {
    let mut path = Vec::new();
    for _ in 0..DEFAULT_MAX_DEPTH {
        let id = flatppl_core::graph::resolve_ref(m, node).0;
        if !cuts.replaced(m, node)
            && !cuts.replaced(m, id)
            && let Some((base, part)) = component(m, id)
        {
            path.push(part);
            node = base;
        } else {
            path.reverse();
            return Some((id, path));
        }
    }
    None
}

// Two separately authored computations may share one emitted SSA value.
// A query use count proves exclusive access only for identifiable storage.
fn input_storage(m: &Module, cuts: &Cuts, node: NodeId) -> bool {
    storage(m, cuts, node).is_some_and(|(source, _)| {
        cuts.abi_input(m, source)
            || matches!(m.node(source), Node::Call(call)
            if matches!(builtin(m, call), Some("external" | "elementof")))
    })
}

fn query_uses(
    m: &Module,
    cuts: &Cuts,
    roots: &[NodeId],
    origins: impl Iterator<Item = NodeId>,
) -> Option<Vec<usize>> {
    let mut uses = HashMap::new();
    let mut seen = HashSet::new();
    let mut work: Vec<_> = roots.iter().map(|&id| (id, true)).collect();
    while let Some((node, whole)) = work.pop() {
        let id = flatppl_core::graph::resolve_ref(m, node).0;
        if whole {
            *uses.entry(storage(m, cuts, node)?).or_insert(0usize) += 1;
        }
        if seen.insert(id) && !cuts.replaced(m, node) && !cuts.replaced(m, id) {
            let base = component(m, id).map(|(base, _)| base);
            m.for_each_child(id, |c| {
                // Selector proofs consume the use counts of vector producers.
                // A scalar literal in a long index vector cannot be one.
                if matches!(m.node(c), Node::Call(_) | Node::Ref(_)) {
                    work.push((c, Some(c) != base));
                }
            });
            if let Node::Call(call) = m.node(id) {
                let entries = match &call.inputs {
                    Some(Inputs::Spec(entries)) => Some(entries.as_ref()),
                    Some(Inputs::Auto) => m.auto_inputs_of(id),
                    None => None,
                };
                for (_, r) in entries.into_iter().flatten() {
                    if r.ns == RefNs::SelfMod
                        && let Some(b) = m.binding_by_name(r.name)
                    {
                        work.push((m.binding(b).rhs, true));
                    }
                }
            }
        }
    }
    origins
        .map(|node| {
            let mut key = storage(m, cuts, node)?;
            let mut count = 0;
            loop {
                count += uses.get(&key).copied().unwrap_or(0);
                if key.1.pop().is_none() {
                    break;
                }
            }
            Some(count)
        })
        .collect()
}

struct Project<'a> {
    m: &'a Module,
    cuts: Cuts,
    profile: Profile,
    terms: Vec<Term>,
    indices: HashMap<Term, usize>,
    by_node: HashMap<NodeId, usize>,
    refs: HashMap<Symbol, usize>,
    origins: Vec<NodeId>,
    types: TypeFacts,
    facts: Vec<Fact>,
    selectors: Selectors,
    concats: Vec<(usize, concat::Witness, Vec<usize>)>,
    poisson: Vec<(usize, usize, Partition)>,
    fixed_poisson: bool,
    glm_operands: HashSet<NodeId>,
}

impl<'a> Project<'a> {
    fn new(m: &'a Module, profile: Profile, sufficient_statistics: bool) -> Self {
        Self {
            m,
            cuts: Cuts::new(m, profile),
            profile,
            terms: vec![],
            indices: HashMap::new(),
            by_node: HashMap::new(),
            refs: HashMap::new(),
            origins: vec![],
            types: TypeFacts::default(),
            facts: vec![],
            selectors: Selectors::default(),
            concats: vec![],
            poisson: vec![],
            glm_operands: (0..m.node_count())
                .map(NodeId::from_usize)
                .filter(|&id| stan_glm_node(m, id, profile))
                .flat_map(|id| m.node(id).children())
                .collect(),
            fixed_poisson: sufficient_statistics
                && matches!(
                    profile,
                    Profile::StableHlo {
                        target: Target::Cpu,
                        ..
                    }
                ),
        }
    }

    fn hlo(&self) -> bool {
        matches!(self.profile, Profile::StableHlo { .. })
    }

    /// A read of a named data value or of one of its record fields by name.
    /// Both are data in a target that reads named data.
    fn reads_data_boundary(&self, id: NodeId) -> bool {
        let m = self.m;
        let named = |id: NodeId| match m.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => m
                .binding_by_name(r.name)
                .is_some_and(|binding| crate::boundary::data_boundary(m, self.profile, binding)),
            _ => false,
        };
        match m.node(id) {
            Node::Call(c) if matches!(builtin(m, c), Some("get" | "get_field")) => {
                c.named.is_empty()
                    && c.inputs.is_none()
                    && matches!(*c.args, [record, key]
                        if named(record) && matches!(m.node(key), Node::Lit(Scalar::Str(_))))
            }
            _ => named(id),
        }
    }

    fn transposed_cells(&self) -> Vec<[usize; 3]> {
        let m = self.m;
        let mut sites = Vec::new();
        for (cell, term) in self.terms.iter().enumerate() {
            let Term::Call(head, args) = term else {
                continue;
            };
            let [transposed, _, _] = args.as_slice() else {
                continue;
            };
            let Term::Call(op, operands) = &self.terms[*transposed] else {
                continue;
            };
            let [matrix] = operands.as_slice() else {
                continue;
            };
            if !matches!(head.as_str(), "get" | "get0")
                || !matches!(op.as_str(), "transpose" | "adjoint")
            {
                continue;
            }
            let Node::Call(original) = m.node(self.origins[cell]) else {
                continue;
            };
            if !original.args[1..].iter().all(|&id| {
                !self.cuts.blocked(m, id)
                    && !m.is_sample_result(id)
                    && matches!(m.node(id), Node::Lit(Scalar::Int(_)))
            }) {
                continue;
            }
            if input_storage(m, &self.cuts, self.origins[*matrix]) {
                sites.push([cell, *transposed, *matrix]);
            }
        }
        sites
    }

    fn literal_rows(&self, id: NodeId) -> Option<Vec<i64>> {
        let Node::Call(c) = self.m.node(id) else {
            return None;
        };
        if builtin(self.m, c) != Some("vector")
            || !c.named.is_empty()
            || c.inputs.is_some()
            || !matches!(self.m.type_of(id),Some(Type::Array {shape,elem})
                if matches!(shape.as_ref(),[Dim::Static(n)] if *n as usize==c.args.len()) &&
                    **elem==Type::Scalar(flatppl_core::ScalarType::Integer))
        {
            return None;
        }
        c.args
            .iter()
            .map(|&id| match self.m.node(id) {
                Node::Lit(Scalar::Int(v))
                    if !self.cuts.blocked(self.m, id)
                        && !self.m.is_sample_result(id)
                        && self.m.sample_origin(id).is_none() =>
                {
                    Some(*v)
                }
                _ => None,
            })
            .collect()
    }

    fn call_name(&self, c: &Call) -> Option<String> {
        if !c.named.is_empty() || c.inputs.is_some() || !(1..=3).contains(&c.args.len()) {
            return None;
        }
        let m = self.m;
        if let Some(name) = builtin(m, c) {
            match name {
                "tile" | "splitblocks" if self.hlo() => {
                    let [_, size] = *c.args else {
                        return None;
                    };
                    let positive = |a: NodeId| {
                        matches!(m.node(a),Node::Lit(Scalar::Int(n)) if *n>0)
                            && !self.cuts.blocked(m, a)
                    };
                    match m.node(size) {
                        Node::Call(v)
                            if builtin(m, v) == Some("vector")
                                && v.named.is_empty()
                                && !self.cuts.blocked(m, size)
                                && v.args.iter().all(|&a| positive(a)) => {}
                        _ if positive(size) => {}
                        _ => return None,
                    }
                }
                "addaxes" if self.hlo() => {
                    let [_, left, right] = *c.args else {
                        return None;
                    };
                    if ![left,right].into_iter().all(|n|!self.cuts.blocked(m,n) &&
                        matches!(m.node(resolve_ref_one(m,n).0),Node::Lit(Scalar::Int(v)) if *v>=0)) {
                        return None;
                    }
                }
                "logsumexp" if self.hlo() && c.args.len() == 1 => {
                    let arg = self.cuts.resolve(m, c.args[0]).unwrap_or(c.args[0]);
                    let (one, _) = resolve_ref_one(m, c.args[0]);
                    if let Some(Type::Array { shape, .. }) = m.type_of(arg)
                        && matches!(shape.as_ref(), [Dim::Static(2)])
                    {
                        if self.cuts.blocked(m, c.args[0]) || self.cuts.blocked(m, arg) {
                            return None;
                        }
                        let Node::Call(pair) = m.node(one) else {
                            return None;
                        };
                        return (arg == one
                            && builtin(m, pair) == Some("vector")
                            && pair.args.len() == 2
                            && pair.named.is_empty()
                            && pair
                                .args
                                .iter()
                                .all(|&n| matches!(m.type_of(n), Some(Type::Scalar(_)))))
                        .then(|| "logsumexp.pair".to_owned());
                    }
                }
                // Stan reads a broadcast law or link by its name. The operation
                // is part of the head, so rules can match it.
                "broadcast"
                    if self.profile == Profile::Stan
                        && let Some(&op) = c.args.first()
                        && let Node::Const(op) = m.node(op)
                        && matches!(
                            m.resolve(*op),
                            "add" | "Normal" | "Bernoulli" | "Poisson" | "invlogit" | "exp"
                        ) =>
                {
                    return Some(format!("broadcast.{}", m.resolve(*op)));
                }
                // Scoped calls and target callbacks have their own lane and status contracts.
                "broadcast"
                    if c.args.first().is_some_and(|&a| {
                        matches!(m.node(a), Node::Const(op)
                    if matches!(m.resolve(*op),"add"|"sub"|"mul"|"div"|"neg"))
                    }) => {}
                "identity" | "normalize" | "truncate" | "interval" | "cat" | "densityof"
                | "logdensityof" | "totalmass" | "lengthof" | "transpose" | "adjoint"
                | "reverse" | "get" | "get0" | "vector" | "row_gram" | "col_gram"
                | "self_outer" | "diagmat" | "diag" | "sum" | "trace" | "tile" | "splitblocks"
                | "joinblocks" | "addaxes" | "rowstack" | "colstack" | "neg" | "add" | "sub"
                | "mul" | "div" | "pow" | "lnot" | "land" | "lor" | "lxor" | "equal"
                | "unequal" | "lt" | "le" | "gt" | "ge" | "ifelse" | "isfinite" | "isinf"
                | "isnan" | "iid" | "joint" | "cartprod" | "eye" | "zeros" | "fill" => {}
                _ => return None,
            }
            return Some(name.to_owned());
        }
        if self.hlo() && c.args.len() == 1 && self.fixed_infinity(c.args[0]) {
            let CallHead::User(fun) = c.head else {
                return None;
            };
            let Node::Ref(r) = m.node(fun) else {
                return None;
            };
            let RefNs::Module(alias) = r.ns else {
                return None;
            };
            let load = m.binding(m.binding_by_name(alias)?).rhs;
            let load = flatppl_core::graph::expect_builtin_call(m, load, "standard_module")?;
            if m.resolve(r.name) == "log_ndtr"
                && matches!(load.args.as_ref(),[n,v]
                if matches!(m.node(*n),Node::Lit(Scalar::Str(s)) if &**s=="special-functions") &&
                    matches!(m.node(*v),Node::Lit(Scalar::Str(s)) if &**s=="0.1"))
            {
                return Some("special-functions.log_ndtr".to_owned());
            }
        }
        None
    }

    fn fixed_infinity(&self, mut id: NodeId) -> bool {
        for _ in 0..DEFAULT_MAX_DEPTH {
            if self.cuts.blocked(self.m, id) {
                return false;
            }
            match self.m.node(id) {
                Node::Const(s) => return self.m.resolve(*s) == "inf",
                Node::Lit(Scalar::Real(v)) => return v.is_infinite(),
                Node::Call(c)
                    if builtin(self.m, c) == Some("neg")
                        && c.args.len() == 1
                        && c.named.is_empty() =>
                {
                    id = c.args[0]
                }
                _ => return false,
            }
        }
        false
    }

    fn project(&mut self, id: NodeId, depth: usize) -> Option<usize> {
        if let Some(&i) = self.by_node.get(&id) {
            return Some(i);
        }
        if depth >= DEFAULT_MAX_DEPTH {
            return None;
        }
        let m = self.m;
        let boundary = self.cuts.blocked(m, id);
        let sampled = m.is_sample_result(id) || m.sample_origin(id).is_some();
        let stochastic = m.phase_of(id) == Some(Phase::Stochastic) || sampled;
        let stan_matrix = stan_matrix_reader(m, id, self.profile);
        let stan_glm = stan_glm_node(m, id, self.profile);
        if let Node::Ref(r) = m.node(id)
            && r.ns == RefNs::SelfMod
        {
            if let Some(&i) = self.refs.get(&r.name) {
                self.by_node.insert(id, i);
                return Some(i);
            }
            if self.hlo()
                && !boundary
                && !stochastic
                && let Some(rhs) = self.cuts.resolve(m, id)
                && rhs != id
            {
                let i = self.project(rhs, depth + 1)?;
                self.by_node.insert(id, i);
                self.refs.insert(r.name, i);
                return Some(i);
            }
        }
        let allowed = !boundary && (!stochastic || stan_matrix || stan_glm);
        let term = match m.node(id) {
            Node::Lit(Scalar::Bool(b)) if allowed => Term::Boolean(*b),
            Node::Lit(Scalar::Real(x)) if allowed => Term::Real(x.to_bits() as i64),
            Node::Lit(Scalar::Int(x))
                if allowed
                    && (matches!(
                        self.profile,
                        Profile::StableHlo {
                            dtype: Dtype::F64,
                            ..
                        }
                    ) || i32::try_from(*x).is_ok()) =>
            {
                Term::Integer(*x)
            }
            Node::Const(s) if allowed && m.resolve(*s) == "inf" => {
                Term::Real(f64::INFINITY.to_bits() as i64)
            }
            Node::Call(c) if allowed && let Some(head) = self.call_name(c) => {
                let named_operation = usize::from(head.starts_with("broadcast."));
                let args = c.args[named_operation..]
                    .iter()
                    .map(|&a| self.project(a, depth + 1))
                    .collect::<Option<_>>()?;
                Term::Call(head, args)
            }
            _ => Term::Leaf(id.index()),
        };
        // Distinct physical numeric operations may accumulate adjoints in different orders.
        // Keep a separate occurrence opaque, even when its operands are identical.
        let term = if matches!(term, Term::Call(_, _)) && self.indices.contains_key(&term) {
            Term::Leaf(id.index())
        } else {
            term
        };
        if let Some(&i) = self.indices.get(&term) {
            self.by_node.insert(id, i);
            return Some(i);
        }
        let i = self.terms.len();
        self.indices.insert(term.clone(), i);
        self.terms.push(term);
        self.origins.push(id);
        self.by_node.insert(id, i);
        if let Node::Ref(r) = m.node(id)
            && r.ns == RefNs::SelfMod
        {
            self.refs.insert(r.name, i);
        }
        if self.reads_data_boundary(id) {
            self.facts.push(fact("data-boundary", i, []));
        }
        if (stan_glm || self.glm_operands.contains(&id))
            && let Some(ty) = m.type_of(id)
        {
            match ty {
                Type::Scalar(ScalarType::Real) => self.facts.push(fact("shape-real", i, [])),
                Type::Array { shape, elem }
                    if **elem == Type::Scalar(ScalarType::Real)
                        && let [Dim::Static(n)] = **shape =>
                {
                    self.facts
                        .push(fact("shape-real-vector", i, [FactArg::Int(i64::from(n))]));
                }
                _ => {}
            }
        }
        let term = &self.terms[i];
        let input = self.cuts.abi_input(m, id);
        let stored_integer = input
            || matches!(term, Term::Integer(_))
            || matches!(term, Term::Call(h,_) if h=="lengthof");
        let integer = !self.hlo() || stored_integer;
        if (!stochastic || stan_matrix)
            && let Some(ty) = m.type_of(id)
        {
            self.types
                .seed(&mut self.facts, ty, i, integer, self.profile);
            if stored_integer
                && !matches!(term, Term::Integer(_))
                && *ty == Type::Scalar(ScalarType::Integer)
            {
                self.facts.push(fact("portable-stored-integer", i, []));
            }
        }
        if !stochastic
            && self.hlo()
            && (input || matches!(term, Term::Integer(_)))
            && m.type_of(id).is_some_and(Type::is_concrete_value)
        {
            self.facts.push(fact("erasable", i, []));
        }
        if allowed && let Some(rows) = self.literal_rows(id) {
            if rows.iter().all(|&n| i32::try_from(n).is_ok()) {
                self.facts.push(fact("portable-discardable", i, []));
            }
            if self.hlo() {
                if rows.len() <= 16 {
                    self.facts
                        .push(fact("math-int-list", i, [dimensions(rows.iter().copied())]));
                }
                self.selectors.insert(i, rows);
            }
        }
        if allowed {
            if let Term::Call(head, args) = term {
                match (head.as_str(), args.as_slice()) {
                    ("interval", [low, high]) => {
                        for &bound in [low, high] {
                            if let Some(key) = interval_bound(&self.terms, bound) {
                                self.facts
                                    .push(fact("measure-bound", bound, [FactArg::Int(key)]));
                            }
                        }
                    }
                    ("truncate", [base, _])
                        if matches!(m.type_of(self.origins[*base]),
                        Some(Type::Measure { domain, mass }) if *mass != Mass::Deferred &&
                            matches!(domain.as_ref(), Type::Scalar(ScalarType::Integer | ScalarType::Real))) =>
                    {
                        self.facts.push(fact("measure-scalar", *base, []));
                    }
                    _ => {}
                }
            }
            if self.fixed_poisson
                && let Term::Call(head, children) = term
                && head == "sum"
                && let [cells] = children.as_slice()
                && let Some(partition) = Partition::prove(m, id, |n| self.cuts.resolve(m, n))
                && self.cuts.abi_input(m, partition.source)
            {
                let cells = *cells;
                let source = self.project(partition.source, depth + 1)?;
                self.poisson.push((cells, source, partition));
            }
            // The matcher reads the original fields of one certified kernel.
            // It can inspect its written scale, but never a replaced value.
            conjugacy::seed(m, id, i, |n| self.cuts.follow(m, n, false), &mut self.facts);
            // Portable output keeps authored bindings. Read their restrictions
            // as evidence for score queries without replacing the measure name.
            if !self.hlo()
                && matches!(m.node(id), Node::Ref(r) if r.ns == RefNs::SelfMod)
                && matches!(m.type_of(id), Some(Type::Measure { domain, .. })
                    if matches!(domain.as_ref(), Type::Scalar(ScalarType::Integer | ScalarType::Real)))
                && let Some(rhs) = self.cuts.resolve(m, id)
                && let Node::Call(call) = m.node(rhs)
                && matches!(builtin(m, call), Some("truncate" | "normalize"))
            {
                let value = self.project(rhs, depth + 1)?;
                self.facts
                    .push(fact("measure-alias", i, [FactArg::Expr(value)]));
            }
            if let Some(witness) = concat::witness(m, id, |n| self.cuts.resolve(m, n)) {
                let args = witness
                    .parts
                    .iter()
                    .map(|&n| self.project(n, depth + 1))
                    .collect::<Option<Vec<_>>>()?;
                self.concats.push((i, witness, args));
            }
            if let Some((head, vector)) =
                concat::fixed_stack_vector(m, id, |n| self.cuts.resolve(m, n))
            {
                let source = self.project(vector, depth + 1)?;
                self.facts.push(fact(
                    "fixed-singleton-stack",
                    i,
                    [FactArg::Str(head), FactArg::Expr(source)],
                ));
            }
            if let Some(witness)=gaussian::witness(m,id) && let Node::Call(call)=m.node(id) &&
                // A promoted record or kernel replaces the structure from which
                // these fields were read. Individual fields still project as inputs.
                (!builtin(m,call).is_some_and(|h|h.starts_with("builtin_")) ||
                    call.args[..2].iter().all(|&n|!self.cuts.blocked(m,n)))
            {
                let args = witness
                    .nodes
                    .iter()
                    .map(|&n| self.project(n, depth + 1))
                    .collect::<Option<Vec<_>>>()?;
                gaussian::seed(&witness, i, &args, &mut self.facts, self.profile);
            }
        }
        Some(i)
    }

    /// Build only the selected DAG. Proofs may mention extra constructors
    /// and selectors that belong to a losing alternative.
    fn rebuild(
        &self,
        m: &mut Module,
        roots: &[usize],
        extraction: &engine::Extraction,
    ) -> Option<Vec<Option<NodeId>>> {
        let winners = &extraction.roots;
        let original = self.origins.len();
        let mut live = HashSet::new();
        let mut stack = winners.clone();
        while let Some(i) = stack.pop() {
            if i >= original
                && live.insert(i)
                && let Term::Call(_, args) = &self.terms[i]
            {
                stack.extend(args);
            }
        }
        let mut created = HashMap::new();
        let resolve = |i: usize, created: &HashMap<usize, NodeId>| {
            self.origins
                .get(i)
                .copied()
                .or_else(|| created.get(&i).copied())
        };
        for (i, term) in self.terms.iter().enumerate().skip(original) {
            if !live.contains(&i) {
                continue;
            }
            let id = match term {
                Term::Leaf(key) => {
                    if *key < self.m.node_count() + self.poisson.len() {
                        // The plan key is interpreted by its only consumer.
                        continue;
                    }
                    let rows = self.selectors.synthetic(*key)?;
                    let args: Vec<_> = rows
                        .iter()
                        .map(|&n| m.alloc(Node::Lit(Scalar::Int(n))))
                        .collect();
                    build_call(m, "vector", &args)
                }
                Term::Boolean(b) => m.alloc(Node::Lit(Scalar::Bool(*b))),
                Term::Integer(n) => m.alloc(Node::Lit(Scalar::Int(*n))),
                Term::Real(bits) => m.alloc(Node::Lit(Scalar::Real(f64::from_bits(*bits as u64)))),
                Term::Call(head, args) if head == "poisson.grouped-score" => {
                    let [source, plan] = *args.as_slice() else {
                        return None;
                    };
                    let source = resolve(source, &created)?;
                    let Term::Leaf(key) = self.terms[plan] else {
                        return None;
                    };
                    let (_, _, partition) =
                        self.poisson.get(key.checked_sub(self.m.node_count())?)?;
                    partition.rebuild(m, source)
                }
                Term::Call(head, args) => {
                    let args: Vec<_> = args
                        .iter()
                        .map(|&n| resolve(n, &created))
                        .collect::<Option<_>>()?;
                    match head.as_str() {
                        "special-functions.log_ndtr" => {
                            let module = flatppl_core::graph::special_functions_alias(m);
                            let name = m.intern("log_ndtr");
                            let fun = m.alloc(Node::Ref(flatppl_core::Ref {
                                ns: RefNs::Module(module),
                                name,
                            }));
                            m.alloc(Node::Call(Call {
                                head: CallHead::User(fun),
                                args: args.into(),
                                named: Box::new([]),
                                inputs: None,
                            }))
                        }
                        "logsumexp.pair" => build_call(m, "logsumexp", &args),
                        head if let Some(member) = head.strip_prefix("regression.") => {
                            let module = flatppl_core::graph::standard_module_alias(
                                m,
                                "regression",
                                "0.1",
                                "__regression",
                            );
                            let name = m.intern(member);
                            let fun = m.alloc(Node::Ref(flatppl_core::Ref {
                                ns: RefNs::Module(module),
                                name,
                            }));
                            m.alloc(Node::Call(Call {
                                head: CallHead::User(fun),
                                args: args.into(),
                                named: Box::new([]),
                                inputs: None,
                            }))
                        }
                        head if let Some(op) = head.strip_prefix("broadcast.") => {
                            let op = m.intern(op);
                            let op = m.alloc(Node::Const(op));
                            let mut operands = vec![op];
                            operands.extend(args);
                            build_call(m, "broadcast", &operands)
                        }
                        head => build_call(m, head, &args),
                    }
                }
            };
            created.insert(i, id);
        }
        // A refusal inside a rebuilt value still points at the written text.
        // A class with no written member takes the span of its consumer.
        let mut spans: Vec<_> = created.keys().copied().collect();
        spans.sort_unstable_by(|a, b| b.cmp(a));
        for i in spans {
            let id = created[&i];
            if let Some(&source) = extraction.sources.get(&i)
                && let Some(span) = self.m.span_of(self.origins[source])
            {
                m.set_span(id, span);
            }
            let (Some(span), Term::Call(_, args)) = (m.span_of(id), &self.terms[i]) else {
                continue;
            };
            for arg in args {
                if let Some(&child) = created.get(arg)
                    && m.span_of(child).is_none()
                {
                    m.set_span(child, span);
                }
            }
        }
        roots
            .iter()
            .zip(winners)
            .map(|(old, new)| {
                if old == new {
                    Some(None)
                } else {
                    Some(Some(resolve(*new, &created)?))
                }
            })
            .collect()
    }
}

pub(super) fn optimise(
    m: &Module,
    names: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
    sufficient_statistics: bool,
    rules: &RulePolicy,
) -> Option<Module> {
    if rules.is_all_disabled() || !trigger(m, profile) {
        return None;
    }
    let mut project = Project::new(m, profile, sufficient_statistics);
    let mut selected: HashSet<_> = names.into_iter().flatten().copied().collect();
    if names.is_some() && !project.hlo() {
        let mut work: Vec<_> = selected.iter().copied().collect();
        while let Some(name) = work.pop() {
            if let Some(bid) = m.binding_by_name(name) {
                let mut deps = HashSet::new();
                m.collect_referenced_names(m.binding(bid).rhs, &mut deps);
                work.extend(deps.into_iter().filter(|s| selected.insert(*s)));
            }
        }
    }
    let bindings: Vec<_> = m
        .bindings()
        .filter(|(bid, b)| {
            (names.is_none() || selected.contains(&b.name))
                && !project.cuts.names.contains(&b.name)
                && !project.cuts.blocked(m, b.rhs)
                && !flatppl_core::query::is_inputs_binding(m, *bid)
        })
        .map(|(bid, b)| (bid, b.rhs))
        .collect();
    let outputs = crate::outputs::Outputs::collect(
        m,
        &project.cuts,
        project.hlo(),
        profile == Profile::Stan,
        &bindings,
    )?;
    let roots: Vec<_> = outputs
        .slots
        .iter()
        .map(|&id| project.project(id, 0))
        .collect::<Option<_>>()?;
    for (bid, b) in m.bindings() {
        if crate::boundary::data_boundary(m, profile, bid)
            && let Some(&i) = project.by_node.get(&b.rhs)
        {
            project.facts.push(fact("data-boundary", i, []));
        }
    }
    let transposed_cells = project.transposed_cells();
    let affine_transports = gaussian::partial_transports(&project.facts);
    let vector_selectors = match profile {
        Profile::StableHlo { dtype, .. }
            if !project.selectors.is_empty()
                && project.terms.iter().any(
                    |term| matches!(term,Term::Call(name,_) if name=="get" || name=="get0"),
                ) =>
        {
            Some(dtype)
        }
        _ => None,
    };
    if !transposed_cells.is_empty()
        || vector_selectors.is_some()
        || !project.poisson.is_empty()
        || !project.concats.is_empty()
        || !affine_transports.is_empty()
    {
        let readers = project
            .origins
            .iter()
            .copied()
            .chain(project.concats.iter().map(|(_, w, _)| w.cat))
            .chain(
                project
                    .poisson
                    .iter()
                    .flat_map(|(_, _, partition)| partition.readers),
            );
        let counts = query_uses(
            m,
            &project.cuts,
            &bindings.iter().map(|&(_, id)| id).collect::<Vec<_>>(),
            readers,
        )?;
        let (uses, other) = counts.split_at(project.origins.len());
        let (cat_uses, poisson_uses) = other.split_at(project.concats.len());
        for (site, point) in affine_transports {
            if uses[point] == 1 && project.cuts.abi_input(m, project.origins[point]) {
                gaussian::seed_exclusive_transport(&mut project.facts, site);
            }
        }
        for ((read, witness, parts), &cat) in project.concats.iter().zip(cat_uses) {
            let counts: Vec<_> = [uses[*read], cat]
                .into_iter()
                .chain(parts.iter().map(|&part| {
                    if input_storage(m, &project.cuts, project.origins[part]) {
                        uses[part]
                    } else {
                        0
                    }
                }))
                .collect();
            concat::seed(witness, *read, parts, &counts, &mut project.facts);
        }
        for [cell, transposed, matrix] in transposed_cells {
            if [cell, transposed, matrix]
                .into_iter()
                .all(|id| uses[id] == 1)
            {
                project.facts.push(fact(
                    "portable-exclusive-transposed-cell",
                    cell,
                    [FactArg::Expr(transposed), FactArg::Expr(matrix)],
                ));
            }
        }
        if let Some(dtype) = vector_selectors {
            selectors::seed(
                &mut project.selectors,
                &mut project.terms,
                &mut project.facts,
                uses,
                m.node_count() + project.poisson.len(),
                dtype,
            );
        }
        let (groups, _) = poisson_uses.as_chunks::<4>();
        for (key, ((cells, source, partition), uses)) in
            project.poisson.iter().zip(groups).enumerate()
        {
            if uses.iter().all(|&count| count == 1) {
                project
                    .facts
                    .push(partition.fact(*cells, *source, m.node_count() + key));
            }
        }
    }
    let winners = engine::optimise(&mut project.terms, &roots, &project.facts, profile, rules)?;
    let mut out = m.clone();
    let chosen = project.rebuild(&mut out, &roots, &winners)?;
    outputs.write(&mut out, &chosen);
    let errors = flatppl_infer::infer_module(&mut out, bundle, flatppl_infer::Level::Shape);
    if errors
        .iter()
        .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return None;
    }
    for (bid, original) in bindings {
        let new = out.binding(bid).rhs;
        if m.type_of(original) != out.type_of(new)
            || ((!matches!(profile, Profile::StableHlo { .. })
                || m.phase_of(original) == Some(Phase::Stochastic))
                && m.phase_of(original) != out.phase_of(new))
        {
            return None;
        }
    }
    Some(out)
}
