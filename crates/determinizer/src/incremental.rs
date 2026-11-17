//! Explicit term queries and an ordinary combiner over cached scalar results.

use std::collections::{HashMap, HashSet};

use flatppl_core::{Binding, CallHead, Module, Node, NodeId, Ref, RefNs, ScalarType, Symbol, Type};

use crate::{AnalysisError, analyze_model};

/// One complete term, evaluated once even when the output uses it repeatedly.
#[derive(Clone, Debug)]
pub struct IncrementalTerm {
    pub dependencies: Vec<String>,
    pub input_paths: Vec<crate::InputPath>,
    pub input_cells: Vec<crate::InputCell>,
    pub cache_input: String,
    root: NodeId,
}

/// A source snapshot, its independent term queries and the original output tree
/// over cached term values. This plan owns no evaluator or mutable data cache.
pub struct IncrementalScorePlan {
    pub inputs: Vec<String>,
    pub input_paths: Vec<crate::InputPath>,
    pub input_arrays: Vec<crate::InputArray>,
    pub terms: Vec<IncrementalTerm>,
    pub combine: Module,
    source: Module,
    input_nodes: Vec<NodeId>,
}

fn binding(m: &Module, name: &str) -> Option<NodeId> {
    m.bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .map(|(_, b)| b.rhs)
}

fn signature(m: &Module) -> Vec<NodeId> {
    let Some(input) = binding(m, "inputs") else {
        return Vec::new();
    };
    match m.node(input) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(name) if m.resolve(name) == "tuple") => {
            c.args.to_vec()
        }
        _ => vec![input],
    }
}

fn set_binding(m: &mut Module, name: &str, rhs: NodeId) {
    let name = m.intern(name);
    if let Some(bid) = m.binding_by_name(name) {
        m.set_binding_rhs(bid, rhs);
    } else {
        m.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: true,
            synthetic: true,
        });
    }
}

fn set_inputs(m: &mut Module, inputs: &[NodeId]) {
    if inputs.is_empty() {
        let keep = m
            .bindings()
            .filter(|(_, b)| m.resolve(b.name) != "inputs")
            .map(|(bid, _)| bid)
            .collect();
        m.retain_bindings(&keep);
    } else {
        let rhs = if let [input] = inputs {
            *input
        } else {
            crate::density::build_call(m, "tuple", inputs)
        };
        set_binding(m, "inputs", rhs);
    }
}

fn retain_query(m: &mut Module) {
    let roots = [m.intern("inputs"), m.intern("outputs")];
    crate::canon::retain_reachable(m, &roots);
}

fn resolve(m: &Module, mut id: NodeId, boundary: &HashSet<Symbol>) -> NodeId {
    let mut seen = HashSet::new();
    while seen.insert(id) {
        let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = *m.node(id)
        else {
            break;
        };
        if boundary.contains(&name) {
            break;
        }
        let Some(bid) = m.binding_by_name(name) else {
            break;
        };
        id = m.binding(bid).rhs;
    }
    id
}

/// Replace term nodes without reassociating arithmetic or cancelling signs.
/// Iterative rebuilding also preserves sharing in deep additive DAGs.
fn combine(
    m: &mut Module,
    root: NodeId,
    boundary: &HashSet<Symbol>,
    replacements: &HashMap<NodeId, NodeId>,
) -> Result<NodeId, AnalysisError> {
    let mut rebuilt = replacements.clone();
    let mut active = HashSet::new();
    let mut pending = vec![(resolve(m, root, boundary), false)];
    while let Some((id, exiting)) = pending.pop() {
        if rebuilt.contains_key(&id) {
            continue;
        }
        if !exiting {
            if !active.insert(id) {
                return Err(AnalysisError("incremental output contains a cycle".into()));
            }
            pending.push((id, true));
            for child in m.node(id).children().into_iter().rev() {
                let child = resolve(m, child, boundary);
                if !rebuilt.contains_key(&child) {
                    pending.push((child, false));
                }
            }
            continue;
        }
        active.remove(&id);
        let mut node = m.node(id).clone();
        let replacement = |child| rebuilt[&resolve(m, child, boundary)];
        if let Node::Call(c) = &mut node {
            if let CallHead::User(head) = &mut c.head {
                *head = replacement(*head);
            }
            for arg in &mut c.args {
                *arg = replacement(*arg);
            }
            for arg in &mut c.named {
                arg.value = replacement(arg.value);
            }
        }
        let ty = m.type_of(id).cloned();
        let value = m.alloc(node);
        if let Some(ty) = ty {
            m.set_type(value, ty);
        }
        rebuilt.insert(id, value);
    }
    Ok(rebuilt[&resolve(m, root, boundary)])
}

/// Split a lowered scalar query without changing its original ABI or output.
/// Freeze file-backed constants before calling this function. Re-evaluate a term
/// after any named dependency changes, including in-place edits to array inputs.
pub fn prepare_incremental_score(
    m: &Module,
    data_inputs: &[String],
) -> Result<IncrementalScorePlan, AnalysisError> {
    let report = analyze_model(m, data_inputs)?;
    let output = binding(m, "outputs")
        .ok_or_else(|| AnalysisError("incremental scoring requires outputs".into()))?;
    if !matches!(
        m.type_of(output),
        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
    ) {
        return Err(AnalysisError(
            "incremental scoring requires a real or integer scalar output".into(),
        ));
    }
    let input_nodes = signature(m);
    let input_symbols = input_nodes
        .iter()
        .map(|id| match m.node(*id) {
            Node::Ref(reference) => reference.name,
            _ => unreachable!("analysis validated the input boundary"),
        })
        .collect::<Vec<_>>();
    let inputs = input_symbols
        .iter()
        .map(|&name| m.resolve(name).to_owned())
        .collect::<Vec<_>>();
    let aliases = input_symbols
        .iter()
        .flat_map(|&declared| {
            m.alias_chain(declared)
                .into_iter()
                .map(move |name| (name, declared))
        })
        .collect::<HashMap<_, _>>();
    let boundary = aliases.keys().copied().collect();
    let mut combine_query = m.clone();
    let mut terms = Vec::new();
    let mut replacements = HashMap::new();
    let mut cache_inputs = Vec::new();
    let mut input_results = HashMap::new();
    for term in report.terms {
        if replacements.contains_key(&term.node) {
            continue;
        }
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = *m.node(term.node)
            && let Some(&reference) =
                input_results.get(&aliases.get(&name).copied().unwrap_or(name))
        {
            replacements.insert(term.node, reference);
            continue;
        }
        let ty = m
            .type_of(term.node)
            .ok_or_else(|| AnalysisError("an incremental term has no type".into()))?;
        let set = match ty {
            Type::Scalar(ScalarType::Real) => "reals",
            Type::Scalar(ScalarType::Integer) => "integers",
            _ => {
                return Err(AnalysisError(
                    "incremental terms must be real or integer scalars".into(),
                ));
            }
        };
        let mut serial = terms.len();
        let name = loop {
            let name = combine_query.intern(&format!("__incremental_term_{serial}"));
            if combine_query.binding_by_name(name).is_none() {
                break name;
            }
            serial += 1;
        };
        let set = combine_query.intern(set);
        let set = combine_query.alloc(Node::Const(set));
        let rhs = crate::density::build_call(&mut combine_query, "external", &[set]);
        combine_query.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: true,
            synthetic: true,
        });
        let reference = combine_query.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }));
        combine_query.set_type(reference, ty.clone());
        replacements.insert(term.node, reference);
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = *m.node(term.node)
        {
            input_results.insert(aliases.get(&name).copied().unwrap_or(name), reference);
        }
        cache_inputs.push(reference);
        let dependencies = inputs
            .iter()
            .filter(|name| term.parameters.contains(name) || term.data.contains(name))
            .cloned()
            .collect();
        terms.push(IncrementalTerm {
            root: term.node,
            dependencies,
            input_paths: term.input_paths,
            input_cells: term.input_cells,
            cache_input: combine_query.resolve(name).to_owned(),
        });
    }
    let output = combine(&mut combine_query, output, &boundary, &replacements)?;
    set_binding(&mut combine_query, "outputs", output);
    set_inputs(&mut combine_query, &cache_inputs);
    retain_query(&mut combine_query);
    Ok(IncrementalScorePlan {
        inputs,
        input_paths: report.input_paths,
        input_arrays: report.input_arrays,
        terms,
        combine: combine_query,
        source: m.clone(),
        input_nodes,
    })
}

impl IncrementalScorePlan {
    /// Construct one query at a time, so callers can serialize or compile terms
    /// without retaining a full source-arena clone for each term.
    pub fn term_query(&self, index: usize) -> Result<Module, AnalysisError> {
        let term = self
            .terms
            .get(index)
            .ok_or_else(|| AnalysisError("incremental term index is out of range".into()))?;
        let mut query = self.source.clone();
        let inputs = self
            .input_nodes
            .iter()
            .zip(&self.inputs)
            .filter(|(_, name)| term.dependencies.contains(name))
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        set_binding(&mut query, "outputs", term.root);
        set_inputs(&mut query, &inputs);
        retain_query(&mut query);
        Ok(query)
    }
}
