//! Read-only structural facts over a lowered query and its declared boundary.

use std::collections::{HashMap, HashSet};
use std::fmt;

use flatppl_core::{
    Call, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar, ScalarType, Symbol, Type,
    ValueSet,
};

/// One additive output term. Node IDs refer to the module passed to analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputTerm {
    pub node: NodeId,
    pub sign: i8,
    pub multiplicity: usize,
    pub parameters: Vec<String>,
    pub data: Vec<String>,
}

/// An advisory scalar Normal coordinate map, never an automatic rewrite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalCoordinate {
    pub parameter: String,
    pub location: NodeId,
    pub scale: NodeId,
    pub location_parents: Vec<String>,
    pub scale_parents: Vec<String>,
    /// False means the map requires a finite runtime location check.
    pub finite_location_proven: bool,
    /// False means the map requires a positive finite runtime scale check.
    pub positive_finite_scale_proven: bool,
    /// A cycle prevents composing these maps as a triangular joint map.
    pub cyclic: bool,
}

/// Structural components describe term incidence, not posterior independence.
/// Arrays and constrained values remain whole inputs. No evaluator cache or
/// query ABI changes accompany this report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelAnalysis {
    pub parameters: Vec<String>,
    pub data: Vec<String>,
    pub terms: Vec<OutputTerm>,
    pub components: Vec<Vec<String>>,
    pub unused_parameters: Vec<String>,
    pub normal_coordinates: Vec<NormalCoordinate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisError(pub String);

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for AnalysisError {}

fn call<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a Call> {
    match m.node(id) {
        Node::Call(c) if matches!(c.head,CallHead::Builtin(h) if m.resolve(h)==name) => Some(c),
        _ => None,
    }
}

fn binding(m: &Module, name: &str) -> Option<NodeId> {
    m.bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .map(|(_, b)| b.rhs)
}

fn resolve(m: &Module, mut id: NodeId, boundary: &HashSet<Symbol>) -> NodeId {
    let mut seen = HashSet::new();
    while seen.insert(id) {
        let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        else {
            break;
        };
        if boundary.contains(name) {
            break;
        }
        let Some(bid) = m.binding_by_name(*name) else {
            break;
        };
        id = m.binding(bid).rhs;
    }
    id
}

struct Dependencies<'a> {
    module: &'a Module,
    boundary: HashSet<Symbol>,
    aliases: HashMap<Symbol, Symbol>,
    alias_names: HashSet<Symbol>,
    order: Vec<Symbol>,
    cache: HashMap<NodeId, HashSet<Symbol>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Reference {
    Current(Symbol),
    Local(Symbol),
    Module(Symbol, Symbol),
}

impl From<Ref> for Reference {
    fn from(value: Ref) -> Self {
        match value.ns {
            RefNs::SelfMod => Self::Current(value.name),
            RefNs::Local => Self::Local(value.name),
            RefNs::Module(module) => Self::Module(module, value.name),
        }
    }
}

impl Dependencies<'_> {
    fn get(&mut self, root: NodeId) -> HashSet<Symbol> {
        if let Some(value) = self.cache.get(&root) {
            return value.clone();
        }
        let m = self.module;
        let mut pending = vec![(root, Vec::<Reference>::new(), false)];
        let mut seen = HashSet::new();
        let mut active = HashSet::new();
        let mut found = HashSet::new();
        while let Some((id, locals, exiting)) = pending.pop() {
            let key = (id, locals.clone());
            if exiting {
                active.remove(&key);
                continue;
            }
            if active.contains(&key) {
                found.extend(self.boundary.iter().copied());
                continue;
            }
            if !seen.insert(key.clone()) {
                continue;
            }
            active.insert(key);
            pending.push((id, locals.clone(), true));
            match m.node(id) {
                Node::Ref(r) if locals.contains(&Reference::from(*r)) => {}
                Node::Ref(Ref {
                    ns: RefNs::SelfMod,
                    name,
                }) => {
                    if let Some(&declared) = self.aliases.get(name) {
                        found.insert(declared);
                    } else if let Some(bid) = m.binding_by_name(*name) {
                        pending.push((m.binding(bid).rhs, locals, false));
                    } else {
                        found.extend(self.boundary.iter().copied());
                    }
                }
                Node::Call(c) => {
                    let mut locals = locals;
                    if let Some(Inputs::Spec(inputs)) = &c.inputs {
                        for (_, reference) in inputs.iter() {
                            let reference = Reference::from(*reference);
                            if !locals.contains(&reference) {
                                locals.push(reference);
                            }
                        }
                    }
                    pending.extend(
                        m.node(id)
                            .children()
                            .into_iter()
                            .map(|child| (child, locals.clone(), false)),
                    );
                }
                Node::Ref(_) => found.extend(self.boundary.iter().copied()),
                _ => {}
            }
        }
        self.cache.insert(root, found.clone());
        found
    }

    fn names(&self, symbols: &HashSet<Symbol>) -> Vec<String> {
        self.order
            .iter()
            .filter(|s| symbols.contains(s))
            .map(|s| self.module.resolve(*s).to_string())
            .collect()
    }
}

fn additive_terms(
    m: &Module,
    root: NodeId,
    boundary: &HashSet<Symbol>,
) -> Vec<(NodeId, i8, usize)> {
    let mut pending = vec![(root, 1_i8)];
    let mut terms = Vec::new();
    let mut positions = HashMap::new();
    let mut steps = 0;
    while let Some((node, sign)) = pending.pop() {
        steps += 1;
        // Repeated DAG additions can denote exponentially many terms. Keep
        // the whole output opaque when a bounded report cannot expand it.
        if steps > 4096 {
            return vec![(root, 1, 1)];
        }
        let id = resolve(m, node, boundary);
        let scalar = matches!(
            m.type_of(id),
            Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
        );
        if scalar {
            if let Some(c) = call(m, id, "add").or_else(|| call(m, id, "sub"))
                && let [a, b] = *c.args
            {
                pending.push((
                    b,
                    if call(m, id, "sub").is_some() {
                        -sign
                    } else {
                        sign
                    },
                ));
                pending.push((a, sign));
                continue;
            }
            if let Some(c) = call(m, id, "sum")
                && let [values] = *c.args
                && let Some(v) = call(m, resolve(m, values, boundary), "vector")
                && v.args.iter().all(|&x| {
                    matches!(
                        m.type_of(x),
                        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
                    )
                })
            {
                pending.extend(v.args.iter().rev().map(|&x| (x, sign)));
                continue;
            }
        }
        let index = *positions.entry((id, sign)).or_insert_with(|| {
            let index = terms.len();
            terms.push((id, sign, 0));
            index
        });
        terms[index].2 += 1;
    }
    terms
}

fn known_number(m: &Module, id: NodeId, depth: usize) -> Option<f64> {
    if depth > 64 {
        return None;
    }
    match m.node(id) {
        Node::Lit(Scalar::Real(value)) => Some(*value),
        Node::Lit(Scalar::Int(value)) => Some(*value as f64),
        Node::Const(name) if m.resolve(*name) == "inf" => Some(f64::INFINITY),
        _ => {
            let c = call(m, id, "neg")?;
            let [value] = *c.args else { return None };
            Some(-known_number(m, value, depth + 1)?)
        }
    }
}

fn normal_coordinate(deps: &mut Dependencies<'_>, node: NodeId) -> Option<NormalCoordinate> {
    let m = deps.module;
    let c = call(m, node, "builtin_logdensityof")?;
    let [family, parameters, point] = *c.args else {
        return None;
    };
    if !matches!(m.node(family),Node::Const(name) if m.resolve(*name)=="Normal") {
        return None;
    }
    let point = resolve(m, point, &deps.alias_names);
    let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = *m.node(point)
    else {
        return None;
    };
    let name = *deps.aliases.get(&name)?;
    if !deps.boundary.contains(&name)
        || m.type_of(point) != Some(&Type::Scalar(ScalarType::Real))
        || m.valueset_of(point) != Some(&ValueSet::Reals)
    {
        return None;
    }
    let parameters = call(m, resolve(m, parameters, &deps.alias_names), "record")?;
    let location = parameters
        .named
        .iter()
        .find(|a| m.resolve(a.name) == "mu")?
        .value;
    let scale = parameters
        .named
        .iter()
        .find(|a| m.resolve(a.name) == "sigma")?
        .value;
    if !matches!(
        m.type_of(location),
        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
    ) || !matches!(
        m.type_of(scale),
        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
    ) {
        return None;
    }
    let location_deps = deps.get(location);
    let scale_deps = deps.get(scale);
    if location_deps.contains(&name) || scale_deps.contains(&name) {
        return None;
    }
    let scale_value = resolve(m, scale, &deps.alias_names);
    let location_value = resolve(m, location, &deps.alias_names);
    if known_number(m, location_value, 0).is_some_and(|value| !value.is_finite())
        || known_number(m, scale_value, 0).is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return None;
    }
    let finite_location_proven = known_number(m, location_value, 0).is_some_and(f64::is_finite)
        || matches!(m.valueset_of(location_value),Some(ValueSet::Interval(lo,hi)) if lo.is_finite() && hi.is_finite());
    let proven = match m.node(scale_value) {
        Node::Lit(Scalar::Real(x)) => *x > 0. && x.is_finite(),
        Node::Lit(Scalar::Int(x)) => *x > 0,
        _ => {
            matches!(m.valueset_of(scale_value),Some(ValueSet::Interval(lo,hi)) if *lo>0. && lo.is_finite() && hi.is_finite())
        }
    };
    Some(NormalCoordinate {
        parameter: m.resolve(name).to_string(),
        location,
        scale,
        location_parents: deps.names(&location_deps),
        scale_parents: deps.names(&scale_deps),
        finite_location_proven,
        positive_finite_scale_proven: proven,
        cyclic: false,
    })
}

/// Analyze a FlatPDL query without changing it. `data_inputs` must be an
/// explicit subset of declared inputs. Every other input is a parameter.
pub fn analyze_model(m: &Module, data_inputs: &[String]) -> Result<ModelAnalysis, AnalysisError> {
    crate::is_flatpdl(m)
        .map_err(|e| AnalysisError(format!("model analysis requires FlatPDL: {e:?}")))?;
    let inputs = binding(m, "inputs")
        .ok_or_else(|| AnalysisError("model analysis requires declared inputs".into()))?;
    let outputs = binding(m, "outputs")
        .ok_or_else(|| AnalysisError("model analysis requires outputs".into()))?;
    let values = call(m, inputs, "tuple").map_or_else(|| vec![inputs], |c| c.args.to_vec());
    let mut order = Vec::new();
    for id in values {
        let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = *m.node(id)
        else {
            return Err(AnalysisError("query inputs must name bindings".into()));
        };
        if order.contains(&name) {
            return Err(AnalysisError(
                "query inputs contain a duplicate binding".into(),
            ));
        }
        order.push(name);
    }
    let mut data = HashSet::new();
    for name in data_inputs {
        let Some(&symbol) = order.iter().find(|s| m.resolve(**s) == name) else {
            return Err(AnalysisError(format!(
                "data input {name:?} is not a declared input"
            )));
        };
        if !data.insert(symbol) {
            return Err(AnalysisError(format!("data input {name:?} occurs twice")));
        }
    }
    let mut aliases = HashMap::new();
    for &declared in &order {
        for name in m.alias_chain(declared) {
            if aliases
                .insert(name, declared)
                .is_some_and(|previous| previous != declared)
            {
                return Err(AnalysisError("query inputs alias the same object".into()));
            }
        }
    }
    let mut deps = Dependencies {
        module: m,
        boundary: order.iter().copied().collect(),
        alias_names: aliases.keys().copied().collect(),
        aliases,
        order,
        cache: HashMap::new(),
    };
    let parameter_set = deps
        .boundary
        .difference(&data)
        .copied()
        .collect::<HashSet<_>>();
    let parameters = deps.names(&parameter_set);
    let data_names = deps.names(&data);
    let mut terms = Vec::new();
    let mut used = HashSet::new();
    let mut normal_coordinates = Vec::new();
    for (node, sign, multiplicity) in additive_terms(m, outputs, &deps.alias_names) {
        let all = deps.get(node);
        let ps = all.intersection(&parameter_set).copied().collect();
        let ds = all.intersection(&data).copied().collect();
        used.extend(&ps);
        terms.push(OutputTerm {
            node,
            sign,
            multiplicity,
            parameters: deps.names(&ps),
            data: deps.names(&ds),
        });
        if sign == 1
            && multiplicity == 1
            && let Some(candidate) = normal_coordinate(&mut deps, node)
            && parameters.contains(&candidate.parameter)
            && !normal_coordinates
                .iter()
                .any(|c: &NormalCoordinate| c.parameter == candidate.parameter)
        {
            normal_coordinates.push(candidate)
        }
    }
    let mut components = parameters
        .iter()
        .map(|p| vec![p.clone()])
        .collect::<Vec<_>>();
    for term in &terms {
        let mut indices = components
            .iter()
            .enumerate()
            .filter(|(_, c)| c.iter().any(|p| term.parameters.contains(p)))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if indices.len() > 1 {
            let first = indices.remove(0);
            for i in indices.into_iter().rev() {
                let extra = components.remove(i);
                components[first].extend(extra);
            }
        }
    }
    for component in &mut components {
        component.sort_by_key(|p| parameters.iter().position(|x| x == p));
    }
    let candidate_names = normal_coordinates
        .iter()
        .map(|c| c.parameter.clone())
        .collect::<HashSet<_>>();
    let edges = normal_coordinates
        .iter()
        .map(|c| {
            (
                c.parameter.clone(),
                c.location_parents
                    .iter()
                    .chain(&c.scale_parents)
                    .filter(|p| candidate_names.contains(*p))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    for candidate in &mut normal_coordinates {
        let mut pending = edges[&candidate.parameter].clone();
        let mut seen = HashSet::new();
        while let Some(name) = pending.pop() {
            if name == candidate.parameter {
                candidate.cyclic = true;
                break;
            }
            if seen.insert(name.clone()) {
                pending.extend(edges[&name].iter().cloned());
            }
        }
    }
    let unused = parameter_set.difference(&used).copied().collect();
    Ok(ModelAnalysis {
        parameters,
        data: data_names,
        terms,
        components,
        unused_parameters: deps.names(&unused),
        normal_coordinates,
    })
}

impl fmt::Display for ModelAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Structural parameter components: {:?}", self.components)?;
        writeln!(f, "Unused parameters: {:?}", self.unused_parameters)?;
        for term in &self.terms {
            writeln!(
                f,
                "Term {:?}: sign {}, multiplicity {}, parameters {:?}, data {:?}",
                term.node, term.sign, term.multiplicity, term.parameters, term.data
            )?;
        }
        for c in &self.normal_coordinates {
            writeln!(
                f,
                "Normal coordinate {}: theta = location + scale*z; z = (theta-location)/scale; forward log Jacobian = log(scale); location {:?}, scale {:?}; location parents {:?}, scale parents {:?}; finite location proven {}; positive finite scale proven {}; cyclic {}",
                c.parameter,
                c.location,
                c.scale,
                c.location_parents,
                c.scale_parents,
                c.finite_location_proven,
                c.positive_finite_scale_proven,
                c.cyclic
            )?;
            if !c.finite_location_proven {
                writeln!(f, "  Required check: finite(location)")?;
            }
            if !c.positive_finite_scale_proven {
                writeln!(f, "  Required check: positive_finite(scale)")?;
            }
        }
        Ok(())
    }
}
