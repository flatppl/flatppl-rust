//! Explicit scalar and triangular Normal companions over the shared lowered query.

use std::collections::{HashMap, HashSet};

use flatppl_core::graph;
use flatppl_core::{
    Binding, CallHead, Idx, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol,
};

use crate::{AnalysisError, InputPath, analyze_lowered_query};
use flatppl_core::Type;

/// Arithmetic precision used to validate a coordinate map. Emit every companion
/// with this precision. These checks describe a point, not a global float bijection.
#[derive(Clone, Copy, Debug)]
pub enum CoordinatePrecision {
    F32,
    F64,
}

/// Ordinary FlatPDL queries. Each returns `(value, valid)` independently of any
/// backend runtime status. Discard invalid results rather than treating them as
/// zero probability. The original query remains unchanged.
pub struct NormalCoordinateQueries {
    pub score: Module,
    pub forward: Module,
    pub inverse: Module,
    pub log_jacobian: Module,
    pub coordinate_input: String,
    pub precision: CoordinatePrecision,
}

/// A triangular map over explicitly selected real paths. Argument slots keep
/// their original names. Maps return selected roots in original ABI order,
/// preserving every unselected sibling. Each result also carries validity.
pub struct NormalCoordinatesQueries {
    pub score: Module,
    pub forward: Module,
    pub inverse: Module,
    pub log_jacobian: Module,
    pub coordinate_inputs: Vec<String>,
    pub paths: Vec<InputPath>,
    pub precision: CoordinatePrecision,
}

fn used_names(m: &Module) -> HashSet<Symbol> {
    let mut used = m.bindings().map(|(_, b)| b.name).collect::<HashSet<_>>();
    for index in 0..m.node_count() {
        let id = NodeId::from_usize(index);
        match m.node(id) {
            Node::Ref(r) => {
                used.insert(r.name);
            }
            Node::Call(c) => {
                let entries = match &c.inputs {
                    Some(Inputs::Spec(inputs)) => Some(inputs.as_ref()),
                    Some(Inputs::Auto) => m.auto_inputs_of(id),
                    None => None,
                };
                for (name, r) in entries.into_iter().flatten() {
                    used.insert(*name);
                    used.insert(r.name);
                }
            }
            _ => {}
        }
    }
    used
}

fn private_value(m: &mut Module, value: NodeId, used: &mut HashSet<Symbol>) -> NodeId {
    let name = (used.len()..)
        .map(|i| m.intern(&format!("__normal_value_{i}")))
        .find(|name| !used.contains(name))
        .unwrap();
    used.insert(name);
    m.add_binding(Binding {
        name,
        rhs: value,
        doc: None,
        public: false,
        synthetic: true,
    });
    m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }))
}

// The printer emits expression trees. Keep shared global scalar guards/maps
// as bindings so composing round trips cannot expand them exponentially.
// Never lift a retained callback's body out of its lexical scope.
fn share_joint_expressions(m: &mut Module) {
    let inputs = binding(m, "inputs").unwrap();
    let inputs = call(m, inputs, "tuple").unwrap_or_else(|| vec![inputs]);
    let boundary: HashSet<_> = inputs
        .iter()
        .flat_map(|id| {
            let Node::Ref(r) = m.node(*id) else {
                unreachable!()
            };
            m.alias_chain(r.name)
        })
        .collect();
    let bindings: Vec<_> = m
        .bindings()
        .filter(|(_, b)| !boundary.contains(&b.name) && m.resolve(b.name) != "inputs")
        .map(|(bid, b)| (bid, b.rhs))
        .collect();
    let mut used = used_names(m);
    let mut memo = HashMap::new();
    for (bid, root) in bindings {
        let mut pending = vec![(root, false)];
        while let Some((id, exiting)) = pending.pop() {
            if memo.contains_key(&id) {
                continue;
            }
            if !matches!(m.node(id), Node::Call(c) if c.inputs.is_none()
                && !matches!(c.head, CallHead::Builtin(name) if matches!(m.resolve(name), "aggregate" | "metricsum")))
            {
                memo.insert(id, id);
                continue;
            }
            let children = m.node(id).children();
            if !exiting {
                pending.push((id, true));
                for child in children.into_iter().rev() {
                    if !memo.contains_key(&child) {
                        pending.push((child, false));
                    }
                }
                continue;
            }
            let changed: Vec<_> = children.iter().map(|child| memo[child]).collect();
            let value = if children == changed {
                id
            } else {
                graph::rebuild_with_children(m, id, &changed)
            };
            // Primitive density arguments and ABI tuples retain explicit
            // structural constructors. Share their scalar children instead.
            let value = if matches!(m.node(value), Node::Call(c) if matches!(c.head, CallHead::Builtin(name) if matches!(m.resolve(name), "record" | "tuple")))
            {
                value
            } else {
                private_value(m, value, &mut used)
            };
            memo.insert(id, value);
        }
        m.set_binding_rhs(bid, memo[&root]);
    }
}

fn finish_joint(mut m: Module, output: NodeId, valid: NodeId) -> Result<Module, AnalysisError> {
    let output = op(&mut m, "tuple", &[output, valid]);
    set(&mut m, "outputs", output);
    share_joint_expressions(&mut m);
    validate(m)
}

fn call(m: &Module, id: NodeId, name: &str) -> Option<Vec<NodeId>> {
    match m.node(id) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if m.resolve(h) == name) => {
            Some(c.args.to_vec())
        }
        _ => None,
    }
}

fn op(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    graph::build_call(m, name, args)
}

fn number(m: &mut Module, value: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(value)))
}

fn project(m: &mut Module, mut root: NodeId, fields: &[String]) -> NodeId {
    for field in fields {
        let key = m.alloc(Node::Lit(Scalar::Str(field.clone().into())));
        root = op(m, "get", &[root, key]);
    }
    root
}

fn replace_field(
    m: &mut Module,
    root: NodeId,
    ty: &Type,
    fields: &[String],
    value: NodeId,
) -> NodeId {
    let Some((field, rest)) = fields.split_first() else {
        return value;
    };
    let Type::Record(entries) = ty else {
        unreachable!("analysis checked the record path")
    };
    let mut result = Vec::new();
    for (name, ty) in entries.iter() {
        let name_text = m.resolve(*name).to_owned();
        let child = project(m, root, std::slice::from_ref(&name_text));
        let child = if &name_text == field {
            replace_field(m, child, ty, rest, value)
        } else {
            child
        };
        result.push((*name, child));
    }
    graph::build_record(m, &result)
}

fn matches_point(
    m: &Module,
    mut id: NodeId,
    selected: Symbol,
    fields: &[String],
    boundary: &HashSet<Symbol>,
) -> bool {
    let mut path = Vec::new();
    let mut seen = HashSet::new();
    while seen.insert(id) {
        match m.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if boundary.contains(&r.name) {
                    path.reverse();
                    return m.alias_target(r.name) == m.alias_target(selected) && path == fields;
                }
                let Some(binding) = m.binding_by_name(r.name) else {
                    return false;
                };
                id = m.binding(binding).rhs;
            }
            Node::Call(c) if matches!(c.head,CallHead::Builtin(h) if m.resolve(h)=="get") => {
                let [receiver, key] = *c.args else {
                    return false;
                };
                let Node::Lit(Scalar::Str(field)) = m.node(key) else {
                    return false;
                };
                path.push(field.to_string());
                id = receiver;
            }
            _ => return false,
        }
    }
    false
}

fn set(m: &mut Module, name: &str, rhs: NodeId) {
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

fn binding(m: &Module, name: &str) -> Option<NodeId> {
    m.bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .map(|(_, b)| b.rhs)
}

fn finish(mut m: Module, output: NodeId, valid: NodeId) -> Result<Module, AnalysisError> {
    let output = op(&mut m, "tuple", &[output, valid]);
    set(&mut m, "outputs", output);
    validate(m)
}

fn validate(mut m: Module) -> Result<Module, AnalysisError> {
    let roots = [m.intern("inputs"), m.intern("outputs")];
    graph::retain_reachable(&mut m, &roots);
    let errors = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(m)
    } else {
        Err(AnalysisError(errors.join("; ")))
    }
}

fn both(m: &mut Module, values: &[NodeId]) -> NodeId {
    values
        .iter()
        .copied()
        .reduce(|a, b| op(m, "land", &[a, b]))
        .unwrap()
}

fn close(m: &mut Module, a: NodeId, b: NodeId, tolerance: f64) -> NodeId {
    let difference = op(m, "sub", &[a, b]);
    let difference = op(m, "abs", &[difference]);
    let size = op(m, "abs", &[b]);
    let one = number(m, 1.0);
    let size = op(m, "max", &[one, size]);
    let tolerance = number(m, tolerance);
    let limit = op(m, "mul", &[tolerance, size]);
    op(m, "le", &[difference, limit])
}

/// Rewrite the query slice, stopping at every other ABI input and every lexical
/// binder. Leave authored binding RHSs unchanged so promoted fixed inputs keep
/// their type/phase even when their original expression mentions the coordinate.
fn substitute(
    m: &mut Module,
    root: NodeId,
    boundary: &HashSet<Symbol>,
    selected: &HashSet<Symbol>,
    value: NodeId,
) -> Result<NodeId, AnalysisError> {
    substitute_inputs(
        m,
        root,
        boundary,
        &selected.iter().map(|&name| (name, value)).collect(),
        true,
    )
}

// Inserted expressions stay opaque, including their original input references.
// This is simultaneous substitution, not repeated application of a transport.
fn substitute_inputs(
    m: &mut Module,
    root: NodeId,
    boundary: &HashSet<Symbol>,
    values: &HashMap<Symbol, NodeId>,
    guard_captures: bool,
) -> Result<NodeId, AnalysisError> {
    let mut scopes = vec![Vec::<Symbol>::new()];
    let mut memo = HashMap::new();
    let mut active = HashSet::new();
    let mut pending = vec![(root, 0, false)];
    while let Some((id, scope, exiting)) = pending.pop() {
        let key = (id, scope);
        if memo.contains_key(&key) {
            continue;
        }
        if !exiting {
            if !active.insert(key) {
                return Err(AnalysisError(
                    "the coordinate query contains a cycle".into(),
                ));
            }
            if let Node::Ref(r) = *m.node(id) {
                if r.ns != RefNs::SelfMod
                    || scopes[scope].contains(&r.name)
                    || boundary.contains(&r.name)
                {
                    memo.insert(key, id);
                    active.remove(&key);
                    continue;
                }
                if let Some(&value) = values.get(&r.name) {
                    if guard_captures && crosses_cut(m, value, &scopes[scope]) {
                        return Err(AnalysisError("a generated Normal map capture crosses a retained callback graph boundary".into()));
                    }
                    memo.insert(key, value);
                    active.remove(&key);
                    continue;
                }
            }
            pending.push((id, scope, true));
        }
        let mut locals = scopes[scope].clone();
        if let Node::Call(c) = m.node(id) {
            let entries = match &c.inputs {
                Some(Inputs::Spec(inputs)) => Some(inputs.as_ref()),
                Some(Inputs::Auto) => return Err(AnalysisError(
                    "coordinate preparation requires explicit boundaries for retained automatic callbacks".into()
                )),
                None => None,
            };
            for (_, r) in entries.into_iter().flatten() {
                if r.ns == RefNs::SelfMod && !locals.contains(&r.name) {
                    locals.push(r.name);
                }
            }
        }
        let child_scope = if let Some(index) = scopes.iter().position(|s| *s == locals) {
            index
        } else {
            scopes.push(locals);
            scopes.len() - 1
        };
        let children = if let Node::Ref(r) = m.node(id) {
            m.binding_by_name(r.name)
                .map(|bid| vec![m.binding(bid).rhs])
                .unwrap_or_default()
        } else {
            m.node(id).children()
        };
        if !exiting {
            for child in children.into_iter().rev() {
                if !memo.contains_key(&(child, child_scope)) {
                    pending.push((child, child_scope, false));
                }
            }
            continue;
        }
        let changed = children
            .iter()
            .map(|child| memo[&(*child, child_scope)])
            .collect::<Vec<_>>();
        let result = if children == changed {
            id
        } else if matches!(m.node(id), Node::Ref(_)) {
            changed[0]
        } else {
            graph::rebuild_with_children(m, id, &changed)
        };
        memo.insert(key, result);
        active.remove(&key);
    }
    Ok(memo[&(root, 0)])
}

// Reification cuts ancestry, not just printed names. A newly inserted map must
// not acquire a bound ancestor through another supplied ABI input or alias.
// Refuse that case rather than changing already-authored callback semantics.
fn crosses_cut(m: &Module, value: NodeId, cuts: &[Symbol]) -> bool {
    if cuts.is_empty() {
        return false;
    }
    let cuts: HashSet<_> = cuts.iter().flat_map(|name| m.alias_chain(*name)).collect();
    let mut pending = vec![(value, Vec::<Symbol>::new())];
    let mut seen = HashSet::new();
    while let Some((id, mut locals)) = pending.pop() {
        if !seen.insert((id, locals.clone())) {
            continue;
        }
        if let Node::Ref(r) = m.node(id) {
            if r.ns != RefNs::SelfMod || locals.contains(&r.name) {
                continue;
            }
            if cuts.contains(&r.name) {
                return true;
            }
            if let Some(bid) = m.binding_by_name(r.name) {
                pending.push((m.binding(bid).rhs, locals));
            }
            continue;
        }
        if let Node::Call(c) = m.node(id) {
            match &c.inputs {
                Some(Inputs::Spec(entries)) => {
                    for (_, r) in entries.iter() {
                        if r.ns == RefNs::SelfMod {
                            locals.extend(m.alias_chain(r.name));
                        }
                    }
                }
                Some(Inputs::Auto) => return true,
                None => {}
            }
        }
        pending.extend(
            m.node(id)
                .children()
                .into_iter()
                .map(|child| (child, locals.clone())),
        );
    }
    false
}

fn limits(precision: CoordinatePrecision) -> (f64, f64, f64) {
    match precision {
        CoordinatePrecision::F32 => (
            (f64::from(f32::MIN_POSITIVE) * 16.).sqrt(),
            (f64::from(f32::MAX) / 16.).sqrt(),
            8. * f64::from(f32::EPSILON),
        ),
        CoordinatePrecision::F64 => (
            (f64::MIN_POSITIVE * 16.).sqrt(),
            (f64::MAX / 16.).sqrt(),
            32. * f64::EPSILON,
        ),
    }
}

struct NormalParameters {
    mu: NodeId,
    sigma: NodeId,
    jacobian: NodeId,
    valid: NodeId,
    zero: NodeId,
    one: NodeId,
}

fn normal_parameters(
    m: &mut Module,
    location: NodeId,
    scale: NodeId,
    precision: CoordinatePrecision,
) -> NormalParameters {
    let (minimum, maximum, _) = limits(precision);
    let zero = number(m, 0.);
    let one = number(m, 1.);
    let mu_ok = op(m, "isfinite", &[location]);
    let sigma_ok = op(m, "isfinite", &[scale]);
    let minimum = number(m, minimum);
    let maximum = number(m, maximum);
    let lo_ok = op(m, "ge", &[scale, minimum]);
    let hi_ok = op(m, "le", &[scale, maximum]);
    let sigma_ok = both(m, &[sigma_ok, lo_ok, hi_ok]);
    let valid = both(m, &[mu_ok, sigma_ok]);
    let mu = op(m, "ifelse", &[mu_ok, location, zero]);
    let sigma = op(m, "ifelse", &[sigma_ok, scale, one]);
    let jacobian = op(m, "log", &[sigma]);
    NormalParameters {
        mu,
        sigma,
        jacobian,
        valid,
        zero,
        one,
    }
}

fn forward_map(
    m: &mut Module,
    p: &NormalParameters,
    z: NodeId,
    precision: CoordinatePrecision,
) -> (NodeId, NodeId) {
    let z_ok = op(m, "isfinite", &[z]);
    let safe_z = op(m, "ifelse", &[z_ok, z, p.zero]);
    let scaled = op(m, "mul", &[p.sigma, safe_z]);
    let theta = op(m, "add", &[p.mu, scaled]);
    let theta_ok = op(m, "isfinite", &[theta]);
    let safe_theta = op(m, "ifelse", &[theta_ok, theta, p.zero]);
    let difference = op(m, "sub", &[safe_theta, p.mu]);
    let recovered = op(m, "divide", &[difference, p.sigma]);
    let recovered_ok = op(m, "isfinite", &[recovered]);
    let roundtrip = close(m, recovered, safe_z, limits(precision).2);
    let valid = both(m, &[p.valid, z_ok, theta_ok, recovered_ok, roundtrip]);
    (safe_theta, valid)
}

fn inverse_map(
    m: &mut Module,
    p: &NormalParameters,
    point: NodeId,
    precision: CoordinatePrecision,
) -> (NodeId, NodeId) {
    let point_ok = op(m, "isfinite", &[point]);
    let point = op(m, "ifelse", &[point_ok, point, p.zero]);
    let delta = op(m, "sub", &[point, p.mu]);
    let inverse = op(m, "divide", &[delta, p.sigma]);
    let inverse_ok = op(m, "isfinite", &[inverse]);
    let scaled = op(m, "mul", &[p.sigma, inverse]);
    let restored = op(m, "add", &[p.mu, scaled]);
    let restored_ok = op(m, "isfinite", &[restored]);
    let roundtrip = scaled_roundtrip(m, restored, point, p, inverse, precision);
    let valid = both(m, &[p.valid, point_ok, inverse_ok, restored_ok, roundtrip]);
    (inverse, valid)
}

fn scaled_roundtrip(
    m: &mut Module,
    restored: NodeId,
    point: NodeId,
    p: &NormalParameters,
    z: NodeId,
    precision: CoordinatePrecision,
) -> NodeId {
    let error = op(m, "sub", &[restored, point]);
    let error = op(m, "divide", &[error, p.sigma]);
    let relative = op(m, "abs", &[z]);
    let relative = op(m, "max", &[p.one, relative]);
    let tolerance = number(m, limits(precision).2);
    let relative = op(m, "mul", &[relative, tolerance]);
    let error = op(m, "abs", &[error]);
    op(m, "le", &[error, relative])
}

/// Build one conditional map, holding every other declared input fixed. No
/// joint composition or automatic sampler change occurs. Location and scale
/// must exclude the selected input. Multiple direct Normal factors are ambiguous.
/// Freeze fixed file data before calling this function.
pub fn prepare_normal_coordinate(
    m: &Module,
    data_inputs: &[String],
    parameter: &str,
    precision: CoordinatePrecision,
    logdensity: bool,
) -> Result<NormalCoordinateQueries, AnalysisError> {
    prepare_normal_coordinate_at_path(
        m,
        data_inputs,
        &InputPath {
            input: parameter.into(),
            fields: vec![],
        },
        precision,
        logdensity,
    )
}

/// Select one real record field while retaining the original record ABI.
/// The selected field contains z in score/forward/Jacobian inputs. Other fields
/// retain their supplied values. Forward and inverse return the whole record.
pub fn prepare_normal_coordinate_at_path(
    m: &Module,
    data_inputs: &[String],
    path: &InputPath,
    precision: CoordinatePrecision,
    logdensity: bool,
) -> Result<NormalCoordinateQueries, AnalysisError> {
    let parameter = &path.input;
    let report = analyze_lowered_query(m, data_inputs)?;
    let candidate = report
        .normal_coordinates
        .iter()
        .find(|c| &c.parameter == parameter && c.fields == path.fields)
        .ok_or_else(|| {
            AnalysisError("the selected input has no scalar Normal coordinate".into())
        })?;
    let inputs = binding(m, "inputs")
        .ok_or_else(|| AnalysisError("coordinate queries require inputs".into()))?;
    let mut inputs = call(m, inputs, "tuple").unwrap_or_else(|| vec![inputs]);
    let selected = inputs
        .iter()
        .position(|&id| matches!(m.node(id),Node::Ref(r) if m.resolve(r.name)==parameter))
        .unwrap();
    let Node::Ref(selected_ref) = *m.node(inputs[selected]) else {
        unreachable!()
    };
    let input_type = m.type_of(inputs[selected]).unwrap().clone();
    let all_boundary = inputs
        .iter()
        .flat_map(|id| {
            let Node::Ref(r) = m.node(*id) else {
                unreachable!()
            };
            m.alias_chain(r.name)
        })
        .collect();
    let mut factors = 0;
    for term in &report.terms {
        let Some(args) = call(m, term.node, "builtin_logdensityof") else {
            continue;
        };
        if let [family, _, point] = args.as_slice()
            && matches!(m.node(*family),Node::Const(name) if m.resolve(*name)=="Normal")
            && matches_point(m, *point, selected_ref.name, &path.fields, &all_boundary)
        {
            factors += term.multiplicity;
        }
    }
    if factors != 1 {
        return Err(AnalysisError(
            "selecting a Normal coordinate requires one unambiguous direct factor".into(),
        ));
    }
    let original = binding(m, "outputs").unwrap();
    let mut q = m.clone();
    // A query argument is a supplied value. Fix its phase in this companion
    // snapshot, preserving its exact set and binding identity. Otherwise a map
    // inserted into a callback can turn a formerly fixed capture into an
    // undeclared parametric input. Retained Auto callbacks refuse rather than
    // silently retracing their interface after this phase change.
    for &input in &inputs {
        let Node::Ref(r) = m.node(input) else {
            unreachable!()
        };
        let target = m.alias_target(r.name).unwrap();
        if let Some(args) = call(m, m.binding(target).rhs, "elementof") {
            let fixed = op(&mut q, "external", &args);
            q.set_binding_rhs(target, fixed);
        }
    }
    // Printed companions must not capture a fresh module reference inside an
    // existing lexical binder with the same spelling.
    let used = used_names(m);
    let fresh_coordinate = (0..)
        .map(|i| format!("__normal_coordinate_{i}"))
        .find(|name| !used.contains(&q.intern(name)))
        .unwrap();
    let name = q.intern(&fresh_coordinate);
    let reals = q.intern("reals");
    let reals = q.alloc(Node::Const(reals));
    let external = op(&mut q, "external", &[reals]);
    q.add_binding(Binding {
        name,
        rhs: external,
        doc: None,
        public: true,
        synthetic: true,
    });
    let fresh_z = q.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }));
    let (coordinate, z) = if path.fields.is_empty() {
        (fresh_coordinate, fresh_z)
    } else {
        (
            parameter.clone(),
            project(&mut q, inputs[selected], &path.fields),
        )
    };
    let parameters = normal_parameters(&mut q, candidate.location, candidate.scale, precision);
    let jacobian = parameters.jacobian;
    let (safe_theta, forward_valid) = forward_map(&mut q, &parameters, z, precision);
    let mut score = q.clone();
    let forward_value = replace_field(
        &mut score,
        inputs[selected],
        &input_type,
        &path.fields,
        safe_theta,
    );
    let selected_aliases = m.alias_chain(selected_ref.name).into_iter().collect();
    let boundary = inputs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != selected)
        .flat_map(|(_, id)| {
            let Node::Ref(r) = m.node(*id) else {
                unreachable!()
            };
            m.alias_chain(r.name)
        })
        .collect();
    let output = substitute(
        &mut score,
        original,
        &boundary,
        &selected_aliases,
        forward_value,
    )?;
    if path.fields.is_empty() {
        inputs[selected] = z;
    }
    let signature = if inputs.len() == 1 {
        inputs[0]
    } else {
        op(&mut score, "tuple", &inputs)
    };
    set(&mut score, "inputs", signature);
    let mut output = output;
    if logdensity {
        output = op(&mut score, "add", &[output, jacobian]);
    }
    let mut forward = score.clone();
    let mut log_jacobian = score.clone();
    forward = finish(forward, forward_value, forward_valid)?;
    log_jacobian = finish(log_jacobian, jacobian, forward_valid)?;
    score = finish(score, output, forward_valid)?;

    // The inverse keeps theta in its original argument slot.
    let point = binding(m, "inputs").unwrap();
    let original_inputs = call(m, point, "tuple").unwrap_or_else(|| vec![point]);
    let original_root = original_inputs[selected];
    let point = project(&mut q, original_root, &path.fields);
    let (inverse, valid) = inverse_map(&mut q, &parameters, point, precision);
    let inverse = replace_field(&mut q, original_root, &input_type, &path.fields, inverse);
    let inverse = finish(q, inverse, valid)?;
    Ok(NormalCoordinateQueries {
        score,
        forward,
        inverse,
        log_jacobian,
        coordinate_input: coordinate,
        precision,
    })
}

/// Compose a selected acyclic set of scalar Normal maps. The selected induced
/// dependency graph must be triangular; unselected inputs stay fixed. Validity
/// includes both complete floating-point round trips, not just local maps.
pub fn prepare_normal_coordinates(
    m: &Module,
    data_inputs: &[String],
    paths: &[InputPath],
    precision: CoordinatePrecision,
    logdensity: bool,
) -> Result<NormalCoordinatesQueries, AnalysisError> {
    let report = analyze_lowered_query(m, data_inputs)?;
    if paths.is_empty()
        || paths
            .iter()
            .enumerate()
            .any(|(i, p)| paths[..i].iter().any(|other| p.overlaps(other)))
    {
        return Err(AnalysisError(
            "select distinct non-overlapping Normal coordinate paths".into(),
        ));
    }
    let mut candidates = Vec::new();
    let input = binding(m, "inputs")
        .ok_or_else(|| AnalysisError("coordinate queries require inputs".into()))?;
    let inputs = call(m, input, "tuple").unwrap_or_else(|| vec![input]);
    let input_roots: HashMap<_, _> = inputs
        .iter()
        .map(|id| {
            let Node::Ref(r) = m.node(*id) else {
                unreachable!()
            };
            (m.resolve(r.name), r.name)
        })
        .collect();
    let all_boundary: HashSet<_> = inputs
        .iter()
        .flat_map(|id| {
            let Node::Ref(r) = m.node(*id) else {
                unreachable!()
            };
            m.alias_chain(r.name)
        })
        .collect();
    for path in paths {
        let candidate = report
            .normal_coordinates
            .iter()
            .find(|c| c.parameter == path.input && c.fields == path.fields)
            .ok_or_else(|| {
                AnalysisError("a selected path has no scalar Normal coordinate".into())
            })?;
        let selected = input_roots[path.input.as_str()];
        let factors: usize = report
            .terms
            .iter()
            .filter_map(|term| {
                let args = call(m, term.node, "builtin_logdensityof")?;
                match args.as_slice() {
                    [family, _, point]
                        if matches!(m.node(*family),Node::Const(name) if m.resolve(*name)=="Normal")
                            && matches_point(m, *point, selected, &path.fields, &all_boundary) =>
                    {
                        Some(term.multiplicity)
                    }
                    _ => None,
                }
            })
            .sum();
        if factors != 1 {
            return Err(AnalysisError(
                "each Normal coordinate requires one unambiguous direct factor".into(),
            ));
        }
        candidates.push(candidate);
    }
    let mut order = Vec::new();
    while order.len() < paths.len() {
        let next = (0..paths.len()).find(|i| {
            !order.contains(i)
                && !(0..paths.len()).any(|j| {
                    !order.contains(&j)
                        && candidates[*i]
                            .location_paths
                            .iter()
                            .chain(&candidates[*i].scale_paths)
                            .any(|p| p.overlaps(&paths[j]))
                })
        });
        let Some(next) = next else {
            return Err(AnalysisError(
                "selected Normal coordinates contain a dependency cycle".into(),
            ));
        };
        order.push(next);
    }

    let mut q = m.clone();
    let mut used = used_names(m);
    let mut raw_roots = HashMap::new();
    let mut raw_aliases = HashMap::new();
    let mut raw_environment = HashMap::new();
    // Keep raw arguments behind private names that no retained binder can
    // capture when this portable module is printed and reparsed.
    for &input in &inputs {
        let Node::Ref(r) = *m.node(input) else {
            unreachable!()
        };
        let target = m.alias_target(r.name).unwrap();
        if let Some(args) = call(m, m.binding(target).rhs, "elementof") {
            let fixed = op(&mut q, "external", &args);
            q.set_binding_rhs(target, fixed);
        }
        let alias = (0..)
            .map(|i| q.intern(&format!("__normal_input_{i}")))
            .find(|name| !used.contains(name))
            .unwrap();
        used.insert(alias);
        q.add_binding(Binding {
            name: alias,
            rhs: input,
            doc: None,
            public: false,
            synthetic: true,
        });
        let raw = q.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: alias,
        }));
        raw_roots.insert(r.name, raw);
        raw_aliases.insert(r.name, alias);
        for name in m.alias_chain(r.name) {
            raw_environment.insert(name, raw);
        }
    }
    let raw_boundary: HashSet<_> = raw_aliases
        .values()
        .copied()
        .chain(all_boundary.iter().copied())
        .collect();
    let transport = |q: &mut Module, expression, roots: &HashMap<Symbol, NodeId>| {
        let environment: HashMap<_, _> = roots
            .iter()
            .map(|(root, value)| (raw_aliases[root], *value))
            .collect();
        let boundary = raw_boundary
            .iter()
            .copied()
            .filter(|name| !environment.contains_key(name))
            .collect();
        substitute_inputs(q, expression, &boundary, &environment, true)
    };
    let raw_output = substitute_inputs(
        &mut q,
        binding(m, "outputs").unwrap(),
        &HashSet::new(),
        &raw_environment,
        false,
    )?;
    let mut forward_roots = HashMap::new();
    let mut inverse_roots = HashMap::new();
    let mut forward_flags = Vec::new();
    let mut inverse_flags = Vec::new();
    let mut inverse_parameters = Vec::new();
    let mut jacobian = number(&mut q, 0.);
    for &i in &order {
        let candidate = candidates[i];
        let root = input_roots[paths[i].input.as_str()];
        let raw_root = raw_roots[&root];
        let ty = m
            .type_of(
                *inputs
                    .iter()
                    .find(|id| matches!(m.node(**id),Node::Ref(r) if r.name==root))
                    .unwrap(),
            )
            .unwrap();
        let location = substitute_inputs(
            &mut q,
            candidate.location,
            &HashSet::new(),
            &raw_environment,
            false,
        )?;
        let scale = substitute_inputs(
            &mut q,
            candidate.scale,
            &HashSet::new(),
            &raw_environment,
            false,
        )?;
        let mu = transport(&mut q, location, &forward_roots)?;
        let sigma = transport(&mut q, scale, &forward_roots)?;
        let p = normal_parameters(&mut q, mu, sigma, precision);
        let coordinate = project(&mut q, raw_root, &paths[i].fields);
        let (theta, valid) = forward_map(&mut q, &p, coordinate, precision);
        forward_flags.push(valid);
        jacobian = op(&mut q, "add", &[jacobian, p.jacobian]);
        let previous = forward_roots.get(&root).copied().unwrap_or(raw_root);
        let value = replace_field(&mut q, previous, ty, &paths[i].fields, theta);
        forward_roots.insert(root, value);
        let p = normal_parameters(&mut q, location, scale, precision);
        let (z, valid) = inverse_map(&mut q, &p, coordinate, precision);
        inverse_flags.push(valid);
        let previous = inverse_roots.get(&root).copied().unwrap_or(raw_root);
        let value = replace_field(&mut q, previous, ty, &paths[i].fields, z);
        inverse_roots.insert(root, value);
        inverse_parameters.push((i, p, z));
    }
    let local_forward = both(&mut q, &forward_flags);
    let local_inverse = both(&mut q, &inverse_flags);
    forward_flags.push(transport(&mut q, local_inverse, &forward_roots)?);
    inverse_flags.push(transport(&mut q, local_forward, &inverse_roots)?);
    let mut round_z = HashMap::new();
    let mut round_theta = HashMap::new();
    for input in &inputs {
        let Node::Ref(r) = m.node(*input) else {
            unreachable!()
        };
        if let Some(&value) = inverse_roots.get(&r.name) {
            round_z.insert(r.name, transport(&mut q, value, &forward_roots)?);
            round_theta.insert(
                r.name,
                transport(&mut q, forward_roots[&r.name], &inverse_roots)?,
            );
        }
    }
    for (i, p, z) in &inverse_parameters {
        let root = input_roots[paths[*i].input.as_str()];
        let raw = project(&mut q, raw_roots[&root], &paths[*i].fields);
        let restored_z = project(&mut q, round_z[&root], &paths[*i].fields);
        let finite = op(&mut q, "isfinite", &[restored_z]);
        let roundtrip = close(&mut q, restored_z, raw, limits(precision).2);
        forward_flags.extend([finite, roundtrip]);
        let restored_theta = project(&mut q, round_theta[&root], &paths[*i].fields);
        let finite = op(&mut q, "isfinite", &[restored_theta]);
        let roundtrip = scaled_roundtrip(&mut q, restored_theta, raw, p, *z, precision);
        inverse_flags.extend([finite, roundtrip]);
    }
    let valid_forward = both(&mut q, &forward_flags);
    let valid_inverse = both(&mut q, &inverse_flags);
    // Map insertions beneath retained callbacks must stay opaque global refs.
    for input in &inputs {
        let Node::Ref(r) = m.node(*input) else {
            unreachable!()
        };
        if let Some(&value) = forward_roots.get(&r.name) {
            let value = private_value(&mut q, value, &mut used);
            forward_roots.insert(r.name, value);
        }
    }
    let mut output = transport(&mut q, raw_output, &forward_roots)?;
    if logdensity {
        output = op(&mut q, "add", &[output, jacobian]);
    }
    let mut coordinate_inputs = Vec::new();
    let mut forward = Vec::new();
    let mut inverse = Vec::new();
    for input in &inputs {
        let Node::Ref(r) = m.node(*input) else {
            unreachable!()
        };
        if let Some(value) = forward_roots.get(&r.name) {
            coordinate_inputs.push(m.resolve(r.name).to_owned());
            forward.push(*value);
            inverse.push(inverse_roots[&r.name]);
        }
    }
    let forward = if forward.len() == 1 {
        forward[0]
    } else {
        op(&mut q, "tuple", &forward)
    };
    let inverse = if inverse.len() == 1 {
        inverse[0]
    } else {
        op(&mut q, "tuple", &inverse)
    };
    Ok(NormalCoordinatesQueries {
        score: finish_joint(q.clone(), output, valid_forward)?,
        forward: finish_joint(q.clone(), forward, valid_forward)?,
        inverse: finish_joint(q.clone(), inverse, valid_inverse)?,
        log_jacobian: finish_joint(q, jacobian, valid_forward)?,
        coordinate_inputs,
        paths: order.into_iter().map(|i| paths[i].clone()).collect(),
        precision,
    })
}
