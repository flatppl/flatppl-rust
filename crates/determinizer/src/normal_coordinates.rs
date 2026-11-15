//! Explicit scalar Normal companions over the shared lowered query.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    Binding, CallHead, Idx, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol,
};

use crate::{AnalysisError, InputPath, analyze_model};
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

fn call(m: &Module, id: NodeId, name: &str) -> Option<Vec<NodeId>> {
    match m.node(id) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if m.resolve(h) == name) => {
            Some(c.args.to_vec())
        }
        _ => None,
    }
}

fn op(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    crate::density::build_call(m, name, args)
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
    crate::density::build_record(m, &result)
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
    let roots = [m.intern("inputs"), m.intern("outputs")];
    crate::canon::retain_reachable(&mut m, &roots);
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
                if selected.contains(&r.name) {
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
            crate::driver::rebuild_with_children(m, id, &changed)
        };
        memo.insert(key, result);
        active.remove(&key);
    }
    Ok(memo[&(root, 0)])
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
    let report = analyze_model(m, data_inputs)?;
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
    let (minimum, maximum, tolerance) = match precision {
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
    };
    let zero = number(&mut q, 0.0);
    let one = number(&mut q, 1.0);
    let mu_ok = op(&mut q, "isfinite", &[candidate.location]);
    let sigma_ok = op(&mut q, "isfinite", &[candidate.scale]);
    let minimum = number(&mut q, minimum);
    let maximum = number(&mut q, maximum);
    let lo_ok = op(&mut q, "ge", &[candidate.scale, minimum]);
    let hi_ok = op(&mut q, "le", &[candidate.scale, maximum]);
    let sigma_ok = both(&mut q, &[sigma_ok, lo_ok, hi_ok]);
    let mu = op(&mut q, "ifelse", &[mu_ok, candidate.location, zero]);
    let sigma = op(&mut q, "ifelse", &[sigma_ok, candidate.scale, one]);
    let jacobian = op(&mut q, "log", &[sigma]);

    let z_ok = op(&mut q, "isfinite", &[z]);
    let safe_z = op(&mut q, "ifelse", &[z_ok, z, zero]);
    let scaled = op(&mut q, "mul", &[sigma, safe_z]);
    let theta = op(&mut q, "add", &[mu, scaled]);
    let theta_ok = op(&mut q, "isfinite", &[theta]);
    let safe_theta = op(&mut q, "ifelse", &[theta_ok, theta, zero]);
    let difference = op(&mut q, "sub", &[safe_theta, mu]);
    let recovered = op(&mut q, "divide", &[difference, sigma]);
    let recovered_ok = op(&mut q, "isfinite", &[recovered]);
    let roundtrip = close(&mut q, recovered, safe_z, tolerance);
    let forward_valid = both(
        &mut q,
        &[mu_ok, sigma_ok, z_ok, theta_ok, recovered_ok, roundtrip],
    );
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
    let point_ok = op(&mut q, "isfinite", &[point]);
    let point = op(&mut q, "ifelse", &[point_ok, point, zero]);
    let delta = op(&mut q, "sub", &[point, mu]);
    let inverse = op(&mut q, "divide", &[delta, sigma]);
    let inverse_ok = op(&mut q, "isfinite", &[inverse]);
    let scaled = op(&mut q, "mul", &[sigma, inverse]);
    let restored = op(&mut q, "add", &[mu, scaled]);
    let restored_ok = op(&mut q, "isfinite", &[restored]);
    let error = op(&mut q, "sub", &[restored, point]);
    let error = op(&mut q, "divide", &[error, sigma]);
    let relative = op(&mut q, "abs", &[inverse]);
    let relative = op(&mut q, "max", &[one, relative]);
    let tolerance = number(&mut q, tolerance);
    let relative = op(&mut q, "mul", &[relative, tolerance]);
    let error = op(&mut q, "abs", &[error]);
    let roundtrip = op(&mut q, "le", &[error, relative]);
    let valid = both(
        &mut q,
        &[
            mu_ok,
            sigma_ok,
            point_ok,
            inverse_ok,
            restored_ok,
            roundtrip,
        ],
    );
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
