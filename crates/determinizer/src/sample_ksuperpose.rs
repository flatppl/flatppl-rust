//! Direct mixture sampling (spec §06 `ksuperpose`): draw a component index in
//! proportion to the broadcast-expanded weights, then draw that component.

use super::*;
use crate::density::broadcast::{flat_coordinates, flatten_terms, map_body, placeholder};
use crate::density::{
    broadcast_collection_shape, ksuperpose_expanded_weights, outer_rank, resolve_ref_chain,
};
use flatppl_core::{Call, Dim, Mass, NamedArg};

/// A `ksuperpose` mixture in a sampled position, with or without `normalize`.
pub(super) fn recognize(m: &Module, resolved: NodeId) -> Option<(NodeId, bool)> {
    if let Some(c) = expect_builtin_call(m, resolved, "normalize") {
        let [inner] = *c.args else {
            return None;
        };
        let inner = resolve_ref(m, inner).0;
        return expect_builtin_call(m, inner, "ksuperpose").map(|_| (inner, true));
    }
    expect_builtin_call(m, resolved, "ksuperpose").map(|_| (resolved, false))
}

struct Mixture {
    node: NodeId,
    component: NodeId,
    weights: NodeId,
    positional: Vec<NodeId>,
    named: Vec<(Symbol, NodeId)>,
    shape: Vec<u32>,
}

/// Sample `mixture` once, or `iid` over `lanes` with one component per lane.
pub(super) fn lower(
    m: &mut Module,
    mixture: NodeId,
    normalized: bool,
    lanes: Option<&[usize]>,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> Result<(NodeId, NodeId), RefuseError> {
    let mix = bind(m, mixture, normalized)?;
    let (index, valid, rng) = draw_index(m, &mix, lanes, rng)?;
    let (value, next) = match lanes {
        None => {
            let component = component_at(m, &mix, index);
            let draw = build_call(m, "draw", &[component]);
            lower_draw(m, draw, rng, sample_name)?
        }
        Some(lanes) => draw_lanes(m, &mix, index, lanes.len(), rng, sample_name)?,
    };
    let domain = match measure_type(m, mixture) {
        Some(Type::Measure { domain, .. }) => Some(*domain),
        _ => None,
    };
    let integer = matches!(
        domain,
        Some(Type::Scalar(ScalarType::Integer | ScalarType::Boolean))
    );
    if lanes.is_none() && matches!(domain, Some(Type::Scalar(_))) {
        let value = crate::density::exponential_tilt::guard_sample(m, value, valid, integer);
        return Ok((value, next));
    }
    // An invalid mass makes every cell undefined, so the guard keeps the shape.
    let sentinel = if integer {
        m.alloc(Node::Lit(Scalar::Int(-1)))
    } else {
        m.alloc(Node::Lit(Scalar::Real(f64::NAN)))
    };
    let value = crate::density::tilt::elementwise(m, "ifelse", &[valid, value, sentinel]);
    Ok((value, next))
}

fn measure_type(m: &mut Module, node: NodeId) -> Option<Type> {
    if let Some(ty) = m.type_of(node) {
        return Some(ty.clone());
    }
    let ty = flatppl_infer::infer_expression(m, node, &[])
        .0
        .remove(&node)?;
    m.set_type(node, ty.clone());
    Some(ty)
}

fn bind(m: &mut Module, node: NodeId, normalized: bool) -> Result<Mixture, RefuseError> {
    let call = expect_builtin_call(m, node, "ksuperpose").unwrap().clone();
    let [component, weights, ref positional @ ..] = *call.args else {
        return Err(refuse(node, m, "ksuperpose needs a kernel and weights"));
    };
    let named: Vec<_> = call
        .named
        .iter()
        .filter(|a| a.kind == NamedKind::Kwarg)
        .map(|a| (a.name, a.value))
        .collect();
    let mass = match measure_type(m, node) {
        Some(Type::Measure { mass, .. }) => mass,
        _ => Mass::Unknown,
    };
    if !normalized && mass != Mass::Normalized {
        // §04 draws only from probability measures and §06 makes
        // normalization explicit, so an unproven unit mass is not a law.
        return Err(refuse(
            node,
            m,
            "sampling a ksuperpose mixture needs a probability measure, but its total mass \
             is the sum of the broadcast-expanded weights (§06); wrap it in normalize(...)",
        ));
    }
    let operands: Vec<_> = std::iter::once(weights)
        .chain(positional.iter().copied())
        .chain(named.iter().map(|(_, v)| *v))
        .collect();
    for &operand in &operands {
        let resolved = resolve_ref_chain(m, operand).0;
        if m.type_of(operand).is_none()
            && let Some(ty) = m.type_of(resolved).cloned()
        {
            m.set_type(operand, ty);
        }
    }
    let shape = broadcast_collection_shape(m, &operands)
        .and_then(|shape| {
            shape
                .into_iter()
                .map(|d| match d {
                    Dim::Static(n) => Some(n),
                    Dim::Dynamic => None,
                })
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| {
            refuse(
                node,
                m,
                "ksuperpose sampling needs a statically known broadcast shape",
            )
        })?;
    if shape.contains(&0) || literal_zero_weights(m, weights) {
        return Err(refuse(
            node,
            m,
            "this ksuperpose mixture is the zero measure (empty broadcast or all-zero weights); \
             normalization and sampling need positive total mass (§06 normalize)",
        ));
    }
    let (positional, named) = if positional.is_empty() || named.is_empty() {
        (positional.to_vec(), named)
    } else {
        // Only the special operation mixes the two forms; the component call
        // binds positional inputs by name.
        let names = input_names(m, component)
            .filter(|names| names.len() >= positional.len())
            .ok_or_else(|| {
                refuse(
                    node,
                    m,
                    "ksuperpose has more positional inputs than the component kernel",
                )
            })?;
        let bound = names.into_iter().zip(positional.iter().copied());
        (Vec::new(), bound.chain(named).collect())
    };
    let mix = Mixture {
        node,
        component,
        weights,
        positional,
        named,
        shape,
    };
    let probe = m.alloc(Node::Lit(Scalar::Int(0)));
    let first = component_at(m, &mix, probe);
    let proper = matches!(
        measure_type(m, first),
        Some(Type::Measure {
            mass: Mass::Normalized,
            ..
        })
    ) || matches!(builtin_name(m, first), Some("Dirac"));
    if !proper {
        return Err(refuse(
            node,
            m,
            "ksuperpose sampling needs a kernel that yields probability measures",
        ));
    }
    Ok(mix)
}

fn input_names(m: &mut Module, component: NodeId) -> Option<Vec<Symbol>> {
    if let Some(kernel) = crate::kernel::resolve_reified(m, component) {
        return Some(kernel.inputs.iter().map(|(name, _)| *name).collect());
    }
    let Node::Const(ctor) = *m.node(resolve_ref(m, component).0) else {
        return None;
    };
    let names = flatppl_infer::constructor_param_names(m.resolve(ctor))?;
    Some(names.iter().map(|name| m.intern(name)).collect())
}

fn literal_zero_weights(m: &Module, weights: NodeId) -> bool {
    let zero = |node: NodeId| {
        matches!(m.node(resolve_ref_chain(m, node).0), Node::Lit(Scalar::Real(x)) if *x == 0.0)
            || matches!(
                m.node(resolve_ref_chain(m, node).0),
                Node::Lit(Scalar::Int(0))
            )
    };
    let resolved = resolve_ref_chain(m, weights).0;
    match expect_builtin_call(m, resolved, "vector") {
        Some(c) => c.args.iter().all(|&x| zero(x)),
        None => zero(resolved),
    }
}

/// Draw the zero-based flat component index. The guard rejects a runtime
/// weight sum that is zero or not finite, and any negative weight.
fn draw_index(
    m: &mut Module,
    mix: &Mixture,
    lanes: Option<&[usize]>,
    rng: NodeId,
) -> Result<(NodeId, NodeId, NodeId), RefuseError> {
    let operands: Vec<_> = std::iter::once(mix.weights)
        .chain(mix.positional.iter().copied())
        .chain(mix.named.iter().map(|(_, v)| *v))
        .collect();
    let shape: Vec<_> = mix.shape.iter().copied().map(Dim::Static).collect();
    let expanded = ksuperpose_expanded_weights(m, &operands, &shape).ok_or_else(|| {
        refuse(
            mix.node,
            m,
            "ksuperpose weights do not broadcast with its parameters",
        )
    })?;
    let weights = match mix.shape.len() {
        0 => build_call(m, "vector", &[expanded]),
        1 => expanded,
        _ => {
            m.set_type(
                expanded,
                Type::Array {
                    shape: shape.into(),
                    elem: Box::new(Type::Scalar(ScalarType::Real)),
                },
            );
            flatten_terms(m, expanded)?.expect("non-empty broadcast")
        }
    };
    let total = build_call(m, "sum", &[weights]);
    let probabilities = crate::density::tilt::elementwise(m, "divide", &[weights, total]);
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let finite = build_call(m, "isfinite", &[total]);
    let positive = build_call(m, "gt", &[total, zero]);
    let smallest = build_call(m, "minimum", &[weights]);
    let nonnegative = build_call(m, "ge", &[smallest, zero]);
    let valid = build_call(m, "land", &[finite, positive]);
    let valid = build_call(m, "land", &[valid, nonnegative]);

    let p = m.intern("p");
    let input = build_record(m, &[(p, probabilities)]);
    let categorical = m.intern("Categorical0");
    let categorical = m.alloc(Node::Const(categorical));
    // Every parameter gather reads the index, so bind one draw by name.
    let mut serial = 0;
    let name = loop {
        let name = m.intern(&format!("__component_{serial}"));
        if m.binding_by_name(name).is_none() {
            break name;
        }
        serial += 1;
    };
    let (index, rng) = match lanes {
        None => build_sample_term(m, categorical, input, rng, Some(name)),
        Some(lanes) => build_iid_sample_term(m, categorical, input, lanes, rng, Some(name)),
    };
    Ok((index, valid, rng))
}

/// The component measure call at a zero-based flat `index`. A unit axis of an
/// operand stays at its single cell; that is broadcast expansion.
fn component_at(m: &mut Module, mix: &Mixture, index: NodeId) -> NodeId {
    let coordinates = flat_coordinates(m, index, &mix.shape);
    let positional: Vec<_> = mix
        .positional
        .clone()
        .into_iter()
        .map(|operand| cell_at(m, operand, &coordinates))
        .collect();
    let named: Vec<_> = mix
        .named
        .clone()
        .into_iter()
        .map(|(name, operand)| NamedArg {
            kind: NamedKind::Kwarg,
            name,
            value: cell_at(m, operand, &coordinates),
        })
        .collect();
    let (resolved, _) = resolve_ref(m, mix.component);
    let head = match *m.node(resolved) {
        Node::Const(sym) => CallHead::Builtin(sym),
        _ => CallHead::User(mix.component),
    };
    m.alloc(Node::Call(Call {
        head,
        args: positional.into(),
        named: named.into(),
        inputs: None,
    }))
}

fn cell_at(m: &mut Module, operand: NodeId, coordinates: &[NodeId]) -> NodeId {
    let resolved = resolve_ref_chain(m, operand).0;
    let ty = m.type_of(operand).or_else(|| m.type_of(resolved)).cloned();
    match ty {
        Some(Type::Array { shape, .. }) if outer_rank(ty.as_ref().unwrap()) > 0 => {
            let mut selectors = vec![operand];
            selectors.extend(select(m, &shape, coordinates));
            build_call(m, "get", &selectors)
        }
        Some(Type::Table { columns, nrows }) => {
            // Tables supply rows; each column is gathered at the row.
            let visible = expect_builtin_call(m, resolved, "table").map(|c| c.named.to_vec());
            let row = select(m, &[nrows], coordinates);
            let fields: Vec<_> = columns
                .iter()
                .map(|(name, _)| {
                    let column = match visible
                        .as_ref()
                        .and_then(|xs| xs.iter().find(|x| x.name == *name))
                    {
                        Some(field) => field.value,
                        None => {
                            let key = m.alloc(Node::Lit(Scalar::Str(m.resolve(*name).into())));
                            build_call(m, "get", &[operand, key])
                        }
                    };
                    (*name, build_call(m, "get", &[column, row[0]]))
                })
                .collect();
            build_record(m, &fields)
        }
        _ => operand,
    }
}

fn select(m: &mut Module, shape: &[Dim], coordinates: &[NodeId]) -> Vec<NodeId> {
    shape
        .iter()
        .zip(coordinates)
        .map(|(dim, &coordinate)| match dim {
            Dim::Static(1) => m.alloc(Node::Lit(Scalar::Int(1))),
            _ => coordinate,
        })
        .collect()
}

/// One component per iid lane: gather each scalar parameter at the lane's
/// index, then sample the component family as one broadcasted product.
fn draw_lanes(
    m: &mut Module,
    mix: &Mixture,
    index: NodeId,
    rank: usize,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> Result<(NodeId, NodeId), RefuseError> {
    let (label, reference, lane) = placeholder(m, Type::Scalar(ScalarType::Integer));
    let mut component = component_at(m, mix, lane);
    while let Some(body) = crate::kernel::reduce_kernel_application(m, component) {
        component = resolve_ref_chain(m, body).0;
    }
    let (ctor, parameters) = split_kernel_constructor(m, component).ok_or_else(|| {
        refuse(
            mix.node,
            m,
            "iid ksuperpose sampling needs a component that is a built-in scalar constructor",
        )
    })?;
    if !matches!(measure_type(m, component), Some(Type::Measure { domain, .. })
        if matches!(*domain, Type::Scalar(_)))
    {
        return Err(refuse(
            mix.node,
            m,
            "iid ksuperpose sampling supports scalar components only",
        ));
    }
    let parameters: Vec<_> = parameters
        .into_iter()
        .map(|(name, value)| {
            let mapped = map_body(m, value, vec![(label, reference)], vec![index], rank);
            (name, mapped)
        })
        .collect();
    if m.resolve(ctor) == "Dirac" {
        // A point mass consumes no randomness.
        let [(_, value)] = *parameters else {
            return Err(refuse(mix.node, m, "Dirac takes one value"));
        };
        return Ok((value, rng));
    }
    let input = build_record(m, &parameters);
    let ctor = m.alloc(Node::Const(ctor));
    let kernel = build_call(m, "broadcasted", &[ctor]);
    Ok(build_sample_term(m, kernel, input, rng, sample_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatppl_core::Idx;

    #[test]
    fn an_empty_broadcast_refuses_as_the_zero_measure() {
        let mut m = flatppl_syntax::parse(
            "w = external(cartpow(nonnegreals, 7))\nmeans = external(cartpow(reals, 7))\n\
             m = ksuperpose(Normal, w, means, sigma = 1.)",
        )
        .unwrap();
        let _ = flatppl_infer::infer(&mut m);
        // A derived zero extent infers dynamically; give the frame a static zero.
        for i in 0..m.node_count() {
            let id = NodeId::from_usize(i);
            if let Some(Type::Array { mut shape, elem }) = m.type_of(id).cloned() {
                for dim in &mut shape {
                    if *dim == Dim::Static(7) {
                        *dim = Dim::Static(0);
                    }
                }
                m.set_type(id, Type::Array { shape, elem });
            }
        }
        let name = m.intern("m");
        let mixture = m.binding(m.binding_by_name(name).unwrap()).rhs;
        let rng = m.alloc(Node::Lit(Scalar::Int(0)));
        let err = lower(&mut m, mixture, true, None, rng, None).unwrap_err();
        assert!(err.reason.contains("zero measure"), "{}", err.reason);
    }
}
