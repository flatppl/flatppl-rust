//! Preserve the recurrent prediction of a two-state probability filter.

use std::collections::{HashMap, HashSet};

use flatppl_core::graph::{build_call, build_record, expect_builtin_call};
use flatppl_core::{
    BindingId, Call, CallHead, Dim, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs,
    Scalar, Symbol,
};

use crate::{Emitter, REAL, Step, affine_scan, callbacks, real, recurrence, tensor};

/// Convert a copied-field projection into the preceding values of its
/// scalar carry. The only evolving field is the next predicted probability.
pub(super) fn projection(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    let [project, trajectory] = *positional(module, node, "broadcast")? else {
        return None;
    };
    let (view, formals) = recurrence::callback(module, inputs, project)?;
    let [view_state] = *formals.as_slice() else {
        return None;
    };
    let projected_field = field_name(module, view, view_state)?;
    let trajectory = callbacks::resolve_unpromoted(module, inputs, trajectory)?;
    let [function, initial, items, parameters] = *positional(module, trajectory, "scan")? else {
        return None;
    };
    let [Dim::Static(length)] = *tensor::real_shape(module, items)? else {
        return None;
    };
    if length == 0 {
        return None;
    }
    let (body, formals) = recurrence::callback(module, inputs, function)?;
    let [state, item, captured] = *formals.as_slice() else {
        return None;
    };
    if formals.iter().any(|reference| reference.ns != RefNs::Local) {
        return None;
    }
    let record = expect_builtin_call(module, body, "record")?;
    let [first, second] = *record.named.as_ref() else {
        return None;
    };
    if !record.args.is_empty() {
        return None;
    }
    let (projected, evolving) = if module.resolve(first.name) == projected_field {
        (first, second)
    } else if module.resolve(second.name) == projected_field {
        (second, first)
    } else {
        return None;
    };
    if field_name(module, projected.value, state)? != module.resolve(evolving.name) {
        return None;
    }
    let next = evolving.name;
    let reads_carry = |node| field_name(module, node, state) == Some(module.resolve(next));
    let (high, low) = transition(module, evolving.value, reads_carry, item)?;
    if !capture_projection(module, high, captured) || !capture_projection(module, low, captured) {
        return None;
    }
    let bindings = [(captured, parameters)];
    let high = recurrence::substitute(module, inputs, high, &bindings, &mut HashMap::new(), 0)?;
    let low = recurrence::substitute(module, inputs, low, &bindings, &mut HashMap::new(), 0)?;
    let initial = get_field(module, initial, next);
    let scan = scan(module, initial, items, high, low);
    let result = affine_scan::lag(module, initial, scan, length);
    let expected = module.type_of(node)?.clone();
    callbacks::annotate_equivalent(module, result, &expected)
}

/// Expose the normalized scan's scalar transition parameters to AD.
pub(super) fn args(
    module: &Module,
    inputs: &HashSet<BindingId>,
    args: &[NodeId],
) -> Option<[NodeId; 4]> {
    let [function, initial, items, params] = *args else {
        return None;
    };
    let (body, formals) = recurrence::callback(module, inputs, function)?;
    let [state, item, capture] = *formals.as_slice() else {
        return None;
    };
    let (high, low) = transition(module, body, |id| is_ref(module, id, state), item)?;
    let high = field_name(module, high, capture)?;
    let low = field_name(module, low, capture)?;
    if high == low
        || module.type_of(initial) != Some(&REAL)
        || !matches!(tensor::real_shape(module, items), Some([Dim::Static(n)]) if *n > 0)
    {
        return None;
    }
    let params = callbacks::resolve_unpromoted(module, inputs, params)?;
    let params = expect_builtin_call(module, params, "record")?;
    let get = |name: &str| {
        let value = params
            .named
            .iter()
            .find(|field| module.resolve(field.name) == name)?
            .value;
        (module.type_of(value) == Some(&REAL)).then_some(value)
    };
    Some([initial, items, get(high)?, get(low)?])
}

fn transition(
    module: &Module,
    body: NodeId,
    reads_state: impl Fn(NodeId) -> bool,
    item: Ref,
) -> Option<(NodeId, NodeId)> {
    let [a, b] = *positional(module, body, "add")? else {
        return None;
    };
    for (base, increment) in [(a, b), (b, a)] {
        let Some([left, right]) = positional(module, increment, "mul") else {
            continue;
        };
        for (&difference, &posterior) in [(left, right), (right, left)] {
            let Some([high, low]) = positional(module, difference, "sub") else {
                continue;
            };
            let Some([log_odds]) = positional(module, posterior, "invlogit") else {
                continue;
            };
            let Some([left, right]) = positional(module, *log_odds, "add") else {
                continue;
            };
            let adds_observation = [(left, right), (right, left)].into_iter().any(|(&prior, &r)| {
                is_ref(module, r, item)
                    && matches!(positional(module, prior, "logit"), Some([q]) if reads_state(*q))
            });
            if adds_observation && module.structural_eq(*low, base) {
                return Some((*high, *low));
            }
        }
    }
    None
}

fn capture_projection(module: &Module, node: NodeId, capture: Ref) -> bool {
    if is_ref(module, node, capture) {
        return true;
    }
    matches!(positional(module, node, "get"),
        Some([source, selector])
            if matches!(module.node(*selector), Node::Lit(Scalar::Int(_) | Scalar::Str(_)))
                && capture_projection(module, *source, capture))
}

fn scan(module: &mut Module, initial: NodeId, items: NodeId, high: NodeId, low: NodeId) -> NodeId {
    let high_name = module.intern("high");
    let low_name = module.intern("low");
    let parameters = build_record(module, &[(high_name, high), (low_name, low)]);
    let (state, sf) = formal(module, "state");
    let (item, itf) = formal(module, "item");
    let (capture, cf) = formal(module, "parameter");
    let high = get_field(module, capture, high_name);
    let low = get_field(module, capture, low_name);
    let prior = build_call(module, "logit", &[state]);
    let logit = build_call(module, "add", &[prior, item]);
    let posterior = build_call(module, "invlogit", &[logit]);
    let difference = build_call(module, "sub", &[high, low]);
    let increment = build_call(module, "mul", &[difference, posterior]);
    let body = build_call(module, "add", &[low, increment]);
    let callback = function(module, body, &[sf, itf, cf]);
    build_call(module, "scan", &[callback, initial, items, parameters])
}

/// Propagate time-varying scalar slopes without expanding the recurrence in IR.
pub(super) fn propagate(
    emitter: &mut Emitter,
    step: &Step<'_>,
    seed: NodeId,
    state_slope: NodeId,
    argument_slope: NodeId,
    slot: usize,
    forward: bool,
) -> Option<NodeId> {
    let [Dim::Static(length)] = *tensor::real_shape(&emitter.module, step.node)? else {
        return None;
    };
    let m = &mut emitter.module;
    let zero = real(m, 0.0);
    if forward {
        let (initial, offsets) = if slot == 0 {
            let length = tensor::int(m, i64::from(length));
            (seed, build_call(m, "zeros", &[length]))
        } else {
            (zero, tensor::broadcast(m, "mul", &[seed, argument_slope]))
        };
        return Some(affine_cells(m, initial, offsets, state_slope, false));
    }

    // At q_i, the next transition uses the slope from step i + 1.
    let mut next_slopes = vec![build_call(m, "vector", &[zero])];
    if length > 1 {
        let length = tensor::int(m, i64::from(length - 1));
        let shape = build_call(m, "zeros", &[length]);
        let indices = build_call(m, "indicesof", &[shape]);
        let one = tensor::int(m, 1);
        let indices = tensor::broadcast(m, "add", &[indices, one]);
        let tail = build_call(m, "get", &[state_slope, indices]);
        next_slopes.insert(0, tail);
    }
    let next_slopes = build_call(m, "cat", &next_slopes);
    let coefficients = build_call(m, "reverse", &[next_slopes]);
    let offsets = build_call(m, "reverse", &[seed]);
    let reverse = affine_cells(m, zero, offsets, coefficients, true);
    let adjoints = build_call(m, "reverse", &[reverse]);
    if slot == 0 {
        let one = tensor::int(m, 1);
        let adjoint = build_call(m, "get", &[adjoints, one]);
        let slope = build_call(m, "get", &[state_slope, one]);
        let contribution = build_call(m, "mul", &[adjoint, slope]);
        Some(tensor::zero_when_masked(m, adjoint, contribution, false))
    } else {
        let contribution = tensor::broadcast(m, "mul", &[adjoints, argument_slope]);
        let contribution = tensor::zero_when_masked(m, adjoints, contribution, true);
        if slot == 1 {
            Some(contribution)
        } else {
            Some(build_call(m, "sum", &[contribution]))
        }
    }
}

fn affine_cells(
    module: &mut Module,
    initial: NodeId,
    offsets: NodeId,
    slopes: NodeId,
    reverse: bool,
) -> NodeId {
    let offset = module.intern("offset");
    let slope = module.intern("slope");
    let table = module.intern("table");
    let sequence = module.alloc(Node::Call(Call {
        head: CallHead::Builtin(table),
        args: Box::new([]),
        named: Box::new([
            NamedArg {
                kind: NamedKind::Field,
                name: offset,
                value: offsets,
            },
            NamedArg {
                kind: NamedKind::Field,
                name: slope,
                value: slopes,
            },
        ]),
        inputs: None,
    }));
    let (state, sf) = formal(module, "state");
    let (cell, cf) = formal(module, "item");
    let coefficient = get_field(module, cell, slope);
    let offset = get_field(module, cell, offset);
    let product = build_call(module, "mul", &[state, coefficient]);
    let product = if reverse {
        tensor::zero_when_masked(module, state, product, false)
    } else {
        product
    };
    let body = build_call(module, "add", &[product, offset]);
    let callback = function(module, body, &[sf, cf]);
    build_call(module, "scan", &[callback, initial, sequence])
}

fn positional<'a>(module: &'a Module, node: NodeId, op: &str) -> Option<&'a [NodeId]> {
    let call = expect_builtin_call(module, node, op)?;
    call.named.is_empty().then_some(&call.args)
}

fn field_name(module: &Module, node: NodeId, formal: Ref) -> Option<&str> {
    let [source, selector] = *positional(module, node, "get")? else {
        return None;
    };
    let Node::Lit(Scalar::Str(name)) = module.node(selector) else {
        return None;
    };
    is_ref(module, source, formal).then_some(name)
}

fn is_ref(module: &Module, node: NodeId, formal: Ref) -> bool {
    matches!(module.node(node), Node::Ref(reference) if *reference == formal)
}

fn get_field(module: &mut Module, value: NodeId, field: Symbol) -> NodeId {
    let key = module.alloc(Node::Lit(Scalar::Str(module.resolve(field).into())));
    build_call(module, "get", &[value, key])
}

fn formal(module: &mut Module, name: &str) -> (NodeId, (Symbol, Ref)) {
    let reference = Ref {
        ns: RefNs::Local,
        name: module.intern(&format!("_{name}_")),
    };
    let label = module.intern(name);
    (module.alloc(Node::Ref(reference)), (label, reference))
}

fn function(module: &mut Module, body: NodeId, formals: &[(Symbol, Ref)]) -> NodeId {
    let head = module.intern("functionof");
    module.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(formals.into())),
    }))
}
