//! AD of a terminal log-space Markov scan with fixed real vectors.
//!
//! The matched finite-domain recurrence has strictly positive transition
//! normalizers. Input vectors and each transition/emission row have the same
//! width. A tied stabilizing maximum splits its cotangent equally.

use std::collections::HashSet;

use flatppl_core::graph::{build_call, build_record, expect_builtin_call, resolve_ref};
use flatppl_core::{BindingId, Dim, Module, Node, NodeId, Ref, Scalar};

use crate::affine_scan::{function_of, lag, placeholder};
use crate::contract::{aggregate, axis, indexed};
use crate::{Emitter, Step, callbacks, real, recurrence, tensor};

pub(super) const OP: &str = "terminal log Markov scan";

pub(super) fn operands(
    module: &Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<[NodeId; 3]> {
    let [scan, terminal] = *positional(module, node, "get")? else {
        return None;
    };
    let scan = callbacks::resolve_unpromoted(module, inputs, scan)?;
    let [step, initial, sequence, params] = *positional(module, scan, "scan")? else {
        return None;
    };
    let [Dim::Static(width)] = *tensor::real_shape(module, initial)? else {
        return None;
    };
    let [Dim::Static(steps)] = *tensor::numeric_shape(module, sequence)? else {
        return None;
    };
    if tensor::nested_vector(module, scan) != Some((steps, width))
        || terminal_index(module, inputs, terminal) != Some(i64::from(steps))
        || tensor::real_shape(module, node) != Some(&[Dim::Static(width)])
    {
        return None;
    }
    let sequence = callbacks::resolve_unpromoted(module, inputs, sequence)?;
    if !contiguous_rows(module, inputs, sequence, steps) {
        return None;
    }

    let (body, placeholders) = recurrence::callback(module, inputs, step)?;
    let [state, row, parameter] = *placeholders.as_slice() else {
        return None;
    };
    let [predictive, emission] = *pointwise(module, body, "add")? else {
        return None;
    };
    let [max, log] = *pointwise(module, predictive, "add")? else {
        return None;
    };
    let [maximum_arg] = *positional(module, max, "maximum")? else {
        return None;
    };
    let [mixed] = *pointwise(module, log, "log")? else {
        return None;
    };
    let [matrix, exp] = *positional(module, mixed, "mul")? else {
        return None;
    };
    let [shifted] = *pointwise(module, exp, "exp")? else {
        return None;
    };
    let [previous, center] = *pointwise(module, shifted, "sub")? else {
        return None;
    };
    if !reference(module, maximum_arg, state)
        || !reference(module, previous, state)
        || !module.structural_eq(max, center)
    {
        return None;
    }
    let [emissions, index, full] = *positional(module, emission, "get")? else {
        return None;
    };
    if !reference(module, index, row)
        || !matches!(module.node(full), Node::Const(symbol) if module.resolve(*symbol) == "all")
    {
        return None;
    }

    let params = callbacks::resolve_unpromoted(module, inputs, params)?;
    let record = expect_builtin_call(module, params, "record")?;
    let field = |node| {
        let [target, key] = *positional(module, node, "get")? else {
            return None;
        };
        if !reference(module, target, parameter) {
            return None;
        }
        let Node::Lit(Scalar::Str(name)) = module.node(key) else {
            return None;
        };
        record
            .named
            .iter()
            .find(|field| module.resolve(field.name) == &**name)
            .map(|field| field.value)
    };
    let matrix = field(matrix)?;
    let emissions = field(emissions)?;
    let k = Dim::Static(width);
    let n = Dim::Static(steps.checked_add(1)?);
    if tensor::real_shape(module, matrix) != Some(&[k, k])
        || tensor::real_shape(module, emissions) != Some(&[n, k])
    {
        return None;
    }
    Some([initial, matrix, emissions])
}

fn positional<'a>(module: &'a Module, node: NodeId, op: &str) -> Option<&'a [NodeId]> {
    let call = expect_builtin_call(module, node, op)?;
    call.named.is_empty().then_some(&call.args)
}

fn pointwise<'a>(module: &'a Module, node: NodeId, op: &str) -> Option<&'a [NodeId]> {
    positional(module, node, op).or_else(|| {
        let (&head, args) = positional(module, node, "broadcast")?.split_first()?;
        matches!(module.node(head), Node::Const(name) if module.resolve(*name) == op)
            .then_some(args)
    })
}

fn reference(module: &Module, node: NodeId, formal: Ref) -> bool {
    matches!(module.node(node), Node::Ref(reference) if *reference == formal)
}

fn integer(module: &Module, node: NodeId, value: i64) -> bool {
    matches!(module.node(node), Node::Lit(Scalar::Int(n)) if *n == value)
}

fn terminal_index(module: &Module, inputs: &HashSet<BindingId>, node: NodeId) -> Option<i64> {
    let literal = |node| {
        let node = callbacks::resolve_unpromoted(module, inputs, node)?;
        match module.node(node) {
            Node::Lit(Scalar::Int(value)) => Some(*value),
            _ => None,
        }
    };
    let source = callbacks::resolve_unpromoted(module, inputs, node)?;
    if let Some([left, right]) = positional(module, source, "sub") {
        return literal(*left)?.checked_sub(literal(*right)?);
    }
    literal(source)
}

fn contiguous_rows(
    module: &Module,
    inputs: &HashSet<BindingId>,
    sequence: NodeId,
    steps: u32,
) -> bool {
    if let Some(cells) = positional(module, sequence, "vector") {
        return cells.len() == steps as usize
            && cells
                .iter()
                .zip(2..)
                .all(|(&cell, row)| integer(module, cell, row));
    }
    let Some([indices, offset]) = pointwise(module, sequence, "add") else {
        return false;
    };
    if !integer(module, *offset, 1) {
        return false;
    }
    let Some(indices) = callbacks::resolve_unpromoted(module, inputs, *indices) else {
        return false;
    };
    let Some([source]) = positional(module, indices, "indicesof") else {
        return false;
    };
    tensor::numeric_shape(module, *source) == Some(&[Dim::Static(steps)])
}

#[derive(Clone, Copy)]
pub(super) struct Prepared {
    steps: u32,
    position: NodeId,
    matrix: NodeId,
    weights: NodeId,
    normalizers: NodeId,
    maxima: NodeId,
}

fn prepare(emitter: &mut Emitter, step: &Step<'_>) -> Option<Prepared> {
    if emitter.inline {
        return None;
    }
    if let Some(&prepared) = emitter.vector_scans.get(&step.node) {
        return Some(prepared);
    }
    let [initial, matrix, emissions] = *step.args.as_slice() else {
        return None;
    };
    let [Dim::Static(rows), _] = *tensor::real_shape(&emitter.module, emissions)? else {
        return None;
    };
    let steps = rows.checked_sub(1)?;
    let &[container, _] = positional(&emitter.module, step.node, "get")? else {
        return None;
    };
    let (scan, _) = resolve_ref(&emitter.module, container);
    let initial = emitter.primal(initial);
    let matrix = emitter.primal(matrix);
    let trajectory = emitter.primal(scan);
    let m = &mut emitter.module;
    let prior = lag(m, initial, trajectory, steps);
    let prior = build_call(m, "rowstack", &[prior]);
    let prior = emitter.define("_ad_markov_prior", prior);
    let m = &mut emitter.module;
    let time = axis(m, "admarkovtime");
    let state = axis(m, "admarkovstate");
    let cell = indexed(m, prior, &[time, state]);
    let maximum = constant(m, "maximum");
    let center = aggregate(m, maximum, &[time], cell);
    let center = emitter.define("_ad_markov_center", center);
    let m = &mut emitter.module;
    let center = indexed(m, center, &[time]);
    let shifted = build_call(m, "sub", &[cell, center]);
    let cell_weight = build_call(m, "exp", &[shifted]);
    let sum = constant(m, "sum");
    let weights = aggregate(m, sum, &[time, state], cell_weight);
    let weights = emitter.define("_ad_markov_weights", weights);
    let m = &mut emitter.module;
    let pivot = build_call(m, "iszero", &[shifted]);
    let pivot = build_call(m, "real", &[pivot]);
    let maxima = aggregate(m, sum, &[time, state], pivot);
    let maxima = emitter.define("_ad_markov_maxima", maxima);

    let m = &mut emitter.module;
    let transposed = build_call(m, "transpose", &[matrix]);
    let normalizers = build_call(m, "mul", &[weights, transposed]);
    let normalizers = emitter.define("_ad_markov_norm", normalizers);
    let m = &mut emitter.module;
    let length = tensor::int(m, i64::from(steps));
    let base = build_call(m, "ones", &[length]);
    let position = build_call(m, "indicesof", &[base]);
    let prepared = Prepared {
        steps,
        position,
        matrix,
        weights,
        normalizers,
        maxima,
    };
    emitter.vector_scans.insert(step.node, prepared);
    Some(prepared)
}

pub(super) fn pullback(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
    masked: bool,
) -> Option<NodeId> {
    let prepared = prepare(emitter, step)?;
    let backward = if let Some(&named) = emitter.vector_scan_bars.get(&step.node) {
        named
    } else {
        let backward = build_reverse(emitter, prepared, bar);
        emitter.vector_scan_bars.insert(step.node, backward);
        backward
    };
    let m = &mut emitter.module;
    let result = if slot == 0 {
        let last = tensor::int(m, i64::from(prepared.steps));
        build_call(m, "get", &[backward, last])
    } else {
        let before = lag(m, bar, backward, prepared.steps);
        let chronological = build_call(m, "reverse", &[before]);
        match slot {
            1 => {
                let bars = build_call(m, "rowstack", &[chronological]);
                let scaled = divide_cotangent(m, bars, prepared.normalizers);
                let left = build_call(m, "transpose", &[scaled]);
                build_call(m, "mul", &[left, prepared.weights])
            }
            2 => {
                let zero = real(&mut emitter.module, 0.0);
                let zero = tensor::fill_like(emitter, zero, step.args[0])?;
                let m = &mut emitter.module;
                let initial = build_call(m, "vector", &[zero]);
                let rows = build_call(m, "cat", &[initial, chronological]);
                build_call(m, "rowstack", &[rows])
            }
            _ => return None,
        }
    };
    if masked {
        let m = &mut emitter.module;
        let absolute = tensor::broadcast(m, "abs", &[bar]);
        let magnitude = build_call(m, "maximum", &[absolute]);
        Some(tensor::zero_when_masked(m, magnitude, result, true))
    } else {
        Some(result)
    }
}

fn build_reverse(emitter: &mut Emitter, prepared: Prepared, bar: NodeId) -> NodeId {
    let m = &mut emitter.module;
    let formals = ["cotangent", "position", "params"].map(|name| placeholder(m, name));
    let [carry, position, params] = formals.map(|(_, r)| m.alloc(Node::Ref(r)));
    let matrix = m.intern("matrix");
    let weights = m.intern("weights");
    let normalizers = m.intern("normalizers");
    let maxima = m.intern("maxima");
    let transposed = build_call(m, "transpose", &[prepared.matrix]);
    let parameters = build_record(
        m,
        &[
            (matrix, transposed),
            (weights, prepared.weights),
            (normalizers, prepared.normalizers),
            (maxima, prepared.maxima),
        ],
    );
    let a = field(m, params, matrix);
    let w = field(m, params, weights);
    let s = field(m, params, normalizers);
    let pivot = field(m, params, maxima);
    let w = row(m, w, position);
    let s = row(m, s, position);
    let pivot = row(m, pivot, position);
    let scaled = divide_cotangent(m, carry, s);
    let propagated = build_call(m, "mul", &[a, scaled]);
    let previous = tensor::broadcast(m, "mul", &[w, propagated]);
    // The maximum terms cancel over reals. Retain their residual in the
    // working precision so a long reverse scan does not accumulate drift.
    let incoming = build_call(m, "sum", &[carry]);
    let outgoing = build_call(m, "sum", &[previous]);
    let remainder = build_call(m, "sub", &[incoming, outgoing]);
    let count = build_call(m, "sum", &[pivot]);
    let remainder = build_call(m, "divide", &[remainder, count]);
    let correction = tensor::broadcast(m, "mul", &[pivot, remainder]);
    let previous = build_call(m, "add", &[previous, correction]);
    let callback = function_of(m, previous, &formals);
    let sequence = build_call(m, "reverse", &[prepared.position]);
    let result = build_call(m, "scan", &[callback, bar, sequence, parameters]);
    emitter.define("_ad_markov_reverse", result)
}

pub(super) fn pushforward(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
) -> Option<NodeId> {
    let prepared = prepare(emitter, step)?;
    let initial = if slot == 0 {
        bar
    } else {
        let zero = real(&mut emitter.module, 0.0);
        tensor::fill_like(emitter, zero, step.args[0])?
    };
    let m = &mut emitter.module;
    let formals = ["tangent", "position", "params"].map(|name| placeholder(m, name));
    let [carry, position, params] = formals.map(|(_, r)| m.alloc(Node::Ref(r)));
    let matrix = m.intern("matrix");
    let normalizers = m.intern("normalizers");
    let weights = m.intern("weights");
    let perturbation = m.intern("perturbation");
    let maxima = m.intern("maxima");
    let mut fields = vec![
        (matrix, prepared.matrix),
        (weights, prepared.weights),
        (normalizers, prepared.normalizers),
        (maxima, prepared.maxima),
    ];
    if slot != 0 {
        fields.push((perturbation, bar));
    }
    let a = field(m, params, matrix);
    let w = field(m, params, weights);
    let s = field(m, params, normalizers);
    let pivot = field(m, params, maxima);
    let w = row(m, w, position);
    let s = row(m, s, position);
    let pivot = row(m, pivot, position);
    let numerator = tensor::broadcast(m, "mul", &[pivot, carry]);
    let numerator = build_call(m, "sum", &[numerator]);
    let count = build_call(m, "sum", &[pivot]);
    let center = build_call(m, "divide", &[numerator, count]);
    let difference = tensor::broadcast(m, "sub", &[carry, center]);
    let weighted = tensor::broadcast(m, "mul", &[w, difference]);
    let mut propagated = build_call(m, "mul", &[a, weighted]);
    if slot == 1 {
        let dot_a = field(m, params, perturbation);
        let changed = build_call(m, "mul", &[dot_a, w]);
        propagated = build_call(m, "add", &[propagated, changed]);
    }
    let tangent = tensor::broadcast(m, "divide", &[propagated, s]);
    let mut tangent = tensor::broadcast(m, "add", &[center, tangent]);
    if slot == 2 {
        let emissions = field(m, params, perturbation);
        let one = tensor::int(m, 1);
        let index = build_call(m, "add", &[position, one]);
        let change = row(m, emissions, index);
        tangent = build_call(m, "add", &[tangent, change]);
    }
    let callback = function_of(m, tangent, &formals);
    let parameters = build_record(m, &fields);
    let path = build_call(
        m,
        "scan",
        &[callback, initial, prepared.position, parameters],
    );
    let last = tensor::int(m, i64::from(prepared.steps));
    Some(build_call(m, "get", &[path, last]))
}

fn constant(module: &mut Module, name: &str) -> NodeId {
    let symbol = module.intern(name);
    module.alloc(Node::Const(symbol))
}

fn row(module: &mut Module, array: NodeId, index: NodeId) -> NodeId {
    let all = constant(module, "all");
    build_call(module, "get", &[array, index, all])
}

fn field(module: &mut Module, value: NodeId, name: flatppl_core::Symbol) -> NodeId {
    let name = module.resolve(name).into();
    let key = module.alloc(Node::Lit(Scalar::Str(name)));
    build_call(module, "get", &[value, key])
}

fn divide_cotangent(module: &mut Module, bar: NodeId, scale: NodeId) -> NodeId {
    let quotient = tensor::broadcast(module, "divide", &[bar, scale]);
    tensor::zero_when_masked(module, bar, quotient, true)
}
