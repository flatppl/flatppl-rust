use flatppl_core::graph::build_call;
use flatppl_core::{Dim, Module, Node, NodeId};

use crate::{Emitter, Step, contract, real, tensor};

/// Transpose a valid convolution or correlation with respect to one operand.
pub(super) fn window_transpose(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
) -> Option<NodeId> {
    let &[signal, kernel] = step.args.as_slice() else {
        return None;
    };
    let &[Dim::Static(width)] = tensor::numeric_shape(&emitter.module, kernel)? else {
        return None;
    };
    if width == 0 {
        return emitter.zero_like(*step.args.get(slot)?).ok();
    }
    let conv = step.op == "conv";
    if slot == 1 {
        let signal = emitter.primal(signal);
        let corr = build_call(&mut emitter.module, "crosscorr", &[signal, bar]);
        return Some(if conv {
            build_call(&mut emitter.module, "reverse", &[corr])
        } else {
            corr
        });
    }
    let kernel = emitter.primal(kernel);
    let module = &mut emitter.module;
    let padded = if width == 1 {
        bar
    } else {
        let length = tensor::int(module, i64::from(width - 1));
        let zeros = build_call(module, "zeros", &[length]);
        build_call(module, "cat", &[zeros, bar, zeros])
    };
    Some(build_call(
        module,
        if conv { "crosscorr" } else { "conv" },
        &[padded, kernel],
    ))
}

fn vector_length(module: &Module, value: NodeId) -> Option<u32> {
    match tensor::numeric_shape(module, value)? {
        [Dim::Static(n)] => Some(*n),
        _ => None,
    }
}

pub(super) fn grid_weight(emitter: &mut Emitter, step: &Step<'_>, slot: usize) -> Option<NodeId> {
    let extra = i64::from(step.op == "extlinspace");
    let length = i64::from(vector_length(&emitter.module, step.node)?) - 2 * extra;
    if slot > 1 || length <= 0 {
        return None;
    }
    let module = &mut emitter.module;
    let zero = real(module, 0.0);
    let one = real(module, 1.0);
    let weights = if length == 1 {
        // On the constrained line from == to, both endpoints move together.
        let half = real(module, 0.5);
        build_call(module, "vector", &[half])
    } else {
        let count = tensor::int(module, length);
        let (start, end) = if slot == 0 { (one, zero) } else { (zero, one) };
        build_call(module, "linspace", &[start, end, count])
    };
    Some(if extra == 1 {
        let end = build_call(module, "vector", &[zero]);
        build_call(module, "cat", &[end, weights, end])
    } else {
        weights
    })
}

/// Locate both order statistics from the primal quantile in one linear pass
/// over the data. Equal observations share the weight of their rank block.
pub(super) fn order_partial(emitter: &mut Emitter, step: &Step<'_>, slot: usize) -> Option<NodeId> {
    let data = step.args[0];
    let n = tensor::real_shape(&emitter.module, data)?
        .iter()
        .try_fold(1u64, |n, d| {
            if let Dim::Static(k) = d {
                n.checked_mul(u64::from(*k))
            } else {
                None
            }
        })?;
    if n == 0 {
        return None;
    }
    if n == 1 {
        let unit = real(&mut emitter.module, if slot == 0 { 1.0 } else { 0.0 });
        return if slot == 0 {
            tensor::fill_like(emitter, unit, data)
        } else {
            Some(unit)
        };
    }
    let data = emitter.primal(data);
    let q = emitter.primal(step.node);
    let p = if let Some(&p) = step.args.get(1) {
        emitter.primal(p)
    } else {
        real(&mut emitter.module, 0.5)
    };
    let m = &mut emitter.module;
    let one = tensor::int(m, 1);
    let scale = real(m, (n - 1) as f64);
    let last_pair = tensor::int(m, i64::try_from(n - 2).ok()?);
    let position = build_call(m, "mul", &[p, scale]);
    let rank = build_call(m, "floor", &[position]);
    let lower = build_call(m, "min", &[rank, last_pair]);
    let upper = build_call(m, "add", &[lower, one]);
    let fraction = build_call(m, "sub", &[position, lower]);
    let less = tensor::broadcast(m, "lt", &[data, q]);
    let greater = tensor::broadcast(m, "gt", &[data, q]);
    let at_most = tensor::broadcast(m, "le", &[data, q]);
    let n_less = build_call(m, "sum", &[less]);
    let n_at_most = build_call(m, "sum", &[at_most]);
    let neg_inf = real(m, f64::NEG_INFINITY);
    let inf = real(m, f64::INFINITY);
    let left = tensor::broadcast(m, "ifelse", &[less, data, neg_inf]);
    let right = tensor::broadcast(m, "ifelse", &[greater, data, inf]);
    let left = build_call(m, "maximum", &[left]);
    let right = build_call(m, "minimum", &[right]);
    let mut value_at = |rank| {
        let before = build_call(m, "lt", &[rank, n_less]);
        let within = build_call(m, "lt", &[rank, n_at_most]);
        let rest = build_call(m, "ifelse", &[within, q, right]);
        build_call(m, "ifelse", &[before, left, rest])
    };
    let lo = value_at(lower);
    let hi = value_at(upper);
    if slot == 1 {
        let spacing = build_call(m, "sub", &[hi, lo]);
        return Some(build_call(m, "mul", &[scale, spacing]));
    }
    let first = build_call(m, "sub", &[one, fraction]);
    let mut distribute = |value, weight| {
        let diff = tensor::broadcast(m, "sub", &[data, value]);
        let matches = tensor::broadcast(m, "iszero", &[diff]);
        let count = build_call(m, "sum", &[matches]);
        let share = build_call(m, "divide", &[weight, count]);
        let zero = real(m, 0.0);
        tensor::broadcast(m, "ifelse", &[matches, share, zero])
    };
    let a = distribute(lo, first);
    let b = distribute(hi, fraction);
    Some(tensor::broadcast(m, "add", &[a, b]))
}

/// dB/dx is a Bernstein polynomial of degree n-1 in the forward differences
/// of the coefficients. Coefficient derivatives are the binomial basis.
pub(super) fn bernstein(emitter: &mut Emitter, step: &Step<'_>, slope: bool) -> Option<NodeId> {
    let &[coefficients, point] = step.args.as_slice() else {
        return None;
    };
    let n = vector_length(&emitter.module, coefficients)?.checked_sub(1)?;
    let coefficients = emitter.primal(coefficients);
    let x = emitter.primal(point);
    let m = &mut emitter.module;
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    if n == 0 {
        return Some(if slope {
            zero
        } else {
            build_call(m, "vector", &[one])
        });
    }
    let count = tensor::int(m, i64::from(n));
    if slope {
        let indices = index_prefix(m, count);
        let one = tensor::int(m, 1);
        let following = tensor::broadcast(m, "add", &[indices, one]);
        let lo = build_call(m, "get", &[coefficients, indices]);
        let hi = build_call(m, "get", &[coefficients, following]);
        let delta = tensor::broadcast(m, "sub", &[hi, lo]);
        let value = build_call(m, "bernstein", &[delta, x]);
        return Some(build_call(m, "mul", &[count, value]));
    }
    let k = build_call(m, "indicesof0", &[coefficients]);
    let r = tensor::broadcast(m, "sub", &[count, k]);
    let n1 = real(m, f64::from(n) + 1.0);
    let cn = build_call(m, "loggamma", &[n1]);
    let k1 = tensor::broadcast(m, "add", &[k, one]);
    let r1 = tensor::broadcast(m, "add", &[r, one]);
    let ck = tensor::broadcast(m, "loggamma", &[k1]);
    let cr = tensor::broadcast(m, "loggamma", &[r1]);
    let log_c = tensor::broadcast(m, "sub", &[cn, ck]);
    let log_c = tensor::broadcast(m, "sub", &[log_c, cr]);
    let at_zero = build_call(m, "iszero", &[x]);
    let one_minus_x = build_call(m, "sub", &[one, x]);
    let at_one = build_call(m, "iszero", &[one_minus_x]);
    let boundary = build_call(m, "lor", &[at_zero, at_one]);
    let middle = real(m, 0.5);
    let safe = build_call(m, "ifelse", &[boundary, middle, x]);
    let lx = build_call(m, "log", &[safe]);
    let minus_x = build_call(m, "neg", &[safe]);
    let ly = build_call(m, "log1p", &[minus_x]);
    let tx = tensor::broadcast(m, "mul", &[k, lx]);
    let ty = tensor::broadcast(m, "mul", &[r, ly]);
    let exponent = tensor::broadcast(m, "add", &[log_c, tx]);
    let exponent = tensor::broadcast(m, "add", &[exponent, ty]);
    let interior = tensor::broadcast(m, "exp", &[exponent]);
    let first = tensor::broadcast(m, "iszero", &[k]);
    let last = tensor::broadcast(m, "iszero", &[r]);
    let first = tensor::broadcast(m, "ifelse", &[first, one, zero]);
    let last = tensor::broadcast(m, "ifelse", &[last, one, zero]);
    let interior = build_call(m, "ifelse", &[at_one, last, interior]);
    Some(build_call(m, "ifelse", &[at_zero, first, interior]))
}

fn index_prefix(m: &mut Module, length: NodeId) -> NodeId {
    let shape = build_call(m, "zeros", &[length]);
    build_call(m, "indicesof", &[shape])
}

pub(super) fn diagonal_scatter(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
) -> Option<NodeId> {
    let matrix = step.args[0];
    let &[Dim::Static(rows), Dim::Static(cols)] = tensor::numeric_shape(&emitter.module, matrix)?
    else {
        return None;
    };
    if rows == cols && step.args.len() == 1 {
        return Some(build_call(&mut emitter.module, "diagmat", &[bar]));
    }
    let length = vector_length(&emitter.module, step.node)?;
    if length == 0 {
        return emitter.zero_like(matrix).ok();
    }
    let offset = step.args.get(1).map(|&k| emitter.primal(k));
    let m = &mut emitter.module;
    let zero = tensor::int(m, 0);
    let offset = offset.unwrap_or(zero);
    let negative = build_call(m, "neg", &[offset]);
    let start = build_call(m, "max", &[zero, negative]);
    let axis_i = contract::axis(m, "adrow");
    let axis_j = contract::axis(m, "adcol");
    let rows = tensor::int(m, i64::from(rows));
    let cols = tensor::int(m, i64::from(cols));
    let row_ids = index_prefix(m, rows);
    let col_ids = index_prefix(m, cols);
    let row = contract::indexed(m, row_ids, &[axis_i]);
    let col = contract::indexed(m, col_ids, &[axis_j]);
    let delta = build_call(m, "sub", &[col, row]);
    let delta = build_call(m, "sub", &[delta, offset]);
    let diagonal = build_call(m, "iszero", &[delta]);
    let index = tensor::broadcast(m, "sub", &[row_ids, start]);
    let one = tensor::int(m, 1);
    let length = tensor::int(m, i64::from(length));
    let index = tensor::broadcast(m, "max", &[index, one]);
    let index = tensor::broadcast(m, "min", &[index, length]);
    let values = build_call(m, "get", &[bar, index]);
    let value = contract::indexed(m, values, &[axis_i]);
    let zero = real(m, 0.0);
    let entry = build_call(m, "ifelse", &[diagonal, value, zero]);
    let sum = m.intern("sum");
    let sum = m.alloc(Node::Const(sum));
    Some(contract::aggregate(m, sum, &[axis_i, axis_j], entry))
}

pub(super) fn step_mask(emitter: &mut Emitter, step: &Step<'_>) -> Option<NodeId> {
    let &[edges, values, point] = step.args.as_slice() else {
        return None;
    };
    let values = emitter.primal(values);
    let edges = emitter.primal(edges);
    let point = emitter.primal(point);
    let m = &mut emitter.module;
    let positions = build_call(m, "indicesof", &[values]);
    let indices = tensor::broadcast(m, "real", &[positions]);
    let chosen = build_call(m, "stepwise", &[edges, indices, point]);
    let difference = tensor::broadcast(m, "sub", &[indices, chosen]);
    let selected = tensor::broadcast(m, "iszero", &[difference]);
    Some(tensor::broadcast(m, "real", &[selected]))
}
