//! Local real spectral derivatives at a simple spectrum.

use super::factorization::{field, guarded};
use super::*;

pub(super) fn check_types(module: &Module, step: &Step<'_>) -> Result<(), Error> {
    if let [input] = step.args.as_slice()
        && math::square_size(module, *input).is_some()
        && module
            .type_of(step.node)
            .is_some_and(record::is_differentiable)
    {
        return Ok(());
    }
    Err(Error::new(
        Some(step.node),
        format!(
            "`{}` AD needs a fixed real square matrix and a real result",
            step.op
        ),
    ))
}

fn nonzero(m: &mut Module, value: NodeId) -> NodeId {
    let zero = tensor::broadcast(m, "iszero", &[value]);
    tensor::broadcast(m, "lnot", &[zero])
}

fn spectral_gaps(m: &mut Module, values: NodeId, n: i64) -> (NodeId, NodeId, NodeId) {
    let idx = indices(m, 1, n);
    let row_idx = column(m, idx);
    let col_idx = transpose(m, row_idx);
    let diag = tensor::broadcast(m, "equal", &[row_idx, col_idx]);
    let row_values = column(m, values);
    let col_values = transpose(m, row_values);
    let diffs = tensor::broadcast(m, "sub", &[col_values, row_values]);
    let one = real(m, 1.0);
    let gaps = tensor::broadcast(m, "ifelse", &[diag, one, diffs]);
    let distinct = nonzero(m, gaps);
    let simple = build_call(m, "lall", &[distinct]);
    (diag, gaps, simple)
}

fn off_diagonal_rates(m: &mut Module, perturbation: NodeId, diag: NodeId, gaps: NodeId) -> NodeId {
    let raw = tensor::broadcast(m, "divide", &[perturbation, gaps]);
    let zero = real(m, 0.0);
    tensor::broadcast(m, "ifelse", &[diag, zero, raw])
}

// Compare each candidate directly with its other lanes. A reduction
// followed by exact scalar equality can recompute a lane at a different
// precision on CPU and fail to select it.
pub(super) fn strict_extreme_mask(m: &mut Module, values: NodeId, n: i64, maximum: bool) -> NodeId {
    let axis = indices(m, 1, n);
    let selectors: Vec<_> = (1..=n)
        .map(|i| {
            let i = tensor::int(m, i);
            let ignore = tensor::broadcast(m, "equal", &[axis, i]);
            let value = build_call(m, "get", &[values, i]);
            let args = if maximum {
                [values, value]
            } else {
                [value, values]
            };
            let wins = tensor::broadcast(m, "lt", &args);
            let wins = tensor::broadcast(m, "lor", &[ignore, wins]);
            build_call(m, "lall", &[wins])
        })
        .collect();
    build_call(m, "vector", &selectors)
}

// Leave a guard band for both backend precisions. A true tie can be
// emitted as unequal magnitudes by the numerical eigensolver.
pub(super) fn phase_separated(m: &mut Module, magnitudes: NodeId) -> NodeId {
    let maximum = build_call(m, "maximum", &[magnitudes]);
    let margin = real(m, 1.0 - 8.0 * f64::from(f32::EPSILON));
    let cutoff = product(m, maximum, margin);
    let contenders = tensor::broadcast(m, "le", &[cutoff, magnitudes]);
    let zero = tensor::int(m, 0);
    let one = tensor::int(m, 1);
    let counts = tensor::broadcast(m, "ifelse", &[contenders, one, zero]);
    let count = build_call(m, "sum", &[counts]);
    build_call(m, "equal", &[count, one])
}

// A tie for the largest coordinate can switch the sign convention.
fn unique_phases(m: &mut Module, vectors: NodeId, n: i64) -> NodeId {
    let all = m.intern("all");
    let all = m.alloc(Node::Const(all));
    let columns: Vec<_> = (1..=n)
        .map(|j| {
            let j = tensor::int(m, j);
            let vector = build_call(m, "get", &[vectors, all, j]);
            let magnitudes = tensor::broadcast(m, "abs", &[vector]);
            phase_separated(m, magnitudes)
        })
        .collect();
    let columns = build_call(m, "vector", &columns);
    build_call(m, "lall", &[columns])
}

pub(crate) fn spectrum_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    forward: bool,
) -> Option<NodeId> {
    let [a] = *step.args.as_slice() else {
        return None;
    };
    let n = math::square_size(&emitter.module, a)?;
    let callee = callee(&emitter.module, step.node)?;
    let a = emitter.primal(a);
    let m = &mut emitter.module;
    if matches!(step.op, "eigmax" | "eigmin") {
        let eigen = member(m, callee, "eigen")?;
        let factors = user_call(m, eigen, &[a]);
        return Some(extremum_part(
            m,
            factors,
            bar,
            n,
            step.op == "eigmax",
            forward,
        ));
    }
    let eigh = member(m, callee, "eigh")?;
    let factors = user_call(m, eigh, &[a]);
    Some(hermitian_part(
        m,
        factors,
        bar,
        n,
        step.op == "eigh",
        forward,
    ))
}

fn hermitian_part(
    m: &mut Module,
    factors: NodeId,
    bar: NodeId,
    n: i64,
    with_vectors: bool,
    forward: bool,
) -> NodeId {
    let values = field(m, factors, "values");
    let vectors = field(m, factors, "vectors");
    let vt = transpose(m, vectors);
    let (diag, gaps, simple) = spectral_gaps(m, values, n);
    if forward {
        let bar = math::symmetric(m, bar);
        let transformed = product(m, vt, bar);
        let transformed = product(m, transformed, vectors);
        let values = build_call(m, "diag", &[transformed]);
        let values = guarded(m, values, simple);
        if !with_vectors {
            return values;
        }
        // Off-diagonal rates are (V^T dA V)_ij / (lambda_j - lambda_i).
        let rates = off_diagonal_rates(m, transformed, diag, gaps);
        let dv = product(m, vectors, rates);
        let phase = unique_phases(m, vectors, n);
        let valid = build_call(m, "land", &[simple, phase]);
        let dv = guarded(m, dv, valid);
        let values_name = m.intern("values");
        let vectors_name = m.intern("vectors");
        return record::build_named(m, "record", &[(values_name, values), (vectors_name, dv)]);
    }
    let values_bar = if with_vectors {
        field(m, bar, "values")
    } else {
        bar
    };
    let diagonal = build_call(m, "diagmat", &[values_bar]);
    let (weights, valid) = if with_vectors {
        let vectors_bar = field(m, bar, "vectors");
        let transformed = product(m, vt, vectors_bar);
        let rates = off_diagonal_rates(m, transformed, diag, gaps);
        let weights = build_call(m, "add", &[diagonal, rates]);
        let inactive = tensor::broadcast(m, "iszero", &[vectors_bar]);
        let inactive = build_call(m, "lall", &[inactive]);
        let phase = unique_phases(m, vectors, n);
        let phase = build_call(m, "lor", &[inactive, phase]);
        (weights, build_call(m, "land", &[simple, phase]))
    } else {
        (diagonal, simple)
    };
    let result = product(m, vectors, weights);
    let result = product(m, result, vt);
    let result = math::symmetric(m, result);
    guarded(m, result, valid)
}

fn extremum_part(
    m: &mut Module,
    factors: NodeId,
    bar: NodeId,
    n: i64,
    maximum: bool,
    forward: bool,
) -> NodeId {
    let complex_values = field(m, factors, "values");
    let complex_vectors = field(m, factors, "vectors");
    let values = tensor::broadcast(m, "real", &[complex_values]);
    let vectors = tensor::broadcast(m, "real", &[complex_vectors]);
    let inverse = build_call(m, "inv", &[vectors]);
    let imag_values = tensor::broadcast(m, "imag", &[complex_values]);
    let imag_vectors = tensor::broadcast(m, "imag", &[complex_vectors]);
    let real_values = tensor::broadcast(m, "iszero", &[imag_values]);
    let real_values = build_call(m, "lall", &[real_values]);
    let real_vectors = tensor::broadcast(m, "iszero", &[imag_vectors]);
    let real_vectors = build_call(m, "lall", &[real_vectors]);
    let real_spectrum = build_call(m, "land", &[real_values, real_vectors]);
    let (_, _, simple) = spectral_gaps(m, values, n);
    let valid = build_call(m, "land", &[real_spectrum, simple]);
    let selected = strict_extreme_mask(m, values, n, maximum);
    let has_selection = build_call(m, "lany", &[selected]);
    let valid = build_call(m, "land", &[valid, has_selection]);
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let selector = tensor::broadcast(m, "ifelse", &[selected, one, zero]);
    // V^-1 dA V yields the rates of simple eigenvalues on its diagonal.
    let result = if forward {
        let transformed = product(m, inverse, bar);
        let transformed = product(m, transformed, vectors);
        let rates = build_call(m, "diag", &[transformed]);
        math::dot(m, rates, selector)
    } else {
        let weights = product(m, bar, selector);
        let diagonal = build_call(m, "diagmat", &[weights]);
        let pullback = product(m, vectors, diagonal);
        let pullback = product(m, pullback, inverse);
        transpose(m, pullback)
    };
    guarded(m, result, valid)
}
