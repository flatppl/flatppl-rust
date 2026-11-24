//! Real-coordinate derivatives of complex eigenvalue and eigenvector projections.

use flatppl_core::graph::expect_builtin_call;

use super::factorization::{field, guarded};
use super::spectrum::{phase_separated, strict_extreme_mask};
use super::*;
use crate::{Key, Query};

pub(crate) fn is_eigen_projection(op: &str) -> bool {
    matches!(
        op,
        "eigen real values" | "eigen imag values" | "eigen real vectors" | "eigen imag vectors"
    )
}

pub(crate) fn eigen_projection(
    m: &Module,
    query: &Query<'_>,
    broadcast: Option<(&str, &[NodeId])>,
) -> Option<(&'static str, Vec<NodeId>)> {
    let (part, [projected]) = broadcast? else {
        return None;
    };
    let Key::Node(projection) = query.key(m, *projected).ok()? else {
        return None;
    };
    let projection = expect_builtin_call(m, projection, "get")?;
    let [factors, key] = *projection.args.as_ref() else {
        return None;
    };
    let Node::Lit(Scalar::Str(name)) = m.node(key) else {
        return None;
    };
    let op = match (part, name.as_ref()) {
        ("real", "values") => "eigen real values",
        ("imag", "values") => "eigen imag values",
        ("real", "vectors") => "eigen real vectors",
        ("imag", "vectors") => "eigen imag vectors",
        _ => return None,
    };
    let Key::Node(factors) = query.key(m, factors).ok()? else {
        return None;
    };
    let Node::Call(call) = m.node(factors) else {
        return None;
    };
    let CallHead::User(callee) = call.head else {
        return None;
    };
    let (module, member) = flatppl_determinizer::standard_function(m, callee)?;
    if module != "ext-linear-algebra" || member != "eigen" {
        return None;
    }
    let [a] = *call.args.as_ref() else {
        return None;
    };
    math::square_size(m, a)?;
    Some((op, vec![a, factors]))
}

pub(super) fn check_types(module: &Module, step: &Step<'_>) -> Result<(), Error> {
    if tensor::real_shape(module, step.node).is_some() {
        Ok(())
    } else {
        Err(Error::new(
            Some(step.node),
            "eigen projection AD needs a fixed real vector or matrix",
        ))
    }
}

fn separation(m: &mut Module, values: NodeId, n: i64) -> (NodeId, NodeId, NodeId) {
    let axis = indices(m, 1, n);
    let row_indices = column(m, axis);
    let col_indices = transpose(m, row_indices);
    let diagonal = tensor::broadcast(m, "equal", &[row_indices, col_indices]);
    let row_values = column(m, values);
    let col_values = transpose(m, row_values);
    let gap = tensor::broadcast(m, "sub", &[col_values, row_values]);
    let modulus = tensor::broadcast(m, "abs", &[gap]);
    let zero = real(m, 0.0);
    let distinct = tensor::broadcast(m, "lt", &[zero, modulus]);
    let allowed = tensor::broadcast(m, "lor", &[diagonal, distinct]);
    let simple = build_call(m, "lall", &[allowed]);
    let one = real(m, 1.0);
    let safe_gap = tensor::broadcast(m, "ifelse", &[diagonal, one, gap]);
    (diagonal, safe_gap, simple)
}

pub(crate) fn projection_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    forward: bool,
) -> Option<NodeId> {
    let [a, factors] = *step.args.as_slice() else {
        return None;
    };
    let n = math::square_size(&emitter.module, a)?;
    let factors = emitter.primal(factors);
    let m = &mut emitter.module;
    let values = field(m, factors, "values");
    let v = field(m, factors, "vectors");
    let (diagonal, gaps, simple) = separation(m, values, n);
    let inv = build_call(m, "inv", &[v]);
    let imaginary = matches!(step.op, "eigen imag values" | "eigen imag vectors");
    if matches!(step.op, "eigen real vectors" | "eigen imag vectors") {
        let pair = Eigenbasis {
            vectors: v,
            inverse: inv,
            diagonal,
            gaps,
            simple,
            size: n,
        };
        return Some(pair.vector_part(m, bar, imaginary, forward));
    }
    let value = if forward {
        let x = product(m, inv, bar);
        let x = product(m, x, v);
        build_call(m, "diag", &[x])
    } else {
        // For a real perturbation the real/imag projection commutes with
        // the bilinear trace pairing without conjugating the eigenbasis.
        let weights = build_call(m, "diagmat", &[bar]);
        let x = product(m, v, weights);
        let x = product(m, x, inv);
        transpose(m, x)
    };
    let part = if imaginary { "imag" } else { "real" };
    let value = tensor::broadcast(m, part, &[value]);
    Some(guarded(m, value, simple))
}

struct Eigenbasis {
    vectors: NodeId,
    inverse: NodeId,
    diagonal: NodeId,
    gaps: NodeId,
    simple: NodeId,
    size: i64,
}

struct PhaseChart {
    weights: NodeId,
    pivot: NodeId,
    valid: NodeId,
}

fn row(m: &mut Module, value: NodeId) -> NodeId {
    let value = column(m, value);
    transpose(m, value)
}

impl PhaseChart {
    fn new(m: &mut Module, v: NodeId, n: i64) -> Self {
        let magnitude = tensor::broadcast(m, "abs", &[v]);
        let real_vectors = tensor::broadcast(m, "real", &[v]);
        let all = m.intern("all");
        let all = m.alloc(Node::Const(all));
        let zero = real(m, 0.0);
        let one = real(m, 1.0);
        let mut columns = Vec::new();
        let mut pivots = Vec::new();
        let mut regular = Vec::new();
        for j in 1..=n {
            let j = tensor::int(m, j);
            let mag = build_call(m, "get", &[magnitude, all, j]);
            let at = strict_extreme_mask(m, mag, n, true);
            regular.push(phase_separated(m, mag));
            let weights = tensor::broadcast(m, "ifelse", &[at, one, zero]);
            let re = build_call(m, "get", &[real_vectors, all, j]);
            pivots.push(math::dot(m, weights, re));
            columns.push(weights);
        }
        let columns = build_call(m, "vector", &columns);
        let weights = build_call(m, "colstack", &[columns]);
        let pivot = build_call(m, "vector", &pivots);
        let regular = build_call(m, "vector", &regular);
        let valid = build_call(m, "lall", &[regular]);
        Self {
            weights,
            pivot,
            valid,
        }
    }

    fn phase_rate(&self, m: &mut Module, value: NodeId, n: i64) -> NodeId {
        let im = tensor::broadcast(m, "imag", &[value]);
        let all = m.intern("all");
        let all = m.alloc(Node::Const(all));
        let coordinates: Vec<_> = (1..=n)
            .map(|j| {
                let j = tensor::int(m, j);
                let im = build_call(m, "get", &[im, all, j]);
                let weights = build_call(m, "get", &[self.weights, all, j]);
                math::dot(m, im, weights)
            })
            .collect();
        let coordinates = build_call(m, "vector", &coordinates);
        tensor::broadcast(m, "divide", &[coordinates, self.pivot])
    }
}

impl Eigenbasis {
    fn offdiag(&self, m: &mut Module, input: NodeId, reverse: bool) -> NodeId {
        let gaps = if reverse {
            tensor::broadcast(m, "conj", &[self.gaps])
        } else {
            self.gaps
        };
        let divided = tensor::broadcast(m, "divide", &[input, gaps]);
        let zero = real(m, 0.0);
        tensor::broadcast(m, "ifelse", &[self.diagonal, zero, divided])
    }

    fn vector_part(&self, m: &mut Module, bar: NodeId, imaginary: bool, forward: bool) -> NodeId {
        let v = self.vectors;
        let vh = build_call(m, "adjoint", &[v]);
        let phase = PhaseChart::new(m, v, self.size);
        let valid = build_call(m, "land", &[self.simple, phase.valid]);
        let real_dot = |m: &mut Module, x| {
            let inner = product(m, vh, x);
            build_call(m, "diag", &[inner])
        };
        if forward {
            let transformed = product(m, self.inverse, bar);
            let transformed = product(m, transformed, v);
            let rates = self.offdiag(m, transformed, false);
            let raw = product(m, v, rates);
            // Remove the parallel component required by unit norm and the
            // imaginary component at the positive-real phase pivot.
            let along = real_dot(m, raw);
            let norm = tensor::broadcast(m, "real", &[along]);
            let theta = phase.phase_rate(m, raw, self.size);
            let scale = tensor::broadcast(m, "complex", &[norm, theta]);
            let scale = row(m, scale);
            let correction = tensor::broadcast(m, "mul", &[v, scale]);
            let tangent = build_call(m, "sub", &[raw, correction]);
            let part = if imaginary { "imag" } else { "real" };
            let tangent = tensor::broadcast(m, part, &[tangent]);
            return guarded(m, tangent, valid);
        }
        let zero = real(m, 0.0);
        let bar = if imaginary {
            tensor::broadcast(m, "complex", &[zero, bar])
        } else {
            bar
        };
        let inner = real_dot(m, bar);
        let norm = tensor::broadcast(m, "real", &[inner]);
        let norm = row(m, norm);
        let norm_part = tensor::broadcast(m, "mul", &[v, norm]);
        let im = tensor::broadcast(m, "imag", &[inner]);
        let theta = tensor::broadcast(m, "divide", &[im, phase.pivot]);
        let theta = row(m, theta);
        let phase_part = tensor::broadcast(m, "mul", &[phase.weights, theta]);
        let phase_part = tensor::broadcast(m, "complex", &[zero, phase_part]);
        let raw = build_call(m, "sub", &[bar, norm_part]);
        let raw = build_call(m, "sub", &[raw, phase_part]);
        let rates = product(m, vh, raw);
        let rates = self.offdiag(m, rates, true);
        let inverse_h = build_call(m, "adjoint", &[self.inverse]);
        let result = product(m, inverse_h, rates);
        let result = product(m, result, vh);
        let result = tensor::broadcast(m, "real", &[result]);
        guarded(m, result, valid)
    }
}
