//! Real-coordinate derivatives for projected complex matrix exponentials.

use flatppl_core::graph::expect_builtin_call;

use super::*;
use crate::{Key, Query};

pub(crate) fn is_matexp_projection(op: &str) -> bool {
    matches!(op, "matexp real projection" | "matexp imag projection")
}

pub(crate) fn matexp_projection(
    m: &Module,
    query: &Query<'_>,
    broadcast: Option<(&str, &[NodeId])>,
) -> Option<(&'static str, Vec<NodeId>)> {
    let (part, [projected]) = broadcast? else {
        return None;
    };
    let op = match part {
        "real" => "matexp real projection",
        "imag" => "matexp imag projection",
        _ => return None,
    };
    let Key::Node(projection) = query.key(m, *projected).ok()? else {
        return None;
    };
    let Node::Call(call) = m.node(projection) else {
        return None;
    };
    let CallHead::User(callee) = call.head else {
        return None;
    };
    let (module, member) = flatppl_determinizer::standard_function(m, callee)?;
    if module != "ext-linear-algebra" || member != "matexp" {
        return None;
    }
    let [complex] = *call.args.as_ref() else {
        return None;
    };
    let Key::Node(complex) = query.key(m, complex).ok()? else {
        return None;
    };
    let [re, im] = if let Some(call) = expect_builtin_call(m, complex, "complex") {
        let [re, im] = *call.args.as_ref() else {
            return None;
        };
        [re, im]
    } else {
        let call = expect_builtin_call(m, complex, "broadcast")?;
        let [head, re, im] = *call.args.as_ref() else {
            return None;
        };
        if !matches!(m.node(head), Node::Const(name) if m.resolve(*name) == "complex") {
            return None;
        }
        [re, im]
    };
    let n = math::square_size(m, re)?;
    if math::square_size(m, im)? != n || n > i64::from(i32::MAX) / 4 {
        return None;
    }
    Some((op, vec![re, im, projection]))
}

pub(super) fn check_types(module: &Module, step: &Step<'_>) -> Result<(), Error> {
    if math::square_size(module, step.node).is_some() {
        Ok(())
    } else {
        Err(Error::new(
            Some(step.node),
            "complex matexp projection AD needs fixed real square coordinates",
        ))
    }
}

fn block(m: &mut Module, parts: [NodeId; 4], n: i64) -> NodeId {
    let all = m.intern("all");
    let all = m.alloc(Node::Const(all));
    let mut rows = Vec::new();
    for [left, right] in [[parts[0], parts[1]], [parts[2], parts[3]]] {
        for i in 1..=n {
            let i = tensor::int(m, i);
            let left = build_call(m, "get", &[left, i, all]);
            let right = build_call(m, "get", &[right, i, all]);
            rows.push(build_call(m, "cat", &[left, right]));
        }
    }
    let rows = build_call(m, "vector", &rows);
    build_call(m, "rowstack", &[rows])
}

fn select(m: &mut Module, value: NodeId, n: i64, bottom: bool, right: bool) -> NodeId {
    let all = m.intern("all");
    let all = m.alloc(Node::Const(all));
    let rows = indices(m, 1 + i64::from(bottom) * n, n);
    let cols = indices(m, 1 + i64::from(right) * n, n);
    let rows = build_call(m, "get", &[value, rows, all]);
    build_call(m, "get", &[rows, all, cols])
}

pub(crate) fn complex_exponential_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
    forward: bool,
) -> Option<NodeId> {
    let [re, im, primal] = *step.args.as_slice() else {
        return None;
    };
    let n = math::square_size(&emitter.module, re)?;
    let callee = callee(&emitter.module, primal)?;
    let re = emitter.primal(re);
    let im = emitter.primal(im);
    let m = &mut emitter.module;
    // Real block embedding preserves matrix multiplication and exp.
    let negative = build_call(m, "neg", &[im]);
    let primal = block(m, [re, negative, im, re], n);
    let zero = real(m, 0.0);
    let size = tensor::int(m, n);
    let dims = build_call(m, "vector", &[size, size]);
    let zero = build_call(m, "fill", &[zero, dims]);
    let imaginary = step.op == "matexp imag projection";
    if forward {
        let direction = if slot == 0 {
            block(m, [bar, zero, zero, bar], n)
        } else {
            let negative = build_call(m, "neg", &[bar]);
            block(m, [zero, negative, bar, zero], n)
        };
        let dx = math::matrix_exp_frechet_block(emitter, callee, 2 * n, primal, direction);
        return Some(select(&mut emitter.module, dx, n, imaginary, false));
    }
    // Lift the real output cotangent into its selected block, and apply the
    // transpose linear map of the input embedding after the real Fréchet VJP.
    let direction = if imaginary {
        block(m, [zero, zero, bar, zero], n)
    } else {
        block(m, [bar, zero, zero, zero], n)
    };
    let primal = transpose(m, primal);
    let dx = math::matrix_exp_frechet_block(emitter, callee, 2 * n, primal, direction);
    let m = &mut emitter.module;
    if slot == 0 {
        let a = select(m, dx, n, false, false);
        let b = select(m, dx, n, true, true);
        Some(build_call(m, "add", &[a, b]))
    } else {
        let a = select(m, dx, n, true, false);
        let b = select(m, dx, n, false, true);
        Some(build_call(m, "sub", &[a, b]))
    }
}
