use flatppl_core::graph::build_call;
use flatppl_core::{Dim, Module, NodeId};

use crate::{Emitter, real, tensor};

pub(super) fn product(
    emitter: &mut Emitter,
    input: NodeId,
    seed: NodeId,
    forward: bool,
    masked: bool,
) -> Option<NodeId> {
    let &[Dim::Static(length)] = tensor::real_shape(&emitter.module, input)? else {
        return None;
    };
    let mut primal = emitter.primal(input);
    let mut derivative = seed;
    let mut levels = Vec::new();
    let mut stride = 1;
    while stride < length {
        let shifted = shift(&mut emitter.module, primal, length, stride, 1.0, false);
        let shifted = bind(emitter, "_ad_cumprod_shift", shifted);
        if forward {
            let displaced = shift(&mut emitter.module, derivative, length, stride, 0.0, false);
            let m = &mut emitter.module;
            let direct = tensor::broadcast(m, "mul", &[derivative, shifted]);
            let indirect = tensor::broadcast(m, "mul", &[primal, displaced]);
            let tangent = tensor::broadcast(m, "add", &[direct, indirect]);
            derivative = bind(emitter, "_ad_cumprod_tangent", tangent);
        } else {
            levels.push((primal, shifted, stride));
        }
        if stride < length - stride {
            let next = tensor::broadcast(&mut emitter.module, "mul", &[primal, shifted]);
            primal = bind(emitter, "_ad_cumprod_partial", next);
        }
        stride = stride.saturating_mul(2);
    }

    for (primal, shifted, stride) in levels.into_iter().rev() {
        let m = &mut emitter.module;
        let direct = tensor::broadcast(m, "mul", &[derivative, shifted]);
        let indirect = tensor::broadcast(m, "mul", &[derivative, primal]);
        let (direct, indirect) = if masked {
            (
                tensor::zero_when_masked(m, derivative, direct, true),
                tensor::zero_when_masked(m, derivative, indirect, true),
            )
        } else {
            (direct, indirect)
        };
        // Transpose the left-padding shift by discarding the leading seed
        // cells and appending zeros at the high end.
        let indirect = shift(m, indirect, length, stride, 0.0, true);
        let cotangent = tensor::broadcast(m, "add", &[direct, indirect]);
        derivative = bind(emitter, "_ad_cumprod_cotangent", cotangent);
    }
    Some(derivative)
}

fn bind(emitter: &mut Emitter, name: &str, node: NodeId) -> NodeId {
    if emitter.inline {
        node
    } else {
        emitter.define(name, node)
    }
}

fn shift(
    module: &mut Module,
    input: NodeId,
    length: u32,
    stride: u32,
    pad: f64,
    transpose: bool,
) -> NodeId {
    let kept = tensor::int(module, i64::from(length - stride));
    let stride = tensor::int(module, i64::from(stride));
    let shape = build_call(module, "zeros", &[kept]);
    let indices = build_call(module, "indicesof", &[shape]);
    let indices = if transpose {
        tensor::broadcast(module, "add", &[indices, stride])
    } else {
        indices
    };
    let cells = build_call(module, "get", &[input, indices]);
    let fill = real(module, pad);
    let fill = build_call(module, "fill", &[fill, stride]);
    if transpose {
        build_call(module, "cat", &[cells, fill])
    } else {
        build_call(module, "cat", &[fill, cells])
    }
}
