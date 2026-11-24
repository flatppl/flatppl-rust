//! Graph construction for the shape-dependent terms in math.egg.

use flatppl_core::graph::build_call;
use flatppl_core::{Call, CallHead, Dim, Inputs, Module, Node, NodeId, Ref, RefNs};

use crate::{Emitter, real, tensor};

pub(super) fn structural(op: &str) -> bool {
    matches!(
        op,
        "indexed sum product"
            | "inner product"
            | "conv"
            | "crosscorr"
            | "cross"
            | "adjoint"
            | "self_outer"
            | "row_gram"
            | "col_gram"
            | "quadform"
            | "l1unit"
            | "l2unit"
            | "softmax"
            | "logsoftmax"
            | "quantile"
            | "median"
            | "linspace"
            | "extlinspace"
            | "bernstein"
            | "stepwise"
            | "polynomial"
            | "l2norm"
            | "l1norm"
            | "linfnorm"
            | "prod"
            | "var"
            | "std"
            | "maximum"
            | "minimum"
            | "trace"
            | "diagmat"
            | "diag"
            | "det"
            | "logabsdet"
            | "inv"
            | "linsolve"
            | "lower_cholesky"
            | "matrix exponential"
    )
}

pub(super) fn dot(module: &mut Module, left: NodeId, right: NodeId) -> NodeId {
    let products = tensor::broadcast(module, "mul", &[left, right]);
    build_call(module, "sum", &[products])
}

pub(super) fn symmetric(module: &mut Module, value: NodeId) -> NodeId {
    let transpose = build_call(module, "transpose", &[value]);
    let sum = build_call(module, "add", &[value, transpose]);
    let half = real(module, 0.5);
    build_call(module, "mul", &[half, sum])
}

pub(super) fn square_size(module: &Module, primal: NodeId) -> Option<i64> {
    match tensor::real_shape(module, primal)? {
        [Dim::Static(rows), Dim::Static(cols)] if rows == cols && *rows > 0 => {
            Some(i64::from(*rows))
        }
        _ => None,
    }
}

pub(super) fn matrix_exp_size(module: &Module, primal: NodeId) -> Option<i64> {
    match tensor::real_shape(module, primal)? {
        [Dim::Static(rows), Dim::Static(cols)] if rows == cols && *rows <= i32::MAX as u32 / 2 => {
            Some(i64::from(*rows))
        }
        _ => None,
    }
}

pub(super) fn matrix_exp_frechet(
    emitter: &mut Emitter,
    node: NodeId,
    primal: NodeId,
    base: NodeId,
    direction: NodeId,
) -> Option<NodeId> {
    let n = matrix_exp_size(&emitter.module, primal)?;
    if n == 0 {
        return Some(direction);
    }
    let Node::Call(Call {
        head: CallHead::User(callee),
        ..
    }) = emitter.module.node(node)
    else {
        return None;
    };
    let callee = *callee;
    Some(matrix_exp_frechet_block(
        emitter, callee, n, base, direction,
    ))
}

pub(super) fn matrix_exp_frechet_block(
    emitter: &mut Emitter,
    callee: NodeId,
    n: i64,
    base: NodeId,
    direction: NodeId,
) -> NodeId {
    let base = emitter.primal(base);
    let direction = emitter.primal(direction);
    let m = &mut emitter.module;

    // Bound the direction before building the block. A large direction would
    // otherwise force excess squarings and lose small diagonal perturbations.
    let magnitude = tensor::broadcast(m, "abs", &[direction]);
    let magnitude = build_call(m, "maximum", &[magnitude]);
    let one = real(m, 1.0);
    let scale = build_call(m, "max", &[one, magnitude]);
    let direction = tensor::broadcast(m, "divide", &[direction, scale]);

    // The upper-right block of exp([[A, H], [0, A]]) is D exp(A)[H].
    let all = m.intern("all");
    let all = m.alloc(Node::Const(all));
    let zero = real(m, 0.0);
    let size = tensor::int(m, n);
    let zeros = build_call(m, "fill", &[zero, size]);
    let mut top = Vec::new();
    let mut bottom = Vec::new();
    for index in 1..=n {
        let index = tensor::int(m, index);
        let a_row = build_call(m, "get", &[base, index, all]);
        let h_row = build_call(m, "get", &[direction, index, all]);
        top.push(build_call(m, "cat", &[a_row, h_row]));
        bottom.push(build_call(m, "cat", &[zeros, a_row]));
    }
    top.extend(bottom);
    let block = stack_rows(m, &top);
    let exponential = m.alloc(Node::Call(Call {
        head: CallHead::User(callee),
        args: Box::new([block]),
        named: Box::new([]),
        inputs: None,
    }));
    let rows = index_vector(m, 1..=n);
    let cols = index_vector(m, n + 1..=2 * n);
    let top = build_call(m, "get", &[exponential, rows, all]);
    let derivative = build_call(m, "get", &[top, all, cols]);
    build_call(m, "mul", &[scale, derivative])
}

pub(super) fn identity(emitter: &mut Emitter, primal: NodeId) -> Option<NodeId> {
    let n = square_size(&emitter.module, primal)?;
    let m = &mut emitter.module;
    let one = real(m, 1.0);
    let n = tensor::int(m, n);
    let ones = build_call(m, "fill", &[one, n]);
    Some(build_call(m, "diagmat", &[ones]))
}

pub(super) fn triangle_half(
    emitter: &mut Emitter,
    primal: NodeId,
    value: NodeId,
) -> Option<NodeId> {
    let n = square_size(&emitter.module, primal)?;
    let m = &mut emitter.module;
    let indices = index_vector(m, 1..=n);
    let zero_axes = tensor::int(m, 0);
    let one_axis = tensor::int(m, 1);
    let rows = build_call(m, "addaxes", &[indices, zero_axes, one_axis]);
    let cols = build_call(m, "addaxes", &[indices, one_axis, zero_axes]);
    let below = tensor::broadcast(m, "lt", &[cols, rows]);
    let diag = tensor::broadcast(m, "equal", &[cols, rows]);
    let zero = real(m, 0.0);
    let half = real(m, 0.5);
    let halved = build_call(m, "mul", &[half, value]);
    let diag = tensor::broadcast(m, "ifelse", &[diag, halved, zero]);
    Some(tensor::broadcast(m, "ifelse", &[below, value, diag]))
}

// The native determinant handles singular minors, so no inverse or division
// is needed. Bound the number of generated determinants until a cheaper
// general cofactor algorithm is available.
pub(super) const MAX_COFACTOR_SIZE: u32 = 8;

pub(super) fn cofactor_shape(module: &Module, primal: NodeId) -> bool {
    square_size(module, primal).is_some_and(|size| size <= i64::from(MAX_COFACTOR_SIZE))
}

pub(super) fn cofactors(emitter: &mut Emitter, primal: NodeId) -> Option<NodeId> {
    let n = square_size(&emitter.module, primal)?;
    let primal = emitter.primal(primal);
    let m = &mut emitter.module;
    if n == 1 {
        let one = real(m, 1.0);
        let row = build_call(m, "vector", &[one]);
        return Some(stack_rows(m, &[row]));
    }
    let exclusions: Vec<_> = (1..=n)
        .map(|excluded| index_vector(m, (1..=n).filter(|&i| i != excluded)))
        .collect();
    let all = m.intern("all");
    let all = m.alloc(flatppl_core::Node::Const(all));
    let rows: Vec<_> = exclusions
        .iter()
        .enumerate()
        .map(|(i, &row)| {
            let selected = build_call(m, "get", &[primal, row]);
            let cells: Vec<_> = exclusions
                .iter()
                .enumerate()
                .map(|(j, &col)| {
                    let minor = build_call(m, "get", &[selected, all, col]);
                    let value = build_call(m, "det", &[minor]);
                    if (i + j) % 2 == 0 {
                        value
                    } else {
                        build_call(m, "neg", &[value])
                    }
                })
                .collect();
            build_call(m, "vector", &cells)
        })
        .collect();
    Some(stack_rows(m, &rows))
}

fn index_vector(module: &mut Module, indices: impl IntoIterator<Item = i64>) -> NodeId {
    let indices: Vec<_> = indices
        .into_iter()
        .map(|i| tensor::int(module, i))
        .collect();
    build_call(module, "vector", &indices)
}

fn stack_rows(module: &mut Module, rows: &[NodeId]) -> NodeId {
    let rows = build_call(module, "vector", rows);
    build_call(module, "rowstack", &[rows])
}

pub(super) fn scalar_complex(module: &Module, node: NodeId) -> bool {
    module.type_of(node)
        == Some(&flatppl_core::Type::Scalar(
            flatppl_core::ScalarType::Complex,
        ))
}

pub(super) fn scalar_arithmetic(module: &Module, node: NodeId) -> bool {
    scalar_complex(module, node) || tensor::numeric_shape(module, node) == Some(&[])
}

pub(super) fn complex_rule(
    module: &Module,
    node: NodeId,
    args: &[NodeId],
    op: &str,
) -> Option<&'static str> {
    if !scalar_complex(module, node) && !args.iter().any(|&arg| scalar_complex(module, arg)) {
        return None;
    }
    Some(match op {
        "identity" => "complex identity",
        "neg" => "complex neg",
        "conj" => "complex conj",
        "add" => "complex add",
        "sub" => "complex sub",
        "mul" => "complex mul",
        "divide" => "complex divide",
        "real" => "complex real",
        "imag" => "complex imag",
        "complex" => "complex complex",
        "cis" => "complex cis",
        "abs2" => "complex abs2",
        "exp" => "complex exp",
        "log" => "complex log",
        "sqrt" => "complex sqrt",
        "abs" => "complex abs",
        "pow" => "complex pow",
        "sin" => "complex sin",
        "cos" => "complex cos",
        "tan" => "complex tan",
        "sinh" => "complex sinh",
        "cosh" => "complex cosh",
        "tanh" => "complex tanh",
        "asin" => "complex asin",
        "acos" => "complex acos",
        "atan" => "complex atan",
        "asinh" => "complex asinh",
        "acosh" => "complex acosh",
        "atanh" => "complex atanh",
        _ => return None,
    })
}

pub(super) fn complex_rule_name(op: &str) -> bool {
    op.starts_with("complex ")
}

pub(super) fn complex_pushforward(op: &str) -> bool {
    complex_rule_name(op)
        && !matches!(
            op,
            "complex identity" | "complex neg" | "complex conj" | "complex add" | "complex sub"
        )
}

// The output and operand can have different scalar kinds and therefore
// different kinds of cotangent. Select real components before rejoining.
pub(super) fn zero_when_masked_complex(
    module: &mut Module,
    bar: NodeId,
    contribution: NodeId,
    complex_bar: bool,
    complex_operand: bool,
) -> NodeId {
    let inactive = if complex_bar {
        let re = build_call(module, "real", &[bar]);
        let im = build_call(module, "imag", &[bar]);
        let re_zero = build_call(module, "iszero", &[re]);
        let im_zero = build_call(module, "iszero", &[im]);
        build_call(module, "land", &[re_zero, im_zero])
    } else {
        build_call(module, "iszero", &[bar])
    };
    let zero = real(module, 0.0);
    if complex_operand {
        let re = build_call(module, "real", &[contribution]);
        let im = build_call(module, "imag", &[contribution]);
        let re = build_call(module, "ifelse", &[inactive, zero, re]);
        let im = build_call(module, "ifelse", &[inactive, zero, im]);
        build_call(module, "complex", &[re, im])
    } else {
        build_call(module, "ifelse", &[inactive, zero, contribution])
    }
}

pub(super) fn requires_unmasked(op: &str) -> bool {
    matches!(
        op,
        "inv" | "logabsdet" | "linsolve" | "lower_cholesky" | "matrix exponential"
    )
}

pub(super) fn scalar_reduction(op: &str) -> bool {
    matches!(
        op,
        "polynomial"
            | "quantile"
            | "median"
            | "bernstein"
            | "stepwise"
            | "l2norm"
            | "l1norm"
            | "linfnorm"
            | "prod"
            | "var"
            | "std"
            | "maximum"
            | "minimum"
            | "trace"
            | "det"
    )
}

fn coefficient_count(emitter: &Emitter, coefficients: NodeId, x: NodeId) -> Option<i64> {
    match tensor::numeric_shape(&emitter.module, coefficients)? {
        [Dim::Static(n)] if tensor::real_shape(&emitter.module, x) == Some(&[]) => {
            Some(i64::from(*n))
        }
        _ => None,
    }
}

pub(super) fn polynomial_slope(
    emitter: &mut Emitter,
    coefficients: NodeId,
    x: NodeId,
) -> Option<NodeId> {
    let n = coefficient_count(emitter, coefficients, x)?;
    if n <= 1 {
        return Some(real(&mut emitter.module, 0.0));
    }
    let coefficients = emitter.primal(coefficients);
    let x = emitter.primal(x);
    let m = &mut emitter.module;
    let degrees = index_vector(m, 1..n);
    let tail_indices = index_vector(m, 2..=n);
    let tail = build_call(m, "get", &[coefficients, tail_indices]);
    let slopes = tensor::broadcast(m, "mul", &[degrees, tail]);
    Some(build_call(m, "polynomial", &[slopes, x]))
}

pub(super) fn polynomial_basis(
    emitter: &mut Emitter,
    coefficients: NodeId,
    x: NodeId,
) -> Option<NodeId> {
    let n = coefficient_count(emitter, coefficients, x)?;
    if n == 0 {
        return emitter.zero_like(coefficients).ok();
    }
    let x = emitter.primal(x);
    let m = &mut emitter.module;
    let degrees = index_vector(m, 0..n);
    Some(tensor::broadcast(m, "pow", &[x, degrees]))
}

pub(super) fn sample_degrees(module: &Module, primal: NodeId) -> Option<u64> {
    let shape = tensor::real_shape(module, primal)?;
    let count = shape.iter().try_fold(1u64, |acc, dim| match dim {
        Dim::Static(n) => acc.checked_mul(u64::from(*n)),
        Dim::Dynamic => None,
    })?;
    count.checked_sub(1).filter(|&degrees| degrees > 0)
}

pub(super) fn product_partials(emitter: &mut Emitter, primal: NodeId) -> Option<NodeId> {
    let &[Dim::Static(length)] = tensor::real_shape(&emitter.module, primal)? else {
        return None;
    };
    let one = real(&mut emitter.module, 1.0);
    if length <= 1 {
        return tensor::fill_like(emitter, one, primal);
    }
    // The partial at each cell is its exclusive prefix times its exclusive
    // suffix. This remains valid with one or several zeros in the input.
    let primal = emitter.primal(primal);
    let accumulate = multiply_callback(&mut emitter.module);
    let accumulate = emitter.define("_ad_product_step", accumulate);
    let m = &mut emitter.module;
    let prefix = build_call(m, "scan", &[accumulate, one, primal]);
    let reversed = build_call(m, "reverse", &[primal]);
    let suffix = build_call(m, "scan", &[accumulate, one, reversed]);
    let suffix = build_call(m, "reverse", &[suffix]);
    let size = tensor::int(m, i64::from(length - 1));
    let extent = build_call(m, "fill", &[one, size]);
    let head = build_call(m, "indicesof", &[extent]);
    let offset = tensor::int(m, 1);
    let tail = tensor::broadcast(m, "add", &[head, offset]);
    let prefix = build_call(m, "get", &[prefix, head]);
    let suffix = build_call(m, "get", &[suffix, tail]);
    let unit = build_call(m, "vector", &[one]);
    let prefix = build_call(m, "cat", &[unit, prefix]);
    let suffix = build_call(m, "cat", &[suffix, unit]);
    Some(tensor::broadcast(m, "mul", &[prefix, suffix]))
}

fn multiply_callback(module: &mut Module) -> NodeId {
    let inputs = ["prefix", "cell"].map(|name| {
        let label = module.intern(name);
        let name = module.intern(&format!("_{name}_"));
        (
            label,
            Ref {
                ns: RefNs::Local,
                name,
            },
        )
    });
    let args = inputs.map(|(_, reference)| module.alloc(Node::Ref(reference)));
    let body = build_call(module, "mul", &args);
    let function = module.intern("functionof");
    module.alloc(Node::Call(Call {
        head: CallHead::Builtin(function),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(Box::new(inputs))),
    }))
}
