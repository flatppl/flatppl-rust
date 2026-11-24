//! Local derivatives for pivoted LU and reduced QR.

use super::*;

pub(super) fn check_types(module: &Module, step: &Step<'_>) -> Result<(), Error> {
    if let [input] = step.args.as_slice()
        && let Some((rows, cols)) = tensor::real_shape(module, *input).and_then(matrix_shape)
        && (if step.op == "lu" {
            rows == cols
        } else {
            rows >= cols
        })
        && matches!(module.type_of(step.node), Some(ty @ Type::Record(_)) if record::is_differentiable(ty))
    {
        return Ok(());
    }
    Err(Error::new(
        Some(step.node),
        format!(
            "`{}` AD needs fixed real matrices and decomposition factors",
            step.op
        ),
    ))
}

fn right_solve(m: &mut Module, tri: NodeId, a: NodeId, upper: NodeId, adjoint: bool) -> NodeId {
    let a = transpose(m, a);
    let upper = if adjoint { upper } else { transpose(m, upper) };
    let solution = triangular_call(m, tri, upper, a, !adjoint, false);
    transpose(m, solution)
}

pub(in crate::linalg) fn field(m: &mut Module, value: NodeId, name: &str) -> NodeId {
    let name = m.intern(name);
    record::project(m, value, name)
}

fn diagonal_nonzero(m: &mut Module, a: NodeId) -> NodeId {
    let diagonal = build_call(m, "diag", &[a]);
    let magnitude = tensor::broadcast(m, "abs", &[diagonal]);
    let minimum = build_call(m, "minimum", &[magnitude]);
    let zero = real(m, 0.0);
    build_call(m, "lt", &[zero, minimum])
}

pub(in crate::linalg) fn guarded(m: &mut Module, value: NodeId, valid: NodeId) -> NodeId {
    let nan = real(m, f64::NAN);
    tensor::broadcast(m, "ifelse", &[valid, value, nan])
}

fn named_factors(m: &mut Module, fields: &[(&str, NodeId)]) -> NodeId {
    let fields = fields
        .iter()
        .map(|&(name, value)| (m.intern(name), value))
        .collect::<Vec<_>>();
    record::build_named(m, "record", &fields)
}

pub(crate) fn factorization_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    forward: bool,
) -> Option<NodeId> {
    let [a] = *step.args.as_slice() else {
        return None;
    };
    let (rows, cols) = matrix_shape(tensor::real_shape(&emitter.module, a)?)?;
    let m = &mut emitter.module;
    let tri = if step.op == "lu" {
        let callee = callee(m, step.node)?;
        member(m, callee, "solve_triangular")?
    } else {
        let alias = flatppl_core::graph::standard_module_alias(
            m,
            "ext-linear-algebra",
            "0.1",
            "__ext_linear_algebra",
        );
        let name = m.intern("solve_triangular");
        m.alloc(Node::Ref(Ref {
            ns: RefNs::Module(alias),
            name,
        }))
    };
    let output = emitter.primal(step.node);
    let m = &mut emitter.module;
    Some(if step.op == "lu" {
        lu_part(m, output, bar, rows, tri, forward)
    } else {
        qr_part(m, output, bar, (rows, cols), tri, forward)
    })
}

fn lu_part(
    m: &mut Module,
    output: NodeId,
    bar: NodeId,
    n: i64,
    tri: NodeId,
    forward: bool,
) -> NodeId {
    let p = field(m, output, "P");
    let l = field(m, output, "L");
    let u = field(m, output, "U");
    let candidates = triangle(m, l, n, true, true);
    let magnitude = tensor::broadcast(m, "abs", &[candidates]);
    let largest = build_call(m, "maximum", &[magnitude]);
    let one = real(m, 1.0);
    // Subdiagonal entries are candidate-to-pivot ratios. A unit magnitude
    // identifies a tie in partial pivoting at that step.
    let unique = build_call(m, "lt", &[largest, one]);
    let invertible = diagonal_nonzero(m, u);
    let valid = build_call(m, "land", &[unique, invertible]);
    if forward {
        let pa = product(m, p, bar);
        let ly = triangular_call(m, tri, l, pa, true, true);
        let y = right_solve(m, tri, ly, u, false);
        let lower = triangle(m, y, n, true, true);
        let upper = triangle(m, y, n, false, false);
        let dl = product(m, l, lower);
        let du = product(m, upper, u);
        let dl = guarded(m, dl, valid);
        let du = guarded(m, du, valid);
        let zero = real(m, 0.0);
        let n = tensor::int(m, n);
        let dims = build_call(m, "vector", &[n, n]);
        let dp = build_call(m, "fill", &[zero, dims]);
        let dp = guarded(m, dp, valid);
        named_factors(m, &[("P", dp), ("L", dl), ("U", du)])
    } else {
        let dl = field(m, bar, "L");
        let du = field(m, bar, "U");
        let lt = transpose(m, l);
        let ut = transpose(m, u);
        let lower = product(m, lt, dl);
        let lower = triangle(m, lower, n, true, true);
        let upper = product(m, du, ut);
        let upper = triangle(m, upper, n, false, false);
        let y = build_call(m, "add", &[lower, upper]);
        let x = triangular_call(m, tri, lt, y, false, true);
        let x = right_solve(m, tri, x, u, true);
        let pt = transpose(m, p);
        let x = product(m, pt, x);
        guarded(m, x, valid)
    }
}

// Peel the native Householder reflectors from thin Q. In the current
// reflector's coordinates, Q[k,k] is the original pivot divided by R[k,k].
// A zero pivot can switch the returned column signs under perturbation.
fn qr_signs_regular(m: &mut Module, q: NodeId, rows: i64, cols: i64) -> NodeId {
    let axis = indices(m, 1, rows);
    let all = m.intern("all");
    let all = m.alloc(Node::Const(all));
    let mut regular = m.alloc(Node::Lit(Scalar::Bool(true)));
    let mut peeled = q;
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let two = real(m, 2.0);
    for k in 1..=cols {
        let index = tensor::int(m, k);
        let pivot = build_call(m, "get", &[peeled, index, index]);
        let pivot_zero = build_call(m, "iszero", &[pivot]);
        let pivot_nonzero = build_call(m, "lnot", &[pivot_zero]);
        regular = build_call(m, "land", &[regular, pivot_nonzero]);
        if k == cols {
            break;
        }
        let vector = build_call(m, "get", &[peeled, all, index]);
        let at = tensor::broadcast(m, "equal", &[axis, index]);
        let basis = tensor::broadcast(m, "ifelse", &[at, one, zero]);
        let w = build_call(m, "sub", &[basis, vector]);
        let norm_squared = math::dot(m, w, w);
        let factor = build_call(m, "divide", &[two, norm_squared]);
        let w = column(m, w);
        let wt = transpose(m, w);
        let projection = product(m, wt, peeled);
        let correction = product(m, w, projection);
        let correction = product(m, factor, correction);
        peeled = build_call(m, "sub", &[peeled, correction]);
    }
    regular
}

fn qr_part(
    m: &mut Module,
    output: NodeId,
    bar: NodeId,
    shape: (i64, i64),
    tri: NodeId,
    forward: bool,
) -> NodeId {
    let (rows, n) = shape;
    let q = field(m, output, "Q");
    let r = field(m, output, "R");
    let qt = transpose(m, q);
    let rt = transpose(m, r);
    let invertible = diagonal_nonzero(m, r);
    let signs_regular = qr_signs_regular(m, q, rows, n);
    let valid = build_call(m, "land", &[invertible, signs_regular]);
    // For F = dA R^-1 and C = Q^T F, dQ = F - Q K and dR = K R,
    // where K is C minus its strictly lower, skew-symmetric part.
    if forward {
        let f = right_solve(m, tri, bar, r, false);
        let c = product(m, qt, f);
        let lower = triangle(m, c, n, true, true);
        let upper = triangle(m, c, n, false, false);
        let lower_t = transpose(m, lower);
        let k = build_call(m, "add", &[upper, lower_t]);
        let qk = product(m, q, k);
        let dq = build_call(m, "sub", &[f, qk]);
        let dr = product(m, k, r);
        let dq = guarded(m, dq, valid);
        let dr = guarded(m, dr, valid);
        named_factors(m, &[("Q", dq), ("R", dr)])
    } else {
        let dq = field(m, bar, "Q");
        let dr = field(m, bar, "R");
        let right = product(m, dr, rt);
        let left = product(m, qt, dq);
        let dk = build_call(m, "sub", &[right, left]);
        let upper = triangle(m, dk, n, false, false);
        let strict_upper = triangle(m, dk, n, false, true);
        let strict_lower = transpose(m, strict_upper);
        let dc = build_call(m, "add", &[upper, strict_lower]);
        let qdc = product(m, q, dc);
        let df = build_call(m, "add", &[dq, qdc]);
        let da = right_solve(m, tri, df, r, true);
        guarded(m, da, valid)
    }
}
