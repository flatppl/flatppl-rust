//! Shape-dependent rules for the real extended linear algebra functions.

use flatppl_core::graph::{build_call, resolve_ref};
use flatppl_core::{Call, CallHead, Dim, Module, Node, NodeId, Ref, RefNs, Scalar, Type};

use crate::{Emitter, Error, Step, math, real, record, tensor};

mod factorization;
pub(super) use factorization::factorization_part;
mod spectrum;
pub(super) use spectrum::spectrum_part;
mod complex_exponential;
pub(super) use complex_exponential::{
    complex_exponential_part, is_matexp_projection, matexp_projection,
};
mod projection;
pub(super) use projection::{eigen_projection, is_eigen_projection, projection_part};

pub(super) fn structural(op: &str) -> bool {
    matches!(
        op,
        "kron"
            | "solve_triangular"
            | "cholesky_solve"
            | "lu_solve"
            | "lu"
            | "qr"
            | "eigh"
            | "eigvalsh"
            | "eigmax"
            | "eigmin"
    ) || is_eigen_projection(op)
        || is_matexp_projection(op)
}

fn matrix_shape(shape: &[Dim]) -> Option<(i64, i64)> {
    match shape {
        [Dim::Static(rows), Dim::Static(cols)] if *rows > 0 && *cols > 0 => {
            Some((i64::from(*rows), i64::from(*cols)))
        }
        _ => None,
    }
}

pub(super) fn check_types(module: &Module, step: &Step<'_>) -> Result<(), Error> {
    if step.broadcast {
        return Err(Error::new(
            Some(step.node),
            format!(
                "broadcasted `{}` needs a cellwise linear algebra rule",
                step.op
            ),
        ));
    }
    if is_matexp_projection(step.op) {
        return complex_exponential::check_types(module, step);
    }
    if is_eigen_projection(step.op) {
        return projection::check_types(module, step);
    }
    if matches!(step.op, "eigh" | "eigvalsh" | "eigmax" | "eigmin") {
        return spectrum::check_types(module, step);
    }
    if matches!(step.op, "lu" | "qr") {
        return factorization::check_types(module, step);
    }
    if step.op == "solve_triangular"
        && let Node::Call(call) = module.node(step.node)
        && triangular_flags(module, &call.args).is_none()
    {
        return Err(Error::new(
            Some(step.node),
            "solve_triangular AD needs fixed Boolean flags",
        ));
    }
    let valid = (|| {
        let [a, b] = *step.args.as_slice() else {
            return None;
        };
        let b = tensor::real_shape(module, b)?;
        let output = tensor::real_shape(module, step.node)?;
        if step.op == "kron" {
            matrix_shape(tensor::real_shape(module, a)?)?;
            matrix_shape(b)?;
            matrix_shape(output)?;
        } else {
            if !matches!(b, [Dim::Static(_)] | [Dim::Static(_), Dim::Static(_)]) || b != output {
                return None;
            }
            if step.op == "lu_solve" {
                let Type::Record(fields) = module.type_of(a)? else {
                    return None;
                };
                if fields.len() != 3 {
                    return None;
                }
                for name in ["L", "U"] {
                    let (_, ty) = fields
                        .iter()
                        .find(|(field, _)| module.resolve(*field) == name)?;
                    let (rows, cols) = matrix_shape(tensor::real_shape_of(ty)?)?;
                    if rows != cols || b[0] != Dim::Static(rows as u32) {
                        return None;
                    }
                }
            } else {
                math::square_size(module, a)?;
            }
        }
        Some(())
    })();
    valid.ok_or_else(|| {
        Error::new(
            Some(step.node),
            format!(
                "`{}` AD needs fixed real matrices and compatible real right-hand sides",
                step.op
            ),
        )
    })
}

fn triangular_flags(module: &Module, args: &[NodeId]) -> Option<(bool, bool)> {
    let [_, _, flags @ ..] = args else {
        return None;
    };
    let flag = |index, default| match flags.get(index) {
        Some(&node) => match module.node(resolve_ref(module, node).0) {
            Node::Lit(Scalar::Bool(value)) => Some(*value),
            _ => None,
        },
        None => Some(default),
    };
    (flags.len() <= 2).then_some(())?;
    Some((flag(0, true)?, flag(1, false)?))
}

fn user_call(module: &mut Module, callee: NodeId, args: &[NodeId]) -> NodeId {
    module.alloc(Node::Call(Call {
        head: CallHead::User(callee),
        args: args.into(),
        named: Box::new([]),
        inputs: None,
    }))
}

fn callee(module: &Module, node: NodeId) -> Option<NodeId> {
    match module.node(node) {
        Node::Call(Call {
            head: CallHead::User(callee),
            ..
        }) => Some(*callee),
        _ => None,
    }
}

fn member(module: &mut Module, callee: NodeId, name: &str) -> Option<NodeId> {
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        ..
    }) = module.node(resolve_ref(module, callee).0)
    else {
        return None;
    };
    let ns = RefNs::Module(*alias);
    let name = module.intern(name);
    Some(module.alloc(Node::Ref(Ref { ns, name })))
}

fn indices(module: &mut Module, start: i64, count: i64) -> NodeId {
    let values: Vec<_> = (start..start + count)
        .map(|i| tensor::int(module, i))
        .collect();
    build_call(module, "vector", &values)
}

pub(super) fn kron_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
    forward: bool,
) -> Option<NodeId> {
    let [a, b] = *step.args.as_slice() else {
        return None;
    };
    let (rows, cols) = matrix_shape(tensor::real_shape(&emitter.module, a)?)?;
    let (block_rows, block_cols) = matrix_shape(tensor::real_shape(&emitter.module, b)?)?;
    let callee = callee(&emitter.module, step.node)?;
    let a = emitter.primal(a);
    let b = emitter.primal(b);
    let m = &mut emitter.module;
    if forward {
        let args = if slot == 0 { [bar, b] } else { [a, bar] };
        return Some(user_call(m, callee, &args));
    }

    // A[i,j] multiplies the entire corresponding block of B.
    let all = m.intern("all");
    let all = m.alloc(Node::Const(all));
    let mut rows_of_d_a = Vec::new();
    let mut d_b = Vec::new();
    let col_selectors: Vec<_> = (0..cols)
        .map(|j| indices(m, 1 + j * block_cols, block_cols))
        .collect();
    for i in 0..rows {
        let row_selector = indices(m, 1 + i * block_rows, block_rows);
        let row = build_call(m, "get", &[bar, row_selector, all]);
        let i = tensor::int(m, i + 1);
        let mut d_a_row = Vec::new();
        for (j, &selector) in (1..=cols).zip(&col_selectors) {
            let block = build_call(m, "get", &[row, all, selector]);
            if slot == 0 {
                d_a_row.push(math::dot(m, block, b));
            } else {
                let j = tensor::int(m, j);
                let cell = build_call(m, "get", &[a, i, j]);
                d_b.push(build_call(m, "mul", &[cell, block]));
            }
        }
        if slot == 0 {
            rows_of_d_a.push(build_call(m, "vector", &d_a_row));
        }
    }
    if slot == 0 {
        let rows = build_call(m, "vector", &rows_of_d_a);
        Some(build_call(m, "rowstack", &[rows]))
    } else {
        d_b.into_iter().reduce(|a, b| build_call(m, "add", &[a, b]))
    }
}

fn triangular_call(
    m: &mut Module,
    callee: NodeId,
    a: NodeId,
    b: NodeId,
    lower: bool,
    unit: bool,
) -> NodeId {
    let lower = m.alloc(Node::Lit(Scalar::Bool(lower)));
    let unit = m.alloc(Node::Lit(Scalar::Bool(unit)));
    user_call(m, callee, &[a, b, lower, unit])
}

fn triangle(m: &mut Module, value: NodeId, n: i64, lower: bool, strict: bool) -> NodeId {
    let indices = indices(m, 1, n);
    let zero_axes = tensor::int(m, 0);
    let one_axis = tensor::int(m, 1);
    let rows = build_call(m, "addaxes", &[indices, zero_axes, one_axis]);
    let cols = build_call(m, "addaxes", &[indices, one_axis, zero_axes]);
    let args = if lower { [cols, rows] } else { [rows, cols] };
    let used = tensor::broadcast(m, if strict { "lt" } else { "le" }, &args);
    let zero = real(m, 0.0);
    tensor::broadcast(m, "ifelse", &[used, value, zero])
}

fn product(m: &mut Module, lhs: NodeId, rhs: NodeId) -> NodeId {
    build_call(m, "mul", &[lhs, rhs])
}

fn transpose(m: &mut Module, x: NodeId) -> NodeId {
    build_call(m, "transpose", &[x])
}

fn column(m: &mut Module, x: NodeId) -> NodeId {
    let before = tensor::int(m, 0);
    let after = tensor::int(m, 1);
    build_call(m, "addaxes", &[x, before, after])
}

// The same real Frobenius pairing covers one or many right-hand sides.
fn paired_rows(m: &mut Module, left: NodeId, right: NodeId, vector: bool) -> NodeId {
    let (left, right) = if vector {
        (column(m, left), column(m, right))
    } else {
        (left, right)
    };
    let right = transpose(m, right);
    product(m, left, right)
}

pub(super) fn solve_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
    forward: bool,
) -> Option<NodeId> {
    let [a, b] = *step.args.as_slice() else {
        return None;
    };
    if step.op == "lu_solve" {
        return lu_solve_part(emitter, step, bar, slot, forward);
    }
    let callee = callee(&emitter.module, step.node)?;
    let n = math::square_size(&emitter.module, a)?;
    let vector = tensor::real_shape(&emitter.module, b)?.len() == 1;
    let a = emitter.primal(a);
    let x = emitter.primal(step.node);
    let m = &mut emitter.module;
    if step.op == "solve_triangular" {
        let Node::Call(call) = m.node(step.node) else {
            return None;
        };
        let (lower, unit) = triangular_flags(m, &call.args)?;
        if forward {
            let rhs = if slot == 0 {
                let da = triangle(m, bar, n, lower, unit);
                let dax = product(m, da, x);
                build_call(m, "neg", &[dax])
            } else {
                bar
            };
            Some(triangular_call(m, callee, a, rhs, lower, unit))
        } else {
            let adjoint = transpose(m, a);
            let z = triangular_call(m, callee, adjoint, bar, !lower, unit);
            if slot == 1 {
                return Some(z);
            }
            let za = paired_rows(m, z, x, vector);
            let za = build_call(m, "neg", &[za]);
            Some(triangle(m, za, n, lower, unit))
        }
    } else if forward {
        let rhs = if slot == 0 {
            let l = triangle(m, a, n, true, false);
            let dl = triangle(m, bar, n, true, false);
            let lt = transpose(m, l);
            let dlt = transpose(m, dl);
            let left = product(m, dl, lt);
            let right = product(m, l, dlt);
            let sum = build_call(m, "add", &[left, right]);
            let dx = product(m, sum, x);
            build_call(m, "neg", &[dx])
        } else {
            bar
        };
        Some(user_call(m, callee, &[a, rhs]))
    } else {
        let z = user_call(m, callee, &[a, bar]);
        if slot == 1 {
            return Some(z);
        }
        let zx = paired_rows(m, z, x, vector);
        let xz = transpose(m, zx);
        let sum = build_call(m, "add", &[zx, xz]);
        let l = triangle(m, a, n, true, false);
        let dz = product(m, sum, l);
        let dz = build_call(m, "neg", &[dz]);
        Some(triangle(m, dz, n, true, false))
    }
}

fn lu_solve_part(
    emitter: &mut Emitter,
    step: &Step<'_>,
    bar: NodeId,
    slot: usize,
    forward: bool,
) -> Option<NodeId> {
    let [factors, b] = *step.args.as_slice() else {
        return None;
    };
    let Type::Record(fields) = emitter.module.type_of(factors)?.clone() else {
        return None;
    };
    let field_name = |name| {
        fields
            .iter()
            .find(|(sym, _)| emitter.module.resolve(*sym) == name)
    };
    let (p_name, p_ty) = field_name("P")?;
    let (l_name, l_ty) = field_name("L")?;
    let (u_name, _) = field_name("U")?;
    let (n, _) = matrix_shape(tensor::real_shape_of(l_ty)?)?;
    let vector = tensor::real_shape(&emitter.module, b)?.len() == 1;
    let callee = callee(&emitter.module, step.node)?;
    let factors = emitter.primal(factors);
    let x = emitter.primal(step.node);
    let m = &mut emitter.module;
    if forward && slot == 1 {
        return Some(user_call(m, callee, &[factors, bar]));
    }
    let tri = member(m, callee, "solve_triangular")?;
    let l = record::project(m, factors, *l_name);
    let u = record::project(m, factors, *u_name);
    // The unit diagonal of L and the permutation matrix are discrete data.
    if forward {
        let upper = triangle(m, u, n, false, false);
        let y = product(m, upper, x);
        let dl = record::project(m, bar, *l_name);
        let dl = triangle(m, dl, n, true, true);
        let dly = product(m, dl, y);
        let dy = triangular_call(m, tri, l, dly, true, true);
        let du = record::project(m, bar, *u_name);
        let du = triangle(m, du, n, false, false);
        let dux = product(m, du, x);
        let rhs = build_call(m, "add", &[dy, dux]);
        let rhs = build_call(m, "neg", &[rhs]);
        return Some(triangular_call(m, tri, u, rhs, false, false));
    }
    let ut = transpose(m, u);
    let z_u = triangular_call(m, tri, ut, bar, true, false);
    let lt = transpose(m, l);
    let z_l = triangular_call(m, tri, lt, z_u, false, true);
    if slot == 1 {
        let p = record::project(m, factors, *p_name);
        let pt = transpose(m, p);
        return Some(product(m, pt, z_l));
    }
    let upper = triangle(m, u, n, false, false);
    let y = product(m, upper, x);
    let dl = paired_rows(m, z_l, y, vector);
    let dl = build_call(m, "neg", &[dl]);
    let dl = triangle(m, dl, n, true, true);
    let du = paired_rows(m, z_u, x, vector);
    let du = build_call(m, "neg", &[du]);
    let du = triangle(m, du, n, false, false);
    let p_bar = record::is_differentiable(p_ty).then(|| {
        let zero = real(m, 0.0);
        let size = tensor::int(m, n);
        let dims = build_call(m, "vector", &[size, size]);
        build_call(m, "fill", &[zero, dims])
    });
    let bar = fields
        .iter()
        .filter_map(|(name, _)| {
            if name == l_name {
                Some((*name, dl))
            } else if name == u_name {
                Some((*name, du))
            } else if name == p_name {
                p_bar.map(|value| (*name, value))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    Some(record::build_named(m, "record", &bar))
}
