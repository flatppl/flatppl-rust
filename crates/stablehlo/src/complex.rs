//! Complex tensor operations and their real component lowerings.

use std::collections::HashMap;

use crate::mlir::{ElemKind, Lowered};
use crate::{EmitError, Emitter, Value};
use flatppl_core::{CallHead, Node, NodeId, ScalarType, Type};

pub(crate) fn complex_leaf(ty: Option<&Type>) -> bool {
    match ty {
        Some(Type::Scalar(ScalarType::Complex)) => true,
        Some(Type::Array { elem, .. } | Type::TVector { elem, .. }) => complex_leaf(Some(elem)),
        _ => false,
    }
}

pub(crate) fn has_complex(a: &Value, b: &Value) -> bool {
    a.elem == ElemKind::Complex || b.elem == ElemKind::Complex
}

pub(crate) fn split(e: &mut Emitter, value: &Value) -> (Value, Value) {
    if value.elem == ElemKind::Complex {
        (e.complex_part(value, false), e.complex_part(value, true))
    } else {
        let re = e.convert(value, ElemKind::Real);
        let im = e.constant_like(0.0, &re);
        (re, im)
    }
}

pub(crate) fn conjugate(e: &mut Emitter, value: &Value) -> Value {
    if value.elem != ElemKind::Complex {
        return value.clone();
    }
    let (re, im) = split(e, value);
    let im = e.neg(&im);
    e.join_complex(&re, &im)
}

pub(crate) fn multiply_parts<'m>(
    e: &mut Emitter<'m>,
    (ar, ai): (Value, Value),
    (br, bi): (Value, Value),
    mut product: impl FnMut(&mut Emitter<'m>, &Value, &Value) -> Value,
) -> (Value, Value) {
    let rr = product(e, &ar, &br);
    let ii = product(e, &ai, &bi);
    let ri = product(e, &ar, &bi);
    let ir = product(e, &ai, &br);
    (e.sub(&rr, &ii), e.add(&ri, &ir))
}

pub(crate) fn binary_parts(
    e: &mut Emitter,
    op: &str,
    left: (Value, Value),
    right: (Value, Value),
) -> (Value, Value) {
    let (ar, ai) = &left;
    let (br, bi) = &right;
    match op {
        "stablehlo.add" => (e.add(ar, br), e.add(ai, bi)),
        "stablehlo.subtract" => (e.sub(ar, br), e.sub(ai, bi)),
        "stablehlo.multiply" => multiply_parts(e, left, right, Emitter::mul),
        "stablehlo.divide" => {
            let abr = e.abs(br);
            let abi = e.abs(bi);
            let scale = e.max(&abr, &abi);
            let ar = e.div(ar, &scale);
            let ai = e.div(ai, &scale);
            let br = e.div(br, &scale);
            let bi = e.div(bi, &scale);
            let denom = squared_magnitude(e, &br, &bi);
            let rr = e.mul(&ar, &br);
            let ii = e.mul(&ai, &bi);
            let ir = e.mul(&ai, &br);
            let ri = e.mul(&ar, &bi);
            let re = e.add(&rr, &ii);
            let im = e.sub(&ir, &ri);
            (e.div(&re, &denom), e.div(&im, &denom))
        }
        "stablehlo.power" => {
            let log = logarithm(e, ar, ai);
            let (re, im) = multiply_parts(e, log, right.clone(), Emitter::mul);
            let (re, im) = exponential(e, &re, &im);
            let zero = e.scalar(0.0);
            let er_zero = e.compare("EQ", br, &zero);
            let ei_zero = e.compare("EQ", bi, &zero);
            let exponent_zero = e.and(&er_zero, &ei_zero);
            let one = e.scalar(1.0);
            (
                e.select(&exponent_zero, &one, &re),
                e.select(&exponent_zero, &zero, &im),
            )
        }
        _ => unreachable!("non-arithmetic complex operation: {op}"),
    }
}

pub(crate) fn squared_magnitude(e: &mut Emitter, re: &Value, im: &Value) -> Value {
    let re2 = e.mul(re, re);
    let im2 = e.mul(im, im);
    e.add(&re2, &im2)
}

/// A common positive scale keeps the squared components in range.
fn magnitude_parts(e: &mut Emitter, re: &Value, im: &Value) -> (Value, Value, Value) {
    let ar = e.abs(re);
    let ai = e.abs(im);
    let scale = e.max(&ar, &ai);
    let one = e.scalar(1.0);
    let zero = e.scalar(0.0);
    let nonzero = e.compare("GT", &scale, &zero);
    let inf = e.inf_like(&scale);
    let finite = e.compare("LT", &scale, &inf);
    let rescale = e.and(&nonzero, &finite);
    let safe = e.select(&rescale, &scale, &one);
    let ar = e.div(&ar, &safe);
    let ai = e.div(&ai, &safe);
    let norm = squared_magnitude(e, &ar, &ai);
    (scale, ar, norm)
}

pub(crate) fn magnitude(e: &mut Emitter, re: &Value, im: &Value) -> Value {
    let (scale, _, norm) = magnitude_parts(e, re, im);
    let norm = e.sqrt(&norm);
    e.mul(&scale, &norm)
}

fn logarithm(e: &mut Emitter, re: &Value, im: &Value) -> (Value, Value) {
    let (scale, _, norm) = magnitude_parts(e, re, im);
    let log_scale = e.log(&scale);
    let half = e.scalar(0.5);
    let log_norm = e.log(&norm);
    let log_norm = e.mul(&half, &log_norm);
    let log_abs = e.add(&log_scale, &log_norm);
    let arg = e.atan2(im, re);
    let zero = e.scalar(0.0);
    let negative_axis = e.compare("LT", re, &zero);
    let on_axis = e.compare("EQ", im, &zero);
    let negative_axis = e.and(&negative_axis, &on_axis);
    let pi = e.scalar(std::f64::consts::PI);
    (log_abs, e.select(&negative_axis, &pi, &arg))
}

fn exponential(e: &mut Emitter, re: &Value, im: &Value) -> (Value, Value) {
    let scale = e.exp(re);
    let cos = e.cos(im);
    let sin = e.sin(im);
    (e.mul(&scale, &cos), e.mul(&scale, &sin))
}

fn square_root(e: &mut Emitter, re: &Value, im: &Value) -> (Value, Value) {
    let (scale, ar, norm) = magnitude_parts(e, re, im);
    let abs = e.sqrt(&norm);
    let half = e.scalar(0.5);
    let sum = e.add(&ar, &abs);
    let normalized = e.mul(&half, &sum);
    let normalized = e.sqrt(&normalized);
    let root_scale = e.sqrt(&scale);
    let principal = e.mul(&root_scale, &normalized);
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let positive = e.compare("GT", &principal, &zero);
    let divisor = e.select(&positive, &principal, &one);
    let half_im = e.mul(&half, im);
    let secondary = e.div(&half_im, &divisor);
    let magnitude = e.abs(&secondary);
    let neg_principal = e.neg(&principal);
    let negative_im = e.compare("LT", im, &zero);
    let signed = e.select(&negative_im, &neg_principal, &principal);
    let left_half = e.compare("LT", re, &zero);
    (
        e.select(&left_half, &magnitude, &principal),
        e.select(&left_half, &signed, &secondary),
    )
}

/// Scaled tangent form. Dividing sin(z) by cos(z) overflows for a large
/// imaginary component even though tan(z) tends to ±i.
fn tangent(e: &mut Emitter, re: &Value, im: &Value) -> (Value, Value) {
    let magnitude = e.abs(im);
    let minus_two = e.scalar(-2.0);
    let exponent = e.mul(&minus_two, &magnitude);
    let decay = e.exp(&exponent);
    let delta = e.expm1(&exponent);
    let delta2 = e.mul(&delta, &delta);
    let four = e.scalar(4.0);
    let weight = e.mul(&four, &decay);
    let c = e.cos(re);
    let s = e.sin(re);
    let wc = e.mul(&weight, &c);
    let denominator = e.mul(&wc, &c);
    let denominator = e.add(&delta2, &denominator);
    let numerator = e.mul(&wc, &s);
    let real = e.div(&numerator, &denominator);
    let one = e.scalar(1.0);
    let sum = e.add(&one, &decay);
    let imaginary = e.mul(&delta, &sum);
    let negative = e.div(&imaginary, &denominator);
    let positive = e.neg(&negative);
    let zero = e.scalar(0.0);
    let lower = e.compare("LT", im, &zero);
    (real, e.select(&lower, &negative, &positive))
}

fn trigonometric(e: &mut Emitter, head: &str, re: &Value, im: &Value) -> (Value, Value) {
    match head {
        "tan" => tangent(e, re, im),
        "tanh" => {
            let rotated = e.neg(im);
            let (r, i) = tangent(e, &rotated, re);
            (i, e.neg(&r))
        }
        "sin" | "cos" => {
            let sr = e.sin(re);
            let cr = e.cos(re);
            let si = e.sinh(im);
            let ci = e.cosh(im);
            if head == "sin" {
                (e.mul(&sr, &ci), e.mul(&cr, &si))
            } else {
                let imaginary = e.mul(&sr, &si);
                (e.mul(&cr, &ci), e.neg(&imaginary))
            }
        }
        "sinh" | "cosh" => {
            let sr = e.sinh(re);
            let cr = e.cosh(re);
            let si = e.sin(im);
            let ci = e.cos(im);
            if head == "sinh" {
                (e.mul(&sr, &ci), e.mul(&cr, &si))
            } else {
                (e.mul(&cr, &ci), e.mul(&sr, &si))
            }
        }
        _ => unreachable!("not a trigonometric function"),
    }
}

// Use even/odd symmetry to evaluate the logarithmic forms on the upper
// half-plane, where their square roots agree with the principal branch.
fn inverse_hyperbolic(e: &mut Emitter, head: &str, re: &Value, im: &Value) -> (Value, Value) {
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let half = e.scalar(0.5);
    let y = e.abs(im);
    let (real, imaginary) = match head {
        "asinh" => {
            let x = e.abs(re);
            let (xx, xy) = multiply_parts(
                e,
                (x.clone(), y.clone()),
                (x.clone(), y.clone()),
                Emitter::mul,
            );
            let xx = e.add(&xx, &one);
            let (r, i) = square_root(e, &xx, &xy);
            let sum_re = e.add(&x, &r);
            let sum_im = e.add(&y, &i);
            let (r, i) = logarithm(e, &sum_re, &sum_im);
            // The logarithm of a sum near one loses small arguments.
            let small_limit = e.scalar(match e.dtype() {
                crate::Dtype::F32 => f32::EPSILON.sqrt() as f64,
                crate::Dtype::F64 => f64::EPSILON.sqrt(),
            });
            let radius = e.max(&x, &y);
            let small = e.compare("LT", &radius, &small_limit);
            let r = e.select(&small, &x, &r);
            let i = e.select(&small, &y, &i);
            let minus = e.neg(&r);
            let left = e.compare("LT", re, &zero);
            (e.select(&left, &minus, &r), i)
        }
        "acosh" => {
            let minus = e.sub(re, &one);
            let plus = e.add(re, &one);
            let (lr, li) = square_root(e, &minus, &y);
            let (rr, ri) = square_root(e, &plus, &y);
            let (real, imaginary) = multiply_parts(e, (lr, li), (rr, ri), Emitter::mul);
            let sum_re = e.add(re, &real);
            let sum_im = e.add(&y, &imaginary);
            logarithm(e, &sum_re, &sum_im)
        }
        "atanh" => {
            let plus = e.add(&one, re);
            let minus = e.sub(&one, re);
            let (log_plus, arg_plus) = logarithm(e, &plus, &y);
            let (log_minus, arg_minus) = logarithm(e, &minus, &y);
            let re = e.sub(&log_plus, &log_minus);
            let im = e.add(&arg_plus, &arg_minus);
            (e.mul(&half, &re), e.mul(&half, &im))
        }
        _ => unreachable!("not an inverse hyperbolic function"),
    };
    let lower = e.compare("LT", im, &zero);
    let minus = e.neg(&imaginary);
    (real, e.select(&lower, &minus, &imaginary))
}

fn unary_parts(e: &mut Emitter, head: &str, re: Value, im: Value) -> Option<(Value, Value)> {
    Some(match head {
        "conj" => (re, e.neg(&im)),
        "neg" => (e.neg(&re), e.neg(&im)),
        "log" => logarithm(e, &re, &im),
        "exp" => exponential(e, &re, &im),
        "sqrt" => square_root(e, &re, &im),
        "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" => trigonometric(e, head, &re, &im),
        "asinh" | "acosh" | "atanh" => inverse_hyperbolic(e, head, &re, &im),
        "asin" | "acos" | "atan" => {
            let rotated = e.neg(&im);
            let inverse = if head == "atan" { "atanh" } else { "asinh" };
            let (i, r) = inverse_hyperbolic(e, inverse, &rotated, &re);
            if head == "acos" {
                let half_pi = e.scalar(std::f64::consts::FRAC_PI_2);
                (e.sub(&half_pi, &r), i)
            } else {
                (r, e.neg(&i))
            }
        }
        _ => return None,
    })
}

pub(crate) fn lower_math(
    e: &mut Emitter,
    head: &str,
    args: &[NodeId],
) -> Option<Result<Value, EmitError>> {
    match (head, args) {
        ("im", []) => Some(Ok({
            let re = e.scalar(0.0);
            let im = e.scalar(1.0);
            e.join_complex(&re, &im)
        })),
        ("complex", [re, im]) => Some((|| {
            let re = e.lower_node(*re)?;
            let im = e.lower_node(*im)?;
            Ok(e.join_complex(&re, &im))
        })()),
        ("cis", [angle]) => Some(e.lower_node(*angle).map(|angle| {
            let re = e.cos(&angle);
            let im = e.sin(&angle);
            e.join_complex(&re, &im)
        })),
        (
            "log" | "exp" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "conj"
            | "neg" | "asin" | "acos" | "atan" | "asinh" | "acosh" | "atanh",
            [arg],
        ) if complex_leaf(e.type_of(*arg)) => {
            Some(parts(e, *arg, &mut HashMap::new()).map(|(re, im)| {
                let (re, im) = unary_parts(e, head, re, im).expect("complex elementary function");
                e.join_complex(&re, &im)
            }))
        }
        _ => None,
    }
}

fn user_parts(
    e: &mut Emitter,
    id: NodeId,
    callee: NodeId,
    args: &[NodeId],
    broadcast: bool,
) -> Option<Result<(Value, Value), EmitError>> {
    e.lower_kronecker_parts(id, callee, args, broadcast)
        .or_else(|| e.lower_matrix_exp_parts(id, callee, args, broadcast))
        .or_else(|| e.lower_bessel_parts(id, callee, args, broadcast))
        .or_else(|| e.lower_lstsq_parts(id, callee, args, broadcast))
        .or_else(|| e.lower_ext_linalg_parts(id, callee, args, broadcast))
}

pub(crate) fn lower_user(
    e: &mut Emitter,
    id: NodeId,
    callee: NodeId,
    args: &[NodeId],
    broadcast: bool,
) -> Option<Result<Value, EmitError>> {
    if !complex_leaf(e.type_of(id)) {
        return None;
    }
    user_parts(e, id, callee, args, broadcast)
        .map(|result| result.map(|(re, im)| e.join_complex(&re, &im)))
}

pub(crate) fn user_field_parts(
    e: &mut Emitter,
    id: NodeId,
    container: NodeId,
    field: &str,
) -> Option<Result<(Value, Value), EmitError>> {
    e.lower_eigh_field_parts(container, field)
        .or_else(|| e.lower_eigen_field_parts(container, field))
        .or_else(|| e.lower_lu_field_parts(container, field))
        .or_else(|| e.lower_svd_field_parts(id, container, field))
}

pub(crate) fn project(
    e: &mut Emitter,
    id: NodeId,
    head: &str,
    value: NodeId,
) -> Result<Value, EmitError> {
    let (re, im) = parts(e, value, &mut HashMap::new())?;
    match head {
        "real" => Ok(re),
        "imag" => Ok(im),
        "abs" => Ok(magnitude(e, &re, &im)),
        "abs2" => Ok(squared_magnitude(e, &re, &im)),
        _ => Err(EmitError::at(id, "unsupported complex projection")),
    }
}

pub(crate) fn parts(
    e: &mut Emitter,
    id: NodeId,
    memo: &mut HashMap<NodeId, (Value, Value)>,
) -> Result<(Value, Value), EmitError> {
    if let Some(result) = memo.get(&id) {
        return Ok(result.clone());
    }
    let resolved = e.resolve_ref_one(id);
    let result = if let Some(value) = e.bound_tensor(id) {
        split(e, &value)
    } else if !complex_leaf(e.type_of(id)) {
        let value = e.lower_node(id)?;
        split(e, &value)
    } else if resolved != id {
        parts(e, resolved, memo)?
    } else {
        match e.node(id).clone() {
            Node::Const(s) if e.resolve(s) == "im" => (e.scalar(0.0), e.scalar(1.0)),
            Node::Call(c) => {
                if let CallHead::User(callee) = c.head {
                    if let Some(result) = user_parts(e, id, callee, &c.args, false) {
                        return result;
                    }
                    let value = e.lower_node(id)?;
                    return Ok(split(e, &value));
                }
                let CallHead::Builtin(head) = c.head else {
                    return Err(EmitError::at(
                        id,
                        "complex expression requires a lowered builtin",
                    ));
                };
                let head = e.resolve(head).to_owned();
                if matches!(head.as_str(), "get" | "get0")
                    && let Some((container, field)) = e.record_projection(&c.args)
                {
                    return field_parts(e, id, container, &field, memo);
                }
                if matches!(head.as_str(), "get" | "get0")
                    && let [container, selector] = c.args.as_ref()
                    && !matches!(e.node(*selector), Node::Lit(flatppl_core::Scalar::Str(_)))
                {
                    let (re, im) = parts(e, *container, memo)?;
                    let base = i64::from(head == "get");
                    return Ok((
                        crate::ops::get_outer(e, id, re, *selector, base)?,
                        crate::ops::get_outer(e, id, im, *selector, base)?,
                    ));
                }
                if head == "vector" {
                    let coordinates = c
                        .args
                        .iter()
                        .map(|&arg| parts(e, arg, memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    let re = coordinates
                        .iter()
                        .map(|(re, _)| re.clone())
                        .collect::<Vec<_>>();
                    let im = coordinates
                        .into_iter()
                        .map(|(_, im)| im)
                        .collect::<Vec<_>>();
                    return Ok((e.vector(&re), e.vector(&im)));
                }
                if matches!(head.as_str(), "rowstack" | "colstack")
                    && let [vectors] = c.args.as_ref()
                {
                    let (mut re, mut im) = parts(e, *vectors, memo)?;
                    if head == "colstack" {
                        let crate::mlir::MlirTy::Ranked(dims) = &re.ty else {
                            return Err(EmitError::at(id, "colstack requires vector cells"));
                        };
                        let rank = dims.len();
                        let mut permutation = (0..rank as u64).collect::<Vec<_>>();
                        permutation.swap(rank - 2, rank - 1);
                        re = e.transpose(&re, &permutation);
                        im = e.transpose(&im, &permutation);
                    }
                    return Ok((e.with_typed_axes(id, re), e.with_typed_axes(id, im)));
                }
                if head == "broadcast"
                    && let Some((callee, args)) = c.args.split_first()
                {
                    if let Some(result) = user_parts(e, id, *callee, args, true) {
                        return result;
                    }
                    if matches!(e.node(*callee), Node::Const(s) if e.resolve(*s) == "complex")
                        && let [re, im] = args
                    {
                        let re = e.lower_node(*re)?;
                        let im = e.lower_node(*im)?;
                        let re = e.convert(&re, ElemKind::Real);
                        let im = e.convert(&im, ElemKind::Real);
                        return Ok(e.broadcast_pair(&re, &im));
                    }
                }
                let scalar = matches!(e.type_of(id), Some(Type::Scalar(ScalarType::Complex)));
                match (head.as_str(), c.args.as_ref()) {
                    ("complex", [re, im]) => {
                        let re = e.lower_node(*re)?;
                        let im = e.lower_node(*im)?;
                        (
                            e.convert(&re, ElemKind::Real),
                            e.convert(&im, ElemKind::Real),
                        )
                    }
                    ("cis", [angle]) => {
                        let angle = e.lower_node(*angle)?;
                        (e.cos(&angle), e.sin(&angle))
                    }
                    ("add" | "sub" | "mul" | "divide" | "pow", [a, b])
                        if scalar && !matches!(e.type_of(*a), Some(Type::TVector { .. })) =>
                    {
                        let a = parts(e, *a, memo)?;
                        let b = parts(e, *b, memo)?;
                        let op = match head.as_str() {
                            "add" => "stablehlo.add",
                            "sub" => "stablehlo.subtract",
                            "mul" => "stablehlo.multiply",
                            "divide" => "stablehlo.divide",
                            _ => "stablehlo.power",
                        };
                        binary_parts(e, op, a, b)
                    }
                    (
                        "conj" | "neg" | "exp" | "log" | "sqrt" | "sin" | "cos" | "tan" | "sinh"
                        | "cosh" | "tanh" | "asin" | "acos" | "atan" | "asinh" | "acosh" | "atanh",
                        [arg],
                    ) => {
                        let (re, im) = parts(e, *arg, memo)?;
                        unary_parts(e, &head, re, im).expect("complex elementary function")
                    }
                    _ => {
                        let value = e.lower_node(id)?;
                        split(e, &value)
                    }
                }
            }
            _ => {
                let value = e.lower_node(id)?;
                split(e, &value)
            }
        }
    };
    memo.insert(id, result.clone());
    Ok(result)
}

/// Record fields commute with selecting rows of their table. Keep both real
/// coordinates through that projection instead of forcing a complex tensor ABI.
pub(crate) fn field_parts(
    e: &mut Emitter,
    id: NodeId,
    container: NodeId,
    field: &str,
    memo: &mut HashMap<NodeId, (Value, Value)>,
) -> Result<(Value, Value), EmitError> {
    let container = e.resolve_tuple_projections(container);
    if let Some(entry) = e.literal_entry(container, field) {
        return parts(e, entry, memo);
    }
    if let Some(Lowered::Tensor(value)) = e.bound_field(container, field) {
        return Ok(split(e, &value));
    }
    if let Node::Call(call) = e.node(container).clone()
        && let CallHead::Builtin(head) = call.head
        && matches!(e.resolve(head), "get" | "get0")
        && let [inner, selector] = call.args.as_ref()
    {
        if let Node::Lit(flatppl_core::Scalar::Str(name)) = e.node(*selector) {
            let name = name.to_string();
            if let Some(entry) = e.literal_entry(*inner, &name) {
                return field_parts(e, id, entry, field, memo);
            }
        } else {
            let (re, im) = field_parts(e, id, *inner, field, memo)?;
            let base = i64::from(e.resolve(head) == "get");
            return Ok((
                crate::ops::get_outer(e, id, re, *selector, base)?,
                crate::ops::get_outer(e, id, im, *selector, base)?,
            ));
        }
    }
    if let Some(result) = user_field_parts(e, id, container, field) {
        return result;
    }
    let Lowered::Tensor(value) = e.lower_field(container, field)? else {
        return Err(EmitError::at(id, "expected a complex tensor field"));
    };
    Ok(split(e, &value))
}
