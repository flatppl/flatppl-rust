//! Real projections of scalar complex expressions, lowered as pairs of real tensors.

use std::collections::HashMap;

use crate::mlir::Lowered;
use crate::{EmitError, Emitter, Value};
use flatppl_core::{CallHead, Node, NodeId, ScalarType, Type};

pub(crate) fn complex_leaf(ty: Option<&Type>) -> bool {
    match ty {
        Some(Type::Scalar(ScalarType::Complex)) => true,
        Some(Type::Array { elem, .. } | Type::TVector { elem, .. }) => complex_leaf(Some(elem)),
        _ => false,
    }
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
        "abs" => {
            let ar = e.abs(&re);
            let ai = e.abs(&im);
            let scale = e.max(&ar, &ai);
            let zero = e.scalar(0.0);
            let one = e.scalar(1.0);
            let nonzero = e.compare("GT", &scale, &zero);
            let safe = e.select(&nonzero, &scale, &one);
            let re = e.div(&re, &safe);
            let im = e.div(&im, &safe);
            let re2 = e.mul(&re, &re);
            let im2 = e.mul(&im, &im);
            let norm = e.add(&re2, &im2);
            let norm = e.sqrt(&norm);
            let norm = e.mul(&scale, &norm);
            let inf = e.inf_like(&scale);
            let infinite = e.compare("EQ", &scale, &inf);
            Ok(e.select(&infinite, &scale, &norm))
        }
        "abs2" => {
            let re2 = e.mul(&re, &re);
            let im2 = e.mul(&im, &im);
            Ok(e.add(&re2, &im2))
        }
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
    let result = if !complex_leaf(e.type_of(id)) {
        (e.lower_node(id)?, e.scalar(0.0))
    } else if resolved != id {
        parts(e, resolved, memo)?
    } else {
        match e.node(id).clone() {
            Node::Const(s) if e.resolve(s) == "im" => (e.scalar(0.0), e.scalar(1.0)),
            Node::Call(c) => {
                if let CallHead::User(callee) = c.head {
                    return e
                        .lower_kronecker_parts(id, callee, &c.args, false)
                        .or_else(|| e.lower_matrix_exp_parts(id, callee, &c.args, false))
                        .or_else(|| e.lower_bessel_parts(id, callee, &c.args, false))
                        .or_else(|| e.lower_lstsq_parts(id, callee, &c.args, false))
                        .or_else(|| e.lower_ext_linalg_parts(id, callee, &c.args, false))
                        .unwrap_or_else(|| {
                            Err(EmitError::at(
                                id,
                                "complex expression requires a supported callable",
                            ))
                        });
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
                    if let Some(result) = e.lower_kronecker_parts(id, *callee, args, true) {
                        return result;
                    }
                    if let Some(result) = e.lower_matrix_exp_parts(id, *callee, args, true) {
                        return result;
                    }
                    if matches!(e.node(*callee), Node::Const(s) if e.resolve(*s) == "complex")
                        && let [re, im] = args
                    {
                        let re = e.lower_node(*re)?;
                        let im = e.lower_node(*im)?;
                        return Ok(e.broadcast_pair(&re, &im));
                    }
                }
                if head == "broadcast"
                    && let Some((callee, args)) = c.args.split_first()
                    && let Some(result) = e
                        .lower_bessel_parts(id, *callee, args, true)
                        .or_else(|| e.lower_lstsq_parts(id, *callee, args, true))
                        .or_else(|| e.lower_ext_linalg_parts(id, *callee, args, true))
                {
                    return result;
                }
                if !matches!(e.type_of(id), Some(Type::Scalar(ScalarType::Complex))) {
                    return Err(EmitError::at(
                        id,
                        "complex array projection requires a supported tensor operation",
                    ));
                }
                match (head.as_str(), c.args.as_ref()) {
                    ("complex", [re, im]) => (e.lower_node(*re)?, e.lower_node(*im)?),
                    ("cis", [angle]) => {
                        let angle = e.lower_node(*angle)?;
                        (e.cos(&angle), e.sin(&angle))
                    }
                    // §07 `exp` over `complexes`: e^(a + ib) = e^a (cos b + i sin b).
                    ("exp", [arg]) => {
                        let (re, im) = parts(e, *arg, memo)?;
                        let magnitude = e.exp(&re);
                        let cos = e.cos(&im);
                        let sin = e.sin(&im);
                        (e.mul(&magnitude, &cos), e.mul(&magnitude, &sin))
                    }
                    ("neg" | "conj", [arg]) => {
                        let (re, im) = parts(e, *arg, memo)?;
                        (if head == "neg" { e.neg(&re) } else { re }, e.neg(&im))
                    }
                    ("add" | "sub" | "mul" | "divide", [a, b]) => {
                        let (ar, ai) = parts(e, *a, memo)?;
                        let (br, bi) = parts(e, *b, memo)?;
                        match head.as_str() {
                            "add" => (e.add(&ar, &br), e.add(&ai, &bi)),
                            "sub" => (e.sub(&ar, &br), e.sub(&ai, &bi)),
                            "mul" => {
                                let rr = e.mul(&ar, &br);
                                let ii = e.mul(&ai, &bi);
                                let ri = e.mul(&ar, &bi);
                                let ir = e.mul(&ai, &br);
                                (e.sub(&rr, &ii), e.add(&ri, &ir))
                            }
                            _ => {
                                let abr = e.abs(&br);
                                let abi = e.abs(&bi);
                                let scale = e.max(&abr, &abi);
                                let ar = e.div(&ar, &scale);
                                let ai = e.div(&ai, &scale);
                                let br = e.div(&br, &scale);
                                let bi = e.div(&bi, &scale);
                                let rr = e.mul(&br, &br);
                                let ii = e.mul(&bi, &bi);
                                let denom = e.add(&rr, &ii);
                                let rr = e.mul(&ar, &br);
                                let ii = e.mul(&ai, &bi);
                                let ir = e.mul(&ai, &br);
                                let ri = e.mul(&ar, &bi);
                                let re = e.add(&rr, &ii);
                                let im = e.sub(&ir, &ri);
                                (e.div(&re, &denom), e.div(&im, &denom))
                            }
                        }
                    }
                    _ => {
                        return Err(EmitError::at(
                            id,
                            format!("complex '{head}' has no real-pair lowering"),
                        ));
                    }
                }
            }
            _ => {
                return Err(EmitError::at(
                    id,
                    "complex inputs and outputs require a complex tensor ABI",
                ));
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
        let zero = e.constant_like(0.0, &value);
        return Ok((value, zero));
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
    e.lower_eigh_field_parts(container, field)
        .or_else(|| e.lower_eigen_field_parts(container, field))
        .or_else(|| e.lower_lu_field_parts(container, field))
        .or_else(|| e.lower_svd_field_parts(id, container, field))
        .unwrap_or_else(|| {
            Err(EmitError::at(
                id,
                format!("complex field `{field}` has no pair lowering"),
            ))
        })
}
