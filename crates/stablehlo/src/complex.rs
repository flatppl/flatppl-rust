//! Real projections of scalar complex expressions, lowered as pairs of real tensors.

use std::collections::HashMap;

use crate::{EmitError, Emitter, Value};
use flatppl_core::{CallHead, Node, NodeId, ScalarType, Type};

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

fn parts(
    e: &mut Emitter,
    id: NodeId,
    memo: &mut HashMap<NodeId, (Value, Value)>,
) -> Result<(Value, Value), EmitError> {
    if let Some(result) = memo.get(&id) {
        return Ok(result.clone());
    }
    let resolved = e.resolve_ref_one(id);
    let result = if !matches!(e.type_of(id), Some(Type::Scalar(ScalarType::Complex))) {
        (e.lower_node(id)?, e.scalar(0.0))
    } else if resolved != id {
        parts(e, resolved, memo)?
    } else {
        match e.node(id).clone() {
            Node::Const(s) if e.resolve(s) == "im" => (e.scalar(0.0), e.scalar(1.0)),
            Node::Call(c) => {
                let CallHead::Builtin(head) = c.head else {
                    return Err(EmitError::at(
                        id,
                        "complex expression requires a lowered builtin",
                    ));
                };
                let head = e.resolve(head).to_owned();
                match (head.as_str(), c.args.as_ref()) {
                    ("complex", [re, im]) => (e.lower_node(*re)?, e.lower_node(*im)?),
                    ("cis", [angle]) => {
                        let angle = e.lower_node(*angle)?;
                        (e.cos(&angle), e.sin(&angle))
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
