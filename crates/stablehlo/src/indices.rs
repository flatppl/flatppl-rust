//! Shape-only index vectors. Multi-axis results stay separate ABI tensors.

use flatppl_core::{CallHead, Dim, Node, NodeId, Type};

use crate::{EmitError, Emitter, Value};

fn dimensions(e: &Emitter, id: NodeId) -> Result<Vec<u32>, EmitError> {
    let dims = match e.type_of(id) {
        Some(Type::Array { shape, .. }) => shape.to_vec(),
        Some(Type::TVector { len, .. } | Type::Table { nrows: len, .. }) => vec![*len],
        _ => vec![],
    };
    if dims.is_empty() || dims.contains(&Dim::Dynamic) {
        return Err(EmitError::at(
            id,
            "indicesof requires a statically-shaped array or table",
        ));
    }
    Ok(dims
        .into_iter()
        .map(|dim| {
            let Dim::Static(n) = dim else { unreachable!() };
            n
        })
        .collect())
}

pub(crate) fn lower(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
    zero_based: bool,
    axis: Option<usize>,
) -> Result<Value, EmitError> {
    let [source] = crate::ops::args_exact(id, args)?;
    let dims = dimensions(e, source)?;
    let dim = match axis {
        Some(axis) => dims.get(axis),
        None if dims.len() == 1 => dims.first(),
        _ => None,
    }
    .ok_or_else(|| {
        EmitError::at(
            id,
            "indicesof axis tuple requires a valid static projection",
        )
    })?;
    Ok(e.index_vector((*dim).into(), zero_based))
}

fn call(e: &Emitter, id: NodeId) -> Option<(Vec<NodeId>, bool)> {
    let Node::Call(c) = e.node(e.resolve_ref_one(id)) else {
        return None;
    };
    let CallHead::Builtin(op) = c.head else {
        return None;
    };
    let zero_based = match e.resolve(op) {
        "indicesof" => false,
        "indicesof0" => true,
        _ => return None,
    };
    Some((c.args.to_vec(), zero_based))
}

pub(crate) fn projection(
    e: &mut Emitter,
    id: NodeId,
    container: NodeId,
    axis: usize,
) -> Option<Result<Value, EmitError>> {
    if !matches!(e.type_of(container), Some(Type::Tuple(_))) {
        return None;
    }
    let (args, zero_based) = call(e, container)?;
    Some(lower(e, id, &args, zero_based, Some(axis)))
}

pub(crate) fn outputs(e: &mut Emitter, id: NodeId) -> Option<Result<Vec<Value>, EmitError>> {
    let Some(Type::Tuple(parts)) = e.type_of(id) else {
        return None;
    };
    let rank = parts.len();
    let (args, zero_based) = call(e, id)?;
    Some(
        (0..rank)
            .map(|axis| lower(e, id, &args, zero_based, Some(axis)))
            .collect(),
    )
}
