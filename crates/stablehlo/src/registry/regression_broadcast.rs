//! Outer collection sampling of vector-valued regression laws. The existing
//! broadcast frame identifies cells; each cell keeps its own threaded RNG draw.

use super::*;

pub(super) fn supports(name: &str) -> bool {
    matches!(
        name,
        "BernoulliLogitGLM"
            | "BinomialLogitGLM"
            | "CategoricalLogitGLM"
            | "NormalGLM"
            | "PoissonLogGLM"
            | "NegativeBinomial2LogGLM"
            | "StudentTGLM"
            | "GammaLogGLM"
            | "OrderedLogitGLM"
            | "BetaLogitGLM"
    )
}

pub(super) fn sample(
    e: &mut Emitter,
    id: NodeId,
    rng: NodeId,
    ctor: &str,
    kernel_input: NodeId,
) -> Result<Value, EmitError> {
    let dist = lookup(ctor).ok_or_else(|| EmitError::at(id, "unknown regression constructor"))?;
    let draw = dist
        .sample
        .ok_or_else(|| EmitError::at(id, "regression sampler is unavailable"))?;
    let Broadcast::PerCell(names) = dist.broadcast else {
        return Err(EmitError::at(
            id,
            "regression sampling requires per-cell parameters",
        ));
    };
    let params = Params {
        kernel_input,
        variate: None,
        sample_shape: None,
        fields: None,
    };
    let mut fields = Vec::with_capacity(names.len());
    let mut iterated = Vec::new();
    for &name in names {
        let field = e.resolve_ref_one(params.field_id(e, name)?);
        let cell = unbatch_field_id(e, &params, name)?;
        if cell == field {
            iterated.push(field);
        }
        fields.push((name.to_owned(), cell));
    }
    let cell_params = Params {
        kernel_input,
        variate: None,
        sample_shape: None,
        fields: Some(fields.clone()),
    };
    check_parameters(e, ctor, &cell_params)?;
    let key = e.lower_node(rng)?;
    e.set_cur_key(key);
    let output = e.lower_broadcast_cells(id, &iterated, |e, _| {
        let originals = fields
            .iter()
            .map(|(_, field)| e.lower_node(*field).map(|value| (*field, value)))
            .collect::<Result<Vec<_>, _>>()?;
        let outer = originals
            .iter()
            .map(|(_, value)| e.batch_rank(value))
            .max()
            .unwrap_or(0);
        let mut shape = vec![1_u64; outer];
        for (_, value) in &originals {
            let rank = e.batch_rank(value);
            let dims = tensor_shape(id, &value.ty)?;
            for axis in 0..rank {
                if dims[axis] != 1 {
                    shape[axis] = dims[axis];
                }
            }
        }
        let count = shape
            .iter()
            .try_fold(1_u64, |n, &d| n.checked_mul(d))
            .ok_or_else(|| EmitError::at(id, "regression broadcast size overflows"))?;
        let batch = e.batch_shape();
        e.clear_batch_shape();
        let result = (|| {
            if count == 0 {
                let x_index = fields.iter().position(|(name, _)| name == "x").unwrap();
                let x = &originals[x_index].1;
                let xd = tensor_shape(id, &e.cell_ty(x))?;
                if xd.len() != 2 {
                    return Err(EmitError::at(id, "regression x must be a matrix cell"));
                }
                let mut dimensions = shape.iter().copied().map(Some).collect::<Vec<_>>();
                dimensions.push(Some(xd[0]));
                let zero = e.scalar(0.0);
                let empty = e.broadcast_in_dim(&zero, &[], MlirTy::Ranked(dimensions));
                return Ok(e.axes_view(
                    &empty,
                    Axes {
                        batch: outer,
                        layers: vec![1],
                    },
                ));
            }
            let mut cells = Vec::with_capacity(count as usize);
            for index in 0..count {
                let mut coordinate = vec![0_u64; outer];
                let mut index = index;
                for axis in (0..outer).rev() {
                    coordinate[axis] = index % shape[axis];
                    index /= shape[axis];
                }
                for (field, value) in &originals {
                    let rank = e.batch_rank(value);
                    let cell = take_cell(e, id, value, &coordinate[..rank])?;
                    e.bind(*field, cell);
                }
                cells.push(draw(e, &cell_params)?);
            }
            let cell_shape = tensor_shape(id, &cells[0].ty)?;
            let stacked = e.vector(&cells);
            let mut dimensions = shape.into_iter().map(Some).collect::<Vec<_>>();
            dimensions.extend(cell_shape.into_iter().map(Some));
            let stacked = e.reshape(&stacked, MlirTy::Ranked(dimensions));
            Ok(e.axes_view(
                &stacked,
                Axes {
                    batch: outer,
                    layers: vec![1],
                },
            ))
        })();
        for (field, value) in originals {
            e.bind(field, value);
        }
        match batch {
            Some(shape) => e.set_batch_shape(shape),
            None => e.clear_batch_shape(),
        }
        result
    })?;
    let output = if let Some(Type::Tuple(parts)) = e.type_of(id)
        && let Some(variate) = parts.first()
        && let Ok((_, kind)) = crate::types::mlir_type_of_ty(id, variate, e.dtype())
    {
        e.convert(&output, kind)
    } else {
        output
    };
    let advanced = e.cur_key();
    e.record_sample_key(id, advanced);
    Ok(output)
}

fn tensor_shape(id: NodeId, ty: &MlirTy) -> Result<Vec<u64>, EmitError> {
    match ty {
        MlirTy::Scalar => Ok(Vec::new()),
        MlirTy::Ranked(dims) => dims
            .iter()
            .map(|d| {
                d.ok_or_else(|| EmitError::at(id, "regression broadcast dimensions must be static"))
            })
            .collect(),
        _ => Err(EmitError::at(
            id,
            "regression broadcast parameters must be tensors",
        )),
    }
}

fn take_cell(
    e: &mut Emitter,
    id: NodeId,
    value: &Value,
    coordinate: &[u64],
) -> Result<Value, EmitError> {
    if coordinate.is_empty() {
        return Ok(value.clone());
    }
    let dims = tensor_shape(id, &value.ty)?;
    let mut starts = vec![0; dims.len()];
    let mut limits = dims.clone();
    for (axis, &index) in coordinate.iter().enumerate() {
        starts[axis] = if dims[axis] == 1 { 0 } else { index };
        limits[axis] = starts[axis] + 1;
    }
    let slice = e.slice(value, &starts, &limits, &vec![1; dims.len()]);
    // Drop the selected batch prefix before reshaping to the original event.
    let slice = e.axes_view(
        &slice,
        Axes {
            batch: 0,
            layers: vec![dims.len()],
        },
    );
    let dimensions = dims[coordinate.len()..]
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    Ok(e.reshape(
        &slice,
        if dimensions.is_empty() {
            MlirTy::Scalar
        } else {
            MlirTy::Ranked(dimensions)
        },
    ))
}
