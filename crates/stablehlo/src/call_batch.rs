//! Host batch axes are independent of each FlatPPL call's tensor dimensions.

use crate::{Dtype, ElemKind, EmitError, MlirTy};

/// A Cartesian batch of calls. Results carry `shape` as a leading prefix.
/// Each input row lists the physical axis for each batch dimension, or `None`
/// for a shared argument. Rows follow the flattened tensor ABI.
#[derive(Clone, Debug, Default)]
pub struct BatchSpec {
    pub shape: Vec<u64>,
    pub input_axes: Vec<Vec<Option<usize>>>,
}

impl BatchSpec {
    /// Insert mapped batch dimensions without changing the cell dimensions.
    pub fn input_shape(&self, cell: &[u64], axes: &[Option<usize>]) -> Result<Vec<u64>, EmitError> {
        if axes.len() != self.shape.len() {
            return Err(EmitError::whole(
                "one input axis is required per batch dimension",
            ));
        }
        let mut shape = vec![None; cell.len() + axes.iter().flatten().count()];
        for (&extent, &axis) in self.shape.iter().zip(axes) {
            if let Some(axis) = axis {
                let slot = shape
                    .get_mut(axis)
                    .ok_or_else(|| EmitError::whole("batch input axis is out of range"))?;
                if slot.replace(extent).is_some() {
                    return Err(EmitError::whole("batch input axes must be distinct"));
                }
            }
        }
        let mut cells = cell.iter();
        Ok(shape
            .into_iter()
            .map(|n| n.unwrap_or_else(|| *cells.next().unwrap()))
            .collect())
    }

    pub(crate) fn wrapper(
        &self,
        name: &str,
        callee: &str,
        inputs: &[(String, MlirTy, ElemKind)],
        outputs: &[(&MlirTy, ElemKind)],
        dtype: Dtype,
    ) -> Result<String, EmitError> {
        if self.input_axes.len() != inputs.len() {
            return Err(EmitError::whole(
                "one batch axis row is required per ABI input",
            ));
        }
        let count = self
            .shape
            .iter()
            .try_fold(1_u64, |a, &b| a.checked_mul(b))
            .filter(|&n| n <= i32::MAX as u64)
            .ok_or_else(|| EmitError::whole("batch size must fit a signed 32-bit loop index"))?;
        let cells = inputs
            .iter()
            .map(|(_, ty, _)| static_shape(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let shapes = cells
            .iter()
            .zip(&self.input_axes)
            .map(|(cell, axes)| self.input_shape(cell, axes))
            .collect::<Result<Vec<_>, _>>()?;
        let input_types = inputs
            .iter()
            .zip(&shapes)
            .map(|((_, ty, elem), shape)| render(ty, *elem, shape, dtype))
            .collect::<Vec<_>>();
        let output_shapes = outputs
            .iter()
            .map(|(ty, _)| {
                let mut shape = self.shape.clone();
                shape.extend(static_shape(ty)?);
                Ok(shape)
            })
            .collect::<Result<Vec<_>, EmitError>>()?;
        let output_types = outputs
            .iter()
            .zip(&output_shapes)
            .map(|((ty, elem), shape)| render(ty, *elem, shape, dtype))
            .collect::<Vec<_>>();
        let signature = inputs
            .iter()
            .zip(&input_types)
            .map(|((name, _, _), ty)| format!("{name}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let results = output_types.join(", ");
        let mut body = format!("  func.func @{name}({signature}) -> ({results}) {{\n");
        for (i, ((ty, elem), rendered)) in outputs.iter().zip(&output_types).enumerate() {
            let zero = if matches!(ty, MlirTy::Key) || *elem != ElemKind::Real {
                "0"
            } else {
                "0.0"
            };
            body.push_str(&format!(
                "    %out{i} = stablehlo.constant dense<{zero}> : {rendered}\n"
            ));
        }
        // Even an unreachable size-one slice of an empty dimension is invalid HLO.
        if count == 0 {
            let values = (0..outputs.len())
                .map(|i| format!("%out{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            body.push_str(&format!("    return {values} : {results}\n  }}\n"));
            return Ok(body);
        }
        body.push_str("    %zero = stablehlo.constant dense<0> : tensor<i32>\n    %one = stablehlo.constant dense<1> : tensor<i32>\n");
        body.push_str(&format!(
            "    %count = stablehlo.constant dense<{count}> : tensor<i32>\n"
        ));
        let carries = (0..outputs.len())
            .map(|i| format!("%b{i} = %out{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let loop_types = format!("tensor<i32>, {results}");
        body.push_str(&format!("    %loop:{} = stablehlo.while(%i = %zero, {carries}) : {loop_types}\n    cond {{\n      %more = stablehlo.compare LT, %i, %count, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>\n      stablehlo.return %more : tensor<i1>\n    }} do {{\n", outputs.len() + 1));
        let mut stride = count;
        for (axis, &extent) in self.shape.iter().enumerate() {
            stride /= extent;
            body.push_str(&format!("      %s{axis} = stablehlo.constant dense<{stride}> : tensor<i32>\n      %n{axis} = stablehlo.constant dense<{extent}> : tensor<i32>\n      %q{axis} = stablehlo.divide %i, %s{axis} : tensor<i32>\n      %x{axis} = stablehlo.remainder %q{axis}, %n{axis} : tensor<i32>\n"));
        }
        let mut call_args = Vec::new();
        for (i, ((name, ty, elem), axes)) in inputs.iter().zip(&self.input_axes).enumerate() {
            if axes.iter().all(Option::is_none) {
                call_args.push(name.clone());
                continue;
            }
            let mut starts = vec!["%zero".to_owned(); shapes[i].len()];
            let mut sizes = shapes[i].clone();
            for (axis, physical) in axes.iter().enumerate() {
                if let Some(physical) = physical {
                    starts[*physical] = format!("%x{axis}");
                    sizes[*physical] = 1;
                }
            }
            let sliced = render(ty, *elem, &sizes, dtype);
            let sizes = sizes
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let indices = vec!["tensor<i32>"; starts.len()].join(", ");
            body.push_str(&format!("      %slice{i} = stablehlo.dynamic_slice {name}, {}, sizes = [{sizes}] : ({}, {indices}) -> {sliced}\n      %cell_arg{i} = stablehlo.reshape %slice{i} : ({sliced}) -> {}\n", starts.join(", "), input_types[i], ty.render(dtype, *elem)));
            call_args.push(format!("%cell_arg{i}"));
        }
        let scalar_inputs = inputs
            .iter()
            .map(|(_, ty, elem)| ty.render(dtype, *elem))
            .collect::<Vec<_>>()
            .join(", ");
        let scalar_outputs = outputs
            .iter()
            .map(|(ty, elem)| ty.render(dtype, *elem))
            .collect::<Vec<_>>()
            .join(", ");
        body.push_str(&format!(
            "      %cell:{} = func.call @{callee}({}) : ({scalar_inputs}) -> ({scalar_outputs})\n",
            outputs.len(),
            call_args.join(", ")
        ));
        for (i, ((ty, elem), shape)) in outputs.iter().zip(&output_shapes).enumerate() {
            let mut sizes = vec![1; self.shape.len()];
            sizes.extend(static_shape(ty)?);
            let update = render(ty, *elem, &sizes, dtype);
            let mut starts = (0..self.shape.len())
                .map(|axis| format!("%x{axis}"))
                .collect::<Vec<_>>();
            starts.resize(shape.len(), "%zero".into());
            let indices = vec!["tensor<i32>"; shape.len()].join(", ");
            body.push_str(&format!("      %update{i} = stablehlo.reshape %cell#{i} : ({}) -> {update}\n      %next{i} = stablehlo.dynamic_update_slice %b{i}, %update{i}, {} : ({}, {update}, {indices}) -> {}\n", ty.render(dtype, *elem), starts.join(", "), output_types[i], output_types[i]));
        }
        let next = (0..outputs.len())
            .map(|i| format!("%next{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let returned = (1..=outputs.len())
            .map(|i| format!("%loop#{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        body.push_str(&format!("      %inc = stablehlo.add %i, %one : tensor<i32>\n      stablehlo.return %inc, {next} : {loop_types}\n    }}\n    return {returned} : {results}\n  }}\n"));
        Ok(body)
    }
}

fn static_shape(ty: &MlirTy) -> Result<Vec<u64>, EmitError> {
    match ty {
        MlirTy::Scalar => Ok(vec![]),
        MlirTy::Key => Ok(vec![2]),
        MlirTy::Ranked(dims) => dims
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| EmitError::whole("batching requires static tensor shapes")),
        MlirTy::Tuple(_) => Err(EmitError::whole(
            "batching requires flattened tensor arguments",
        )),
    }
}

fn render(ty: &MlirTy, elem: ElemKind, shape: &[u64], dtype: Dtype) -> String {
    if matches!(ty, MlirTy::Key) {
        let dims = shape.iter().map(|n| format!("{n}x")).collect::<String>();
        format!("tensor<{dims}ui64>")
    } else {
        MlirTy::Ranked(shape.iter().copied().map(Some).collect()).render(dtype, elem)
    }
}
