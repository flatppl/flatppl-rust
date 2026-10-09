//! Batch the scalar program's typed primitive recipes, independent of FlatPPL
//! function names. Original physical axes are cells; only host axes are batches.

use super::*;

struct TensorBatch<'a, 'm> {
    source: &'a Emitter<'m>,
    out: Emitter<'m>,
    literals: HashMap<&'a str, &'a str>,
    values: HashMap<String, Value>,
    visiting: HashSet<String>,
}

impl Emitter<'_> {
    /// Lift supported primitive graphs, retaining the existing fallback for
    /// regions, RNG state, or operations without a typed recipe.
    pub fn tensorize_batched(
        &self,
        func_name: &str,
        args: &[(String, MlirTy, ElemKind)],
        rets: &[&Value],
        batch: &crate::BatchSpec,
    ) -> Option<String> {
        if batch.shape.is_empty()
            || self.cur_key.is_some()
            || args.len() != batch.input_axes.len()
            || args.iter().any(|(_, ty, _)| matches!(ty, MlirTy::Key))
        {
            return None;
        }
        let mut replay = TensorBatch {
            source: self,
            out: self.scratch_emitter(),
            // Exact constant RHS text also preserves infinities and NaNs,
            // which the finite-only constant-folding metadata omits.
            literals: self
                .pure_ops
                .iter()
                .filter(|((rhs, _), _)| rhs.starts_with("stablehlo.constant "))
                .map(|((rhs, _), ssa)| (ssa.as_str(), rhs.as_str()))
                .collect(),
            values: HashMap::new(),
            visiting: HashSet::new(),
        };
        let mut physical = Vec::with_capacity(args.len());
        for ((ssa, ty, elem), axes) in args.iter().zip(&batch.input_axes) {
            let value = Value {
                ssa: ssa.clone(),
                ty: ty.clone(),
                elem: *elem,
            };
            let (input, internal) = replay.out.batch_input(&value, batch, axes).ok()?;
            physical.push((input.ssa, input.ty, input.elem));
            replay.values.insert(ssa.clone(), internal);
        }
        let outputs = rets
            .iter()
            .map(|value| {
                let value = replay.value(value)?;
                Some(replay.out.batch_output(&value, batch))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(
            replay
                .out
                .finish(func_name, &physical, &outputs.iter().collect::<Vec<_>>()),
        )
    }
}

impl TensorBatch<'_, '_> {
    fn value(&mut self, value: &Value) -> Option<Value> {
        if let Some(mapped) = self.values.get(&value.ssa) {
            return Some(mapped.clone());
        }
        if !self.visiting.insert(value.ssa.clone()) {
            return None;
        }
        let result = if let Some(rhs) = self.literals.get(value.ssa.as_str()) {
            if !rhs.ends_with(&format!(
                " : {}",
                value.ty.render(self.out.dtype, value.elem)
            )) {
                return None;
            }
            let ssa = self.out.pure((*rhs).to_owned());
            if let Some(data) = self.source.constants.get(&value.ssa) {
                self.out.constants.insert(ssa.clone(), data.clone());
            }
            Value {
                ssa,
                ..value.clone()
            }
        } else {
            let op = self.source.pointwise.get(&value.ssa)?.op.clone();
            let inputs = op
                .inputs()
                .iter()
                .map(|input| self.value(input))
                .collect::<Option<Vec<_>>>()?;
            self.operation(value, &op, &inputs)?
        };
        let batch = self.out.batch_rank(&result);
        if shape(&result.ty).get(batch..)? != shape(&value.ty) || result.elem != value.elem {
            return None;
        }
        // Discard source callable layers. Those axes are already explicit in
        // the scalar primitive graph and must not become host batch axes.
        self.out.axes.insert(
            result.ssa.clone(),
            Axes {
                batch,
                layers: match shape(&value.ty).len() {
                    0 => vec![],
                    rank => vec![rank],
                },
            },
        );
        self.visiting.remove(&value.ssa);
        self.values.insert(value.ssa.clone(), result.clone());
        Some(result)
    }

    fn operation(&mut self, value: &Value, op: &Pointwise, inputs: &[Value]) -> Option<Value> {
        let a = &inputs[0];
        let batch = self.out.batch_rank(a);
        let prefix = &shape(&a.ty)[..batch];
        let target = || batching::tensor(prefix.iter().chain(shape(&value.ty)).copied().collect());
        Some(match op {
            Pointwise::Unary(op, _) => self.out.unary(op, a),
            Pointwise::ChloUnary(op, _) => self.out.chlo_unary(op, a),
            Pointwise::Binary(op, ..) => self.out.binary(op, a, &inputs[1]),
            Pointwise::Compare(dir, ..) => self.out.compare(dir, a, &inputs[1]),
            Pointwise::Select(..) => self.out.select(a, &inputs[1], &inputs[2]),
            Pointwise::Convert(_, elem) => self.out.convert(a, *elem),
            Pointwise::Reshape(_) => self.out.reshape(a, target()),
            Pointwise::Broadcast(_, dims) => {
                let dims = (0..batch as u64)
                    .chain(dims.iter().map(|d| d + batch as u64))
                    .collect::<Vec<_>>();
                let ty = target();
                self.out.expand_axes(
                    a,
                    &dims,
                    ty,
                    Axes {
                        batch,
                        layers: vec![shape(&value.ty).len()],
                    },
                )
            }
            Pointwise::Slice(_, starts, limits, strides) => {
                let starts = [vec![0; batch], starts.clone()].concat();
                let limits = [
                    prefix.iter().copied().collect::<Option<Vec<_>>>()?,
                    limits.clone(),
                ]
                .concat();
                let strides = [vec![1; batch], strides.clone()].concat();
                self.out.slice(a, &starts, &limits, &strides)
            }
            Pointwise::Transpose(_, perm) => {
                let perm = (0..batch as u64)
                    .chain(perm.iter().map(|d| d + batch as u64))
                    .collect::<Vec<_>>();
                let result = self.out.transpose(a, &perm);
                self.out
                    .axes
                    .insert(result.ssa.clone(), self.out.axes_of(a));
                result
            }
            Pointwise::Gather(_, axis, indices) => {
                let indices = self.out.folded_constant(
                    indices.iter().map(|&i| Scalar::Int(i as i64)).collect(),
                    MlirTy::Ranked(vec![Some(indices.len() as u64)]),
                    Axes::default(),
                )?;
                self.out.gather_axis(a, &indices, 0, *axis)
            }
            Pointwise::DynamicGather(_, _, dimensions) => {
                self.gather(value, a, &inputs[1], dimensions)?
            }
            Pointwise::Dot(_, _, left, right) => self.dot(value, a, &inputs[1], *left, *right),
            Pointwise::Reduce(_, axis, op, init) => {
                self.out.reduce_axis_lit(op, init, a, batch + axis)
            }
            Pointwise::Concat(_, axis) => self.concatenate(inputs, *axis)?,
        })
    }

    fn dot(&mut self, value: &Value, a: &Value, b: &Value, left: usize, right: usize) -> Value {
        let (a, b) = self.out.broadcast_batches(a, b);
        let batch = self.out.batch_rank(&a);
        let ty = batching::tensor(
            shape(&a.ty)[..batch]
                .iter()
                .chain(shape(&value.ty))
                .copied()
                .collect(),
        );
        let dimensions = (0..batch)
            .map(|axis| axis.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let left = left + batch;
        let right = right + batch;
        let ssa = self.out.pure_axes(format!(
            "stablehlo.dot_general {}, {}, batching_dims = [{dimensions}] x [{dimensions}], contracting_dims = [{left}] x [{right}], precision = [DEFAULT, DEFAULT] : ({}, {}) -> {}",
            a.ssa, b.ssa, a.ty.render(self.out.dtype, a.elem), b.ty.render(self.out.dtype, b.elem), ty.render(self.out.dtype, value.elem)
        ), Axes { batch, layers: vec![shape(&value.ty).len()] });
        Value {
            ssa,
            ty,
            elem: value.elem,
        }
    }

    fn gather(
        &mut self,
        value: &Value,
        operand: &Value,
        indices: &Value,
        dimensions: &pointwise::GatherDimensions,
    ) -> Option<Value> {
        let mapped_indices = self.out.batch_rank(indices) != 0;
        let (operand, indices) = if mapped_indices {
            self.out.broadcast_batches(operand, indices)
        } else {
            (operand.clone(), indices.clone())
        };
        let batch = self.out.batch_rank(&operand);
        let prefix = shape(&operand.ty)[..batch]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()?;
        let shifted = |dims: &[usize]| dims.iter().map(|d| d + batch).collect::<Vec<_>>();
        let mut offsets = shifted(&dimensions.offsets);
        let collapsed = shifted(&dimensions.collapsed);
        let mut operand_batches = shifted(&dimensions.operand_batches);
        let mut index_batches = dimensions.index_batches.clone();
        let index_map = shifted(&dimensions.index_map);
        let mut index_vector = dimensions.index_vector;
        let mut sizes = prefix.clone();
        if mapped_indices {
            operand_batches.splice(..0, 0..batch);
            index_batches = (0..batch).chain(shifted(&index_batches)).collect();
            index_vector += batch;
            sizes.iter_mut().for_each(|size| *size = (*size).min(1));
        } else {
            offsets.splice(..0, 0..batch);
        }
        sizes.extend_from_slice(&dimensions.slice_sizes);
        let list = |dims: &[usize]| {
            dims.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let offsets = list(&offsets);
        let collapsed = list(&collapsed);
        let operand_batches = list(&operand_batches);
        let index_batches = list(&index_batches);
        let index_map = list(&index_map);
        let sizes = sizes
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let ty = batching::tensor(
            prefix
                .into_iter()
                .map(Some)
                .chain(shape(&value.ty).iter().copied())
                .collect(),
        );
        let ssa = self.out.pure_axes(format!(
            "\"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<offset_dims = [{offsets}], collapsed_slice_dims = [{collapsed}], operand_batching_dims = [{operand_batches}], start_indices_batching_dims = [{index_batches}], start_index_map = [{index_map}], index_vector_dim = {index_vector}>, indices_are_sorted = false, slice_sizes = array<i64: {sizes}>}}> : ({}, {}) -> {}",
            operand.ssa, indices.ssa, operand.ty.render(self.out.dtype, operand.elem),
            indices.ty.render(self.out.dtype, indices.elem), ty.render(self.out.dtype, value.elem),
        ), Axes { batch, layers: vec![shape(&value.ty).len()] });
        Some(Value {
            ssa,
            ty,
            elem: value.elem,
        })
    }

    fn concatenate(&mut self, inputs: &[Value], axis: usize) -> Option<Value> {
        let mut target = inputs[0].clone();
        for value in &inputs[1..] {
            target = self.out.broadcast_batches(&target, value).0;
        }
        let inputs = inputs
            .iter()
            .map(|value| self.out.broadcast_batches(value, &target).0)
            .collect::<Vec<_>>();
        let axis = self.out.batch_rank(&target) + axis;
        let mut dims = shape(&target.ty).to_vec();
        dims[axis] = Some(
            inputs
                .iter()
                .try_fold(0u64, |n, v| n.checked_add(shape(&v.ty)[axis]?))?,
        );
        let ty = batching::tensor(dims);
        let names = inputs
            .iter()
            .map(|v| v.ssa.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = inputs
            .iter()
            .map(|v| v.ty.render(self.out.dtype, v.elem))
            .collect::<Vec<_>>()
            .join(", ");
        let ssa = self.out.pure_axes(
            format!(
                "stablehlo.concatenate {names}, dim = {axis} : ({types}) -> {}",
                ty.render(self.out.dtype, target.elem),
            ),
            self.out.axes_of(&target),
        );
        Some(Value {
            ssa,
            ty,
            elem: target.elem,
        })
    }
}
