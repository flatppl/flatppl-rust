//! Lowerings selected by the Enzyme compatibility option. Enzyme rewrites sliced
//! multiply trees into product reductions whose adjoints divide by each factor.
//! Extent-one tensor dots preserve the product rule without that rewrite.

use super::*;

impl Emitter<'_> {
    /// Select every `stride`th cell without Enzyme's faulty strided-slice adjoint.
    /// Reshape complete stride blocks; at most one selected tail remains.
    pub(super) fn enzyme_strided_axis(&mut self, input: &Value, axis: usize, stride: u64) -> Value {
        let mut dims = shape(&input.ty)
            .iter()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>();
        let length = dims[axis];
        if length == 0 {
            return input.clone();
        }
        let count = length / stride;
        let starts = vec![0; dims.len()];
        let ones = vec![1; dims.len()];
        if count == 0 {
            dims[axis] = 1;
            return self.slice(input, &starts, &dims, &ones);
        }
        dims[axis] = count * stride;
        let prefix = self.slice(input, &starts, &dims, &ones);
        dims[axis] = count;
        dims.insert(axis + 1, stride);
        let grouped = self.reshape(
            &prefix,
            MlirTy::Ranked(dims.iter().copied().map(Some).collect()),
        );
        dims[axis + 1] = 1;
        let selected = self.slice(&grouped, &vec![0; dims.len()], &dims, &vec![1; dims.len()]);
        dims.remove(axis + 1);
        let mut output = self.reshape_axes(
            &selected,
            MlirTy::Ranked(dims.iter().copied().map(Some).collect()),
            self.axes_of(input),
        );
        if !length.is_multiple_of(stride) {
            let mut tail_starts = starts;
            tail_starts[axis] = count * stride;
            let mut limits = shape(&input.ty)
                .iter()
                .map(|d| d.unwrap())
                .collect::<Vec<_>>();
            limits[axis] = tail_starts[axis] + 1;
            let tail = self.slice(input, &tail_starts, &limits, &ones);
            dims[axis] += 1;
            let ty = MlirTy::Ranked(dims.into_iter().map(Some).collect());
            let ssa = self.pure_like(
                format!(
                    "stablehlo.concatenate {}, {}, dim = {axis} : ({}, {}) -> {}",
                    output.ssa,
                    tail.ssa,
                    output.ty.render(self.dtype, input.elem),
                    tail.ty.render(self.dtype, input.elem),
                    ty.render(self.dtype, input.elem),
                ),
                input,
            );
            output = Value {
                ssa,
                ty,
                elem: input.elem,
            };
        }
        output
    }

    /// Pair factors in logarithmic depth, keeping every other axis tensorized.
    /// Empty axes retain the ordinary identity. Public mode rejects dynamic shapes.
    pub(super) fn product_tree(
        &mut self,
        input: &Value,
        axis: usize,
        result_ty: &MlirTy,
        result_axes: &Axes,
    ) -> Option<Value> {
        let mut dims = shape(&input.ty)
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()?;
        if dims[axis] == 0 {
            return None;
        }
        let mut current = input.clone();
        let strides = vec![1; dims.len()];
        while dims[axis] > 1 {
            let half = dims[axis] / 2;
            let mut starts = vec![0; dims.len()];
            let mut limits = dims.clone();
            limits[axis] = 2 * half;
            let prefix = self.slice(&current, &starts, &limits, &strides);
            let mut paired_dims = dims.clone();
            paired_dims[axis] = half;
            paired_dims.insert(axis + 1, 2);
            let paired = self.reshape(
                &prefix,
                MlirTy::Ranked(paired_dims.iter().copied().map(Some).collect()),
            );
            let mut pair_starts = vec![0; paired_dims.len()];
            let pair_strides = vec![1; paired_dims.len()];
            paired_dims[axis + 1] = 1;
            let left = self.slice(&paired, &pair_starts, &paired_dims, &pair_strides);
            pair_starts[axis + 1] = 1;
            paired_dims[axis + 1] = 2;
            let right = self.slice(&paired, &pair_starts, &paired_dims, &pair_strides);
            let product = self.product_pair(&left, &right);
            paired_dims.remove(axis + 1);
            let mut product = self.reshape(
                &product,
                MlirTy::Ranked(paired_dims.into_iter().map(Some).collect()),
            );
            if dims[axis] % 2 != 0 {
                starts[axis] = 2 * half;
                limits[axis] = dims[axis];
                let tail = self.slice(&current, &starts, &limits, &strides);
                let mut next_dims = dims.clone();
                next_dims[axis] = half + 1;
                let ty = MlirTy::Ranked(next_dims.into_iter().map(Some).collect());
                let lhs_ty = product.ty.render(self.dtype, product.elem);
                let rhs_ty = tail.ty.render(self.dtype, tail.elem);
                let output_ty = ty.render(self.dtype, product.elem);
                let ssa = self.pure_like(
                    format!(
                        "stablehlo.concatenate {}, {}, dim = {axis} : ({lhs_ty}, {rhs_ty}) -> {output_ty}",
                        product.ssa, tail.ssa
                    ),
                    &product,
                );
                product = Value {
                    ssa,
                    ty,
                    elem: input.elem,
                };
            }
            dims[axis] = dims[axis].div_ceil(2);
            current = product;
        }
        Some(self.reshape_axes(&current, result_ty.clone(), result_axes.clone()))
    }

    /// A one-element contraction is multiplication, with no quotient in its
    /// adjoint. HIGHEST avoids dot's default reduced-precision accelerator path.
    pub(super) fn product_pair(&mut self, left: &Value, right: &Value) -> Value {
        let mut dims = shape(&left.ty).to_vec();
        let axis = dims.len();
        let batch = (0..axis)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        dims.push(Some(1));
        let dot_ty = MlirTy::Ranked(dims);
        let lhs = self.reshape(left, dot_ty.clone());
        let rhs = self.reshape(right, dot_ty.clone());
        let operand_ty = dot_ty.render(self.dtype, left.elem);
        let output_ty = left.ty.render(self.dtype, left.elem);
        let ssa = self.pure_like(
            format!(
                "stablehlo.dot_general {}, {}, batching_dims = [{batch}] x [{batch}], contracting_dims = [{axis}] x [{axis}], precision = [HIGHEST, HIGHEST] : ({operand_ty}, {operand_ty}) -> {output_ty}",
                lhs.ssa, rhs.ssa
            ),
            left,
        );
        Value {
            ssa,
            ty: left.ty.clone(),
            elem: left.elem,
        }
    }
}
