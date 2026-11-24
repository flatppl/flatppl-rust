//! Compensated f32 contractions using only portable floating-point operations.

use super::*;

// The complete f32 significand rounded up to a power of two for the pair tree.
const TILE: u64 = (f32::MANTISSA_DIGITS as u64).next_power_of_two();
const SPLITTER: f64 = ((1u64 << f32::MANTISSA_DIGITS.div_ceil(2)) + 1) as f64;

impl Emitter<'_> {
    pub(super) fn compensated_dot(
        &mut self,
        a: &Value,
        b: &Value,
        [left, right]: [usize; 2],
        batch: usize,
        result_ty: &MlirTy,
        result_axes: &Axes,
    ) -> Option<Value> {
        if !matches!(self.dtype, Dtype::F32) || a.elem != ElemKind::Real || b.elem != ElemKind::Real
        {
            return None;
        }
        let result = shape(result_ty);
        let groups = tile_count(shape(&a.ty), shape(&b.ty), result, [left, right], batch)?;
        // Fixed nonnegative vector weights commonly appear in repeated
        // reductions. Preserve their compact native matrix-vector lowering.
        if batch == 0
            && matches!((shape(&a.ty).len(), shape(&b.ty).len()), (1, 2) | (2, 1))
            && (self.fixed_nonnegative_weights(a) || self.fixed_nonnegative_weights(b))
        {
            return None;
        }
        let axis = shape(&a.ty).len() - 1;
        let mut product = result.to_vec();
        product.insert(axis, shape(&a.ty)[left]);
        let product = batching::tensor(product);
        let left_dims = (0..=axis as u64).collect::<Vec<_>>();
        let right_dims = (0..batch as u64)
            .chain(axis as u64..shape(&product).len() as u64)
            .collect::<Vec<_>>();
        let a = self.contraction_broadcast(a, left, axis, &left_dims, &product);
        let b = self.contraction_broadcast(b, right, batch, &right_dims, &product);
        let pair = self.product_with_error(&a, &b);

        let mut tiles = shape(&product).to_vec();
        tiles.splice(axis..axis + 1, [Some(groups), Some(TILE)]);
        let tiles = batching::tensor(tiles);
        let pair = pair.map(|value| {
            let value = self.pad_contraction(&value, axis, groups * TILE);
            self.reshape_axes(&value, tiles.clone(), Axes::default())
        });
        let pair = self.sum_pairs(pair, axis + 1, TILE);
        let [hi, lo] = self.sum_pairs(pair, axis, groups);
        let result = self.add(&hi, &lo);
        Some(self.axes_view(&result, result_axes.clone()))
    }

    fn fixed_nonnegative_weights(&self, value: &Value) -> bool {
        shape(&value.ty).len() == 1
            && self.fixed_operand(value).is_some_and(|cells| {
                cells
                    .iter()
                    .all(|cell| matches!(cell, Scalar::Real(x) if x.is_finite() && *x >= 0.0))
            })
    }

    fn contraction_broadcast(
        &mut self,
        value: &Value,
        from: usize,
        to: usize,
        dims: &[u64],
        ty: &MlirTy,
    ) -> Value {
        let mut value = value.clone();
        if from != to {
            let mut permutation = (0..shape(&value.ty).len() as u64).collect::<Vec<_>>();
            permutation.remove(from);
            permutation.insert(to, from as u64);
            value = self.transpose(&value, &permutation);
        }
        self.expand_axes(&value, dims, ty.clone(), Axes::default())
    }

    fn split_significand(&mut self, value: &Value) -> (Value, Value, Value) {
        let splitter = self.constant_like(SPLITTER, value);
        let shifted = self.mul(&splitter, value);
        let tail = self.sub(&shifted, value);
        let hi = self.sub(&shifted, &tail);
        let lo = self.sub(value, &hi);
        let finite = self.contraction_finite(&shifted);
        (hi, lo, finite)
    }

    fn product_with_error(&mut self, a: &Value, b: &Value) -> [Value; 2] {
        let product = self.mul(a, b);
        let (a_hi, a_lo, a_finite) = self.split_significand(a);
        let (b_hi, b_lo, b_finite) = self.split_significand(b);
        let (p_hi, p_lo, p_finite) = self.split_significand(&product);
        // Dekker's split keeps the product rounding error in the low part.
        // Splitting the rounded product also prevents a multiply-add from
        // erasing its rounding before the residual is measured.
        let hi = self.mul(&a_hi, &b_hi);
        let residual = self.sub(&hi, &p_hi);
        let cross = self.mul(&a_lo, &b_hi);
        let residual = self.add(&residual, &cross);
        let cross = self.mul(&a_hi, &b_lo);
        let residual = self.add(&residual, &cross);
        let lo = self.mul(&a_lo, &b_lo);
        let residual = self.add(&residual, &lo);
        let residual = self.sub(&residual, &p_lo);
        let finite_operands = self.and(&a_finite, &b_finite);
        let finite = self.and(&finite_operands, &p_finite);
        let zero = self.constant_like(0.0, &residual);
        [product, self.select(&finite, &residual, &zero)]
    }

    fn sum_pairs(&mut self, mut pair: [Value; 2], axis: usize, length: u64) -> [Value; 2] {
        let mut count = length.next_power_of_two();
        pair = pair.map(|value| self.pad_contraction(&value, axis, count));
        let mut dims = shape(&pair[0].ty).to_vec();
        while count > 1 {
            count /= 2;
            dims[axis] = Some(count);
            let ty = batching::tensor(dims.clone());
            let mut split_dims = dims.clone();
            split_dims.insert(axis + 1, Some(2));
            let split_ty = batching::tensor(split_dims);
            let limits = shape(&split_ty)
                .iter()
                .map(|dim| dim.expect("compensated contractions are static"))
                .collect::<Vec<_>>();
            let parts =
                pair.map(|value| self.reshape_axes(&value, split_ty.clone(), Axes::default()));
            let mut starts = vec![0; limits.len()];
            let mut ends = limits.clone();
            let strides = vec![1; starts.len()];
            ends[axis + 1] = 1;
            let lhs = parts.each_ref().map(|value| {
                let half = self.slice(value, &starts, &ends, &strides);
                self.reshape_axes(&half, ty.clone(), Axes::default())
            });
            starts[axis + 1] = 1;
            let rhs = parts.each_ref().map(|value| {
                let half = self.slice(value, &starts, &limits, &strides);
                self.reshape_axes(&half, ty.clone(), Axes::default())
            });
            pair = self.add_pairs(lhs, rhs);
        }
        dims.remove(axis);
        let ty = batching::tensor(dims);
        pair.map(|value| self.reshape_axes(&value, ty.clone(), Axes::default()))
    }

    fn add_pairs(&mut self, [a_hi, a_lo]: [Value; 2], [b_hi, b_lo]: [Value; 2]) -> [Value; 2] {
        // TwoSum keeps the roundoff of each addition. Only the high part
        // participates when the sum is nonfinite, as it would in a plain dot.
        let sum = self.add(&a_hi, &b_hi);
        let bv = self.sub(&sum, &a_hi);
        let av = self.sub(&sum, &bv);
        let da = self.sub(&a_hi, &av);
        let db = self.sub(&b_hi, &bv);
        let error = self.add(&da, &db);
        let low = self.add(&a_lo, &b_lo);
        let low = self.add(&low, &error);
        let finite = self.contraction_finite(&sum);
        let zero = self.constant_like(0.0, &sum);
        [sum, self.select(&finite, &low, &zero)]
    }

    fn contraction_finite(&mut self, value: &Value) -> Value {
        let ty = value.ty.render(self.dtype, ElemKind::Real);
        let out = value.ty.render(self.dtype, ElemKind::Bool);
        let ssa = self.pure_like(
            format!("stablehlo.is_finite {} : ({ty}) -> {out}", value.ssa),
            value,
        );
        Value {
            ssa,
            ty: value.ty.clone(),
            elem: ElemKind::Bool,
        }
    }

    fn pad_contraction(&mut self, value: &Value, axis: usize, size: u64) -> Value {
        let dims = shape(&value.ty);
        let length = dims[axis].expect("compensated contractions are static");
        if length == size {
            return value.clone();
        }
        let low = vec![0; dims.len()];
        let mut high = low.clone();
        high[axis] = size - length;
        self.pad_zeros(value, &low, &high)
    }
}

fn tile_count(
    a: &[Option<u64>],
    b: &[Option<u64>],
    result: &[Option<u64>],
    [left, right]: [usize; 2],
    batch: usize,
) -> Option<u64> {
    let length = a[left]?;
    if length <= TILE || b[right] != Some(length) {
        return None;
    }
    let groups = length.div_ceil(TILE);
    groups.checked_mul(TILE)?;

    fn cells(dims: &[Option<u64>]) -> Option<u128> {
        dims.iter().try_fold(1u128, |size, dim| {
            size.checked_mul(u128::from(dim.filter(|&n| n > 0)?))
        })
    }
    // Keep both partial sums within the input working set. Large square
    // matrix multiplies keep their native dot lowering.
    cells(a.get(..batch)?)?;
    let input = cells(a.get(batch..)?)?.checked_add(cells(b.get(batch..)?)?)?;
    let partials = cells(result.get(batch..)?)?.checked_mul(2 * u128::from(groups))?;
    (partials <= input).then_some(groups)
}

#[cfg(test)]
mod tests {
    use super::tile_count;

    #[test]
    fn empty_or_dynamic_contractions_use_the_native_path() {
        assert_eq!(
            tile_count(&[Some(2), Some(0)], &[Some(0)], &[Some(2)], [1, 0], 0),
            None
        );
        assert_eq!(
            tile_count(&[Some(0), Some(65)], &[Some(65)], &[Some(0)], [1, 0], 0),
            None
        );
        assert_eq!(
            tile_count(&[None, Some(65)], &[Some(65)], &[None], [1, 0], 0),
            None
        );
    }
}
