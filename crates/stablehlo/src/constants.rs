//! Fold pure value operations at the target precision. Unknown inputs and
//! unsupported/non-finite results retain their normal runtime lowering.
//! Views stay compact; folds materialize at most a bounded number of elements.

use super::*;
use std::fmt::Write;
use std::sync::Arc;

pub(super) type Constant = Arc<[Scalar]>;

// Bound newly materialized fixed tables and remembered index vectors.
pub(super) const MAX_FOLDED_ELEMENTS: usize = 4096;

impl Emitter<'_> {
    /// Read a fixed element through pure views without expanding them:
    /// reshape, broadcast, transpose, slice, concatenation, static gather and
    /// exact element-kind conversion. A reverse remembers its fixed values
    /// where it is emitted.
    pub(crate) fn fixed_scalar_at(&self, value: &Value, index: usize) -> Option<Scalar> {
        if index >= element_count(&value.ty)? {
            return None;
        }
        if let Some(data) = self.constants.get(&value.ssa) {
            return data.get(if data.len() == 1 { 0 } else { index }).cloned();
        }
        let at = |input: &Value, coordinates: &[usize]| {
            self.fixed_scalar_at(input, flat_index(&input.ty, coordinates)?)
        };
        match &self.pointwise.get(&value.ssa)?.op {
            Pointwise::Reshape(input) => self.fixed_scalar_at(input, index),
            Pointwise::Broadcast(input, dimensions) => {
                let out = coordinates_of(&value.ty, index)?;
                let source = shape(&input.ty)
                    .iter()
                    .zip(dimensions)
                    .map(|(size, axis)| match size {
                        Some(1) => Some(0),
                        _ => out.get(usize::try_from(*axis).ok()?).copied(),
                    })
                    .collect::<Option<Vec<_>>>()?;
                at(input, &source)
            }
            // Result axis i is input axis perm[i].
            Pointwise::Transpose(input, perm) => {
                let out = coordinates_of(&value.ty, index)?;
                let mut source = vec![0usize; out.len()];
                for (coordinate, axis) in out.iter().zip(perm) {
                    *source.get_mut(usize::try_from(*axis).ok()?)? = *coordinate;
                }
                at(input, &source)
            }
            Pointwise::Slice(input, starts, _, strides) => {
                let out = coordinates_of(&value.ty, index)?;
                let source = out
                    .iter()
                    .zip(starts.iter().zip(strides))
                    .map(|(c, (start, stride))| {
                        usize::try_from(*start)
                            .ok()?
                            .checked_add(c.checked_mul(usize::try_from(*stride).ok()?)?)
                    })
                    .collect::<Option<Vec<_>>>()?;
                at(input, &source)
            }
            Pointwise::Concat(parts, axis) => {
                let mut source = coordinates_of(&value.ty, index)?;
                let coordinate = source.get_mut(*axis)?;
                for part in parts {
                    let extent = usize::try_from((*shape(&part.ty).get(*axis)?)?).ok()?;
                    if *coordinate < extent {
                        return at(part, &source);
                    }
                    *coordinate -= extent;
                }
                None
            }
            // Static, in-bounds, zero-based selections along one axis.
            Pointwise::Gather(input, axis, indices) => {
                let mut source = coordinates_of(&value.ty, index)?;
                let coordinate = source.get_mut(*axis)?;
                *coordinate = usize::try_from(*indices.get(*coordinate)?).ok()?;
                at(input, &source)
            }
            Pointwise::Convert(input, target) => {
                convert_scalar(self.fixed_scalar_at(input, index)?, *target, self.dtype)
            }
            _ => None,
        }
    }

    /// The fixed values of `a` reversed along `dims`, bounded like other folds.
    pub(super) fn fixed_reversed(&self, a: &Value, dims: &[u64]) -> Option<Vec<Scalar>> {
        let count = element_count(&a.ty)?;
        if count > MAX_FOLDED_ELEMENTS {
            return None;
        }
        let extents = shape(&a.ty)
            .iter()
            .map(|d| usize::try_from((*d)?).ok())
            .collect::<Option<Vec<_>>>()?;
        (0..count)
            .map(|i| {
                let mut source = coordinates_of(&a.ty, i)?;
                for axis in dims {
                    let axis = usize::try_from(*axis).ok()?;
                    source[axis] = extents
                        .get(axis)?
                        .checked_sub(1)?
                        .checked_sub(source[axis])?;
                }
                self.fixed_scalar_at(a, flat_index(&a.ty, &source)?)
            })
            .collect()
    }

    fn fixed_count_at(&self, value: &Value, index: usize) -> Option<i64> {
        match self.fixed_scalar_at(value, index)? {
            Scalar::Int(n) => Some(n),
            _ => None,
        }
    }

    /// Round the whole fixed log-binomial coefficient once. Rounding three
    /// large lgamma values separately loses precision in mixture weights.
    pub(crate) fn fold_log_choose(&mut self, n: &Value, k: &Value) -> Option<Value> {
        let (n, k) = self.broadcast_pair(n, k);
        let count = element_count(&n.ty)?;
        if count > MAX_FOLDED_ELEMENTS {
            return None;
        }
        let values = (0..count)
            .map(|i| {
                let n = self.fixed_count_at(&n, i)?;
                let k = self.fixed_count_at(&k, i)?;
                if n < 0 {
                    return None;
                }
                // The caller masks out-of-support observations separately.
                let result = if k < 0 || k > n {
                    0.0
                } else {
                    libm::lgamma(n as f64 + 1.0)
                        - libm::lgamma(k as f64 + 1.0)
                        - libm::lgamma((n - k) as f64 + 1.0)
                };
                Some(Scalar::Real(result))
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, n.ty.clone(), self.axes_of(&n))
    }

    /// A fixed diagonal metric can be inverted without a matrix factorization.
    /// Query arguments have no constant entry, even when their source is fixed.
    pub(crate) fn constant_diagonal(&self, value: &Value) -> Option<Vec<f64>> {
        let [Some(rows), Some(cols)] = shape(&value.ty) else {
            return None;
        };
        if rows != cols || *rows == 0 || self.batch_rank(value) != 0 {
            return None;
        }
        let n = usize::try_from(*rows).ok()?;
        let data = self.constants.get(&value.ssa)?;
        let mut diagonal = Vec::with_capacity(n);
        for i in 0..n {
            for j in 0..n {
                let x = match data.get(if data.len() == 1 { 0 } else { i * n + j })? {
                    Scalar::Real(x) => *x,
                    Scalar::Int(x) => *x as f64,
                    _ => return None,
                };
                if i == j {
                    if !x.is_finite() || x == 0.0 {
                        return None;
                    }
                    diagonal.push(x);
                } else if x != 0.0 {
                    return None;
                }
            }
        }
        Some(diagonal)
    }

    pub(crate) fn constant_indices_in_bounds(
        &self,
        value: &Value,
        lower: i64,
        upper: u64,
    ) -> Option<bool> {
        if value.elem != ElemKind::Int {
            return None;
        }
        Some(
            self.constants
                .get(&value.ssa)?
                .iter()
                .all(|value| matches!(value, Scalar::Int(n) if *n >= lower && *n as u64 <= upper)),
        )
    }

    pub(crate) fn constant_integer_vector(&self, value: &Value) -> Option<Vec<i64>> {
        let MlirTy::Ranked(dims) = &value.ty else {
            return None;
        };
        let [Some(n)] = dims.as_slice() else {
            return None;
        };
        let data = self.constants.get(&value.ssa)?;
        (0..usize::try_from(*n).ok()?)
            .map(|i| match data.get(if data.len() == 1 { 0 } else { i })? {
                Scalar::Int(n) => Some(*n),
                _ => None,
            })
            .collect()
    }

    /// The values of a fixed, unbatched rank-1 real or integer vector, at the
    /// emitted dtype's precision.
    pub(crate) fn constant_reals(&self, value: &Value) -> Option<Vec<f64>> {
        let [Some(n)] = shape(&value.ty) else {
            return None;
        };
        if self.batch_rank(value) != 0 {
            return None;
        }
        let data = self.constants.get(&value.ssa)?;
        (0..usize::try_from(*n).ok()?)
            .map(|i| match data.get(if data.len() == 1 { 0 } else { i })? {
                Scalar::Real(x) => Some(*x),
                Scalar::Int(n) => Some(*n as f64),
                _ => None,
            })
            .collect()
    }

    /// Whether `value` is a fixed tensor of any shape. Only finite reals fold.
    pub(crate) fn is_constant(&self, value: &Value) -> bool {
        self.constants.contains_key(&value.ssa)
    }

    /// The value of a fixed, unbatched real scalar at the emitted precision.
    pub(crate) fn constant_real(&self, value: &Value) -> Option<f64> {
        if value.ty != MlirTy::Scalar || self.batch_rank(value) != 0 {
            return None;
        }
        match self.constants.get(&value.ssa)?.as_ref() {
            [Scalar::Real(x)] => Some(*x),
            _ => None,
        }
    }

    /// The value of a fixed, unbatched integer scalar.
    pub(crate) fn constant_int(&self, value: &Value) -> Option<i64> {
        if value.ty != MlirTy::Scalar || self.batch_rank(value) != 0 {
            return None;
        }
        match self.constants.get(&value.ssa)?.as_ref() {
            [Scalar::Int(n)] => Some(*n),
            _ => None,
        }
    }

    pub(crate) fn constant_extent(&self, value: &Value) -> Option<u64> {
        if value.ty != MlirTy::Scalar || self.batch_rank(value) != 0 {
            return None;
        }
        match self.constants.get(&value.ssa)?.as_ref() {
            [Scalar::Int(n)] => u64::try_from(*n).ok(),
            _ => None,
        }
    }

    /// Retain a fixed array element through the slice used by scalar indexing.
    pub(super) fn fold_extract(&mut self, a: &Value, starts: &[u64], ty: MlirTy) -> Option<Value> {
        if element_count(&ty) != Some(1) {
            return None;
        }
        let index = starts
            .iter()
            .zip(shape(&a.ty))
            .try_fold(0usize, |i, (&start, &dim)| {
                i.checked_mul(usize::try_from(dim?).ok()?)?
                    .checked_add(usize::try_from(start).ok()?)
            })?;
        let value = self.fixed_scalar_at(a, index)?;
        self.folded_constant(vec![value], ty, self.axes_of(a))
    }

    pub(super) fn fold_vector(&mut self, elems: &[Value], ty: MlirTy, axes: Axes) -> Option<Value> {
        let mut values = Vec::new();
        for elem in elems {
            let data = self.constants.get(&elem.ssa)?;
            // Do not materialize a broadcast expansion just to fold it.
            if element_count(&elem.ty) != Some(data.len()) {
                return None;
            }
            values.extend(data.iter().cloned());
        }
        self.folded_constant(values, ty, axes)
    }

    fn rounded_constant(&self, value: Scalar) -> Option<Scalar> {
        match value {
            Scalar::Real(x) => {
                let x = match self.dtype {
                    Dtype::F32 => (x as f32) as f64,
                    Dtype::F64 => x,
                };
                x.is_finite().then_some(Scalar::Real(x))
            }
            Scalar::Int(x) if self.int_fits(x) => Some(Scalar::Int(x)),
            Scalar::Bool(x) => Some(Scalar::Bool(x)),
            _ => None,
        }
    }

    pub(super) fn remember_constant(
        &mut self,
        ssa: &str,
        values: impl IntoIterator<Item = Scalar>,
    ) {
        if let Some(values) = values
            .into_iter()
            .map(|v| self.rounded_constant(v))
            .collect::<Option<Vec<_>>>()
        {
            self.constants.insert(ssa.to_owned(), values.into());
        }
    }

    /// Reshapes preserve element order. A broadcast can reuse a splat without
    /// allocating repeated data. Other extent changes remain backend work.
    pub(super) fn copy_constant(&mut self, from: &Value, to: &Value) {
        if let Some(data) = self.constants.get(&from.ssa)
            && (data.len() == 1 || element_count(&to.ty) == Some(data.len()))
        {
            self.constants.insert(to.ssa.clone(), data.clone());
        }
    }

    pub(super) fn folded_constant(
        &mut self,
        values: Vec<Scalar>,
        ty: MlirTy,
        axes: Axes,
    ) -> Option<Value> {
        let values = values
            .into_iter()
            .map(|v| self.rounded_constant(v))
            .collect::<Option<Vec<_>>>()?;
        let elem = match values.first()? {
            Scalar::Real(_) => ElemKind::Real,
            Scalar::Int(_) => ElemKind::Int,
            Scalar::Bool(_) => ElemKind::Bool,
            _ => return None,
        };
        let dims = if values.len() == 1 {
            &[][..]
        } else {
            if element_count(&ty) != Some(values.len()) {
                return None;
            }
            shape(&ty)
        };
        let literal = if elem == ElemKind::Real {
            real_buffer_literal(&values, self.dtype)?
        } else {
            dense_literal(&values, dims)?
        };
        let rendered = ty.render(self.dtype, elem);
        let ssa = self.pure_axes(
            format!("stablehlo.constant dense<{literal}> : {rendered}"),
            axes,
        );
        self.constants.insert(ssa.clone(), values.into());
        Some(Value { ssa, ty, elem })
    }

    pub(super) fn fold_unary(&mut self, op: &str, a: &Value) -> Option<Value> {
        let data = self.fixed_operand(a)?;
        let values = data
            .iter()
            .map(|v| match v {
                Scalar::Real(x) => real_unary(op, *x, self.dtype).map(Scalar::Real),
                Scalar::Int(x) => match op {
                    "stablehlo.negate" => x.checked_neg(),
                    "stablehlo.abs" => x.checked_abs(),
                    _ => None,
                }
                .map(Scalar::Int),
                Scalar::Bool(x) if op == "stablehlo.not" => Some(Scalar::Bool(!x)),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }

    /// Reduce fixed cells in axis order, rounding arithmetic at target precision.
    /// Bound reads as well as output storage; large views stay backend work.
    pub(super) fn fold_reduce_axis(
        &mut self,
        op: &str,
        init: &str,
        a: &Value,
        axis: usize,
        ty: MlirTy,
        axes: Axes,
    ) -> Option<Value> {
        let source_count = element_count(&a.ty)?;
        if source_count > MAX_FOLDED_ELEMENTS {
            return None;
        }
        let count = element_count(&ty)?;
        if count > MAX_FOLDED_ELEMENTS {
            return None;
        }
        if source_count != 0 {
            self.fixed_scalar_at(a, 0)?;
        }
        let width = usize::try_from((*shape(&a.ty).get(axis)?)?).ok()?;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let mut source = coordinates_of(&ty, i)?;
            source.insert(axis, 0);
            let mut total = match (a.elem, op) {
                (ElemKind::Real, "stablehlo.add" | "stablehlo.minimum" | "stablehlo.maximum") => {
                    Scalar::Real(if init == pos_inf_literal(self.dtype) {
                        f64::INFINITY
                    } else if init == reduce_max_identity(self.dtype) {
                        f64::NEG_INFINITY
                    } else {
                        init.parse().ok()?
                    })
                }
                (ElemKind::Int, "stablehlo.add" | "stablehlo.minimum" | "stablehlo.maximum") => {
                    Scalar::Int(init.parse().ok()?)
                }
                (ElemKind::Bool, "stablehlo.and" | "stablehlo.or") => {
                    Scalar::Bool(init.parse().ok()?)
                }
                _ => return None,
            };
            for k in 0..width {
                source[axis] = k;
                let value = self.fixed_scalar_at(a, flat_index(&a.ty, &source)?)?;
                total = self.rounded_constant(match (total, value) {
                    (Scalar::Real(x), Scalar::Real(y)) => {
                        Scalar::Real(real_binary(op, x, y, self.dtype)?)
                    }
                    (Scalar::Int(x), Scalar::Int(y)) => Scalar::Int(match op {
                        "stablehlo.add" => x.checked_add(y)?,
                        "stablehlo.minimum" => x.min(y),
                        "stablehlo.maximum" => x.max(y),
                        _ => return None,
                    }),
                    (Scalar::Bool(x), Scalar::Bool(y)) => Scalar::Bool(match op {
                        "stablehlo.and" => x && y,
                        "stablehlo.or" => x || y,
                        _ => return None,
                    }),
                    _ => return None,
                })?;
            }
            values.push(total);
        }
        self.folded_constant(values, ty, axes)
    }

    pub(super) fn fold_binary(&mut self, op: &str, a: &Value, b: &Value) -> Option<Value> {
        let (Some(aa), Some(bb)) = (self.fixed_operand(a), self.fixed_operand(b)) else {
            return None;
        };
        let n = aa.len().max(bb.len());
        if (aa.len() != 1 && aa.len() != n) || (bb.len() != 1 && bb.len() != n) {
            return None;
        }
        if aa.is_empty() || bb.is_empty() {
            return None;
        }
        let values = (0..n)
            .map(|i| {
                match (
                    &aa[if aa.len() == 1 { 0 } else { i }],
                    &bb[if bb.len() == 1 { 0 } else { i }],
                ) {
                    (Scalar::Real(x), Scalar::Real(y)) => {
                        real_binary(op, *x, *y, self.dtype).map(Scalar::Real)
                    }
                    (Scalar::Bool(x), Scalar::Bool(y)) => match op {
                        "stablehlo.and" => Some(Scalar::Bool(*x && *y)),
                        "stablehlo.or" => Some(Scalar::Bool(*x || *y)),
                        "stablehlo.xor" => Some(Scalar::Bool(x != y)),
                        _ => None,
                    },
                    (Scalar::Int(x), Scalar::Int(y)) => match op {
                        "stablehlo.add" => x.checked_add(*y),
                        "stablehlo.subtract" => x.checked_sub(*y),
                        "stablehlo.multiply" => x.checked_mul(*y),
                        "stablehlo.maximum" => Some((*x).max(*y)),
                        "stablehlo.minimum" => Some((*x).min(*y)),
                        // StableHLO integer divide and remainder truncate toward zero,
                        // as Rust's do. A zero divisor stays a run-time op.
                        "stablehlo.divide" => x.checked_div(*y),
                        "stablehlo.remainder" => x.checked_rem(*y),
                        _ => None,
                    }
                    .map(Scalar::Int),
                    _ => None,
                }
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }

    /// Dense fixed data, or a bounded expansion of a pure fixed view.
    /// Direct constants and splats reuse their shared storage.
    fn fixed_operand(&self, value: &Value) -> Option<Constant> {
        if let Some(data) = self.constants.get(&value.ssa) {
            return Some(data.clone());
        }
        let count = element_count(&value.ty)?;
        if count > MAX_FOLDED_ELEMENTS {
            return None;
        }
        (0..count)
            .map(|i| self.fixed_scalar_at(value, i))
            .collect::<Option<Vec<_>>>()
            .map(Into::into)
    }

    /// Gather outer cells from fixed data, reading broadcast sources lazily.
    /// Both operands already share a batch prefix; indices are zero-based.
    pub(super) fn fold_gather_cells(
        &mut self,
        operand: &Value,
        indices: &Value,
        ty: MlirTy,
        axes: Axes,
    ) -> Option<Value> {
        let count = element_count(&ty)?;
        if count > MAX_FOLDED_ELEMENTS {
            return None;
        }
        let batch = self.batch_rank(operand);
        let dims = shape(&operand.ty);
        let width = usize::try_from(*dims.get(batch)?.as_ref()?).ok()?;
        let cell = element_count(&MlirTy::Ranked(dims.get(batch + 1..)?.to_vec()))?;
        let selectors = element_count(&MlirTy::Ranked(shape(&indices.ty).get(batch..)?.to_vec()))?;
        let values = (0..count)
            .map(|i| {
                let selector = i.checked_div(cell)?;
                let index = usize::try_from(self.fixed_count_at(indices, selector)?).ok()?;
                if index >= width {
                    return None;
                }
                let source = selector
                    .checked_div(selectors)?
                    .checked_mul(width)?
                    .checked_add(index)?
                    .checked_mul(cell)?
                    .checked_add(i.checked_rem(cell)?)?;
                self.fixed_scalar_at(operand, source)
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, ty, axes)
    }

    pub(super) fn fold_convert(&mut self, a: &Value, target: ElemKind) -> Option<Value> {
        let data = self.fixed_operand(a)?;
        let values = data
            .iter()
            .map(|v| convert_scalar(v.clone(), target, self.dtype))
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }

    pub(super) fn fold_compare(&mut self, dir: &str, a: &Value, b: &Value) -> Option<Value> {
        use std::cmp::Ordering::{Equal, Greater, Less};

        let aa = self.fixed_operand(a)?;
        let bb = self.fixed_operand(b)?;
        let n = aa.len().max(bb.len());
        if aa.is_empty()
            || bb.is_empty()
            || (aa.len() != 1 && aa.len() != n)
            || (bb.len() != 1 && bb.len() != n)
        {
            return None;
        }
        let values = (0..n)
            .map(|i| {
                let order = match (&aa[i % aa.len()], &bb[i % bb.len()]) {
                    (Scalar::Real(x), Scalar::Real(y)) => x.partial_cmp(y)?,
                    (Scalar::Int(x), Scalar::Int(y)) => x.cmp(y),
                    (Scalar::Bool(x), Scalar::Bool(y)) => x.cmp(y),
                    _ => return None,
                };
                Some(Scalar::Bool(match dir {
                    "EQ" => order == Equal,
                    "NE" => order != Equal,
                    "LT" => order == Less,
                    "LE" => order != Greater,
                    "GT" => order == Greater,
                    "GE" => order != Less,
                    _ => return None,
                }))
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }

    pub(super) fn fold_select(&mut self, c: &Value, a: &Value, b: &Value) -> Option<Value> {
        if c.ty != MlirTy::Scalar && c.ty != a.ty {
            return None;
        }
        let cc = self.fixed_operand(c)?;
        let Scalar::Bool(first) = cc.first()? else {
            return None;
        };
        if cc
            .iter()
            .all(|v| matches!(v, Scalar::Bool(v) if v == first))
        {
            return Some(if *first { a.clone() } else { b.clone() });
        }
        let aa = self.fixed_operand(a)?;
        let bb = self.fixed_operand(b)?;
        let n = cc.len().max(aa.len()).max(bb.len());
        if [cc.len(), aa.len(), bb.len()]
            .iter()
            .any(|&len| len != 1 && len != n)
        {
            return None;
        }
        let values = (0..n)
            .map(|i| match cc[i % cc.len()] {
                Scalar::Bool(true) => Some(aa[i % aa.len()].clone()),
                Scalar::Bool(false) => Some(bb[i % bb.len()].clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }
}

/// The canonical embeddings `booleans ⊂ integers ⊂ reals` (§03), with an
/// integer rounded once to the target precision.
fn convert_scalar(value: Scalar, target: ElemKind, dtype: Dtype) -> Option<Scalar> {
    match (value, target) {
        (Scalar::Int(x), ElemKind::Real) => Some(Scalar::Real(match dtype {
            Dtype::F32 => (x as f32) as f64,
            Dtype::F64 => x as f64,
        })),
        (Scalar::Bool(x), ElemKind::Real) => Some(Scalar::Real(f64::from(x))),
        (Scalar::Bool(x), ElemKind::Int) => Some(Scalar::Int(i64::from(x))),
        _ => None,
    }
}

/// Row-major coordinates of a flat index in a static shape.
fn coordinates_of(ty: &MlirTy, index: usize) -> Option<Vec<usize>> {
    let dims = shape(ty);
    let mut coordinates = vec![0usize; dims.len()];
    let mut remaining = index;
    for (coordinate, size) in coordinates.iter_mut().zip(dims).rev() {
        let size = usize::try_from((*size)?).ok()?;
        *coordinate = remaining.checked_rem(size)?;
        remaining = remaining.checked_div(size)?;
    }
    Some(coordinates)
}

/// The flat row-major index of in-bounds coordinates in a static shape.
fn flat_index(ty: &MlirTy, coordinates: &[usize]) -> Option<usize> {
    let dims = shape(ty);
    if dims.len() != coordinates.len() {
        return None;
    }
    let mut index = 0usize;
    for (coordinate, size) in coordinates.iter().zip(dims) {
        let size = usize::try_from((*size)?).ok()?;
        if *coordinate >= size {
            return None;
        }
        index = index.checked_mul(size)?.checked_add(*coordinate)?;
    }
    Some(index)
}

pub(super) fn element_count(ty: &MlirTy) -> Option<usize> {
    shape(ty)
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(usize::try_from((*d)?).ok()?))
}

fn real_buffer_literal(values: &[Scalar], dtype: Dtype) -> Option<String> {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let width = match dtype {
        Dtype::F32 => 8,
        Dtype::F64 => 16,
    };
    let mut out = String::with_capacity(values.len().checked_mul(width)?.checked_add(4)?);
    out.push_str("\"0x");
    for value in values {
        let Scalar::Real(x) = value else {
            return None;
        };
        // MLIR raw dense buffers use little-endian bytes on every host.
        let bits = match dtype {
            Dtype::F32 => u64::from((*x as f32).to_bits()),
            Dtype::F64 => x.to_bits(),
        };
        for byte in &bits.to_le_bytes()[..width / 2] {
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 15)]));
        }
    }
    out.push('"');
    Some(out)
}

fn dense_literal(values: &[Scalar], dims: &[Option<u64>]) -> Option<String> {
    fn append(out: &mut String, values: &[Scalar], dims: &[Option<u64>]) -> Option<()> {
        if dims.is_empty() {
            match values.first()? {
                Scalar::Real(x) => {
                    let start = out.len();
                    write!(out, "{x}").ok()?;
                    // Match render_float_literal without allocating each token.
                    if !out[start..].contains(['.', 'e', 'E']) {
                        out.push_str(".0");
                    }
                }
                Scalar::Int(x) => write!(out, "{x}").ok()?,
                Scalar::Bool(x) => write!(out, "{x}").ok()?,
                _ => return None,
            }
            return Some(());
        }
        let n = usize::try_from(dims[0]?).ok()?;
        out.push('[');
        if let Some(width) = values.len().checked_div(n) {
            for i in 0..n {
                if i != 0 {
                    out.push_str(", ");
                }
                append(out, &values[i * width..(i + 1) * width], &dims[1..])?;
            }
        }
        out.push(']');
        Some(())
    }
    let mut out = String::new();
    append(&mut out, values, dims)?;
    Some(out)
}

fn real_unary(op: &str, x: f64, dtype: Dtype) -> Option<f64> {
    // Use each target's math implementation, not f64 followed by f32 rounding.
    macro_rules! unary {
        ($x:expr, $log:path, $exp:path, $sqrt:path, $lgamma:path, $log1p:path, $expm1:path, $erf:path, $erfc:path) => {{
            let x = $x;
            match op {
                "stablehlo.negate" => -x,
                "stablehlo.abs" => x.abs(),
                "stablehlo.log" => $log(x),
                "stablehlo.exponential" => $exp(x),
                "stablehlo.sqrt" => $sqrt(x),
                "stablehlo.log_plus_one" => $log1p(x),
                "stablehlo.exponential_minus_one" => $expm1(x),
                "chlo.lgamma" => $lgamma(x),
                "chlo.erf" => $erf(x),
                "chlo.erfc" => $erfc(x),
                _ => return None,
            }
        }};
    }
    Some(match dtype {
        Dtype::F32 => unary!(
            x as f32,
            libm::logf,
            libm::expf,
            libm::sqrtf,
            libm::lgammaf,
            libm::log1pf,
            libm::expm1f,
            libm::erff,
            libm::erfcf
        ) as f64,
        Dtype::F64 => unary!(
            x,
            libm::log,
            libm::exp,
            libm::sqrt,
            libm::lgamma,
            libm::log1p,
            libm::expm1,
            libm::erf,
            libm::erfc
        ),
    })
}

fn real_binary(op: &str, x: f64, y: f64, dtype: Dtype) -> Option<f64> {
    macro_rules! binary {
        ($x:expr, $y:expr, $pow:path) => {{
            let (x, y) = ($x, $y);
            match op {
                "stablehlo.add" => x + y,
                "stablehlo.subtract" => x - y,
                "stablehlo.multiply" => x * y,
                "stablehlo.divide" => x / y,
                "stablehlo.power" => $pow(x, y),
                // IEEE maximum/minimum order -0 below +0, unlike Rust's
                // unspecified choice for equal zeros in max/min.
                "stablehlo.maximum" if x == 0.0 && y == 0.0 => x + y,
                "stablehlo.minimum" if x == 0.0 && y == 0.0 => -(-x + -y),
                "stablehlo.maximum" => x.max(y),
                "stablehlo.minimum" => x.min(y),
                _ => return None,
            }
        }};
    }
    Some(match dtype {
        Dtype::F32 => binary!(x as f32, y as f32, libm::powf) as f64,
        Dtype::F64 => binary!(x, y, libm::pow),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_powers_and_axis_sums_preserve_cells_and_precision() {
        let module = Module::new();
        for dtype in [Dtype::F32, Dtype::F64] {
            let mut e = Emitter::new(&module, dtype);
            let a = e
                .folded_constant(
                    [-1., -1., -1., 2., 2., 2.].map(Scalar::Real).to_vec(),
                    MlirTy::Ranked(vec![Some(2), Some(3)]),
                    Axes {
                        batch: 1,
                        layers: vec![1],
                    },
                )
                .unwrap();
            let b = e
                .folded_constant(
                    [0., 1., 2., 0., 1., 2.].map(Scalar::Real).to_vec(),
                    a.ty.clone(),
                    e.axes_of(&a),
                )
                .unwrap();
            let powers = e.fold_binary("stablehlo.power", &a, &b).unwrap();
            let sums = e.reduce_sum_last_axis(&powers);
            assert_eq!(
                e.fixed_operand(&sums).unwrap().as_ref(),
                &[Scalar::Real(1.), Scalar::Real(7.)]
            );
            assert_eq!(
                e.axes_of(&sums),
                Axes {
                    batch: 1,
                    layers: vec![]
                }
            );
            let a = e
                .folded_constant(
                    [16777216., 1., -16777216.].map(Scalar::Real).to_vec(),
                    MlirTy::Ranked(vec![Some(3)]),
                    Axes::default(),
                )
                .unwrap();
            let sum = e.reduce_sum(&a);
            assert_eq!(
                e.constant_real(&sum),
                Some(if matches!(dtype, Dtype::F32) { 0. } else { 1. })
            );
        }
    }

    #[test]
    fn fixed_sum_and_power_fallbacks_remain_runtime() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        let unknown = Value {
            ssa: "%arg0".into(),
            ty: MlirTy::Ranked(vec![Some(2)]),
            elem: ElemKind::Real,
        };
        assert!(
            e.fold_reduce_axis(
                "stablehlo.add",
                "0",
                &unknown,
                0,
                MlirTy::Scalar,
                Axes::default()
            )
            .is_none()
        );
        for values in [
            vec![Scalar::Real(f64::from(f32::MAX)); 2],
            vec![Scalar::Int(i64::from(i32::MAX)), Scalar::Int(1)],
        ] {
            let a = e
                .folded_constant(values, unknown.ty.clone(), Axes::default())
                .unwrap();
            assert!(
                e.fold_reduce_axis("stablehlo.add", "0", &a, 0, MlirTy::Scalar, Axes::default())
                    .is_none()
            );
        }
        let huge = e
            .folded_constant(
                vec![Scalar::Real(1.)],
                MlirTy::Ranked(vec![Some(4097)]),
                Axes::default(),
            )
            .unwrap();
        assert!(
            e.fold_reduce_axis(
                "stablehlo.add",
                "0",
                &huge,
                0,
                MlirTy::Scalar,
                Axes::default()
            )
            .is_none()
        );
        let a = e
            .folded_constant(vec![Scalar::Real(-1.)], MlirTy::Scalar, Axes::default())
            .unwrap();
        let b = e
            .folded_constant(vec![Scalar::Real(0.5)], MlirTy::Scalar, Axes::default())
            .unwrap();
        assert!(e.fold_binary("stablehlo.power", &a, &b).is_none());
        let a = e
            .folded_constant(vec![Scalar::Real(0.)], MlirTy::Scalar, Axes::default())
            .unwrap();
        let b = e
            .folded_constant(vec![Scalar::Real(-1.)], MlirTy::Scalar, Axes::default())
            .unwrap();
        assert!(e.fold_binary("stablehlo.power", &a, &b).is_none());
    }

    #[test]
    fn empty_axis_sum_bounds_output_storage() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        for width in [2, 1000000000] {
            let empty = Value {
                ssa: "%empty".into(),
                ty: MlirTy::Ranked(vec![Some(0), Some(width)]),
                elem: ElemKind::Real,
            };
            let result = e.fold_reduce_axis(
                "stablehlo.add",
                "0",
                &empty,
                0,
                MlirTy::Ranked(vec![Some(width)]),
                Axes::default(),
            );
            if width == 2 {
                assert_eq!(
                    e.fixed_operand(&result.unwrap()).unwrap().as_ref(),
                    &[Scalar::Real(0.), Scalar::Real(0.)]
                );
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn real_folds_read_broadcast_and_transpose_coordinates() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        let source = Value {
            ssa: "%source".into(),
            ty: MlirTy::Ranked(vec![Some(2)]),
            elem: ElemKind::Real,
        };
        e.remember_constant(&source.ssa, [Scalar::Real(1.), Scalar::Real(2.)]);
        let view = e.broadcast_in_dim(&source, &[0], MlirTy::Ranked(vec![Some(2), Some(3)]));
        let view = e.transpose(&view, &[1, 0]);
        let folded = e.fold_unary("stablehlo.negate", &view).unwrap();
        let values = e.fixed_operand(&folded).unwrap();
        assert_eq!(
            values.as_ref(),
            &[-1., -2., -1., -2., -1., -2.].map(Scalar::Real)
        );
        let gamma = e.fold_unary("chlo.lgamma", &view).unwrap();
        assert!(
            e.fixed_operand(&gamma)
                .unwrap()
                .iter()
                .all(|x| *x == Scalar::Real(0.))
        );
    }

    #[test]
    fn real_view_folds_leave_unknown_and_oversized_values_live() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        let source = Value {
            ssa: "%source".into(),
            ty: MlirTy::Ranked(vec![Some(2)]),
            elem: ElemKind::Real,
        };
        let view = e.broadcast_in_dim(&source, &[0], MlirTy::Ranked(vec![Some(2), Some(2)]));
        assert!(e.fold_unary("chlo.lgamma", &view).is_none());
        e.remember_constant(&source.ssa, [Scalar::Real(1.), Scalar::Real(2.)]);
        let view = e.broadcast_in_dim(&source, &[0], MlirTy::Ranked(vec![Some(2), Some(4097)]));
        assert!(e.fixed_operand(&view).is_none());
        assert!(e.fold_unary("chlo.lgamma", &view).is_none());
        let data = e.constants.get(&source.ssa).unwrap();
        assert!(Arc::ptr_eq(data, &e.fixed_operand(&source).unwrap()));
    }

    #[test]
    fn view_conversion_rounds_at_target_precision_and_extract_reads_one_element() {
        let module = Module::new();
        for dtype in [Dtype::F32, Dtype::F64] {
            let mut e = Emitter::new(&module, dtype);
            let source = Value {
                ssa: "%source".into(),
                ty: MlirTy::Ranked(vec![Some(2)]),
                elem: ElemKind::Int,
            };
            e.remember_constant(&source.ssa, [Scalar::Int(16777217), Scalar::Int(2)]);
            let view = e.broadcast_in_dim(&source, &[0], MlirTy::Ranked(vec![Some(2), Some(3)]));
            let real = e.fold_convert(&view, ElemKind::Real).unwrap();
            let expected = if matches!(dtype, Dtype::F32) {
                16777216.
            } else {
                16777217.
            };
            assert_eq!(e.fixed_scalar_at(&real, 2), Some(Scalar::Real(expected)));
            let view = e.transpose(&real, &[1, 0]);
            let scalar = e.fold_extract(&view, &[2, 0], MlirTy::Scalar).unwrap();
            assert_eq!(e.constant_real(&scalar), Some(expected));
            let huge = e.broadcast_in_dim(
                &real,
                &[0, 1],
                MlirTy::Ranked(vec![Some(2), Some(3), Some(4097)]),
            );
            assert!(e.fixed_operand(&huge).is_none());
            let scalar = e
                .fold_extract(&huge, &[1, 2, 4096], MlirTy::Scalar)
                .unwrap();
            assert_eq!(e.constant_real(&scalar), Some(2.));
        }
    }

    #[test]
    fn real_min_max_preserve_signed_zero_order() {
        for dtype in [Dtype::F32, Dtype::F64] {
            for (x, y, maximum, minimum) in [
                (0., 0., 0., 0.),
                (-0., 0., 0., -0.),
                (0., -0., 0., -0.),
                (-0., -0., -0., -0.),
                (-3., 2., 2., -3.),
            ] {
                assert_eq!(
                    real_binary("stablehlo.maximum", x, y, dtype)
                        .unwrap()
                        .to_bits(),
                    f64::to_bits(maximum)
                );
                assert_eq!(
                    real_binary("stablehlo.minimum", x, y, dtype)
                        .unwrap()
                        .to_bits(),
                    f64::to_bits(minimum)
                );
            }
        }
    }

    #[test]
    fn fixed_integer_clamp_converts_and_folds_lgamma() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        let source = Value {
            ssa: "%source".into(),
            ty: MlirTy::Ranked(vec![Some(2)]),
            elem: ElemKind::Int,
        };
        e.remember_constant(&source.ssa, [Scalar::Int(-1), Scalar::Int(0)]);
        let zeros = e
            .folded_constant(vec![Scalar::Int(0)], source.ty.clone(), Axes::default())
            .unwrap();
        let clamped = e.fold_binary("stablehlo.maximum", &source, &zeros).unwrap();
        let ones = e
            .folded_constant(vec![Scalar::Int(1)], source.ty.clone(), Axes::default())
            .unwrap();
        let shifted = e.fold_binary("stablehlo.add", &clamped, &ones).unwrap();
        let real = e.fold_convert(&shifted, ElemKind::Real).unwrap();
        let gamma = e.fold_unary("chlo.lgamma", &real).unwrap();
        assert!(
            e.fixed_operand(&gamma)
                .unwrap()
                .iter()
                .all(|x| *x == Scalar::Real(0.))
        );
    }

    #[test]
    fn fixed_extrema_and_boolean_reductions_keep_cells_and_target_precision() {
        let module = Module::new();
        for dtype in [Dtype::F32, Dtype::F64] {
            let mut e = Emitter::new(&module, dtype);
            let axes = Axes {
                batch: 1,
                layers: vec![1],
            };
            let values = e
                .folded_constant(
                    [0., -0., 1., 2., 16777217., 16777218.]
                        .map(Scalar::Real)
                        .to_vec(),
                    MlirTy::Ranked(vec![Some(3), Some(2)]),
                    axes.clone(),
                )
                .unwrap();
            let minimum = e.reduce_min(&values);
            let maximum = e.reduce_max(&values);
            let magnitude = e.abs(&values);
            let infinity = e.inf_like(&magnitude);
            let finite = e.compare("LT", &magnitude, &infinity);
            let finite = e.reduce_all(&finite);
            assert_eq!(
                e.fixed_operand(&finite).unwrap().as_ref(),
                &[Scalar::Bool(true), Scalar::Bool(true), Scalar::Bool(true)]
            );
            let rounded = if matches!(dtype, Dtype::F32) {
                16777216.
            } else {
                16777217.
            };
            for (value, expected) in [
                (&minimum, [-0., 1., rounded]),
                (&maximum, [0., 2., 16777218.]),
            ] {
                for (actual, expected) in e.fixed_operand(value).unwrap().iter().zip(expected) {
                    let Scalar::Real(actual) = actual else {
                        panic!("expected real extremum")
                    };
                    assert_eq!(actual.to_bits(), f64::to_bits(expected));
                }
                assert_eq!(
                    e.axes_of(value),
                    Axes {
                        batch: 1,
                        layers: vec![]
                    }
                );
            }
            let flags = e
                .folded_constant(
                    [true, false, true, true, false, false]
                        .map(Scalar::Bool)
                        .to_vec(),
                    values.ty.clone(),
                    axes,
                )
                .unwrap();
            let all = e.reduce_all(&flags);
            let any = e.reduce_boolean("stablehlo.or", "false", &flags);
            assert_eq!(
                e.fixed_operand(&all).unwrap().as_ref(),
                &[Scalar::Bool(false), Scalar::Bool(true), Scalar::Bool(false)]
            );
            assert_eq!(
                e.fixed_operand(&any).unwrap().as_ref(),
                &[Scalar::Bool(true), Scalar::Bool(true), Scalar::Bool(false)]
            );
        }
    }
}
