//! Fold pure value operations at the target precision. Unknown inputs and
//! unsupported/non-finite results retain their normal runtime lowering.
//! Data stays compact: broadcasts only propagate splats or unchanged extents.

use super::*;
use std::sync::Arc;

pub(super) type Constant = Arc<[Scalar]>;

impl Emitter<'_> {
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
        let data = self.constants.get(&a.ssa)?;
        let index = starts
            .iter()
            .zip(shape(&a.ty))
            .try_fold(0usize, |i, (&start, &dim)| {
                i.checked_mul(usize::try_from(dim?).ok()?)?
                    .checked_add(usize::try_from(start).ok()?)
            })?;
        let value = data.get(if data.len() == 1 { 0 } else { index })?.clone();
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
            Scalar::Int(x) if matches!(self.dtype, Dtype::F64) || i32::try_from(x).is_ok() => {
                Some(Scalar::Int(x))
            }
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

    fn folded_constant(&mut self, values: Vec<Scalar>, ty: MlirTy, axes: Axes) -> Option<Value> {
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
        let literals = values
            .iter()
            .map(|v| match v {
                Scalar::Real(x) => render_float_literal(*x),
                Scalar::Int(x) => x.to_string(),
                Scalar::Bool(x) => x.to_string(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let literal = if values.len() == 1 {
            literals[0].clone()
        } else {
            if element_count(&ty) != Some(values.len()) {
                return None;
            }
            dense_literal(&literals, shape(&ty))?
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
        let data = self.constants.get(&a.ssa)?;
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
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }

    pub(super) fn fold_binary(&mut self, op: &str, a: &Value, b: &Value) -> Option<Value> {
        let aa = self.constants.get(&a.ssa)?;
        let bb = self.constants.get(&b.ssa)?;
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
                    (Scalar::Int(x), Scalar::Int(y)) => match op {
                        "stablehlo.add" => x.checked_add(*y),
                        "stablehlo.subtract" => x.checked_sub(*y),
                        "stablehlo.multiply" => x.checked_mul(*y),
                        _ => None,
                    }
                    .map(Scalar::Int),
                    _ => None,
                }
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }

    pub(super) fn fold_convert(&mut self, a: &Value, target: ElemKind) -> Option<Value> {
        let data = self.constants.get(&a.ssa)?;
        let values = data
            .iter()
            .map(|v| match (v, target) {
                (Scalar::Int(x), ElemKind::Real) => Some(Scalar::Real(match self.dtype {
                    Dtype::F32 => (*x as f32) as f64,
                    Dtype::F64 => *x as f64,
                })),
                (Scalar::Bool(x), ElemKind::Real) => Some(Scalar::Real(f64::from(*x))),
                (Scalar::Bool(x), ElemKind::Int) => Some(Scalar::Int(i64::from(*x))),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        self.folded_constant(values, a.ty.clone(), self.axes_of(a))
    }
}

fn element_count(ty: &MlirTy) -> Option<usize> {
    shape(ty)
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(usize::try_from((*d)?).ok()?))
}

fn dense_literal(values: &[String], dims: &[Option<u64>]) -> Option<String> {
    if dims.is_empty() {
        return values.first().cloned();
    }
    let n = usize::try_from(dims[0]?).ok()?;
    if n == 0 {
        return Some("[]".to_owned());
    }
    let width = values.len() / n;
    let rows = (0..n)
        .map(|i| dense_literal(&values[i * width..(i + 1) * width], &dims[1..]))
        .collect::<Option<Vec<_>>>()?;
    Some(format!("[{}]", rows.join(", ")))
}

fn real_unary(op: &str, x: f64, dtype: Dtype) -> Option<f64> {
    // Use each target's math implementation, not f64 followed by f32 rounding.
    macro_rules! unary {
        ($x:expr, $log:path, $exp:path, $sqrt:path, $lgamma:path, $log1p:path, $expm1:path) => {{
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
            libm::expm1f
        ) as f64,
        Dtype::F64 => unary!(
            x,
            libm::log,
            libm::exp,
            libm::sqrt,
            libm::lgamma,
            libm::log1p,
            libm::expm1
        ),
    })
}

fn real_binary(op: &str, x: f64, y: f64, dtype: Dtype) -> Option<f64> {
    macro_rules! binary {
        ($x:expr, $y:expr) => {{
            let (x, y) = ($x, $y);
            match op {
                "stablehlo.add" => x + y,
                "stablehlo.subtract" => x - y,
                "stablehlo.multiply" => x * y,
                "stablehlo.divide" => x / y,
                _ => return None,
            }
        }};
    }
    Some(match dtype {
        Dtype::F32 => binary!(x as f32, y as f32) as f64,
        Dtype::F64 => binary!(x, y),
    })
}
