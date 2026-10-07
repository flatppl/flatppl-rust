//! Pivoted elimination and tensor-axis contractions using portable StableHLO.

use super::batching::tensor;
use super::*;

impl Emitter<'_> {
    /// Enzyme's reverse-loop tape uses i64 slice offsets. Keep its induction
    /// variable i64 too; convert only the index consumed by tensor arithmetic.
    fn matrix_loop(
        &mut self,
        n: u64,
        inits: &[Value],
        body: impl FnOnce(&mut Self, &Value, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let counter = Value {
            ssa: self.pure("stablehlo.constant dense<0> : tensor<i64>".into()),
            ty: MlirTy::Scalar,
            elem: ElemKind::Int,
        };
        let mut states = vec![counter];
        states.extend_from_slice(inits);
        // The control value uses the explicit carried type, not the ABI's Int
        // width. It never escapes this helper or enters the typed arithmetic.
        let mut types = vec!["tensor<i64>".to_owned()];
        types.extend(inits.iter().map(|v| v.ty.render(self.dtype, v.elem)));
        let result = self.while_loop(
            &states,
            &types,
            |e, args| {
                let limit = e.pure(format!("stablehlo.constant dense<{n}> : tensor<i64>"));
                let ssa = e.pure(format!(
                    "stablehlo.compare LT, {}, {limit}, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>",
                    args[0].ssa,
                ));
                Value { ssa, ty: MlirTy::Scalar, elem: ElemKind::Bool }
            },
            |e, args| {
                let ty = MlirTy::Scalar.render(e.dtype, ElemKind::Int);
                let ssa = e.pure(format!("stablehlo.convert {} : (tensor<i64>) -> {ty}", args[0].ssa));
                let index = Value { ssa, ty: MlirTy::Scalar, elem: ElemKind::Int };
                let one = e.pure("stablehlo.constant dense<1> : tensor<i64>".into());
                let ssa = e.pure(format!("stablehlo.add {}, {one} : tensor<i64>", args[0].ssa));
                let mut next = vec![Value { ssa, ..args[0].clone() }];
                next.extend(body(e, &index, &args[1..]));
                next
            },
        );
        result.into_iter().skip(1).collect()
    }

    /// Right-looking Cholesky through rank-one Schur updates. Symmetrizing the
    /// input gives the symmetric cotangent on the positive-definite domain.
    pub(super) fn cholesky_decomposed(&mut self, input: &Value) -> Value {
        let batch = self.batch_rank(input);
        let axes = self.axes_of(input);
        let n = shape(&input.ty)[batch].expect("static Cholesky dimension");
        let mut permutation = (0..batch as u64).collect::<Vec<_>>();
        permutation.extend([batch as u64 + 1, batch as u64]);
        let transpose = self.transpose(input, &permutation);
        let transpose = self.axes_view(&transpose, axes.clone());
        let difference = self.sub(&transpose, input);
        let half = self.scalar(0.5);
        let correction = self.mul(&half, &difference);
        let symmetric = self.add(input, &correction);
        let zero = self.constant_like(0.0, input);
        let cols = self.axis_indices(input, batch + 1);
        let result = self.matrix_loop(n, &[symmetric, zero.clone()], |e, index, args| {
            let schur = e.axes_view(&args[0], axes.clone());
            let factor = e.axes_view(&args[1], axes.clone());
            let at_col = e.compare("EQ", &cols, index);
            let column = e.select(&at_col, &schur, &zero);
            let column = e.reduce_sum_last_axis(&column);
            let indices = e.axis_indices(&column, batch);
            let at_pivot = e.compare("EQ", &indices, index);
            let column_zero = e.constant_like(0.0, &column);
            let pivot = e.select(&at_pivot, &column, &column_zero);
            let pivot = e.reduce_sum_last_axis(&pivot);
            let diagonal = e.sqrt(&pivot);
            let active = e.compare("GT", &indices, index);
            let column = e.select(&active, &column, &column_zero);
            let column = e.div(&column, &diagonal);
            let column = e.select(&at_pivot, &diagonal, &column);
            let col = e.matrix_expand(&column, &schur, batch);
            let row = e.matrix_expand(&column, &schur, batch + 1);
            let product = e.mul(&col, &row);
            vec![e.sub(&schur, &product), e.select(&at_col, &col, &factor)]
        });
        self.axes_view(&result[1], axes)
    }

    /// Forward substitution for lower-triangular matrix cells and matrix RHSs.
    /// The fixed loop bound permits reverse AD without a native solve adjoint.
    pub(super) fn tri_solve_decomposed(&mut self, l: &Value, b: &Value) -> Value {
        let batch = self.batch_rank(l);
        let n = shape(&l.ty)[batch].expect("static triangular solve dimension");
        let axes = self.axes_of(b);
        let rows = self.axis_indices(l, batch);
        let rhs_rows = self.axis_indices(b, batch);
        let zero = self.constant_like(0.0, b);
        let result = self.matrix_loop(n, &[zero], |e, index, args| {
            let x = e.axes_view(&args[0], axes.clone());
            let at_row = e.compare("EQ", &rows, index);
            let rhs_at_row = e.compare("EQ", &rhs_rows, index);
            let row = e.selected_row(l, &at_row);
            let rhs = e.selected_row(b, &rhs_at_row);
            let indices = e.axis_indices(&row, batch);
            let at_pivot = e.compare("EQ", &indices, index);
            let row_zero = e.constant_like(0.0, &row);
            let pivot = e.select(&at_pivot, &row, &row_zero);
            let pivot = e.reduce_sum_last_axis(&pivot);
            let before = e.compare("LT", &indices, index);
            let coefficients = e.select(&before, &row, &row_zero);
            let coefficients = e.matrix_expand(&coefficients, &x, batch);
            let products = e.mul(&coefficients, &x);
            let sum = e.reduce_axis("stablehlo.add", "0.000000e+00", &products, batch);
            let residual = e.sub(&rhs, &sum);
            let solution = e.div(&residual, &pivot);
            let solution = e.matrix_expand(&solution, &x, batch + 1);
            vec![e.select(&rhs_at_row, &solution, &x)]
        });
        self.axes_view(&result[0], axes)
    }

    pub(crate) fn diagonal_axes(&mut self, value: &Value, first: usize, second: usize) -> Value {
        let first = self.axis_indices(value, first);
        let second_indices = self.axis_indices(value, second);
        let mask = self.compare("EQ", &first, &second_indices);
        let zero = self.int_value_const(0);
        let zero = self.convert(&zero, value.elem);
        let selected = self.select(&mask, value, &zero);
        self.reduce_axis("stablehlo.add", "0.000000e+00", &selected, second)
    }

    pub(crate) fn axis_indices(&mut self, value: &Value, axis: usize) -> Value {
        let ty = value.ty.render(self.dtype, ElemKind::Int);
        let ssa = self.pure_like(format!("stablehlo.iota dim = {axis} : {ty}"), value);
        Value {
            ssa,
            ty: value.ty.clone(),
            elem: ElemKind::Int,
        }
    }

    /// Apply a matrix to one tensor cell axis, preserving its original position.
    pub(crate) fn transform_axis(&mut self, value: &Value, matrix: &Value, axis: usize) -> Value {
        let value = self.convert(value, ElemKind::Real);
        let matrix = self.convert(matrix, ElemKind::Real);
        let (value, matrix) = self.broadcast_batches(&value, &matrix);
        let batch = self.batch_rank(&value);
        let physical = batch + axis;
        let mut dims = shape(&value.ty).to_vec();
        dims.remove(physical);
        dims.push(shape(&matrix.ty)[batch]);
        let ty = tensor(dims);
        let batches = (0..batch)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let batching = if batch == 0 {
            String::new()
        } else {
            format!("batching_dims = [{batches}] x [{batches}], ")
        };
        let ssa = self.pure_like(format!(
            "stablehlo.dot_general {}, {}, {batching}contracting_dims = [{physical}] x [{}], precision = [HIGHEST, HIGHEST] : ({}, {}) -> {}",
            value.ssa, matrix.ssa, batch + 1, value.ty.render(self.dtype, value.elem),
            matrix.ty.render(self.dtype, matrix.elem), ty.render(self.dtype, ElemKind::Real),
        ), &value);
        let product = Value {
            ssa,
            ty,
            elem: ElemKind::Real,
        };
        let rank = shape(&product.ty).len();
        let mut permutation = (0..rank - 1).map(|i| i as u64).collect::<Vec<_>>();
        permutation.insert(physical, (rank - 1) as u64);
        let result = self.transpose(&product, &permutation);
        self.axes_view(&result, self.axes_of(&value))
    }

    fn matrix_expand(&mut self, vector: &Value, matrix: &Value, axis: usize) -> Value {
        let batch = self.batch_rank(matrix);
        let map = (0..batch as u64).chain([axis as u64]).collect::<Vec<_>>();
        self.expand_axes(vector, &map, matrix.ty.clone(), self.axes_of(matrix))
    }

    fn selected_row(&mut self, matrix: &Value, mask: &Value) -> Value {
        let zero = self.constant_like(0.0, matrix);
        let selected = self.select(mask, matrix, &zero);
        self.reduce_axis(
            "stablehlo.add",
            "0.000000e+00",
            &selected,
            self.batch_rank(matrix),
        )
    }

    fn swap_rows(&mut self, matrix: &Value, current: &Value, pivot: &Value) -> Value {
        let batch = self.batch_rank(matrix);
        let current_row = self.selected_row(matrix, current);
        let pivot_row = self.selected_row(matrix, pivot);
        let current_row = self.matrix_expand(&current_row, matrix, batch + 1);
        let pivot_row = self.matrix_expand(&pivot_row, matrix, batch + 1);
        let swapped = self.select(pivot, &current_row, matrix);
        self.select(current, &pivot_row, &swapped)
    }

    /// Complete pivoting moves rank deficiency to the final diagonal entries.
    /// This retains the determinant's cofactor derivative at a zero column.
    pub(crate) fn matrix_determinant(&mut self, input: &Value, n: u64, logarithm: bool) -> Value {
        if n == 1 {
            let value = self.reduce_sum(input);
            return if logarithm {
                let magnitude = self.abs(&value);
                self.log(&magnitude)
            } else {
                value
            };
        }
        let batch = self.batch_rank(input);
        let axes = self.axes_of(input);
        let scalar_axes = Axes {
            batch,
            layers: vec![],
        };
        let parity = self.constant(1.0, tensor(shape(&input.ty)[..batch].to_vec()));
        let parity = self.axes_view(&parity, scalar_axes.clone());
        let rows = self.axis_indices(input, batch);
        let cols = self.axis_indices(input, batch + 1);
        let mut permutation = (0..batch as u64).collect::<Vec<_>>();
        permutation.extend([batch as u64 + 1, batch as u64]);
        let result = self.matrix_loop(n - 1, &[input.clone(), parity], |e, index, args| {
            let a = e.axes_view(&args[0], axes.clone());
            let parity = e.axes_view(&args[1], scalar_axes.clone());
            let row_active = e.compare("GE", &rows, index);
            let col_active = e.compare("GE", &cols, index);
            let active = e.and(&row_active, &col_active);
            let magnitude = e.abs(&a);
            let infinity = e.inf_like(&a);
            let minus_inf = e.neg(&infinity);
            let magnitude = e.select(&active, &magnitude, &minus_inf);
            let largest = e.reduce_max(&magnitude);
            let candidates = e.compare("EQ", &magnitude, &largest);
            let limit = e.int_value_const(n as i64);
            let candidate_rows = e.select(&candidates, &rows, &limit);
            let pivot_row = e.reduce_axis_lit(
                "stablehlo.minimum",
                &n.to_string(),
                &candidate_rows,
                batch + 1,
            );
            let pivot_row =
                e.reduce_axis_lit("stablehlo.minimum", &n.to_string(), &pivot_row, batch);
            let pivot_rows = e.compare("EQ", &rows, &pivot_row);
            let candidates = e.and(&candidates, &pivot_rows);
            let candidate_cols = e.select(&candidates, &cols, &limit);
            let pivot_col = e.reduce_axis_lit(
                "stablehlo.minimum",
                &n.to_string(),
                &candidate_cols,
                batch + 1,
            );
            let pivot_col =
                e.reduce_axis_lit("stablehlo.minimum", &n.to_string(), &pivot_col, batch);
            let row_changed = e.compare("NE", &pivot_row, index);
            let col_changed = e.compare("NE", &pivot_col, index);
            let negative = e.neg(&parity);
            let parity = e.select(&row_changed, &negative, &parity);
            let negative = e.neg(&parity);
            let parity = e.select(&col_changed, &negative, &parity);
            let at_row = e.compare("EQ", &rows, index);
            let a = e.swap_rows(&a, &at_row, &pivot_rows);
            let a = e.transpose(&a, &permutation);
            let a = e.axes_view(&a, axes.clone());
            let pivot_rows = e.compare("EQ", &rows, &pivot_col);
            let a = e.swap_rows(&a, &at_row, &pivot_rows);
            let a = e.transpose(&a, &permutation);
            let a = e.axes_view(&a, axes.clone());
            let zero = e.constant_like(0.0, &a);
            let at_col = e.compare("EQ", &cols, index);
            let column = e.select(&at_col, &a, &zero);
            let column = e.reduce_sum_last_axis(&column);
            let row = e.selected_row(&a, &at_row);
            let indices = e.axis_indices(&row, batch);
            let at_pivot = e.compare("EQ", &indices, index);
            let zero = e.constant_like(0.0, &row);
            let pivot = e.select(&at_pivot, &row, &zero);
            let pivot = e.reduce_sum_last_axis(&pivot);
            let pivot_zero = e.scalar(0.0);
            let is_zero = e.compare("EQ", &pivot, &pivot_zero);
            let one = e.scalar(1.0);
            let safe_pivot = e.select(&is_zero, &one, &pivot);
            let below = e.compare("GT", &indices, index);
            let column = e.select(&below, &column, &zero);
            let factors = e.div(&column, &safe_pivot);
            let factors = e.matrix_expand(&factors, &a, batch);
            let row = e.matrix_expand(&row, &a, batch + 1);
            let change = e.mul(&factors, &row);
            vec![e.sub(&a, &change), parity]
        });
        let upper = self.axes_view(&result[0], axes);
        let parity = self.axes_view(&result[1], scalar_axes.clone());
        let diagonal = self.diag(&upper);
        let magnitude = self.abs(&diagonal);
        if logarithm {
            let logs = self.log(&magnitude);
            return self.reduce_sum(&logs);
        }
        let indices = self.axis_indices(&diagonal, batch);
        let last = self.int_value_const(n as i64 - 1);
        let at_last = self.compare("EQ", &indices, &last);
        let one = self.scalar(1.0);
        let zero = self.scalar(0.0);
        let prefix = self.select(&at_last, &one, &diagonal);
        let nonzero = self.compare("NE", &prefix, &zero);
        let regular_prefix = self.reduce_boolean("stablehlo.and", "true", &nonzero);
        let last_value = self.select(&at_last, &diagonal, &zero);
        let last_value = self.reduce_sum(&last_value);
        let nonzero_last = self.compare("NE", &last_value, &zero);
        let nonsingular = self.and(&regular_prefix, &nonzero_last);
        let negative = self.scalar(-1.0);
        let is_negative = self.compare("LT", &prefix, &zero);
        let signs = self.select(&is_negative, &negative, &one);
        let prefix_sign = self.reduce_full("stablehlo.multiply", "1.000000e+00", &signs);
        let prefix_sign = self.mul(&parity, &prefix_sign);
        let prefix = self.abs(&prefix);
        let prefix = self.select(&regular_prefix, &prefix, &one);
        let logs = self.log(&prefix);
        let prefix_log = self.reduce_sum(&logs);
        let last_magnitude = self.abs(&last_value);
        let last_magnitude = self.select(&nonsingular, &last_magnitude, &one);
        let last_log = self.log(&last_magnitude);
        let log_magnitude = self.add(&prefix_log, &last_log);
        let log_magnitude = self.select(&nonsingular, &log_magnitude, &zero);
        let magnitude = self.exp(&log_magnitude);
        let negative_last = self.compare("LT", &last_value, &zero);
        let last_sign = self.select(&negative_last, &negative, &one);
        let sign = self.mul(&prefix_sign, &last_sign);
        let determinant = self.mul(&sign, &magnitude);

        // At corank one, retain the final pivot's derivative. Equal bounded
        // factors avoid overflowing a prefix product with finite cofactors.
        let zero_last = self.compare("EQ", &last_value, &zero);
        let corank_one = self.and(&regular_prefix, &zero_last);
        let prefix_log = self.select(&corank_one, &prefix_log, &zero);
        let steps = 2 * (n - 1);
        let divisor = self.scalar(steps as f64);
        let mean_log = self.div(&prefix_log, &divisor);
        let factor = self.exp(&mean_log);
        let initial = self.select(&corank_one, &last_value, &zero);
        let singular = self.matrix_loop(steps, &[initial], |e, _, args| {
            let value = e.axes_view(&args[0], scalar_axes.clone());
            vec![e.mul(&value, &factor)]
        });
        let singular = self.axes_view(&singular[0], scalar_axes);
        let singular = self.mul(&prefix_sign, &singular);
        self.select(&nonsingular, &determinant, &singular)
    }

    /// Partial pivoting handles indefinite metrics and zero leading pivots.
    /// Two rolled loops avoid code growth and unsupported solve adjoints.
    pub(crate) fn matrix_inverse(&mut self, id: NodeId, input: &Value) -> Result<Value, EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(input) else {
            return Err(EmitError::at(id, "matrix inverse requires a square matrix"));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(
                id,
                "matrix inverse requires a static square matrix",
            ));
        };
        if n != m || *n == 0 || *n > i32::MAX as u64 {
            return Err(EmitError::at(
                id,
                "matrix inverse requires a nonempty square matrix",
            ));
        }
        let n = *n;
        let input = self.convert(input, ElemKind::Real);
        if let Some(diagonal) = self.constant_diagonal(&input) {
            let rows = diagonal
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    let row = (0..diagonal.len())
                        .map(|j| self.scalar(if i == j { 1.0 / d } else { 0.0 }))
                        .collect::<Vec<_>>();
                    self.vector(&row)
                })
                .collect::<Vec<_>>();
            let result = self.vector(&rows);
            return Ok(self.axes_view(&result, self.axes_of(&input)));
        }
        let batch = self.batch_rank(&input);
        let axes = self.axes_of(&input);
        let rows = self.axis_indices(&input, batch);
        let cols = self.axis_indices(&input, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let reduced = self.matrix_loop(n, &[input, identity], |e, index, args| {
            let a = e.axes_view(&args[0], axes.clone());
            let b = e.axes_view(&args[1], axes.clone());
            let at_row = e.compare("EQ", &rows, index);
            let at_col = e.compare("EQ", &cols, index);
            let zero = e.constant_like(0.0, &a);
            let column = e.select(&at_col, &a, &zero);
            let column = e.reduce_sum_last_axis(&column);
            let indices = e.axis_indices(&column, batch);
            let eligible = e.compare("GE", &indices, index);
            let magnitude = e.abs(&column);
            let infinity = e.inf_like(&column);
            let minus_inf = e.neg(&infinity);
            let magnitude = e.select(&eligible, &magnitude, &minus_inf);
            let largest = e.reduce_max(&magnitude);
            let candidates = e.compare("EQ", &magnitude, &largest);
            let limit = e.int_value_const(n as i64);
            let candidates = e.select(&candidates, &indices, &limit);
            let pivot = e.reduce_axis_lit("stablehlo.minimum", &n.to_string(), &candidates, batch);
            let pivot = e.fill_cell(&pivot, tensor(vec![Some(n), Some(n)]));
            let pivot_rows = e.compare("EQ", &rows, &pivot);
            let a = e.swap_rows(&a, &at_row, &pivot_rows);
            let b = e.swap_rows(&b, &at_row, &pivot_rows);
            let column = e.select(&at_col, &a, &zero);
            let column = e.reduce_sum_last_axis(&column);
            let row_a = e.selected_row(&a, &at_row);
            let row_b = e.selected_row(&b, &at_row);
            let indices = e.axis_indices(&row_a, batch);
            let at_pivot = e.compare("EQ", &indices, index);
            let row_zero = e.constant_like(0.0, &row_a);
            let pivot = e.select(&at_pivot, &row_a, &row_zero);
            let pivot = e.reduce_sum_last_axis(&pivot);
            let below = e.compare("GT", &indices, index);
            let column = e.select(&below, &column, &row_zero);
            let factors = e.div(&column, &pivot);
            let factors = e.matrix_expand(&factors, &a, batch);
            let row_a = e.matrix_expand(&row_a, &a, batch + 1);
            let row_b = e.matrix_expand(&row_b, &b, batch + 1);
            let change_a = e.mul(&factors, &row_a);
            let change_b = e.mul(&factors, &row_b);
            vec![e.sub(&a, &change_a), e.sub(&b, &change_b)]
        });
        let a = self.axes_view(&reduced[0], axes.clone());
        let b = self.axes_view(&reduced[1], axes.clone());
        let zero = self.constant_like(0.0, &a);
        let result = self.matrix_loop(n, &[zero], |e, index, args| {
            let last = e.int_value_const(n as i64 - 1);
            let k = e.sub(&last, index);
            let at_row = e.compare("EQ", &rows, &k);
            let row_a = e.selected_row(&a, &at_row);
            let row_b = e.selected_row(&b, &at_row);
            let indices = e.axis_indices(&row_a, batch);
            let at_pivot = e.compare("EQ", &indices, &k);
            let zero = e.constant_like(0.0, &row_a);
            let pivot = e.select(&at_pivot, &row_a, &zero);
            let pivot = e.reduce_sum_last_axis(&pivot);
            let above = e.compare("GT", &indices, &k);
            let offdiag = e.select(&above, &row_a, &zero);
            let offdiag = e.matrix_expand(&offdiag, &a, batch);
            let x = e.axes_view(&args[0], axes.clone());
            let products = e.mul(&offdiag, &x);
            let sum = e.reduce_axis("stablehlo.add", "0.000000e+00", &products, batch);
            let residual = e.sub(&row_b, &sum);
            let row = e.div(&residual, &pivot);
            let row = e.matrix_expand(&row, &x, batch + 1);
            vec![e.select(&at_row, &row, &x)]
        });
        Ok(self.axes_view(&result[0], axes))
    }
}
