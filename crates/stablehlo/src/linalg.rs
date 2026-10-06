//! Pivoted elimination and tensor-axis contractions using portable StableHLO.

use super::batching::tensor;
use super::*;

impl Emitter<'_> {
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
        let counter = self.int_value_const(0);
        let inits = [counter.clone(), input, identity];
        let types = inits
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let reduced = self.while_loop(
            &inits,
            &types,
            |e, args| {
                let n = e.int_value_const(n as i64);
                e.compare("LT", &args[0], &n)
            },
            |e, args| {
                let a = e.axes_view(&args[1], axes.clone());
                let b = e.axes_view(&args[2], axes.clone());
                let at_row = e.compare("EQ", &rows, &args[0]);
                let at_col = e.compare("EQ", &cols, &args[0]);
                let zero = e.constant_like(0.0, &a);
                let column = e.select(&at_col, &a, &zero);
                let column = e.reduce_sum_last_axis(&column);
                let indices = e.axis_indices(&column, batch);
                let eligible = e.compare("GE", &indices, &args[0]);
                let magnitude = e.abs(&column);
                let infinity = e.inf_like(&column);
                let minus_inf = e.neg(&infinity);
                let magnitude = e.select(&eligible, &magnitude, &minus_inf);
                let largest = e.reduce_max(&magnitude);
                let candidates = e.compare("EQ", &magnitude, &largest);
                let limit = e.int_value_const(n as i64);
                let index = e.select(&candidates, &indices, &limit);
                let pivot = e.reduce_axis_lit("stablehlo.minimum", &n.to_string(), &index, batch);
                let pivot = e.fill_cell(&pivot, tensor(vec![Some(n), Some(n)]));
                let pivot_rows = e.compare("EQ", &rows, &pivot);
                let a = e.swap_rows(&a, &at_row, &pivot_rows);
                let b = e.swap_rows(&b, &at_row, &pivot_rows);
                let column = e.select(&at_col, &a, &zero);
                let column = e.reduce_sum_last_axis(&column);
                let row_a = e.selected_row(&a, &at_row);
                let row_b = e.selected_row(&b, &at_row);
                let indices = e.axis_indices(&row_a, batch);
                let at_pivot = e.compare("EQ", &indices, &args[0]);
                let row_zero = e.constant_like(0.0, &row_a);
                let pivot = e.select(&at_pivot, &row_a, &row_zero);
                let pivot = e.reduce_sum_last_axis(&pivot);
                let factors = e.div(&column, &pivot);
                let below = e.compare("GT", &indices, &args[0]);
                let factors = e.select(&below, &factors, &row_zero);
                let factors = e.matrix_expand(&factors, &a, batch);
                let row_a = e.matrix_expand(&row_a, &a, batch + 1);
                let row_b = e.matrix_expand(&row_b, &b, batch + 1);
                let change_a = e.mul(&factors, &row_a);
                let change_b = e.mul(&factors, &row_b);
                let one = e.int_value_const(1);
                vec![
                    e.add(&args[0], &one),
                    e.sub(&a, &change_a),
                    e.sub(&b, &change_b),
                ]
            },
        );
        let a = self.axes_view(&reduced[1], axes.clone());
        let b = self.axes_view(&reduced[2], axes.clone());
        let zero = self.constant_like(0.0, &a);
        let inits = [counter, zero];
        let types = inits
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let result = self.while_loop(
            &inits,
            &types,
            |e, args| {
                let n = e.int_value_const(n as i64);
                e.compare("LT", &args[0], &n)
            },
            |e, args| {
                let last = e.int_value_const(n as i64 - 1);
                let k = e.sub(&last, &args[0]);
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
                let x = e.axes_view(&args[1], axes.clone());
                let products = e.mul(&offdiag, &x);
                let sum = e.reduce_axis("stablehlo.add", "0.000000e+00", &products, batch);
                let residual = e.sub(&row_b, &sum);
                let row = e.div(&residual, &pivot);
                let row = e.matrix_expand(&row, &x, batch + 1);
                let one = e.int_value_const(1);
                vec![e.add(&args[0], &one), e.select(&at_row, &row, &x)]
            },
        );
        Ok(self.axes_view(&result[1], axes))
    }
}
