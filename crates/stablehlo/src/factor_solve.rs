//! Factor solves preserve vector/matrix RHS cells and complex arithmetic.

use super::*;

type Pair = (Value, Value);

impl Emitter<'_> {
    pub(super) fn finite_matrix_parts(&mut self, a: &Pair) -> Value {
        let infinity = self.scalar(f64::INFINITY);
        let ar = self.abs(&a.0);
        let ai = self.abs(&a.1);
        let re = self.compare("LT", &ar, &infinity);
        let im = self.compare("LT", &ai, &infinity);
        let finite = self.and(&re, &im);
        self.reduce_all(&finite)
    }

    pub(super) fn factor_matmul(&mut self, a: &Pair, b: &Pair) -> Pair {
        let rr = self.matmat(&a.0, &b.0);
        let ii = self.matmat(&a.1, &b.1);
        let ri = self.matmat(&a.0, &b.1);
        let ir = self.matmat(&a.1, &b.0);
        (self.sub(&rr, &ii), self.add(&ri, &ir))
    }

    pub(super) fn factor_apply(&mut self, a: &Pair, b: &Pair) -> Pair {
        if matches!(self.cell_ty(&b.0), MlirTy::Ranked(d) if d.len() == 2) {
            return self.factor_matmul(a, b);
        }
        let rr = self.matvec(&a.0, &b.0);
        let ii = self.matvec(&a.1, &b.1);
        let ri = self.matvec(&a.0, &b.1);
        let ir = self.matvec(&a.1, &b.0);
        (self.sub(&rr, &ii), self.add(&ri, &ir))
    }

    pub(super) fn factor_adjoint(&mut self, a: &Pair) -> Pair {
        let re = self.svd_transpose(&a.0);
        let im = self.svd_transpose(&a.1);
        (re, self.neg(&im))
    }

    pub(super) fn matrix_factor_solve(
        &mut self,
        id: NodeId,
        a: &Pair,
        b: &Pair,
        lower: &Value,
        unit_diagonal: &Value,
    ) -> Result<Pair, EmitError> {
        let MlirTy::Ranked(adims) = self.cell_ty(&a.0) else {
            return Err(EmitError::at(
                id,
                "factor solve requires a static square matrix",
            ));
        };
        let [Some(n), Some(m)] = adims.as_slice() else {
            return Err(EmitError::at(
                id,
                "factor solve requires static matrix dimensions",
            ));
        };
        let MlirTy::Ranked(bdims) = self.cell_ty(&b.0) else {
            return Err(EmitError::at(
                id,
                "factor solve requires a vector or matrix RHS",
            ));
        };
        if n != m || bdims.first() != Some(&Some(*n)) || !(1..=2).contains(&bdims.len()) {
            return Err(EmitError::at(
                id,
                "factor solve RHS rows must match the square matrix",
            ));
        }
        let (n, vector) = (*n, bdims.len() == 1);
        let b = if vector {
            let column = MlirTy::Ranked(vec![Some(n), Some(1)]);
            (
                self.reshape_cell(&b.0, column.clone()),
                self.reshape_cell(&b.1, column),
            )
        } else {
            b.clone()
        };
        let (ar, br) = self.broadcast_batches(&a.0, &b.0);
        let (ai, bi) = self.broadcast_batches(&a.1, &b.1);
        let a = (
            self.convert(&ar, ElemKind::Real),
            self.convert(&ai, ElemKind::Real),
        );
        let b = (
            self.convert(&br, ElemKind::Real),
            self.convert(&bi, ElemKind::Real),
        );
        let batch = self.batch_rank(&a.0);
        let axes = self.axes_of(&b.0);
        let arows = self.axis_indices(&a.0, batch);
        let acols = self.axis_indices(&a.0, batch + 1);
        let below = self.compare("GE", &arows, &acols);
        let above = self.compare("LE", &arows, &acols);
        let triangle = self.select(lower, &below, &above);
        let off_diagonal = self.compare("NE", &arows, &acols);
        let nonunit = self.not(unit_diagonal);
        let used = self.or(&nonunit, &off_diagonal);
        let used = self.and(&triangle, &used);
        let zeros = self.constant_like(0.0, &a.0);
        let used_a = (
            self.select(&used, &a.0, &zeros),
            self.select(&used, &a.1, &zeros),
        );
        let finite_a = self.finite_matrix_parts(&used_a);
        let finite_b = self.finite_matrix_parts(&b);
        let domain = self.and(&finite_a, &finite_b);
        self.record_runtime_error(&domain, 12);
        let rows = self.axis_indices(&b.0, batch);
        let zero = self.constant_like(0.0, &b.0);
        let scalar_axes = Axes {
            batch,
            layers: vec![],
        };
        let valid_zero = self.constant(
            0.0,
            super::batching::tensor(shape(&a.0.ty)[..batch].to_vec()),
        );
        let valid_zero = self.axes_view(&valid_zero, scalar_axes.clone());
        let valid = self.compare("EQ", &valid_zero, &valid_zero);
        let result = self.matrix_loop(
            n,
            MATRIX_UNROLL_LIMIT,
            &[zero.clone(), zero, valid],
            |e, index, state| {
                let last = e.int_value_const(n as i64 - 1);
                let backwards = e.sub(&last, index);
                let k = e.select(lower, index, &backwards);
                let row = (
                    e.indexed_matrix_axis(&a.0, &k, 0),
                    e.indexed_matrix_axis(&a.1, &k, 0),
                );
                let rhs = (
                    e.indexed_matrix_axis(&b.0, &k, 0),
                    e.indexed_matrix_axis(&b.1, &k, 0),
                );
                let indices = e.axis_indices(&row.0, batch);
                let at_pivot = e.compare("EQ", &indices, &k);
                let before = e.compare("LT", &indices, &k);
                let after = e.compare("GT", &indices, &k);
                let active = e.select(lower, &before, &after);
                let row_zero = e.constant_like(0.0, &row.0);
                let coefficients = (
                    e.select(&active, &row.0, &row_zero),
                    e.select(&active, &row.1, &row_zero),
                );
                let x = (
                    e.axes_view(&state[0], axes.clone()),
                    e.axes_view(&state[1], axes.clone()),
                );
                let coefficients = (
                    e.matrix_expand(&coefficients.0, &x.0, batch),
                    e.matrix_expand(&coefficients.1, &x.1, batch),
                );
                let rr = e.mul(&coefficients.0, &x.0);
                let ii = e.mul(&coefficients.1, &x.1);
                let ri = e.mul(&coefficients.0, &x.1);
                let ir = e.mul(&coefficients.1, &x.0);
                let re = e.sub(&rr, &ii);
                let im = e.add(&ri, &ir);
                let re = e.reduce_axis("stablehlo.add", "0.000000e+00", &re, batch);
                let im = e.reduce_axis("stablehlo.add", "0.000000e+00", &im, batch);
                let residual = (e.sub(&rhs.0, &re), e.sub(&rhs.1, &im));
                let pr = e.select(&at_pivot, &row.0, &row_zero);
                let pi = e.select(&at_pivot, &row.1, &row_zero);
                let pr = e.reduce_sum_last_axis(&pr);
                let pi = e.reduce_sum_last_axis(&pi);
                let one = e.scalar(1.0);
                let zero = e.scalar(0.0);
                let pr = e.select(unit_diagonal, &one, &pr);
                let pi = e.select(unit_diagonal, &zero, &pi);
                let abs_r = e.abs(&pr);
                let abs_i = e.abs(&pi);
                let scale = e.max(&abs_r, &abs_i);
                let nonzero = e.compare("GT", &scale, &zero);
                let infinity = e.scalar(f64::INFINITY);
                let finite = e.compare("LT", &scale, &infinity);
                let pivot_ok = e.and(&nonzero, &finite);
                let valid = e.axes_view(&state[2], scalar_axes.clone());
                let valid = e.and(&valid, &pivot_ok);
                let safe = e.select(&pivot_ok, &scale, &one);
                let pr = e.div(&pr, &safe);
                let pi = e.div(&pi, &safe);
                let rr = e.mul(&pr, &pr);
                let ii = e.mul(&pi, &pi);
                let denominator = e.add(&rr, &ii);
                let denominator = e.select(&pivot_ok, &denominator, &one);
                let rr = e.div(&residual.0, &safe);
                let ri = e.div(&residual.1, &safe);
                let ar = e.mul(&rr, &pr);
                let ai = e.mul(&ri, &pi);
                let br = e.mul(&ri, &pr);
                let bi = e.mul(&rr, &pi);
                let re = e.add(&ar, &ai);
                let im = e.sub(&br, &bi);
                let re = e.div(&re, &denominator);
                let im = e.div(&im, &denominator);
                let re = e.matrix_expand(&re, &x.0, batch + 1);
                let im = e.matrix_expand(&im, &x.1, batch + 1);
                let at_row = e.compare("EQ", &rows, &k);
                vec![
                    e.select(&at_row, &re, &x.0),
                    e.select(&at_row, &im, &x.1),
                    valid,
                ]
            },
        );
        let valid = self.axes_view(&result[2], scalar_axes);
        self.record_runtime_error(&valid, 4);
        let nan = self.scalar(f64::NAN);
        let re = self.axes_view(&result[0], axes.clone());
        let im = self.axes_view(&result[1], axes);
        let valid = self.and(&valid, &domain);
        let re = self.select(&valid, &re, &nan);
        let im = self.select(&valid, &im, &nan);
        Ok(if vector {
            let vector = MlirTy::Ranked(vec![Some(n)]);
            (
                self.reshape_cell(&re, vector.clone()),
                self.reshape_cell(&im, vector),
            )
        } else {
            (re, im)
        })
    }
}
