//! Adaptive positive scalar quadrature. The embedded Gauss 7 / Kronrod 15
//! rule and variation-based error estimate follow QUADPACK's DQK15.
//! Like Quadax's direct adjoint, AD differentiates a fresh evaluation on the
//! selected mesh, with live interval bounds. The mesh consists of dyadic unit
//! intervals selected by integer indices, so it has no parameter derivatives.
//! The primal error estimate does not bound AD error.
//! All panels and totals stay in log space, including the error estimates.

use super::*;

const NODES: [f64; 15] = [
    -0.9914553711208126,
    -0.9491079123427585,
    -0.8648644233597691,
    -0.7415311855993945,
    -0.5860872354676911,
    -0.4058451513773972,
    -0.2077849550078985,
    0.0,
    0.2077849550078985,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993945,
    0.8648644233597691,
    0.9491079123427585,
    0.9914553711208126,
];
const KRONROD: [f64; 15] = [
    0.02293532201052922,
    0.06309209262997855,
    0.1047900103222502,
    0.1406532597155259,
    0.1690047266392679,
    0.1903505780647854,
    0.2044329400752989,
    0.2094821410847278,
    0.2044329400752989,
    0.1903505780647854,
    0.1690047266392679,
    0.1406532597155259,
    0.1047900103222502,
    0.06309209262997855,
    0.02293532201052922,
];
const GAUSS: [f64; 15] = [
    0.0,
    0.1294849661688697,
    0.0,
    0.2797053914892767,
    0.0,
    0.3818300505051189,
    0.0,
    0.4179591836734694,
    0.0,
    0.3818300505051189,
    0.0,
    0.2797053914892767,
    0.0,
    0.1294849661688697,
    0.0,
];

impl Emitter<'_> {
    pub(crate) fn lower_integral(
        &mut self,
        id: NodeId,
        args: &[NodeId],
    ) -> Result<Value, EmitError> {
        let options = self
            .integration
            .ok_or_else(|| EmitError::at(id, "numerical integration requires explicit opt-in"))?;
        if args.len() != 4 || !self.broadcast_frame.is_empty() {
            return Err(EmitError::at(
                id,
                "numerical integration requires an unbatched scalar integral",
            ));
        }
        if !options.rtol.is_finite()
            || options.rtol < 0.0
            || !options.atol.is_finite()
            || options.atol < 0.0
            || (options.rtol == 0.0 && options.atol == 0.0)
            || !(2..=i32::MAX as u32).contains(&options.max_intervals)
        {
            return Err(EmitError::at(
                id,
                "integration requires finite nonnegative tolerances, at least one positive tolerance, and 2..=i32::MAX intervals",
            ));
        }
        let lo = self.lower_node(args[2])?;
        let hi = self.lower_node(args[3])?;
        if lo.ty != MlirTy::Scalar || hi.ty != MlirTy::Scalar {
            return Err(EmitError::at(
                id,
                "numerical integration bounds must be scalar",
            ));
        }
        let lo = self.convert(&lo, ElemKind::Real);
        let hi = self.convert(&hi, ElemKind::Real);
        let inf = self.scalar(f64::INFINITY);
        let neg_inf = self.scalar(f64::NEG_INFINITY);
        let nan = self.scalar(f64::NAN);
        let izero = self.int_value_const(0);
        let ione = self.int_value_const(1);
        let capacity = self.int_value_const(options.max_intervals as i64);
        let buffer_ty = MlirTy::Ranked(vec![Some(options.max_intervals as u64)]);
        let left = self.constant(0.0, buffer_ty.clone());
        let right = self.constant(1.0, buffer_ty.clone());
        let masses = self.constant(f64::NEG_INFINITY, buffer_ty.clone());
        let errors = self.scan_update(&masses, &inf, &izero);
        let valid = self.compare("LT", &lo, &hi);
        let log_rtol = self.scalar(options.rtol.ln());
        let log_atol = self.scalar(options.atol.ln());
        // The first iteration bisects the whole domain. Every later iteration
        // replaces the worst panel by two children. Inactive slots have zero mass.
        let initial = vec![
            ione.clone(),
            left,
            right,
            masses,
            errors,
            neg_inf.clone(),
            inf.clone(),
            valid,
        ];
        let tys = initial
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let result = self.try_while_loop(
            &initial,
            &tys,
            |e, s| {
                let relative = e.add(&s[5], &log_rtol);
                let tolerance = e.max(&relative, &log_atol);
                let inaccurate = e.compare("GT", &s[6], &tolerance);
                let room = e.compare("LT", &s[0], &capacity);
                let more = e.and(&inaccurate, &room);
                e.and(&more, &s[7])
            },
            |e, s| {
                let worst = e.reduce_max(&s[4]);
                let indices = e.axis_indices(&s[4], 0);
                let is_worst = e.compare("EQ", &s[4], &worst);
                let candidates = e.select(&is_worst, &indices, &capacity);
                let index = e.reduce_axis_lit(
                    "stablehlo.minimum",
                    &options.max_intervals.to_string(),
                    &candidates,
                    0,
                );
                let a = e.scan_slice(&s[1], &index);
                let b = e.scan_slice(&s[2], &index);
                let half = e.scalar(0.5);
                let sum = e.add(&a, &b);
                let middle = e.mul(&sum, &half);
                let above = e.compare("GT", &middle, &a);
                let below = e.compare("LT", &middle, &b);
                let split = e.and(&above, &below);
                let valid = e.and(&s[7], &split);
                let mut left = s[1].clone();
                let mut right = s[2].clone();
                let mut masses = s[3].clone();
                let mut errors = s[4].clone();
                let mut valid = valid;
                // A fixed two-panel rule is emitted directly; quadrature nodes are
                // batched. Only the adaptive worklist needs a runtime loop.
                for (child_a, child_b, target) in [(&a, &middle, &index), (&middle, &b, &s[0])] {
                    let (mass, error, panel_valid) =
                        e.quadrature_panel(args[0], args[1], &lo, &hi, child_a, child_b)?;
                    left = e.scan_update(&left, child_a, target);
                    right = e.scan_update(&right, child_b, target);
                    masses = e.scan_update(&masses, &mass, target);
                    errors = e.scan_update(&errors, &error, target);
                    valid = e.and(&valid, &panel_valid);
                }
                let total = e.integral_logsum(&masses);
                let error = e.integral_logsum(&errors);
                Ok(vec![
                    e.add(&s[0], &ione),
                    left,
                    right,
                    masses,
                    errors,
                    total,
                    error,
                    valid,
                ])
            },
        )?;
        let relative = self.add(&result[5], &log_rtol);
        let tolerance = self.max(&relative, &log_atol);
        let converged = self.compare("LE", &result[6], &tolerance);
        let valid = self.and(&converged, &result[7]);
        // Enzyme cannot tape an adaptive while. Reevaluate the selected mesh
        // outside it, as Quadax's direct adjoint does. Only discrete decisions
        // depend on parameters inside the loop; the live physical bounds are
        // applied again here. Unused slots retain the valid initial interval.
        let value = if self.restrict_enzyme_compatible {
            let panel_axes = Axes {
                batch: 1,
                layers: vec![],
            };
            let left = self.axes_view(&result[1], panel_axes.clone());
            let right = self.axes_view(&result[2], panel_axes);
            let (masses, _, _) =
                self.quadrature_panel(args[0], args[1], &lo, &hi, &left, &right)?;
            let masses = self.axes_view(
                &masses,
                Axes {
                    batch: 0,
                    layers: vec![1],
                },
            );
            let indices = self.axis_indices(&masses, 0);
            let active = self.compare("LT", &indices, &result[0]);
            let masses = self.select(&active, &masses, &neg_inf);
            self.integral_logsum(&masses)
        } else {
            result[5].clone()
        };
        let value = self.select(&valid, &value, &nan);
        let empty = self.compare("EQ", &lo, &hi);
        let finite = self.abs(&lo);
        let finite = self.compare("LT", &finite, &inf);
        let empty = self.and(&empty, &finite);
        // No evaluations are needed for an empty finite interval.
        Ok(self.select(&empty, &neg_inf, &value))
    }

    fn integral_logsum(&mut self, values: &Value) -> Value {
        let maximum = self.reduce_max(values);
        let neg_inf = self.scalar(f64::NEG_INFINITY);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let nonzero = self.compare("GT", &maximum, &neg_inf);
        // Any finite shift gives the same log sum. Rounding it also gives AD
        // an inactive scale, avoiding Enzyme's maximum-adjoint loop bug.
        let rounded = self.floor(&maximum);
        let shift = self.select(&nonzero, &rounded, &zero);
        let shifted = self.sub(values, &shift);
        let weights = self.exp(&shifted);
        let total = self.reduce_sum(&weights);
        let safe = self.select(&nonzero, &total, &one);
        let log = self.log(&safe);
        let log = self.add(&shift, &log);
        self.select(&nonzero, &log, &neg_inf)
    }

    fn quadrature_panel(
        &mut self,
        body: NodeId,
        point: NodeId,
        lo: &Value,
        hi: &Value,
        a: &Value,
        b: &Value,
    ) -> Result<(Value, Value, Value), EmitError> {
        let nodes = self.quadrature_constants(&NODES);
        let kronrod = self.quadrature_constants(&KRONROD);
        let gauss = self.quadrature_constants(&GAUSS);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let half = self.scalar(0.5);
        let width = self.sub(b, a);
        let radius = self.mul(&width, &half);
        let sum = self.add(a, b);
        let center = self.mul(&sum, &half);
        let offset = self.mul(&radius, &nodes);
        let t = self.add(&center, &offset);
        let (x, jacobian) = self.quadrature_map(&t, lo, hi);
        let count = shape(&x.ty).iter().map(|d| d.unwrap()).product::<u64>();
        let x = self.reshape(&x, MlirTy::Ranked(vec![Some(count)]));
        let parent = std::mem::replace(&mut self.broadcast_frame, vec![Some(count)]);
        let x = self.axes_view(
            &x,
            Axes {
                batch: 1,
                layers: vec![],
            },
        );
        let log = self.lower_with_bindings(body, vec![(point, x)]);
        self.broadcast_frame = parent;
        let log = log?;
        if self.cell_ty(&log) != MlirTy::Scalar || log.elem != ElemKind::Real {
            return Err(EmitError::at(
                body,
                "numerical integrand must be a real scalar log density",
            ));
        }
        let log = if log.ty == MlirTy::Scalar {
            self.expand_axes(
                &log,
                &[],
                MlirTy::Ranked(vec![Some(count)]),
                Axes {
                    batch: 1,
                    layers: vec![],
                },
            )
        } else {
            log
        };
        let log = self.reshape(&log, t.ty.clone());
        let log = self.axes_view(&log, self.axes_of(&t));
        let log = self.add(&log, &jacobian);
        let inf = self.scalar(f64::INFINITY);
        let finite = self.compare("LT", &log, &inf);
        let lower = self.compare("GT", &t, &zero);
        let upper = self.compare("LT", &t, &one);
        let interior = self.and(&lower, &upper);
        let valid = self.and(&finite, &interior);
        let valid = self.reduce_all(&valid);
        let neg_inf = self.scalar(f64::NEG_INFINITY);
        let logs = self.select(&finite, &log, &neg_inf);
        let maximum = self.reduce_max(&logs);
        let neg_inf = self.scalar(f64::NEG_INFINITY);
        let nonzero = self.compare("GT", &maximum, &neg_inf);
        // Any finite shift gives the same log sum. Rounding it also gives AD
        // an inactive scale, avoiding Enzyme's maximum-adjoint loop bug.
        let rounded = self.floor(&maximum);
        let shift = self.select(&nonzero, &rounded, &zero);
        let shifted = self.sub(&logs, &shift);
        let values = self.exp(&shifted);
        let weighted = self.mul(&values, &kronrod);
        let k = self.reduce_sum(&weighted);
        let weighted = self.mul(&values, &gauss);
        let g = self.reduce_sum(&weighted);
        let mean = self.mul(&k, &half);
        let deviation = self.sub(&values, &mean);
        let deviation = self.abs(&deviation);
        let deviation = self.mul(&deviation, &kronrod);
        let variation = self.reduce_sum(&deviation);
        let difference = self.sub(&k, &g);
        let difference = self.abs(&difference);
        let changing = self.compare("GT", &variation, &zero);
        let disagree = self.compare("GT", &difference, &zero);
        let scalable = self.and(&changing, &disagree);
        let denominator = self.select(&scalable, &variation, &one);
        let ratio = self.div(&difference, &denominator);
        let ratio = self.select(&scalable, &ratio, &one);
        let ratio = self.min(&ratio, &one);
        let inflation = self.scalar(200.0);
        let ratio = self.mul(&inflation, &ratio);
        let ratio = self.min(&ratio, &one);
        let power = self.scalar(1.5);
        let ratio = self.pow(&ratio, &power);
        let rescaled = self.mul(&variation, &ratio);
        let error = self.select(&scalable, &rescaled, &difference);
        let epsilon = match self.dtype {
            Dtype::F32 => f32::EPSILON as f64,
            Dtype::F64 => f64::EPSILON,
        };
        let floor = self.scalar(50.0 * epsilon);
        let floor = self.mul(&floor, &k);
        let error = self.max(&error, &floor);
        let safe_k = self.select(&nonzero, &k, &one);
        let safe_error = self.select(&nonzero, &error, &one);
        let log_radius = self.log(&radius);
        let scale = self.add(&shift, &log_radius);
        let log_k = self.log(&safe_k);
        let log_error = self.log(&safe_error);
        let mass = self.add(&scale, &log_k);
        let error = self.add(&scale, &log_error);
        Ok((
            self.select(&nonzero, &mass, &neg_inf),
            self.select(&nonzero, &error, &neg_inf),
            valid,
        ))
    }

    fn quadrature_constants(&mut self, xs: &[f64]) -> Value {
        let values = xs.iter().map(|&x| self.scalar(x)).collect::<Vec<_>>();
        self.vector(&values)
    }

    /// Rational maps avoid cancellation near a finite endpoint. Keep every
    /// inactive branch finite so its zero adjoint cannot multiply an infinity.
    fn quadrature_map(&mut self, t: &Value, lo: &Value, hi: &Value) -> (Value, Value) {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let two = self.scalar(2.0);
        let inf = self.scalar(f64::INFINITY);
        let abs_lo = self.abs(lo);
        let abs_hi = self.abs(hi);
        let finite_lo = self.compare("LT", &abs_lo, &inf);
        let finite_hi = self.compare("LT", &abs_hi, &inf);
        let finite = self.and(&finite_lo, &finite_hi);
        let a = self.select(&finite_lo, lo, &zero);
        let b = self.select(&finite_hi, hi, &one);
        let width = self.sub(&b, &a);
        let width = self.select(&finite, &width, &one);
        let offset = self.mul(&width, t);
        let bounded = self.add(&a, &offset);
        let bounded_jacobian = self.log(&width);
        let u = self.sub(&one, t);
        let positive = self.div(t, &u);
        let negative = self.div(&u, t);
        let right = self.add(&a, &positive);
        let left = self.sub(&b, &negative);
        let whole = self.sub(&positive, &negative);
        let log_t = self.log(t);
        let log_u = self.log(&u);
        let negative_two = self.neg(&two);
        let right_jacobian = self.mul(&negative_two, &log_u);
        let left_jacobian = self.mul(&negative_two, &log_t);
        let shift = self.max(&right_jacobian, &left_jacobian);
        let shifted_right = self.sub(&right_jacobian, &shift);
        let shifted_left = self.sub(&left_jacobian, &shift);
        let weight_right = self.exp(&shifted_right);
        let weight_left = self.exp(&shifted_left);
        let sum = self.add(&weight_right, &weight_left);
        let log = self.log(&sum);
        let whole_jacobian = self.add(&shift, &log);
        let unbounded = self.select(&finite_lo, &right, &whole);
        let unbounded = self.select(&finite_hi, &left, &unbounded);
        let unbounded_jacobian = self.select(&finite_lo, &right_jacobian, &whole_jacobian);
        let unbounded_jacobian = self.select(&finite_hi, &left_jacobian, &unbounded_jacobian);
        (
            self.select(&finite, &bounded, &unbounded),
            self.select(&finite, &bounded_jacobian, &unbounded_jacobian),
        )
    }
}
