//! Adaptive cubature for a finite rectangle. The tensor rule uses the same
//! Gauss and Kronrod nodes as scalar quadrature. Directional discrepancies
//! select the split axis; the selected mesh is replayed for differentiation.

use super::*;

const NODES_PER_PANEL: u32 = (NODES.len() * NODES.len()) as u32;

struct Integrand2 {
    body: NodeId,
    points: [NodeId; 2],
    kind: IntegralKind,
    origins: [Value; 2],
    widths: [Value; 2],
}

struct TensorRule {
    nodes: [Value; 2],
    kronrod: Value,
    mixed: [Value; 2],
    gauss: Value,
}

struct Panel {
    log_mass: Value,
    log_error: Value,
    split_x: Value,
    valid: Value,
    linear: Value,
}

impl TensorRule {
    fn new(e: &mut Emitter<'_>) -> Self {
        let mut product = |x: &[f64], y: &[f64]| {
            let weights = x.iter().flat_map(|a| y.iter().map(move |b| a * b));
            e.quadrature_constants(&weights.collect::<Vec<_>>())
        };
        let ones = [1.0; NODES.len()];
        Self {
            nodes: [product(&NODES, &ones), product(&ones, &NODES)],
            kronrod: product(&KRONROD, &KRONROD),
            mixed: [product(&GAUSS, &KRONROD), product(&KRONROD, &GAUSS)],
            gauss: product(&GAUSS, &GAUSS),
        }
    }
}

impl Emitter<'_> {
    pub(super) fn lower_rectangle_integral(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        kind: IntegralKind,
    ) -> Result<Value, EmitError> {
        let options = self.checked_integration(id)?;
        if options.max_intervals > i32::MAX as u32 / NODES_PER_PANEL {
            return Err(EmitError::at(
                id,
                "rectangle integration node budget exceeds int32 range",
            ));
        }
        let points = [args[1], args[4]];
        for point in points {
            self.integral_breakpoints(args[0], point, false)?;
        }

        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let infinity = self.scalar(f64::INFINITY);
        let mut valid_bounds = self.bool_value_const(true);
        let mut empty = self.bool_value_const(false);
        let mut bounds = Vec::with_capacity(2);
        for (lo, hi) in [(args[2], args[3]), (args[5], args[6])] {
            let lo = self.lower_node(lo)?;
            let hi = self.lower_node(hi)?;
            if lo.ty != MlirTy::Scalar || hi.ty != MlirTy::Scalar {
                return Err(EmitError::at(
                    id,
                    "rectangle integration bounds must be scalar",
                ));
            }
            let lo = self.convert(&lo, ElemKind::Real);
            let hi = self.convert(&hi, ElemKind::Real);
            let width = self.sub(&hi, &lo);
            let abs_lo = self.abs(&lo);
            let finite_lo = self.compare("LT", &abs_lo, &infinity);
            let finite_width = self.compare("LT", &width, &infinity);
            let ordered = self.compare("GE", &width, &zero);
            let finite = self.and(&finite_lo, &finite_width);
            let valid = self.and(&finite, &ordered);
            valid_bounds = self.and(&valid_bounds, &valid);
            let collapsed = self.compare("EQ", &width, &zero);
            empty = self.or(&empty, &collapsed);
            let positive = self.compare("GT", &width, &zero);
            let active = self.and(&valid, &positive);
            bounds.push((
                self.select(&active, &lo, &zero),
                self.select(&active, &width, &one),
            ));
        }
        let [(x, dx), (y, dy)] = bounds.try_into().unwrap_or_else(|_| unreachable!());
        let integrand = Integrand2 {
            body: args[0],
            points,
            kind,
            origins: [x, y],
            widths: [dx, dy],
        };
        let rule = TensorRule::new(self);
        let unit = [zero.clone(), one.clone(), zero.clone(), one.clone()];
        let initial_panel = self.rectangle_panel(&integrand, &rule, &unit)?;
        let count = self.int_value_const(1);
        let first = self.int_value_const(0);
        let capacity = self.int_value_const(options.max_intervals as i64);
        let buffer_ty = MlirTy::Ranked(vec![Some(options.max_intervals as u64)]);
        let start = self.constant(0.0, buffer_ty.clone());
        let end = self.constant(1.0, buffer_ty.clone());
        let inactive = self.constant(f64::NEG_INFINITY, buffer_ty.clone());
        let masses = self.scan_update(&inactive, &initial_panel.log_mass, &first);
        let errors = self.scan_update(&inactive, &initial_panel.log_error, &first);
        let choose_x = self.convert(&start, ElemKind::Bool);
        let choose_x = self.scan_update(&choose_x, &initial_panel.split_x, &first);
        let nonempty = self.not(&empty);
        let active = self.and(&valid_bounds, &nonempty);
        let valid = self.and(&active, &initial_panel.valid);
        let initial = [
            count,
            start.clone(),
            end.clone(),
            start,
            end,
            masses,
            errors,
            choose_x,
            initial_panel.log_mass,
            initial_panel.log_error,
            valid,
        ];
        let types = initial
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let log_rtol = self.scalar(options.rtol.ln());
        let log_atol = self.scalar(options.atol.ln());
        let result = self.try_while_loop(
            &initial,
            &types,
            |e, state| {
                let relative = e.add(&state[8], &log_rtol);
                let tolerance = e.max(&relative, &log_atol);
                let inaccurate = e.compare("GT", &state[9], &tolerance);
                let room = e.compare("LT", &state[0], &capacity);
                let more = e.and(&inaccurate, &room);
                e.and(&more, &state[10])
            },
            |e, state| {
                let [count, x0, x1, y0, y1, masses, errors, choose_x, _, _, valid] = state else {
                    unreachable!()
                };
                let worst = e.reduce_max(errors);
                let indices = e.axis_indices(errors, 0);
                let is_worst = e.compare("EQ", errors, &worst);
                let candidates = e.select(&is_worst, &indices, &capacity);
                let index = e.reduce_axis_lit(
                    "stablehlo.minimum",
                    &options.max_intervals.to_string(),
                    &candidates,
                    0,
                );
                let parent = [x0, x1, y0, y1].map(|edge| e.scan_slice(edge, &index));
                let [x0p, x1p, y0p, y1p] = &parent;
                let split_x = e.scan_slice(choose_x, &index);
                let half = e.scalar(0.5);
                let xmid = e.add(x0p, x1p);
                let xmid = e.mul(&half, &xmid);
                let ymid = e.add(y0p, y1p);
                let ymid = e.mul(&half, &ymid);
                let low = e.select(&split_x, x0p, y0p);
                let high = e.select(&split_x, x1p, y1p);
                let mid = e.select(&split_x, &xmid, &ymid);
                let above = e.compare("GT", &mid, &low);
                let below = e.compare("LT", &mid, &high);
                let split = e.and(&above, &below);
                let mut valid = e.and(valid, &split);
                let left = [
                    x0p.clone(),
                    e.select(&split_x, &xmid, x1p),
                    y0p.clone(),
                    e.select(&split_x, y1p, &ymid),
                ];
                let right = [
                    e.select(&split_x, &xmid, x0p),
                    x1p.clone(),
                    e.select(&split_x, y0p, &ymid),
                    y1p.clone(),
                ];
                let mut edges = [x0.clone(), x1.clone(), y0.clone(), y1.clone()];
                let mut masses = masses.clone();
                let mut errors = errors.clone();
                let mut choose_x = choose_x.clone();
                // Score both children in one node batch.
                let children = std::array::from_fn(|axis| {
                    let pair = e.vector(&[left[axis].clone(), right[axis].clone()]);
                    e.rectangle_panel_axis(&pair)
                });
                let panel = e.rectangle_panel(&integrand, &rule, &children)?;
                for (i, (child, slot)) in [(&left, &index), (&right, count)].into_iter().enumerate()
                {
                    let mut element = |values: &Value| {
                        let i = i as u64;
                        let lane = e.slice(values, &[i], &[i + 1], &[1]);
                        e.reshape_axes(&lane, MlirTy::Scalar, Axes::default())
                    };
                    let (mass, error, split_x, child_valid) = (
                        element(&panel.log_mass),
                        element(&panel.log_error),
                        element(&panel.split_x),
                        element(&panel.valid),
                    );
                    for (buffer, coordinate) in edges.iter_mut().zip(child) {
                        *buffer = e.scan_update(buffer, coordinate, slot);
                    }
                    masses = e.scan_update(&masses, &mass, slot);
                    errors = e.scan_update(&errors, &error, slot);
                    choose_x = e.scan_update(&choose_x, &split_x, slot);
                    valid = e.and(&valid, &child_valid);
                }
                let one = e.int_value_const(1);
                let count = e.add(count, &one);
                let total = e.integral_logsum(&masses);
                let error = e.integral_logsum(&errors);
                let mut next = vec![count];
                next.extend(edges);
                next.extend([masses, errors, choose_x, total, error, valid]);
                Ok(next)
            },
        )?;
        let relative = self.add(&result[8], &log_rtol);
        let tolerance = self.max(&relative, &log_atol);
        let converged = self.compare("LE", &result[9], &tolerance);
        let valid = self.and(&result[10], &converged);
        let empty_value = self.scalar(if kind == IntegralKind::Mass {
            0.0
        } else {
            f64::NEG_INFINITY
        });
        // The mesh carries only dyadic coordinates. Apply the physical bounds
        // again outside the adaptive loop so AD sees their derivatives.
        let value = if self.restrict_enzyme_compatible || kind == IntegralKind::Mass {
            let edges = [1, 2, 3, 4].map(|i| self.rectangle_panel_axis(&result[i]));
            let panel = self.rectangle_panel(&integrand, &rule, &edges)?;
            let masses = if kind == IntegralKind::Mass {
                panel.linear
            } else {
                panel.log_mass
            };
            let masses = self.axes_view(
                &masses,
                Axes {
                    batch: 0,
                    layers: vec![1],
                },
            );
            let indices = self.axis_indices(&masses, 0);
            let used = self.compare("LT", &indices, &result[0]);
            let masses = self.select(&used, &masses, &empty_value);
            if kind == IntegralKind::Mass {
                self.reduce_sum(&masses)
            } else {
                self.integral_logsum(&masses)
            }
        } else {
            result[8].clone()
        };
        let nan = self.scalar(f64::NAN);
        let value = self.select(&valid, &value, &nan);
        let empty = self.and(&valid_bounds, &empty);
        Ok(self.select(&empty, &empty_value, &value))
    }

    fn rectangle_panel_axis(&mut self, value: &Value) -> Value {
        self.axes_view(
            value,
            Axes {
                batch: 1,
                layers: vec![],
            },
        )
    }

    fn rectangle_panel(
        &mut self,
        integrand: &Integrand2,
        rule: &TensorRule,
        edges: &[Value; 4],
    ) -> Result<Panel, EmitError> {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let half = self.scalar(0.5);
        let neg_inf = self.scalar(f64::NEG_INFINITY);
        let infinity = self.scalar(f64::INFINITY);
        let mut interior = self.bool_value_const(true);
        let mut jacobian = zero.clone();
        let mut log_radius = zero.clone();
        let mut panel_widths = Vec::with_capacity(2);
        let mut points = Vec::with_capacity(2);
        for axis in 0..2 {
            let [a, b] = [&edges[axis * 2], &edges[axis * 2 + 1]];
            let width = self.sub(b, a);
            let radius = self.mul(&width, &half);
            let center = self.add(a, &radius);
            let displacement = self.mul(&radius, &rule.nodes[axis]);
            let t = self.add(&center, &displacement);
            let physical_width = &integrand.widths[axis];
            let origin = &integrand.origins[axis];
            let physical_offset = self.mul(&t, physical_width);
            let physical = self.add(origin, &physical_offset);
            let end = self.add(origin, physical_width);
            let above = self.compare("GT", &physical, origin);
            let below = self.compare("LT", &physical, &end);
            let inside = self.and(&above, &below);
            interior = self.and(&interior, &inside);
            let log_width = self.log(physical_width);
            jacobian = self.add(&jacobian, &log_width);
            let log = self.log(&radius);
            log_radius = self.add(&log_radius, &log);
            panel_widths.push(width);
            points.push((integrand.points[axis], physical));
        }
        let body = self.quadrature_body::<2>(
            integrand.body,
            points.try_into().unwrap_or_else(|_| unreachable!()),
        )?;
        let (log, raw_valid, linear) = if integrand.kind == IntegralKind::Mass {
            let (log, valid, linear) =
                self.quadrature_raw_mass(&body, &jacobian, &log_radius, &rule.kronrod);
            (log, Some(valid), linear)
        } else {
            (body, None, zero.clone())
        };
        let finite = self.compare("LT", &log, &infinity);
        let valid = self.and(&finite, &interior);
        let valid = raw_valid
            .as_ref()
            .map_or(valid.clone(), |raw| self.and(&valid, raw));
        let valid = self.reduce_all(&valid);
        let logs = self.select(&finite, &log, &neg_inf);
        let max = self.reduce_max(&logs);
        let nonzero = self.compare("GT", &max, &neg_inf);
        let rounded = self.floor(&max);
        let shift = self.select(&nonzero, &rounded, &zero);
        let shifted = self.sub(&logs, &shift);
        let values = self.exp(&shifted);
        let weighted = self.mul(&values, &rule.kronrod);
        let k = self.reduce_sum(&weighted);
        let discrepancy = |e: &mut Self, weights: &Value| {
            let weighted = e.mul(&values, weights);
            let reduced = e.reduce_sum(&weighted);
            let difference = e.sub(&k, &reduced);
            e.abs(&difference)
        };
        let x_error = discrepancy(self, &rule.mixed[0]);
        let y_error = discrepancy(self, &rule.mixed[1]);
        let joint_error = discrepancy(self, &rule.gauss);
        let directional = self.max(&x_error, &y_error);
        let difference = self.max(&joint_error, &directional);
        let quarter = self.scalar(0.25);
        let mean = self.mul(&k, &quarter);
        let deviation = self.sub(&values, &mean);
        let deviation = self.abs(&deviation);
        let deviation = self.mul(&deviation, &rule.kronrod);
        let variation = self.reduce_sum(&deviation);
        let error = self.quadrature_error(&difference, &variation, &k);
        let scale = self.add(&shift, &jacobian);
        let scale = self.add(&scale, &log_radius);
        let mass = self.select(&nonzero, &k, &one);
        let mass = self.log(&mass);
        let mass = self.add(&scale, &mass);
        let error = self.select(&nonzero, &error, &one);
        let error = self.log(&error);
        let error = self.add(&scale, &error);
        let x_worse = self.compare("GT", &x_error, &y_error);
        let tied = self.compare("EQ", &x_error, &y_error);
        let wider = self.compare("GE", &panel_widths[0], &panel_widths[1]);
        let split_x = self.select(&tied, &wider, &x_worse);
        Ok(Panel {
            log_mass: self.select(&nonzero, &mass, &neg_inf),
            log_error: self.select(&nonzero, &error, &neg_inf),
            split_x,
            valid,
            linear,
        })
    }
}
