//! Adaptive positive scalar quadrature. The embedded Gauss 7 / Kronrod 15
//! rule and variation-based error estimate follow QUADPACK's DQK15.
//! Like Quadax's direct adjoint, AD differentiates a fresh evaluation on the
//! selected mesh, with live interval bounds and structural breakpoints. The
//! mesh uses dyadic coordinates within each segment, without parameter derivatives.
//! The primal error estimate does not bound AD error.
//! All panels and totals stay in log space, including the error estimates.

use super::*;

#[derive(Clone, PartialEq, Eq, Hash)]
enum IntegralInput {
    Node(NodeId),
    Column(NodeId, String),
}

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
        if args.len() != 4 || self.integrating {
            return Err(EmitError::at(
                id,
                "numerical integration requires a non-nested scalar integral",
            ));
        }
        let inputs = self.integral_inputs(args);
        if !inputs.is_empty() {
            return self.lower_batched_integral(id, args, &inputs);
        }
        let frame = std::mem::take(&mut self.broadcast_frame);
        let result = (|| {
            // Independent inner normalizers dominate adaptation and mesh replay.
            self.lower_integral_dependencies(args[0])?;
            self.integrating = true;
            self.lower_scalar_integral(id, args)
        })();
        self.integrating = false;
        self.broadcast_frame = frame;
        result
    }

    fn lower_scalar_integral(&mut self, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
        let options = self
            .integration
            .ok_or_else(|| EmitError::at(id, "numerical integration requires explicit opt-in"))?;
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
        let cuts = self.integral_breakpoints(args[0], args[1])?;
        let mut bounds = vec![lo.clone()];
        let mut seen = HashSet::from([lo.ssa.clone(), hi.ssa.clone()]);
        for cut in cuts {
            if cut.ty != MlirTy::Scalar {
                return Err(EmitError::at(id, "integration breakpoints must be scalar"));
            }
            let cut = self.convert(&cut, ElemKind::Real);
            if seen.insert(cut.ssa.clone()) {
                let below = self.compare("LT", &cut, &lo);
                let cut = self.select(&below, &lo, &cut);
                let above = self.compare("GT", &cut, &hi);
                let cut = self.select(&above, &hi, &cut);
                bounds.push(cut);
            }
        }
        // Comparisons choose the ordering. The selected endpoint values stay live.
        for i in 2..bounds.len() {
            for j in (2..=i).rev() {
                let swap = self.compare("LT", &bounds[j], &bounds[j - 1]);
                let left = self.select(&swap, &bounds[j], &bounds[j - 1]);
                let right = self.select(&swap, &bounds[j - 1], &bounds[j]);
                bounds[j - 1] = left;
                bounds[j] = right;
            }
        }
        bounds.push(hi.clone());
        let segments = bounds.len() - 1;
        if segments > options.max_intervals as usize {
            return Err(EmitError::at(
                id,
                "integration capacity must cover every breakpoint segment",
            ));
        }
        let inf = self.scalar(f64::INFINITY);
        let neg_inf = self.scalar(f64::NEG_INFINITY);
        let nan = self.scalar(f64::NAN);
        let ione = self.int_value_const(1);
        let capacity = self.int_value_const(options.max_intervals as i64);
        let buffer_ty = MlirTy::Ranked(vec![Some(options.max_intervals as u64)]);
        let left = self.constant(0.0, buffer_ty.clone());
        let right = self.constant(1.0, buffer_ty.clone());
        let mut segment_ids = self.convert(&left, ElemKind::Int);
        let mut masses = self.constant(f64::NEG_INFINITY, buffer_ty.clone());
        let mut errors = masses.clone();
        let mut valid = self.compare("LT", &lo, &hi);
        // One worklist and one error budget cover the entire domain. Separate
        // integer segment IDs preserve full precision in local unit coordinates.
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        for segment in 0..segments {
            let index = self.int_value_const(segment as i64);
            let (mass, error, panel_valid) =
                self.quadrature_panel(args[0], args[1], &bounds, &index, &zero, &one)?;
            segment_ids = self.scan_update(&segment_ids, &index, &index);
            masses = self.scan_update(&masses, &mass, &index);
            errors = self.scan_update(&errors, &error, &index);
            valid = self.and(&valid, &panel_valid);
        }
        let total = self.integral_logsum(&masses);
        let error = self.integral_logsum(&errors);
        let log_rtol = self.scalar(options.rtol.ln());
        let log_atol = self.scalar(options.atol.ln());
        let initial = vec![
            self.int_value_const(segments as i64),
            left,
            right,
            masses,
            errors,
            total,
            error,
            valid,
            segment_ids,
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
                let segment = e.scan_slice(&s[8], &index);
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
                        e.quadrature_panel(args[0], args[1], &bounds, &segment, child_a, child_b)?;
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
                    e.scan_update(&s[8], &segment, &s[0]),
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
            let right = self.axes_view(&result[2], panel_axes.clone());
            let segments = self.axes_view(&result[8], panel_axes);
            let (masses, _, _) =
                self.quadrature_panel(args[0], args[1], &bounds, &segments, &left, &right)?;
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

    fn integral_inputs(&self, args: &[NodeId]) -> Vec<(IntegralInput, Value)> {
        let mut inputs = Vec::new();
        let mut bound = HashSet::new();
        let mut pending = vec![args[0], args[2], args[3]];
        let mut seen = HashSet::new();
        while let Some(id) = pending.pop() {
            if id == args[1] || !seen.insert(id) {
                continue;
            }
            if let Some(value) = self.memo.get(&id) {
                if self.batch_rank(value) != 0 {
                    inputs.push((IntegralInput::Node(id), value.clone()));
                }
                continue;
            }
            let fields = match self.type_of(id) {
                Some(
                    Type::Record(fields)
                    | Type::Table {
                        columns: fields, ..
                    },
                ) => fields.as_ref(),
                _ => &[],
            };
            let resolved = self.resolve_ref_one(id);
            for (name, _) in fields {
                let key = (resolved, self.resolve(*name).to_owned());
                if let Some(value) = self.columns.get(&key)
                    && self.batch_rank(value) != 0
                    && bound.insert(key.clone())
                {
                    inputs.push((IntegralInput::Column(key.0, key.1), value.clone()));
                }
            }
            if resolved != id {
                pending.push(resolved);
            } else {
                pending.extend(self.node(id).children());
            }
        }
        inputs
    }

    fn lower_batched_integral(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        inputs: &[(IntegralInput, Value)],
    ) -> Result<Value, EmitError> {
        let mut frame = Vec::new();
        for (_, value) in inputs {
            let rank = self.batch_rank(value);
            frame.resize(frame.len().max(rank), Some(1));
            for (out, &dim) in frame.iter_mut().zip(&shape(&value.ty)[..rank]) {
                if *out == Some(1) {
                    *out = dim;
                } else if dim != Some(1) && *out != dim {
                    return Err(EmitError::at(id, "integral batch extents differ"));
                }
            }
        }
        let count = frame.iter().try_fold(1u64, |n, &dim| n.checked_mul(dim?));
        let count = count.filter(|&n| n <= i32::MAX as u64).ok_or_else(|| {
            EmitError::at(
                id,
                "integral batch requires a static size within int32 range",
            )
        })?;
        let mut initial = Vec::new();
        for (_, value) in inputs {
            let axes = self.axes_of(value);
            let cell = &shape(&value.ty)[axes.batch..];
            let mut dims = frame.clone();
            dims.extend_from_slice(cell);
            let map = (0..axes.batch as u64)
                .chain((frame.len()..dims.len()).map(|i| i as u64))
                .collect::<Vec<_>>();
            let expanded = self.expand_axes(
                value,
                &map,
                batching::tensor(dims),
                Axes {
                    batch: frame.len(),
                    layers: axes.layers.clone(),
                },
            );
            let mut dims = vec![Some(count)];
            dims.extend_from_slice(cell);
            let mut layers = vec![1];
            layers.extend(axes.layers);
            initial.push(self.reshape_axes(
                &expanded,
                batching::tensor(dims),
                Axes { batch: 0, layers },
            ));
        }
        let buffer = self.constant(0.0, MlirTy::Ranked(vec![Some(count)]));
        initial.push(buffer.clone());
        let memo = self.memo.clone();
        let columns = self.columns.clone();
        let parent = std::mem::take(&mut self.broadcast_frame);
        let result = if count == 0 {
            Ok(buffer)
        } else {
            // Enzyme's reverse loop cache uses i64 slice indices. Keep the
            // counter compatible without widening query integers or real values.
            let counter = self.pure("stablehlo.constant dense<0> : tensor<i64>".into());
            let mut names = vec![counter];
            names.extend(initial.iter().map(|v| v.ssa.clone()));
            let mut types = vec!["tensor<i64>".to_owned()];
            types.extend(initial.iter().map(|v| v.ty.render(self.dtype, v.elem)));
            self.try_while_ssa(
                &names,
                &types,
                |e, state| {
                    let limit = e.pure(format!("stablehlo.constant dense<{count}> : tensor<i64>"));
                    e.pure(format!(
                        "stablehlo.compare LT, {}, {limit}, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>", state[0]
                    ))
                },
                |e, state| {
                    let index_ty = MlirTy::Scalar.render(e.dtype, ElemKind::Int);
                    let index = Value {
                        ssa: e.pure(format!("stablehlo.convert {} : (tensor<i64>) -> {index_ty}", state[0])),
                        ty: MlirTy::Scalar,
                        elem: ElemKind::Int,
                    };
                    let values = state[1..].iter().zip(&initial).map(|(ssa, init)| Value {
                        ssa: ssa.clone(),
                        ty: init.ty.clone(),
                        elem: init.elem,
                    }).collect::<Vec<_>>();
                    for (i, (input, _)) in inputs.iter().enumerate() {
                        let value = e.axes_view(&values[i], e.axes_of(&initial[i]));
                        let value = e.scan_slice(&value, &index);
                        match input {
                            IntegralInput::Node(node) => e.bind(*node, value),
                            IntegralInput::Column(node, field) => {
                                e.columns.insert((*node, field.clone()), value);
                            }
                        }
                    }
                    let value = e.lower_integral(id, args)?;
                    let one = e.pure("stablehlo.constant dense<1> : tensor<i64>".into());
                    let mut next = vec![e.pure(format!("stablehlo.add {}, {one} : tensor<i64>", state[0]))];
                    next.extend_from_slice(&state[1..state.len() - 1]);
                    next.push(e.scan_update(values.last().unwrap(), &value, &index).ssa);
                    Ok(next)
                },
            )
            .map(|state| Value {
                ssa: state.last().unwrap().clone(),
                ty: buffer.ty.clone(),
                elem: buffer.elem,
            })
        };
        self.memo = memo;
        self.columns = columns;
        self.broadcast_frame = parent;
        result.map(|value| {
            self.reshape_axes(
                &value,
                batching::tensor(frame.clone()),
                Axes {
                    batch: frame.len(),
                    layers: vec![],
                },
            )
        })
    }

    fn lower_integral_dependencies(&mut self, body: NodeId) -> Result<(), EmitError> {
        let mut pending = vec![body];
        let mut seen = HashSet::new();
        while let Some(id) = pending.pop() {
            if !seen.insert(id) || self.memo.contains_key(&id) {
                continue;
            }
            let resolved = self.resolve_ref_one(id);
            if resolved != id {
                if matches!(self.node(resolved), Node::Call(c)
                    if matches!(c.head, CallHead::Builtin(h)
                        if self.resolve(h) == flatppl_core::LOG_INTEGRAL))
                {
                    self.lower_node(id)?;
                } else {
                    pending.push(resolved);
                }
            } else {
                pending.extend(self.node(id).children());
            }
        }
        Ok(())
    }

    /// Find explicit scalar-coordinate cuts after callable reduction. General
    /// root finding is not an AD contract: refuse unknown cuts in derivative mode.
    fn integral_breakpoints(
        &mut self,
        body: NodeId,
        point: NodeId,
    ) -> Result<Vec<Value>, EmitError> {
        let mut pending = vec![body];
        let mut seen = HashSet::new();
        let mut cuts = Vec::new();
        while let Some(id) = pending.pop() {
            if !seen.insert(id) || self.memo.contains_key(&id) {
                continue;
            }
            let resolved = self.resolve_ref_one(id);
            if resolved != id {
                pending.push(resolved);
                continue;
            }
            if let Node::Call(c) = self.node(id).clone()
                && let CallHead::Builtin(head) = c.head
            {
                match self.resolve(head) {
                    "lt" | "le" | "gt" | "ge" | "equal" | "unequal" if c.args.len() == 2 => {
                        let a = self.integral_depends_on(c.args[0], point);
                        let b = self.integral_depends_on(c.args[1], point);
                        if a || b {
                            let (coordinate, cut) = if a {
                                (c.args[0], c.args[1])
                            } else {
                                (c.args[1], c.args[0])
                            };
                            if !(a && b) && self.resolve_ref_one(coordinate) == point {
                                cuts.push(self.lower_node(cut)?);
                            } else if self.restrict_enzyme_compatible {
                                return Err(EmitError::at(
                                    id,
                                    "numerical gradients require comparisons of the integration coordinate with coordinate-independent scalar breakpoints",
                                ));
                            }
                        }
                    }
                    "in" if c.args.len() == 2
                        && c.args.iter().any(|&n| self.integral_depends_on(n, point)) =>
                    {
                        let set = self.resolve_ref_one(c.args[1]);
                        if self.resolve_ref_one(c.args[0]) == point
                            && let Node::Call(set) = self.node(set).clone()
                            && matches!(set.head, CallHead::Builtin(h) if self.resolve(h) == "interval")
                            && set.args.len() == 2
                            && !set.args.iter().any(|&n| self.integral_depends_on(n, point))
                        {
                            for cut in set.args {
                                cuts.push(self.lower_node(cut)?);
                            }
                        } else if self.resolve_ref_one(c.args[0]) == point
                            && let Node::Const(set) = self.node(set)
                            && matches!(
                                self.resolve(*set),
                                "reals" | "posreals" | "nonnegreals" | "unitinterval"
                            )
                        {
                            let set = self.resolve(*set).to_owned();
                            if set != "reals" {
                                cuts.push(self.scalar(0.0));
                            }
                            if set == "unitinterval" {
                                cuts.push(self.scalar(1.0));
                            }
                        } else if self.restrict_enzyme_compatible {
                            return Err(EmitError::at(
                                id,
                                "numerical gradients require an explicit coordinate-independent interval for integration-coordinate membership",
                            ));
                        }
                    }
                    "floor" | "ceil" | "round" | "div" | "mod" | "stepwise"
                        if self.restrict_enzyme_compatible
                            && c.args.iter().any(|&n| self.integral_depends_on(n, point)) =>
                    {
                        return Err(EmitError::at(
                            id,
                            "numerical gradients require explicit integration-coordinate breakpoints; this discontinuous operation has no breakpoint lowering",
                        ));
                    }
                    "get" | "get0"
                        if self.restrict_enzyme_compatible
                            && c.args
                                .iter()
                                .skip(1)
                                .any(|&n| self.integral_depends_on(n, point)) =>
                    {
                        return Err(EmitError::at(
                            id,
                            "numerical gradients require coordinate-independent array selectors inside an integral",
                        ));
                    }
                    _ => (),
                }
            }
            self.m.for_each_child(id, |child| pending.push(child));
        }
        Ok(cuts)
    }

    fn integral_depends_on(&self, root: NodeId, point: NodeId) -> bool {
        let mut pending = vec![root];
        let mut seen = HashSet::new();
        while let Some(id) = pending.pop() {
            if id == point {
                return true;
            }
            if !seen.insert(id) || self.memo.contains_key(&id) {
                continue;
            }
            let resolved = self.resolve_ref_one(id);
            if resolved != id {
                pending.push(resolved);
            }
            self.m.for_each_child(id, |child| pending.push(child));
        }
        false
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
        bounds: &[Value],
        segment: &Value,
        a: &Value,
        b: &Value,
    ) -> Result<(Value, Value, Value), EmitError> {
        let nodes = self.quadrature_constants(&NODES);
        let kronrod = self.quadrature_constants(&KRONROD);
        let gauss = self.quadrature_constants(&GAUSS);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let half = self.scalar(0.5);
        let mut lo = bounds[0].clone();
        let mut hi = bounds[1].clone();
        let mut fallback_lo = self.scalar(0.0);
        let mut fallback_hi = self.scalar(1.0);
        for (i, pair) in bounds.windows(2).enumerate() {
            let index = self.int_value_const(i as i64);
            let selected = self.compare("EQ", segment, &index);
            lo = self.select(&selected, &pair[0], &lo);
            hi = self.select(&selected, &pair[1], &hi);
            let nonempty = self.compare("LT", &pair[0], &pair[1]);
            fallback_lo = self.select(&nonempty, &pair[0], &fallback_lo);
            fallback_hi = self.select(&nonempty, &pair[1], &fallback_hi);
        }
        let nonempty = self.compare("LT", &lo, &hi);
        let ordered = self.compare("LE", &lo, &hi);
        // A clipped or coincident segment has zero mass. Evaluate a nonempty
        // segment in its inactive branch, avoiding log(0) and infinite endpoints.
        lo = self.select(&nonempty, &lo, &fallback_lo);
        hi = self.select(&nonempty, &hi, &fallback_hi);
        let width = self.sub(b, a);
        let radius = self.mul(&width, &half);
        let sum = self.add(a, b);
        let center = self.mul(&sum, &half);
        let offset = self.mul(&radius, &nodes);
        let t = self.add(&center, &offset);
        let (x, jacobian) = self.quadrature_map(&t, &lo, &hi);
        // Interior unit nodes can round onto a physical cut in a narrow panel.
        // Sampling the other side can yield a false zero error estimate.
        let above = self.compare("GT", &x, &lo);
        let below = self.compare("LT", &x, &hi);
        let physical_interior = self.and(&above, &below);
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
        let interior = self.and(&interior, &physical_interior);
        let valid = self.and(&finite, &interior);
        let valid = self.reduce_all(&valid);
        let empty = self.not(&nonempty);
        let valid = self.or(&empty, &valid);
        let valid = self.and(&valid, &ordered);
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
        let nonzero = self.and(&nonzero, &nonempty);
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
