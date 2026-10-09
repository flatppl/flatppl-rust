//! StableHLO fast paths for retained `pyhf_helpers` standard-module calls.
//! Other targets expand the portable reference during determinization.

use super::*;

impl Emitter<'_> {
    pub(super) fn lower_pyhf_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "pyhf_helpers" {
            return None;
        }
        if matches!(self.m.node(id), Node::Call(call) if !call.named.is_empty()) {
            return Some(Err(EmitError::at(
                id,
                "pyhf helper arguments must be bound by determinize",
            )));
        }
        Some(self.emit_pyhf_call(id, &member, args, broadcast))
    }

    fn emit_pyhf_call(
        &mut self,
        id: NodeId,
        member: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        let mut values = args
            .iter()
            .map(|&arg| {
                let value = self.lower_node(arg)?;
                Ok(self.typed_axes(arg, value))
            })
            .collect::<Result<Vec<_>, EmitError>>()?;
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let result = match (member, values.as_slice()) {
            ("normsys_factor", [lo, hi, alpha]) => Ok(self.normsys_factor(lo, hi, alpha)),
            ("histosys_shift", [lo, nominal, hi, alpha]) => {
                Ok(self.histosys_shift(lo, nominal, hi, alpha))
            }
            ("sample_yields", [nominal, shifts, factors]) => {
                let sums = self.pyhf_reduce(shifts, 1, false);
                let products = self.pyhf_reduce(factors, 1, true);
                let shifted = self.add(nominal, &sums);
                Ok(self.mul(&shifted, &products))
            }
            ("expected_counts", [samples]) => Ok(self.pyhf_reduce(samples, 0, false)),
            _ => Err(EmitError::at(
                id,
                format!("unsupported pyhf helper call `{member}`"),
            )),
        };
        match parent {
            Some(parent) => {
                let frame = self.broadcast_frame.clone();
                let result =
                    result.map(|value| self.finish_broadcast(&value, &frame, parent.len()));
                self.broadcast_frame = parent;
                result
            }
            None => result,
        }
    }

    /// Reduce a model axis without transposing it past bin or batch axes.
    fn pyhf_reduce(&mut self, value: &Value, axis: usize, product: bool) -> Value {
        let value = self.convert(value, ElemKind::Real);
        let identity = if product {
            "1.000000e+00"
        } else {
            "0.000000e+00"
        };
        let op = if product {
            "stablehlo.multiply"
        } else {
            "stablehlo.add"
        };
        self.reduce_axis_lit(op, identity, &value, self.batch_rank(&value) + axis)
    }

    /// HistFactory code4p returns the additive shift, not the shifted template.
    /// Keep modifier/bin axes intact and avoid cancellation against the nominal.
    fn histosys_shift(&mut self, lo: &Value, nominal: &Value, hi: &Value, alpha: &Value) -> Value {
        let up = self.sub(hi, nominal);
        let up = self.convert(&up, ElemKind::Real);
        let left = self.sub(lo, nominal);
        let zero = self.scalar(0.0);
        let down = self.sub(&zero, &left);
        // §09 requires finite alpha. Fixed positive-zero shifts reduce to
        // alpha * 0 in both regions, including the sign of a zero result.
        // Runtime anchors must retain their derivatives even when equal.
        if [&up, &down].iter().all(|value| {
            self.constants.get(&value.ssa).is_some_and(|data| {
                data.iter()
                    .all(|x| matches!(x, Scalar::Real(v) if v.to_bits() == 0))
            })
        }) {
            let (shape, _) = self.broadcast_pair(&up, &down);
            let (shape, _) = self.broadcast_pair(&shape, alpha);
            let zero = self.constant_like(0.0, &shape);
            return self.mul(alpha, &zero);
        }
        let sum = self.add(&up, &down);
        let half = self.scalar(0.5);
        let symmetric = self.mul(&sum, &half);
        let difference = self.sub(&up, &down);
        let sixteenth = self.scalar(0.0625);
        let asymmetric = self.mul(&difference, &sixteenth);
        let alpha = self.convert(alpha, ElemKind::Real);
        let one = self.scalar(1.0);
        let minus_one = self.scalar(-1.0);
        let x = self.max(&alpha, &minus_one);
        let x = self.min(&x, &one);
        let x2 = self.mul(&x, &x);
        let three = self.scalar(3.0);
        let minus_ten = self.scalar(-10.0);
        let fifteen = self.scalar(15.0);
        let term = self.mul(&x2, &three);
        let term = self.add(&minus_ten, &term);
        let term = self.mul(&x2, &term);
        let term = self.add(&fifteen, &term);
        // Share the alpha-only scale across bins. Keep the outer x so tiny
        // shifts survive large anchors and negative zero keeps its sign.
        let scale = self.mul(&x, &term);
        let term = self.mul(&asymmetric, &scale);
        let term = self.add(&symmetric, &term);
        let polynomial = self.mul(&x, &term);
        let positive = self.compare("GT", &alpha, &zero);
        let slope = self.select(&positive, &up, &down);
        let tail = self.mul(&alpha, &slope);
        let abs_alpha = self.abs(&alpha);
        let interior = self.compare("LT", &abs_alpha, &one);
        self.select(&interior, &polynomial, &tail)
    }

    /// Unit-centered §09 interpolation. Coefficient operations fold at target
    /// precision when anchors are fixed, without expanding a callable body.
    fn normsys_factor(&mut self, lo: &Value, hi: &Value, alpha: &Value) -> Value {
        // The reference divides both anchors by its Real unit center first.
        let lo = &self.convert(lo, ElemKind::Real);
        let hi = &self.convert(hi, ElemKind::Real);
        let alpha = &self.convert(alpha, ElemKind::Real);
        let log_hi = self.log(hi);
        let log_lo = self.log(lo);
        let up1 = self.mul(hi, &log_hi);
        let down1 = self.mul(lo, &log_lo);
        let up2 = self.mul(&up1, &log_hi);
        let down2 = self.mul(&down1, &log_lo);
        let half = self.scalar(0.5);
        let s0 = self.add(hi, lo);
        let s0 = self.mul(&s0, &half);
        let a0 = self.sub(hi, lo);
        let a0 = self.mul(&a0, &half);
        let s1 = self.sub(&up1, &down1);
        let s1 = self.mul(&s1, &half);
        let a1 = self.add(&up1, &down1);
        let a1 = self.mul(&a1, &half);
        let s2 = self.add(&up2, &down2);
        let s2 = self.mul(&s2, &half);
        let a2 = self.sub(&up2, &down2);
        let a2 = self.mul(&a2, &half);

        let coefficients = [
            self.interpolation_coefficient([&a0, &s1, &a2], [15.0, -7.0, 1.0], 0.0, 0.125),
            self.interpolation_coefficient([&s0, &a1, &s2], [24.0, -9.0, 1.0], -24.0, 0.125),
            self.interpolation_coefficient([&a0, &s1, &a2], [-5.0, 5.0, -1.0], 0.0, 0.25),
            self.interpolation_coefficient([&s0, &a1, &s2], [-12.0, 7.0, -1.0], 12.0, 0.25),
            self.interpolation_coefficient([&a0, &s1, &a2], [3.0, -3.0, 1.0], 0.0, 0.125),
            self.interpolation_coefficient([&s0, &a1, &s2], [8.0, -5.0, 1.0], -8.0, 0.125),
        ];
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let abs_alpha = self.abs(alpha);
        let interior = self.compare("LT", &abs_alpha, &one);
        // Bound inactive powers without making x depend on the output predicate.
        // The tails own the knots, so clamp's boundary derivative is never used.
        let minus_one = self.scalar(-1.0);
        let x = self.max(alpha, &minus_one);
        let x = self.min(&x, &one);
        let pairs = [0, 2, 4].map(|i| {
            let term = self.mul(&x, &coefficients[i + 1]);
            self.add(&coefficients[i], &term)
        });
        let x2 = self.mul(&x, &x);
        let term = self.mul(&x2, &pairs[2]);
        let term = self.add(&pairs[1], &term);
        let term = self.mul(&x2, &term);
        let term = self.add(&pairs[0], &term);
        let term = self.mul(&x, &term);
        let polynomial = self.add(&one, &term);

        // Choose by sign even inside the polynomial region. Then the inactive
        // exponential stays between one and its finite positive anchor.
        let positive = self.compare("GT", alpha, &zero);
        let minus_log_lo = self.neg(&log_lo);
        let slope = self.select(&positive, &log_hi, &minus_log_lo);
        let exponent = self.mul(alpha, &slope);
        let tail = self.exp(&exponent);
        self.select(&interior, &polynomial, &tail)
    }

    fn interpolation_coefficient(
        &mut self,
        terms: [&Value; 3],
        weights: [f64; 3],
        offset: f64,
        scale: f64,
    ) -> Value {
        let mut result = self.scalar(offset);
        for (term, weight) in terms.into_iter().zip(weights) {
            let weight = self.scalar(weight);
            let term = self.mul(term, &weight);
            result = self.add(&result, &term);
        }
        let scale = self.scalar(scale);
        self.mul(&result, &scale)
    }
}
