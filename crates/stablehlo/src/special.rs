//! `special-functions` standard-module calls (§09), which FlatPDL keeps as calls.
//! `erf`, `erfc`, `digamma` and `polygamma` lower to their CHLO ops. `erfcx` has no
//! CHLO op, so it lowers to StableHLO arithmetic that Enzyme differentiates.

use super::*;

#[path = "special/bessel.rs"]
mod bessel;
#[path = "special/probability.rs"]
mod probability;

/// The `special-functions` members this backend lowers, with their arity.
pub(crate) const SPECIAL_FUNCTIONS: [(&str, usize); 12] = [
    ("erf", 1),
    ("erfc", 1),
    ("erfcx", 1),
    ("digamma", 1),
    ("polygamma", 2),
    ("gammainc", 2),
    ("betainc", 3),
    ("bessel_j", 2),
    ("bessel_y", 2),
    ("bessel_i", 2),
    ("bessel_k", 2),
    ("airy", 1),
];

/// Below this `erfcx(x) = exp(x²)·erfc(x)`. From here up the continued fraction
/// converges to rounding in [`ERFCX_DEPTH_F64`] terms (mpmath, x up to 1e4).
const ERFCX_SPLIT: f64 = 2.5;
const ERFCX_DEPTH_F64: usize = 40;
const ERFCX_DEPTH_F32: usize = 12;
const FRAC_1_SQRT_PI: f64 = 0.5641895835477563;

impl Emitter<'_> {
    /// A static tape bound with a safe, runtime-budgeted prefix. Final
    /// convergence checks reject workspaces too small for the requested rule.
    fn special_loop(
        &mut self,
        maximum: u64,
        limit: &Value,
        initial: &[Value],
        mut step: impl FnMut(&mut Self, &Value, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let requested = self.ceil(limit);
        let rank = shape(&requested.ty).len();
        let requested = self.axes_view(
            &requested,
            Axes {
                batch: 0,
                layers: if rank == 0 { vec![] } else { vec![rank] },
            },
        );
        let requested = self.reduce_max(&requested);
        let count = self.convert(&requested, ElemKind::Int);
        let one = self.int_value_const(1);
        let last = self.sub(&count, &one);
        let output = self.matrix_loop(maximum, 0, initial, |e, index, state| {
            let state: Vec<_> = state
                .iter()
                .zip(initial)
                .map(|(value, initial)| e.axes_view(value, e.axes_of(initial)))
                .collect();
            let active = e.compare("LT", index, &count);
            let safe_index = e.min(index, &last);
            let next = step(e, &safe_index, &state);
            next.into_iter()
                .zip(&state)
                .map(|(next, old)| {
                    let next = e.select(&active, &next, old);
                    e.axes_view(&next, e.axes_of(old))
                })
                .collect()
        });
        let bound = self.scalar(maximum as f64);
        let fits = self.compare("LE", &requested, &bound);
        output
            .iter()
            .zip(initial)
            .map(|(value, initial)| {
                let value = self.axes_view(value, self.axes_of(initial));
                let nan = self.constant_like(f64::NAN, &value);
                self.select(&fits, &value, &nan)
            })
            .collect()
    }
    /// Principal J/I continuation. Complex values retain the backend's pair
    /// representation until a real/imaginary projection consumes them.
    pub(crate) fn lower_bessel_parts(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<(Value, Value), EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "special-functions" || !matches!(member.as_str(), "bessel_j" | "bessel_i") {
            return None;
        }
        Some((|| {
            let [order, x] = args else {
                return Err(EmitError::at(id, "Bessel J/I require order and argument"));
            };
            let order_value = self.lower_node(*order)?;
            let x_value = self.lower_node(*x)?;
            let mut values = [
                self.typed_axes(*order, order_value),
                self.typed_axes(*x, x_value),
            ];
            let parent = if broadcast {
                Some(self.enter_broadcast(id, &mut values)?)
            } else {
                None
            };
            let (re, im) = self.bessel_principal_parts(&member, &values[0], &values[1]);
            Ok(match parent {
                Some(parent) => {
                    let frame = self.broadcast_frame.clone();
                    let re = self.finish_broadcast(&re, &frame, parent.len());
                    let im = self.finish_broadcast(&im, &frame, parent.len());
                    self.broadcast_frame = parent;
                    (re, im)
                }
                None => (re, im),
            })
        })())
    }
    pub(super) fn lower_special_function_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "special-functions" {
            return None;
        }
        let Some(&(_, arity)) = SPECIAL_FUNCTIONS.iter().find(|(name, _)| *name == member) else {
            return Some(Err(EmitError::at(
                id,
                format!("special-functions.{member} has no StableHLO lowering"),
            )));
        };
        Some(self.emit_special_function_call(id, &member, arity, args, broadcast))
    }

    fn emit_special_function_call(
        &mut self,
        id: NodeId,
        member: &str,
        arity: usize,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        if matches!(self.m.node(id), Node::Call(call) if !call.named.is_empty()) {
            return Err(EmitError::at(
                id,
                "special-functions arguments must be bound by determinize",
            ));
        }
        if args.len() != arity {
            return Err(EmitError::at(
                id,
                format!("special-functions.{member} takes {arity} argument(s)"),
            ));
        }
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
        let result = match member {
            "erf" => self.chlo_unary("chlo.erf", &values[0]),
            // `1 - erf(x)` cancels to zero in the upper tail.
            "erfc" => self.chlo_unary("chlo.erfc", &values[0]),
            "erfcx" => self.erfcx(&values[0]),
            "digamma" => self.chlo_unary("chlo.digamma", &values[0]),
            "gammainc" => self.regularized_gamma(&values[0], &values[1]),
            "betainc" => self.regularized_beta(&values[0], &values[1], &values[2]),
            "bessel_j" | "bessel_i" => {
                return Err(EmitError::at(
                    id,
                    "complex Bessel J/I result requires real, imag, abs or abs2 projection",
                ));
            }
            "bessel_y" | "bessel_k" => self.bessel_real(member, &values[0], &values[1]),
            "airy" => self.airy_ai(&values[0]),
            // Enzyme differentiates polygamma in `x` only. The order `n` is a §09
            // integer, so it reaches the op as an inactive constant or integer input.
            _ => self.chlo_binary("chlo.polygamma", &values[0], &values[1]),
        };
        Ok(match parent {
            Some(parent) => {
                let frame = self.broadcast_frame.clone();
                let result = self.finish_broadcast(&result, &frame, parent.len());
                self.broadcast_frame = parent;
                result
            }
            None => result,
        })
    }

    /// `erfcx(x) = exp(x²)·erfc(x)`, the product below [`ERFCX_SPLIT`] and the
    /// Laplace continued fraction `1/(√π·(x + ½/(x + 1/(x + (3/2)/(x + …)))))`
    /// from there up, where `erfc` underflows. Each branch reads a clamped input,
    /// so the branch `select` drops never feeds `inf` or `NaN` into a gradient.
    fn erfcx(&mut self, x: &Value) -> Value {
        let x = &self.convert(x, ElemKind::Real);
        let split = self.constant_like(ERFCX_SPLIT, x);
        let small = self.compare("LT", x, &split);

        let zero = self.constant_like(0.0, x);
        let near = self.select(&small, x, &zero);
        let square = self.mul(&near, &near);
        let scale = self.exp(&square);
        let tail = self.chlo_unary("chlo.erfc", &near);
        let product = self.mul(&scale, &tail);

        let far = self.select(&small, &split, x);
        let depth = match self.dtype {
            Dtype::F32 => ERFCX_DEPTH_F32,
            Dtype::F64 => ERFCX_DEPTH_F64,
        };
        let mut fraction = far.clone();
        for n in (1..=depth).rev() {
            let a = self.constant_like(n as f64 / 2.0, x);
            let step = self.div(&a, &fraction);
            fraction = self.add(&far, &step);
        }
        let c = self.constant_like(FRAC_1_SQRT_PI, x);
        let asymptotic = self.div(&c, &fraction);

        self.select(&small, &product, &asymptotic)
    }
}
