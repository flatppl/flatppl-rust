//! Runtime checks of distribution parameters against their declared value sets.
//!
//! §08 and §09 declare a value set for each parameter, such as `sigma =
//! elementof(posreals)`. A density or sampler outside that set has no meaning,
//! and its closed form returns a plausible number or NaN. So each parameter that
//! the emitter cannot prove inside its set reports status
//! [`PARAMETER_DOMAIN_STATUS`]. A NaN parameter propagates without an error, as
//! status 3 already does for quantile probabilities. So a `reals` parameter needs
//! no check.
//!
//! The emitter proves membership without runtime work in two cases. A constant is
//! checked at emit time, and a value outside its set refuses. A value produced by
//! a function whose §07 result set lies inside the declared set needs no check,
//! for example `exp(a)` for a `posreals` parameter. Host inputs are never proven:
//! their declared sets are not enforced at the ABI.

use flatppl_infer::ParamSet;

use super::constants::element_count;
use super::*;

/// Status for a distribution parameter outside its declared value set.
pub(crate) const PARAMETER_DOMAIN_STATUS: i64 = 9;

/// Largest constant matrix factored at emit time.
const MAX_STATIC_FACTOR: usize = 64;

impl Emitter<'_> {
    /// Check each parameter `(name, node, set)` of distribution `ctor`. A
    /// `PosDefinite` set is left to [`Emitter::positive_definite_factor`], which
    /// reuses the factorization the builder needs.
    pub(crate) fn check_parameter_domains(
        &mut self,
        ctor: &str,
        params: &[(&str, NodeId, ParamSet)],
    ) -> Result<(), EmitError> {
        let mut violations: Vec<Value> = Vec::new();
        for &(name, id, set) in params {
            let id = self.sole_element(id);
            if matches!(
                set,
                ParamSet::Unconstrained | ParamSet::PosDefinite | ParamSet::Reals
            ) {
                continue;
            }
            if self
                .proven_set(id)
                .is_some_and(|proven| set_includes(set, proven))
            {
                continue;
            }
            let value = self.lower_node(id)?;
            if let Some(values) = self.constant_elements(&value) {
                if constant_violates(set, &values, self.cell_len(&value), self.dtype) {
                    return Err(outside(id, ctor, name, set));
                }
                continue;
            }
            // A check outside every region holds wherever the value is used.
            let key = (value.ssa.clone(), format!("{set:?}"));
            if self.checked_parameters.contains(&key) {
                continue;
            }
            if self.region_depth == 0 {
                self.checked_parameters.insert(key);
            }
            violations.extend(self.violations(&value, set));
        }
        // One status merge per shape: same-shaped violations combine first.
        let mut groups: Vec<Value> = Vec::new();
        for violation in violations {
            let axes = self.axes_of(&violation);
            match groups
                .iter_mut()
                .find(|group| group.ty == violation.ty && self.axes_of(group) == axes)
            {
                Some(group) => {
                    let combined = self.either(group, &violation);
                    *group = combined;
                }
                None => groups.push(violation),
            }
        }
        for violation in groups {
            self.record_runtime_violation(&violation, PARAMETER_DOMAIN_STATUS);
        }
        Ok(())
    }

    /// The lower Cholesky factor of positive definite parameter `name` of `ctor`,
    /// and its diagonal. A runtime matrix whose diagonal is nonpositive or
    /// non-finite reports [`PARAMETER_DOMAIN_STATUS`]. A constant matrix is
    /// factored here.
    pub(crate) fn positive_definite_factor(
        &mut self,
        id: NodeId,
        ctor: &str,
        name: &str,
        value: &Value,
    ) -> Result<(Value, Value), EmitError> {
        let known = if self.proven_set(id) == Some(ParamSet::PosDefinite) {
            true
        } else if let Some(values) = self.constant_elements(value) {
            if !constant_positive_definite(&values) {
                return Err(outside(id, ctor, name, ParamSet::PosDefinite));
            }
            true
        } else {
            false
        };
        let factor = self.cholesky(value);
        let diagonal = self.diag(&factor);
        if !known {
            // A NaN diagonal fails both tests: it marks a failed factorization.
            let least = self.per_call(&diagonal, false);
            let most = self.per_call(&diagonal, true);
            let zero = self.scalar(0.0);
            let infinity = self.scalar(f64::INFINITY);
            let positive = self.compare("GT", &least, &zero);
            let finite = self.compare("LT", &most, &infinity);
            let valid = self.select(&positive, &finite, &positive);
            self.record_runtime_error(&valid, PARAMETER_DOMAIN_STATUS);
        }
        Ok((factor, diagonal))
    }

    /// Every element of a constant value, through layout views, or `None`.
    fn constant_elements(&self, value: &Value) -> Option<Vec<f64>> {
        if self.batch_rank(value) != 0 {
            return None;
        }
        let count = element_count(&value.ty)?;
        if count > constants::MAX_FOLDED_ELEMENTS {
            return None;
        }
        (0..count)
            .map(|i| match self.fixed_scalar_at(value, i)? {
                Scalar::Real(x) => Some(self.rounded(*x)),
                Scalar::Int(n) => Some(*n as f64),
                Scalar::Bool(b) => Some(f64::from(*b)),
                _ => None,
            })
            .collect()
    }

    fn rounded(&self, x: f64) -> f64 {
        match self.dtype {
            Dtype::F32 => (x as f32) as f64,
            Dtype::F64 => x,
        }
    }

    /// Length of the innermost cell axis, the simplex dimension.
    fn cell_len(&self, value: &Value) -> usize {
        match self.cell_ty(value) {
            MlirTy::Ranked(dims) => dims
                .last()
                .copied()
                .flatten()
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(1),
            _ => 1,
        }
    }

    /// The element of a one-element `vector(x)`, the wrapper a batched kernel
    /// input puts around a parameter shared by every cell.
    fn sole_element(&self, id: NodeId) -> NodeId {
        let field = self.resolve_ref_one(id);
        match self.m.node(field) {
            Node::Call(c)
                if matches!(c.head, CallHead::Builtin(s) if self.resolve(s) == "vector")
                    && c.args.len() == 1
                    && c.named.is_empty() =>
            {
                c.args[0]
            }
            _ => id,
        }
    }

    /// The value set a computation guarantees for `id`, without reading inputs.
    fn proven_set(&self, mut id: NodeId) -> Option<ParamSet> {
        for _ in 0..self.m.binding_count() + 1 {
            let next = self.resolve_ref_one(id);
            if next == id {
                break;
            }
            id = next;
        }
        let Node::Call(call) = self.m.node(id) else {
            return None;
        };
        let CallHead::Builtin(head) = call.head else {
            return None;
        };
        match self.resolve(head) {
            // §07 `softmax`: "a probability vector".
            "softmax" => Some(ParamSet::StdSimplex),
            // §07 `eye`: the identity matrix.
            "eye" => Some(ParamSet::PosDefinite),
            // A dotted call applies its function to each element. A reified
            // function proves what its body proves.
            "broadcast" => {
                let &f = call.args.first()?;
                if let Node::Const(f) = self.m.node(f) {
                    return flatppl_infer::function_result_set(self.resolve(*f));
                }
                let function = self.resolve_functionof(f)?;
                let Node::Call(function) = self.m.node(function) else {
                    return None;
                };
                match function.args.as_ref() {
                    [body] => self.proven_set(*body),
                    _ => None,
                }
            }
            name => flatppl_infer::function_result_set(name),
        }
    }

    /// Where a runtime `value` lies outside `set`, one flag per call. A vector
    /// reduces to its minimum and maximum first: a float `min` or `max` stays
    /// in the surrounding fusion, where XLA's CPU backend runs a boolean
    /// `or` reduction as a slow separate loop. Both propagate NaN, and every
    /// test is an ordered comparison, so NaN never violates.
    fn violations(&mut self, value: &Value, set: ParamSet) -> Vec<Value> {
        use ParamSet::*;
        // (lower bound, whether the bound itself violates, upper bound, integral)
        let (lower, strict, upper, whole) = match set {
            Unconstrained | PosDefinite | Reals => return vec![],
            StdSimplex => return self.simplex_violations(value),
            PosReals => (Some(0.0), true, None, false),
            NonNegReals => (Some(0.0), false, None, false),
            UnitInterval => (Some(0.0), false, Some(1.0), false),
            Interval(lo, hi) => (Some(lo), false, Some(hi), false),
            Integers => (None, false, None, true),
            PosIntegers => (Some(1.0), false, None, true),
            NonNegIntegers => (Some(0.0), false, None, true),
            Booleans => (Some(0.0), false, Some(1.0), true),
        };
        let holds_bool = |x: f64| set.contains(x) == Some(true);
        if value.elem == ElemKind::Bool && holds_bool(0.0) && holds_bool(1.0) {
            return vec![];
        }
        let whole = whole && value.elem == ElemKind::Real;
        let x = self.convert(value, ElemKind::Real);
        let mut flags = Vec::new();
        if let Some(lo) = lower {
            let least = self.per_call(&x, false);
            let lo = self.scalar(lo);
            flags.push(self.compare(if strict { "LE" } else { "LT" }, &least, &lo));
        }
        if let Some(hi) = upper {
            let most = self.per_call(&x, true);
            let hi = self.scalar(hi);
            flags.push(self.compare("GT", &most, &hi));
        }
        if whole {
            // `x - floor(x)` is zero for an integer and NaN for an infinity.
            let floor = self.floor(&x);
            let fraction = self.sub(&x, &floor);
            let fraction = self.per_call(&fraction, true);
            let zero = self.scalar(0.0);
            flags.push(self.compare("GT", &fraction, &zero));
        }
        let first = flags.remove(0);
        vec![
            flags
                .iter()
                .fold(first, |any, flag| self.either(&any, flag)),
        ]
    }

    /// The minimum or maximum of `x` over every axis within one call, keeping
    /// the host batch axes. Both propagate NaN.
    fn per_call(&mut self, x: &Value, maximum: bool) -> Value {
        let rank = shape(&x.ty).len();
        let batch = self.batch_rank(x).min(self.runtime_batch_rank);
        if rank == batch {
            return x.clone();
        }
        let (op, identity) = if maximum {
            ("stablehlo.maximum", reduce_max_identity(self.dtype))
        } else {
            ("stablehlo.minimum", pos_inf_literal(self.dtype))
        };
        let mut x = self.axes_view(
            x,
            Axes {
                batch,
                layers: vec![rank - batch],
            },
        );
        for _ in batch..rank {
            x = self.reduce_axis(op, identity, &x, batch);
        }
        x
    }

    /// Boolean OR as a `select`, which the packing pass treats as pure.
    fn either(&mut self, a: &Value, b: &Value) -> Value {
        self.select(a, a, b)
    }

    /// `stdsimplex(n)`: a negative cell, or a sum farther than
    /// [`simplex_tolerance`] from one.
    fn simplex_violations(&mut self, value: &Value) -> Vec<Value> {
        if self.cell_ty(value) == MlirTy::Scalar {
            return Vec::new();
        }
        let n = self.cell_len(value);
        let p = self.convert(value, ElemKind::Real);
        let least = self.per_call(&p, false);
        let zero = self.scalar(0.0);
        let negative = self.compare("LT", &least, &zero);
        let total = self.reduce_sum_last_axis(&p);
        let one = self.scalar(1.0);
        let error = self.sub(&total, &one);
        let error = self.abs(&error);
        let error = self.per_call(&error, true);
        let tolerance = self.scalar(simplex_tolerance(n, self.dtype));
        let far = self.compare("GT", &error, &tolerance);
        vec![self.either(&negative, &far)]
    }
}

/// The rounding allowance on the sum of an `n`-cell probability vector: each
/// normalized cell and each partial sum rounds once.
pub(crate) fn simplex_tolerance(n: usize, dtype: Dtype) -> f64 {
    let epsilon = match dtype {
        Dtype::F32 => f64::from(f32::EPSILON),
        Dtype::F64 => f64::EPSILON,
    };
    4.0 * n.max(1) as f64 * epsilon
}

/// Whether every value a computation proves for a parameter lies in `set`.
fn set_includes(set: ParamSet, proven: ParamSet) -> bool {
    use ParamSet::*;
    let interval = |s: ParamSet| match s {
        Reals => Some((f64::NEG_INFINITY, f64::INFINITY, true)),
        PosReals | PosIntegers => Some((0.0, f64::INFINITY, false)),
        NonNegReals | NonNegIntegers => Some((0.0, f64::INFINITY, true)),
        UnitInterval | Booleans => Some((0.0, 1.0, true)),
        Interval(lo, hi) => Some((lo, hi, true)),
        _ => None,
    };
    let integral = |s: ParamSet| matches!(s, Integers | PosIntegers | NonNegIntegers | Booleans);
    match (set, proven) {
        (StdSimplex, StdSimplex) => true,
        (Unconstrained, _) => true,
        (StdSimplex | PosDefinite, _) | (_, StdSimplex | PosDefinite | Unconstrained) => false,
        (Integers | PosIntegers | NonNegIntegers | Booleans, _) if !integral(proven) => false,
        _ => match (interval(set), interval(proven)) {
            (Some((lo, hi, closed)), Some((plo, phi, pclosed))) => {
                let low = plo > lo || (plo == lo && (closed || !pclosed));
                low && phi <= hi && (set != Booleans || proven == Booleans)
            }
            (Some(_), None) => matches!(proven, Integers) && set == Reals,
            _ => false,
        },
    }
}

/// Whether constant `values` lie outside `set`, with the runtime test's
/// semantics: NaN never violates. `n` is the simplex dimension.
fn constant_violates(set: ParamSet, values: &[f64], n: usize, dtype: Dtype) -> bool {
    let violates = |x: f64| {
        let fractional = x - x.floor() > 0.0;
        match set {
            ParamSet::Unconstrained
            | ParamSet::Reals
            | ParamSet::StdSimplex
            | ParamSet::PosDefinite => false,
            ParamSet::PosReals => x <= 0.0,
            ParamSet::NonNegReals => x < 0.0,
            ParamSet::UnitInterval => x < 0.0 || x > 1.0,
            ParamSet::Interval(lo, hi) => x < lo || x > hi,
            ParamSet::Integers => fractional,
            ParamSet::PosIntegers => x < 1.0 || fractional,
            ParamSet::NonNegIntegers => x < 0.0 || fractional,
            ParamSet::Booleans => x < 0.0 || x > 1.0 || (x > 0.0 && x < 1.0),
        }
    };
    if set == ParamSet::StdSimplex {
        return values.iter().any(|p| *p < 0.0)
            || values
                .chunks(n.max(1))
                .any(|cell| (cell.iter().sum::<f64>() - 1.0).abs() > simplex_tolerance(n, dtype));
    }
    values.iter().any(|x| violates(*x))
}

/// Whether a constant square matrix, in row-major order, has a Cholesky factor
/// with a positive finite diagonal. A larger matrix is left to run time.
fn constant_positive_definite(values: &[f64]) -> bool {
    let n = (values.len() as f64).sqrt() as usize;
    if n * n != values.len() || n > MAX_STATIC_FACTOR {
        return true;
    }
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = values[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if !(d > 0.0 && d.is_finite()) {
            return false;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = values[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    true
}

fn outside(id: NodeId, ctor: &str, name: &str, set: ParamSet) -> EmitError {
    EmitError::at(
        id,
        format!(
            "`{ctor}`'s `{name}` is outside its declared value set `{}`",
            set.describe()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proven_sets_include_only_their_subsets() {
        use ParamSet::*;
        assert!(set_includes(PosReals, PosReals));
        assert!(set_includes(NonNegReals, PosReals));
        assert!(set_includes(Reals, UnitInterval));
        assert!(set_includes(PosReals, Interval(0.5, 2.0)));
        assert!(!set_includes(PosReals, NonNegReals));
        assert!(!set_includes(PosReals, Interval(0.0, 2.0)));
        assert!(!set_includes(UnitInterval, PosReals));
        assert!(!set_includes(PosIntegers, PosReals));
        assert!(set_includes(StdSimplex, StdSimplex));
        assert!(!set_includes(StdSimplex, UnitInterval));
        assert!(!set_includes(PosDefinite, PosReals));
    }

    #[test]
    fn constant_violations_follow_the_closed_bounds() {
        use ParamSet::*;
        assert!(!constant_violates(NonNegReals, &[0.0, 2.0], 1, Dtype::F64));
        assert!(constant_violates(PosReals, &[0.0], 1, Dtype::F64));
        assert!(!constant_violates(PosReals, &[f64::NAN], 1, Dtype::F64));
        assert!(constant_violates(PosIntegers, &[1.5], 1, Dtype::F64));
        assert!(!constant_violates(
            StdSimplex,
            &[0.1, 0.2, 0.7],
            3,
            Dtype::F64
        ));
        assert!(constant_violates(StdSimplex, &[0.5, 0.6], 2, Dtype::F64));
        assert!(constant_violates(StdSimplex, &[-0.5, 1.5], 2, Dtype::F64));
        assert!(constant_positive_definite(&[2.0, 1.0, 1.0, 2.0]));
        assert!(!constant_positive_definite(&[1.0, 2.0, 2.0, 1.0]));
        assert!(!constant_positive_definite(&[0.0, 0.0, 0.0, 1.0]));
    }
}
