//! A GPU `scan` whose carry is affine in itself lowers to a log-depth doubling
//! scan instead of a `while`. A step `x_t = a_t * x_(t-1) + b_t` composes
//! associatively, `(a2, b2) . (a1, b1) = (a2 * a1, a2 * b1 + b2)`, so
//! `ceil(log2 n)` levels of shift, multiply and add replace `n` serial steps.
//!
//! The step body is lowered once over the whole sequence, with each carry
//! leaf bound to an opaque zero. A leaf is solved when its output depends on
//! no unsolved leaf but itself. Solved leaves then enter later lowerings as
//! their shifted sequences. A self-dependent output must be affine in its own
//! zero: the coefficient is read off the recorded pointwise producers, and the
//! output itself, evaluated at zero, is the offset. Anything else keeps the
//! `while`.

use std::collections::HashSet;

use super::batching::tensor;
use super::sequences::Components;
use super::*;

pub(super) struct AffineScan<'a> {
    pub id: NodeId,
    pub body: NodeId,
    pub entries: &'a [(Symbol, Ref)],
    pub state: &'a Components,
    /// The state type, then one item type per sequence.
    pub types: &'a [Type],
    pub seqs: &'a [Components],
    /// The traversed length of each sequence; `None` for a constant input.
    pub lengths: &'a [Option<u64>],
    pub n: u64,
}

/// A carry leaf's coefficient in a step output. A value is a scalar or has
/// the output's shape.
enum Coef {
    One,
    Of(Value),
}

impl Emitter<'_> {
    /// The scan outputs, one `[frame..., n]` sequence per carry leaf, or
    /// `None` with the emitter unchanged when the step is not proven affine.
    pub(super) fn try_affine_scan(&mut self, scan: &AffineScan<'_>) -> Option<Vec<Value>> {
        let mut trial = self.clone();
        let values = trial.affine_scan(scan).ok().flatten()?;
        *self = trial;
        Some(values)
    }

    fn affine_scan(&mut self, scan: &AffineScan<'_>) -> Result<Option<Vec<Value>>, EmitError> {
        let state = scan.state;
        if scan.n < 2
            || state
                .values
                .iter()
                .any(|v| v.elem != ElemKind::Real || self.cell_ty(v) != MlirTy::Scalar)
        {
            return Ok(None);
        }
        let mut leaves = Vec::new();
        for (sequence, length) in scan.seqs.iter().zip(scan.lengths) {
            if length.is_some() {
                leaves.extend(sequence.values.iter().cloned());
            }
        }
        let parent = self.enter_broadcast(scan.id, &mut leaves)?;
        let frame = self.broadcast_frame.clone();
        if frame.len() != parent.len() + 1
            || frame.last() != Some(&Some(scan.n))
            || frame.contains(&None)
        {
            return Ok(None);
        }
        let mut leaves = leaves.into_iter();
        let items = scan
            .seqs
            .iter()
            .zip(scan.lengths)
            .map(|(sequence, length)| match length {
                Some(_) => {
                    let values = leaves
                        .by_ref()
                        .take(sequence.values.len())
                        .collect::<Vec<_>>();
                    sequence.with_values(&values)
                }
                None => sequence.clone(),
            })
            .collect::<Vec<_>>();

        let ty = tensor(frame.clone());
        let mark = self.body.len();
        let zeros = (0..state.values.len())
            .map(|_| self.opaque_zero(&ty))
            .collect::<Vec<_>>();
        let mut carries = zeros.clone();
        let mut solved: Vec<Option<Value>> = vec![None; state.values.len()];
        while solved.iter().any(Option::is_none) {
            let mut arguments = vec![state.with_values(&carries)];
            arguments.extend(items.iter().cloned());
            let output = self
                .call_components(scan.body, scan.entries, &arguments, scan.types)?
                .align(state, scan.id)?;
            let mut outputs = Vec::with_capacity(output.values.len());
            for value in &output.values {
                let value = self.finish_broadcast(value, &frame, frame.len());
                if value.ty != ty || value.elem == ElemKind::Bool {
                    return Ok(None);
                }
                outputs.push(self.convert(&value, ElemKind::Real));
            }
            let unsolved = (0..solved.len())
                .filter(|&j| solved[j].is_none())
                .collect::<Vec<_>>();
            let mut dependents = HashMap::new();
            for &i in &unsolved {
                let Some(set) = dependents_of(&self.body[mark..], &zeros[i].ssa) else {
                    return Ok(None);
                };
                dependents.insert(i, set);
            }
            let mut progress = Vec::new();
            for &j in &unsolved {
                let output = &outputs[j];
                let depends = |i: &usize| dependents[i].contains(base(&output.ssa));
                if unsolved.iter().any(|i| *i != j && depends(i)) {
                    continue;
                }
                let value = if depends(&j) {
                    let Some(coef) = self.coefficient(output, &zeros[j], &dependents[&j]) else {
                        return Ok(None);
                    };
                    self.doubling(coef, output, &state.values[j], &frame)
                } else {
                    output.clone()
                };
                progress.push((j, value));
            }
            if progress.is_empty() {
                return Ok(None);
            }
            for (j, value) in progress {
                let first = self.time_cells(&state.values[j], 1, &frame);
                carries[j] = self.shift(&value, 1, &first);
                solved[j] = Some(value);
            }
        }
        self.broadcast_frame = parent;
        Ok(Some(solved.into_iter().map(Option::unwrap).collect()))
    }

    /// A zero that constant folding and CSE cannot see through, so every use
    /// stays visible in the emitted text.
    fn opaque_zero(&mut self, ty: &MlirTy) -> Value {
        let ssa = self.fresh();
        let text = ty.render(self.dtype, ElemKind::Real);
        let literal = float_literal(self.dtype, 0.0);
        self.push(&format!(
            "{ssa} = stablehlo.constant dense<{literal}> : {text}"
        ));
        let batch = shape(ty).len();
        self.remember_axes(
            &ssa,
            Axes {
                batch,
                layers: vec![],
            },
        );
        // A unique key marks the line pure for dead-code removal only.
        self.pure_ops
            .insert((format!("affine carry {ssa}"), None), ssa.clone());
        Value {
            ssa,
            ty: ty.clone(),
            elem: ElemKind::Real,
        }
    }

    /// The coefficient of `zero` in `v`, or `None` unless `v` is provably
    /// affine in it. `dependent` holds every SSA name that may use `zero`.
    fn coefficient(
        &mut self,
        v: &Value,
        zero: &Value,
        dependent: &HashSet<String>,
    ) -> Option<Coef> {
        if v.ssa == zero.ssa {
            return Some(Coef::One);
        }
        let op = self.pointwise.get(&v.ssa)?.op.clone();
        let part = |e: &mut Self, x: &Value| -> Option<Option<Coef>> {
            if dependent.contains(base(&x.ssa)) {
                e.coefficient(x, zero, dependent).map(Some)
            } else {
                Some(None)
            }
        };
        match op {
            Pointwise::Binary(name, a, b) => {
                let ca = part(self, &a)?;
                let cb = part(self, &b)?;
                match (name.as_str(), ca, cb) {
                    ("stablehlo.add" | "stablehlo.subtract", Some(c), None)
                    | ("stablehlo.add", None, Some(c)) => Some(c),
                    ("stablehlo.subtract", None, Some(c)) => Some(self.coef_neg(c)),
                    ("stablehlo.add", Some(x), Some(y)) => {
                        let (x, y) = (self.coef_value(x), self.coef_value(y));
                        Some(Coef::Of(self.add(&x, &y)))
                    }
                    ("stablehlo.subtract", Some(x), Some(y)) => {
                        let (x, y) = (self.coef_value(x), self.coef_value(y));
                        Some(Coef::Of(self.sub(&x, &y)))
                    }
                    ("stablehlo.multiply", Some(c), None) => Some(self.coef_scale(c, &b, false)),
                    ("stablehlo.multiply", None, Some(c)) => Some(self.coef_scale(c, &a, false)),
                    ("stablehlo.divide", Some(c), None) => Some(self.coef_scale(c, &b, true)),
                    _ => None,
                }
            }
            Pointwise::Unary(name, a) if name == "stablehlo.negate" => {
                let c = part(self, &a)??;
                Some(self.coef_neg(c))
            }
            Pointwise::Reshape(a) => match part(self, &a)?? {
                Coef::Of(x) if x.ty == a.ty && x.ty != MlirTy::Scalar => {
                    let axes = self.axes_of(v);
                    Some(Coef::Of(self.reshape_axes(&x, v.ty.clone(), axes)))
                }
                Coef::Of(x) if x.ty != MlirTy::Scalar => None,
                c => Some(c),
            },
            // Increasing dimensions keep each output cell on its own time step.
            Pointwise::Broadcast(a, dims) if dims.windows(2).all(|d| d[0] < d[1]) => {
                match part(self, &a)?? {
                    Coef::Of(x) if x.ty == a.ty && x.ty != MlirTy::Scalar => {
                        let axes = self.axes_of(v);
                        Some(Coef::Of(self.expand_axes(&x, &dims, v.ty.clone(), axes)))
                    }
                    Coef::Of(x) if x.ty != MlirTy::Scalar => None,
                    c => Some(c),
                }
            }
            _ => None,
        }
    }

    fn coef_value(&mut self, c: Coef) -> Value {
        match c {
            Coef::One => self.scalar(1.0),
            Coef::Of(x) => x,
        }
    }

    fn coef_neg(&mut self, c: Coef) -> Coef {
        Coef::Of(match c {
            Coef::One => self.scalar(-1.0),
            Coef::Of(x) => self.neg(&x),
        })
    }

    /// `c * factor` or `c / factor`. A factor broadcast from a scalar keeps
    /// the coefficient a scalar, so the doubling scan squares it.
    fn coef_scale(&mut self, c: Coef, factor: &Value, divide: bool) -> Coef {
        let factor = self.scalar_source(factor);
        let c = match (c, divide) {
            (Coef::One, false) => return Coef::Of(factor),
            (c, _) => self.coef_value(c),
        };
        Coef::Of(if divide {
            self.div(&c, &factor)
        } else {
            self.mul(&c, &factor)
        })
    }

    fn scalar_source(&self, x: &Value) -> Value {
        let mut source = x.clone();
        while source.ty != MlirTy::Scalar {
            match self.pointwise.get(&source.ssa).map(|p| &p.op) {
                Some(Pointwise::Broadcast(a, _) | Pointwise::Reshape(a)) => source = a.clone(),
                _ => return x.clone(),
            }
        }
        source
    }

    /// Hillis-Steele scan of `x_t = coef_t * x_(t-1) + offset_t` from `init`.
    /// After the level with shift `s`, each cell composes the last `2s` steps.
    fn doubling(
        &mut self,
        coef: Coef,
        offset: &Value,
        init: &Value,
        frame: &[Option<u64>],
    ) -> Value {
        let n = frame.last().copied().flatten().expect("static scan length");
        let mut coef = match coef {
            Coef::One => None,
            Coef::Of(a) => Some(a),
        };
        // Fold the initial carry into the first step, so that every later
        // window that reaches the first step also carries it.
        let zero = self.constants.get(&init.ssa).is_some_and(|data| {
            data.iter()
                .all(|v| matches!(v, Scalar::Real(x) if *x == 0.0) || matches!(v, Scalar::Int(0)))
        });
        let mut x = offset.clone();
        if !zero {
            let first = self.time_cells(init, 1, frame);
            let rest = self.time_constant(0.0, n - 1, frame);
            let start = self.concat_time(&first, &rest, offset);
            let start = match &coef {
                Some(a) => self.mul(a, &start),
                None => start,
            };
            x = self.add(&x, &start);
        }
        let mut fill = self.time_constant(0.0, 1, frame);
        let mut s = 1;
        while s < n {
            let shifted = self.shift(&x, s, &fill);
            let term = match &coef {
                Some(a) => self.mul(a, &shifted),
                None => shifted,
            };
            x = self.add(&x, &term);
            if 2 * s < n
                && let Some(a) = coef.take()
            {
                let earlier = if a.ty == MlirTy::Scalar {
                    a.clone()
                } else {
                    let ones = self.time_constant(1.0, s, frame);
                    self.shift(&a, s, &ones)
                };
                coef = Some(self.mul(&a, &earlier));
            }
            s *= 2;
            fill = self.time_constant(0.0, s, frame);
        }
        x
    }

    /// `value`, whose shape is the frame without the time axis, as `s` time cells.
    fn time_cells(&mut self, value: &Value, s: u64, frame: &[Option<u64>]) -> Value {
        let mut dims = frame[..frame.len() - 1].to_vec();
        dims.push(Some(1));
        let one = self.reshape(value, tensor(dims.clone()));
        if s == 1 {
            return one;
        }
        *dims.last_mut().expect("time axis") = Some(s);
        let map = (0..dims.len() as u64).collect::<Vec<_>>();
        self.broadcast_in_dim(&one, &map, tensor(dims))
    }

    fn time_constant(&mut self, x: f64, s: u64, frame: &[Option<u64>]) -> Value {
        let mut dims = frame[..frame.len() - 1].to_vec();
        dims.push(Some(s));
        self.constant(x, tensor(dims))
    }

    /// `x` moved `s` cells later along the time axis, with `fill` in front.
    fn shift(&mut self, x: &Value, s: u64, fill: &Value) -> Value {
        let dims = shape(&x.ty)
            .iter()
            .map(|d| d.expect("static scan shape"))
            .collect::<Vec<_>>();
        let axis = dims.len() - 1;
        let mut limits = dims.clone();
        limits[axis] -= s;
        let kept = self.slice(x, &vec![0; dims.len()], &limits, &vec![1; dims.len()]);
        self.concat_time(fill, &kept, x)
    }

    /// `a` then `b` along the time axis, typed and laid out like `like`.
    fn concat_time(&mut self, a: &Value, b: &Value, like: &Value) -> Value {
        let axis = shape(&like.ty).len() - 1;
        let render = |v: &Value| v.ty.render(self.dtype, ElemKind::Real);
        let rhs = format!(
            "stablehlo.concatenate {}, {}, dim = {axis} : ({}, {}) -> {}",
            a.ssa,
            b.ssa,
            render(a),
            render(b),
            render(like)
        );
        let ssa = self.pure_like(rhs, like);
        Value {
            ssa,
            ..like.clone()
        }
    }
}

/// The SSA name without a result index (`%r#1` -> `%r`).
fn base(ssa: &str) -> &str {
    ssa.split_once('#').map_or(ssa, |(name, _)| name)
}

/// Every name defined by a straight-line operation that transitively uses
/// `seed`. A use inside a region could reach the region's results through its
/// terminator, so it returns `None`.
fn dependents_of(text: &str, seed: &str) -> Option<HashSet<String>> {
    let mut set = HashSet::from([seed.to_owned()]);
    let mut depth = 0usize;
    for line in text.lines() {
        let trimmed = line.trim_start();
        let (defined, used) = match trimmed.split_once(" = ") {
            Some((lhs, rhs)) if lhs.starts_with('%') => (lhs, rhs),
            _ => ("", trimmed),
        };
        if packing::ssa_uses(used).any(|(_, name)| set.contains(name)) {
            if depth > 0 {
                return None;
            }
            set.extend(packing::ssa_uses(defined).map(|(_, name)| name.to_owned()));
        }
        let opens = line.matches('{').count();
        let closes = line.matches('}').count();
        depth = (depth + opens).checked_sub(closes)?;
    }
    Some(set)
}

#[cfg(test)]
mod tests {
    use super::dependents_of;

    #[test]
    fn dependents_follow_straight_line_uses() {
        let text = "%2 = stablehlo.multiply %0, %1 : tensor<4xf64>\n\
                    %3 = stablehlo.add %2, %1 : tensor<4xf64>\n\
                    %4 = stablehlo.negate %1 : tensor<4xf64>\n";
        let set = dependents_of(text, "%0").expect("straight line");
        assert!(set.contains("%2") && set.contains("%3"));
        assert!(!set.contains("%4"));
    }

    #[test]
    fn a_use_inside_a_region_is_refused() {
        let text = "%5:2 = stablehlo.while(%1 = %4, %2 = %4) : tensor<i64>, tensor<i64>\n\
                    cond {\n\
                    \x20 stablehlo.return %3 : tensor<i1>\n\
                    } do {\n\
                    \x20 %6 = stablehlo.add %2, %0 : tensor<i64>\n\
                    \x20 stablehlo.return %1, %6 : tensor<i64>, tensor<i64>\n\
                    }\n";
        assert!(dependents_of(text, "%0").is_none());
        assert!(
            dependents_of(text, "%4")
                .expect("operands only")
                .contains("%5")
        );
    }
}
