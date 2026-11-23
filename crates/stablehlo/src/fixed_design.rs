//! Count labelled Bernoulli outcomes whose emitted logits agree for every scoring input.

use super::constants::{MAX_FOLDED_ELEMENTS, element_count};
use super::*;
use std::collections::HashSet;

#[derive(Default)]
struct Design {
    // One fixed, typed column per component of a row key. Compare the real bits,
    // including signed zero, after conversion to the emitted precision.
    columns: Vec<Vec<u64>>,
    visited: HashSet<String>,
}

impl Design {
    fn fixed_rows(&mut self, e: &Emitter, value: &Value, rows: usize, width: usize) -> Option<()> {
        if self.columns.len().checked_add(width)?.checked_mul(rows)? > MAX_FOLDED_ELEMENTS {
            return None;
        }
        for j in 0..width {
            let keys = (0..rows)
                .map(|i| match e.fixed_scalar_at(value, i * width + j)? {
                    Scalar::Real(x) => Some(x.to_bits()),
                    Scalar::Int(x) => Some(x as u64),
                    Scalar::Bool(x) => Some(u64::from(x)),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            self.columns.push(keys);
        }
        Some(())
    }

    /// A scalar broadcast shares all of its live inputs. For a pointwise
    /// operation, equal input rows imply equal results, whatever the operation.
    /// Keep the actual arithmetic in the emitted program and gather its result.
    fn visit(&mut self, e: &Emitter, value: &Value, rows: usize) -> Option<()> {
        let elements = element_count(&value.ty)?;
        if elements == 1 || !self.visited.insert(value.ssa.clone()) {
            return Some(());
        }
        if elements != rows || self.visited.len() > 64 {
            return None;
        }
        if let Some(values) = e.constants.get(&value.ssa) {
            return if values.len() == 1 {
                Some(())
            } else {
                self.fixed_rows(e, value, rows, 1)
            };
        }
        let op = &e.pointwise.get(&value.ssa)?.op;
        match op {
            Pointwise::Reshape(_)
            | Pointwise::Unary(..)
            | Pointwise::Binary(..)
            | Pointwise::ChloUnary(..)
            | Pointwise::ChloBinary(..)
            | Pointwise::Compare(..)
            | Pointwise::Convert(..)
            | Pointwise::Select(..) => {
                for input in op.inputs() {
                    self.visit(e, input, rows)?;
                }
                Some(())
            }
            Pointwise::Broadcast(input, _) if element_count(&input.ty) == Some(1) => Some(()),
            // A fixed matrix times one shared vector. Compare every ordered
            // component of each matrix row, preserving the original contraction.
            Pointwise::Dot(matrix, coefficients, 1, 0, 0) => {
                let [Some(n), Some(k)] = shape(&matrix.ty) else {
                    return None;
                };
                if *n != rows as u64 || shape(&coefficients.ty) != [Some(*k)] {
                    return None;
                }
                self.fixed_rows(e, matrix, rows, usize::try_from(*k).ok()?)
            }
            // Fixed, possibly strided views are already understood by the cache.
            _ => self.fixed_rows(e, value, rows, 1),
        }
    }
}

impl Emitter<'_> {
    pub(super) fn reduce_bernoulli_design(
        &mut self,
        probability: NodeId,
        observed: NodeId,
    ) -> Option<Value> {
        if self.target != crate::Target::Cpu
            || self.fold_design_scans
            || !self.call_frame.is_empty()
        {
            return None;
        }
        let observed_type = self.type_of(observed)?;
        let (ty, _) = crate::types::mlir_type_of_ty(observed, observed_type, self.dtype).ok()?;
        if !matches!(element_count(&ty), Some(2..=MAX_FOLDED_ELEMENTS)) {
            return None;
        }
        // A bound argument has no fixed cells. This also catches the usual
        // self-module reference to a promoted file-backed observation.
        for id in [observed, self.resolve_ref_one(observed)] {
            if let Some(y) = self.bound_tensor(id) {
                self.fixed_scalar_at(&y, 0)?;
                break;
            }
        }
        // A failed proof must leave the latent predictor, its status checks
        // and the backend's original scan choice intact.
        let mut trial = self.clone();
        let y = trial.lower_node(observed).ok()?;
        if y.ty != ty || trial.batch_rank(&y) != 0 || trial.fixed_scalar_at(&y, 0).is_none() {
            return None;
        }
        trial.fold_design_scans = true;
        let p = trial.lower_node(probability).ok()?;
        trial.fold_design_scans = false;
        if p.ty != y.ty || trial.batch_rank(&p) != 0 {
            return None;
        }
        let z = trial.invlogit_operand(&p)?;
        let result = trial.count_bernoulli_rows(&z, &y)?;
        *self = trial;
        Some(result)
    }

    fn count_bernoulli_rows(&mut self, z: &Value, y: &Value) -> Option<Value> {
        let rows = element_count(&y.ty)?;
        let mut design = Design::default();
        design.visit(self, z, rows)?;
        if design.columns.is_empty() {
            return None;
        }
        let mut groups = HashMap::new();
        let mut indices = Vec::new();
        // At most MAX_FOLDED_ELEMENTS observations contribute to any count;
        // hits, misses and their sum are exact even at f32 precision.
        let mut counts: Vec<[u32; 2]> = Vec::new();
        for i in 0..rows {
            let success = match self.fixed_scalar_at(y, i)? {
                Scalar::Int(0) | Scalar::Real(0.0) | Scalar::Bool(false) => 0,
                Scalar::Int(1) | Scalar::Real(1.0) | Scalar::Bool(true) => 1,
                _ => return None,
            };
            let key = design.columns.iter().map(|c| c[i]).collect::<Vec<_>>();
            let group = *groups.entry(key).or_insert_with(|| {
                let group = indices.len();
                indices.push(Scalar::Int(i as i64));
                counts.push([0; 2]);
                group
            });
            // A gather and its reverse scatter have a cost. Require at least
            // two observations per representative on average.
            if indices.len() > rows / 2 {
                return None;
            }
            counts[group][success] += 1;
        }
        let ty = MlirTy::Ranked(vec![Some(indices.len() as u64)]);
        let axes = Axes {
            batch: 0,
            layers: vec![1],
        };
        let hits = counts
            .iter()
            .map(|c| Scalar::Real(f64::from(c[1])))
            .collect();
        let misses = counts
            .iter()
            .map(|c| Scalar::Real(f64::from(c[0])))
            .collect();
        let hits = self.folded_constant(hits, ty.clone(), axes.clone())?;
        let misses = self.folded_constant(misses, ty.clone(), axes.clone())?;
        let indices = self.folded_constant(indices, ty, axes.clone())?;
        let flat = self.reshape_axes(z, MlirTy::Ranked(vec![Some(rows as u64)]), axes);
        let reps = self.gather(&flat, &indices, 0);
        let scores = crate::registry::binomial_logit_score(self, &reps, &hits, &misses);
        Some(self.reduce_sum(&scores))
    }
}
