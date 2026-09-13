//! Expand batched scalar producers into their consumer's cell shape without
//! changing arithmetic order. Batch-invariant work and non-pointwise producers
//! remain computed leaves.

use super::*;

#[derive(Clone)]
pub(super) enum Pointwise {
    Unary(String, Value),
    Binary(String, Value, Value),
    Compare(String, Value, Value),
    Select(Value, Value, Value),
    Convert(Value, ElemKind),
}

impl Emitter<'_> {
    pub(super) fn remember_pointwise(&mut self, ssa: &str, shape_source: &Value, op: Pointwise) {
        if self.batch_rank(shape_source) > 0 && self.cell_ty(shape_source) == MlirTy::Scalar {
            self.pointwise.insert(ssa.to_owned(), op);
        }
    }

    pub(super) fn expand_pointwise(
        &mut self,
        value: &Value,
        dims: &[u64],
        ty: &MlirTy,
        axes: &Axes,
    ) -> Option<Value> {
        let source = shape(&value.ty);
        let target = shape(ty);
        // Keep batch-invariant scalar work hoisted. Only add cell axes to an
        // existing batch; never enlarge or reinterpret its prefix.
        if source.is_empty()
            || self.batch_rank(value) != source.len()
            || axes.batch != source.len()
            || target.len() <= source.len()
            || target[..source.len()] != *source
            || !dims.iter().copied().eq(0..source.len() as u64)
        {
            return None;
        }
        let key = (value.ssa.clone(), target.to_vec(), axes.clone());
        if let Some(out) = self.expanded.get(&key) {
            return Some(out.clone());
        }
        let op = self.pointwise.get(&value.ssa)?.clone();
        let mut expand = |v: &Value| self.expand_axes(v, dims, ty.clone(), axes.clone());
        let out = match op {
            Pointwise::Unary(op, a) => {
                let a = expand(&a);
                self.unary(&op, &a)
            }
            Pointwise::Binary(op, a, b) => {
                let a = expand(&a);
                let b = expand(&b);
                self.emit_binary(&op, &a, &b)
            }
            Pointwise::Compare(dir, a, b) => {
                let a = expand(&a);
                let b = expand(&b);
                self.compare(&dir, &a, &b)
            }
            Pointwise::Select(c, a, b) => {
                let c = expand(&c);
                let a = expand(&a);
                let b = expand(&b);
                self.select(&c, &a, &b)
            }
            Pointwise::Convert(a, elem) => {
                let a = expand(&a);
                self.convert(&a, elem)
            }
        };
        self.expanded.insert(key, out.clone());
        Some(out)
    }
}
