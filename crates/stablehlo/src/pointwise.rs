//! Typed pointwise recipes shared by consumer expansion and horizontal packing.
//! Expansion keeps batch-invariant work hoisted and preserves arithmetic order.

use super::*;

#[derive(Clone)]
pub(super) enum Pointwise {
    Unary(String, Value),
    Binary(String, Value, Value),
    Compare(String, Value, Value),
    Select(Value, Value, Value),
    Convert(Value, ElemKind),
    Broadcast(Value, Vec<u64>),
    Reshape(Value),
    Slice(Value, Vec<u64>, Vec<u64>, Vec<u64>),
}

#[derive(Clone)]
pub(super) struct Producer {
    pub value: Value,
    pub op: Pointwise,
}

impl Pointwise {
    pub(super) fn inputs(&self) -> Vec<&Value> {
        match self {
            Self::Unary(_, a)
            | Self::Convert(a, _)
            | Self::Broadcast(a, _)
            | Self::Reshape(a)
            | Self::Slice(a, ..) => vec![a],
            Self::Binary(_, a, b) | Self::Compare(_, a, b) => vec![a, b],
            Self::Select(c, a, b) => vec![c, a, b],
        }
    }

    pub(super) fn signature(&self) -> Option<(String, Vec<u64>)> {
        let name = match self {
            Self::Unary(op, _) | Self::Binary(op, ..) => op.clone(),
            Self::Compare(dir, ..) => format!("compare {dir}"),
            Self::Select(..) => "select".to_owned(),
            Self::Convert(..) => "convert".to_owned(),
            Self::Broadcast(_, dims) => return Some(("broadcast".to_owned(), dims.clone())),
            Self::Reshape(_) | Self::Slice(..) => return None,
        };
        Some((name, vec![]))
    }
}

impl Emitter<'_> {
    pub(super) fn remember_pointwise(&mut self, ssa: &str, shape_source: &Value, op: Pointwise) {
        let elem = match &op {
            Pointwise::Compare(..) => ElemKind::Bool,
            Pointwise::Convert(_, target) => *target,
            _ => shape_source.elem,
        };
        let value = Value {
            ssa: ssa.to_owned(),
            ty: shape_source.ty.clone(),
            elem,
        };
        self.pointwise
            .insert(ssa.to_owned(), Producer { value, op });
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
        let mut op = self.pointwise.get(&value.ssa)?.op.clone();
        // A semantic axes view has the same physical shape. Follow it while
        // retaining the viewed value's expansion guard and target layout.
        while let Pointwise::Reshape(ref a) = op {
            if a.ty != value.ty {
                return None;
            }
            op = self.pointwise.get(&a.ssa)?.op.clone();
        }
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
            Pointwise::Broadcast(..) | Pointwise::Reshape(_) | Pointwise::Slice(..) => return None,
        };
        self.expanded.insert(key, out.clone());
        Some(out)
    }
}
