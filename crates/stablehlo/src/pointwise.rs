//! Typed pointwise recipes shared by consumer expansion and horizontal packing.
//! Expansion keeps batch-invariant work hoisted and preserves arithmetic order.
//! Reductions participate only in packing, never pointwise expansion.

use super::*;

#[derive(Clone)]
pub(super) struct GatherDimensions {
    pub offsets: Vec<usize>,
    pub collapsed: Vec<usize>,
    pub operand_batches: Vec<usize>,
    pub index_batches: Vec<usize>,
    pub index_map: Vec<usize>,
    pub index_vector: usize,
    pub slice_sizes: Vec<u64>,
}

#[derive(Clone)]
pub(super) enum Pointwise {
    Unary(String, Value),
    ChloUnary(String, Value),
    Binary(String, Value, Value),
    Compare(String, Value, Value),
    Select(Value, Value, Value),
    Convert(Value, ElemKind),
    Broadcast(Value, Vec<u64>),
    Reshape(Value),
    Slice(Value, Vec<u64>, Vec<u64>, Vec<u64>),
    Transpose(Value, Vec<u64>),
    // In-bounds, zero-based selections, including equivalent vector concatenations.
    Gather(Value, usize, Vec<u64>),
    DynamicGather(Value, Value, GatherDimensions),
    Dot(Value, Value, usize, usize),
    Concat(Vec<Value>, usize),
    Reduce(Value, usize, String, String),
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
            | Self::ChloUnary(_, a)
            | Self::Convert(a, _)
            | Self::Broadcast(a, _)
            | Self::Reshape(a)
            | Self::Slice(a, ..)
            | Self::Transpose(a, _)
            | Self::Gather(a, ..)
            | Self::Reduce(a, ..) => vec![a],
            Self::Binary(_, a, b)
            | Self::Compare(_, a, b)
            | Self::DynamicGather(a, b, _)
            | Self::Dot(a, b, ..) => {
                vec![a, b]
            }
            Self::Select(c, a, b) => vec![c, a, b],
            Self::Concat(values, _) => values.iter().collect(),
        }
    }

    pub(super) fn signature(&self) -> Option<(String, Vec<u64>)> {
        let name = match self {
            Self::Unary(op, _) | Self::ChloUnary(op, _) | Self::Binary(op, ..) => op.clone(),
            Self::Compare(dir, ..) => format!("compare {dir}"),
            Self::Select(..) => "select".to_owned(),
            Self::Convert(..) => "convert".to_owned(),
            Self::Broadcast(_, dims) => return Some(("broadcast".to_owned(), dims.clone())),
            Self::Reduce(_, axis, op, init) => {
                return Some((format!("reduce {op} {init}"), vec![*axis as u64]));
            }
            Self::Reshape(_)
            | Self::Slice(..)
            | Self::Transpose(..)
            | Self::Gather(..)
            | Self::DynamicGather(..)
            | Self::Dot(..)
            | Self::Concat(..) => {
                return None;
            }
        };
        Some((name, vec![]))
    }

    /// Preserve segment provenance through opaque replay without exposing new
    /// broadcast recipes to the horizontal packer's hoisting decisions.
    pub(super) fn remap_segment(&self, mut rename: impl FnMut(&Value) -> Value) -> Option<Self> {
        Some(match self {
            Self::Reshape(a) => Self::Reshape(rename(a)),
            Self::Transpose(a, perm) => Self::Transpose(rename(a), perm.clone()),
            Self::Gather(a, axis, indices) => Self::Gather(rename(a), *axis, indices.clone()),
            Self::Slice(a, starts, limits, strides) => {
                Self::Slice(rename(a), starts.clone(), limits.clone(), strides.clone())
            }
            Self::Concat(values, axis) => Self::Concat(values.iter().map(rename).collect(), *axis),
            Self::Reduce(a, axis, op, init) => {
                Self::Reduce(rename(a), *axis, op.clone(), init.clone())
            }
            _ => return None,
        })
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
            // Compute parameter-only functions once per point, then broadcast
            // their values instead of repeating them along every cell axis.
            Pointwise::Unary(..) | Pointwise::ChloUnary(..) => return None,
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
            Pointwise::Broadcast(..)
            | Pointwise::Reshape(_)
            | Pointwise::Slice(..)
            | Pointwise::Transpose(..)
            | Pointwise::Gather(..)
            | Pointwise::DynamicGather(..)
            | Pointwise::Dot(..)
            | Pointwise::Concat(..)
            | Pointwise::Reduce(..) => return None,
        };
        self.expanded.insert(key, out.clone());
        Some(out)
    }
}
