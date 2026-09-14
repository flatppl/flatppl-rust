//! Tensorized callable scopes. Physical axes are `[batch prefix][cell layers]`.
//! A distinct SSA view keeps a mapped input separate from the same captured
//! array. Nesting survives independently of MLIR's flattened tensor shape.

use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Axes {
    pub batch: usize,
    pub layers: Vec<usize>,
}

pub(super) fn shape(ty: &MlirTy) -> &[Option<u64>] {
    match ty {
        MlirTy::Scalar => &[],
        MlirTy::Ranked(dims) => dims,
        _ => panic!("expected numeric tensor shape"),
    }
}

pub(super) fn tensor(dims: Vec<Option<u64>>) -> MlirTy {
    if dims.is_empty() {
        MlirTy::Scalar
    } else {
        MlirTy::Ranked(dims)
    }
}

impl Emitter<'_> {
    pub(crate) fn batch_rank(&self, v: &Value) -> usize {
        self.axes.get(&v.ssa).map_or(0, |a| a.batch)
    }

    pub(crate) fn cell_ty(&self, v: &Value) -> MlirTy {
        tensor(shape(&v.ty)[self.batch_rank(v)..].to_vec())
    }

    pub(crate) fn constant_like(&mut self, x: f64, v: &Value) -> Value {
        let out = self.const_lit_like(&render_float_literal(x), v);
        self.remember_constant(&out.ssa, [Scalar::Real(x)]);
        out
    }

    pub(crate) fn inf_like(&mut self, v: &Value) -> Value {
        self.const_lit_like(pos_inf_literal(self.dtype), v)
    }

    pub(crate) fn const_lit_like(&mut self, lit: &str, v: &Value) -> Value {
        let ty = v.ty.render(self.dtype, ElemKind::Real);
        let ssa = self.pure_like(format!("stablehlo.constant dense<{lit}> : {ty}"), v);
        Value {
            ssa,
            ty: v.ty.clone(),
            elem: ElemKind::Real,
        }
    }

    pub(crate) fn fill_cell(&mut self, v: &Value, cell: MlirTy) -> Value {
        let batch = self.batch_rank(v);
        let mut dims = shape(&v.ty)[..batch].to_vec();
        dims.extend_from_slice(shape(&cell));
        let map = (0..batch as u64).collect::<Vec<_>>();
        let axes = Axes {
            batch,
            layers: vec![shape(&cell).len()],
        };
        self.expand_axes(v, &map, tensor(dims), axes)
    }

    pub(crate) fn reshape_cell(&mut self, v: &Value, cell: MlirTy) -> Value {
        let mut dims = shape(&v.ty)[..self.batch_rank(v)].to_vec();
        dims.extend_from_slice(shape(&cell));
        self.reshape(v, tensor(dims))
    }

    /// Align only batch prefixes; matrix operands may have different cell shapes.
    pub(super) fn broadcast_batches(&mut self, a: &Value, b: &Value) -> (Value, Value) {
        let rank = self.batch_rank(a).max(self.batch_rank(b));
        let mut frame = vec![Some(1); rank];
        for v in [a, b] {
            for (out, &dim) in frame.iter_mut().zip(&shape(&v.ty)[..self.batch_rank(v)]) {
                if *out == Some(1) {
                    *out = dim;
                } else {
                    assert!(dim == Some(1) || *out == dim, "incompatible batch extents");
                }
            }
        }
        let mut expand = |v: &Value| {
            let mut axes = self.axes_of(v);
            let mut dims = frame.clone();
            dims.extend_from_slice(&shape(&v.ty)[axes.batch..]);
            let map = (0..axes.batch as u64)
                .chain((rank..dims.len()).map(|i| i as u64))
                .collect::<Vec<_>>();
            axes.batch = rank;
            self.expand_axes(v, &map, tensor(dims), axes)
        };
        (expand(a), expand(b))
    }

    pub(super) fn axes_of(&self, v: &Value) -> Axes {
        self.axes.get(&v.ssa).cloned().unwrap_or_else(|| Axes {
            batch: 0,
            layers: if shape(&v.ty).is_empty() {
                vec![]
            } else {
                vec![shape(&v.ty).len()]
            },
        })
    }

    pub(super) fn remember_axes(&mut self, ssa: &str, axes: Axes) {
        if axes.batch != 0 || axes.layers.len() > 1 {
            self.axes.insert(ssa.to_owned(), axes);
        }
    }

    pub(super) fn pure_axes(&mut self, rhs: String, axes: Axes) -> String {
        let special = (axes.batch != 0 || axes.layers.len() > 1).then_some(axes.clone());
        let key = (rhs, special);
        if let Some(ssa) = self.pure_ops.get(&key) {
            return ssa.clone();
        }
        let ssa = self.fresh();
        self.push(&format!("{ssa} = {}", key.0));
        self.remember_axes(&ssa, axes);
        self.pure_ops.insert(key, ssa.clone());
        ssa
    }

    pub(super) fn pure_like(&mut self, rhs: String, v: &Value) -> String {
        self.pure_axes(rhs, self.axes_of(v))
    }

    /// A no-copy semantic view; backend canonicalization removes the reshape.
    pub(super) fn axes_view(&mut self, v: &Value, axes: Axes) -> Value {
        if self.axes_of(v) == axes {
            return v.clone();
        }
        let ty = v.ty.render(self.dtype, v.elem);
        let ssa = self.pure_axes(
            format!("stablehlo.reshape {} : ({ty}) -> {ty}", v.ssa),
            axes,
        );
        let out = Value { ssa, ..v.clone() };
        self.copy_constant(v, &out);
        self.remember_pointwise(&out.ssa, &out, Pointwise::Reshape(v.clone()));
        out
    }

    pub(super) fn typed_axes(&mut self, id: NodeId, v: Value) -> Value {
        if !matches!(v.ty, MlirTy::Scalar | MlirTy::Ranked(_)) {
            return v;
        }
        let mut ty = self.m.type_of(id);
        let mut layers = Vec::new();
        loop {
            match ty {
                Some(Type::Array { shape, elem }) => {
                    layers.push(shape.len());
                    ty = Some(elem);
                }
                Some(Type::TVector { elem, .. }) => {
                    layers.push(1);
                    ty = Some(elem);
                }
                _ => break,
            }
        }
        let batch = self.batch_rank(&v);
        if matches!(ty, Some(Type::Scalar(_)))
            && layers.iter().sum::<usize>() + batch == shape(&v.ty).len()
        {
            return self.axes_view(&v, Axes { batch, layers });
        }
        v
    }

    pub(super) fn expand_axes(&mut self, v: &Value, dims: &[u64], ty: MlirTy, axes: Axes) -> Value {
        if v.ty == ty && dims.iter().copied().eq(0..shape(&ty).len() as u64) {
            return self.axes_view(v, axes);
        }
        if let Some(value) = self.expand_pointwise(v, dims, &ty, &axes) {
            return value;
        }
        let from = v.ty.render(self.dtype, v.elem);
        let to = ty.render(self.dtype, v.elem);
        let map = dims
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let ssa = self.pure_axes(
            format!(
                "stablehlo.broadcast_in_dim {}, dims = [{map}] : ({from}) -> {to}",
                v.ssa
            ),
            axes,
        );
        let out = Value {
            ssa,
            ty,
            elem: v.elem,
        };
        if dims.windows(2).all(|pair| pair[0] < pair[1]) {
            self.copy_constant(v, &out);
        }
        self.remember_pointwise(
            &out.ssa,
            &out,
            Pointwise::Broadcast(v.clone(), dims.to_vec()),
        );
        out
    }

    /// Align batch axes separately from cell axes. Missing batch levels never
    /// consume a cell axis, even when their sizes happen to be equal.
    pub(super) fn broadcast_batched_pair(&mut self, a: &Value, b: &Value) -> (Value, Value) {
        let aa = self.axes_of(a);
        let ba = self.axes_of(b);
        let ad = shape(&a.ty);
        let bd = shape(&b.ty);
        let batch = aa.batch.max(ba.batch);
        let ac = &ad[aa.batch..];
        let bc = &bd[ba.batch..];
        assert!(
            ac.is_empty() || bc.is_empty() || ac.len() == bc.len(),
            "incompatible cell ranks"
        );
        let common = |x, y| {
            if x == y {
                x
            } else if x == Some(1) {
                y
            } else if y == Some(1) {
                x
            } else {
                panic!("incompatible axis extents {x:?} and {y:?}")
            }
        };
        let mut dims = (0..batch)
            .map(|i| {
                common(
                    if i < aa.batch { ad[i] } else { Some(1) },
                    if i < ba.batch { bd[i] } else { Some(1) },
                )
            })
            .collect::<Vec<_>>();
        let cell = if ac.is_empty() {
            bc.to_vec()
        } else if bc.is_empty() {
            ac.to_vec()
        } else {
            ac.iter().zip(bc).map(|(&x, &y)| common(x, y)).collect()
        };
        dims.extend(cell);
        let layers = if ac.is_empty() {
            ba.layers.clone()
        } else {
            aa.layers.clone()
        };
        let axes = Axes { batch, layers };
        let ty = tensor(dims);
        let map = |n: usize, rank: usize| {
            (0..n as u64)
                .chain((batch..batch + rank - n).map(|i| i as u64))
                .collect::<Vec<_>>()
        };
        let a = self.expand_axes(a, &map(aa.batch, ad.len()), ty.clone(), axes.clone());
        let b = self.expand_axes(b, &map(ba.batch, bd.len()), ty, axes);
        (a, b)
    }

    /// Restore a callable's result into its caller's frame, including an
    /// invariant body that never used any iterated input.
    pub(super) fn finish_broadcast(
        &mut self,
        value: &Value,
        frame: &[Option<u64>],
        parent: usize,
    ) -> Value {
        let mut axes = self.axes_of(value);
        let mut dims = frame.to_vec();
        dims.extend_from_slice(&shape(&value.ty)[axes.batch..]);
        let map = (0..axes.batch as u64)
            .chain((frame.len()..dims.len()).map(|i| i as u64))
            .collect::<Vec<_>>();
        let added = frame.len() - parent;
        axes.batch = parent;
        if added > 0 {
            axes.layers.insert(0, added);
        }
        self.expand_axes(value, &map, tensor(dims), axes)
    }

    /// Consume exactly the outer collection layer. Captured values retain their
    /// original layout even if a callable argument names the same SSA value.
    pub(super) fn enter_broadcast(
        &mut self,
        id: NodeId,
        values: &mut [(Symbol, Value)],
    ) -> Result<Vec<Option<u64>>, EmitError> {
        let parent = self.broadcast_frame.len();
        let mut outer: Option<Vec<Option<u64>>> = None;
        for (_, v) in values.iter() {
            let axes = self.axes_of(v);
            let Some(&rank) = axes.layers.first() else {
                continue;
            };
            let dims = &shape(&v.ty)[axes.batch..axes.batch + rank];
            if let Some(common) = &mut outer {
                if common.len() != rank {
                    return Err(EmitError::at(id, "broadcast: collection ranks differ"));
                }
                for (a, &b) in common.iter_mut().zip(dims) {
                    if *a == Some(1) {
                        *a = b;
                    } else if *a != b && b != Some(1) {
                        return Err(EmitError::at(id, "broadcast: collection extents differ"));
                    }
                }
            } else {
                outer = Some(dims.to_vec());
            }
        }
        let mut frame = self.broadcast_frame.clone();
        frame.extend(outer.unwrap_or_default());
        for (_, v) in values.iter_mut() {
            let mut axes = self.axes_of(v);
            if axes.layers.is_empty() {
                continue;
            }
            let rank = axes.layers.remove(0);
            // A singleton collection supplies the same cell at every point.
            // Keep that cell outside the new frame: eager expansion would
            // repeat parameter-only arithmetic inside the observation loop.
            let source = shape(&v.ty);
            if source[axes.batch..axes.batch + rank]
                .iter()
                .all(|&d| d == Some(1))
            {
                let mut dims = source[..axes.batch].to_vec();
                dims.extend_from_slice(&source[axes.batch + rank..]);
                *v = self.reshape_axes(v, tensor(dims), axes);
                continue;
            }
            // Missing parent axes stay invariant. Expanding them to the whole
            // frame here would repeat observation-only work for every point
            // and lose compact constant payloads before folding can use them.
            let mut dims = source[..axes.batch].to_vec();
            dims.resize(parent, Some(1));
            dims.extend_from_slice(&source[axes.batch..]);
            let map = (0..axes.batch as u64)
                .chain((parent..parent + rank).map(|i| i as u64))
                .chain((frame.len()..dims.len()).map(|i| i as u64))
                .collect::<Vec<_>>();
            axes.batch = frame.len();
            *v = self.expand_axes(v, &map, tensor(dims), axes);
        }
        Ok(std::mem::replace(&mut self.broadcast_frame, frame))
    }

    pub(super) fn vector_batched(&mut self, elems: &[Value]) -> Value {
        let mut target = elems[0].clone();
        for v in &elems[1..] {
            target = self.broadcast_batched_pair(&target, v).0;
        }
        let mut axes = self.axes_of(&target);
        let batch = axes.batch;
        axes.layers.insert(0, 1);
        let mut dims = shape(&target.ty).to_vec();
        dims.insert(batch, Some(1));
        let part_ty = tensor(dims.clone());
        let parts = elems
            .iter()
            .map(|v| {
                let aligned = self.broadcast_batched_pair(v, &target).0;
                self.reshape_axes(&aligned, part_ty.clone(), axes.clone())
            })
            .collect::<Vec<_>>();
        if parts.len() == 1 {
            return parts[0].clone();
        }
        dims[batch] = Some(parts.len() as u64);
        let ty = tensor(dims);
        let names = parts
            .iter()
            .map(|v| v.ssa.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = parts
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>()
            .join(", ");
        let result = ty.render(self.dtype, target.elem);
        let ssa = self.pure_axes(
            format!("stablehlo.concatenate {names}, dim = {batch} : ({types}) -> {result}"),
            axes,
        );
        Value {
            ssa,
            ty,
            elem: target.elem,
        }
    }
}
