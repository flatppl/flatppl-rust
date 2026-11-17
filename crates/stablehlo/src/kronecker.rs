//! §09 "Module ext-linear-algebra": Kronecker products preserve matrix cells.

use super::*;

impl Emitter<'_> {
    pub(crate) fn lower_kronecker_parts(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<(Value, Value), EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra" || member != "kron" {
            return None;
        }
        Some((|| {
            let [left, right] = args else {
                return Err(EmitError::at(id, "kron requires two matrices"));
            };
            let mut memo = HashMap::new();
            let (ar, ai) = crate::complex::parts(self, *left, &mut memo)?;
            let (br, bi) = crate::complex::parts(self, *right, &mut memo)?;
            // A real operand's zero imaginary coordinate has its matrix shape.
            let (ar, ai) = self.broadcast_pair(&ar, &ai);
            let (br, bi) = self.broadcast_pair(&br, &bi);
            let mut values = [
                self.typed_axes(*left, ar),
                self.typed_axes(*left, ai),
                self.typed_axes(*right, br),
                self.typed_axes(*right, bi),
            ];
            let parent = if broadcast {
                Some(self.enter_broadcast(id, &mut values)?)
            } else {
                None
            };
            let rr = self.kronecker_product(id, &values[0], &values[2])?;
            let ii = self.kronecker_product(id, &values[1], &values[3])?;
            let ri = self.kronecker_product(id, &values[0], &values[3])?;
            let ir = self.kronecker_product(id, &values[1], &values[2])?;
            let re = self.sub(&rr, &ii);
            let im = self.add(&ri, &ir);
            Ok(if let Some(parent) = parent {
                let frame = self.broadcast_frame.clone();
                let re = self.finish_broadcast(&re, &frame, parent.len());
                let im = self.finish_broadcast(&im, &frame, parent.len());
                self.broadcast_frame = parent;
                (re, im)
            } else {
                (re, im)
            })
        })())
    }
    pub(super) fn lower_kronecker_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra" || member != "kron" {
            return None;
        }
        Some(self.emit_kronecker(id, args, broadcast))
    }

    fn emit_kronecker(
        &mut self,
        id: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        let [left, right] = args else {
            return Err(EmitError::at(id, "kron requires two matrices"));
        };
        let a = self.lower_node(*left)?;
        let b = self.lower_node(*right)?;
        let mut values = [self.typed_axes(*left, a), self.typed_axes(*right, b)];
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let result = self.kronecker_product(id, &values[0], &values[1])?;
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

    fn kronecker_product(&mut self, id: NodeId, a: &Value, b: &Value) -> Result<Value, EmitError> {
        let mut values = [a.clone(), b.clone()];
        let mut sizes = Vec::with_capacity(2);
        for input in &mut values {
            let MlirTy::Ranked(dims) = self.cell_ty(input) else {
                return Err(EmitError::at(id, "kron requires static matrix cells"));
            };
            let [Some(rows), Some(cols)] = dims.as_slice() else {
                return Err(EmitError::at(id, "kron requires static matrix cells"));
            };
            sizes.push((*rows, *cols));
            // §03 arithmetic promotes booleans to integers. Keep integer
            // products integer instead of forcing every matrix to real.
            if input.elem == ElemKind::Bool {
                *input = self.convert(input, ElemKind::Int);
            }
        }
        let [(m, n), (p, q)] = sizes.as_slice() else {
            unreachable!()
        };
        let rows = m.checked_mul(*p).filter(|&n| n <= i64::MAX as u64);
        let cols = n.checked_mul(*q).filter(|&n| n <= i64::MAX as u64);
        let (Some(rows), Some(cols)) = (rows, cols) else {
            return Err(EmitError::at(id, "kron output dimensions overflow"));
        };
        let (a, b) = self.broadcast_batches(&values[0], &values[1]);
        let batch = self.batch_rank(&a);
        let prefix = shape(&a.ty)[..batch].to_vec();
        let mut a_dims = prefix.clone();
        a_dims.extend([Some(*m), Some(1), Some(*n), Some(1)]);
        let mut b_dims = prefix.clone();
        b_dims.extend([Some(1), Some(*p), Some(1), Some(*q)]);
        let a = self.reshape(&a, MlirTy::Ranked(a_dims));
        let b = self.reshape(&b, MlirTy::Ranked(b_dims));
        // The four axes are (row A, row B, column A, column B), so the
        // reshape gives K[i*p+k, j*q+l] = A[i,j] * B[k,l].
        let product = self.mul(&a, &b);
        let mut output_dims = prefix;
        output_dims.extend([Some(rows), Some(cols)]);
        Ok(self.reshape(&product, MlirTy::Ranked(output_dims)))
    }
}
