//! Standard-module entry points for Hermitian eigenpairs and factor solves.

use super::*;
use flatppl_core::ScalarType;

type Pair = (Value, Value);

impl Emitter<'_> {
    fn eigh_call(&self, id: NodeId) -> Option<(NodeId, bool)> {
        let id = self.resolve_tuple_projections(id);
        let Node::Call(call) = self.node(id) else {
            return None;
        };
        let (callee, args, broadcast) = match call.head {
            CallHead::User(callee) => (callee, call.args.as_ref(), false),
            CallHead::Builtin(head) if self.resolve(head) == "broadcast" => {
                let (&callee, args) = call.args.split_first()?;
                (callee, args, true)
            }
            _ => return None,
        };
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        let [arg] = args else { return None };
        (module == "ext-linear-algebra" && member == "eigh").then_some((*arg, broadcast))
    }

    fn emit_hermitian(
        &mut self,
        id: NodeId,
        arg: NodeId,
        broadcast: bool,
        vectors: bool,
    ) -> Result<hermitian::HermitianEigen, EmitError> {
        let (re, im) = crate::complex::parts(self, arg, &mut HashMap::new())?;
        let (re, im) = self.broadcast_pair(&re, &im);
        let mut values = [self.typed_axes(arg, re), self.typed_axes(arg, im)];
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let real = matches!(
            self.type_of(arg),
            Some(Type::Array { elem, .. }) if matches!(elem.as_ref(), Type::Scalar(ScalarType::Real | ScalarType::Integer))
        );
        let [re, im] = values;
        let mut result = self.matrix_hermitian(id, (re, im), vectors, real)?;
        if let Some(parent) = parent {
            let frame = self.broadcast_frame.clone();
            result.values = self.finish_broadcast(&result.values, &frame, parent.len());
            result.vectors = result.vectors.map(|(re, im)| {
                (
                    self.finish_broadcast(&re, &frame, parent.len()),
                    self.finish_broadcast(&im, &frame, parent.len()),
                )
            });
            self.broadcast_frame = parent;
        }
        Ok(result)
    }

    pub(super) fn lower_eigh_record(
        &mut self,
        id: NodeId,
        field: &str,
    ) -> Option<Result<Lowered, EmitError>> {
        let (arg, broadcast) = self.eigh_call(id)?;
        Some((|| {
            let complex = crate::complex::complex_leaf(self.type_of(arg));
            let result = self.emit_hermitian(id, arg, broadcast, field == "vectors" || !complex)?;
            self.bind_field(id, "values".into(), Lowered::Tensor(result.values.clone()));
            if !complex && let Some((vectors, _)) = &result.vectors {
                self.bind_field(id, "vectors".into(), Lowered::Tensor(vectors.clone()));
            }
            if field == "values" {
                return Ok(Lowered::Tensor(result.values));
            }
            if field != "vectors" {
                return Err(EmitError::at(id, "eigh has fields values and vectors"));
            }
            if complex {
                return Err(EmitError::at(
                    id,
                    "complex Hermitian eigenvectors require real/imaginary projections",
                ));
            }
            let vectors = result.vectors.expect("requested eigenvectors").0;
            Ok(Lowered::Tensor(vectors))
        })())
    }

    pub(crate) fn lower_eigh_field_parts(
        &mut self,
        id: NodeId,
        field: &str,
    ) -> Option<Result<Pair, EmitError>> {
        let (arg, broadcast) = self.eigh_call(id)?;
        Some((|| {
            let result = self.emit_hermitian(id, arg, broadcast, field == "vectors")?;
            match field {
                "values" => {
                    let zero = self.constant_like(0.0, &result.values);
                    Ok((result.values, zero))
                }
                "vectors" => Ok(result.vectors.expect("requested eigenvectors")),
                _ => Err(EmitError::at(id, "eigh has fields values and vectors")),
            }
        })())
    }

    pub(super) fn lower_ext_linalg_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra"
            || !matches!(
                member.as_str(),
                "solve_triangular" | "lu_solve" | "cholesky_solve" | "eigvalsh"
            )
        {
            return None;
        }
        Some((|| {
            if member == "eigvalsh" {
                let [arg] = args else {
                    return Err(EmitError::at(id, "eigvalsh requires one matrix"));
                };
                return Ok(self.emit_hermitian(id, *arg, broadcast, false)?.values);
            }
            let result = self.emit_factor_solve(id, &member, args, broadcast)?;
            if crate::complex::complex_leaf(self.type_of(id)) {
                return Err(EmitError::at(
                    id,
                    "complex solve requires real/imaginary projections",
                ));
            }
            Ok(result.0)
        })())
    }

    pub(crate) fn lower_ext_linalg_parts(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Pair, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "ext-linear-algebra"
            || !matches!(
                member.as_str(),
                "solve_triangular" | "lu_solve" | "cholesky_solve"
            )
        {
            return None;
        }
        Some(self.emit_factor_solve(id, &member, args, broadcast))
    }

    fn fixed_solve_flag(
        &mut self,
        id: NodeId,
        arg: Option<&NodeId>,
        default: bool,
    ) -> Result<Value, EmitError> {
        let Some(arg) = arg else {
            return Ok(self.bool_value_const(default));
        };
        let flag = self.lower_node(*arg)?;
        if self.cell_ty(&flag) != MlirTy::Scalar || self.batch_rank(&flag) != 0 {
            return Err(EmitError::at(
                id,
                "triangular solve flags require fixed scalar boolean arguments",
            ));
        }
        match self.fixed_scalar_at(&flag, 0) {
            Some(Scalar::Bool(value)) => Ok(self.bool_value_const(value)),
            _ => Err(EmitError::at(
                id,
                "triangular solve flags require fixed boolean arguments",
            )),
        }
    }

    fn solve_pair(
        &mut self,
        arg: NodeId,
        memo: &mut HashMap<NodeId, Pair>,
    ) -> Result<Pair, EmitError> {
        let (re, im) = crate::complex::parts(self, arg, memo)?;
        let (re, im) = self.broadcast_pair(&re, &im);
        Ok((self.typed_axes(arg, re), self.typed_axes(arg, im)))
    }

    fn emit_factor_solve(
        &mut self,
        id: NodeId,
        member: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Pair, EmitError> {
        let [a, b, flags @ ..] = args else {
            return Err(EmitError::at(id, "factor solve requires factors and RHS"));
        };
        let lower = self.fixed_solve_flag(id, flags.first(), true)?;
        let unit = self.fixed_solve_flag(id, flags.get(1), false)?;
        let mut memo = HashMap::new();
        let mut values = Vec::new();
        if member == "lu_solve" {
            if let Some(factors) = self.lower_lu_record_parts(*a) {
                for (re, im) in factors? {
                    values.extend([re, im]);
                }
            } else {
                for field in ["P", "L", "U"] {
                    let (re, im) = crate::complex::field_parts(self, id, *a, field, &mut memo)?;
                    let (re, im) = self.broadcast_pair(&re, &im);
                    values.extend([re, im]);
                }
            }
        } else {
            let pair = self.solve_pair(*a, &mut memo)?;
            values.extend([pair.0, pair.1]);
        }
        let pair = self.solve_pair(*b, &mut memo)?;
        values.extend([pair.0, pair.1]);
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let one = self.bool_value_const(true);
        let no = self.bool_value_const(false);
        let result = if member == "lu_solve" {
            let p = (values[0].clone(), values[1].clone());
            let l = (values[2].clone(), values[3].clone());
            let u = (values[4].clone(), values[5].clone());
            let b = (values[6].clone(), values[7].clone());
            self.factor_shapes(id, &[&p.0, &l.0, &u.0], &b.0)?;
            let valid = self.validate_permutation(&p);
            let batch = self.batch_rank(&l.0);
            let lr = self.diagonal_axes(&l.0, batch, batch + 1);
            let li = self.diagonal_axes(&l.1, batch, batch + 1);
            let scalar_one = self.scalar(1.0);
            let scalar_zero = self.scalar(0.0);
            let real_unit = self.compare("EQ", &lr, &scalar_one);
            let imaginary_zero = self.compare("EQ", &li, &scalar_zero);
            let unit_diagonal = self.and(&real_unit, &imaginary_zero);
            let unit_diagonal = self.reduce_all(&unit_diagonal);
            let valid = self.and(&valid, &unit_diagonal);
            self.record_runtime_error(&valid, 12);
            let vector = matches!(self.cell_ty(&b.0), MlirTy::Ranked(d) if d.len() == 1);
            let pb = if vector {
                (self.matvec(&p.0, &b.0), self.matvec(&p.0, &b.1))
            } else {
                self.factor_matmul(&p, &b)
            };
            let y = self.matrix_factor_solve(id, &l, &pb, &one, &one)?;
            let x = self.matrix_factor_solve(id, &u, &y, &no, &no)?;
            let nan = self.scalar(f64::NAN);
            (
                self.select(&valid, &x.0, &nan),
                self.select(&valid, &x.1, &nan),
            )
        } else {
            let a = (values[0].clone(), values[1].clone());
            let b = (values[2].clone(), values[3].clone());
            self.factor_shapes(id, &[&a.0], &b.0)?;
            if member == "cholesky_solve" {
                let batch = self.batch_rank(&a.0);
                let dr = self.diagonal_axes(&a.0, batch, batch + 1);
                let di = self.diagonal_axes(&a.1, batch, batch + 1);
                let zero = self.scalar(0.0);
                let positive = self.compare("GT", &dr, &zero);
                let real = self.compare("EQ", &di, &zero);
                let valid = self.and(&positive, &real);
                let valid = self.reduce_all(&valid);
                self.record_runtime_error(&valid, 12);
                let y = self.matrix_factor_solve(id, &a, &b, &one, &no)?;
                let adjoint = self.factor_adjoint(&a);
                let x = self.matrix_factor_solve(id, &adjoint, &y, &no, &no)?;
                let nan = self.scalar(f64::NAN);
                (
                    self.select(&valid, &x.0, &nan),
                    self.select(&valid, &x.1, &nan),
                )
            } else {
                self.matrix_factor_solve(id, &a, &b, &lower, &unit)?
            }
        };
        Ok(if let Some(parent) = parent {
            let frame = self.broadcast_frame.clone();
            let result = (
                self.finish_broadcast(&result.0, &frame, parent.len()),
                self.finish_broadcast(&result.1, &frame, parent.len()),
            );
            self.broadcast_frame = parent;
            result
        } else {
            result
        })
    }

    fn factor_shapes(&self, id: NodeId, factors: &[&Value], rhs: &Value) -> Result<(), EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(factors[0]) else {
            return Err(EmitError::at(
                id,
                "factor solve requires a static square matrix",
            ));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(
                id,
                "factor solve requires static matrix dimensions",
            ));
        };
        if n != m
            || *n > i32::MAX as u64
            || factors
                .iter()
                .any(|v| self.cell_ty(v) != self.cell_ty(factors[0]))
        {
            return Err(EmitError::at(
                id,
                "factor matrices must have the same square shape",
            ));
        }
        let MlirTy::Ranked(rhs) = self.cell_ty(rhs) else {
            return Err(EmitError::at(
                id,
                "factor solve requires a vector or matrix RHS",
            ));
        };
        if !(1..=2).contains(&rhs.len()) || rhs.first() != Some(&Some(*n)) {
            return Err(EmitError::at(
                id,
                "factor solve RHS rows must match the square matrix",
            ));
        }
        Ok(())
    }

    fn validate_permutation(&mut self, p: &Pair) -> Value {
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let zeros = self.compare("EQ", &p.0, &zero);
        let ones = self.compare("EQ", &p.0, &one);
        let entries = self.or(&zeros, &ones);
        let real = self.compare("EQ", &p.1, &zero);
        let entries = self.and(&entries, &real);
        let entries = self.reduce_all(&entries);
        let row_sum = self.reduce_sum_last_axis(&p.0);
        let transpose = self.svd_transpose(&p.0);
        let col_sum = self.reduce_sum_last_axis(&transpose);
        let rows = self.compare("EQ", &row_sum, &one);
        let cols = self.compare("EQ", &col_sum, &one);
        let sums = self.and(&rows, &cols);
        let sums = self.reduce_all(&sums);
        self.and(&entries, &sums)
    }
}
