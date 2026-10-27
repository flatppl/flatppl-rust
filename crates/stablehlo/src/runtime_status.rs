//! Device-side status values for runtime checks.

use super::*;

pub(super) fn append_runtime_status<'a>(
    rets: &mut Vec<&'a Value>,
    status: Option<&'a Value>,
) -> String {
    let Some(status) = status else {
        return String::new();
    };
    let index = rets
        .iter()
        .position(|value| value.ssa == status.ssa)
        .unwrap_or_else(|| {
            rets.push(status);
            rets.len() - 1
        });
    format!(" attributes {{flatppl.runtime_status = {index} : i64}}")
}

impl Emitter<'_> {
    pub(super) fn final_runtime_status(&mut self) -> Option<Value> {
        let status = self.runtime_status.clone()?;
        if self.runtime_batch_rank == 0 {
            return Some(status);
        }
        let frame = self.call_frame[..self.runtime_batch_rank].to_vec();
        Some(self.finish_broadcast(&status, &frame, 0))
    }

    /// Number of leading physical host batch axes to retain in the status.
    pub fn set_runtime_batch_rank(&mut self, rank: usize) {
        self.runtime_batch_rank = rank;
    }

    /// The accumulated first error, or no status when no check was emitted.
    pub fn runtime_status(&self) -> Option<Value> {
        self.runtime_status.clone()
    }

    /// Record the first failure per host batch element. Callable and authored
    /// cell axes describe work within one call and are reduced together.
    pub(crate) fn record_runtime_error(&mut self, valid: &Value, code: i64) {
        assert!(code > 0, "runtime error codes must be positive");
        if self.runtime_batch_rank > 0
            && self.call_frame.len() >= self.runtime_batch_rank
            && self.broadcast_frame.len() < self.runtime_batch_rank
        {
            self.runtime_unmapped_check = true;
        }
        let valid = Value {
            elem: ElemKind::Bool,
            ..valid.clone()
        };
        let rank = shape(&valid.ty).len();
        let batch = self.batch_rank(&valid).min(self.runtime_batch_rank);
        let valid = self.axes_view(
            &valid,
            Axes {
                batch,
                layers: if rank == batch {
                    vec![]
                } else {
                    vec![rank - batch]
                },
            },
        );
        let valid = if rank == 0 {
            valid
        } else {
            self.reduce_boolean("stablehlo.and", "true", &valid)
        };
        let zero = self.int_value_const(0);
        let failure = self.int_value_const(code);
        let mut current = self.select(&valid, &zero, &failure);
        if self.runtime_batch_rank > 0 && self.call_frame.len() >= self.runtime_batch_rank {
            let frame = self.call_frame[..self.runtime_batch_rank].to_vec();
            current = self.finish_broadcast(&current, &frame, frame.len());
        }
        self.runtime_status = Some(match self.runtime_status.take() {
            Some(previous) => self.merge_runtime_status(&previous, &current),
            None => current,
        });
    }

    /// Keep the earlier error, taking the later one only if it was clear.
    pub(super) fn merge_runtime_status(&mut self, previous: &Value, current: &Value) -> Value {
        let zero = self.int_value_const(0);
        let clear = self.compare("EQ", previous, &zero);
        self.select(&clear, current, previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_reduces_call_axes_and_retains_host_axes() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        e.set_runtime_batch_rank(1);
        let valid = Value {
            ssa: "%valid".into(),
            ty: MlirTy::Ranked(vec![Some(2), Some(3), Some(4)]),
            elem: ElemKind::Bool,
        };
        e.remember_axes(
            &valid.ssa,
            Axes {
                batch: 2,
                layers: vec![1],
            },
        );
        e.record_runtime_error(&valid, 1);
        assert_eq!(
            e.runtime_status().unwrap().ty,
            MlirTy::Ranked(vec![Some(2)])
        );
        assert!(e.body.contains("across dimensions = [1]"));
        let scratch = e.scratch_emitter();
        assert_eq!(scratch.runtime_batch_rank, 1);
        assert!(scratch.runtime_status().is_none());
    }

    #[test]
    fn partial_host_axes_share_the_complete_status_frame() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        e.set_runtime_batch_rank(2);
        e.call_frame = vec![Some(2), Some(3)];
        e.broadcast_frame = e.call_frame.clone();
        let first = Value {
            ssa: "%a".into(),
            ty: MlirTy::Ranked(vec![Some(2)]),
            elem: ElemKind::Bool,
        };
        let second = Value {
            ssa: "%b".into(),
            ty: MlirTy::Ranked(vec![Some(1), Some(3)]),
            elem: ElemKind::Bool,
        };
        e.remember_axes(
            &first.ssa,
            Axes {
                batch: 1,
                layers: vec![],
            },
        );
        e.remember_axes(
            &second.ssa,
            Axes {
                batch: 2,
                layers: vec![],
            },
        );
        e.record_runtime_error(&first, 1);
        e.record_runtime_error(&second, 2);
        let status = e.runtime_status().unwrap();
        assert_eq!(status.ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
        assert_eq!(e.batch_rank(&status), 2);
    }

    #[test]
    fn checked_loop_carries_status_and_rejects_condition_checks() {
        let module = Module::new();
        let mut e = Emitter::new(&module, Dtype::F32);
        let zero = e.int_value_const(0);
        let ty = zero.ty.render(e.dtype, zero.elem);
        let result = e
            .try_while_loop(
                std::slice::from_ref(&zero),
                std::slice::from_ref(&ty),
                |e, args| {
                    let two = e.int_value_const(2);
                    e.compare("LT", &args[0], &two)
                },
                |e, args| {
                    let one = e.int_value_const(1);
                    let valid = e.compare("NE", &args[0], &one);
                    e.record_runtime_error(&valid, 1);
                    Ok(vec![e.add(&args[0], &one)])
                },
            )
            .unwrap();
        let status = e.runtime_status().unwrap();
        assert!(status.ssa.ends_with("#1"));
        assert!(result[0].ssa.ends_with("#0"));
        assert!(e.body.contains(":2 = stablehlo.while"));
        assert!(e.body.contains("stablehlo.return"));

        let mut e = Emitter::new(&module, Dtype::F32);
        let zero = e.int_value_const(0);
        let ty = zero.ty.render(e.dtype, zero.elem);
        let error = e
            .try_while_loop(
                std::slice::from_ref(&zero),
                std::slice::from_ref(&ty),
                |e, args| {
                    let valid = e.bool_value_const(true);
                    e.record_runtime_error(&valid, 1);
                    e.compare("EQ", &args[0], &zero)
                },
                |_, args| Ok(args.to_vec()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("while condition"));
        assert!(e.runtime_status().is_none());

        e.set_runtime_batch_rank(1);
        e.call_frame = vec![Some(2)];
        let error = e
            .try_while_loop(
                std::slice::from_ref(&zero),
                std::slice::from_ref(&ty),
                |e, args| e.compare("EQ", &args[0], &zero),
                |e, args| {
                    let valid = e.bool_value_const(true);
                    e.record_runtime_error(&valid, 1);
                    Ok(args.to_vec())
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("flattened host batch"));
        assert!(e.runtime_status().is_none());
    }
}
