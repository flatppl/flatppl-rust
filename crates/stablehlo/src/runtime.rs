//! Runtime source-domain checks, before backend operations lose invalid inputs.

use flatppl_core::NodeId;

use crate::{Dtype, ElemKind, EmitError, Emitter, Value};

impl Emitter<'_> {
    /// Check a source index before base subtraction or StableHLO's clamping
    /// gather. A failed call returns a safe operand and a separate error status.
    pub(crate) fn checked_index(
        &mut self,
        id: NodeId,
        index: &Value,
        extent: u64,
        base: i64,
    ) -> Result<Value, EmitError> {
        if extent == 0 {
            return Err(EmitError::at(id, "index into an empty axis"));
        }
        let upper = extent.saturating_sub(1).saturating_add(base as u64);
        if let Some(valid) = self.constant_indices_in_bounds(index, base, upper) {
            if !valid {
                return Err(EmitError::at(id, "index out of range"));
            }
            // Keep static selectors visible to the gather and segment passes.
            return Ok(index.clone());
        }
        let lower = self.int_value_const(base);
        let mut valid = self.compare("GE", index, &lower);
        let max = match self.dtype() {
            Dtype::F32 => i32::MAX as u64,
            Dtype::F64 => i64::MAX as u64,
        };
        if upper < max {
            let upper = self.int_value_const(upper as i64);
            let below = self.compare("LE", index, &upper);
            valid = self.and(&valid, &below);
        }
        self.record_runtime_error(&valid, 1);
        Ok(self.select(&valid, index, &lower))
    }

    /// Integral real values must also fit the target integer width. Checking
    /// an exclusive power-of-two upper bound avoids rounded MAX constants.
    pub(crate) fn checked_integer(&mut self, value: &Value) -> Value {
        let limit = match self.dtype() {
            Dtype::F32 => 2_f64.powi(31),
            Dtype::F64 => 2_f64.powi(63),
        };
        let lower = self.scalar(-limit);
        let upper = self.scalar(limit);
        let above = self.compare("GE", value, &lower);
        let below = self.compare("LT", value, &upper);
        let valid = self.and(&above, &below);
        self.record_runtime_error(&valid, 2);
        let zero = self.scalar(0.0);
        let safe = self.select(&valid, value, &zero);
        self.convert(&safe, ElemKind::Int)
    }
}
