//! Retained `special-functions` standard-module calls (§09), lowered to CHLO.
//! FlatPDL has no base-op form for them, so portable determinization refuses.

use super::*;

/// The `special-functions` members this backend retains, with their CHLO op.
/// `erfc` keeps its own op: `1 - erf(x)` cancels to zero in the upper tail.
pub(crate) const SPECIAL_FUNCTIONS: [(&str, &str); 2] =
    [("erf", "chlo.erf"), ("erfc", "chlo.erfc")];

impl Emitter<'_> {
    pub(super) fn lower_special_function_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "special-functions" {
            return None;
        }
        let Some(&(_, op)) = SPECIAL_FUNCTIONS.iter().find(|(name, _)| *name == member) else {
            return Some(Err(EmitError::at(
                id,
                format!("special-functions.{member} has no StableHLO lowering"),
            )));
        };
        Some(self.emit_special_function_call(id, op, args, broadcast))
    }

    fn emit_special_function_call(
        &mut self,
        id: NodeId,
        op: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        if matches!(self.m.node(id), Node::Call(call) if !call.named.is_empty()) {
            return Err(EmitError::at(
                id,
                "special-functions arguments must be bound by determinize",
            ));
        }
        let [arg] = args else {
            return Err(EmitError::at(id, format!("{op} takes one argument")));
        };
        let value = self.lower_node(*arg)?;
        let mut values = vec![self.typed_axes(*arg, value)];
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let result = self.chlo_unary(op, &values[0]);
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
}
