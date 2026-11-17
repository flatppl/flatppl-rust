//! Retained spline calls share the portable tensor formula and report domains.

use super::*;

impl Emitter<'_> {
    pub(super) fn lower_spline_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "splines" || member.starts_with("thinplate_") {
            return None;
        }
        Some(self.emit_spline_call(id, &member, args, broadcast))
    }

    fn emit_spline_call(
        &mut self,
        id: NodeId,
        member: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        let fixed = |i: usize| match member {
            "interp_linear" | "interp_cubic" | "interp_pchip" => i == 3,
            "bspline_basis" => i == 2,
            "bspline_derivative" => i >= 2,
            "bspline_integral" => i == 3,
            "bspline_penalty" => i >= 1,
            _ => false,
        };
        let mut local = Module::new();
        let mut local_args = Vec::new();
        let mut bindings = Vec::new();
        let mut types = Vec::new();
        for (i, &arg) in args.iter().enumerate() {
            let (node, value, ty) = if fixed(i) {
                let resolved = self.resolve_ref_one(arg);
                let Node::Lit(literal) = self.node(resolved) else {
                    return Err(EmitError::at(
                        id,
                        format!(
                            "splines.{member} requires a fixed literal degree, order or extrapolation"
                        ),
                    ));
                };
                (Node::Lit(literal.clone()), None, self.type_of(arg).cloned())
            } else {
                let value = self.lower_node(arg)?;
                let value = self.typed_axes(arg, value);
                let mut ty = self.type_of(arg).cloned();
                if broadcast {
                    ty = ty.map(|ty| match ty {
                        Type::Array { elem, .. } | Type::TVector { elem, .. } => *elem,
                        other => other,
                    });
                }
                let name = local.intern(&format!("spline_arg_{i}"));
                (Node::Const(name), Some(value), ty)
            };
            let node = local.alloc(node);
            if let Some(ty) = ty {
                local.set_type(node, ty.clone());
                types.push((node, ty));
            }
            if let Some(value) = value {
                bindings.push((node, value));
            }
            local_args.push(node);
        }
        let (result, valid) = flatppl_determinizer::expand_spline(&mut local, &local_args, member)
            .ok_or_else(|| EmitError::at(id, format!("splines.{member} requires matching static vector shapes and fixed valid degree/order/extrapolation")))?;
        let (inferred, _) = flatppl_infer::infer_expression(&mut local, result, &types);
        for (node, ty) in &inferred {
            local.set_type(*node, ty.clone());
        }
        // Only the temporary formula uses these NodeIds. SSA-indexed emitter
        // state is moved through, so existing model size does not affect cost.
        let original_module = self.m;
        let original_memo = std::mem::take(&mut self.memo);
        let original_types = std::mem::take(&mut self.scoped_types);
        let original_inputs = std::mem::take(&mut self.scoped_inputs);
        let original_inference = self.inference_module.take();
        let original_columns = std::mem::take(&mut self.columns);
        let original_sample_keys = std::mem::take(&mut self.sample_keys);
        let empty = Emitter::new(original_module, self.dtype);
        let mut child = Emitter {
            m: &local,
            ..std::mem::replace(self, empty)
        };
        child.scoped_types = inferred;
        let mut values: Vec<_> = bindings.iter().map(|(_, value)| value.clone()).collect();
        let parent = if broadcast {
            Some(child.enter_broadcast(id, &mut values))
        } else {
            None
        };
        let emitted = match parent {
            Some(Err(error)) => Err(error),
            parent => {
                for ((node, _), value) in bindings.iter().zip(values) {
                    child.memo.insert(*node, value);
                }
                let output = (|| {
                    let value = child.lower_node(result)?;
                    let condition = child.lower_node(valid)?;
                    child.record_runtime_error(&condition, 12);
                    Ok(value)
                })();
                match parent {
                    Some(Ok(parent)) => {
                        let frame = child.broadcast_frame.clone();
                        let output =
                            output.map(|v| child.finish_broadcast(&v, &frame, parent.len()));
                        child.broadcast_frame = parent;
                        output
                    }
                    _ => output,
                }
            }
        };
        child.memo = original_memo;
        child.scoped_types = original_types;
        child.scoped_inputs = original_inputs;
        child.inference_module = original_inference;
        child.columns = original_columns;
        child.sample_keys = original_sample_keys;
        *self = Emitter {
            m: original_module,
            ..child
        };
        emitted
    }
}
