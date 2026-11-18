//! `filter(pred, xs)` over data (spec §07 "Membership, filtering, and bin
//! selection"): the elements of `xs` for which `pred` holds. Its length is
//! known only after evaluation, so the count and the kept elements are
//! computed once in `transformed data`, and every extent derived from them
//! is that count. Stan has no `filter` or `which`, so both are loops over
//! data that run once.
//!
//! A predicate that reads parameters gives a parameter-dependent length,
//! which no Stan container can have.

use super::Emitter;
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{NodeId, Phase, Type};

impl<'m> Emitter<'m> {
    /// The kept elements of the `filter` call `id` under the name `base`.
    pub(super) fn filter_value(&mut self, base: &str, id: NodeId) -> Result<String> {
        if let Some(n) = self.recall(id) {
            return Ok(n);
        }
        let c = self.call(id).unwrap();
        let [pred, xs] = c.args[..] else {
            return Err(Refusal::not_yet(
                Some(id),
                "filter with a callback parameter",
            ));
        };
        if !c.named.is_empty() {
            return Err(Refusal::not_yet(Some(id), "filter with keyword arguments"));
        }
        if matches!(self.ty(xs), Some(Type::Table { .. })) {
            return Err(Refusal::not_yet(Some(id), "filter over a table"));
        }
        if self.m.phase_of(id) != Some(Phase::Fixed) {
            return Err(Refusal::new(
                RefusalKind::ParameterDependentExtent,
                Some(id),
                "filter",
                "`filter` of a predicate or data that depend on parameters has a length that \
                 depends on parameters; a Stan container's size is data",
            ));
        }
        let elem = match self.ty(xs) {
            Some(Type::Array { shape, elem }) if shape.len() == 1 => *elem,
            _ => return Err(Refusal::not_yet(Some(id), "filter over this value")),
        };
        let fun = self
            .user_callee(pred)
            .ok_or_else(|| Refusal::not_yet(Some(id), "filter with this predicate"))?;
        // A filtered `xs` (a filter of a filter) has an extent only once its
        // count is declared.
        self.at_module(|me| me.expr(xs))?;
        let n = self
            .sym_extent(xs, 0, 0)
            .ok_or_else(|| Refusal::not_yet(Some(id), "filter over an array of unknown length"))?;
        let hi = if n.atomic() {
            n.print()
        } else {
            format!("({})", n.print())
        };
        let (fname, caps) = self.at_module(|me| me.user_function(id, fun, &[elem]))?;
        let count = self.fresh(&format!("{base}_n"));
        let name = self.fresh(base);
        self.filter_counts.insert(id, count.clone());
        let var = self.loop_var();
        let x = self.at_module(|me| me.elem(xs, std::slice::from_ref(&var)));
        let x = match x {
            Ok(x) => x,
            Err(e) => {
                self.used.remove(&var);
                return Err(e);
            }
        };
        let test = Self::apply_user(&fname, vec![x.s.clone()], &caps);
        let t = self.declare(id, None, false, &name, &[])?;
        let b = self.fresh_local("b");
        self.tdata.push(format!("int {count} = 0;"));
        self.tdata.push(format!(
            "for ({var} in 1:{hi}) {{\n  {count} += {test};\n}}"
        ));
        self.tdata.push(format!("{};", t.render(&name)));
        self.tdata.push(format!(
            "{{\n  int {b} = 1;\n  for ({var} in 1:{hi}) {{\n    if ({test}) {{\n      {name}[{b}] = {};\n      {b} += 1;\n    }}\n  }}\n}}",
            x.s
        ));
        self.used.remove(&var);
        self.used.remove(&b);
        self.record_decl(&name, t);
        self.remember(id, &name);
        Ok(name)
    }
}
