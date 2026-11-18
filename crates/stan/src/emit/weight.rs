//! `logweighted(lw, M)`: one `target += <lw>;` beside M's own statements
//! (spec §06 "Density reweighting"), and `logdensityof(D, x)` as `D_lpdf`.
//!
//! A log weight that is a function is inlined at model scope: its body is
//! instantiated at the record type of M's variate, and a field read of its
//! argument prints as that field's parameter. It never becomes `~` terms.

use super::functions::FnCtx;
use super::{Emitter, Ex};
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Node, NodeId, Type};

impl<'m> Emitter<'m> {
    /// Emit each queued log weight as one `target +=` line of the prior.
    pub(super) fn emit_weights(&mut self) -> Result<()> {
        for (lw, base) in std::mem::take(&mut self.weights) {
            let line = self.weight(lw, base)?;
            self.prior.push(line);
        }
        Ok(())
    }

    fn weight(&mut self, lw: NodeId, base: NodeId) -> Result<String> {
        let Some(fun) = self.user_callee(lw) else {
            // A constant log weight.
            return Ok(format!("target += {};", self.expr(lw)?.s));
        };
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != 1 {
            return Err(Refusal::not_yet(
                Some(lw),
                "a log weight of several arguments",
            ));
        }
        let ty = match self.ty(base) {
            Some(Type::Measure { domain, .. }) => *domain,
            _ => {
                return Err(Refusal::not_yet(
                    Some(base),
                    "a log weight on an untyped measure",
                ));
            }
        };
        let scoped = self.instantiate(lw, body, &entries, &[ty])?;
        let name = match self.self_ref(lw) {
            Some(s) => self.name(s).to_string(),
            None => "weight".to_string(),
        };
        let mut ctx = FnCtx {
            scoped,
            record_local: Some(entries[0].1.name),
            ..Default::default()
        };
        self.mark_boundary(&mut ctx, lw, body, &entries)?;
        let outer = self.fctx.replace(ctx);
        let outer_owner = self.owner.replace((name.clone(), body));
        let value = match self.expr(body) {
            Ok(e) => Ok(e.s),
            Err(r) if r.kind == RefusalKind::NotYetSupported && self.elementwise(body) => {
                self.place(&format!("{name}_value"), body, Err(r))
            }
            Err(r) => Err(r),
        };
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        self.owner = outer_owner;
        let value = value?;
        // The weight's own statements are model-block locals.
        self.locals.extend(ctx.lines);
        Ok(format!("target += {value};"))
    }

    /// The parameter of field `f` of a log weight's record argument.
    pub(super) fn record_field(&mut self, at: NodeId, f: &str) -> Result<Ex> {
        let stan = self
            .fields
            .iter()
            .find(|(k, _)| self.name(**k) == f)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| Refusal::not_yet(Some(at), format!("the weight field `.{f}`")))?;
        Ok(Ex::atom(stan))
    }

    /// `logdensityof.(D, v)` with `D` held: the `(D, v)` a `sum` reduces.
    pub(super) fn density_broadcast(&self, a: NodeId) -> Option<(NodeId, NodeId)> {
        let r = self.resolve(a);
        if self.head(r) != Some("broadcast") {
            return None;
        }
        let c = self.call(r)?;
        match (self.m.node(c.args[0]), &c.args[1..]) {
            (Node::Const(f), [m, x])
                if self.name(*f) == "logdensityof"
                    && matches!(self.ty(*m), Some(Type::Measure { .. })) =>
            {
                Some((*m, *x))
            }
            _ => None,
        }
    }

    /// `logdensityof(D, x)` as `D_lpdf(x | …)`, or `_lpmf` for a discrete law,
    /// from the same `distributions.ron` row as `~`. Over an array `x` Stan's
    /// density is the sum, which is `sum(logdensityof.(D, x))`.
    pub(super) fn log_density(&mut self, id: NodeId, m: NodeId, x: NodeId) -> Result<Ex> {
        self.density_call(id, m, x).map(Ex::atom)
    }
}
