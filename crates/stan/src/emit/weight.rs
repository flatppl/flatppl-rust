//! `logweighted(lw, M)`: one `target += <lw>;` beside M's own statements
//! (spec §06 "Density reweighting"), and `logdensityof(D, x)` as `D_lpdf`.
//!
//! A log weight that is a function is inlined at model scope: its body is
//! instantiated at the record type of M's variate, and a field read of its
//! argument prints as that field's parameter. It never becomes `~` terms.

use super::functions::FnCtx;
use super::mixture::ElementLp;
use super::structure::LOOP;
use super::{Emitter, Ex};
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Node, NodeId, RefNs, Scalar, Type};

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
        let line = match self.weighted_loop(body) {
            Ok(Some(block)) => Ok(block),
            Ok(None) => match self.expr(body) {
                Ok(e) => Ok(format!("target += {};", e.s)),
                Err(r) if r.kind == RefusalKind::NotYetSupported && self.elementwise(body) => self
                    .place(&format!("{name}_value"), body, Err(r))
                    .map(|v| format!("target += {v};")),
                Err(r) => Err(r),
            },
            Err(r) => Err(r),
        };
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        self.owner = outer_owner;
        let line = line?;
        // The weight's own statements are model-block locals.
        self.locals.extend(ctx.lines);
        Ok(line)
    }

    /// `sum(w .* f(k = point.k, …))` with `f` a `functionof` whose body is
    /// `((p…) -> logdensityof(D(p…), value)).(args…)`: one weighted term
    /// per element (Ruling 4). Stan has no elementwise `_lpdf` (that of a
    /// container is the sum), so each element's density takes its weight
    /// in a loop.
    fn weighted_loop(&mut self, body: NodeId) -> Result<Option<String>> {
        let r = self.resolve(body);
        if self.head(r) != Some("sum") {
            return Ok(None);
        }
        let [prod] = self.call(r).unwrap().args[..] else {
            return Ok(None);
        };
        let prod = self.resolve(prod);
        if self.head(prod) != Some("broadcast") {
            return Ok(None);
        }
        let (f, x, y) = match &self.call(prod).unwrap().args[..] {
            [f, x, y] => (*f, *x, *y),
            _ => return Ok(None),
        };
        if !matches!(self.m.node(f), Node::Const(c) if self.name(*c) == "mul") {
            return Ok(None);
        }
        let keyword = |me: &Self, a: NodeId| {
            me.is_user_call(a).is_some() && !me.call(a).unwrap().named.is_empty()
        };
        let (w, call) = if keyword(self, y) {
            (x, y)
        } else if keyword(self, x) {
            (y, x)
        } else {
            return Ok(None);
        };
        let e = self.keyword_body(call)?;
        let refuse = || Refusal::not_yet(Some(call), "a weighted log likelihood of this form");
        let er = self.resolve(e);
        if self.head(er) != Some("broadcast") {
            return Err(refuse());
        }
        let lam = self
            .user_callee(self.call(er).unwrap().args[0])
            .ok_or_else(refuse)?;
        let (lbody, entries) = self.callable_parts(lam)?;
        let lb = self.resolve(lbody);
        let (law, point) = match (self.head(lb), &self.call(lb).ok_or_else(refuse)?.args[..]) {
            (Some("logdensityof"), [law, p]) => (*law, *p),
            _ => return Err(refuse()),
        };
        let value = match self.m.node(point) {
            Node::Ref(p) if p.ns == RefNs::Local && entries.iter().any(|(_, e)| e == p) => p.name,
            _ => return Err(refuse()),
        };
        // The law's arguments, and a weight inside it, must not read the
        // observed value.
        if self.reads_local(law, value) {
            return Err(Refusal::not_yet(
                Some(law),
                "a weighted log likelihood whose law reads the observed value",
            ));
        }
        if self.shape(w).is_none_or(|s| s.rank != 1) {
            return Err(refuse());
        }
        // The keyword call is the identity substitution: its body reads at
        // model scope, outside the log weight's own context.
        let weight_ctx = self.fctx.take();
        let out = self.weighted_terms(er, law, point, w);
        self.fctx = weight_ctx;
        let (wt, n, out) = out?;
        let i = self.fresh_loop();
        let wi = format!("{}[{i}]", wt.at(super::P_ATOM));
        let times = |lp: &str| {
            if single_call(lp) {
                format!("{wi} * {lp}")
            } else {
                format!("{wi} * ({lp})")
            }
        };
        let inner = match out {
            ElementLp::One(lp) => format!("  target += {};\n", times(&lp)),
            ElementLp::Split(test, at, rest) => format!(
                "  if ({test}) {{\n    target += {};\n  }} else {{\n    target += {};\n  }}\n",
                times(&at),
                times(&rest)
            ),
        };
        self.used.remove(&i);
        Ok(Some(format!(
            "for ({i} in 1:{n}) {{\n{}}}",
            inner.replace(LOOP, &i)
        )))
    }

    /// The weight's text and extent, and one element's log density under
    /// `law` at `point`, with E's lambda locals bound per element.
    fn weighted_terms(
        &mut self,
        er: NodeId,
        law: NodeId,
        point: NodeId,
        w: NodeId,
    ) -> Result<(Ex, String, ElementLp)> {
        let wt = self.expr(w)?;
        let n = self
            .sym_extent(w, 0, 0)
            .map(|l| l.print())
            .unwrap_or_else(|| format!("num_elements({})", wt.s));
        let (_, ctx) = self.user_law_body(er, LOOP)?;
        let outer = self.fctx.replace(ctx);
        let out = self.expr(point).and_then(|yv| self.element_lp(law, &yv.s));
        let ctx = std::mem::replace(&mut self.fctx, outer);
        if ctx.is_some_and(|c| !c.lines.is_empty()) {
            return Err(Refusal::not_yet(
                Some(law),
                "a weighted log likelihood whose law needs statements",
            ));
        }
        Ok((wt, n, out?))
    }

    /// The body of the keyword call `f(k = point.k, …)` of `f =
    /// functionof(body, k = k, …)`: each argument is the log weight's point
    /// field of the input's own name, and each input is the module binding
    /// of that name, so the substitution is the identity and the body is
    /// read at model scope.
    fn keyword_body(&mut self, call: NodeId) -> Result<NodeId> {
        let refuse = || {
            Refusal::not_yet(
                Some(call),
                "a keyword call of a functionof whose arguments are not the point's matching fields",
            )
        };
        let fun = self.is_user_call(call).ok_or_else(refuse)?;
        let (body, entries) = self.callable_parts(fun)?;
        let c = self.call(call).unwrap();
        if !c.args.is_empty() || c.named.len() != entries.len() {
            return Err(refuse());
        }
        let named: Vec<(flatppl_core::Symbol, NodeId)> =
            c.named.iter().map(|na| (na.name, na.value)).collect();
        let point = self
            .fctx
            .as_ref()
            .and_then(|x| x.record_local)
            .ok_or_else(refuse)?;
        for (key, value) in named {
            let input = entries.iter().find(|(k, _)| *k == key).ok_or_else(refuse)?;
            if !(input.1.ns == RefNs::SelfMod && input.1.name == key) {
                return Err(refuse());
            }
            let v = self.resolve(value);
            let field = match (self.head(v), self.call(v).map(|x| &x.args[..])) {
                (Some("get"), Some([x, f])) => match (self.m.node(*x), self.m.node(*f)) {
                    (Node::Ref(rf), Node::Lit(Scalar::Str(s))) if rf.name == point => {
                        Some(s.clone())
                    }
                    _ => None,
                },
                _ => None,
            };
            let key = self.name(key).to_string();
            if field.as_deref() != Some(key.as_str()) {
                return Err(refuse());
            }
            // The point field is a prior field. A free input (`elementof`) of
            // the same name is that field's parameter here.
            let stan = self.record_field(value, &key)?.s;
            let sym = input.1.name;
            // The input reads the point's field only where the binding is
            // that field's draw, or a free input with no value of its own. A
            // binding with its own value (`b = 3.0`) would be read instead.
            match self.env.get(&sym) {
                Some(s) if *s == stan => {}
                Some(_) => return Err(refuse()),
                None => {
                    let free = self
                        .m
                        .binding_by_name(sym)
                        .is_some_and(|b| self.head(self.m.binding(b).rhs) == Some("elementof"));
                    if !free {
                        return Err(refuse());
                    }
                    self.env.insert(sym, stan);
                }
            }
        }
        Ok(body)
    }

    /// `true` when `x` reads the local `name`.
    fn reads_local(&self, x: NodeId, name: flatppl_core::Symbol) -> bool {
        let mut stack = vec![x];
        while let Some(id) = stack.pop() {
            if let Node::Ref(r) = self.m.node(id)
                && r.ns == RefNs::Local
                && r.name == name
            {
                return true;
            }
            self.m.for_each_child(id, |c| stack.push(c));
        }
        false
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

/// `name(…)` with its parentheses closing at the end: one call, which needs
/// no parentheses as a factor.
fn single_call(s: &str) -> bool {
    let Some(open) = s.find('(') else {
        return false;
    };
    if open == 0
        || !s[..open]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return false;
    }
    let mut depth = 0i32;
    for (i, c) in s.char_indices().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return i == s.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}
