//! User functions as Stan `functions`, one per argument types.
//!
//! A FlatPPL function is polymorphic and its body is typed generically, so
//! each use instantiates it: `flatppl_infer::infer_expression` types the body
//! at the argument types, and the body is emitted under those types. A Stan
//! function sees only its arguments, so every module value the body reads is
//! passed as a trailing argument under its own name.
//!
//! `f.(xs…)` fills its result by loops over the broadcast axes, calling `f`
//! per element. A broadcast user law is inlined per element.

use super::{Emitter, Ex, P_ATOM};
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::{StanType, declare};
use flatppl_core::{CallHead, Inputs, Node, NodeId, Phase, Ref, RefNs, Symbol, Type};
use std::collections::{HashMap, HashSet};

/// The state of a function body being emitted.
#[derive(Default)]
pub(crate) struct FnCtx {
    /// Statements before the `return`.
    pub lines: Vec<String>,
    /// Input placeholder → its Stan expression.
    pub locals: HashMap<Symbol, String>,
    /// Node types of this instantiation.
    pub scoped: HashMap<NodeId, Type>,
    /// Names declared inside the body (never captured).
    pub declared: HashSet<String>,
    /// Law arguments lifted over whole arrays, printed as is.
    pub lifted: HashMap<NodeId, Ex>,
}

/// The unsized Stan type of a function argument or result.
pub(crate) fn arg_type(t: &StanType) -> String {
    let base = match t.base {
        "simplex" | "ordered" | "positive_ordered" | "unit_vector" | "sum_to_zero_vector" => {
            "vector"
        }
        "cholesky_factor_corr" | "cholesky_factor_cov" | "corr_matrix" | "cov_matrix" => "matrix",
        b => b,
    };
    if t.array.is_empty() {
        base.to_string()
    } else {
        format!("array[{}] {base}", ",".repeat(t.array.len() - 1))
    }
}

/// Identifiers of `text` that are not function names.
fn identifiers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let called = text[i..].trim_start().starts_with('(');
            if !called {
                out.push(text[start..i].to_string());
            }
        } else if c.is_ascii_digit() || c == '.' {
            // Skip number literals (`1e5`).
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'.') {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

/// The outer array rank of a type (`cartpow(cartpow(reals, m), n)` is 1).
fn top_rank(ty: Option<&Type>) -> usize {
    match ty {
        Some(Type::Array { shape, .. }) => shape.len(),
        _ => 0,
    }
}

impl<'m> Emitter<'m> {
    /// The `functionof` a callee denotes: a reference to a function binding or
    /// an inline function.
    pub(super) fn user_callee(&self, f: NodeId) -> Option<NodeId> {
        let r = self.resolve(f);
        if self.head(r) != Some("functionof") {
            return None;
        }
        Some(r)
    }

    /// The body and input entries of a `functionof`.
    fn callable_parts(&self, fun: NodeId) -> Result<(NodeId, Vec<(Symbol, Ref)>)> {
        let c = self.call(fun).unwrap();
        let body = c.args[0];
        let entries = match &c.inputs {
            Some(Inputs::Spec(e)) => e.to_vec(),
            Some(Inputs::Auto) => self
                .m
                .auto_inputs_of(fun)
                .ok_or_else(|| Refusal::not_yet(Some(fun), "a function with unresolved inputs"))?
                .to_vec(),
            None => return Err(Refusal::not_yet(Some(fun), "a callable without inputs")),
        };
        Ok((body, entries))
    }

    /// The types of `body` with its input placeholders at `arg_tys`.
    fn instantiate(
        &mut self,
        at: NodeId,
        body: NodeId,
        entries: &[(Symbol, Ref)],
        arg_tys: &[Type],
    ) -> Result<HashMap<NodeId, Type>> {
        let mut seeds = Vec::new();
        let mut stack = vec![body];
        let mut seen = HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Node::Ref(r) = self.m.node(id)
                && let Some(k) = entries.iter().position(|(_, e)| e == r)
            {
                seeds.push((id, arg_tys[k].clone()));
            }
            self.m.for_each_child(id, |c| stack.push(c));
        }
        let module = self.scratch.get_or_insert_with(|| self.m.clone());
        let (types, diags) = flatppl_infer::infer_expression(module, body, &seeds);
        if let Some(d) = diags
            .iter()
            .find(|d| d.severity == flatppl_infer::Severity::Error)
        {
            return Err(Refusal::new(
                RefusalKind::NotYetSupported,
                Some(at),
                "a user function at these argument types",
                format!(
                    "the user function does not type at its arguments: {}",
                    d.message
                ),
            ));
        }
        Ok(types)
    }

    /// The Stan function for `fun` at `arg_tys`, and the module names it
    /// takes as trailing arguments.
    pub(super) fn user_function(
        &mut self,
        at: NodeId,
        fun: NodeId,
        arg_tys: &[Type],
    ) -> Result<(String, Vec<String>)> {
        let key = (fun, format!("{arg_tys:?}"));
        if let Some(v) = self.fn_memo.get(&key) {
            return Ok(v.clone());
        }
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != arg_tys.len() {
            return Err(Refusal::not_yet(
                Some(at),
                "a user function called with another arity",
            ));
        }
        let types = self.instantiate(at, body, &entries, arg_tys)?;
        let base = self.function_name(fun);

        let outer_ctx = self.fctx.replace(FnCtx {
            scoped: types,
            ..Default::default()
        });
        let outer_owner = self.owner.take();
        let out = self.function_body(at, fun, &base, body, &entries, arg_tys);
        self.fctx = outer_ctx;
        self.owner = outer_owner;
        let (name, caps) = out?;
        self.fn_memo.insert(key, (name.clone(), caps.clone()));
        Ok((name, caps))
    }

    fn function_name(&self, fun: NodeId) -> String {
        self.m
            .bindings()
            .find(|(_, b)| b.rhs == fun)
            .map(|(_, b)| self.name(b.name).to_string())
            .unwrap_or_else(|| "lambda".to_string())
    }

    fn function_body(
        &mut self,
        at: NodeId,
        fun: NodeId,
        base: &str,
        body: NodeId,
        entries: &[(Symbol, Ref)],
        arg_tys: &[Type],
    ) -> Result<(String, Vec<String>)> {
        let mut params = Vec::new();
        let mut names = Vec::new();
        for ((name, r), ty) in entries.iter().zip(arg_tys) {
            let p = self.fresh(self.name(*name));
            let t = declare(ty, None, false, &mut |_| String::new(), at)?;
            params.push(format!("{} {p}", arg_type(&t)));
            self.record_decl(&p, t);
            let ctx = self.fctx.as_mut().unwrap();
            ctx.locals.insert(r.name, p.clone());
            ctx.declared.insert(p.clone());
            names.push(p);
        }
        self.owner = Some((base.to_string(), body));
        let attempt = self.expr(body);
        let ret = match attempt {
            Ok(e) => e.s,
            Err(r) if r.kind == RefusalKind::NotYetSupported && self.elementwise(body) => {
                self.place(&format!("{base}_value"), body, Err(r))?
            }
            Err(r) => return Err(r),
        };
        let ret_ty = self
            .ty(body)
            .ok_or_else(|| Refusal::not_yet(Some(fun), "a function with an untyped result"))?;
        let ret_t = declare(&ret_ty, None, false, &mut |_| String::new(), body)?;
        let ctx = self.fctx.take().unwrap();
        let mut text_body: Vec<String> = ctx.lines.clone();
        text_body.push(format!("return {ret};"));
        let joined = text_body.join("\n");
        let mut caps = Vec::new();
        for id in identifiers(&joined) {
            if !ctx.declared.contains(&id) && !caps.contains(&id) && self.decls.contains_key(&id) {
                caps.push(id);
            }
        }
        for c in &caps {
            params.push(format!("{} {c}", arg_type(&self.decls[c])));
        }
        let name = self.fresh(base);
        let mut text = format!("{} {name}({}) {{\n", arg_type(&ret_t), params.join(", "));
        for l in &text_body {
            for part in l.lines() {
                text.push_str("  ");
                text.push_str(part);
                text.push('\n');
            }
        }
        text.push('}');
        self.functions.push(text);
        self.fctx = Some(ctx);
        Ok((name, caps))
    }

    /// `name(args…, captures…)`.
    fn apply_user(name: &str, args: Vec<String>, caps: &[String]) -> String {
        let all: Vec<String> = args.into_iter().chain(caps.iter().cloned()).collect();
        format!("{name}({})", all.join(", "))
    }

    /// A user call `f(args…)`.
    pub(super) fn user_call(&mut self, id: NodeId, fun: NodeId, args: &[NodeId]) -> Result<Ex> {
        let mut tys = Vec::new();
        let mut parts = Vec::new();
        for &a in args {
            tys.push(
                self.ty(a)
                    .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?,
            );
            parts.push(self.expr(a)?.s);
        }
        let (name, caps) = self.user_function(id, fun, &tys)?;
        Ok(Ex::atom(Self::apply_user(&name, parts, &caps)))
    }

    /// `f.(args…)` as one call of a vectorized Stan function: the body at
    /// whole-array arguments, each operation in its elementwise form. `Err`
    /// gives the reason it loops instead.
    pub(super) fn vectorized_broadcast(
        &mut self,
        id: NodeId,
    ) -> Result<std::result::Result<Ex, String>> {
        let c = self.call(id).unwrap();
        if !c.named.is_empty() {
            return Ok(Err("keyword arguments".into()));
        }
        let fun = self.user_callee(c.args[0]).unwrap();
        let args = c.args[1..].to_vec();
        let tys: Vec<Option<Type>> = args.iter().map(|&a| self.ty(a)).collect();
        let brank = tys.iter().map(|t| top_rank(t.as_ref())).max().unwrap_or(0);
        if brank == 0 || brank > 2 {
            return Ok(Err(format!("a broadcast over {brank} axes")));
        }
        match self.ty(id) {
            Some(Type::Array { elem, .. })
                if matches!(*elem, Type::Scalar(flatppl_core::ScalarType::Real)) => {}
            Some(Type::Array { elem, .. }) if matches!(*elem, Type::Scalar(_)) => {
                return Ok(Err(
                    "an integer result, and Stan has no integer-array arithmetic".into(),
                ));
            }
            _ => return Ok(Err("a non-scalar result per element".into())),
        }
        let mut elem_tys = Vec::new();
        let mut full = Vec::new();
        for (a, t) in args.iter().zip(&tys) {
            let Some(t) = t.clone() else {
                return Err(Refusal::not_yet(Some(*a), "an untyped argument"));
            };
            match (top_rank(Some(&t)), t) {
                (r, Type::Array { elem, .. }) if r == brank && matches!(*elem, Type::Scalar(_)) => {
                    elem_tys.push(*elem);
                    full.push(true);
                }
                (r, Type::Array { .. }) if r == brank => {
                    return Ok(Err("a non-scalar argument element".into()));
                }
                (0, t) => {
                    elem_tys.push(t);
                    full.push(false);
                }
                _ => return Ok(Err("arguments of mixed ranks".into())),
            }
        }
        let key = (fun, format!("vectorized {tys:?}"));
        let (name, caps) = match self.fn_memo.get(&key) {
            Some(v) => v.clone(),
            None => {
                let (body, entries) = self.callable_parts(fun)?;
                if entries.len() != args.len() {
                    return Err(Refusal::not_yet(
                        Some(id),
                        "a user function called with another arity",
                    ));
                }
                let types = self.instantiate(id, body, &entries, &elem_tys)?;
                let outer_ctx = self.fctx.replace(FnCtx {
                    scoped: types,
                    ..Default::default()
                });
                let outer_owner = self.owner.take();
                let outer_v = self.vlocals.take();
                let out = self.vectorized_body(id, fun, body, &entries, &args, &tys, &full, brank);
                self.fctx = outer_ctx;
                self.owner = outer_owner;
                self.vlocals = outer_v;
                match out? {
                    Ok(v) => {
                        self.fn_memo.insert(key, v.clone());
                        v
                    }
                    Err(reason) => return Ok(Err(reason)),
                }
            }
        };
        let mut parts = Vec::new();
        for &a in &args {
            parts.push(self.expr(a)?.s);
        }
        Ok(Ok(Ex::atom(Self::apply_user(&name, parts, &caps))))
    }

    #[allow(clippy::too_many_arguments)]
    fn vectorized_body(
        &mut self,
        at: NodeId,
        fun: NodeId,
        body: NodeId,
        entries: &[(Symbol, Ref)],
        args: &[NodeId],
        tys: &[Option<Type>],
        full: &[bool],
        brank: usize,
    ) -> Result<std::result::Result<(String, Vec<String>), String>> {
        let mut params = Vec::new();
        let mut leaves = HashMap::new();
        let mut frame = Vec::new();
        for (((name, r), _a), (ty, &f)) in entries.iter().zip(args).zip(tys.iter().zip(full)) {
            let p = self.fresh(self.name(*name));
            let ty = ty.clone().unwrap();
            // A traversed argument is passed whole; a held one as it is.
            let t = declare(&ty, None, false, &mut |_| String::new(), at)?;
            params.push(format!("{} {p}", arg_type(&t)));
            let array = f && !t.array.is_empty();
            self.record_decl(&p, t);
            self.fctx.as_mut().unwrap().declared.insert(p.clone());
            leaves.insert(
                r.name,
                super::aggregate::Lifted {
                    ex: Ex::atom(p.clone()),
                    full: f,
                    array,
                },
            );
            if frame.len() < brank {
                frame.push(*name);
            }
        }
        while frame.len() < brank {
            frame.push(entries[0].0);
        }
        self.vlocals = Some(leaves);
        let lifted = self.lift(body, &frame, &HashMap::new(), &HashMap::new());
        self.vlocals = None;
        let l = match lifted {
            Ok(l) if l.full => l,
            Ok(_) => {
                return Ok(Err(
                    "a body that does not depend on the broadcast arguments".into(),
                ));
            }
            Err(r) => return Ok(Err(r.reason)),
        };
        let ctx = self.fctx.take().unwrap();
        if !ctx.lines.is_empty() {
            self.fctx = Some(ctx);
            return Ok(Err("a body that needs statements".into()));
        }
        let ex = if l.array {
            self.to_real(&l.ex, brank)
        } else {
            l.ex
        };
        let ret_ty = self
            .m
            .type_of(at)
            .cloned()
            .ok_or_else(|| Refusal::not_yet(Some(at), "an untyped broadcast"))?;
        let ret_t = declare(&ret_ty, None, false, &mut |_| String::new(), at)?;
        let mut caps = Vec::new();
        for id in identifiers(&ex.s) {
            if !ctx.declared.contains(&id) && !caps.contains(&id) && self.decls.contains_key(&id) {
                caps.push(id);
            }
        }
        for c in &caps {
            params.push(format!("{} {c}", arg_type(&self.decls[c])));
        }
        let base = self.function_name(fun);
        let name = self.fresh(&base);
        self.functions.push(format!(
            "{} {name}({}) {{\n  return {};\n}}",
            arg_type(&ret_t),
            params.join(", "),
            ex.s
        ));
        self.fctx = Some(ctx);
        Ok(Ok((name, caps)))
    }

    /// `base` holding `f.(args…)`: one vectorized call when the body lifts,
    /// else filled by loops over the broadcast axes.
    pub(super) fn user_broadcast_value(&mut self, base: &str, id: NodeId) -> Result<String> {
        if let Ok(e) = self.vectorized_broadcast(id)? {
            return self.place(base, id, Ok(e));
        }
        self.looped_broadcast(base, id)
    }

    /// `base` holding `f.(args…)`, filled by loops over the broadcast axes.
    pub(super) fn looped_broadcast(&mut self, base: &str, id: NodeId) -> Result<String> {
        self.looped_broadcast_into(base, id, id)
    }

    /// `rowstack(f.(args…))` filled row by row from the calls.
    pub(super) fn stacked_broadcast(&mut self, base: &str, stack: NodeId) -> Result<String> {
        let bc = self.call(stack).unwrap().args[0];
        self.looped_broadcast_into(base, bc, stack)
    }

    /// `f.(args…)` declared as the value `decl` (itself, or the `rowstack`
    /// of it) and filled by loops over the broadcast axes.
    fn looped_broadcast_into(&mut self, base: &str, id: NodeId, decl: NodeId) -> Result<String> {
        let c = self.call(id).unwrap();
        if !c.named.is_empty() {
            return Err(Refusal::not_yet(
                Some(id),
                "a user broadcast with keyword arguments",
            ));
        }
        let fun = self.user_callee(c.args[0]).unwrap();
        let args = c.args[1..].to_vec();
        let tys: Vec<Option<Type>> = args.iter().map(|&a| self.ty(a)).collect();
        let brank = tys.iter().map(|t| top_rank(t.as_ref())).max().unwrap_or(0);
        if brank == 0 {
            let table = tys.iter().any(|t| matches!(t, Some(Type::Table { .. })));
            let what = if table {
                "a broadcast over a table"
            } else {
                "a broadcast over no array"
            };
            return Err(Refusal::not_yet(Some(id), what));
        }
        let mut elem_tys = Vec::new();
        for (a, t) in args.iter().zip(&tys) {
            let t = t
                .clone()
                .ok_or_else(|| Refusal::not_yet(Some(*a), "an untyped argument"))?;
            elem_tys.push(match (top_rank(Some(&t)), t) {
                (r, Type::Array { elem, .. }) if r == brank => *elem,
                (0, t) => t,
                _ => return Err(Refusal::not_yet(Some(*a), "a broadcast over mixed ranks")),
            });
        }
        let (name, caps) = self.user_function(id, fun, &elem_tys)?;

        let stan = self.fresh(base);
        self.peers = self.operand_extents(decl);
        let t = self.declare(decl, None, false, &stan, &[])?;
        let extents: Vec<String> = t.array.iter().chain(&t.dims).take(brank).cloned().collect();
        // A matrix row takes the call's vector as a row_vector.
        let row = decl != id && t.base == "matrix";
        let mut vars = Vec::new();
        for _ in 0..brank {
            vars.push(self.loop_var());
        }
        let mut parts = Vec::new();
        let mut failed = None;
        for (a, ty) in args.iter().zip(&tys) {
            let p = if top_rank(ty.as_ref()) == brank {
                self.elem_at(*a, &vars)
            } else {
                self.expr(*a)
            };
            match p {
                Ok(e) => parts.push(e.s),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        for v in &vars {
            self.used.remove(v);
        }
        if let Some(e) = failed {
            return Err(e);
        }
        let mut text = format!(
            "{stan}[{}] = {}{};",
            vars.join(", "),
            Self::apply_user(&name, parts, &caps),
            if row { "'" } else { "" }
        );
        for (v, n) in vars.iter().zip(&extents).rev() {
            let inner: String = text.lines().map(|l| format!("  {l}\n")).collect();
            text = format!("for ({v} in 1:{n}) {{\n{inner}}}");
        }
        let lines = vec![format!("{};", t.render(&stan)), text];
        self.record_decl(&stan, t);
        let fixed = self.m.phase_of(decl) == Some(Phase::Fixed);
        self.put(fixed, lines);
        Ok(stan)
    }

    /// Element `idx` of `a` along its outer axes; a `fill(x, n)` gives `x`.
    pub(super) fn elem_at(&mut self, a: NodeId, idx: &[String]) -> Result<Ex> {
        let r = self.resolve(a);
        if self.head(r) == Some("fill")
            && let [x, _] = self.call(r).unwrap().args[..]
        {
            return self.expr(x);
        }
        self.elem(a, idx)
    }

    /// Run `f` in the module context: module bindings never go into a
    /// function body.
    pub(super) fn at_module<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let ctx = self.fctx.take();
        let owner = self.owner.take();
        let out = f(self);
        self.fctx = ctx;
        self.owner = owner;
        out
    }

    /// The user law of a broadcast `f.(args…)` per element, with
    /// `index` standing for the loop index.
    pub(super) fn user_law_body(&mut self, m: NodeId, index: &str) -> Result<(NodeId, FnCtx)> {
        let c = self.call(m).unwrap();
        let fun = self
            .user_callee(c.args[0])
            .ok_or_else(|| Refusal::not_yet(Some(m), "a broadcast of a user law"))?;
        let args = c.args[1..].to_vec();
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != args.len() {
            return Err(Refusal::not_yet(
                Some(m),
                "a user law called with another arity",
            ));
        }
        let mut elem_tys = Vec::new();
        let mut locals = HashMap::new();
        for ((_, r), &a) in entries.iter().zip(&args) {
            let t = self
                .ty(a)
                .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?;
            let e = self.expr(a)?;
            let (et, s) = match t {
                Type::Array { shape, elem } if shape.len() == 1 => {
                    (*elem, format!("{}[{index}]", e.at(P_ATOM)))
                }
                Type::Array { .. } => {
                    return Err(Refusal::not_yet(
                        Some(a),
                        "a user law broadcast over a matrix",
                    ));
                }
                t => (t, e.s),
            };
            elem_tys.push(et);
            locals.insert(r.name, s);
        }
        let scoped = self.instantiate(m, body, &entries, &elem_tys)?;
        Ok((
            body,
            FnCtx {
                locals,
                scoped,
                ..Default::default()
            },
        ))
    }

    /// A broadcast user law as one vectorized law over whole arrays, when
    /// each law argument is a parameter itself or does not depend on one.
    pub(super) fn vectorized_user_law(
        &mut self,
        m: NodeId,
    ) -> Result<Option<(NodeId, FnCtx, usize)>> {
        let c = self.call(m).unwrap();
        let Some(fun) = self.user_callee(c.args[0]) else {
            return Ok(None);
        };
        let args = c.args[1..].to_vec();
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != args.len() || !c.named.is_empty() {
            return Ok(None);
        }
        let refs: Vec<Ref> = entries.iter().map(|(_, r)| *r).collect();
        let body_r = self.resolve(body);
        let Some(bc) = self.call(body_r) else {
            return Ok(None);
        };
        let mentions = |me: &Self, root: NodeId| {
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                if matches!(me.m.node(id), Node::Ref(r) if refs.contains(r)) {
                    return true;
                }
                me.m.for_each_child(id, |k| stack.push(k));
            }
            false
        };
        let law_args: Vec<NodeId> = bc
            .args
            .iter()
            .chain(bc.named.iter().map(|n| &n.value))
            .copied()
            .collect();
        let mut elem_tys = Vec::new();
        let mut leaves = HashMap::new();
        let mut nfull = 0;
        for ((name, r), &a) in entries.iter().zip(&args) {
            let Some(t) = self.ty(a) else {
                return Ok(None);
            };
            let e = self.expr(a)?;
            let (et, full) = match t {
                Type::Array { shape, elem }
                    if shape.len() == 1 && matches!(*elem, Type::Scalar(_)) =>
                {
                    nfull += 1;
                    (*elem, true)
                }
                Type::Array { .. } => return Ok(None),
                t => (t, false),
            };
            let array = full && self.shape(a).is_some_and(|s| s.int);
            elem_tys.push(et);
            leaves.insert(
                r.name,
                (*name, super::aggregate::Lifted { ex: e, full, array }),
            );
        }
        let scoped = self.instantiate(m, body, &entries, &elem_tys)?;
        // Each law argument that reads a parameter lifts over whole arrays.
        let outer_ctx = self.fctx.replace(FnCtx {
            scoped,
            ..Default::default()
        });
        let outer_v = self
            .vlocals
            .replace(leaves.iter().map(|(k, (_, l))| (*k, l.clone())).collect());
        let frame = vec![entries[0].0];
        let mut lifted = HashMap::new();
        let mut ok = true;
        for &a in &law_args {
            if !mentions(self, a) {
                continue;
            }
            match self.lift(a, &frame, &HashMap::new(), &HashMap::new()) {
                Ok(l) if l.array => {
                    lifted.insert(a, self.to_real(&l.ex, 1));
                }
                Ok(l) => {
                    lifted.insert(a, l.ex);
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        self.vlocals = outer_v;
        let mut ctx = std::mem::replace(&mut self.fctx, outer_ctx).unwrap();
        if !ok || !ctx.lines.is_empty() {
            return Ok(None);
        }
        ctx.lifted = lifted;
        Ok(Some((body, ctx, nfull)))
    }

    /// The local or bound Stan expression of a `%local` reference.
    pub(super) fn local_ref(&self, id: NodeId, name: Symbol) -> Result<Ex> {
        match self.fctx.as_ref().and_then(|c| c.locals.get(&name)) {
            Some(s) => Ok(Ex::atom(s.clone())),
            None => Err(Refusal::not_yet(Some(id), "a function body")),
        }
    }

    /// A `scan` whose step adds: `cumulative_sum`, from `init = 0`.
    pub(super) fn additive_scan(
        &mut self,
        id: NodeId,
        f: NodeId,
        init: NodeId,
        xs: NodeId,
    ) -> Result<Ex> {
        let zero = matches!(
            self.m.node(self.resolve(init)),
            Node::Lit(flatppl_core::Scalar::Int(0))
        ) || matches!(self.m.node(self.resolve(init)), Node::Lit(flatppl_core::Scalar::Real(x)) if *x == 0.0);
        let adds = match self.m.node(f) {
            Node::Const(c) => self.name(*c) == "add",
            _ => {
                match self.user_callee(f) {
                    Some(fun) => {
                        let (body, entries) = self.callable_parts(fun)?;
                        let refs: Vec<Ref> = entries.iter().map(|(_, r)| *r).collect();
                        self.head(body) == Some("add")
                            && refs.len() == 2
                            && self.call(body).unwrap().args.iter().all(
                                |&a| matches!(self.m.node(a), Node::Ref(r) if refs.contains(r)),
                            )
                            && self.call(body).unwrap().args.len() == 2
                            && !matches!(
                                (self.m.node(self.call(body).unwrap().args[0]), self.m.node(self.call(body).unwrap().args[1])),
                                (Node::Ref(a), Node::Ref(b)) if a == b
                            )
                    }
                    None => false,
                }
            }
        };
        if !(zero && adds) {
            return Err(Refusal::not_yet(Some(id), "scan"));
        }
        let x = self.expr(xs)?;
        self.table_fn(id, "cumsum", None, vec![x])
    }

    /// `true` when `c` calls a user function (not a standard module member).
    pub(super) fn is_user_call(&self, id: NodeId) -> Option<NodeId> {
        let c = self.call(id)?;
        let CallHead::User(callee) = c.head else {
            return None;
        };
        if let Node::Ref(r) = self.m.node(callee)
            && matches!(r.ns, RefNs::Module(_))
        {
            return None;
        }
        self.user_callee(callee)
    }
}
