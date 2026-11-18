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
use crate::stan_type::{StanType, declare, mark_dynamic};
use flatppl_core::{CallHead, Inputs, Node, NodeId, Phase, Ref, RefNs, ScalarType, Symbol, Type};
use std::collections::{HashMap, HashSet};

/// The state of a function body being emitted.
#[derive(Default)]
pub(crate) struct FnCtx {
    /// Temporaries shared by a data summary block guarded on its inferred extents.
    pub plan_arrays: Option<Vec<(NodeId, Ex)>>,
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
    /// Values placed in this body, by node.
    pub placed: HashMap<NodeId, String>,
    /// Non-scalar inlined calls placed in this body: callee, arguments, name.
    pub inlined: Vec<(NodeId, Vec<NodeId>, String)>,
    /// The body must stay one expression, with no statements: a law
    /// argument, a broadcast user law, a lifted map. An inlined call keeps
    /// its text here, unmemoised.
    pub inline_only: bool,
    /// A log weight's record argument: `get(x, "f")` reads the parameter of
    /// field `f`.
    pub record_local: Option<Symbol>,
    /// Record parameters bound field by field: `get(r, "f")` is the name.
    pub records: HashMap<Symbol, HashMap<String, String>>,
    /// Scans emitted in this body, by node.
    pub scans: HashMap<NodeId, super::scan::ScanOut>,
    /// Boundary inputs that name module bindings (`functionof(e, a = a0)`):
    /// a reference to one is the input, bound in `locals` or `records`.
    pub self_inputs: HashSet<Symbol>,
    /// Module bindings between the body and those inputs (spec §04
    /// "Reification to functions and kernels"): their values change with the
    /// inputs, so the body computes them. Every other binding is captured.
    pub through: HashSet<Symbol>,
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
pub(super) fn identifiers(text: &str) -> Vec<String> {
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

/// A type with no unresolved part.
fn concrete(t: &Type) -> bool {
    match t {
        Type::Any | Type::Deferred | Type::Failed(_) => false,
        Type::Array { elem, .. } | Type::TVector { elem, .. } => concrete(elem),
        Type::Record(fs) => fs.iter().all(|(_, t)| concrete(t)),
        Type::Tuple(ts) => ts.iter().all(concrete),
        _ => true,
    }
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
    pub(super) fn callable_parts(&self, fun: NodeId) -> Result<(NodeId, Vec<(Symbol, Ref)>)> {
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
    pub(super) fn instantiate(
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
        // A module-binding input is seeded at its right-hand side, as infer
        // seeds a call's body: the bindings that read it are typed at the
        // argument too.
        for ((_, r), ty) in entries.iter().zip(arg_tys) {
            if r.ns == RefNs::SelfMod
                && let Some(b) = self.m.binding_by_name(r.name)
            {
                seeds.push((self.m.binding(b).rhs, ty.clone()));
            }
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

    /// Mark the module-binding inputs of a callable body and the bindings
    /// between them and the body (see [`FnCtx::through`]).
    pub(super) fn mark_boundary(
        &self,
        ctx: &mut FnCtx,
        at: NodeId,
        body: NodeId,
        entries: &[(Symbol, Ref)],
    ) -> Result<()> {
        let targets: HashSet<Symbol> = entries
            .iter()
            .filter(|(_, r)| r.ns == RefNs::SelfMod)
            .map(|(_, r)| r.name)
            .collect();
        if targets.is_empty() {
            return Ok(());
        }
        ctx.through = self.boundary_reach(at, body, &targets)?;
        ctx.self_inputs = targets;
        Ok(())
    }

    /// The module bindings from which `body` reaches one of `targets` through
    /// self references. A draw binding is not entered: a captured draw keeps
    /// its one realization. A path through a nested reification or a module
    /// reference refuses.
    pub(super) fn boundary_reach(
        &self,
        at: NodeId,
        body: NodeId,
        targets: &HashSet<Symbol>,
    ) -> Result<HashSet<Symbol>> {
        let mut memo = HashMap::new();
        let mut through = HashSet::new();
        self.reach_walk(at, body, targets, false, &mut memo, &mut through)?;
        Ok(through)
    }

    /// `absorbed`: inside a `lawof` or `kernelof`, where a draw is part of
    /// the reified law and not a captured realization.
    fn reach_walk(
        &self,
        at: NodeId,
        node: NodeId,
        targets: &HashSet<Symbol>,
        absorbed: bool,
        memo: &mut HashMap<(Symbol, bool), bool>,
        through: &mut HashSet<Symbol>,
    ) -> Result<bool> {
        match self.m.node(node) {
            Node::Ref(r) => match r.ns {
                RefNs::SelfMod => {
                    if targets.contains(&r.name) {
                        return Ok(true);
                    }
                    if let Some(&v) = memo.get(&(r.name, absorbed)) {
                        return Ok(v);
                    }
                    memo.insert((r.name, absorbed), false);
                    let Some(b) = self.m.binding_by_name(r.name) else {
                        return Ok(false);
                    };
                    let rhs = self.m.binding(b).rhs;
                    if !absorbed && self.head(rhs) == Some("draw") {
                        return Ok(false);
                    }
                    let v = self.reach_walk(at, rhs, targets, absorbed, memo, through)?;
                    memo.insert((r.name, absorbed), v);
                    if v {
                        through.insert(r.name);
                    }
                    Ok(v)
                }
                RefNs::Module(alias) => {
                    let load = self.m.binding_by_name(alias).map(|b| self.m.binding(b).rhs);
                    if let Some(load) = load
                        && self.reach_walk(at, load, targets, absorbed, memo, through)?
                    {
                        return Err(Refusal::not_yet(
                            Some(at),
                            "a boundary input reached through a module reference",
                        ));
                    }
                    Ok(false)
                }
                RefNs::Local => Ok(false),
            },
            Node::Call(c) => {
                let head = self.head(node);
                let nested = matches!(head, Some("functionof" | "kernelof" | "lawof"));
                // A nested reification that declares a name rebinds it.
                let inner: HashSet<Symbol> = match (&c.inputs, nested) {
                    (Some(Inputs::Spec(e)), true) => targets
                        .iter()
                        .filter(|t| !e.iter().any(|(_, r)| r.name == **t))
                        .copied()
                        .collect(),
                    _ => targets.clone(),
                };
                let inner_absorbed = absorbed || matches!(head, Some("lawof" | "kernelof"));
                let mut kids = Vec::new();
                self.m.for_each_child(node, |k| kids.push(k));
                let mut any = false;
                for k in kids {
                    any |= self.reach_walk(at, k, &inner, inner_absorbed, memo, through)?;
                }
                if nested && any {
                    return Err(Refusal::not_yet(
                        Some(at),
                        format!(
                            "a boundary input reached inside a nested `{}`",
                            head.unwrap_or("reification")
                        ),
                    ));
                }
                Ok(any)
            }
            _ => {
                let mut kids = Vec::new();
                self.m.for_each_child(node, |k| kids.push(k));
                let mut any = false;
                for k in kids {
                    any |= self.reach_walk(at, k, targets, absorbed, memo, through)?;
                }
                Ok(any)
            }
        }
    }

    /// A module-binding reference inside a callable body: the input itself,
    /// a binding between the body and an input (computed in the body, an
    /// array once as a local), or `None` for a captured value.
    pub(super) fn boundary_ref(&mut self, id: NodeId, name: Symbol) -> Result<Option<Ex>> {
        let Some(ctx) = &self.fctx else {
            return Ok(None);
        };
        if ctx.self_inputs.contains(&name) {
            return match ctx.locals.get(&name) {
                Some(s) => Ok(Some(Ex::atom(s.clone()))),
                None => Err(Refusal::not_yet(
                    Some(id),
                    "a record boundary input used whole",
                )),
            };
        }
        if !ctx.through.contains(&name) {
            return Ok(None);
        }
        let b = self.m.binding_by_name(name).unwrap();
        let rhs = self.m.binding(b).rhs;
        if let Some(n) = self.recall(rhs) {
            return Ok(Some(Ex::atom(n)));
        }
        let e = self.expr(rhs)?;
        if self.shape(rhs).is_none_or(|s| s.rank == 0) {
            return Ok(Some(e));
        }
        let placed = self.place(self.name(name), rhs, Ok(e))?;
        Ok(Some(Ex::atom(placed)))
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
        let mut ctx = FnCtx {
            scoped: types,
            ..Default::default()
        };
        self.mark_boundary(&mut ctx, at, body, &entries)?;
        let outer_ctx = self.fctx.replace(ctx);
        let outer_owner = self.owner.take();
        self.function_depth += 1;
        let out = self.function_body(at, fun, &base, body, &entries, arg_tys);
        self.function_depth -= 1;
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
            let t = declare(&mark_dynamic(ty), None, false, &mut |_| String::new(), at)?;
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
        let ret_t = declare(
            &mark_dynamic(&ret_ty),
            None,
            false,
            &mut |_| String::new(),
            body,
        )?;
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
        let name =
            self.finish_function(base, &arg_type(&ret_t), &params, &text_body, &ctx.declared);
        self.fctx = Some(ctx);
        Ok((name, caps))
    }

    /// Add a function to the `functions` block, or reuse an identical one,
    /// and name it. Its own names are released, so another instance of the
    /// same function prints the same text.
    fn finish_function(
        &mut self,
        base: &str,
        ret: &str,
        params: &[String],
        body: &[String],
        local: &HashSet<String>,
    ) -> String {
        for n in local {
            self.used.remove(n);
        }
        let mut text = format!("{ret} \u{2}({}) {{\n", params.join(", "));
        for l in body {
            for part in l.lines() {
                text.push_str("  ");
                text.push_str(part);
                text.push('\n');
            }
        }
        text.push('}');
        if let Some(name) = self.fn_texts.get(&text) {
            return name.clone();
        }
        let name = self.function_stan_name(base);
        self.functions.push(text.replace('\u{2}', &name));
        self.fn_texts.insert(text, name.clone());
        name
    }

    /// A Stan name for a user function: never a Stan library function, a
    /// reserved word, or a name with a Stan density or generator suffix.
    fn function_stan_name(&mut self, base: &str) -> String {
        let mut name = super::sanitize(base);
        while crate::tables::clashes_with_stan(&name) {
            name.push_str("_fn");
        }
        self.fresh(&name)
    }

    /// `name(args…, captures…)`.
    pub(super) fn apply_user(name: &str, args: Vec<String>, caps: &[String]) -> String {
        let all: Vec<String> = args.into_iter().chain(caps.iter().cloned()).collect();
        format!("{name}({})", all.join(", "))
    }

    /// A user call `f(args…)`.
    pub(super) fn user_call(&mut self, id: NodeId, fun: NodeId, args: &[NodeId]) -> Result<Ex> {
        if self.has_record_argument(args) {
            return self.inline_call(id, fun, args);
        }
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

    /// The type of `f.(args…)` inside an instantiated body, where inference
    /// leaves a user broadcast untyped: the traversed argument's outer shape
    /// around the callee's result at the element types.
    pub(super) fn ensure_broadcast_type(&mut self, id: NodeId) {
        if self.fctx.is_none() || self.ty(id).as_ref().is_some_and(concrete) {
            return;
        }
        let Some(c) = self.call(id) else { return };
        let Some(fun) = self.user_callee(c.args[0]) else {
            return;
        };
        let args = c.args[1..].to_vec();
        let Some(tys) = args.iter().map(|&a| self.ty(a)).collect::<Option<Vec<_>>>() else {
            return;
        };
        let brank = tys.iter().map(|t| top_rank(Some(t))).max().unwrap_or(0);
        if brank == 0 {
            return;
        }
        let mut shape = None;
        let mut elem_tys = Vec::new();
        for t in tys {
            match (top_rank(Some(&t)), t) {
                (r, Type::Array { shape: s, elem }) if r == brank => {
                    shape.get_or_insert(s);
                    elem_tys.push(*elem);
                }
                (0, t) => elem_tys.push(t),
                _ => return,
            }
        }
        let Ok((body, entries)) = self.callable_parts(fun) else {
            return;
        };
        if entries.len() != elem_tys.len() {
            return;
        }
        let Ok(types) = self.instantiate(id, body, &entries, &elem_tys) else {
            return;
        };
        let (Some(shape), Some(elem)) = (shape, types.get(&body).cloned()) else {
            return;
        };
        if !concrete(&elem) {
            return;
        }
        let ty = Type::Array {
            shape,
            elem: Box::new(elem),
        };
        if let Some(ctx) = self.fctx.as_mut() {
            ctx.scoped.insert(id, ty);
        }
    }

    /// `f.(args…)` as one call of a vectorized Stan function: the body at
    /// whole-array arguments, each operation in its elementwise form. `Err`
    /// gives the reason it loops instead.
    pub(super) fn vectorized_broadcast(
        &mut self,
        id: NodeId,
    ) -> Result<std::result::Result<Ex, String>> {
        self.ensure_broadcast_type(id);
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
        // The broadcast's type in the caller's scope.
        let ret_ty = self.ty(id);
        let (name, caps) = match self
            .vectorized_function(id, fun, &args, &tys, &elem_tys, &full, brank, ret_ty)?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let mut parts = Vec::new();
        for &a in &args {
            parts.push(self.expr(a)?.s);
        }
        Ok(Ok(Ex::atom(Self::apply_user(&name, parts, &caps))))
    }

    /// The vectorized Stan function for `fun` with the arguments marked
    /// `full` passed as whole arrays of rank `brank` and the others held.
    /// `Err` gives the reason the body does not lift.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vectorized_function(
        &mut self,
        id: NodeId,
        fun: NodeId,
        args: &[NodeId],
        tys: &[Option<Type>],
        elem_tys: &[Type],
        full: &[bool],
        brank: usize,
        ret_ty: Option<Type>,
    ) -> Result<std::result::Result<(String, Vec<String>), String>> {
        let key = (fun, format!("vectorized {tys:?}"));
        if let Some(v) = self.fn_memo.get(&key) {
            return Ok(Ok(v.clone()));
        }
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != args.len() {
            return Err(Refusal::not_yet(
                Some(id),
                "a user function called with another arity",
            ));
        }
        let types = self.instantiate(id, body, &entries, elem_tys)?;
        let mut ctx = FnCtx {
            scoped: types,
            inline_only: true,
            ..Default::default()
        };
        self.mark_boundary(&mut ctx, id, body, &entries)?;
        let outer_ctx = self.fctx.replace(ctx);
        let outer_owner = self.owner.take();
        let outer_v = self.vlocals.take();
        self.function_depth += 1;
        let out = self.vectorized_body(id, fun, body, &entries, args, tys, full, brank, ret_ty);
        self.function_depth -= 1;
        let inner = std::mem::replace(&mut self.fctx, outer_ctx);
        self.owner = outer_owner;
        self.vlocals = outer_v;
        let out = out?;
        match &out {
            Ok(v) => {
                self.fn_memo.insert(key, v.clone());
            }
            // A body that does not lift leaves no names behind.
            Err(_) => {
                for n in inner.iter().flat_map(|c| c.declared.iter()) {
                    self.used.remove(n);
                }
            }
        }
        Ok(out)
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
        ret_ty: Option<Type>,
    ) -> Result<std::result::Result<(String, Vec<String>), String>> {
        let mut params = Vec::new();
        let mut leaves = HashMap::new();
        let mut frame = Vec::new();
        for (((name, r), _a), (ty, &f)) in entries.iter().zip(args).zip(tys.iter().zip(full)) {
            let p = self.fresh(self.name(*name));
            let ty = ty.clone().unwrap();
            // A traversed argument is passed whole; a held one as it is.
            let t = declare(&mark_dynamic(&ty), None, false, &mut |_| String::new(), at)?;
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
        let ret_ty = ret_ty.ok_or_else(|| Refusal::not_yet(Some(at), "an untyped broadcast"))?;
        let ret_t = declare(
            &mark_dynamic(&ret_ty),
            None,
            false,
            &mut |_| String::new(),
            at,
        )?;
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
        let body = [format!("return {};", ex.s)];
        let name = self.finish_function(&base, &arg_type(&ret_t), &params, &body, &ctx.declared);
        self.fctx = Some(ctx);
        Ok(Ok((name, caps)))
    }

    /// `base` holding `f.(args…)`: one vectorized call when the body lifts,
    /// else filled by loops over the broadcast axes.
    pub(super) fn user_broadcast_value(&mut self, base: &str, id: NodeId) -> Result<String> {
        if let Some(e) = self.row_selector_broadcast(id)? {
            if super::structure::is_identifier(&e.s) {
                self.remember(id, &e.s);
                return Ok(e.s);
            }
            return self.place(base, id, Ok(e));
        }
        if let Ok(e) = self.vectorized_broadcast(id)? {
            return self.place(base, id, Ok(e));
        }
        self.looped_broadcast(base, id)
    }

    /// `f.(indicesof(…))` where `f(i) = A[i, :, …]` and the range covers A's
    /// whole outer axis: the rows of A in order, which is A itself. Only for
    /// an integer A or a vector, whose array-of-rows and whole-array Stan
    /// types are the same.
    pub(super) fn row_selector_broadcast(&mut self, id: NodeId) -> Result<Option<Ex>> {
        let c = self.call(id).unwrap();
        if !c.named.is_empty() || c.args.len() != 2 {
            return Ok(None);
        }
        let Some(fun) = self.user_callee(c.args[0]) else {
            return Ok(None);
        };
        // Over a whole outer range the rows are A; over any other integer
        // index array they are the gather `A[idx]`.
        let range = self.resolve(c.args[1]);
        let whole = self.head(range) == Some("indicesof");
        // `Shape.int` counts Booleans too; a mask is no index array.
        let int_index = matches!(self.ty(c.args[1]), Some(Type::Array { shape, elem })
            if shape.len() == 1 && matches!(*elem, Type::Scalar(ScalarType::Integer)));
        if !whole && !int_index {
            return Ok(None);
        }
        let (body, entries) = self.callable_parts(fun)?;
        let [(_, param)] = entries[..] else {
            return Ok(None);
        };
        let b = self.resolve(body);
        if self.head(b) != Some("get") {
            return Ok(None);
        }
        let gc = self.call(b).unwrap();
        let [a, first, rest @ ..] = &gc.args[..] else {
            return Ok(None);
        };
        let all =
            |me: &Self, s: NodeId| matches!(me.m.node(s), Node::Const(k) if me.name(*k) == "all");
        let is_param = matches!(self.m.node(*first), Node::Ref(r) if *r == param);
        let Some(shape) = self.shape(*a) else {
            return Ok(None);
        };
        if !is_param
            || !rest.iter().all(|&s| all(self, s))
            || rest.len() + 1 != shape.rank
            || !(shape.int || shape.rank == 1)
        {
            return Ok(None);
        }
        // A must not read the selector anywhere else.
        let mut stack = vec![*a];
        while let Some(n) = stack.pop() {
            if matches!(self.m.node(n), Node::Ref(r) if *r == param) {
                return Ok(None);
            }
            self.m.for_each_child(n, |k| stack.push(k));
        }
        let a = *a;
        if !whole {
            let idx = c.args[1];
            let base = self.at_module(|me| me.expr(a))?;
            let ix = self.expr(idx)?;
            return Ok(Some(super::Ex::atom(format!(
                "{}[{}]",
                base.at(P_ATOM),
                ix.s
            ))));
        }
        match (self.sym_extent(range, 0, 0), self.sym_extent(a, 0, 0)) {
            (Some(n), Some(m)) if n == m => {}
            _ => return Ok(None),
        }
        Ok(Some(self.at_module(|me| me.expr(a))?))
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
        self.ensure_broadcast_type(id);
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
            if tys.iter().any(|t| matches!(t, Some(Type::Table { .. }))) && decl == id {
                return self.table_broadcast(base, id);
            }
            return Err(Refusal::not_yet(Some(id), "a broadcast over no array"));
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
        self.remember(decl, &stan);
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
        let depth = std::mem::take(&mut self.function_depth);
        let out = f(self);
        self.function_depth = depth;
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
        for ((pname, r), &a) in entries.iter().zip(&args) {
            // `splitblocks(A, [1, n])`: element i is row i of A, typed as a
            // row (a 1×n block and its row hold the same entries).
            if let Some(m_) = self.row_blocks(a)
                && let Some(Type::Array { shape, elem }) = self.ty(m_)
                && shape.len() == 2
            {
                let e = self.expr(m_)?;
                elem_tys.push(Type::Array {
                    shape: vec![shape[1]].into(),
                    elem,
                });
                locals.insert(r.name, format!("{}[{index}]", e.at(P_ATOM)));
                continue;
            }
            let t = self
                .ty(a)
                .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?;
            let e = self.expr(a)?;
            let (et, s) = match t {
                Type::Array { shape, elem } if shape.len() == 1 => {
                    // A computed array is computed once, under the parameter's
                    // name, not once per element.
                    let plain = e.s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                    let e = if plain || self.fctx.is_some() {
                        e
                    } else {
                        let base = self.name(*pname).to_string();
                        super::Ex::atom(self.place(&base, a, Ok(e))?)
                    };
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
        let mut ctx = FnCtx {
            locals,
            scoped,
            inline_only: true,
            ..Default::default()
        };
        self.mark_boundary(&mut ctx, m, body, &entries)?;
        Ok((body, ctx))
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
        let mut marks = FnCtx::default();
        self.mark_boundary(&mut marks, m, body, &entries)?;
        let through = marks.through.clone();
        let mentions = |me: &Self, root: NodeId| {
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                match me.m.node(id) {
                    Node::Ref(r) if refs.contains(r) => return true,
                    Node::Ref(r) if r.ns == RefNs::SelfMod && through.contains(&r.name) => {
                        return true;
                    }
                    _ => {}
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
            self_inputs: marks.self_inputs,
            through: marks.through,
            scoped,
            inline_only: true,
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
