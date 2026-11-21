//! `aggregate(f, [.i, …], body)` (spec §04 "Multi-axis aggregation").
//!
//! With no reduced axis the body is one whole-array expression over the
//! output axes: each axis-indexed operand becomes a slice, transposed or
//! repeated (`rep_matrix`) into the output frame. A body with no such form
//! (`ifelse`, a comparison) fills the output by loops. Reduced axes become
//! the table reduction of a slice inside loops over the output axes:
//! `capture_totals[i] = sum(y[i, :])`.

use super::{Emitter, Ex, P_ADD, P_ATOM, P_MUL, P_POW, P_UNARY};
use crate::refuse::{Refusal, Result};
use crate::tables;
use flatppl_core::{Node, NodeId, Phase, Scalar, Symbol};
use std::collections::HashMap;

/// A body value over the lifted axes: either a scalar or the whole frame.
#[derive(Clone)]
pub(super) struct Lifted {
    pub ex: Ex,
    pub full: bool,
    /// A Stan array (integer or real), which needs `to_vector`/`to_matrix`
    /// before vector arithmetic.
    pub array: bool,
}

/// What an axis-indexed operand reads.
struct Site {
    base: String,
    /// The axis at each index position, `None` for a fixed index.
    axes: Vec<Option<Symbol>>,
}

impl<'m> Emitter<'m> {
    /// The binding `base` holding the value of the `aggregate` call `id`.
    pub(super) fn aggregate_value(&mut self, base: &str, id: NodeId) -> Result<String> {
        let c = self.call(id).unwrap();
        let [f, out_list, body] = c.args[..] else {
            return Err(Refusal::not_yet(Some(id), "aggregate with this arity"));
        };
        let Node::Const(fs) = self.m.node(f) else {
            return Err(Refusal::not_yet(
                Some(id),
                "aggregate with a user reduction",
            ));
        };
        let fname = self.name(*fs);
        let outs = self
            .axis_list(out_list)
            .ok_or_else(|| Refusal::not_yet(Some(id), "aggregate with a computed axis list"))?;
        let mut all = Vec::new();
        let sites = self.sites(body, &mut all)?;
        let reduced: Vec<Symbol> = all.iter().filter(|a| !outs.contains(a)).copied().collect();
        let mut ext = HashMap::new();
        for s in &sites {
            for (pos, a) in s.axes.iter().enumerate() {
                if let Some(a) = a
                    && let Some(e) = self.extents.get(&s.base).and_then(|e| e.get(pos))
                {
                    ext.entry(*a).or_insert_with(|| e.1.clone());
                }
            }
        }

        // `sum` over one axis of a squared matrix: one dot-self call.
        if fname == "sum"
            && let ([out], [red]) = (outs.as_slice(), reduced.as_slice())
            && let Some(e) = self.dot_self(body, *out, *red)?
        {
            return self.place(base, id, Ok(e));
        }
        // `sum` over one axis of a matrix: one product with a ones vector.
        if fname == "sum"
            && let ([out], [red]) = (outs.as_slice(), reduced.as_slice())
            && let [site] = sites.as_slice()
            && self.indexes_axis(self.resolve(body))
            && self
                .decls
                .get(&site.base)
                .is_some_and(|t| t.base == "matrix" && t.array.is_empty())
        {
            let b = &site.base;
            let e = match site.axes.as_slice() {
                [Some(a), Some(r)] if (a, r) == (out, red) => Some(Ex {
                    s: format!("{b} * rep_vector(1, cols({b}))"),
                    prec: P_MUL,
                }),
                [Some(r), Some(a)] if (a, r) == (out, red) => {
                    Some(Ex::atom(format!("(rep_row_vector(1, rows({b})) * {b})'")))
                }
                _ => None,
            };
            if let Some(e) = e {
                return self.place(base, id, Ok(e));
            }
        }
        if reduced.is_empty()
            && outs.len() <= 2
            && let Ok(l) = self.lift(body, &outs, &HashMap::new(), &ext)
            && (l.full || outs.is_empty())
        {
            let e = if l.array && !self.shape(id).is_some_and(|s| s.int) {
                self.to_real(&l.ex, outs.len())
            } else {
                l.ex
            };
            return self.place(base, id, Ok(e));
        }

        // A rank-3 result has no whole-array form, and element loops are no answer.
        if self.shape(id).is_some_and(|s| s.rank == 3) {
            return Err(Refusal::not_yet(
                Some(id),
                "an aggregate with a rank-3 result",
            ));
        }
        let stan = self.fresh(base);
        let t = self.declare(id, None, false, &stan, &[])?;
        let extents: Vec<String> = t.array.iter().chain(&t.dims).cloned().collect();
        if extents.len() != outs.len() {
            return Err(Refusal::not_yet(Some(id), "aggregate of this shape"));
        }
        // A user function that vectorizes over one output axis: a loop over
        // the other axis fills whole rows or columns.
        if reduced.is_empty()
            && outs.len() == 2
            && t.base == "matrix"
            && let Some(line) = self.axis_vectorized_call(id, body, &outs, &ext, &stan, &extents)?
        {
            let lines = vec![format!("{};", t.render(&stan)), line];
            self.record_decl(&stan, t);
            let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
            self.put(fixed, lines);
            self.remember(id, &stan);
            return Ok(stan);
        }
        let mut vars = HashMap::new();
        let mut order = Vec::new();
        for a in &outs {
            let v = self.loop_var();
            vars.insert(*a, v.clone());
            order.push(v);
        }
        let value = if reduced.is_empty() {
            self.lift(body, &[], &vars, &ext).map(|l| l.ex)
        } else {
            match self.lift(body, &reduced, &vars, &ext) {
                Ok(l) if l.full => self.table_fn(id, fname, None, vec![l.ex]),
                _ => Err(Refusal::not_yet(
                    Some(id),
                    "aggregate reducing over this body",
                )),
            }
        };
        for v in &order {
            self.used.remove(v);
        }
        let value = value?;
        let mut lines = vec![];
        if order.is_empty() {
            lines.push(format!("{} = {};", t.render(&stan), value.s));
        } else {
            lines.push(format!("{};", t.render(&stan)));
            let mut text = format!("{stan}[{}] = {};", order.join(", "), value.s);
            for (v, n) in order.iter().zip(&extents).rev() {
                let inner: String = text.lines().map(|l| format!("  {l}\n")).collect();
                text = format!("for ({v} in 1:{n}) {{\n{inner}}}");
            }
            lines.push(text);
        }
        self.record_decl(&stan, t);
        let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
        self.put(fixed, lines);
        self.remember(id, &stan);
        Ok(stan)
    }

    /// `X[.out, .red]^2` summed over `.red` is `rows_dot_self(X)`, and
    /// `X[.red, .out]^2` is `columns_dot_self(X)'` (Functions Reference,
    /// "Dot products and specialized products"), for a declared `matrix` X.
    fn dot_self(&mut self, body: NodeId, out: Symbol, red: Symbol) -> Result<Option<Ex>> {
        let b = self.resolve(body);
        let Some(c) = self.call(b).filter(|_| self.head(b) == Some("pow")) else {
            return Ok(None);
        };
        let [site, k] = c.args[..] else {
            return Ok(None);
        };
        let two = matches!(self.m.node(self.resolve(k)), Node::Lit(Scalar::Int(2)))
            || matches!(self.m.node(self.resolve(k)), Node::Lit(Scalar::Real(x)) if *x == 2.0);
        if !two || !self.indexes_axis(site) {
            return Ok(None);
        }
        let sc = self.call(site).unwrap();
        let axis = |me: &Self, n: NodeId| match me.m.node(n) {
            Node::Axis(ax) if ax.variance.is_none() => Some(ax.name),
            _ => None,
        };
        let [x, s0, s1] = sc.args[..] else {
            return Ok(None);
        };
        let x = self.expr(x)?;
        let matrix = self
            .decls
            .get(&x.s)
            .is_some_and(|t| t.base == "matrix" && t.array.is_empty());
        if !matrix {
            return Ok(None);
        }
        Ok(match (axis(self, s0), axis(self, s1)) {
            (Some(a), Some(r)) if (a, r) == (out, red) => {
                Some(Ex::atom(format!("rows_dot_self({})", x.s)))
            }
            (Some(r), Some(a)) if (a, r) == (out, red) => {
                Some(Ex::atom(format!("columns_dot_self({})'", x.s)))
            }
            _ => None,
        })
    }

    /// `stan[i] = f(…)'` or `stan[:, k] = f(…)` in one loop, when the body is
    /// a user call that slice B vectorizes over one of the two output axes.
    /// The loop runs over the shorter axis when both lift.
    fn axis_vectorized_call(
        &mut self,
        id: NodeId,
        body: NodeId,
        outs: &[Symbol],
        ext: &HashMap<Symbol, String>,
        stan: &str,
        extents: &[String],
    ) -> Result<Option<String>> {
        let Some(fun) = self.is_user_call(body) else {
            return Ok(None);
        };
        let c = self.call(body).unwrap();
        if !c.named.is_empty() {
            return Ok(None);
        }
        let args = c.args.to_vec();
        let lens: Vec<u32> = match self.ty(id) {
            Some(flatppl_core::Type::Array { shape, .. }) => shape
                .iter()
                .map(|d| match d {
                    flatppl_core::Dim::Static(n) => *n,
                    flatppl_core::Dim::Dynamic => u32::MAX,
                })
                .collect(),
            _ => return Ok(None),
        };
        // The vectorized axis first: the longer one, so the loop is shorter.
        let order: [usize; 2] = if lens.get(1) >= lens.first() {
            [1, 0]
        } else {
            [0, 1]
        };
        for inner in order {
            let outer = 1 - inner;
            let v = self.loop_var();
            let attempt = self.axis_call(
                body,
                fun,
                &args,
                outs[inner],
                outs[outer],
                &v,
                ext,
                lens[inner],
            );
            self.used.remove(&v);
            let Some(call) = attempt? else {
                continue;
            };
            let line = if inner == 1 {
                format!("{stan}[{v}] = {}';", Ex::atom(call).at(P_ATOM))
            } else {
                format!("{stan}[:, {v}] = {call};")
            };
            let n = &extents[outer];
            let inner_text: String = line.lines().map(|l| format!("  {l}\n")).collect();
            return Ok(Some(format!("for ({v} in 1:{n}) {{\n{inner_text}}}")));
        }
        Ok(None)
    }

    /// `f(…)` with the arguments over axis `inner` as whole vectors and the
    /// axis `outer` at the loop variable `v`; `None` when the body does not
    /// lift.
    #[allow(clippy::too_many_arguments)]
    fn axis_call(
        &mut self,
        body: NodeId,
        fun: NodeId,
        args: &[NodeId],
        inner: Symbol,
        outer: Symbol,
        v: &str,
        ext: &HashMap<Symbol, String>,
        n: u32,
    ) -> Result<Option<String>> {
        use flatppl_core::{Dim, ScalarType, Type};
        let vars = HashMap::from([(outer, v.to_string())]);
        let mut parts = Vec::new();
        let mut tys = Vec::new();
        let mut elem_tys = Vec::new();
        let mut full = Vec::new();
        for &a in args {
            let Ok(l) = self.lift(a, &[inner], &vars, ext) else {
                return Ok(None);
            };
            let Some(et) = self.ty(a) else {
                return Ok(None);
            };
            if l.full {
                let ex = if l.array {
                    self.to_real(&l.ex, 1)
                } else {
                    l.ex
                };
                parts.push(ex.s);
                tys.push(Some(Type::Array {
                    shape: vec![Dim::Static(n)].into(),
                    elem: Box::new(Type::Scalar(ScalarType::Real)),
                }));
                full.push(true);
            } else {
                parts.push(l.ex.s);
                tys.push(Some(et.clone()));
                full.push(false);
            }
            elem_tys.push(et);
        }
        if !full.iter().any(|f| *f) {
            return Ok(None);
        }
        let ret = Type::Array {
            shape: vec![Dim::Static(n)].into(),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        };
        match self.vectorized_function(body, fun, args, &tys, &elem_tys, &full, 1, Some(ret))? {
            Ok((name, caps)) => Ok(Some(Self::apply_user(&name, parts, &caps))),
            Err(_) => Ok(None),
        }
    }

    /// The axis names of an `[.i, .k]` list.
    fn axis_list(&self, list: NodeId) -> Option<Vec<Symbol>> {
        if self.head(list) != Some("vector") {
            return None;
        }
        self.call(list)
            .unwrap()
            .args
            .iter()
            .map(|&a| match self.m.node(a) {
                Node::Axis(ax) if ax.variance.is_none() => Some(ax.name),
                _ => None,
            })
            .collect()
    }

    /// `true` when `body` reads a lift leaf: an axis, a function parameter
    /// in `self.vlocals`, or a binding that changes with the inputs.
    fn reads_leaf(&self, body: NodeId) -> bool {
        let mut stack = vec![body];
        while let Some(id) = stack.pop() {
            match self.m.node(id) {
                Node::Axis(_) => return true,
                Node::Ref(r)
                    if self
                        .vlocals
                        .as_ref()
                        .is_some_and(|v| v.contains_key(&r.name)) =>
                {
                    return true;
                }
                Node::Ref(r)
                    if r.ns == flatppl_core::RefNs::SelfMod
                        && self
                            .fctx
                            .as_ref()
                            .is_some_and(|c| c.through.contains(&r.name)) =>
                {
                    return true;
                }
                _ => {}
            }
            self.m.for_each_child(id, |k| stack.push(k));
        }
        false
    }

    fn indexes_axis(&self, id: NodeId) -> bool {
        matches!(self.head(id), Some("get"))
            && self.call(id).unwrap().args[1..]
                .iter()
                .any(|&s| matches!(self.m.node(s), Node::Axis(_)))
    }

    /// Every axis-indexed operand of `body`, with the axis names in order of
    /// first appearance.
    fn sites(&mut self, body: NodeId, all: &mut Vec<Symbol>) -> Result<Vec<Site>> {
        let mut out = Vec::new();
        let mut stack = vec![body];
        while let Some(id) = stack.pop() {
            if !self.indexes_axis(id) {
                if matches!(self.m.node(id), Node::Axis(_)) {
                    return Err(Refusal::not_yet(Some(id), "an axis name used as a value"));
                }
                let mut kids = Vec::new();
                self.m.for_each_child(id, |c| kids.push(c));
                stack.extend(kids.into_iter().rev());
                continue;
            }
            let c = self.call(id).unwrap();
            let (x, sels) = (c.args[0], c.args[1..].to_vec());
            let mut base = self.expr(x)?;
            // A computed array is placed once and indexed by name.
            if base.prec < P_ATOM || !super::structure::is_identifier(&base.s) {
                let name = match &self.owner {
                    Some((n, _)) => format!("{n}_site"),
                    None => "site".to_string(),
                };
                base = Ex::atom(self.place(&name, x, Ok(base))?);
            }
            let mut axes = Vec::new();
            for s in sels {
                match self.m.node(s) {
                    Node::Axis(ax) if ax.variance.is_none() => {
                        if axes.contains(&Some(ax.name)) {
                            return Err(Refusal::not_yet(
                                Some(id),
                                "a repeated axis in one index list",
                            ));
                        }
                        if !all.contains(&ax.name) {
                            all.push(ax.name);
                        }
                        axes.push(Some(ax.name));
                    }
                    _ => axes.push(None),
                }
            }
            out.push(Site { base: base.s, axes });
        }
        Ok(out)
    }

    /// `body` over the axes `frame` (outermost first), with other axes bound
    /// to the loop variables `vars`. A function parameter in `self.vlocals`
    /// is a leaf too: `f.(x)` lifted over the whole array `x`.
    pub(super) fn lift(
        &mut self,
        body: NodeId,
        frame: &[Symbol],
        vars: &HashMap<Symbol, String>,
        ext: &HashMap<Symbol, String>,
    ) -> Result<Lifted> {
        if frame.len() > 2 {
            return Err(Refusal::not_yet(
                Some(body),
                "an aggregate frame of rank > 2",
            ));
        }
        if self.indexes_axis(body) {
            return self.lift_site(body, frame, vars, ext);
        }
        // A subexpression that reads no leaf is held: it prints as written,
        // whatever its type (`sum(W * v)`, `W[2, 1]`).
        if !self.reads_leaf(body) {
            return Ok(Lifted {
                ex: self.expr(body)?,
                full: false,
                array: false,
            });
        }
        let scalar = |ex: Ex| Lifted {
            ex,
            full: false,
            array: false,
        };
        let node = self.m.node(body);
        if let Node::Ref(r) = node
            && let Some(l) = self.vlocals.as_ref().and_then(|v| v.get(&r.name))
        {
            return Ok(l.clone());
        }
        // A binding between the body and a module-binding input: its value
        // in the body.
        if let Node::Ref(r) = node
            && r.ns == flatppl_core::RefNs::SelfMod
            && self
                .fctx
                .as_ref()
                .is_some_and(|c| c.through.contains(&r.name))
        {
            let rhs = self.m.binding(self.m.binding_by_name(r.name).unwrap()).rhs;
            return self.lift(rhs, frame, vars, ext);
        }
        let Node::Call(c) = node else {
            if self.shape(body).is_some_and(|s| s.rank == 0) {
                return Ok(scalar(self.expr(body)?));
            }
            return Err(Refusal::not_yet(
                Some(body),
                "an array inside an aggregate body",
            ));
        };
        // A scalar field of a record argument (`p.a`), or any field of a
        // held record (`q.w` of `q.w[1]`).
        let held_record = c.args.first().is_some_and(|&a| {
            matches!(self.m.node(a), Node::Ref(r)
                if self.fctx.as_ref().is_some_and(|x| x.records.contains_key(&r.name)))
        });
        if self.head(body) == Some("get")
            && c.args.len() == 2
            && matches!(
                self.m.node(c.args[1]),
                Node::Lit(flatppl_core::Scalar::Str(_))
            )
            && (held_record || self.shape(body).is_some_and(|s| s.rank == 0))
        {
            return Ok(scalar(self.expr(body)?));
        }
        // A user function of scalars, per element: its Stan function.
        if let Some(fun) = self.is_user_call(body) {
            let args = c.args.to_vec();
            let mut parts = Vec::new();
            let mut tys = Vec::new();
            for &a in &args {
                let p = self.lift(a, frame, vars, ext)?;
                if p.full {
                    return Err(Refusal::not_yet(
                        Some(body),
                        "a user function over an aggregate frame",
                    ));
                }
                parts.push(p.ex.s);
                tys.push(
                    self.ty(a)
                        .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?,
                );
            }
            let (name, caps) = self.user_function(body, fun, &tys)?;
            return Ok(scalar(Ex::atom(Self::apply_user(&name, parts, &caps))));
        }
        let (f, args): (&str, Vec<NodeId>) = match self.head(body) {
            Some("broadcast") => match self.m.node(c.args[0]) {
                Node::Const(f) => (self.name(*f), c.args[1..].to_vec()),
                _ => return Err(Refusal::not_yet(Some(body), "broadcast of a user function")),
            },
            Some(h) => (h, c.args.to_vec()),
            None => unreachable!(),
        };
        if !c.named.is_empty() {
            return Err(Refusal::not_yet(
                Some(body),
                format!("{f} with keyword arguments"),
            ));
        }
        let mut parts = Vec::new();
        for &a in &args {
            parts.push(self.lift(a, frame, vars, ext)?);
        }
        if parts.iter().all(|p| !p.full) {
            // An element of a held array (`w[1]`) is one scalar.
            if f == "get" {
                return Ok(scalar(self.expr(body)?));
            }
            let parts = parts.into_iter().map(|p| p.ex).collect();
            return Ok(scalar(self.apply(body, f, &args, parts)?));
        }
        let rank = frame.len();
        let parts: Vec<(Ex, bool)> = parts
            .into_iter()
            .map(|p| {
                let ex = if p.array {
                    self.to_real(&p.ex, rank)
                } else {
                    p.ex
                };
                (ex, p.full)
            })
            .collect();
        let bin = |op: &str, prec: u8| Ex {
            s: format!("{} {op} {}", parts[0].0.at(prec), parts[1].0.at(prec + 1)),
            prec,
        };
        let full = |ex| Lifted {
            ex,
            full: true,
            array: false,
        };
        Ok(full(match (f, parts.len()) {
            ("add", 2) => bin("+", P_ADD),
            ("sub", 2) => bin("-", P_ADD),
            ("mul", 2) if parts[0].1 && parts[1].1 => bin(".*", P_MUL),
            ("mul", 2) => bin("*", P_MUL),
            ("divide", 2) if parts[1].1 => bin("./", P_MUL),
            ("divide", 2) => bin("/", P_MUL),
            ("pow", 2) => Ex {
                s: format!("{} .^ {}", parts[0].0.at(P_POW + 1), parts[1].0.at(P_POW)),
                prec: P_POW,
            },
            ("neg", 1) => Ex {
                s: format!("-{}", parts[0].0.at(P_UNARY)),
                prec: P_UNARY,
            },
            (name, _) => match tables::function(name, None) {
                Some(row) if row.vectorized && row.kind != "operator" => {
                    let parts = parts.into_iter().map(|p| p.0).collect();
                    self.table_fn(body, name, None, parts)?
                }
                _ => {
                    return Err(Refusal::not_yet(
                        Some(body),
                        format!("{name} over an aggregate frame"),
                    ));
                }
            },
        }))
    }

    /// One axis-indexed operand over `frame`.
    fn lift_site(
        &mut self,
        id: NodeId,
        frame: &[Symbol],
        vars: &HashMap<Symbol, String>,
        ext: &HashMap<Symbol, String>,
    ) -> Result<Lifted> {
        let c = self.call(id).unwrap();
        let (x, sels) = (c.args[0], c.args[1..].to_vec());
        let base = self.expr(x)?.s;
        let decl =
            self.decls.get(&base).cloned().ok_or_else(|| {
                Refusal::not_yet(Some(id), "axis indexing of an undeclared value")
            })?;
        let mut idx = Vec::new();
        let mut kept = Vec::new();
        let mut kept_pos = Vec::new();
        for (pos, s) in sels.iter().enumerate() {
            match self.m.node(*s) {
                Node::Axis(ax) => {
                    if let Some(v) = vars.get(&ax.name) {
                        idx.push(v.clone());
                    } else if frame.contains(&ax.name) {
                        idx.push(":".to_string());
                        kept.push(ax.name);
                        kept_pos.push(pos);
                    } else {
                        return Err(Refusal::not_yet(Some(id), "an unbound axis"));
                    }
                }
                _ => idx.push(self.expr(*s)?.s),
            }
        }
        let text = if idx.iter().all(|i| i == ":") {
            base.clone()
        } else {
            format!("{base}[{}]", idx.join(", "))
        };
        if kept.is_empty() {
            return Ok(Lifted {
                ex: Ex::atom(text),
                full: false,
                array: false,
            });
        }
        let narray = decl.array.len();
        let (mut ex, array) = if narray == 1 && decl.base == "matrix" && sels.len() == 3 {
            // A real rank-3 value, an array of matrices: the index arm's rule.
            let k: Vec<bool> = (0..3).map(|p| kept_pos.contains(&p)).collect();
            let e = super::matrices_slice(&k, text).ok_or_else(|| {
                Refusal::not_yet(Some(id), "`x[:, :, k]` of a rank-3 array of matrices")
            })?;
            (e, false)
        } else {
            let array = narray > 0;
            if array && kept_pos.iter().any(|&p| p >= narray) {
                return Err(Refusal::not_yet(Some(id), "a slice of an array of vectors"));
            }
            // A matrix row slice is a row_vector; the frame keeps column vectors.
            let row_slice =
                !array && ((decl.base == "matrix" && kept_pos == [1]) || decl.base == "row_vector");
            (if row_slice { format!("{text}'") } else { text }, array)
        };
        let in_order: Vec<Symbol> = frame.iter().filter(|a| kept.contains(a)).copied().collect();
        if kept.len() == 2 && kept != in_order {
            if array {
                return Err(Refusal::not_yet(Some(id), "a transposed array slice"));
            }
            ex = format!("{}'", Ex::atom(ex).at(P_ATOM));
        }
        if kept.len() == frame.len() {
            return Ok(Lifted {
                ex: Ex::atom(ex),
                full: true,
                array,
            });
        }
        // One kept axis in a rank-2 frame: repeat it along the other axis.
        let v = if array {
            format!("to_vector({ex})")
        } else {
            ex
        };
        let other = if kept[0] == frame[0] {
            frame[1]
        } else {
            frame[0]
        };
        let n = ext
            .get(&other)
            .cloned()
            .ok_or_else(|| Refusal::not_yet(Some(id), "an aggregate axis of unknown extent"))?;
        let s = if kept[0] == frame[0] {
            format!("rep_matrix({v}, {n})")
        } else {
            format!("rep_matrix({}', {n})", Ex::atom(v).at(P_ATOM))
        };
        Ok(Lifted {
            ex: Ex::atom(s),
            full: true,
            array: false,
        })
    }

    pub(super) fn to_real(&self, ex: &Ex, rank: usize) -> Ex {
        let f = if rank == 2 { "to_matrix" } else { "to_vector" };
        Ex::atom(format!("{f}({})", ex.s))
    }
}
