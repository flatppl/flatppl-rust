//! The structural walk: target → prior and kernel → Stan blocks.

use crate::data::{self, DataSource};
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::{StanType, declare};
use crate::tables;
use crate::{EmitOptions, ManifestParam, Placement};
use flatppl_core::{
    CallHead, Module, Node, NodeId, Phase, RefNs, Scalar, ScalarType, Symbol, Type, ValueSet,
};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

mod elementwise;
mod pattern;
mod structure;

/// Expression precedence, loosest first, as Stan parses it.
const P_TERNARY: u8 = 0;
const P_CMP: u8 = 3;
const P_ADD: u8 = 4;
const P_MUL: u8 = 5;
const P_UNARY: u8 = 6;
const P_POW: u8 = 7;
const P_ATOM: u8 = 9;

#[derive(Clone, Debug)]
pub(crate) struct Ex {
    s: String,
    prec: u8,
}

impl Ex {
    fn atom(s: impl Into<String>) -> Ex {
        Ex {
            s: s.into(),
            prec: P_ATOM,
        }
    }
    fn at(&self, prec: u8) -> String {
        if self.prec >= prec {
            self.s.clone()
        } else {
            format!("({})", self.s)
        }
    }
}

/// Rank and integer-ness of a value, read from its inferred type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    rank: usize,
    int: bool,
}

fn shape_of(ty: &Type) -> Option<Shape> {
    match ty {
        Type::Scalar(s) => Some(Shape {
            rank: 0,
            int: matches!(s, ScalarType::Integer | ScalarType::Boolean),
        }),
        Type::TVector { elem, .. } => shape_of(elem).map(|s| Shape {
            rank: s.rank + 1,
            int: s.int,
        }),
        Type::Array { shape, elem } => shape_of(elem).map(|s| Shape {
            rank: s.rank + shape.len(),
            int: s.int,
        }),
        _ => None,
    }
}

/// One prior field of a posterior target.
struct PriorField {
    field: Symbol,
    /// The draw or elementof binding the field denotes, when it is a binding.
    binding: Option<Symbol>,
    /// The law the field is drawn from; `None` for a flat (Lebesgue) part.
    law: Option<NodeId>,
    node: NodeId,
}

pub(crate) struct Emitter<'m> {
    m: &'m Module,
    opts: &'m EmitOptions,
    used: HashSet<String>,
    /// FlatPPL binding → the Stan name it is bound to.
    env: HashMap<Symbol, String>,
    /// Prior field or kernel input name → its Stan parameter.
    fields: HashMap<Symbol, String>,
    /// Stan name → its declaration.
    decls: HashMap<String, StanType>,
    /// Kernel inputs inside the prior, bound by name to prior fields.
    links: Vec<(Symbol, Symbol)>,
    /// `functions` block sources, deduplicated.
    functions: Vec<String>,
    /// (record binding, field) → data leaf name.
    leaves: HashMap<(Symbol, String), String>,
    in_progress: HashSet<Symbol>,
    sizes: HashMap<u32, Vec<Symbol>>,
    data: Vec<String>,
    tdata: Vec<String>,
    params: Vec<String>,
    tparams: Vec<String>,
    locals: Vec<String>,
    prior: Vec<String>,
    lik: Vec<String>,
    json: Map<String, Value>,
    deferred: Vec<DataSource>,
    manifest: Vec<ManifestParam>,
}

fn sanitize(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !s.starts_with(|c: char| c.is_ascii_alphabetic()) {
        s = format!("v{s}");
    }
    if s.ends_with("__") {
        s.push('x');
    }
    if tables::RESERVED.contains(&s.as_str()) {
        s.push('_');
    }
    s
}

impl<'m> Emitter<'m> {
    pub(crate) fn new(m: &'m Module, opts: &'m EmitOptions) -> Self {
        let mut sizes: HashMap<u32, Vec<Symbol>> = HashMap::new();
        for (_, b) in m.bindings() {
            if let Node::Lit(Scalar::Int(n)) = m.node(b.rhs)
                && *n > 0
                && b.public
                && let Ok(n) = u32::try_from(*n)
            {
                sizes.entry(n).or_default().push(b.name);
            }
        }
        Emitter {
            m,
            opts,
            used: HashSet::new(),
            env: HashMap::new(),
            fields: HashMap::new(),
            decls: HashMap::new(),
            links: vec![],
            functions: vec![],
            leaves: HashMap::new(),
            in_progress: HashSet::new(),
            sizes,
            data: vec![],
            tdata: vec![],
            params: vec![],
            tparams: vec![],
            locals: vec![],
            prior: vec![],
            lik: vec![],
            json: Map::new(),
            deferred: vec![],
            manifest: vec![],
        }
    }

    fn record_decl(&mut self, name: &str, t: StanType) {
        self.decls.insert(name.to_string(), t);
    }

    fn name(&self, s: Symbol) -> &'m str {
        self.m.resolve(s)
    }

    fn fresh(&mut self, base: &str) -> String {
        let base = sanitize(base);
        let mut name = base.clone();
        let mut k = 2;
        while self.used.contains(&name) {
            name = format!("{base}_{k}");
            k += 1;
        }
        self.used.insert(name.clone());
        name
    }

    fn head(&self, id: NodeId) -> Option<&'m str> {
        match self.m.node(id) {
            Node::Call(c) => match c.head {
                CallHead::Builtin(h) => Some(self.m.resolve(h)),
                CallHead::User(_) => Some("%call"),
            },
            _ => None,
        }
    }

    fn call(&self, id: NodeId) -> Option<&'m flatppl_core::Call> {
        match self.m.node(id) {
            Node::Call(c) => Some(c),
            _ => None,
        }
    }

    /// Follow `self` references through bindings whose value is not a draw.
    fn resolve(&self, mut id: NodeId) -> NodeId {
        for _ in 0..64 {
            let Node::Ref(r) = self.m.node(id) else {
                return id;
            };
            if r.ns != RefNs::SelfMod {
                return id;
            }
            let Some(b) = self.m.binding_by_name(r.name) else {
                return id;
            };
            let rhs = self.m.binding(b).rhs;
            if self.head(rhs) == Some("draw") {
                return id;
            }
            id = rhs;
        }
        id
    }

    fn self_ref(&self, id: NodeId) -> Option<Symbol> {
        match self.m.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => Some(r.name),
            _ => None,
        }
    }

    /// Positional plus keyword arguments in the law's canonical order.
    fn law_args(&self, name: &str, id: NodeId) -> Result<Vec<NodeId>> {
        let c = self.call(id).expect("law is a call");
        let params = flatppl_infer::distribution_param_names(name).unwrap_or_default();
        let mut out: Vec<Option<NodeId>> = c.args.iter().map(|&a| Some(a)).collect();
        for na in c.named.iter() {
            let key = self.name(na.name);
            let Some(pos) = params.iter().position(|p| p == key) else {
                return Err(Refusal::not_yet(Some(id), format!("{name}({key} = …)")));
            };
            if out.len() <= pos {
                out.resize(pos + 1, None);
            }
            out[pos] = Some(na.value);
        }
        out.into_iter()
            .map(|a| {
                a.ok_or_else(|| {
                    Refusal::not_yet(Some(id), format!("{name} with a missing argument"))
                })
            })
            .collect()
    }

    // ---------------------------------------------------------------- sizes

    /// A static extent, named by the unique fixed integer binding with that value.
    fn size(&mut self, n: u32) -> String {
        match self.sizes.get(&n).map(Vec::as_slice) {
            Some([sym]) => {
                let sym = *sym;
                self.bind(sym).unwrap_or_else(|_| n.to_string())
            }
            _ => n.to_string(),
        }
    }

    fn declare(&mut self, node: NodeId, vs: Option<&ValueSet>, bounded: bool) -> Result<StanType> {
        let ty = self
            .m
            .type_of(node)
            .cloned()
            .ok_or_else(|| Refusal::not_yet(Some(node), "an untyped value"))?;
        self.declare_ty(&ty, vs, bounded, node)
    }

    fn declare_ty(
        &mut self,
        ty: &Type,
        vs: Option<&ValueSet>,
        bounded: bool,
        node: NodeId,
    ) -> Result<StanType> {
        // A two-pass size resolution: collect extents, then name them, so the
        // size closure does not borrow `self` while `declare` runs.
        let mut extents = Vec::new();
        declare(
            ty,
            vs,
            bounded,
            &mut |n| {
                extents.push(n);
                format!("\u{0}{}\u{0}", extents.len() - 1)
            },
            node,
        )?;
        let names: Vec<String> = extents.iter().map(|&n| self.size(n)).collect();
        let mut k = 0;
        let t = declare(
            ty,
            vs,
            bounded,
            &mut |_| {
                k += 1;
                names[k - 1].clone()
            },
            node,
        )?;
        Ok(t)
    }

    // ------------------------------------------------------------- bindings

    /// The Stan name for a top-level binding, emitting its declaration on
    /// first use.
    fn bind(&mut self, sym: Symbol) -> Result<String> {
        if let Some(s) = self.env.get(&sym) {
            return Ok(s.clone());
        }
        let Some(bid) = self.m.binding_by_name(sym) else {
            return Err(Refusal::not_yet(
                None,
                format!("the unbound name `{}`", self.name(sym)),
            ));
        };
        if !self.in_progress.insert(sym) {
            return Err(Refusal::not_yet(
                None,
                format!("the cyclic binding `{}`", self.name(sym)),
            ));
        }
        let out = self.bind_new(sym, self.m.binding(bid).rhs);
        self.in_progress.remove(&sym);
        let out = out?;
        self.env.insert(sym, out.clone());
        Ok(out)
    }

    fn bind_new(&mut self, sym: Symbol, rhs: NodeId) -> Result<String> {
        let name = self.name(sym);
        if let Some(other) = self.self_ref(rhs) {
            return self.bind(other);
        }
        match self.head(rhs) {
            Some("draw") => return Err(self.latent(sym, rhs)),
            Some("elementof") => {
                return Err(Refusal::new(
                    RefusalKind::NotYetSupported,
                    Some(rhs),
                    format!("elementof in `{name}`"),
                    format!("`{name}` is neither a prior field nor a kernel input of the target"),
                ));
            }
            Some("external") => {
                return Err(Refusal::not_yet(Some(rhs), format!("external in `{name}`")));
            }
            Some("load_data") => {
                if matches!(
                    self.m.type_of(rhs),
                    Some(Type::Record(_) | Type::Table { .. })
                ) {
                    return Err(Refusal::not_yet(
                        Some(rhs),
                        format!("the whole data record `{name}`"),
                    ));
                }
                let stan = self.fresh(name);
                let t = self.declare(rhs, self.m.valueset_of(rhs), true)?;
                self.data.push(format!("{};", t.render(&stan)));
                self.record_decl(&stan, t);
                self.deferred.push(DataSource {
                    stan: stan.clone(),
                    binding: name.to_string(),
                    path: vec![],
                });
                return Ok(stan);
            }
            _ => {}
        }
        if matches!(
            self.m.type_of(rhs),
            Some(
                Type::Measure { .. }
                    | Type::Kernel { .. }
                    | Type::Likelihood { .. }
                    | Type::Function { .. }
            )
        ) {
            return Err(Refusal::not_yet(
                Some(rhs),
                format!("`{name}` used as a value"),
            ));
        }
        let phase = self.m.phase_of(rhs).unwrap_or(Phase::Stochastic);
        if phase == Phase::Fixed
            && let Some(v) = data::const_json(self.m, rhs)
        {
            let stan = self.fresh(name);
            let mut t = self.declare(rhs, None, false)?;
            if t.is_int() && t.rank() == 0 && self.sizes.values().any(|v| v.contains(&sym)) {
                t.bounds.lower = Some("0".into());
            }
            self.data.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.json.insert(stan.clone(), v);
            return Ok(stan);
        }
        self.computed(name, rhs)
    }

    /// A computed value under a fresh name: `transformed data` when it is
    /// fixed, else a `model` local (see [`Placement`]). A value Stan has no
    /// whole-array expression for is filled elementwise by loops.
    fn computed(&mut self, base: &str, rhs: NodeId) -> Result<String> {
        let attempt = self.expr(rhs);
        self.place(base, rhs, attempt)
    }

    /// Place a value whose whole-array expression is `attempt`, falling back
    /// to elementwise loops.
    fn place(&mut self, base: &str, rhs: NodeId, attempt: Result<Ex>) -> Result<String> {
        let fixed = self.m.phase_of(rhs) == Some(Phase::Fixed);
        let lines = match attempt {
            Ok(e) => {
                let stan = self.fresh(base);
                let t = self.declare(rhs, None, false)?;
                let line = format!("{} = {};", t.render(&stan), e.s);
                self.record_decl(&stan, t);
                (stan, vec![line])
            }
            Err(r) if r.kind == RefusalKind::NotYetSupported && self.elementwise(rhs) => {
                let stan = self.fresh(base);
                let t = self.declare(rhs, None, false)?;
                let lines = self.fill_loops(&stan, &t, rhs)?;
                self.record_decl(&stan, t);
                (stan, lines)
            }
            Err(r) => return Err(r),
        };
        let (stan, lines) = lines;
        if fixed {
            self.tdata.extend(lines);
        } else {
            match self.opts.derived {
                Placement::ModelLocals => self.locals.extend(lines),
                Placement::TransformedParameters => self.tparams.extend(lines),
            }
        }
        Ok(stan)
    }

    fn latent(&self, sym: Symbol, rhs: NodeId) -> Refusal {
        let name = self.name(sym);
        let discrete = self
            .m
            .type_of(rhs)
            .and_then(shape_of)
            .is_some_and(|s| s.int);
        if discrete {
            Refusal::new(
                RefusalKind::NotYetSupported,
                Some(rhs),
                format!("draw of `{name}`"),
                format!(
                    "`{name}` is a discrete draw the target marginalizes: it is neither a prior \
                     field nor a kernel output; enumerating it is not supported yet"
                ),
            )
        } else {
            Refusal::new(
                RefusalKind::LatentMarginalized,
                Some(rhs),
                format!("draw of `{name}`"),
                format!(
                    "`{name}` is a draw the target marginalizes: it is neither a prior field nor a \
                     kernel output. Stan cannot integrate out a continuous latent. Add `{name}` to \
                     the prior record to sample it"
                ),
            )
        }
    }

    /// A data leaf: one field of a fixed record binding.
    fn leaf(&mut self, rec: Symbol, fname: &str, node: NodeId) -> Result<String> {
        if let Some(s) = self.leaves.get(&(rec, fname.to_string())) {
            return Ok(s.clone());
        }
        let bid = self.m.binding_by_name(rec).expect("checked by caller");
        let rhs = self.m.binding(bid).rhs;
        let stan = if self.used.contains(&sanitize(fname)) {
            self.fresh(&format!("{}_{fname}", self.name(rec)))
        } else {
            self.fresh(fname)
        };
        let vs = self.m.valueset_of(node).cloned();
        let t = self.declare(node, vs.as_ref(), true)?;
        self.data.push(format!("{};", t.render(&stan)));
        self.record_decl(&stan, t);
        match self.head(rhs) {
            Some("load_data") => self.deferred.push(DataSource {
                stan: stan.clone(),
                binding: self.name(rec).to_string(),
                path: vec![fname.to_string()],
            }),
            _ => {
                let v = data::field_json(self.m, rhs, fname).ok_or_else(|| {
                    Refusal::not_yet(
                        Some(node),
                        format!("the computed field `{}.{fname}`", self.name(rec)),
                    )
                })?;
                self.json.insert(stan.clone(), v);
            }
        }
        self.leaves.insert((rec, fname.to_string()), stan.clone());
        Ok(stan)
    }

    /// A fixed binding whose fields are data leaves.
    fn data_record(&self, id: NodeId) -> Option<Symbol> {
        let sym = self.self_ref(id)?;
        let rhs = self.m.binding(self.m.alias_target(sym)?).rhs;
        let sym = *self.m.alias_chain(sym).last()?;
        let fixed = self.m.phase_of(rhs) == Some(Phase::Fixed);
        match self.head(rhs) {
            Some("load_data") => Some(sym),
            Some("record") if fixed && data::const_json_record(self.m, rhs) => Some(sym),
            _ => None,
        }
    }

    // ----------------------------------------------------------- expressions

    fn shape(&self, id: NodeId) -> Option<Shape> {
        self.m.type_of(id).and_then(shape_of)
    }

    /// An expression used in real arithmetic: integer arrays become vectors.
    fn real(&mut self, id: NodeId) -> Result<Ex> {
        let e = self.expr(id)?;
        match self.shape(id) {
            Some(Shape { rank: 1, int: true }) => Ok(Ex::atom(format!("to_vector({})", e.s))),
            _ => Ok(e),
        }
    }

    fn expr(&mut self, id: NodeId) -> Result<Ex> {
        match self.m.node(id) {
            Node::Lit(Scalar::Int(n)) => Ok(Ex::atom(n.to_string()).neg_prec(*n < 0)),
            Node::Lit(Scalar::Real(x)) => Ok(Ex::atom(real_literal(*x)).neg_prec(*x < 0.0)),
            Node::Lit(Scalar::Bool(b)) => Ok(Ex::atom(if *b { "1" } else { "0" })),
            Node::Lit(Scalar::Str(s)) => {
                Err(Refusal::not_yet(Some(id), format!("the string \"{s}\"")))
            }
            Node::Const(c) => match self.name(*c) {
                "inf" => Ok(Ex::atom("positive_infinity()")),
                "pi" => Ok(Ex::atom("pi()")),
                other => Err(Refusal::not_yet(
                    Some(id),
                    format!("the constant `{other}`"),
                )),
            },
            Node::Ref(r) => match r.ns {
                RefNs::SelfMod => Ok(Ex::atom(self.bind(r.name)?)),
                RefNs::Local => Err(Refusal::not_yet(Some(id), "a function body")),
                RefNs::Module(alias) => Err(Refusal::not_yet(
                    Some(id),
                    format!("a reference into module `{}`", self.name(alias)),
                )),
            },
            Node::Call(_) => self.call_expr(id),
            _ => Err(Refusal::not_yet(Some(id), "this expression form")),
        }
    }

    fn call_expr(&mut self, id: NodeId) -> Result<Ex> {
        let c = self.call(id).unwrap();
        let head = self.head(id).unwrap();
        let args: Vec<NodeId> = c.args.to_vec();
        let bin = |me: &mut Self, op: &str, prec: u8, a: NodeId, b: NodeId| -> Result<Ex> {
            let l = me.real(a)?;
            let r = me.real(b)?;
            Ok(Ex {
                s: format!("{} {op} {}", l.at(prec), r.at(prec + 1)),
                prec,
            })
        };
        match (head, args.as_slice()) {
            ("add", [a, b]) => bin(self, "+", P_ADD, *a, *b),
            ("sub", [a, b]) => bin(self, "-", P_ADD, *a, *b),
            ("mul", [a, b]) => bin(self, "*", P_MUL, *a, *b),
            ("divide", [a, b]) => {
                let ints =
                    self.shape(*a).is_some_and(|s| s.int) && self.shape(*b).is_some_and(|s| s.int);
                let l = self.real(*a)?;
                let r = self.real(*b)?;
                let l = if ints {
                    format!("1.0 * {}", l.at(P_MUL))
                } else {
                    l.at(P_MUL)
                };
                Ok(Ex {
                    s: format!("{l} / {}", r.at(P_MUL + 1)),
                    prec: P_MUL,
                })
            }
            ("neg", [a]) => {
                let e = self.real(*a)?;
                Ok(Ex {
                    s: format!("-{}", e.at(P_UNARY)),
                    prec: P_UNARY,
                })
            }
            ("pow", [a, b]) => {
                let l = self.real(*a)?;
                let r = self.real(*b)?;
                Ok(Ex {
                    s: format!("{} ^ {}", l.at(P_POW + 1), r.at(P_POW)),
                    prec: P_POW,
                })
            }
            ("equal", [a, b]) => self.cmp("==", *a, *b),
            ("unequal", [a, b]) => self.cmp("!=", *a, *b),
            ("lt", [a, b]) => self.cmp("<", *a, *b),
            ("le", [a, b]) => self.cmp("<=", *a, *b),
            ("gt", [a, b]) => self.cmp(">", *a, *b),
            ("ge", [a, b]) => self.cmp(">=", *a, *b),
            ("ifelse", [c0, a, b]) => {
                let c0 = self.expr(*c0)?;
                let a = self.expr(*a)?;
                let b = self.expr(*b)?;
                Ok(Ex {
                    s: format!(
                        "{} ? {} : {}",
                        c0.at(P_CMP),
                        a.at(P_TERNARY + 1),
                        b.at(P_TERNARY)
                    ),
                    prec: P_TERNARY,
                })
            }
            ("broadcast", [f, rest @ ..]) => self.broadcast(id, *f, rest),
            ("get", [x, sel @ ..]) => self.get(id, *x, sel),
            ("vector", elems) => self.vector(id, elems),
            ("ones" | "zeros", [n])
                if self.shape(id)
                    == Some(Shape {
                        rank: 1,
                        int: false,
                    }) =>
            {
                let size = self.expr(*n)?;
                self.table_fn(id, head, None, vec![size])
            }
            ("ones", [n]) => self.filled(id, "1", *n),
            ("zeros", [n]) => self.filled(id, "0", *n),
            ("fill", [x, n]) => {
                let v = self.expr(*x)?;
                self.filled(id, &v.at(P_ATOM), *n)
            }
            ("colstack", [cols]) => self.stack(id, *cols, "append_col", false),
            ("rowstack", [rows]) => self.stack(id, *rows, "append_row", true),
            ("%call", _) => self.module_call(id),
            _ => {
                if !c.named.is_empty() {
                    return Err(Refusal::not_yet(
                        Some(id),
                        format!("{head} with keyword arguments"),
                    ));
                }
                let mut parts = Vec::new();
                for a in &args {
                    parts.push(self.real(*a)?);
                }
                self.table_fn(id, head, None, parts)
            }
        }
    }

    /// A function from `functions.ron` applied to printed arguments.
    fn table_fn(
        &mut self,
        id: NodeId,
        name: &str,
        module: Option<&str>,
        parts: Vec<Ex>,
    ) -> Result<Ex> {
        let row = tables::function(name, module)
            .filter(|r| matches!(r.kind.as_str(), "builtin" | "helper"))
            .filter(|r| r.args.len() == parts.len() && !r.args.iter().any(|a| a == "..."))
            .ok_or_else(|| Refusal::not_yet(Some(id), name.to_string()))?;
        if row.kind == "helper" && !self.functions.contains(&row.functions) {
            self.functions.push(row.functions.clone());
        }
        let binds = row
            .args
            .iter()
            .cloned()
            .zip(parts.into_iter().map(pattern::Bound::Expr))
            .collect();
        let s = self.fill(&row.stan, &binds, &|_| false, None)?;
        let prec = if is_call(&s) { P_ATOM } else { P_TERNARY };
        Ok(Ex { s, prec })
    }

    /// A call into a standard module: refused until the `functions` block lands,
    /// and refused for good where Stan has no counterpart.
    fn module_call(&mut self, id: NodeId) -> Result<Ex> {
        let c = self.call(id).unwrap();
        let CallHead::User(callee) = c.head else {
            unreachable!()
        };
        let member = match self.m.node(callee) {
            Node::Ref(r) => match r.ns {
                RefNs::Module(alias) => format!("{}.{}", self.name(alias), self.name(r.name)),
                _ => self.name(r.name).to_string(),
            },
            _ => "a callable expression".to_string(),
        };
        let (Node::Ref(r), Some(module)) = (self.m.node(callee), self.standard_module(callee))
        else {
            return Err(Refusal::not_yet(
                Some(id),
                format!("the call `{member}(…)`"),
            ));
        };
        let fname = self.name(r.name);
        if !c.named.is_empty() {
            return Err(Refusal::not_yet(
                Some(id),
                format!("`{member}` with keyword arguments"),
            ));
        }
        let args = c.args.to_vec();
        // polygamma(n, x): Stan has the orders 0 and 1 only.
        if fname == "polygamma" && args.len() == 2 {
            let order = match self.m.node(self.resolve(args[0])) {
                Node::Lit(Scalar::Int(n)) => Some(*n),
                _ => None,
            };
            match order {
                Some(0) => {
                    let x = self.real(args[1])?;
                    return self.table_fn(id, "digamma", Some(&module), vec![x]);
                }
                Some(1) => {}
                Some(_) => {
                    return Err(Refusal::new(
                        RefusalKind::NoStanFunction,
                        Some(id),
                        member.clone(),
                        format!(
                            "`{member}` of order 2 or more has no Stan function; Stan has \
                             `digamma` and `trigamma` (orders 0 and 1) only"
                        ),
                    ));
                }
                None => {
                    return Err(Refusal::not_yet(
                        Some(id),
                        format!("`{member}` of a computed order"),
                    ));
                }
            }
        }
        let mut parts = Vec::new();
        for a in &args {
            parts.push(self.real(*a)?);
        }
        self.table_fn(id, fname, Some(&module), parts)
            .map_err(|_| Refusal::not_yet(Some(id), format!("the call `{member}(…)`")))
    }

    /// The standard-module name a callee reference resolves into.
    fn standard_module(&self, callee: NodeId) -> Option<String> {
        let Node::Ref(r) = self.m.node(callee) else {
            return None;
        };
        let RefNs::Module(alias) = r.ns else {
            return None;
        };
        let rhs = self.m.binding(self.m.binding_by_name(alias)?).rhs;
        if self.head(rhs) != Some("standard_module") {
            return None;
        }
        match self.m.node(*self.call(rhs)?.args.first()?) {
            Node::Lit(Scalar::Str(s)) => Some(s.to_string()),
            _ => None,
        }
    }

    fn cmp(&mut self, op: &str, a: NodeId, b: NodeId) -> Result<Ex> {
        let l = self.expr(a)?;
        let r = self.expr(b)?;
        Ok(Ex {
            s: format!("{} {op} {}", l.at(P_CMP + 1), r.at(P_CMP + 1)),
            prec: P_CMP,
        })
    }

    /// `ones`, `zeros` and `fill`, by the result's rank and element type.
    fn filled(&mut self, id: NodeId, value: &str, n: NodeId) -> Result<Ex> {
        let sh = self
            .shape(id)
            .ok_or_else(|| Refusal::not_yet(Some(id), "an untyped array"))?;
        let r = self.resolve(n);
        let extents: Vec<String> = if self.head(r) == Some("vector") {
            let items = self.call(r).unwrap().args.to_vec();
            items
                .iter()
                .map(|&i| self.expr(i).map(|e| e.s))
                .collect::<Result<_>>()?
        } else {
            vec![self.expr(n)?.s]
        };
        let value = if sh.int || value.contains('.') || value.contains('(') {
            value.to_string()
        } else {
            format!("{value}.0")
        };
        let f = match (sh.rank, sh.int) {
            (1, false) => "rep_vector",
            (2, false) => "rep_matrix",
            (_, true) => "rep_array",
            _ => return Err(Refusal::not_yet(Some(id), "a filled array of this rank")),
        };
        Ok(Ex::atom(format!("{f}({value}, {})", extents.join(", "))))
    }

    fn stack(&mut self, id: NodeId, list: NodeId, join: &str, transpose: bool) -> Result<Ex> {
        let original = list;
        let list = self.resolve(list);
        if self.head(list) != Some("vector") {
            // An array of vectors: copy it into a matrix by loops.
            let base = match self.self_ref(original) {
                Some(s) => format!("{}_matrix", self.name(s)),
                None => "stacked".to_string(),
            };
            let refused = Err(Refusal::not_yet(Some(id), "a stacked array"));
            return Ok(Ex::atom(self.place(&base, id, refused)?));
        }
        let items = self.call(list).unwrap().args.to_vec();
        let mut parts = Vec::new();
        for item in items {
            let e = self.real(item)?;
            parts.push(if transpose {
                format!("{}'", e.at(P_ATOM))
            } else {
                e.s
            });
        }
        let mut acc = if parts.len() == 1 {
            if transpose {
                parts[0].clone()
            } else {
                format!("to_matrix({})", parts[0])
            }
        } else {
            parts[0].clone()
        };
        for p in &parts[1..] {
            acc = format!("{join}({acc}, {p})");
        }
        Ok(Ex::atom(acc))
    }

    fn vector(&mut self, id: NodeId, elems: &[NodeId]) -> Result<Ex> {
        let sh = self
            .shape(id)
            .ok_or_else(|| Refusal::not_yet(Some(id), "an untyped array"))?;
        let mut parts = Vec::new();
        for e in elems {
            parts.push(self.expr(*e)?.s);
        }
        if sh.rank == 1 && !sh.int {
            Ok(Ex::atom(format!("[{}]'", parts.join(", "))))
        } else {
            Ok(Ex::atom(format!("{{{}}}", parts.join(", "))))
        }
    }

    fn get(&mut self, id: NodeId, x: NodeId, sel: &[NodeId]) -> Result<Ex> {
        // A field of a data record.
        if let [field] = sel
            && let Node::Lit(Scalar::Str(f)) = self.m.node(*field)
        {
            if let Some(rec) = self.data_record(x) {
                return Ok(Ex::atom(self.leaf(rec, f, id)?));
            }
            let r = self.resolve(x);
            if self.head(r) == Some("record")
                && let Some(na) = self
                    .call(r)
                    .unwrap()
                    .named
                    .iter()
                    .find(|na| self.name(na.name) == &**f)
            {
                return self.expr(na.value);
            }
            return Err(Refusal::not_yet(
                Some(id),
                format!("the field access `.{f}`"),
            ));
        }
        let base = self.expr(x)?;
        let mut idx = Vec::new();
        for s in sel {
            match self.m.node(*s) {
                Node::Const(c) if self.name(*c) == "all" => idx.push(":".to_string()),
                _ => idx.push(self.expr(*s)?.s),
            }
        }
        let s = format!("{}[{}]", base.at(P_ATOM), idx.join(", "));
        // A matrix row is a Stan row_vector; a FlatPPL vector is a Stan vector.
        let matrix = matches!(self.m.type_of(x), Some(Type::Array { shape, elem })
            if shape.len() == 2 && matches!(elem.as_ref(), Type::Scalar(ScalarType::Real)));
        let row = sel.len() == 2
            && self.shape(sel[0]).is_some_and(|s| s.rank == 0)
            && self.shape(id).is_some_and(|s| s.rank == 1);
        if matrix && row {
            return Ok(Ex::atom(format!("{s}'")));
        }
        Ok(Ex::atom(s))
    }

    fn broadcast(&mut self, id: NodeId, f: NodeId, args: &[NodeId]) -> Result<Ex> {
        let Node::Const(fs) = self.m.node(f) else {
            return Err(Refusal::not_yet(Some(id), "broadcast of a user function"));
        };
        let fname = self.name(*fs);
        let scalar = |me: &Self, a: NodeId| me.shape(a).is_some_and(|s| s.rank == 0);
        match (fname, args) {
            ("add", [a, b]) => self.call_expr_bin("+", P_ADD, *a, *b),
            ("sub", [a, b]) => self.call_expr_bin("-", P_ADD, *a, *b),
            ("mul", [a, b]) => {
                let op = if scalar(self, *a) || scalar(self, *b) {
                    "*"
                } else {
                    ".*"
                };
                self.call_expr_bin(op, P_MUL, *a, *b)
            }
            ("divide", [a, b]) => {
                let op = if scalar(self, *b) { "/" } else { "./" };
                self.call_expr_bin(op, P_MUL, *a, *b)
            }
            ("pow", [a, b]) => {
                let op = if scalar(self, *a) && scalar(self, *b) {
                    "^"
                } else {
                    ".^"
                };
                let l = self.real(*a)?;
                let r = self.real(*b)?;
                Ok(Ex {
                    s: format!("{} {op} {}", l.at(P_POW + 1), r.at(P_POW)),
                    prec: P_POW,
                })
            }
            ("neg", [a]) => {
                let e = self.real(*a)?;
                Ok(Ex {
                    s: format!("-{}", e.at(P_UNARY)),
                    prec: P_UNARY,
                })
            }
            (name, [a]) if tables::ELEMENTWISE.contains(&name) => {
                let e = self.real(*a)?;
                self.table_fn(id, name, None, vec![e])
            }
            (name, _) => Err(Refusal::not_yet(Some(id), format!("{name}.(…)"))),
        }
    }

    fn call_expr_bin(&mut self, op: &str, prec: u8, a: NodeId, b: NodeId) -> Result<Ex> {
        let l = self.real(a)?;
        let r = self.real(b)?;
        Ok(Ex {
            s: format!("{} {op} {}", l.at(prec), r.at(prec + 1)),
            prec,
        })
    }
}

/// `true` when `s` is one call `f(...)` (so it needs no parentheses).
fn is_call(s: &str) -> bool {
    let Some(open) = s.find('(') else {
        return false;
    };
    if !s[..open]
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
        || open == 0
    {
        return false;
    }
    let mut depth = 0;
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

fn real_literal(x: f64) -> String {
    if x.is_infinite() {
        return if x > 0.0 {
            "positive_infinity()".into()
        } else {
            "negative_infinity()".into()
        };
    }
    format!("{x:?}")
}

impl Ex {
    fn neg_prec(mut self, negative: bool) -> Ex {
        if negative {
            self.prec = P_UNARY;
        }
        self
    }
}

/// The target binding: the named one, or the module's single `bayesupdate`.
fn select_target(m: &Module, opts: &EmitOptions) -> Result<Symbol> {
    if let Some(name) = &opts.target {
        return m
            .bindings()
            .find(|(_, b)| m.resolve(b.name) == name)
            .map(|(_, b)| b.name)
            .ok_or_else(|| {
                Refusal::new(
                    RefusalKind::NoTarget,
                    None,
                    name.clone(),
                    format!("the module has no binding `{name}`"),
                )
            });
    }
    let posts: Vec<Symbol> = m
        .bindings()
        .filter(|(_, b)| matches!(m.node(b.rhs), Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if m.resolve(h) == "bayesupdate")))
        .map(|(_, b)| b.name)
        .collect();
    match posts.as_slice() {
        [one] => Ok(*one),
        [] => Err(Refusal::new(
            RefusalKind::NoTarget,
            None,
            "the target",
            "the module has no `bayesupdate`; pass `--target NAME` naming a posterior or likelihood",
        )),
        many => {
            let names: Vec<&str> = many.iter().map(|s| m.resolve(*s)).collect();
            Err(Refusal::new(
                RefusalKind::AmbiguousTarget,
                None,
                "the target",
                format!(
                    "{} posterior bindings ({}); pass `--target NAME`",
                    many.len(),
                    names.join(", ")
                ),
            ))
        }
    }
}

fn block(out: &mut String, name: &str, lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    out.push_str(name);
    out.push_str(" {\n");
    for l in lines {
        if l.is_empty() {
            out.push('\n');
        }
        for part in l.lines() {
            out.push_str("  ");
            out.push_str(part);
            out.push('\n');
        }
    }
    out.push_str("}\n");
}

pub(crate) fn program(m: &Module, opts: &EmitOptions) -> Result<crate::StanProgram> {
    let target = select_target(m, opts)?;
    let mut e = Emitter::new(m, opts);
    e.emit_target(target)?;
    e.query_inputs(target);
    let mut model = e.locals.clone();
    if !model.is_empty() && !(e.prior.is_empty() && e.lik.is_empty()) {
        model.push(String::new());
    }
    model.extend(e.prior.iter().cloned());
    model.extend(e.lik.iter().cloned());
    let mut code = String::new();
    block(&mut code, "functions", &e.functions);
    block(&mut code, "data", &e.data);
    block(&mut code, "transformed data", &e.tdata);
    block(&mut code, "parameters", &e.params);
    block(&mut code, "transformed parameters", &e.tparams);
    block(&mut code, "model", &model);
    Ok(crate::StanProgram {
        code,
        data: e.json,
        deferred_data: e.deferred,
        manifest: crate::Manifest {
            target: m.resolve(target).to_string(),
            parameters: e.manifest,
        },
    })
}
