//! Extent names from their symbolic source.
//!
//! An axis is named by the expression that produces it: the declared set
//! (`cartpow(reals, N)`), an `iid` count, the length argument of `zeros(T - 1)`,
//! or the matching axis of the operand an operation takes it from, through a
//! user function's body too. A value never picks a name. With no source, an
//! extent is a fresh data size for a `data` declaration and its number
//! elsewhere. So a program stays correct for data of another shape.

use super::Emitter;
use flatppl_core::{Node, NodeId, Phase, RefNs, Scalar, Symbol, Type};
use std::collections::{HashMap, HashSet};

/// Builtins whose result has the shape of an argument of its rank.
const SHAPE_PRESERVING: &[&str] = &[
    "ifelse",
    "lower_cholesky",
    "cholesky_decompose",
    "inv",
    "softmax",
    "log_softmax",
    "cumsum",
    "reverse",
    "sort",
];

/// Marks a size variable inside emitted text until it is named.
const TOK: char = '\u{3}';

fn token(k: usize) -> String {
    format!("{TOK}{k}{TOK}")
}

/// The size variable a form is, when it is exactly one.
fn as_var(l: &Lin) -> Option<usize> {
    match (l.konst, l.terms.as_slice()) {
        (0, [(t, 1)]) => t.strip_prefix(TOK)?.strip_suffix(TOK)?.parse().ok(),
        _ => None,
    }
}

fn has_token(l: &Lin) -> bool {
    l.terms.iter().any(|(t, _)| t.contains(TOK))
}

/// One unsourced size of a data value.
pub(crate) struct SizeVar {
    owner: String,
    axis: usize,
    naxes: usize,
    n: u32,
    parent: usize,
    bound: Option<Lin>,
}

/// An integer `konst + Σ coef · term`: sizes add and cancel symbolically.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Lin {
    pub konst: i64,
    pub terms: Vec<(String, i64)>,
}

impl Lin {
    pub fn konst(k: i64) -> Lin {
        Lin {
            konst: k,
            terms: vec![],
        }
    }

    pub fn term(s: String) -> Lin {
        Lin {
            konst: 0,
            terms: vec![(s, 1)],
        }
    }

    pub fn plus(&self, o: &Lin) -> Lin {
        let mut terms = self.terms.clone();
        for (s, c) in &o.terms {
            match terms.iter_mut().find(|(t, _)| t == s) {
                Some(e) => e.1 += c,
                None => terms.push((s.clone(), *c)),
            }
        }
        terms.retain(|(_, c)| *c != 0);
        Lin {
            konst: self.konst + o.konst,
            terms,
        }
    }

    pub fn scale(&self, k: i64) -> Lin {
        let mut terms: Vec<_> = self.terms.iter().map(|(s, c)| (s.clone(), c * k)).collect();
        terms.retain(|(_, c)| *c != 0);
        Lin {
            konst: self.konst * k,
            terms,
        }
    }

    pub fn add(&self, k: i64) -> Lin {
        self.plus(&Lin::konst(k))
    }

    pub fn as_const(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.konst)
    }

    /// `true` when it prints as one atom.
    pub fn atomic(&self) -> bool {
        match (self.terms.as_slice(), self.konst) {
            ([], k) => k >= 0,
            ([(_, 1)], 0) => true,
            _ => false,
        }
    }

    pub fn print(&self) -> String {
        let mut out = String::new();
        for (s, c) in &self.terms {
            let first = out.is_empty();
            let (sign, mag) = if *c < 0 { ("-", -c) } else { ("+", *c) };
            let body = if mag == 1 {
                s.clone()
            } else {
                format!("{mag} * {s}")
            };
            if first {
                out = if sign == "-" {
                    format!("-{body}")
                } else {
                    body
                };
            } else {
                out.push_str(&format!(" {sign} {body}"));
            }
        }
        match (out.is_empty(), self.konst) {
            (true, k) => k.to_string(),
            (false, 0) => out,
            (false, k) if k > 0 => format!("{out} + {k}"),
            (false, k) => format!("{out} - {}", -k),
        }
    }
}

impl<'m> Emitter<'m> {
    /// The name of axis `axis` of `naxes` (extent `n`) of the value `node`
    /// declared as `owner`, with `hint` the size expression a set or law gives
    /// that axis.
    pub(super) fn axis_name(
        &mut self,
        n: u32,
        hint: Option<NodeId>,
        node: NodeId,
        axis: usize,
        naxes: usize,
        owner: &str,
    ) -> Lin {
        // Only this value's own unsourced extents are data sizes.
        let data = std::mem::replace(&mut self.data_sizes, false);
        let written = hint.and_then(|h| self.lin_int(h));
        let found = match written {
            Some(l) => Some(l),
            // A literal array's element count is no source for data.
            None => self
                .sym_extent(node, axis, 0)
                .filter(|l| !(data && l.as_const().is_some())),
        };
        self.data_sizes = data;
        // A dynamic extent with no symbolic source stays unnamed (refused).
        if n == crate::stan_type::DYNAMIC && found.is_none() {
            return Lin::konst(i64::from(n));
        }
        // A function sees no unnamed data size: its body takes the number.
        if self.fctx.is_some() && found.as_ref().is_some_and(has_token) {
            return Lin::konst(i64::from(n));
        }
        if let Some(l) = found {
            return l;
        }
        if !data {
            return Lin::konst(i64::from(n));
        }
        // A data value of its own shape: a size variable, unified with the
        // sizes the model requires equal and named at the end.
        let k = self.svars.len();
        self.svars.push(SizeVar {
            owner: owner.to_string(),
            axis,
            naxes,
            n,
            parent: k,
            bound: None,
        });
        Lin::term(token(k))
    }

    fn find(&mut self, k: usize) -> usize {
        let p = self.svars[k].parent;
        if p == k {
            return k;
        }
        let r = self.find(p);
        self.svars[k].parent = r;
        r
    }

    /// Record that two sizes are equal. Size variables merge, and a variable
    /// takes a sourced size. Two different numbers never merge: that is a
    /// model error, left as it is.
    pub(super) fn unify(&mut self, a: &Lin, b: &Lin) {
        match (as_var(a), as_var(b)) {
            (Some(x), Some(y)) => {
                let (rx, ry) = (self.find(x), self.find(y));
                if rx == ry || self.svars[rx].n != self.svars[ry].n {
                    return;
                }
                let (bx, by) = (self.svars[rx].bound.clone(), self.svars[ry].bound.clone());
                if let (Some(p), Some(q)) = (&bx, &by)
                    && p != q
                {
                    return;
                }
                // The earlier variable represents the class.
                let (root, child) = if rx < ry { (rx, ry) } else { (ry, rx) };
                self.svars[child].parent = root;
                self.svars[root].bound = bx.or(by);
            }
            (Some(x), None) | (None, Some(x)) => {
                let l = if as_var(a).is_some() { b } else { a };
                if has_token(l) {
                    return;
                }
                let r = self.find(x);
                if l.as_const()
                    .is_some_and(|c| c != i64::from(self.svars[r].n))
                {
                    return;
                }
                if self.svars[r].bound.is_none() {
                    self.svars[r].bound = Some(l.clone());
                }
            }
            _ => {}
        }
    }

    /// The size of axis `axis` of `id` as Stan text, as a declaration names
    /// it ([`Self::axis_name`]): the symbolic size, or the number `n` where
    /// it has none, or where a body would need an unnamed data size.
    pub(super) fn size_text(&mut self, id: NodeId, axis: usize, n: u32) -> String {
        match self.sym_extent(id, axis, 0) {
            Some(l) if !(self.fctx.is_some() && has_token(&l)) => l.print(),
            _ => n.to_string(),
        }
    }

    /// `true` when two sizes are the same, recording it where one is an
    /// unsourced data size (as an `iid` count records its variate's size).
    /// Two different sourced sizes are never the same.
    pub(super) fn same_size(&mut self, a: &Lin, b: &Lin) -> bool {
        if a == b {
            return true;
        }
        match (as_var(a), as_var(b)) {
            (Some(x), Some(y)) => {
                self.unify(a, b);
                self.find(x) == self.find(y)
            }
            (Some(x), None) | (None, Some(x)) => {
                let other = if as_var(a).is_some() { b } else { a };
                self.unify(a, b);
                let r = self.find(x);
                self.svars[r].bound.as_ref() == Some(other)
            }
            _ => false,
        }
    }

    /// Name each size class and write the names into the program. A class
    /// that took a sourced size prints it; any other is one `data` size, `N`
    /// first, then `N_<first member>`.
    pub(super) fn resolve_sizes(&mut self) {
        let mut text: HashMap<usize, String> = HashMap::new();
        let mut decls: Vec<(usize, String)> = Vec::new();
        // A data size declares a `data` value, so it takes a sourced size
        // only when the `data` block can read it (a `filter` count is
        // computed in `transformed data`). Otherwise it keeps its own name.
        let data_names: HashSet<String> = self
            .data
            .iter()
            .filter_map(|l| l.trim_end_matches(';').rsplit(' ').next())
            .map(str::to_string)
            .collect();
        let readable = |l: &Lin| {
            super::functions::identifiers(&l.print())
                .iter()
                .all(|n| data_names.contains(n))
        };
        for k in 0..self.svars.len() {
            let r = self.find(k);
            if text.contains_key(&r) {
                let t = text[&r].clone();
                text.insert(k, t);
                continue;
            }
            let t = match self.svars[r].bound.clone() {
                Some(l) if readable(&l) => l.print(),
                _ => {
                    let v = &self.svars[r];
                    let base = if self.used.contains("N") {
                        if v.naxes > 1 {
                            format!("N_{}_{}", v.owner, v.axis + 1)
                        } else {
                            format!("N_{}", v.owner)
                        }
                    } else {
                        "N".to_string()
                    };
                    let name = self.fresh(&base);
                    decls.push((r, name.clone()));
                    name
                }
            };
            text.insert(r, t.clone());
            text.insert(k, t);
        }
        let replace = |s: &str| -> String {
            let mut out = s.to_string();
            for (k, t) in &text {
                let tok = token(*k);
                if !out.contains(&tok) {
                    continue;
                }
                // A whole extent prints bare; inside arithmetic, a compound
                // size is parenthesized.
                let compound = t.contains(' ');
                for (before, after) in [("[", "]"), ("[", ","), (", ", "]"), (", ", ",")] {
                    out = out.replace(
                        &format!("{before}{tok}{after}"),
                        &format!("{before}{t}{after}"),
                    );
                }
                let wrapped = if compound {
                    format!("({t})")
                } else {
                    t.clone()
                };
                out = out.replace(&tok, &wrapped);
            }
            out
        };
        for block in [
            &mut self.data,
            &mut self.tdata,
            &mut self.params,
            &mut self.tparams,
            &mut self.locals,
            &mut self.prior,
            &mut self.lik,
            &mut self.functions,
        ] {
            for line in block.iter_mut() {
                if line.contains(TOK) {
                    *line = replace(line);
                }
            }
        }
        // New size declarations lead the data block.
        let mut data: Vec<String> = decls
            .iter()
            .map(|(_, n)| format!("int<lower=0> {n};"))
            .collect();
        for (r, n) in &decls {
            self.json
                .insert(n.clone(), serde_json::Value::from(self.svars[*r].n));
        }
        data.extend(std::mem::take(&mut self.data));
        // A size a class took can be declared after a line that now reads it:
        // move its declaration up to the first such line.
        let mut i = 0;
        while i < data.len() {
            let line = &data[i];
            let name = (line.starts_with("int") && !line.contains('[')).then(|| {
                line.trim_end_matches(';')
                    .rsplit(' ')
                    .next()
                    .unwrap()
                    .to_string()
            });
            if let Some(name) = name
                && let Some(j) = data[..i]
                    .iter()
                    .position(|l| super::functions::identifiers(l).contains(&name))
            {
                let moved = data.remove(i);
                data.insert(j, moved);
            }
            i += 1;
        }
        self.data = data;
    }

    /// A scalar integer expression as a linear form. Literals fold; a module
    /// size stays its name; any other integer value is one opaque term.
    pub(super) fn lin_int(&mut self, id: NodeId) -> Option<Lin> {
        if !self.shape(id).is_some_and(|s| s.rank == 0 && s.int) {
            return None;
        }
        let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
        match self.m.node(id) {
            Node::Lit(Scalar::Int(k)) => return Some(Lin::konst(*k)),
            Node::Ref(r) if r.ns == RefNs::Local => {
                return self.expr(id).ok().map(|e| Lin::term(e.s));
            }
            _ => {}
        }
        if !fixed {
            return None;
        }
        let args: Vec<NodeId> = self.call(id).map(|c| c.args.to_vec()).unwrap_or_default();
        match (self.head(id), args.as_slice()) {
            (Some("add"), [a, b]) => Some(self.lin_int(*a)?.plus(&self.lin_int(*b)?)),
            (Some("sub"), [a, b]) => Some(self.lin_int(*a)?.plus(&self.lin_int(*b)?.scale(-1))),
            (Some("neg"), [a]) => Some(self.lin_int(*a)?.scale(-1)),
            (Some("mul"), [a, b]) => {
                let (x, y) = (self.lin_int(*a)?, self.lin_int(*b)?);
                match (x.as_const(), y.as_const()) {
                    (Some(k), _) => Some(y.scale(k)),
                    (_, Some(k)) => Some(x.scale(k)),
                    _ => self.opaque(id),
                }
            }
            _ => self.opaque(id),
        }
    }

    fn opaque(&mut self, id: NodeId) -> Option<Lin> {
        let e = self.expr(id).ok()?;
        Some(Lin::term(e.at(super::P_ATOM)))
    }

    /// `true` when the module binding `name` reads a module-binding input of
    /// the body being sized, through self references (not through a draw).
    fn reads_substituted(&self, name: Symbol) -> bool {
        match self.esubst_self.last() {
            Some(targets) if !targets.is_empty() => self.binding_reads(name, targets, 0),
            _ => false,
        }
    }

    fn binding_reads(&self, name: Symbol, targets: &HashSet<Symbol>, depth: u32) -> bool {
        if depth > 32 {
            return false;
        }
        let Some(b) = self.m.binding_by_name(name) else {
            return false;
        };
        let rhs = self.m.binding(b).rhs;
        if self.head(rhs) == Some("draw") {
            return false;
        }
        let mut stack = vec![rhs];
        while let Some(id) = stack.pop() {
            if let Node::Ref(r) = self.m.node(id)
                && r.ns == RefNs::SelfMod
                && (targets.contains(&r.name) || self.binding_reads(r.name, targets, depth + 1))
            {
                return true;
            }
            self.m.for_each_child(id, |c| stack.push(c));
        }
        false
    }

    /// The symbolic extent of axis `axis` of the value `id`.
    pub(super) fn sym_extent(&mut self, id: NodeId, axis: usize, depth: u32) -> Option<Lin> {
        if depth > 48 {
            return None;
        }
        let d = depth + 1;
        // An unknown rank (an untyped body) stops the derivation.
        let rank = |me: &Self, x: NodeId| me.arank(x).unwrap_or(usize::MAX);
        if let Node::Ref(r) = self.m.node(id) {
            match r.ns {
                // A module-binding input of a body being sized: its argument.
                RefNs::SelfMod if self.esubst_self.last().is_some_and(|s| s.contains(&r.name)) => {
                    let (arg, offset) = *self.esubst.last()?.get(&r.name)?;
                    let top = self.esubst.pop();
                    let out = self.sym_extent(arg, axis + offset, d);
                    self.esubst.extend(top);
                    return out;
                }
                // Inside a body, a module-binding input has its argument's
                // extents (the body's types), not the module binding's.
                RefNs::SelfMod
                    if self
                        .fctx
                        .as_ref()
                        .is_some_and(|c| c.self_inputs.contains(&r.name)) =>
                {
                    return None;
                }
                RefNs::SelfMod
                    if self
                        .fctx
                        .as_ref()
                        .is_some_and(|c| c.through.contains(&r.name)) =>
                {
                    let rhs = self.m.binding(self.m.binding_by_name(r.name)?).rhs;
                    return self.sym_extent(rhs, axis, d);
                }
                // A binding that reads such an input: sized through its value.
                RefNs::SelfMod if self.reads_substituted(r.name) => {
                    let rhs = self.m.binding(self.m.binding_by_name(r.name)?).rhs;
                    return self.sym_extent(rhs, axis, d);
                }
                RefNs::SelfMod => {}
                RefNs::Local => {
                    // A parameter of a body being sized: its argument.
                    let (arg, offset) = *self.esubst.last()?.get(&r.name)?;
                    let top = self.esubst.pop();
                    let out = self.sym_extent(arg, axis + offset, d);
                    self.esubst.extend(top);
                    return out;
                }
                RefNs::Module(_) => return None,
            }
            let sym = *self.m.alias_chain(r.name).last().unwrap();
            // A fixed value the expression reads is declared before it is
            // sized, so its own data size names the axis.
            if !self.env.contains_key(&sym)
                && let Some(b) = self.m.binding_by_name(sym)
                && self.m.phase_of(self.m.binding(b).rhs) == Some(Phase::Fixed)
            {
                let _ = self.bind(sym);
            }
            if let Some(name) = self.env.get(&sym)
                && let Some(l) = self.ext_lin.get(name).and_then(|e| e.get(axis))
            {
                return Some(l.clone());
            }
            let rhs = self.m.binding(self.m.binding_by_name(sym)?).rhs;
            return self.sym_extent(rhs, axis, d);
        }
        let c = self.call(id)?;
        let args = c.args.to_vec();
        match self.head(id)? {
            "load_data" | "elementof" => {
                let set = if self.head(id) == Some("elementof") {
                    Some(args[0])
                } else {
                    self.load_set(id)
                };
                let h = self.set_hints(set?).get(axis).copied().flatten()?;
                self.lin_int(h)
            }
            "draw" => match self.law_hints(args[0]).get(axis).copied().flatten() {
                Some(h) => self.lin_int(h),
                None => self.law_extent(args[0], axis, d),
            },
            "eye" if axis < 2 => self.lin_int(args[0]),
            "onehot" if axis == 0 => self.lin_int(*args.get(1)?),
            "zeros" | "ones" => {
                let h = self.extent_list(args[0]).get(axis).copied().flatten()?;
                self.lin_int(h)
            }
            "fill" => {
                let ns = self.extent_list(args[1]);
                match ns.get(axis) {
                    Some(h) => self.lin_int((*h)?),
                    None => self.sym_extent(args[0], axis - ns.len(), d),
                }
            }
            "indicesof" if axis == 0 => self.sym_extent(args[0], 0, d),
            "vector" if axis == 0 => Some(Lin::konst(args.len() as i64)),
            "vector" => self.sym_extent(*args.first()?, axis - 1, d),
            "broadcast" => {
                let operands = &args[1..];
                if matches!(self.m.node(args[0]), Node::Const(_)) {
                    return self.carried(id, operands, axis, d);
                }
                let fun = self.user_callee(args[0])?;
                let top = |me: &Self, a: NodeId| match me.ty(a) {
                    Some(Type::Array { shape, .. }) => shape.len(),
                    _ => 0,
                };
                let brank = operands.iter().map(|&a| top(self, a)).max()?;
                if axis < brank {
                    let src = operands.iter().find(|&&a| top(self, a) == brank).copied()?;
                    return self.sym_extent(src, axis, d);
                }
                let subst = operands
                    .iter()
                    .map(|&a| (a, if top(self, a) == brank { brank } else { 0 }))
                    .collect::<Vec<_>>();
                self.body_extent(fun, &subst, axis - brank, d)
            }
            "%call" => {
                let fun = self.is_user_call(id)?;
                let subst = args.iter().map(|&a| (a, 0)).collect::<Vec<_>>();
                self.body_extent(fun, &subst, axis, d)
            }
            "add" | "sub" | "divide" | "neg" | "pow" => self.carried(id, &args, axis, d),
            "addaxes" => {
                let lit = |me: &Self, x: NodeId| match me.m.node(x) {
                    Node::Lit(Scalar::Int(k)) => usize::try_from(*k).ok(),
                    _ => None,
                };
                let lead = lit(self, args[1])?;
                let inner = self.arank(args[0])?;
                if axis < lead || axis >= lead + inner {
                    Some(Lin::konst(1))
                } else {
                    self.sym_extent(args[0], axis - lead, d)
                }
            }
            "mul" => {
                let [a, b] = args[..] else { return None };
                // The inner axes of a product are equal.
                let inner = match (rank(self, a), rank(self, b)) {
                    (2, 1) | (2, 2) => Some((1, 0)),
                    (1, 2) => Some((0, 0)),
                    _ => None,
                };
                if let Some((ia, ib)) = inner
                    && let (Some(x), Some(y)) =
                        (self.sym_extent(a, ia, d), self.sym_extent(b, ib, d))
                {
                    self.unify(&x, &y);
                }
                match (rank(self, a), rank(self, b)) {
                    (0, _) => self.sym_extent(b, axis, d),
                    (_, 0) => self.sym_extent(a, axis, d),
                    (2, 1) => self.sym_extent(a, 0, d),
                    (1, 2) => self.sym_extent(b, 1, d),
                    (2, 2) if axis == 0 => self.sym_extent(a, 0, d),
                    (2, 2) => self.sym_extent(b, 1, d),
                    _ => None,
                }
            }
            "transpose" if rank(self, id) == 2 => self.sym_extent(args[0], 1 - axis, d),
            "rowstack" => match axis {
                0 => self.sym_extent(args[0], 0, d),
                _ => self.stack_inner(args[0], axis, d),
            },
            "colstack" => match axis {
                0 => self.stack_inner(args[0], 1, d),
                _ => self.sym_extent(args[0], 0, d),
            },
            "cat" if axis == 0 => {
                let a = self.sym_extent(args[0], 0, d)?;
                let b = self.sym_extent(args[1], 0, d)?;
                Some(a.plus(&b))
            }
            "scan" if axis == 0 => self.sym_extent(*args.get(2)?, 0, d),
            // A `filter` result: its count, computed in `transformed data`.
            "filter" if axis == 0 => self.filter_counts.get(&id).map(|c| Lin::term(c.clone())),
            // A scalar or vector accumulator: each entry has the init's shape.
            "scan" if !self.record_scan(id) => self.sym_extent(*args.get(1)?, axis - 1, d),
            "get" => self.get_extent(&args, axis, d),
            "aggregate" => self.aggregate_extent(&args, axis, d),
            // A shape-preserving function: the argument of its rank.
            h if SHAPE_PRESERVING.contains(&h)
                || crate::tables::function(h, None).is_some_and(|r| r.vectorized) =>
            {
                self.carried(id, &args, axis, d)
            }
            _ => None,
        }
    }

    /// Axis `axis` of the variate of a law: `iid` counts, then the law's own
    /// length from its vector argument.
    fn law_extent(&mut self, law: NodeId, axis: usize, d: u32) -> Option<Lin> {
        let m = self.resolve(law);
        let c = self.call(m)?;
        let args = c.args.to_vec();
        match self.head(m)? {
            "iid" => {
                let ns = self.extent_list(args[1]);
                match ns.get(axis) {
                    Some(h) => self.lin_int((*h)?),
                    None => self.law_extent(args[0], axis - ns.len(), d),
                }
            }
            "normalize" | "truncate" => self.law_extent(args[0], axis, d),
            "broadcast" => {
                // The variate has the shape of the highest-rank argument.
                let r = args[1..].iter().filter_map(|&a| self.arank(a)).max()?;
                let full: Vec<NodeId> = args[1..]
                    .iter()
                    .copied()
                    .filter(|&a| self.arank(a) == Some(r))
                    .collect();
                self.unified(&full, axis, d)
            }
            "Dirichlet" | "MvNormal" | "Multinomial" | "MvNormalCholesky" if axis == 0 => {
                let a = self.law_args(self.head(m)?, m).ok()?;
                let v = *a.iter().find(|&&x| self.arank(x) == Some(1))?;
                self.sym_extent(v, 0, d)
            }
            _ => None,
        }
    }

    /// The array rank of a value, from its type even when the element type is
    /// unknown (a generic function body).
    fn arank(&self, id: NodeId) -> Option<usize> {
        match self.shape(id) {
            Some(s) => Some(s.rank),
            None => match self.ty(id)? {
                Type::Array { shape, .. } => Some(shape.len()),
                Type::TVector { .. } => Some(1),
                _ => None,
            },
        }
    }

    /// The static extent of axis `axis` of a value, when its type has one.
    fn dim_at(&self, id: NodeId, axis: usize) -> Option<u32> {
        match self.ty(id)? {
            Type::Array { shape, .. } => match shape.get(axis)? {
                flatppl_core::Dim::Static(n) => Some(*n),
                flatppl_core::Dim::Dynamic => None,
            },
            _ => None,
        }
    }

    /// The operand of an elementwise operation that carries axis `axis`: of
    /// full rank, and not a broadcast singleton there.
    fn carrier(&self, id: NodeId, operands: &[NodeId], axis: usize) -> Option<NodeId> {
        let r = self.arank(id)?;
        if operands.iter().any(|&a| self.arank(a).is_none()) {
            return None;
        }
        let full: Vec<NodeId> = operands
            .iter()
            .copied()
            .filter(|&a| self.arank(a) == Some(r))
            .collect();
        full.iter()
            .copied()
            .find(|&a| self.dim_at(a, axis) != Some(1))
            .or_else(|| full.first().copied())
    }

    /// A variate observed or drawn under `m` has the shape the law gives it:
    /// its `iid` counts, or the array arguments of a broadcast law.
    pub(super) fn constrain_variate(&mut self, lhs: &str, m: NodeId) {
        let Some(axes) = self.ext_lin.get(lhs).cloned() else {
            return;
        };
        let m = self.resolve(m);
        let Some(c) = self.call(m) else {
            return;
        };
        let args = c.args.to_vec();
        match self.head(m) {
            Some("iid") => {
                let ns = self.extent_list(args[1]);
                for (a, h) in ns.into_iter().enumerate() {
                    if let (Some(h), Some(l)) = (h, axes.get(a))
                        && let Some(x) = self.lin_int(h)
                    {
                        self.unify(l, &x);
                    }
                }
            }
            Some("broadcast") => {
                for &v in &args[1..] {
                    let Some(r) = self.arank(v) else { continue };
                    for (a, l) in axes.iter().enumerate().take(r) {
                        if self.dim_at(v, a) == Some(1) {
                            continue;
                        }
                        if let Some(x) = self.sym_extent(v, a, 0) {
                            self.unify(l, &x);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Axis `axis` of an elementwise result: its full-rank operands share it.
    fn carried(&mut self, id: NodeId, operands: &[NodeId], axis: usize, d: u32) -> Option<Lin> {
        self.carrier(id, operands, axis)?;
        let r = self.arank(id)?;
        let full: Vec<NodeId> = operands
            .iter()
            .copied()
            .filter(|&a| self.arank(a) == Some(r))
            .collect();
        self.unified(&full, axis, d)
    }

    /// The common extent of axis `axis` of `values`, broadcast singletons
    /// aside, recording that they are equal.
    fn unified(&mut self, values: &[NodeId], axis: usize, d: u32) -> Option<Lin> {
        let mut out: Option<Lin> = None;
        for &v in values {
            if self.dim_at(v, axis) == Some(1) && values.len() > 1 {
                continue;
            }
            if let Some(l) = self.sym_extent(v, axis, d) {
                match &out {
                    Some(o) => {
                        let o = o.clone();
                        self.unify(&o, &l);
                    }
                    None => out = Some(l),
                }
            }
        }
        out
    }

    /// Axis `axis` of a user function's result, its parameters standing for
    /// `args` (each with the axes a broadcast takes off it).
    fn body_extent(
        &mut self,
        fun: NodeId,
        args: &[(NodeId, usize)],
        axis: usize,
        d: u32,
    ) -> Option<Lin> {
        let c = self.call(fun)?;
        let body = c.args[0];
        let entries: Vec<Symbol> = match &c.inputs {
            Some(flatppl_core::Inputs::Spec(e)) => e.iter().map(|(_, r)| r.name).collect(),
            _ => return None,
        };
        if entries.len() != args.len() {
            return None;
        }
        // The body's node types at these arguments.
        let mut tys = Vec::new();
        for &(a, offset) in args {
            let t = match (self.ty(a)?, offset) {
                (t, 0) => t,
                (Type::Array { shape, elem }, k) if shape.len() == k => *elem,
                _ => return None,
            };
            tys.push(t);
        }
        let spec = match &c.inputs {
            Some(flatppl_core::Inputs::Spec(e)) => e.to_vec(),
            _ => return None,
        };
        let types = self.instantiate(fun, body, &spec, &tys).ok()?;
        let selfs: HashSet<Symbol> = spec
            .iter()
            .filter(|(_, r)| r.ns == RefNs::SelfMod)
            .map(|(_, r)| r.name)
            .collect();
        let map: HashMap<Symbol, (NodeId, usize)> =
            entries.into_iter().zip(args.iter().copied()).collect();
        self.esubst.push(map);
        self.esubst_self.push(selfs);
        self.sized_scope.push(types);
        let out = self.sym_extent(body, axis, d);
        self.sized_scope.pop();
        self.esubst_self.pop();
        self.esubst.pop();
        out
    }

    /// Axis `axis` of a stack's rows: the element axis `axis - 1`.
    fn stack_inner(&mut self, list: NodeId, axis: usize, d: u32) -> Option<Lin> {
        let l = self.resolve(list);
        if self.head(l) == Some("vector") {
            // Stacked columns or rows share their length.
            let items = self.call(l)?.args.to_vec();
            let mut out: Option<Lin> = None;
            for it in items {
                if let Some(x) = self.sym_extent(it, axis - 1, d) {
                    match &out {
                        Some(o) => {
                            let o = o.clone();
                            self.unify(&o, &x);
                        }
                        None => out = Some(x),
                    }
                }
            }
            return out;
        }
        self.sym_extent(list, axis, d)
    }

    /// `x[sel…]`: each kept axis comes from a slice of `x` or an index array.
    fn get_extent(&mut self, args: &[NodeId], axis: usize, d: u32) -> Option<Lin> {
        let (x, sels) = (args[0], &args[1..]);
        if let [field] = sels
            && let Node::Lit(Scalar::Str(f)) = self.m.node(*field)
        {
            // A field of a data record: its declared set.
            let f = f.to_string();
            let rec = self.resolve(x);
            if let Some(set) = self.load_set(rec) {
                let set = self.resolve(set);
                let fset = self
                    .call(set)?
                    .named
                    .iter()
                    .find(|na| self.name(na.name) == f)?
                    .value;
                let h = self.set_hints(fset).get(axis).copied().flatten()?;
                return self.lin_int(h);
            }
            // A field of a record argument: the extent of its Stan name.
            if let Node::Ref(rf) = self.m.node(x)
                && rf.ns == RefNs::Local
            {
                let ctx = self.fctx.as_ref()?;
                let name = if ctx.record_local == Some(rf.name) {
                    self.fields
                        .iter()
                        .find(|(k, _)| self.name(**k) == f)
                        .map(|(_, v)| v.clone())?
                } else {
                    ctx.records.get(&rf.name)?.get(&f)?.clone()
                };
                return self.ext_lin.get(&name)?.get(axis).cloned();
            }
            // A literal record field already declared as data.
            let sym = *self.m.alias_chain(self.self_ref(x)?).last()?;
            let leaf = self.leaves.get(&(sym, f))?.clone();
            return self.ext_lin.get(&leaf)?.get(axis).cloned();
        }
        let mut kept = 0;
        for (k, &s) in sels.iter().enumerate() {
            let all = matches!(self.m.node(s), Node::Const(c) if self.name(*c) == "all");
            let array = !all && self.arank(s)? >= 1;
            if !(all || array) {
                continue;
            }
            if kept == axis {
                return if all {
                    self.sym_extent(x, k, d)
                } else {
                    self.sym_extent(s, 0, d)
                };
            }
            kept += 1;
        }
        // Axes of `x` past the selectors.
        self.sym_extent(x, sels.len() + axis - kept, d)
    }

    /// Output axis `axis` of an `aggregate`: an operand indexed by that axis.
    fn aggregate_extent(&mut self, args: &[NodeId], axis: usize, d: u32) -> Option<Lin> {
        let list = self.resolve(args[1]);
        let target = match self.m.node(*self.call(list)?.args.get(axis)?) {
            Node::Axis(a) => a.name,
            _ => return None,
        };
        // Every operand an axis indexes has that axis's length.
        let mut stack = vec![args[2]];
        let mut out: Option<Lin> = None;
        while let Some(n) = stack.pop() {
            if self.head(n) == Some("get") {
                let c = self.call(n)?;
                let pos = c.args[1..]
                    .iter()
                    .position(|&s| matches!(self.m.node(s), Node::Axis(a) if a.name == target));
                if let Some(p) = pos {
                    let x = c.args[0];
                    if let Some(l) = self.sym_extent(x, p, d) {
                        match &out {
                            Some(o) => {
                                let o = o.clone();
                                self.unify(&o, &l);
                            }
                            None => out = Some(l),
                        }
                    }
                    continue;
                }
            }
            self.m.for_each_child(n, |k| stack.push(k));
        }
        out
    }
}
