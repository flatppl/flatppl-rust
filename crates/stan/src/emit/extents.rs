//! Extent names from their symbolic source.
//!
//! An axis is named by the expression that produces it: the declared set
//! (`cartpow(reals, N)`), an `iid` count, the length argument of `zeros(T - 1)`,
//! or the matching axis of the operand an operation takes it from, through a
//! user function's body too. A value never picks a name. With no source, an
//! extent is a fresh data size for a `data` declaration and its number
//! elsewhere. So a program stays correct for data of another shape.

use super::Emitter;
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Dim, Node, NodeId, Phase, RefNs, Scalar, Symbol, Type};
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

/// The variate a size rule ties: its Stan name, its size forms where it has
/// them, and its inferred type.
pub(crate) struct Variate<'a> {
    lhs: &'a str,
    axes: Option<&'a [Lin]>,
    domain: Option<&'a Type>,
}

/// One axis of a law's size argument: an extent node (`n`, an entry of
/// `[2, n]`), or axis `axis` of `sizeof(of)`, whose node `at` is the whole
/// `sizeof`.
#[derive(Clone, Copy)]
enum SizeEntry {
    Node(NodeId),
    Axis { at: NodeId, of: NodeId, axis: usize },
}

/// A variate axis against the size its law gives, which may differ for some
/// data. Decided in [`Emitter::resolve_sizes`], on the size forms.
pub(crate) struct SizeCheck {
    axis: Lin,
    law: Lin,
    /// The variate's Stan name, for the message.
    var: String,
    /// The law's size argument, the refusal's location.
    at: NodeId,
    /// Neither side reads a `data` value: both are fixed in the program, and
    /// their static extents from the inferred types decide.
    fixed: bool,
    /// The static extent on this axis of the observed value and of the law's
    /// variate, from the inferred types.
    statics: (Option<u32>, Option<u32>),
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
        if self.function_depth > 0 && found.as_ref().is_some_and(has_token) {
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
            Some(l) if !(self.function_depth > 0 && has_token(&l)) => l.print(),
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
    pub(super) fn resolve_sizes(&mut self) -> Result<()> {
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
        // A size never names a value of its own class: `num_elements(y)`
        // cannot size `y` itself.
        let mut owners: HashMap<usize, HashSet<String>> = HashMap::new();
        for k in 0..self.svars.len() {
            let r = self.find(k);
            owners
                .entry(r)
                .or_default()
                .insert(self.svars[k].owner.clone());
        }
        let readable = |l: &Lin, own: &HashSet<String>| {
            super::functions::identifiers(&l.print())
                .iter()
                .all(|n| data_names.contains(n) && !own.contains(n))
        };
        // The data each accepted bound reads, by the value it sizes: a bound
        // that would read back into its own class through other bounds is a
        // cycle (`x` sized by `y`, `y` by `x`), and keeps its own data size.
        let mut reads: HashMap<String, Vec<String>> = HashMap::new();
        let cyclic =
            |reads: &HashMap<String, Vec<String>>, from: &[String], own: &HashSet<String>| {
                let mut seen: HashSet<String> = HashSet::new();
                let mut stack: Vec<String> = from.to_vec();
                while let Some(n) = stack.pop() {
                    if own.contains(&n) {
                        return true;
                    }
                    if seen.insert(n.clone())
                        && let Some(next) = reads.get(&n)
                    {
                        stack.extend(next.iter().cloned());
                    }
                }
                false
            };
        // The roots that print their bound: the check compares on it.
        let mut printed_bound: HashMap<usize, Lin> = HashMap::new();
        for k in 0..self.svars.len() {
            let r = self.find(k);
            if text.contains_key(&r) {
                let t = text[&r].clone();
                text.insert(k, t);
                continue;
            }
            let bound = self.svars[r].bound.clone().filter(|l| {
                readable(l, &owners[&r])
                    && !cyclic(
                        &reads,
                        &super::functions::identifiers(&l.print()),
                        &owners[&r],
                    )
            });
            let t = match bound {
                Some(l) => {
                    let names = super::functions::identifiers(&l.print());
                    for o in &owners[&r] {
                        reads.insert(o.clone(), names.clone());
                    }
                    printed_bound.insert(r, l.clone());
                    l.print()
                }
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
        // A size the law gives that the variate's declaration does not state
        // (a static input against a data count, a size only `transformed
        // data` computes): Stan checks it once the data is read. Decided on
        // the size forms: each class stands for its printed bound or for
        // itself.
        let roots: Vec<usize> = (0..self.svars.len()).map(|k| self.find(k)).collect();
        let resolved = |l: &Lin| -> Lin {
            let mut out = Lin::konst(l.konst);
            for (t, c) in &l.terms {
                let class = as_var(&Lin::term(t.clone())).map(|k| roots[k]);
                let part = match class {
                    Some(r) => match printed_bound.get(&r) {
                        Some(b) => b.scale(*c),
                        None => Lin::term(token(r)).scale(*c),
                    },
                    None => Lin::term(t.clone()).scale(*c),
                };
                out = out.plus(&part);
            }
            out
        };
        let mut checks: Vec<String> = Vec::new();
        for c in std::mem::take(&mut self.size_checks) {
            let (a, b) = (resolved(&c.axis), resolved(&c.law));
            if a == b {
                continue;
            }
            let var = &c.var;
            let ill = |p: i64, q: i64| {
                Refusal::new(
                    RefusalKind::IllFormedModule,
                    Some(c.at),
                    format!("the size of `{var}`"),
                    format!("`{var}` has size {p}, but its law gives size {q}"),
                )
            };
            // Both sides fixed in the program: their static extents decide.
            if c.fixed && !has_token(&a) {
                match c.statics {
                    (Some(p), Some(q)) if p == q => continue,
                    (Some(p), Some(q)) => return Err(ill(i64::from(p), i64::from(q))),
                    _ => {}
                }
            }
            if let (Some(p), Some(q)) = (a.as_const(), b.as_const()) {
                return Err(ill(p, q));
            }
            let (sa, sb) = (replace(&c.axis.print()), replace(&c.law.print()));
            let line = format!(
                "if ({sa} != {sb}) {{\n  reject(\"the size of {var} is \", {sa}, \", but its law gives \", {sb});\n}}"
            );
            if !checks.contains(&line) {
                checks.push(line);
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
        // A declaration that an earlier line reads moves up to just before its
        // first reader, repeated until nothing moves: the data block in
        // dependency order, otherwise as emitted. A text Stan accepts has no
        // such read, so it does not change. The bound on passes guards a
        // cycle, which the bound choice above already leaves out.
        let name_of = |line: &str| -> String {
            line.trim_end_matches(';')
                .rsplit(' ')
                .next()
                .unwrap_or("")
                .to_string()
        };
        for _ in 0..=data.len() * data.len() {
            let moved = (0..data.len()).find_map(|i| {
                let name = name_of(&data[i]);
                data[..i]
                    .iter()
                    .position(|l| super::functions::identifiers(l).contains(&name))
                    .map(|j| (i, j))
            });
            let Some((i, j)) = moved else {
                break;
            };
            let line = data.remove(i);
            data.insert(j, line);
        }
        self.data = data;
        self.tdata.extend(checks);
        Ok(())
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
                if matches!(self.m.node(args[0]), Node::Const(_)) {
                    return self.carried(id, &args[1..], axis, d);
                }
                let fun = self.user_callee(args[0])?;
                let operands = &self.broadcast_args(id).ok()?;
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
            // `cat(x, y, ...)`: the outer lengths add (a scalar counts one),
            // and the inner axes are the operands' common extent.
            "cat" if axis == 0 => {
                let mut n = Lin::konst(0);
                for &a in &args {
                    let l = match self.arank(a) {
                        Some(0) => Lin::konst(1),
                        _ => self.sym_extent(a, 0, d)?,
                    };
                    n = n.plus(&l);
                }
                Some(n)
            }
            "cat" => self.unified(&args, axis, d),
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

    /// The design `x` of a §09 regression law that prints as the vectorized
    /// statement of its broadcast law: a design that is not data. The
    /// response then takes the design's rows, as under the broadcast law.
    /// A GLM function checks its own sizes.
    fn regression_design(&self, law: NodeId) -> Option<NodeId> {
        let c = self.call(law)?;
        let flatppl_core::CallHead::User(callee) = c.head else {
            return None;
        };
        let Node::Ref(r) = self.m.node(callee) else {
            return None;
        };
        let member = self.name(r.name);
        if self.standard_module(callee).as_deref() != Some("regression")
            || !matches!(member, "NormalGLM" | "BernoulliLogitGLM" | "PoissonLogGLM")
        {
            return None;
        }
        let x = *self.law_args(member, law).ok()?.first()?;
        (self.m.phase_of(self.resolve(x)) != Some(Phase::Fixed)).then_some(x)
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
                // The variate has the shape of the highest-rank argument,
                // positional or keyword.
                let all: Vec<NodeId> = args[1..]
                    .iter()
                    .copied()
                    .chain(c.named.iter().map(|na| na.value))
                    .collect();
                let r = all.iter().filter_map(|&a| self.arank(a)).max()?;
                let full: Vec<NodeId> = all
                    .into_iter()
                    .filter(|&a| self.arank(a) == Some(r))
                    .collect();
                self.unified(&full, axis, d)
            }
            "Dirichlet" | "MvNormal" | "Multinomial" | "MvNormalCholesky" if axis == 0 => {
                let a = self.law_args(self.head(m)?, m).ok()?;
                let v = *a.iter().find(|&&x| self.arank(x) == Some(1))?;
                self.sym_extent(v, 0, d)
            }
            "%call" if axis == 0 => {
                let x = self.regression_design(m)?;
                self.sym_extent(x, 0, d)
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
    /// its `iid` counts, an LKJ dimension, or the array arguments of a
    /// broadcast law. A size the Stan function does not take must tie the
    /// data, so one the program cannot read refuses: data of another size
    /// would score silently.
    pub(super) fn constrain_variate(&mut self, lhs: &str, m: NodeId) -> Result<()> {
        let m = self.resolve(m);
        let axes = self.ext_lin.get(lhs).cloned();
        // The whole variate's static extents, from the inferred law type.
        let domain = match self.m.type_of(m) {
            Some(Type::Measure { domain, .. }) => Some((**domain).clone()),
            _ => None,
        };
        let Some(c) = self.call(m) else {
            return Ok(());
        };
        if self.head(m) == Some("broadcast")
            && let Some(axes) = &axes
        {
            for &v in &c.args.to_vec()[1..] {
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
            return Ok(());
        }
        let shape = Variate {
            lhs,
            axes: axes.as_deref(),
            domain: domain.as_ref(),
        };
        self.constrain_law(&shape, m, 0)
    }

    /// The size rule of the law `m` whose variate axes start at `offset` of
    /// the whole variate: `iid` counts (then the inner law), an LKJ
    /// dimension, a `Multinomial` count. One recursion, so a rule holds at
    /// every `iid` depth.
    fn constrain_law(&mut self, shape: &Variate<'_>, m: NodeId, offset: usize) -> Result<()> {
        let m = self.resolve(m);
        let Some(c) = self.call(m) else {
            return Ok(());
        };
        let args = c.args.to_vec();
        match self.head(m) {
            Some("iid") => {
                let ns = self.size_entries(args[1]);
                let k = ns.len();
                for (a, h) in ns.into_iter().enumerate() {
                    match h {
                        None => {}
                        Some(SizeEntry::Node(h)) => {
                            let x = self.size_argument(h, "an `iid` count")?;
                            self.tie_axis(shape, offset + a, &x, h)?;
                        }
                        // Axis j of `sizeof(v)`: v's own extent on this axis
                        // when v is the observed value.
                        Some(SizeEntry::Axis { at, of, axis: j }) => {
                            let x = self.axis_size(at, of, j)?;
                            let own =
                                (j == offset + a && self.is_observed(of, shape.lhs)).then_some(0);
                            self.tie_axis_own(shape, offset + a, &x, at, own)?;
                        }
                    }
                }
                self.constrain_law(shape, args[0], offset + k)
            }
            // §08: an n × n matrix. Stan's `lkj_corr` takes no n.
            Some(h @ ("LKJ" | "LKJCholesky")) => {
                if let Some(&n) = self.law_args(h, m)?.first() {
                    let x = self.size_argument(n, &format!("an `{h}` dimension"))?;
                    self.tie_axis(shape, offset, &x, n)?;
                    self.tie_axis(shape, offset + 1, &x, n)?;
                }
                Ok(())
            }
            // §09: the response has one entry for each row of the design.
            Some("%call") if offset == 0 => {
                if let Some(x) = self.regression_design(m)
                    && let Some(l) = shape.axes.and_then(|a| a.first()).cloned()
                    && let Some(rows) = self.sym_extent(x, 0, 0)
                {
                    self.unify(&l, &rows);
                }
                Ok(())
            }
            // §08 `Multinomial`: the support is the vectors summing to n,
            // which the row guard tests, so n must print. n read as the
            // observed value's own sum holds by construction: no guard.
            Some("Multinomial") => {
                if let Some(&n) = self.law_args("Multinomial", m)?.first() {
                    self.size_argument(n, "a `Multinomial` count")?;
                    if offset == 0 && self.own_sum(n, shape.lhs) {
                        self.own_sum.insert(shape.lhs.to_string());
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Axis `axis` of the variate takes the law size `x` (argument node
    /// `at`). Unify where the variate's size is a data size, and record a
    /// check where the two can differ for some data.
    fn tie_axis(&mut self, shape: &Variate<'_>, axis: usize, x: &Lin, at: NodeId) -> Result<()> {
        let own = self.own_extent_offset(at, shape.lhs, axis);
        self.tie_axis_own(shape, axis, x, at, own)
    }

    /// [`Self::tie_axis`] with the own-extent offset (see
    /// [`Self::own_extent_offset`]) given by the caller.
    fn tie_axis_own(
        &mut self,
        shape: &Variate<'_>,
        axis: usize,
        x: &Lin,
        at: NodeId,
        own: Option<i64>,
    ) -> Result<()> {
        let Some(l) = shape.axes.and_then(|a| a.get(axis)).cloned() else {
            return Ok(());
        };
        self.unify(&l, x);
        // The law size is the observed value's own extent on this axis,
        // through any chain of bindings: equal for every data set, or, with
        // an offset, for none.
        if let Some(k) = own {
            if k == 0 {
                return Ok(());
            }
            let var = shape.lhs;
            return Err(Refusal::new(
                RefusalKind::IllFormedModule,
                Some(at),
                format!("the size of `{var}`"),
                format!(
                    "the law gives `{var}` its own extent on axis {} {} {}, so no data fits it",
                    axis + 1,
                    if k > 0 { "plus" } else { "minus" },
                    k.abs()
                ),
            ));
        }
        // A size inside a function body or a loop reads a local, which no
        // `transformed data` check can read.
        let mut stack = vec![at];
        while let Some(id) = stack.pop() {
            if matches!(self.m.node(id), Node::Ref(r) if r.ns == RefNs::Local) {
                return Ok(());
            }
            self.m.for_each_child(id, |k| stack.push(k));
        }
        if self.function_depth > 0 {
            return Ok(());
        }
        // A size is an integer, never a parameter. The static extent of a
        // draw (`lengthof(X)` of `X ~ iid(…, 3)`) reads no draw value.
        if self.m.phase_of(at) == Some(Phase::Stochastic) && x.as_const().is_none() {
            return Err(Refusal::not_yet(Some(at), "a size that reads a draw"));
        }
        // A side is fixed in the program when it is Fixed phase and reaches no
        // `data` value: its Stan value is computed from the program text alone.
        let fixed_in_program = |me: &Self, node: NodeId| {
            me.m.phase_of(node) == Some(Phase::Fixed)
                && !me
                    .reached(node, &HashSet::new())
                    .iter()
                    .any(|b| me.env.get(b).is_some_and(|s| me.in_data_block(s)))
        };
        let observed = self.observing;
        let fixed = fixed_in_program(self, at)
            && !self.in_data_block(shape.lhs)
            && observed.is_some_and(|v| fixed_in_program(self, v));
        let static_at = |t: Option<&Type>| match t {
            Some(Type::Array { shape, .. }) => match shape.get(axis) {
                Some(Dim::Static(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        };
        let statics = (
            static_at(observed.and_then(|v| self.m.type_of(v))),
            static_at(shape.domain),
        );
        self.size_checks.push(SizeCheck {
            axis: l,
            law: x.clone(),
            var: shape.lhs.to_string(),
            at,
            fixed,
            statics,
        });
        Ok(())
    }

    /// `lengthof(v)` or `sizeof(v)[i]` (through fixed bindings) where v's
    /// inferred type states that extent.
    fn static_extent(&self, n: NodeId) -> Option<u32> {
        let r = self.resolve(n);
        let c = self.call(r)?;
        let (v, axis) = match (self.head(r)?, &c.args[..]) {
            // §07 gives a matrix no `lengthof` (its print refuses).
            ("lengthof", [v]) if !self.is_matrix(*v) => (*v, 0),
            ("get", [s, i]) => {
                let s = self.resolve(*s);
                let (Some("sizeof"), Some([v])) = (self.head(s), self.call(s).map(|c| &c.args[..]))
                else {
                    return None;
                };
                let Node::Lit(Scalar::Int(i)) = self.m.node(self.resolve(*i)) else {
                    return None;
                };
                // `sizeof` of a vector of vectors gives its outer extents only.
                let axis = usize::try_from(*i - 1).ok()?;
                if self.outer_axes(*v).is_some_and(|k| axis >= k) {
                    return None;
                }
                (*v, axis)
            }
            _ => return None,
        };
        match self.m.type_of(v)? {
            Type::Array { shape, .. } => match shape.get(axis)? {
                Dim::Static(k) => Some(*k),
                Dim::Dynamic => None,
            },
            _ => None,
        }
    }

    /// `true` when the Stan name is a `data` value (literal or loaded).
    fn in_data_block(&self, name: &str) -> bool {
        self.json.contains_key(name) || self.deferred.iter().any(|d| d.stan == name)
    }

    /// `true` when the binding behind `v` is the value observed as `lhs`.
    fn is_observed(&self, v: NodeId, lhs: &str) -> bool {
        let field = |me: &Self, n: NodeId| {
            let r = me.resolve(n);
            match (me.head(r), me.call(r).map(|c| &c.args[..])) {
                (Some("get"), Some(&[x, f])) => match (me.self_ref(x), me.m.node(f)) {
                    (Some(s), Node::Lit(Scalar::Str(f))) => {
                        Some((*me.m.alias_chain(s).last().unwrap(), f.to_string()))
                    }
                    _ => None,
                },
                _ => None,
            }
        };
        // The observed value is a record field (`r.u`) the size reads too.
        let same_field = || {
            self.observing
                .and_then(|o| field(self, o))
                .is_some_and(|f| field(self, v) == Some(f))
        };
        self.self_ref(v)
            .map(|s| *self.m.alias_chain(s).last().unwrap())
            .and_then(|s| self.env.get(&s))
            .is_some_and(|stan| stan == lhs)
            || same_field()
    }

    /// `Some(k)` when the size `h` is, through fixed bindings and literal
    /// offsets, the extent on `axis` of the value observed as `lhs` plus k
    /// (`lengthof(v)` is axis 0, `sizeof(v)[i]` axis i - 1).
    fn own_extent_offset(&self, h: NodeId, lhs: &str, axis: usize) -> Option<i64> {
        let lit = |me: &Self, x: NodeId| match me.m.node(me.resolve(x)) {
            Node::Lit(Scalar::Int(k)) => Some(*k),
            _ => None,
        };
        let r = self.resolve(h);
        let c = self.call(r)?;
        match (self.head(r)?, &c.args[..]) {
            ("lengthof", [v]) => (axis == 0 && self.is_observed(*v, lhs)).then_some(0),
            ("get", [s, i]) => {
                let s = self.resolve(*s);
                let (Some("sizeof"), Some([v])) = (self.head(s), self.call(s).map(|c| &c.args[..]))
                else {
                    return None;
                };
                let i = lit(self, *i)?;
                let outer = self.outer_axes(*v).is_none_or(|k| axis < k);
                (i >= 1 && i - 1 == axis as i64 && outer && self.is_observed(*v, lhs)).then_some(0)
            }
            ("add", [a, b]) => match (lit(self, *a), lit(self, *b)) {
                (_, Some(k)) => Some(self.own_extent_offset(*a, lhs, axis)? + k),
                (Some(k), _) => Some(self.own_extent_offset(*b, lhs, axis)? + k),
                _ => None,
            },
            ("sub", [a, b]) => Some(self.own_extent_offset(*a, lhs, axis)? - lit(self, *b)?),
            _ => None,
        }
    }

    /// `true` when the `Multinomial` count `n` is, through fixed bindings,
    /// `sum(v)` of the value observed as `lhs`.
    fn own_sum(&self, n: NodeId, lhs: &str) -> bool {
        let r = self.resolve(n);
        self.head(r) == Some("sum")
            && matches!(self.call(r).map(|c| &c.args[..]), Some([v]) if self.is_observed(*v, lhs))
    }

    /// The axes of a size argument (§06 `iid(M, size)`): `sizeof(v)` of a
    /// value of static rank gives one entry per axis of v; any other size
    /// its extents ([`Emitter::extent_list`]).
    fn size_entries(&self, n: NodeId) -> Vec<Option<SizeEntry>> {
        let r = self.resolve(n);
        if self.head(r) == Some("sizeof")
            && let Some(&[of]) = self.call(r).map(|c| &c.args[..])
            && let Some(k) = self.sizeof_rank(of)
        {
            return (0..k)
                .map(|axis| Some(SizeEntry::Axis { at: r, of, axis }))
                .collect();
        }
        self.extent_list(n)
            .into_iter()
            .map(|h| h.map(SizeEntry::Node))
            .collect()
    }

    /// Axis `axis` of `sizeof(of)` as a Stan integer: the size
    /// [`Self::size_argument`] gives the entry `sizeof(of)[axis + 1]`, step
    /// for step (that node is not in the module, so the steps repeat here on
    /// the same texts). `at` is the whole `sizeof`.
    fn axis_size(&mut self, at: NodeId, of: NodeId, axis: usize) -> Result<Lin> {
        let from = self.sizeof_form(of).map_or(0, |(_, from)| from);
        // The text the entry prints (see `outer_size_at`), or `None` where
        // `of` cannot print, as `opaque` gives.
        let entry = |me: &mut Self| {
            let v = me.expr(of).ok()?;
            Some(Lin::term(format!("dims({})[{}]", v.s, from + axis + 1)))
        };
        // `lin_int`: a fixed size prints, so data written for it keeps its
        // check.
        if self.m.phase_of(at) == Some(Phase::Fixed)
            && let Some(x) = entry(self)
        {
            return Ok(x);
        }
        // `static_extent`: the static extent of of's type.
        let fixed = match self.ty(of) {
            Some(Type::Array { shape, .. }) => shape.get(axis).cloned(),
            Some(Type::TVector { len, .. }) if axis == 0 => Some(len),
            _ => None,
        };
        if let Some(Dim::Static(k)) = fixed {
            return Ok(Lin::konst(i64::from(k)));
        }
        // A size that reads a draw: the binder's refusal names it.
        if self.m.phase_of(at) == Some(Phase::Stochastic) {
            self.expr(of)?;
        }
        // An integer input is a Stan data integer.
        if let Some(x) = entry(self) {
            return Ok(x);
        }
        Err(Refusal::not_yet(
            Some(at),
            "an `iid` count that the Stan program cannot read as an integer",
        ))
    }

    /// A law's size argument `n` as a Stan integer, or a refusal that names
    /// it (`what` is "an `iid` count"): a module member or an untyped
    /// `load_data` value has no integer the program can read.
    fn size_argument(&mut self, n: NodeId, what: &str) -> Result<Lin> {
        if let Some(x) = self.lin_int(n) {
            return Ok(x);
        }
        // The static extent of a value of fixed shape (a parameter in
        // `cartpow(reals, 3)`), read from its inferred type.
        if let Some(k) = self.static_extent(n) {
            return Ok(Lin::konst(i64::from(k)));
        }
        // A count that reads a draw: the binder's refusal names the latent
        // draw the target integrates out.
        if self.m.phase_of(n) == Some(Phase::Stochastic) {
            self.expr(n)?;
        }
        // A size outside its function's domain (`lengthof` of a matrix) is
        // ill-formed, not unreadable.
        if let Err(e) = self.expr(n)
            && e.kind == RefusalKind::IllFormedModule
        {
            return Err(e);
        }
        // An integer input or kernel input is a Stan data integer, declared
        // before the statements that read it.
        if self.shape(n).is_some_and(|s| s.rank == 0 && s.int)
            && let Some(x) = self.opaque(n)
        {
            return Ok(x);
        }
        let name = match self.m.node(n) {
            Node::Ref(r) => match r.ns {
                RefNs::Module(alias) => format!(" `{}.{}`", self.name(alias), self.name(r.name)),
                _ => format!(" `{}`", self.name(r.name)),
            },
            _ => String::new(),
        };
        Err(Refusal::not_yet(
            Some(n),
            format!("{what}{name} that the Stan program cannot read as an integer"),
        ))
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
            // A field of a record written in place: the field's value.
            if self.head(rec) == Some("record") {
                let v = self
                    .call(rec)?
                    .named
                    .iter()
                    .find(|na| self.name(na.name) == f)?
                    .value;
                return self.sym_extent(v, axis, d);
            }
            // A field of a parameter of a body being sized: of its argument.
            if let Node::Ref(rf) = self.m.node(x)
                && rf.ns == RefNs::Local
                && let Some(&(arg, 0)) = self.esubst.last().and_then(|m| m.get(&rf.name))
            {
                let top = self.esubst.pop();
                let out = self.get_extent(&[arg, sels[0]], axis, d);
                self.esubst.extend(top);
                return out;
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
                    ctx.records.get(&rf.name)?.get(&f)?.at(super::P_ATOM)
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
