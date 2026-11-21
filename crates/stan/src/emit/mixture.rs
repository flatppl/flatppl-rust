//! Mixtures: `superpose` and `ksuperpose`, normalized or not (spec §06
//! "Density of composed measures", "Additive superposition").
//!
//! A mixture's log density is `logsumexp_i(log w_i + lp_i)`. Its normalizer
//! is `Σ w_i` because every component is a probability measure. Stan cannot
//! vectorize a mixture over observations (User's Guide "Vectorizing
//! mixtures"), so an iid variate takes one `target +=` per element. The
//! provably normalized forms of §06 ([`Emitter::provably_normalized`]: exact
//! literal weights, the complement pair, simplex weights) need no
//! normalizer, written `normalize` or not; the complement pair and a simplex
//! `ksuperpose` weight use `log_mix`.
//! A `Dirac` component beside a discrete law splits observed data in
//! `transformed data` into the values at the atom and the rest (User's Guide
//! "Zero-inflated and hurdle models").

use super::Emitter;
use super::functions::FnCtx;
use super::structure::Law;
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::number;
use flatppl_core::{Node, NodeId, Type, ValueSet};
use std::collections::HashMap;

mod nested;

/// One element's log density in a weighted log likelihood (Ruling 4).
pub(super) enum ElementLp {
    One(String),
    /// A `Dirac(c)` atom beside a law: the test `y == c`, the term at the
    /// atom, the term elsewhere.
    Split(String, String, String),
}

/// The loop index a per-component argument is rendered with.
const COMP: &str = "\u{4}";

/// One component: its weight (none for weight 1) and its law, or an atom.
enum Part {
    Law(Option<NodeId>, NodeId),
    Dirac(Option<NodeId>, NodeId),
}

/// Stands for a row index (or the rest rows) in a [`RowArg`] template.
pub(super) const ROW: &str = "\u{7}";

/// One argument of a law over the rows of a data matrix.
pub(super) enum RowArg {
    /// The same for every row.
    Shared(String),
    /// A row-length vector, the same for every row.
    Repeated(String),
    /// Per row: `row` at one row index, `rest` over an index array of rows,
    /// flattened row by row. [`ROW`] marks the index. `atom`: the value at
    /// every atom row, when they all share one.
    PerRow {
        row: String,
        rest: String,
        atom: Option<String>,
    },
}

/// A discrete law over each row of a data matrix: `name(y | args…)`.
pub(super) struct RowLaw {
    /// `bernoulli_lpmf`.
    pub name: String,
    pub args: Vec<RowArg>,
}

/// Which rows a [`RowLaw`] density covers.
#[derive(Clone, Copy)]
pub(super) enum Render<'a> {
    /// One row, at a row index.
    Row,
    /// Index-selected rows flattened, with this many rows.
    Rest(&'a str),
    /// All rows flattened, with this many rows.
    All(&'a str),
    /// Any one atom row, where every atom row has the same arguments.
    Atom,
}

impl RowLaw {
    /// The argument texts for `render`, [`ROW`] replaced by `rows`.
    fn args_text(&self, rows: &str, render: Render<'_>) -> Vec<String> {
        self.args
            .iter()
            .map(|a| match (a, render) {
                (RowArg::Shared(s), _) => s.clone(),
                (RowArg::Repeated(v), Render::Row | Render::Atom) => v.clone(),
                (RowArg::Repeated(v), Render::Rest(n) | Render::All(n)) => {
                    format!("to_vector(rep_matrix({v}, {n}))")
                }
                (RowArg::PerRow { atom: Some(t), .. }, Render::Atom) => t.clone(),
                (RowArg::PerRow { row, .. }, Render::Row | Render::Atom) => row.replace(ROW, rows),
                (RowArg::PerRow { rest, .. }, Render::Rest(_)) => rest.replace(ROW, rows),
                (RowArg::PerRow { rest, .. }, Render::All(_)) => {
                    rest.replace(&format!("[{ROW}]"), "")
                }
            })
            .collect()
    }

    /// `name(y | args)` with [`ROW`] replaced by `rows`.
    pub(super) fn density(&self, y: &str, rows: &str, render: Render<'_>) -> String {
        let args = self.args_text(rows, render);
        if args.is_empty() {
            format!("{}({y})", self.name)
        } else {
            format!("{}({y} | {})", self.name, args.join(", "))
        }
    }

    /// `y ~ dist(args)` over all `n` rows of `y`, flattened row by row.
    pub(super) fn statement(&self, y: &str, n: &str) -> String {
        let dist = self.name.trim_end_matches("_lpmf");
        let args = self.args_text("", Render::All(n));
        format!("to_array_1d({y}) ~ {dist}({});", args.join(", "))
    }

    /// `true` when every atom row has the same arguments: no per-row
    /// argument, or each one's value at an atom row is known.
    fn atom_shared(&self) -> bool {
        self.args
            .iter()
            .all(|a| !matches!(a, RowArg::PerRow { atom: None, .. }))
    }
}

impl<'m> Emitter<'m> {
    /// `(normalize, superpose-or-ksuperpose node)` when `m` is a mixture.
    fn mixture(&self, m: NodeId) -> Option<(bool, NodeId)> {
        let m = self.resolve(m);
        match self.head(m)? {
            "normalize" => {
                let inner = self.resolve(self.call(m)?.args[0]);
                matches!(self.head(inner), Some("superpose" | "ksuperpose"))
                    .then_some((true, inner))
            }
            "superpose" | "ksuperpose" => Some((false, m)),
            _ => None,
        }
    }

    /// The statements for `lhs ~ m` when `m` is a mixture, or an iid product
    /// or a broadcast user law of one.
    pub(super) fn mixture_statements(
        &mut self,
        lhs: &str,
        m: NodeId,
    ) -> Result<Option<Vec<String>>> {
        if let Some((norm, mix)) = self.mixture(m) {
            // The whole variate is one draw from the mixture.
            let (pre, lp) = self.mixture_lp(mix, norm, lhs)?;
            return Ok(Some(vec![block(pre, format!("target += {lp};"))]));
        }
        let r = self.resolve(m);
        if self.head(r) == Some("broadcast")
            && let Some(fun) = self.user_callee(self.call(r).unwrap().args[0])
            && self.mixture(self.call(fun).unwrap().args[0]).is_some()
        {
            return self.broadcast_mixture(lhs, r).map(Some);
        }
        if self.head(r) != Some("iid") {
            return Ok(None);
        }
        let Some((norm, mix)) = self.mixture(self.call(r).unwrap().args[0]) else {
            return Ok(None);
        };
        if let Some(lines) = self.dirac_split(lhs, mix, norm)? {
            return Ok(Some(lines));
        }
        let i = self.fresh_loop();
        let n = self.extent_text(lhs);
        let out = self.mixture_lp(mix, norm, &format!("{lhs}[{i}]"));
        self.used.remove(&i);
        let (pre, lp) = out?;
        let body = block(pre, format!("target += {lp};"));
        let inner: String = body.lines().map(|l| format!("  {l}\n")).collect();
        Ok(Some(vec![format!("for ({i} in 1:{n}) {{\n{inner}}}")]))
    }

    /// `f.(args…)` of a user law whose body is a mixture: the body per
    /// element, its parameters at that element.
    fn broadcast_mixture(&mut self, lhs: &str, m: NodeId) -> Result<Vec<String>> {
        let loop_ = super::structure::LOOP;
        let (body, ctx) = self.user_law_body(m, loop_)?;
        let (norm, mix) = self.mixture(body).unwrap();
        let outer = self.fctx.replace(ctx);
        let out = match self.dirac_split(lhs, mix, norm) {
            Ok(Some(lines)) => Ok(lines),
            Ok(None) => {
                let i = self.fresh_loop();
                let n = self.extent_text(lhs);
                let r = self.mixture_lp(mix, norm, &format!("{lhs}[{loop_}]"));
                self.used.remove(&i);
                r.map(|(pre, lp)| {
                    let body = block(pre, format!("target += {lp};")).replace(loop_, &i);
                    let inner: String = body.lines().map(|l| format!("  {l}\n")).collect();
                    vec![format!("for ({i} in 1:{n}) {{\n{inner}}}")]
                })
            }
            Err(e) => Err(e),
        };
        let ctx = std::mem::replace(&mut self.fctx, outer);
        if ctx.is_some_and(|c| !c.lines.is_empty()) {
            return Err(Refusal::not_yet(
                Some(m),
                "a mixture law whose arguments need statements",
            ));
        }
        out
    }

    /// `true` when `x` reads only literals and fixed module values.
    fn fixed_value(&self, x: NodeId) -> bool {
        let mut stack = vec![x];
        while let Some(id) = stack.pop() {
            match self.m.node(id) {
                Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
                    let fixed = self.m.binding_by_name(r.name).is_some_and(|b| {
                        self.m.phase_of(self.m.binding(b).rhs) == Some(flatppl_core::Phase::Fixed)
                    });
                    if !fixed {
                        return false;
                    }
                }
                Node::Ref(_) => return false,
                _ => {}
            }
            self.m.for_each_child(id, |c| stack.push(c));
        }
        true
    }

    /// `fill(c, [1, n])`: a 1×n block of one value, the atom of a row.
    pub(super) fn row_fill(&self, atom: NodeId) -> bool {
        let f = self.resolve(atom);
        if self.head(f) != Some("fill") {
            return false;
        }
        let n = self.resolve(self.call(f).unwrap().args[1]);
        self.call(n).is_some_and(|c| {
            self.head(n) == Some("vector")
                && c.args.len() == 2
                && matches!(
                    self.m.node(c.args[0]),
                    Node::Lit(flatppl_core::Scalar::Int(1))
                )
        })
    }

    /// The Dirac split over the rows of a two-dimensional data variate: an
    /// atom `fill(c, T)` (or `fill(c, [1, T])`) against a law over each row.
    /// A row is at the atom when its largest and smallest values are `c`.
    /// With arguments shared by all rows, the atom rows are one term; with
    /// per-row arguments, one term per atom row (Stan has no per-row
    /// `_lpmf`). The other rows score in one call, flattened row by row.
    #[allow(clippy::too_many_arguments)]
    fn dirac_split_rows(
        &mut self,
        lhs: &str,
        mix: NodeId,
        norm: bool,
        dirac_first: bool,
        (wd, atom): (Option<NodeId>, NodeId),
        (wl, law): (Option<NodeId>, NodeId),
    ) -> Result<Vec<String>> {
        let fill = self.resolve(atom);
        let row_law = self.resolve(law);
        let data = self.is_data(lhs);
        let shaped = self
            .decls
            .get(lhs)
            .is_some_and(|t| t.is_int() && t.array.len() == 2);
        if self.head(fill) != Some("fill") || !data || !shaped {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom over a vector variate in this form",
            ));
        }
        if !self.dirac_weights(wl, wd, norm) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` mixture with these weights",
            ));
        }
        let [cv, fill_n] = self.call(fill).unwrap().args[..] else {
            unreachable!()
        };
        // The atom test runs in `transformed data`: its value must be data.
        if !self.fixed_value(cv) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom whose value is not data",
            ));
        }
        // The atom is one whole row: `fill(c, n)` or `fill(c, [1, n])` with n
        // the rows' length.
        let len_node = if self.row_fill(atom) {
            self.call(self.resolve(fill_n)).unwrap().args[1]
        } else {
            fill_n
        };
        let row_len = self.ext_lin.get(lhs).and_then(|e| e.get(1)).cloned();
        let same = match (self.lin_int(len_node), row_len) {
            (Some(a), Some(b)) => self.same_size(&a, &b),
            _ => false,
        };
        if !same {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom of another length than its rows",
            ));
        }
        let c = self.expr(cv)?;
        let rl = match self.head(row_law) {
            Some("broadcast") => self.broadcast_row_law(mix, law)?,
            Some("markovchain") => self.markovchain_row_law(lhs, row_law, Some(&c.s))?,
            _ => {
                return Err(Refusal::not_yet(
                    Some(mix),
                    "a `Dirac` mixture with this row law",
                ));
            }
        };
        let n = self.extent_text(lhs);
        let per_row = !rl.atom_shared();
        let count = self.fresh(&format!("{lhs}_n_atom"));
        let at_idx = per_row.then(|| self.fresh(&format!("{lhs}_atom")));
        let rest = self.fresh(&format!("{lhs}_rest"));
        let i = self.fresh_loop();
        let (a, b) = (self.fresh_local("a"), self.fresh_local("b"));
        let at = format!("max({lhs}[{i}]) == {c} && min({lhs}[{i}]) == {c}", c = c.s);
        self.tdata.push(format!(
            "int {count} = 0;\nfor ({i} in 1:{n}) {{\n  {count} += {at};\n}}"
        ));
        match &at_idx {
            Some(at_idx) => {
                self.tdata.push(format!("array[{count}] int {at_idx};"));
                self.tdata.push(format!("array[{n} - {count}] int {rest};"));
                self.tdata.push(format!(
                    "{{\n  int {a} = 1;\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    if ({at}) {{\n      {at_idx}[{a}] = {i};\n      {a} += 1;\n    }} else {{\n      {rest}[{b}] = {i};\n      {b} += 1;\n    }}\n  }}\n}}"
                ));
            }
            None => {
                self.tdata.push(format!("array[{n} - {count}] int {rest};"));
                self.tdata.push(format!(
                    "{{\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    if (!({at})) {{\n      {rest}[{b}] = {i};\n      {b} += 1;\n    }}\n  }}\n}}"
                ));
            }
        }
        for v in [&i, &a, &b] {
            self.used.remove(v);
        }
        let wlx = self.real(wl.unwrap())?.s;
        let wdx = self.real(wd.unwrap())?.s;
        if wlx.contains(super::structure::LOOP) || wdx.contains(super::structure::LOOP) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` row mixture with a weight per row",
            ));
        }
        let mix_with = |me: &mut Self, lp: &str| -> Result<String> {
            Ok(if dirac_first {
                format!("log_mix({}, 0, {lp})", me.real(wd.unwrap())?.s)
            } else {
                format!("log_mix({wlx}, {lp}, 0)")
            })
        };
        let mut out = Vec::new();
        match &at_idx {
            Some(at_idx) => {
                let k = self.fresh_loop();
                let row = format!("{at_idx}[{k}]");
                let lp = rl.density(&format!("{lhs}[{row}]"), &row, Render::Row);
                let term = mix_with(self, &lp)?;
                out.push(format!(
                    "for ({k} in 1:{count}) {{\n  target += {term};\n}}"
                ));
                self.used.remove(&k);
            }
            None => {
                // Every atom row scores the same: the atom itself.
                let atom_x = if self.row_fill(atom) {
                    let len = self
                        .decls
                        .get(lhs)
                        .and_then(|t| t.array.get(1).cloned())
                        .unwrap_or_else(|| format!("size({lhs}[1])"));
                    format!("rep_array({}, {len})", c.s)
                } else {
                    self.expr(atom)?.s
                };
                let lp = rl.density(&atom_x, "", Render::Atom);
                let term = mix_with(self, &lp)?;
                out.push(format!("target += {count} * {term};"));
            }
        }
        let reps = format!("{n} - {count}");
        let lp_rest = rl.density(
            &format!("to_array_1d({lhs}[{rest}])"),
            &rest,
            Render::Rest(&reps),
        );
        let log_l = self.log_weight(wl.unwrap())?;
        out.push(format!("target += ({n} - {count}) * {log_l} + {lp_rest};"));
        Ok(out)
    }

    /// The row law of a broadcast `D.(args…)` over one row: each argument
    /// shared, a vector repeated per row, or a per-row matrix row `X[i]`.
    fn broadcast_row_law(&mut self, mix: NodeId, law: NodeId) -> Result<RowLaw> {
        let row_law = self.resolve(law);
        let l = self.law_any(law)?;
        if !l.discrete
            || !l.vectorized
            || l.trunc.is_some()
            || l.support.is_some()
            || l.declared
            || l.lhs("{y}") != "{y}"
        {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` mixture with this row law",
            ));
        }
        let loop_ = super::structure::LOOP;
        let mut repeated = Vec::new();
        for &arg in &self.call(row_law).unwrap().args[1..].to_vec() {
            if self.shape(arg).is_some_and(|s| s.rank == 1) {
                let e = self.real(arg)?.s;
                if !e.contains(loop_) {
                    // A row-length vector the law printed in another form
                    // (`invlogit.(v)` as `bernoulli_logit(v)`) cannot be
                    // repeated per row by its text.
                    if !l.call.contains(&e) {
                        return Err(Refusal::not_yet(
                            Some(mix),
                            "a `Dirac` mixture whose row law rewrites a vector argument",
                        ));
                    }
                    repeated.push(e);
                }
            } else if self.shape(arg).is_some_and(|s| s.rank > 1) {
                return Err(Refusal::not_yet(
                    Some(mix),
                    "a `Dirac` mixture with a matrix row argument",
                ));
            }
        }
        let open = l.call.find('(').unwrap_or(l.call.len());
        let name = l.call[..open].to_string();
        let inner = l.call[open..]
            .strip_prefix('(')
            .and_then(|x| x.strip_suffix(')'))
            .unwrap_or("");
        let mut args = Vec::new();
        for a in split_args(inner) {
            args.push(if a.contains(loop_) {
                // Only a row of a named matrix: `X[i]`, and `X[rest]` for the
                // other rows.
                // `to_vector(X[rest]')` is row-major only for a matrix.
                let base = a.strip_suffix(&format!("[{loop_}]")).filter(|b| {
                    !b.is_empty()
                        && b.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                        && self
                            .decls
                            .get(*b)
                            .is_some_and(|t| t.base == "matrix" && t.array.is_empty())
                });
                let Some(base) = base else {
                    return Err(Refusal::not_yet(
                        Some(mix),
                        "a `Dirac` mixture with a per-row argument of this form",
                    ));
                };
                RowArg::PerRow {
                    row: format!("{base}[{ROW}]"),
                    rest: format!("to_vector({base}[{ROW}]')"),
                    atom: None,
                }
            } else if repeated.contains(&a) {
                RowArg::Repeated(a)
            } else {
                RowArg::Shared(a)
            });
        }
        Ok(RowLaw {
            name: format!("{name}_lpmf"),
            args,
        })
    }

    pub(super) fn fresh_loop(&mut self) -> String {
        for c in ["i", "j", "n", "k"] {
            if !self.used.contains(c) && !self.reserved.contains(c) {
                self.used.insert(c.to_string());
                return c.to_string();
            }
        }
        self.fresh_local("idx")
    }

    fn extent_text(&self, name: &str) -> String {
        self.decls
            .get(name)
            .and_then(|t| t.array.first().or(t.dims.first()).cloned())
            .unwrap_or_else(|| format!("size({name})"))
    }

    fn parts(&self, mix: NodeId) -> Result<Vec<Part>> {
        let c = self.call(mix).unwrap();
        let mut out = Vec::new();
        for &a in c.args.iter() {
            let r = self.resolve(a);
            let (w, law) = if self.head(r) == Some("weighted") {
                let wc = self.call(r).unwrap();
                self.nonnegative_weight(wc.args[0])?;
                (Some(wc.args[0]), self.resolve(wc.args[1]))
            } else {
                (None, r)
            };
            out.push(if self.head(law) == Some("Dirac") {
                Part::Dirac(w, self.call(law).unwrap().args[0])
            } else {
                Part::Law(w, law)
            });
        }
        Ok(out)
    }

    /// A negative numeric literal weight (`-0.5`) refuses: §06 `weighted`
    /// takes "a non-negative weight", and its `log(w)` would be NaN. A
    /// computed weight is the model's own.
    fn nonnegative_weight(&self, w: NodeId) -> Result<()> {
        let lit = |n: NodeId| match self.m.node(self.resolve(n)) {
            Node::Lit(flatppl_core::Scalar::Int(k)) => Some(*k as f64),
            Node::Lit(flatppl_core::Scalar::Real(x)) => Some(*x),
            _ => None,
        };
        let r = self.resolve(w);
        let value = match self.head(r) {
            Some("neg") => lit(self.call(r).unwrap().args[0]).map(|x| -x),
            _ => lit(w),
        };
        match value {
            Some(x) if x < 0.0 => Err(Refusal::new(
                RefusalKind::IllFormedModule,
                Some(w),
                format!("the negative weight {x}"),
                format!(
                    "the weight {x} is negative; §06 `weighted` takes a non-negative weight, so \
                     the module is ill-formed"
                ),
            )),
            _ => Ok(()),
        }
    }

    /// `log(w)` of a mixture weight, `log1m(x)` when `w` is `1 - x`: the
    /// complement keeps its precision near 0.
    pub(super) fn log_weight(&mut self, w: NodeId) -> Result<String> {
        let r = self.resolve(w);
        if self.head(r) == Some("sub") {
            let c = self.call(r).unwrap();
            let (a, b) = (self.resolve(c.args[0]), c.args[1]);
            let one = matches!(self.m.node(a), Node::Lit(flatppl_core::Scalar::Int(1)))
                || matches!(self.m.node(a), Node::Lit(flatppl_core::Scalar::Real(x)) if *x == 1.0);
            if one {
                return Ok(format!("log1m({})", self.real(b)?.s));
            }
        }
        Ok(format!("log({})", self.real(w)?.s))
    }

    /// `weighted(w, A), weighted(1 - w, B)`: the second weight is `1 - w`,
    /// `w` one node.
    fn complement(&self, a: Option<NodeId>, b: Option<NodeId>) -> bool {
        let (Some(a), Some(b)) = (a, b) else {
            return false;
        };
        let b = self.resolve(b);
        if self.head(b) != Some("sub") {
            return false;
        }
        let bc = self.call(b).unwrap();
        let one = matches!(
            self.m.node(self.resolve(bc.args[0])),
            Node::Lit(flatppl_core::Scalar::Int(1))
        ) || matches!(self.m.node(self.resolve(bc.args[0])), Node::Lit(flatppl_core::Scalar::Real(x)) if *x == 1.0);
        one && self.m.structural_eq(bc.args[1], a)
    }

    /// `true` when the value set of `x` lies within `unitinterval`.
    fn within_unit(&self, x: NodeId) -> bool {
        match self.value_set(x) {
            Some(ValueSet::UnitInterval | ValueSet::Booleans) => true,
            Some(ValueSet::Interval(lo, hi)) => lo >= 0.0 && hi <= 1.0,
            _ => false,
        }
    }

    /// The value set of `x`, through the binding it names.
    fn value_set(&self, x: NodeId) -> Option<ValueSet> {
        self.m.valueset_of(x).cloned().or_else(|| {
            let s = self.self_ref(x)?;
            let b = self
                .m
                .binding_by_name(*self.m.alias_chain(s).last().unwrap())?;
            self.m.valueset_of(self.m.binding(b).rhs).cloned()
        })
    }

    /// §06 "Provably normalized superpositions": weights under which a
    /// superposition of probability measures is one. The forms every engine
    /// must recognize: numeric literals whose values as written sum to
    /// exactly one; the complement pair `w`, `1 - w`; and `p[1]`, …, `p[n]`
    /// of one node `p` whose value set lies within `stdsimplex(n)`. An
    /// unweighted part has weight 1.
    pub(super) fn provably_normalized(&self, weights: &[Option<NodeId>]) -> bool {
        if let [a, b] = weights
            && let Some(w) = if self.complement(*a, *b) {
                *a
            } else if self.complement(*b, *a) {
                *b
            } else {
                None
            }
        {
            return self.within_unit(w);
        }
        // Exact decimal arithmetic on the literals, never a floating-point sum.
        let literal = |w: &Option<NodeId>| -> Option<String> {
            match w {
                None => Some("1".to_string()),
                Some(n) => match self.m.node(self.resolve(*n)) {
                    Node::Lit(flatppl_core::Scalar::Int(k)) => Some(k.to_string()),
                    Node::Lit(flatppl_core::Scalar::Real(x)) => Some(format!("{x}")),
                    _ => None,
                },
            }
        };
        if let Some(texts) = weights.iter().map(literal).collect::<Option<Vec<_>>>() {
            return decimal_sum_is_one(&texts);
        }
        // `p[1], …, p[n]` of one simplex vector.
        let Some(Some(first)) = weights.first() else {
            return false;
        };
        let base = |w: NodeId| -> Option<(NodeId, i64)> {
            let r = self.resolve(w);
            let c = self.call(r).filter(|_| self.head(r) == Some("get"))?;
            match (c.args.len(), self.m.node(c.args[1])) {
                (2, Node::Lit(flatppl_core::Scalar::Int(k))) => Some((c.args[0], *k)),
                _ => None,
            }
        };
        let Some((p, _)) = base(*first) else {
            return false;
        };
        let mut seen = vec![false; weights.len()];
        for w in weights {
            let Some((q, k)) = w.and_then(base) else {
                return false;
            };
            let Some(slot) = usize::try_from(k - 1).ok().and_then(|i| seen.get_mut(i)) else {
                return false;
            };
            if *slot || !self.m.structural_eq(p, q) {
                return false;
            }
            *slot = true;
        }
        matches!(self.value_set(p), Some(ValueSet::StdSimplex(flatppl_core::Dim::Static(n))) if n as usize == weights.len())
    }

    /// The two weights of a `Dirac` atom beside one law, where the split
    /// terms may read the law's weight as one minus the atom's.
    // Without `normalize` the mixture is a probability measure only by a §06
    // form, so the pair needs `w` within `unitinterval`. Under a written
    // `normalize`, Z = w + (1 - w) = 1 for any `w`: it is the identity.
    fn dirac_weights(&self, wl: Option<NodeId>, wd: Option<NodeId>, norm: bool) -> bool {
        wl.is_some()
            && wd.is_some()
            && (self.provably_normalized(&[wl, wd])
                || (norm && (self.complement(wl, wd) || self.complement(wd, wl))))
    }

    /// The log density of the scalar `y` under `law`, for one element of a
    /// weighted log likelihood. A `Dirac(c)` atom beside a law splits on
    /// `y == c`, from the same terms as the Dirac split: at the atom the
    /// mixture's mass there, elsewhere the law's weight and density.
    pub(super) fn element_lp(&mut self, law: NodeId, y: &str) -> Result<ElementLp> {
        if let Some((norm, mix)) = self.mixture(law) {
            if self.head(mix) == Some("superpose") {
                let parts = self.parts(mix)?;
                let split = match parts.as_slice() {
                    [Part::Dirac(wd, c), Part::Law(wl, l)] => Some((true, (*wd, *c), (*wl, *l))),
                    [Part::Law(wl, l), Part::Dirac(wd, c)] => Some((false, (*wd, *c), (*wl, *l))),
                    _ => None,
                };
                if let Some((dirac_first, (wd, atom), (wl, inner))) = split {
                    if !self.dirac_weights(wl, wd, norm) {
                        return Err(Refusal::not_yet(
                            Some(mix),
                            "a `Dirac` mixture with these weights",
                        ));
                    }
                    if self.shape(atom).is_none_or(|s| s.rank != 0) || !self.fixed_value(atom) {
                        return Err(Refusal::not_yet(
                            Some(mix),
                            "a `Dirac` atom of this form in a weighted log likelihood",
                        ));
                    }
                    let l = self.law_any(inner)?;
                    let c = self.at_module(|me| me.expr(atom))?.s;
                    // Beside a continuous law the atom's mass is its weight
                    // alone (Ruling 2).
                    let at = if !l.discrete {
                        self.log_weight(wd.unwrap())?
                    } else {
                        match self.atom_lp(mix, inner, &l, &c)? {
                            None => self.log_weight(wd.unwrap())?,
                            Some(lp) if dirac_first => {
                                format!("log_mix({}, 0, {lp})", self.real(wd.unwrap())?.s)
                            }
                            Some(lp) => format!("log_mix({}, {lp}, 0)", self.real(wl.unwrap())?.s),
                        }
                    };
                    let rest = format!(
                        "{} + {}",
                        self.log_weight(wl.unwrap())?,
                        self.law_lp(inner, &l, y, None)?
                    );
                    return Ok(ElementLp::Split(format!("{y} == {c}"), at, rest));
                }
            }
            let (pre, lp) = self.mixture_lp(mix, norm, y)?;
            if !pre.is_empty() {
                return Err(Refusal::not_yet(
                    Some(mix),
                    "a mixture that needs statements in a weighted log likelihood",
                ));
            }
            return Ok(ElementLp::One(lp));
        }
        let l = self.law_any(law)?;
        // Outside a mixture an unnormalized density is exact as it is.
        if l.unnormalized {
            self.integer_data(law, &l, y)?;
            return Ok(ElementLp::One(self.law_density(law, &l, y, None)?));
        }
        Ok(ElementLp::One(self.law_lp(law, &l, y, None)?))
    }

    /// The log density of `y` under the mixture `mix`, after `pre` statements.
    fn mixture_lp(&mut self, mix: NodeId, norm: bool, y: &str) -> Result<(Vec<String>, String)> {
        if self.head(mix) == Some("ksuperpose") {
            return self.kmixture_lp(mix, norm, y);
        }
        let parts = self.parts(mix)?;
        let mut lps = Vec::new();
        let mut weights = Vec::new();
        // The log mass of each unnormalized truncated component: it enters
        // the normalizer `Σ w_i M_i(S_i)` (§06 "Normalization and mass").
        let mut masses: Vec<Option<String>> = Vec::new();
        // Components on different supports (a Normal beside a Gamma): at a
        // point outside one component's support its density is 0, where
        // Stan's lpdf throws, so a component whose effective support is not
        // the union of all of them tests its own (`MixSupport`).
        let mut sides = Vec::new();
        for p in &parts {
            if let Part::Law(_, law) = *p {
                let l = self.law_any(law)?;
                sides.push(MixSupport::of(&l));
            }
        }
        let union = MixSupport::union(&sides);
        let differ = sides
            .iter()
            .any(|s| s.guard != sides[0].guard || s.ends != sides[0].ends);
        let mut k = 0;
        for p in &parts {
            match *p {
                Part::Law(w, law) => {
                    if norm {
                        self.truncated_component_has_mass(law)?;
                    }
                    let mut l = self.law_any(law)?;
                    let support = l.support.take();
                    let mut lp = self.law_lp(law, &l, y, None)?;
                    if !l.total_support
                        && !l.unchecked_support
                        && l.trunc.is_none()
                        && support.is_none()
                    {
                        let array = self.decls.get(y).is_some_and(|d| d.rank() > 0);
                        let (lo, hi) = sides[k].narrower(&union);
                        if let Some(out) =
                            Self::outside_test(y, array, lo.as_deref(), hi.as_deref())
                        {
                            lp = format!("({out} ? negative_infinity() : {lp})");
                        }
                        // An end that depends on the law's arguments is the
                        // row guard's (`binomial` at y > n); `law_lp` prints
                        // it already where it varies per component.
                        if differ
                            && let Some((whole, _)) = &l.guard
                            && !whole.contains(COMP)
                        {
                            let yy = if array {
                                format!("max({y})")
                            } else {
                                y.to_string()
                            };
                            lp = format!(
                                "({} ? negative_infinity() : {lp})",
                                whole.replace("{y}", &yy)
                            );
                        }
                    }
                    k += 1;
                    let mut mass = None;
                    if let Some((lo, hi)) = support {
                        if let Some(out) = Self::outside_test(
                            y,
                            false,
                            l.within(lo.as_deref(), hi.as_deref()).0.as_deref(),
                            l.within(lo.as_deref(), hi.as_deref()).1.as_deref(),
                        ) {
                            lp = format!("({out} ? negative_infinity() : {lp})");
                        }
                        mass = Some(self.interval_lmass(&l, lo.as_deref(), hi.as_deref()));
                    }
                    lps.push(lp);
                    weights.push(w);
                    masses.push(mass);
                }
                Part::Dirac(..) => {
                    return Err(Refusal::not_yet(
                        Some(mix),
                        "a `Dirac` component beside this variate",
                    ));
                }
            }
        }
        if let [a, b] = weights[..]
            && self.complement(a, b)
            && masses.iter().all(Option::is_none)
        {
            let w = self.real(a.unwrap())?;
            return Ok((vec![], format!("log_mix({}, {}, {})", w.s, lps[0], lps[1])));
        }
        let terms: Vec<String> = weights
            .iter()
            .zip(&lps)
            .map(|(w, lp)| -> Result<String> {
                Ok(match w {
                    Some(w) => format!("{} + {lp}", self.log_weight(*w)?),
                    None => lp.clone(),
                })
            })
            .collect::<Result<_>>()?;
        let mut lp = format!("log_sum_exp([{}]')", terms.join(", "));
        // An unweighted part has mass 1, so `normalize` divides by the
        // weight sum even when no part is weighted.
        if norm && masses.iter().any(Option::is_some) {
            let zs: Vec<String> = weights
                .iter()
                .zip(&masses)
                .map(|(w, m)| -> Result<String> {
                    let lw = match w {
                        Some(w) => Some(self.log_weight(*w)?),
                        None => None,
                    };
                    Ok(match (lw, m) {
                        (Some(lw), Some(m)) => format!("{lw} + {m}"),
                        (Some(lw), None) => lw,
                        (None, Some(m)) => m.clone(),
                        (None, None) => "0".to_string(),
                    })
                })
                .collect::<Result<_>>()?;
            lp = format!("{lp} - log_sum_exp([{}]')", zs.join(", "));
        } else if norm && weights.iter().all(Option::is_none) {
            lp = format!("{lp} - log({})", weights.len());
        } else if norm && !self.provably_normalized(&weights) {
            let ws: Vec<String> = weights
                .iter()
                .map(|w| -> Result<String> {
                    Ok(match w {
                        Some(w) => self.real(*w)?.at(super::P_ADD),
                        None => "1".to_string(),
                    })
                })
                .collect::<Result<_>>()?;
            lp = format!("{lp} - log({})", ws.join(" + "));
        }
        Ok((vec![], lp))
    }

    /// `ksuperpose(kernel, w, params…)`: the components `kernel(params_k)`,
    /// their log densities filled into a local vector.
    fn kmixture_lp(&mut self, mix: NodeId, norm: bool, y: &str) -> Result<(Vec<String>, String)> {
        let c = self.call(mix).unwrap();
        if c.args.len() < 2 {
            return Err(Refusal::not_yet(Some(mix), "ksuperpose with no weights"));
        }
        let (kernel, weights) = (c.args[0], c.args[1]);
        let wr = self.resolve(weights);
        if self.head(wr) == Some("vector") {
            for &w in self.call(wr).unwrap().args.iter() {
                self.nonnegative_weight(w)?;
            }
        }
        let params: Vec<NodeId> = c.args[2..].to_vec();
        let named: Vec<(String, NodeId)> = c
            .named
            .iter()
            .map(|na| (self.name(na.name).to_string(), na.value))
            .collect();
        let k = self.fresh_loop();
        let out = self.kernel_component(mix, kernel, &params, &named, y);
        let lp = match out {
            Ok(lp) => lp.replace(COMP, &k),
            Err(e) => {
                self.used.remove(&k);
                return Err(e);
            }
        };
        let n = self
            .sym_extent(weights, 0, 0)
            .map(|l| l.print())
            .or_else(|| {
                self.expr(weights)
                    .ok()
                    .map(|e| format!("num_elements({})", e.s))
            })
            .unwrap();
        let lps = self.fresh("lps");
        self.used.remove(&k);
        let pre = vec![
            format!("vector[{n}] {lps};"),
            format!("for ({k} in 1:{n}) {{\n  {lps}[{k}] = {lp};\n}}"),
        ];
        let w = self.real(weights)?;
        // §06 "Provably normalized superpositions": a weights vector whose
        // value set lies within `stdsimplex(n)`, passed whole.
        let simplex = matches!(self.value_set(weights), Some(ValueSet::StdSimplex(_)));
        let lp = if norm && simplex {
            format!("log_mix({}, {lps})", w.s)
        } else if norm {
            format!("log_sum_exp(log({0}) + {lps}) - log(sum({0}))", w.s)
        } else {
            format!("log_sum_exp(log({}) + {lps})", w.s)
        };
        Ok((pre, lp))
    }

    /// The log density of `y` under `kernel(params[k]…)`, with [`COMP`] for k.
    fn kernel_component(
        &mut self,
        at: NodeId,
        kernel: NodeId,
        params: &[NodeId],
        named: &[(String, NodeId)],
        y: &str,
    ) -> Result<String> {
        // Each argument with an axis is per component; a scalar is shared.
        let per = |me: &mut Self, a: NodeId| -> Result<String> {
            let e = me.expr(a)?;
            Ok(if me.shape(a).is_some_and(|s| s.rank >= 1) {
                // Stan indexes a name or a parenthesized expression, not a
                // literal (`[3.0, 4.0]'[k]` is a syntax error).
                if super::structure::is_identifier(&e.s) {
                    format!("{}[{COMP}]", e.s)
                } else {
                    format!("({})[{COMP}]", e.s)
                }
            } else {
                e.s
            })
        };
        if let Node::Const(f) = self.m.node(kernel) {
            // A law constructor: its canonical parameter order.
            let name = self.name(*f).to_string();
            let order = flatppl_infer::distribution_param_names(&name).unwrap_or_default();
            let mut args: Vec<Option<NodeId>> = params.iter().map(|&a| Some(a)).collect();
            for (key, v) in named {
                let pos = order.iter().position(|p| p == key).ok_or_else(|| {
                    Refusal::not_yet(Some(at), format!("{name}({key} = …) in ksuperpose"))
                })?;
                if args.len() <= pos {
                    args.resize(pos + 1, None);
                }
                args[pos] = Some(*v);
            }
            let args: Vec<NodeId> = args.into_iter().collect::<Option<_>>().ok_or_else(|| {
                Refusal::not_yet(Some(at), format!("{name} with a missing argument"))
            })?;
            let law = self.table_law_per(at, &name, &args)?;
            return self.law_lp(at, &law, y, Some(COMP));
        }
        // A user kernel: its law body per component.
        let fun = self
            .user_callee(kernel)
            .ok_or_else(|| Refusal::not_yet(Some(at), "ksuperpose of this kernel"))?;
        let (body, entries) = self.callable_parts(fun)?;
        // §04 "Auto-splatting" of a lone record or table, then keyword
        // arguments bound to the kernel's inputs by name (§06 `ksuperpose`:
        // a table's columns are the kernel's inputs).
        let mut named = named.to_vec();
        let mut params = params.to_vec();
        if let [a] = params[..]
            && named.is_empty()
            && entries.len() >= 2
            && let Some(fields) = self.splat_fields(a)?
        {
            named = fields;
            params.clear();
        }
        if !named.is_empty() {
            if !params.is_empty() || named.len() != entries.len() {
                return Err(Refusal::not_yet(
                    Some(at),
                    "ksuperpose of a user kernel with positional and keyword arguments",
                ));
            }
            params = entries
                .iter()
                .map(|(n, _)| {
                    named
                        .iter()
                        .find(|(k, _)| k == self.name(*n))
                        .map(|(_, v)| *v)
                        .ok_or_else(|| {
                            Refusal::not_yet(
                                Some(at),
                                format!(
                                    "a ksuperpose without the kernel input `{}`",
                                    self.name(*n)
                                ),
                            )
                        })
                })
                .collect::<Result<_>>()?;
        }
        if entries.len() != params.len() {
            return Err(Refusal::not_yet(Some(at), "ksuperpose with another arity"));
        }
        let mut locals = HashMap::new();
        let mut tys = Vec::new();
        for ((_, r), &a) in entries.iter().zip(&params) {
            let t = self
                .ty(a)
                .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?;
            tys.push(match t {
                Type::Array { shape, elem } if shape.len() == 1 => *elem,
                Type::Array { .. } => {
                    return Err(Refusal::not_yet(
                        Some(a),
                        "ksuperpose over a matrix argument",
                    ));
                }
                t => t,
            });
            locals.insert(r.name, per(self, a)?);
        }
        let scoped = self.instantiate(at, body, &entries, &tys)?;
        let mut ctx = FnCtx {
            locals,
            scoped,
            inline_only: true,
            ..Default::default()
        };
        self.mark_boundary(&mut ctx, at, body, &entries)?;
        let outer = self.fctx.replace(ctx);
        // An iid body over the whole variate: the component's density is the
        // vectorized density, the sum over its elements.
        let mut law_node = self.resolve(body);
        if self.head(law_node) == Some("iid") {
            law_node = self.resolve(self.call(law_node).unwrap().args[0]);
        }
        let out = self
            .law_any(law_node)
            .and_then(|law| self.law_lp(at, &law, y, None));
        let ctx = std::mem::replace(&mut self.fctx, outer);
        if ctx.is_some_and(|c| !c.lines.is_empty()) {
            return Err(Refusal::not_yet(
                Some(at),
                "a ksuperpose kernel that needs statements",
            ));
        }
        out
    }

    /// A normalized mixture needs each unnormalized truncated component's
    /// mass, which the emitter takes from the law's Stan CDF. The laws without
    /// one that reach here (`Poisson(exp(a))`, `Categorical`) still have a
    /// closed-form mass, so this is not yet, not never.
    fn truncated_component_has_mass(&mut self, law: NodeId) -> Result<()> {
        let r = self.resolve(law);
        if self.head(r) != Some("truncate") {
            return Ok(());
        }
        let (inner, set) = {
            let c = self.call(r).unwrap();
            (c.args[0], c.args[1])
        };
        let base = self.law_any(inner)?;
        // Stan's integer CDFs take integer ends (the F3 refusal, ruling R4).
        if base.discrete && !self.integer_ends(set) {
            return Err(Refusal::not_yet(
                Some(law),
                "a discrete law truncated at a non-integer end",
            ));
        }
        if base.lcdf {
            return Ok(());
        }
        Err(Refusal::not_yet(
            Some(law),
            "an unnormalized truncated mixture component of a law with no Stan CDF",
        ))
    }

    /// The law's log density at a Dirac atom `c` beside it, or `None` when
    /// `c` is outside the law's support: the mixture's density there is the
    /// atom's weight alone, and `lpmf(c | …)` would throw. The support is the
    /// row's variate bounds and its guard. A literal atom is decided here; any
    /// other is tested where it is scored.
    pub(super) fn atom_lp(
        &mut self,
        mix: NodeId,
        law: NodeId,
        l: &Law,
        c: &str,
    ) -> Result<Option<String>> {
        let num = |s: &str| s.trim().parse::<f64>().ok();
        let cv = num(c);
        if let (Some(x), Some((lo, hi))) = (cv, &l.trunc) {
            let below = lo.as_deref().and_then(num).is_some_and(|b| x < b);
            let above = hi.as_deref().and_then(num).is_some_and(|b| x > b);
            if below || above {
                return Ok(None);
            }
        }
        let lp = self.law_lp(law, l, c, None)?;
        let (lo, hi) = (l.bounds.0.as_deref(), l.bounds.1.as_deref());
        let mut tests = Vec::new();
        for (bound, below) in [(lo, true), (hi, false)] {
            let Some(b) = bound else { continue };
            let outside = |x: f64, y: f64| if below { x < y } else { x > y };
            match (cv, num(b)) {
                (Some(x), Some(y)) if outside(x, y) => return Ok(None),
                (Some(_), Some(_)) => {}
                // A literal atom at the lower bound is inside for any valid
                // upper bound.
                (Some(x), None) if !below && lo.and_then(num) == Some(x) => {}
                (_, Some(_)) => tests.push(format!("{c} {} {b}", if below { "<" } else { ">" })),
                (_, None) => {
                    return Err(Refusal::not_yet(
                        Some(mix),
                        "a `Dirac` atom beside a law whose support depends on its arguments",
                    ));
                }
            }
        }
        if let Some((whole, _)) = &l.guard
            && !(cv.is_some() && cv == lo.and_then(num))
        {
            tests.push(format!("({})", whole.replace("{y}", c)));
        }
        Ok(Some(if tests.is_empty() {
            lp
        } else {
            format!("({} ? negative_infinity() : {lp})", tests.join(" || "))
        }))
    }

    /// `D_lpdf(y | args)` of a component, with its support guard. A
    /// normalized truncation divides by its mass on `[lo, hi]` and is zero
    /// outside it (§06 "Normalization and mass"). An unnormalized one is
    /// taken apart by `mixture_lp`, which knows the mixture's normalizer.
    fn law_lp(&mut self, at: NodeId, law: &Law, y: &str, index: Option<&str>) -> Result<String> {
        if law.support.is_some() {
            return Err(Refusal::not_yet(
                Some(at),
                "an unnormalized truncated mixture component in this form",
            ));
        }
        if !law.extra.is_empty() {
            return Err(Refusal::not_yet(
                Some(at),
                "a mixture component with extra terms",
            ));
        }
        let array = self.decls.get(y).is_some_and(|d| d.rank() > 0);
        if law.trunc.is_some() && array {
            return Err(Refusal::not_yet(
                Some(at),
                "a truncated mixture component over an array variate",
            ));
        }
        if law.declared {
            return Err(Refusal::not_yet(
                Some(at),
                "an ordered restriction as a mixture component",
            ));
        }
        self.integer_data(at, law, y)?;
        // A mixture component is a probability measure; an unnormalized row
        // has no closed-form share of it.
        if law.unnormalized {
            return Err(Refusal::not_yet(
                Some(at),
                "an unnormalized law as a mixture component",
            ));
        }
        let lp = self.law_density(at, law, y, index)?;
        // A guard matters only where the support varies by component
        // (binomial `n` per component): Stan's lpmf throws outside it.
        let guard = law
            .guard
            .as_ref()
            .filter(|(whole, _)| index.is_some() || whole.contains(COMP))
            .map(|(whole, each)| {
                let t = if index.is_some() { each } else { whole };
                // An array variate is outside the support when its largest value is.
                let yy = if self.decls.get(y).is_some_and(|d| d.rank() > 0) {
                    format!("max({y})")
                } else {
                    y.to_string()
                };
                t.replace("{y}", &yy)
            });
        let lp = match &law.trunc {
            Some((lo, hi)) => {
                let mass = self.interval_lmass(law, lo.as_deref(), hi.as_deref());
                let mass = match index {
                    Some(i) => mass.replace(super::structure::LOOP, i),
                    None => mass,
                };
                match Self::outside_test(
                    y,
                    false,
                    law.within(lo.as_deref(), hi.as_deref()).0.as_deref(),
                    law.within(lo.as_deref(), hi.as_deref()).1.as_deref(),
                ) {
                    Some(out) => format!("({out} ? negative_infinity() : {lp} - {mass})"),
                    None => lp,
                }
            }
            None => lp,
        };
        if law
            .support_type
            .is_some_and(crate::stan_type::square_support)
        {
            return Err(Refusal::not_yet(
                None,
                "a mixture component on a constrained matrix set",
            ));
        }
        // An `unchecked_support` row (von Mises) does not test its own
        // support: outside it the component's density is zero.
        let outside = if law.unchecked_support {
            Self::outside_test(
                y,
                self.decls.get(y).is_some_and(|d| d.rank() > 0),
                law.bounds.0.as_deref(),
                law.bounds.1.as_deref(),
            )
        } else {
            None
        };
        let lp = match outside {
            Some(out) => format!("({out} ? negative_infinity() : {lp})"),
            None => lp,
        };
        Ok(match guard {
            Some(g) => {
                let g = match index {
                    Some(i) => g.replace(super::structure::LOOP, i),
                    None => g,
                };
                format!("({g} ? negative_infinity() : {lp})")
            }
            None => lp,
        })
    }

    /// A Dirac atom beside one discrete law, over observed data: the
    /// positions at the atom and the rest, found once in `transformed data`.
    /// At the atom the mixture is `log_mix(w, lp(atom), 0)`, a loop over the
    /// atom's positions when the law's arguments vary per element. Elsewhere
    /// the atom contributes nothing: `log(w)` plus one vectorized density.
    fn dirac_split(&mut self, lhs: &str, mix: NodeId, norm: bool) -> Result<Option<Vec<String>>> {
        if self.head(mix) != Some("superpose") {
            return Ok(None);
        }
        let parts = self.parts(mix)?;
        let (dirac_first, (wd, atom), (wl, law)) = match parts.as_slice() {
            [Part::Dirac(wd, c), Part::Law(wl, l)] => (true, (*wd, *c), (*wl, *l)),
            [Part::Law(wl, l), Part::Dirac(wd, c)] => (false, (*wd, *c), (*wl, *l)),
            _ => return Ok(None),
        };
        // A chain's law is per row: only the row split takes it.
        if self.head(self.resolve(law)) == Some("markovchain") {
            if !(self.shape(atom).is_some_and(|s| s.rank == 1) || self.row_fill(atom)) {
                return Err(Refusal::not_yet(Some(mix), "a `Dirac` atom of this shape"));
            }
            return self
                .dirac_split_rows(lhs, mix, norm, dirac_first, (wd, atom), (wl, law))
                .map(Some);
        }
        // An iid mixture per site beside a whole-row atom.
        if let Some(lines) =
            self.dirac_split_nested(lhs, mix, norm, dirac_first, (wd, atom), (wl, law))?
        {
            return Ok(Some(lines));
        }
        let l = self.law_any(law)?;
        if l.declared {
            return Err(Refusal::not_yet(
                Some(mix),
                "an ordered restriction as a mixture component",
            ));
        }
        if l.support.is_some() {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom beside an unnormalized truncated law",
            ));
        }
        // Beside a continuous law the mixture has a density with respect to
        // `Dirac(c) + Lebesgue` (Ruling 2): only observed data, whose values
        // at the atom are known, can be scored. A parameter would need a
        // density with respect to Lebesgue measure alone, and it has none.
        if !l.discrete && !self.is_data(lhs) {
            return Err(Refusal::new(
                RefusalKind::MixedReference,
                Some(mix),
                "superpose",
                "a `Dirac` atom beside a continuous law has no density with respect to Lebesgue \
                 measure, so it cannot be the law of a parameter (§06)",
            ));
        }
        if !l.discrete {
            let l = Self::normalized(mix, l)?;
            return self
                .hurdle_split(lhs, mix, norm, (wd, atom), (wl, law), &l)
                .map(Some);
        }
        if self.shape(atom).is_some_and(|s| s.rank == 1) || self.row_fill(atom) {
            return self
                .dirac_split_rows(lhs, mix, norm, dirac_first, (wd, atom), (wl, law))
                .map(Some);
        }
        if self.shape(atom).is_none_or(|s| s.rank != 0) {
            return Err(Refusal::not_yet(Some(mix), "a `Dirac` atom of this shape"));
        }
        // Observed data only: the positions are found once.
        let data = self.is_data(lhs);
        if !data
            || self
                .decls
                .get(lhs)
                .is_none_or(|t| !(t.is_int() && t.rank() == 1))
        {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` mixture over this variate",
            ));
        }
        if !self.dirac_weights(wl, wd, norm) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` mixture with these weights",
            ));
        }
        if !self.fixed_value(atom) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom whose value is not data",
            ));
        }
        let c = self.at_module(|me| me.expr(atom))?;
        let n = self.extent_text(lhs);
        let count = self.fresh(&format!("{lhs}_n_atom"));
        let at_idx = self.fresh(&format!("{lhs}_atom"));
        let rest = self.fresh(&format!("{lhs}_rest"));
        let i = self.fresh_loop();
        let (a, b) = (self.fresh_local("a"), self.fresh_local("b"));
        self.tdata.push(format!(
            "int {count} = 0;\nfor ({i} in 1:{n}) {{\n  {count} += {lhs}[{i}] == {c};\n}}",
            c = c.s
        ));
        self.tdata.push(format!("array[{count}] int {at_idx};"));
        self.tdata.push(format!("array[{n} - {count}] int {rest};"));
        self.tdata.push(format!(
            "{{\n  int {a} = 1;\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    if ({lhs}[{i}] == {c}) {{\n      {at_idx}[{a}] = {i};\n      {a} += 1;\n    }} else {{\n      {rest}[{b}] = {i};\n      {b} += 1;\n    }}\n  }}\n}}",
            c = c.s
        ));
        for v in [&i, &a, &b] {
            self.used.remove(v);
        }
        let loop_ = super::structure::LOOP;
        let wlx = self.real(wl.unwrap())?.s;
        let mix_at = match self.atom_lp(mix, law, &l, &c.s)? {
            None => self.log_weight(wd.unwrap())?,
            Some(lp_atom) if dirac_first => {
                format!("log_mix({}, 0, {lp_atom})", self.real(wd.unwrap())?.s)
            }
            Some(lp_atom) => format!("log_mix({wlx}, {lp_atom}, 0)"),
        };
        let mut out = Vec::new();
        if mix_at.contains(loop_) {
            // Per-element arguments: no whole-array form, one term per position.
            let k = self.fresh_loop();
            out.push(format!(
                "for ({k} in 1:{count}) {{\n  target += {};\n}}",
                mix_at.replace(loop_, &format!("{at_idx}[{k}]"))
            ));
            self.used.remove(&k);
        } else {
            out.push(format!("target += {count} * {mix_at};"));
        }
        let lp_rest = self.rest_lp(mix, law, &l, lhs, &rest, &n, &count)?;
        let log_l = self.log_weight(wl.unwrap())?;
        let log_w = if log_l.contains(loop_) {
            format!("sum({})", log_l.replace(loop_, &rest))
        } else {
            format!("({n} - {count}) * {log_l}")
        };
        out.push(format!("target += {log_w} + {lp_rest};"));
        Ok(Some(out))
    }

    /// The vectorized log density of `lhs[rest]` under `l`. A normalized
    /// truncation subtracts each element's log mass (one `_lccdf` or `_lcdf`
    /// call sums them when the arguments vary per element) and is `-inf`
    /// when an element lies outside `[lo, hi]`.
    #[allow(clippy::too_many_arguments)]
    fn rest_lp(
        &mut self,
        mix: NodeId,
        law: NodeId,
        l: &Law,
        lhs: &str,
        rest: &str,
        n: &str,
        count: &str,
    ) -> Result<String> {
        let loop_ = super::structure::LOOP;
        let y = format!("{lhs}[{rest}]");
        let lp = self.law_density(law, l, &y, None)?.replace(loop_, rest);
        let Some((lo, hi)) = &l.trunc else {
            return Ok(lp);
        };
        let mass = self.interval_lmass(l, lo.as_deref(), hi.as_deref());
        let mass = if !mass.contains(loop_) {
            format!("({n} - {count}) * {mass}")
        } else if lo.is_some() && hi.is_some() {
            // `log_diff_exp` of two summed `_lcdf` calls is not the sum of
            // the per-element masses.
            return Err(Refusal::not_yet(
                Some(mix),
                "a two-sided truncation beside a `Dirac` atom with per-element arguments",
            ));
        } else {
            mass.replace(loop_, rest)
        };
        // `min` of an empty integer array raises: no rest, nothing outside.
        Ok(
            match Self::outside_test(
                &y,
                true,
                l.within(lo.as_deref(), hi.as_deref()).0.as_deref(),
                l.within(lo.as_deref(), hi.as_deref()).1.as_deref(),
            ) {
                Some(out) => {
                    format!("({n} - {count} > 0 && ({out}) ? negative_infinity() : {lp} - {mass})")
                }
                None => format!("{lp} - {mass}"),
            },
        )
    }

    /// A `Dirac(c)` atom beside a continuous law over observed data (the
    /// hurdle form, Ruling 2): with respect to `Dirac(c) + Lebesgue`, the
    /// density is `w` at `c` and `(1 - w)·m(y)` elsewhere, since the law
    /// gives the single point `c` no mass. The positions at `c` are counted
    /// once in `transformed data`.
    fn hurdle_split(
        &mut self,
        lhs: &str,
        mix: NodeId,
        norm: bool,
        (wd, atom): (Option<NodeId>, NodeId),
        (wl, law): (Option<NodeId>, NodeId),
        l: &Law,
    ) -> Result<Vec<String>> {
        if self.shape(atom).is_none_or(|s| s.rank != 0)
            || self
                .decls
                .get(lhs)
                .is_none_or(|t| t.is_int() || t.rank() != 1)
        {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom beside a continuous law over this variate",
            ));
        }
        if !self.dirac_weights(wl, wd, norm) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` mixture with these weights",
            ));
        }
        if !self.fixed_value(atom) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom whose value is not data",
            ));
        }
        let c = self.at_module(|me| me.expr(atom))?.s;
        let n = self.extent_text(lhs);
        let count = self.fresh(&format!("{lhs}_n_atom"));
        let rest = self.fresh(&format!("{lhs}_rest"));
        let i = self.fresh_loop();
        let b = self.fresh_local("b");
        self.tdata.push(format!(
            "int {count} = 0;\nfor ({i} in 1:{n}) {{\n  {count} += {lhs}[{i}] == {c};\n}}"
        ));
        self.tdata.push(format!("array[{n} - {count}] int {rest};"));
        self.tdata.push(format!(
            "{{\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    if ({lhs}[{i}] != {c}) {{\n      {rest}[{b}] = {i};\n      {b} += 1;\n    }}\n  }}\n}}"
        ));
        for v in [&i, &b] {
            self.used.remove(v);
        }
        let loop_ = super::structure::LOOP;
        let log_d = self.log_weight(wd.unwrap())?;
        let log_l = self.log_weight(wl.unwrap())?;
        let mut out = Vec::new();
        if log_d.contains(loop_) {
            // A weight per element: the atom positions are needed by index.
            let at_idx = self.fresh(&format!("{lhs}_atom"));
            let (i, a) = (self.fresh_loop(), self.fresh_local("a"));
            self.tdata.push(format!("array[{count}] int {at_idx};"));
            self.tdata.push(format!(
                "{{\n  int {a} = 1;\n  for ({i} in 1:{n}) {{\n    if ({lhs}[{i}] == {c}) {{\n      {at_idx}[{a}] = {i};\n      {a} += 1;\n    }}\n  }}\n}}"
            ));
            for v in [&i, &a] {
                self.used.remove(v);
            }
            out.push(format!("target += sum({});", log_d.replace(loop_, &at_idx)));
        } else {
            out.push(format!("target += {count} * {log_d};"));
        }
        let lp_rest = self.rest_lp(mix, law, l, lhs, &rest, &n, &count)?;
        let log_w = if log_l.contains(loop_) {
            format!("sum({})", log_l.replace(loop_, &rest))
        } else {
            format!("({n} - {count}) * {log_l}")
        };
        let statement = format!("target += {log_w} + {lp_rest};");
        // Off the atom, data outside the law's support scores -inf (spec);
        // Stan's `_lpdf` would raise there.
        let (lo, hi) = self.support_ends(mix, l)?;
        let y = format!("{lhs}[{rest}]");
        match Self::outside_test(&y, true, lo.as_deref(), hi.as_deref()) {
            Some(test) => {
                let body: String = statement.lines().map(|x| format!("  {x}\n")).collect();
                out.push(format!(
                    "if ({n} - {count} > 0 && ({test})) {{\n  target += negative_infinity();\n}} else {{\n{body}}}"
                ));
            }
            None => out.push(statement),
        }
        Ok(out)
    }

    /// The literal ends of a law's support: its variate bounds, or the
    /// argument its row names as `floor`.
    fn support_ends(&self, mix: NodeId, l: &Law) -> Result<(Option<String>, Option<String>)> {
        let num = |b: &Option<String>| b.clone().filter(|x| x.parse::<f64>().is_ok());
        let lo = match l.floor_arg {
            Some(k) => {
                let open = l.call.find('(').unwrap_or(l.call.len());
                let args = l.call[open..]
                    .strip_prefix('(')
                    .and_then(|x| x.strip_suffix(')'))
                    .map(split_args)
                    .unwrap_or_default();
                match args.get(k) {
                    Some(a) if !a.contains(super::structure::LOOP) => Some(a.clone()),
                    _ => {
                        return Err(Refusal::not_yet(
                            Some(mix),
                            "a `Dirac` atom beside a law whose support varies per element",
                        ));
                    }
                }
            }
            None => num(&l.bounds.0),
        };
        Ok((lo, num(&l.bounds.1)))
    }
}

/// The top-level comma-separated arguments of a call's argument text.
/// A mixture component's support for the outside test: its literal ends
/// (a truncation's ends within the law's own, `Law::within`), `None` for an
/// unbounded end, and its row guard, the argument-dependent end.
struct MixSupport {
    /// Each end: `None` unbounded, `Some(None)` not a literal, `Some(Some(x))`.
    ends: (Option<Option<f64>>, Option<Option<f64>>),
    guard: Option<String>,
}

impl MixSupport {
    fn of(l: &Law) -> Self {
        let (lo, hi) = match l.trunc.as_ref().or(l.support.as_ref()) {
            Some((lo, hi)) => l.within(lo.as_deref(), hi.as_deref()),
            None => l.bounds.clone(),
        };
        // A template end (`{resonance}`) is not a literal; the row guard or
        // `total_support` covers it.
        let end = |e: Option<String>| e.map(|t| t.parse::<f64>().ok());
        MixSupport {
            ends: (end(lo), end(hi)),
            guard: l.guard.as_ref().map(|g| g.0.clone()),
        }
    }

    /// The union's ends: unbounded where any component is, unknown where any
    /// end is not a literal, else the widest literal.
    fn union(all: &[MixSupport]) -> (Option<Option<f64>>, Option<Option<f64>>) {
        let side = |pick: &dyn Fn(&MixSupport) -> Option<Option<f64>>,
                    wider: fn(f64, f64) -> f64| {
            let mut out = Some(Some(f64::NAN));
            for s in all {
                out = match (out, pick(s)) {
                    (None, _) | (_, None) => None,
                    (Some(None), _) | (_, Some(None)) => Some(None),
                    (Some(Some(a)), Some(Some(b))) => {
                        Some(Some(if a.is_nan() { b } else { wider(a, b) }))
                    }
                };
            }
            out
        };
        (side(&|s| s.ends.0, f64::min), side(&|s| s.ends.1, f64::max))
    }

    /// This component's literal ends that lie inside the union's: each one
    /// needs its test.
    fn narrower(
        &self,
        union: &(Option<Option<f64>>, Option<Option<f64>>),
    ) -> (Option<String>, Option<String>) {
        let side = |mine: Option<Option<f64>>,
                    theirs: Option<Option<f64>>,
                    inside: fn(f64, f64) -> bool| {
            match (mine, theirs) {
                (Some(Some(x)), None | Some(None)) => Some(number(x)),
                (Some(Some(x)), Some(Some(u))) if inside(x, u) => Some(number(x)),
                _ => None,
            }
        };
        (
            side(self.ends.0, union.0, |x, u| x > u),
            side(self.ends.1, union.1, |x, u| x < u),
        )
    }
}

pub(super) fn split_args(inner: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, ch) in inner.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !inner.trim().is_empty() {
        parts.push(inner[start..].trim().to_string());
    }
    parts
}

/// `true` when the decimal literals `texts` (`0.3`, `7`, `0.25`) sum to
/// exactly one, by integer arithmetic at their common scale. Each text is
/// the literal's shortest round-trip decimal, which is the literal as
/// written for every literal a user writes at that precision.
fn decimal_sum_is_one(texts: &[String]) -> bool {
    let parts: Option<Vec<(i128, u32)>> = texts
        .iter()
        .map(|t| {
            let (int, frac) = t.split_once('.').unwrap_or((t, ""));
            if frac.len() > 30 || int.starts_with('-') {
                return None;
            }
            let digits = format!("{int}{frac}");
            Some((digits.parse::<i128>().ok()?, frac.len() as u32))
        })
        .collect();
    let Some(parts) = parts else {
        return false;
    };
    let scale = parts.iter().map(|p| p.1).max().unwrap_or(0);
    let mut sum: i128 = 0;
    for (v, s) in parts {
        let Some(term) = 10i128.checked_pow(scale - s).and_then(|f| v.checked_mul(f)) else {
            return false;
        };
        sum += term;
    }
    10i128.checked_pow(scale) == Some(sum)
}

/// Statements `pre` then `last`, in a block when `pre` declares locals.
fn block(pre: Vec<String>, last: String) -> String {
    if pre.is_empty() {
        return last;
    }
    let mut out = String::from("{\n");
    for l in pre.iter().chain(std::iter::once(&last)) {
        for part in l.lines() {
            out.push_str("  ");
            out.push_str(part);
            out.push('\n');
        }
    }
    out.push('}');
    out
}
