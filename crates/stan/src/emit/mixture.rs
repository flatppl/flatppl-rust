//! Mixtures: `superpose` and `ksuperpose`, normalized or not (spec §06
//! "Density of composed measures", "Additive superposition").
//!
//! A mixture's log density is `logsumexp_i(log w_i + lp_i)`. Its normalizer
//! is `Σ w_i` because every component is a probability measure. Stan cannot
//! vectorize a mixture over observations (User's Guide "Vectorizing
//! mixtures"), so an iid variate takes one `target +=` per element. The
//! provably normalized forms of §06 use `log_mix`: the complement pair
//! `weighted(w, A), weighted(1 - w, B)` and a simplex `ksuperpose` weight.
//! A `Dirac` component beside a discrete law splits observed data in
//! `transformed data` into the values at the atom and the rest (User's Guide
//! "Zero-inflated and hurdle models").

use super::Emitter;
use super::functions::FnCtx;
use super::structure::Law;
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Node, NodeId, Type};
use std::collections::HashMap;

mod nested;

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
        if !(norm && (self.complement(wl, wd) || self.complement(wd, wl))) {
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
        out.push(format!(
            "target += ({n} - {count}) * log({wlx}) + {lp_rest};"
        ));
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

    fn fresh_loop(&mut self) -> String {
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

    /// `weighted(w, A), weighted(1 - w, B)`: the second weight is `1 - w`.
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

    /// The log density of `y` under the mixture `mix`, after `pre` statements.
    fn mixture_lp(&mut self, mix: NodeId, norm: bool, y: &str) -> Result<(Vec<String>, String)> {
        if self.head(mix) == Some("ksuperpose") {
            return self.kmixture_lp(mix, norm, y);
        }
        let parts = self.parts(mix)?;
        let mut lps = Vec::new();
        let mut weights = Vec::new();
        for p in &parts {
            match *p {
                Part::Law(w, law) => {
                    lps.push(self.component_lp(law, y, None)?);
                    weights.push(w);
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
        {
            let w = self.real(a.unwrap())?;
            return Ok((vec![], format!("log_mix({}, {}, {})", w.s, lps[0], lps[1])));
        }
        let terms: Vec<String> = weights
            .iter()
            .zip(&lps)
            .map(|(w, lp)| -> Result<String> {
                Ok(match w {
                    Some(w) => format!("log({}) + {lp}", self.real(*w)?.s),
                    None => lp.clone(),
                })
            })
            .collect::<Result<_>>()?;
        let mut lp = format!("log_sum_exp([{}]')", terms.join(", "));
        // An unweighted part has mass 1, so `normalize` divides by the
        // weight sum even when no part is weighted.
        if norm && weights.iter().all(Option::is_none) {
            lp = format!("{lp} - log({})", weights.len());
        } else if norm {
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
        let simplex = self
            .self_ref(weights)
            .and_then(|s| self.env.get(&s).cloned())
            .and_then(|n| self.decls.get(&n))
            .is_some_and(|t| t.base == "simplex");
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
        if !named.is_empty() {
            return Err(Refusal::not_yet(
                Some(at),
                "ksuperpose of a user kernel with keyword arguments",
            ));
        }
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != params.len() {
            return Err(Refusal::not_yet(Some(at), "ksuperpose with another arity"));
        }
        let mut locals = HashMap::new();
        let mut tys = Vec::new();
        for ((_, r), &a) in entries.iter().zip(params) {
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

    /// The log density of `y` under one component law.
    fn component_lp(&mut self, law: NodeId, y: &str, index: Option<&str>) -> Result<String> {
        let l = self.law_any(law)?;
        self.law_lp(law, &l, y, index)
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
        let lp = self.law_lp(law, l, c, None)?;
        let num = |s: &str| s.trim().parse::<f64>().ok();
        let cv = num(c);
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
    /// component must be a probability measure: an unnormalized truncation
    /// has no closed-form share of the mixture.
    fn law_lp(&mut self, at: NodeId, law: &Law, y: &str, index: Option<&str>) -> Result<String> {
        if law.support.is_some() {
            return Err(Refusal::new(
                RefusalKind::NumericNormalizer,
                Some(at),
                "a mixture component",
                "a mixture component that is not normalized (a `truncate` without `normalize`) has \
                 no closed-form weight in the mixture",
            ));
        }
        if law.trunc.is_some() || !law.extra.is_empty() {
            return Err(Refusal::not_yet(Some(at), "a truncated mixture component"));
        }
        if law.declared {
            return Err(Refusal::not_yet(
                Some(at),
                "an ordered restriction as a mixture component",
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
        if !l.discrete {
            return Err(Refusal::new(
                RefusalKind::MixedReference,
                Some(mix),
                "superpose",
                "a `Dirac` atom beside a continuous law mixes two reference measures (§06)",
            ));
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
        if !(norm && (self.complement(wl, wd) || self.complement(wd, wl))) {
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
            None => format!("log({})", self.real(wd.unwrap())?.s),
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
        let lp_rest = self
            .law_density(law, &l, &format!("{lhs}[{rest}]"), None)?
            .replace(loop_, &rest);
        let log_w = if wlx.contains(loop_) {
            format!("sum(log({}))", wlx.replace(loop_, &rest))
        } else {
            format!("({n} - {count}) * log({wlx})")
        };
        out.push(format!("target += {log_w} + {lp_rest};"));
        Ok(Some(out))
    }
}

/// The top-level comma-separated arguments of a call's argument text.
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
