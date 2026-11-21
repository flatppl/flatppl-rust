//! A row law that is itself a Dirac mixture: `iid(mix, J)` beside
//! `Dirac(fill(c, J))` over the rows of observed integer data, where `mix`
//! is a normalized mixture of one discrete law and a scalar atom at the same
//! `c` (a zero-inflated count per site, inside a zero-inflated row).
//!
//! The rows split once in `transformed data` (the row split of
//! [`Emitter::dirac_split_rows`]). Then, within each row, the sites at the
//! atom are counted and the other values are stored flat, row after row,
//! with a start offset per row: Stan has no ragged arrays, so a row's other
//! values are a range slice (Reference Manual "Indexing"). A row's law
//! arguments vary per row, and `_lpmf` returns a sum, so the rows are loops.

use super::super::Emitter;
use super::super::functions::FnCtx;
use super::Part;
use crate::refuse::{Refusal, Result};
use flatppl_core::{NodeId, Type};
use std::collections::HashMap;

/// The inner mixture's texts, [`LOOP`](super::super::structure::LOOP)
/// standing for the row.
struct Inner {
    /// Its log density at the atom: `log_mix(w, lp(c), 0)`.
    at_atom: String,
    /// `log(w)` of the law's weight.
    log_w: String,
    /// The law's log density at [`SLICE`].
    lp_slice: String,
}

/// Stands for the other values of one row in [`Inner::lp_slice`].
const SLICE: &str = "\u{5}";

impl<'m> Emitter<'m> {
    /// The nested split, or `None` when `law` is not `iid(mix, J)` of a
    /// Dirac mixture.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dirac_split_nested(
        &mut self,
        lhs: &str,
        mix: NodeId,
        norm: bool,
        dirac_first: bool,
        (wd, atom): (Option<NodeId>, NodeId),
        (wl, law): (Option<NodeId>, NodeId),
    ) -> Result<Option<Vec<String>>> {
        let iid = self.resolve(law);
        if self.head(iid) != Some("iid") {
            return Ok(None);
        }
        let ic = self.call(iid).unwrap();
        let (site, count_node) = (ic.args[0], ic.args[1]);
        let Some((body, ctx)) = self.mixture_body(mix, site)? else {
            return Ok(None);
        };
        let fill = self.resolve(atom);
        let data = self.is_data(lhs);
        let shaped = self
            .decls
            .get(lhs)
            .is_some_and(|t| t.is_int() && t.array.len() == 2);
        if self.head(fill) != Some("fill") || self.row_fill(atom) || !data || !shaped {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom beside an iid mixture in this form",
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
        if !self.fixed_value(cv) {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` atom whose value is not data",
            ));
        }
        // The atom and the iid product both cover one whole row.
        let row_len = self.ext_lin.get(lhs).and_then(|e| e.get(1)).cloned();
        for n in [fill_n, count_node] {
            let same = match (self.lin_int(n), row_len.clone()) {
                (Some(a), Some(b)) => self.same_size(&a, &b),
                _ => false,
            };
            if !same {
                return Err(Refusal::not_yet(
                    Some(mix),
                    "a `Dirac` atom or iid count of another length than its rows",
                ));
            }
        }
        let c = self.at_module(|me| me.expr(cv))?.s;
        // A called mixture law reads its parameters at the call's arguments.
        let inner = match ctx {
            Some(ctx) => {
                let outer = self.fctx.replace(ctx);
                let inner = self.inner_mixture(mix, body, &c);
                let ctx = std::mem::replace(&mut self.fctx, outer);
                if ctx.is_some_and(|c| !c.lines.is_empty()) {
                    return Err(Refusal::not_yet(
                        Some(mix),
                        "an iid mixture whose arguments need statements",
                    ));
                }
                inner
            }
            None => self.inner_mixture(mix, body, &c),
        };
        let inner = inner?;
        let loop_ = super::super::structure::LOOP;
        let wlx = self.real(wl.unwrap())?.s;
        let wdx = self.real(wd.unwrap())?.s;

        let n = self.extent_text(lhs);
        let len = self
            .decls
            .get(lhs)
            .and_then(|t| t.array.get(1).cloned())
            .unwrap_or_else(|| format!("size({lhs}[1])"));
        // Every row has the same arguments: no loop. An atom row has every
        // site at the atom, so it adds no other value.
        let texts = [&wlx, &wdx, &inner.at_atom, &inner.log_w, &inner.lp_slice];
        if !texts.iter().any(|t| t.contains(loop_)) {
            let count = self.fresh(&format!("{lhs}_n_atom"));
            let rest_n = self.fresh(&format!("{lhs}_rest_n_atom"));
            let n_other = self.fresh(&format!("{lhs}_n_other"));
            let other = self.fresh(&format!("{lhs}_other"));
            let (i, j) = (self.fresh_loop(), self.fresh_loop());
            let (r, b) = (self.fresh_local("r"), self.fresh_local("b"));
            self.tdata.push(format!(
                "int {count} = 0;\nint {rest_n} = 0;\nint {n_other} = 0;\nfor ({i} in 1:{n}) {{\n  int {r} = 0;\n  for ({j} in 1:{len}) {{\n    {r} += {lhs}[{i}, {j}] == {c};\n  }}\n  if ({r} == {len}) {{\n    {count} += 1;\n  }} else {{\n    {rest_n} += {r};\n  }}\n  {n_other} += {len} - {r};\n}}"
            ));
            self.tdata.push(format!("array[{n_other}] int {other};"));
            self.tdata.push(format!(
                "{{\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    for ({j} in 1:{len}) {{\n      if ({lhs}[{i}, {j}] != {c}) {{\n        {other}[{b}] = {lhs}[{i}, {j}];\n        {b} += 1;\n      }}\n    }}\n  }}\n}}"
            ));
            for v in [&i, &j, &r, &b] {
                self.used.remove(v);
            }
            let rows = format!("{len} * {}", inner.at_atom);
            let at = if dirac_first {
                format!("log_mix({wdx}, 0, {rows})")
            } else {
                format!("log_mix({wlx}, {rows}, 0)")
            };
            let lp = inner.lp_slice.replace(SLICE, &other);
            let log_l = self.log_weight(wl.unwrap())?;
            return Ok(Some(vec![
                format!("target += {count} * {at};"),
                format!(
                    "target += ({n} - {count}) * {log_l} + {rest_n} * {} + {n_other} * {} + {lp};",
                    inner.at_atom, inner.log_w
                ),
            ]));
        }
        let count = self.fresh(&format!("{lhs}_n_atom"));
        let at_idx = self.fresh(&format!("{lhs}_atom"));
        let rest = self.fresh(&format!("{lhs}_rest"));
        let row_n = self.fresh(&format!("{lhs}_row_n_atom"));
        let n_other = self.fresh(&format!("{lhs}_n_other"));
        let other = self.fresh(&format!("{lhs}_other"));
        let start = self.fresh(&format!("{lhs}_other_start"));
        let i = self.fresh_loop();
        let j = self.fresh_loop();
        let (a, b) = (self.fresh_local("a"), self.fresh_local("b"));
        let at_row = format!("max({lhs}[{i}]) == {c} && min({lhs}[{i}]) == {c}");
        self.tdata.push(format!(
            "int {count} = 0;\nfor ({i} in 1:{n}) {{\n  {count} += {at_row};\n}}"
        ));
        self.tdata.push(format!("array[{count}] int {at_idx};"));
        self.tdata.push(format!("array[{n} - {count}] int {rest};"));
        self.tdata.push(format!(
            "{{\n  int {a} = 1;\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    if ({at_row}) {{\n      {at_idx}[{a}] = {i};\n      {a} += 1;\n    }} else {{\n      {rest}[{b}] = {i};\n      {b} += 1;\n    }}\n  }}\n}}"
        ));
        self.tdata.push(format!("array[{n}] int {row_n};"));
        self.tdata.push(format!(
            "int {n_other} = 0;\nfor ({i} in 1:{n}) {{\n  {row_n}[{i}] = 0;\n  for ({j} in 1:{len}) {{\n    {row_n}[{i}] += {lhs}[{i}, {j}] == {c};\n  }}\n  {n_other} += {len} - {row_n}[{i}];\n}}"
        ));
        self.tdata.push(format!("array[{n_other}] int {other};"));
        self.tdata.push(format!("array[{n}] int {start};"));
        self.tdata.push(format!(
            "{{\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    {start}[{i}] = {b};\n    for ({j} in 1:{len}) {{\n      if ({lhs}[{i}, {j}] != {c}) {{\n        {other}[{b}] = {lhs}[{i}, {j}];\n        {b} += 1;\n      }}\n    }}\n  }}\n}}"
        ));
        for v in [&i, &j, &a, &b] {
            self.used.remove(v);
        }

        let row = self.fresh_loop();
        let k = self.fresh_loop();
        let m = self.fresh_local("m");
        let at_row_lp = format!("{len} * {}", inner.at_atom).replace(loop_, &row);
        let outer_at = if dirac_first {
            format!("log_mix({}, 0, {at_row_lp})", wdx.replace(loop_, &row))
        } else {
            format!("log_mix({}, {at_row_lp}, 0)", wlx.replace(loop_, &row))
        };
        let slice = format!("{other}[{start}[{row}]:({start}[{row}] + {m} - 1)]");
        let rest_lp = format!(
            "{} + {row_n}[{row}] * {} + {m} * {} + {}",
            self.log_weight(wl.unwrap())?,
            inner.at_atom,
            inner.log_w,
            inner.lp_slice.replace(SLICE, &slice)
        )
        .replace(loop_, &row);
        let out = vec![
            format!(
                "for ({k} in 1:{count}) {{\n  int {row} = {at_idx}[{k}];\n  target += {outer_at};\n}}"
            ),
            format!(
                "for ({k} in 1:({n} - {count})) {{\n  int {row} = {rest}[{k}];\n  int {m} = {len} - {row_n}[{row}];\n  target += {rest_lp};\n}}"
            ),
        ];
        for v in [&k, &row, &m] {
            self.used.remove(v);
        }
        Ok(Some(out))
    }

    /// The body of `site` when it is a normalized mixture: a user law call
    /// `f(args…)` inlined with its parameters at the arguments, or the
    /// mixture itself.
    fn mixture_body(
        &mut self,
        at: NodeId,
        site: NodeId,
    ) -> Result<Option<(NodeId, Option<FnCtx>)>> {
        let s = self.resolve(site);
        if self.mixture(s).is_some() {
            return Ok(Some((s, None)));
        }
        let Some(fun) = self.is_user_call(s) else {
            return Ok(None);
        };
        let (body, entries) = self.callable_parts(fun)?;
        if self.mixture(body).is_none() {
            return Ok(None);
        }
        let c = self.call(s).unwrap();
        let args = c.args.to_vec();
        if entries.len() != args.len() || !c.named.is_empty() {
            return Err(Refusal::not_yet(
                Some(at),
                "an iid mixture law called with another arity",
            ));
        }
        let mut tys = Vec::new();
        let mut locals = HashMap::new();
        for ((_, r), &a) in entries.iter().zip(&args) {
            let t = self
                .ty(a)
                .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?;
            if !matches!(t, Type::Scalar(_)) {
                return Err(Refusal::not_yet(
                    Some(at),
                    "an iid mixture law with an array argument",
                ));
            }
            let e = self.expr(a)?;
            tys.push(t);
            locals.insert(r.name, e.operand());
        }
        let scoped = self.instantiate(s, body, &entries, &tys)?;
        let mut ctx = FnCtx {
            locals,
            scoped,
            inline_only: true,
            ..Default::default()
        };
        self.mark_boundary(&mut ctx, s, body, &entries)?;
        Ok(Some((body, Some(ctx))))
    }

    /// The inner mixture's texts, under its own context.
    fn inner_mixture(&mut self, mix: NodeId, body: NodeId, c: &str) -> Result<Inner> {
        let (norm, inner) = self.mixture(body).unwrap();
        let parts = if self.head(inner) == Some("superpose") {
            self.parts(inner)?
        } else {
            vec![]
        };
        let (dirac_first, (wd, atom), (wl, law)) = match parts.as_slice() {
            [Part::Dirac(wd, a), Part::Law(wl, l)] => (true, (*wd, *a), (*wl, *l)),
            [Part::Law(wl, l), Part::Dirac(wd, a)] => (false, (*wd, *a), (*wl, *l)),
            _ => {
                return Err(Refusal::not_yet(
                    Some(mix),
                    "an iid mixture of this form beside a `Dirac` row",
                ));
            }
        };
        if !self.dirac_weights(wl, wd, norm) {
            return Err(Refusal::not_yet(
                Some(mix),
                "an iid mixture with these weights",
            ));
        }
        if self.shape(atom).is_none_or(|s| s.rank != 0) || !self.fixed_value(atom) {
            return Err(Refusal::not_yet(
                Some(mix),
                "an iid mixture whose atom is not a data scalar",
            ));
        }
        // One atom for both levels: an atom row has every site at the inner
        // atom.
        let c_in = self.at_module(|me| me.expr(atom))?.s;
        if c_in != c {
            return Err(Refusal::not_yet(
                Some(mix),
                "an iid mixture whose atom differs from the row atom",
            ));
        }
        let l = self.law_any(law)?;
        // The other values score in one call with no per-value mass.
        if l.trunc.is_some() || l.support.is_some() {
            return Err(Refusal::not_yet(Some(mix), "a truncated mixture component"));
        }
        // Over data the mixture has a density with respect to the atom plus
        // Lebesgue measure (Ruling 2); the row form does not split it yet.
        if !l.discrete {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` row atom beside an iid mixture of a continuous law",
            ));
        }
        // A shifted variate (`{y} + 1`) would add to the integer slice,
        // which Stan has no arithmetic for. `law_lp` below refuses a
        // truncation, an unnormalized support and an ordered restriction.
        if !l.vectorized || l.multivariate || l.lhs("{y}") != "{y}" {
            return Err(Refusal::not_yet(Some(mix), "an iid mixture with this law"));
        }
        let wlx = self.real(wl.unwrap())?.s;
        let at_atom = match self.atom_lp(mix, law, &l, c)? {
            None => self.log_weight(wd.unwrap())?,
            Some(lp_c) if dirac_first => {
                format!("log_mix({}, 0, {lp_c})", self.real(wd.unwrap())?.s)
            }
            Some(lp_c) => format!("log_mix({wlx}, {lp_c}, 0)"),
        };
        let lp_slice = self.law_density(law, &l, SLICE, None)?;
        Ok(Inner {
            at_atom,
            log_w: self.log_weight(wl.unwrap())?,
            lp_slice,
        })
    }
}
