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

use super::functions::FnCtx;
use super::structure::Law;
use super::{Emitter, P_ATOM};
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Node, NodeId, Type};
use std::collections::HashMap;

/// The loop index a per-component argument is rendered with.
const COMP: &str = "\u{4}";

/// One component: its weight (none for weight 1) and its law, or an atom.
enum Part {
    Law(Option<NodeId>, NodeId),
    Dirac(Option<NodeId>, NodeId),
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

    /// The Dirac split over the rows of a two-dimensional data variate: an
    /// atom `fill(c, T)` against a broadcast law over each row. A row is at
    /// the atom when its largest and smallest values are `c`. The other rows
    /// score in one call, flattened row by row, each row-length argument
    /// repeated once per row.
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
        let data = self.json.contains_key(lhs)
            || self.deferred.iter().any(|d| d.stan == lhs)
            || self.tdata.iter().any(|t| t.contains(&format!(" {lhs}")));
        let shaped = self
            .decls
            .get(lhs)
            .is_some_and(|t| t.is_int() && t.array.len() == 2);
        if self.head(fill) != Some("fill")
            || self.head(row_law) != Some("broadcast")
            || !data
            || !shaped
        {
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
        let l = self.law_any(law)?;
        if !l.discrete || !l.vectorized || l.trunc.is_some() || l.support.is_some() || l.declared {
            return Err(Refusal::not_yet(
                Some(mix),
                "a `Dirac` mixture with this row law",
            ));
        }
        let c = self.expr(self.call(fill).unwrap().args[0])?;
        let atom_x = self.expr(atom)?;
        let n = self.extent_text(lhs);
        let count = self.fresh(&format!("{lhs}_n_atom"));
        let rest = self.fresh(&format!("{lhs}_rest"));
        let i = self.fresh_loop();
        let b = self.fresh("b");
        let at = format!("max({lhs}[{i}]) == {c} && min({lhs}[{i}]) == {c}", c = c.s);
        self.tdata.push(format!(
            "int {count} = 0;\nfor ({i} in 1:{n}) {{\n  {count} += {at};\n}}"
        ));
        self.tdata.push(format!("array[{n} - {count}] int {rest};"));
        self.tdata.push(format!(
            "{{\n  int {b} = 1;\n  for ({i} in 1:{n}) {{\n    if (!({at})) {{\n      {rest}[{b}] = {i};\n      {b} += 1;\n    }}\n  }}\n}}"
        ));
        self.used.remove(&i);
        self.used.remove(&b);
        let wlx = self.real(wl.unwrap())?.s;
        let lp_atom = self.law_density(law, &l, &atom_x.s, None)?;
        let mix_at = if dirac_first {
            format!("log_mix({}, 0, {lp_atom})", self.real(wd.unwrap())?.s)
        } else {
            format!("log_mix({wlx}, {lp_atom}, 0)")
        };
        // The other rows: row-major `to_array_1d` against each vector
        // argument repeated per row (`to_vector` of `rep_matrix` is
        // column-major, one copy of the vector per column).
        let mut repeated = Vec::new();
        for &arg in &self.call(row_law).unwrap().args[1..].to_vec() {
            if self.shape(arg).is_some_and(|s| s.rank == 1) {
                repeated.push(self.real(arg)?.s);
            } else if self.shape(arg).is_some_and(|s| s.rank > 1) {
                return Err(Refusal::not_yet(
                    Some(mix),
                    "a `Dirac` mixture with a matrix row argument",
                ));
            }
        }
        let open = l.call.find('(').unwrap_or(l.call.len());
        let (name, inner) = (
            &l.call[..open],
            l.call[open..].trim_start_matches('(').trim_end_matches(')'),
        );
        let args: Vec<String> = split_args(inner)
            .into_iter()
            .map(|a| {
                if repeated.contains(&a) {
                    format!("to_vector(rep_matrix({a}, {n} - {count}))")
                } else {
                    a
                }
            })
            .collect();
        let flat = format!(
            "{name}_lpmf(to_array_1d({lhs}[{rest}]) | {})",
            args.join(", ")
        );
        Ok(vec![
            format!("target += {count} * {mix_at};"),
            format!("target += ({n} - {count}) * log({wlx}) + {flat};"),
        ])
    }

    fn fresh_loop(&mut self) -> String {
        for c in ["i", "j", "n", "k"] {
            if !self.used.contains(c) {
                self.used.insert(c.to_string());
                return c.to_string();
            }
        }
        self.fresh("idx")
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
                format!("{}[{COMP}]", e.at(P_ATOM))
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
        let outer = self.fctx.replace(FnCtx {
            locals,
            scoped,
            ..Default::default()
        });
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
        if self.shape(atom).is_some_and(|s| s.rank == 1) {
            return self
                .dirac_split_rows(lhs, mix, norm, dirac_first, (wd, atom), (wl, law))
                .map(Some);
        }
        if self.shape(atom).is_none_or(|s| s.rank != 0) {
            return Err(Refusal::not_yet(Some(mix), "a `Dirac` atom of this shape"));
        }
        // Observed data only: the positions are found once.
        let data = self.json.contains_key(lhs)
            || self.deferred.iter().any(|d| d.stan == lhs)
            || self.tdata.iter().any(|t| t.contains(&format!(" {lhs}")));
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
        let c = self.at_module(|me| me.expr(atom))?;
        let n = self.extent_text(lhs);
        let count = self.fresh(&format!("{lhs}_n_atom"));
        let at_idx = self.fresh(&format!("{lhs}_atom"));
        let rest = self.fresh(&format!("{lhs}_rest"));
        let i = self.fresh_loop();
        let (a, b) = (self.fresh("a"), self.fresh("b"));
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
        let lp_atom = self.law_lp(law, &l, &c.s, None)?;
        let mix_at = if dirac_first {
            format!("log_mix({}, 0, {lp_atom})", self.real(wd.unwrap())?.s)
        } else {
            format!("log_mix({wlx}, {lp_atom}, 0)")
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
fn split_args(inner: &str) -> Vec<String> {
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
