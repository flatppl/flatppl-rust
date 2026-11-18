//! `markovchain(κ, init, n[, p])` over observed data (spec §06 "Dependent
//! composition"): step i is `traj_i ~ κ(traj_{i−1}, p)` with
//! `traj_0 = init`, so its density at an observed trajectory y is the
//! density of κ at the lagged trajectory `[init, y_1, …, y_{n−1}]`. The lag
//! is data, built once in `transformed data` by slice assignment.
//!
//! A kernel argument that reads the state is `ifelse(cond(state), a, b)`
//! with scalar branches that do not: an index select `[b, a]'[sel]` from a
//! data index array. Any other use of the state refuses.

use super::functions::FnCtx;
use super::mixture::{ROW, RowArg, RowLaw};
use super::{Emitter, P_ATOM};
use crate::refuse::{Refusal, Result};
use crate::stan_type::StanType;
use flatppl_core::{Node, NodeId, Phase, Ref, RefNs, ScalarType, Type};

/// Stands for the chain's state in the kernel's printed law.
const PREV: &str = "\u{5}";

impl<'m> Emitter<'m> {
    /// The law of each row of the data matrix `lhs` under the chain `mc`.
    /// `atom`: the value of every entry of a Dirac atom row, whose lag is
    /// then the same `[init, atom, …, atom]` for every atom row.
    pub(super) fn markovchain_row_law(
        &mut self,
        lhs: &str,
        mc: NodeId,
        atom: Option<&str>,
    ) -> Result<RowLaw> {
        let c = self.call(mc).unwrap();
        let args = c.args.to_vec();
        let (kernel, init, n, p) = match args[..] {
            [k, i, n] => (k, i, n, None),
            [k, i, n, p] => (k, i, n, Some(p)),
            _ => return Err(Refusal::not_yet(Some(mc), "markovchain with this arity")),
        };
        if !c.named.is_empty() {
            return Err(Refusal::not_yet(
                Some(mc),
                "markovchain with keyword arguments",
            ));
        }
        if matches!(self.ty(init), Some(Type::Record(_))) {
            return Err(Refusal::not_yet(Some(mc), "a record-state markovchain"));
        }
        let fun = self
            .user_callee(kernel)
            .ok_or_else(|| Refusal::not_yet(Some(mc), "a markovchain of this kernel"))?;
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != 1 + usize::from(p.is_some()) {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain kernel of another arity",
            ));
        }
        // The rows of an integer data matrix, each one trajectory.
        let t = self.decls.get(lhs).cloned();
        let Some(t) = t.filter(|t| t.is_int() && t.array.len() == 2 && t.dims.is_empty()) else {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain over this variate",
            ));
        };
        let length = self.lin_int(n);
        let row_len = self.ext_lin.get(lhs).and_then(|e| e.get(1)).cloned();
        let same = match (&length, &row_len) {
            (Some(a), Some(b)) => self.same_size(a, b),
            _ => false,
        };
        if !same {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain whose length is not its rows' length",
            ));
        }
        if self.m.phase_of(init) != Some(Phase::Fixed)
            || !self.shape(init).is_some_and(|s| s.rank == 0 && s.int)
        {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain whose initial state is not integer data",
            ));
        }
        let (rows, cols) = (t.array[0].clone(), t.array[1].clone());
        let init_x = self.expr(init)?;

        // The kernel's law at one step, its state a placeholder.
        let state = entries[0].1;
        let mut tys = vec![Type::Scalar(ScalarType::Integer)];
        if let Some(p) = p {
            tys.push(
                self.ty(p)
                    .ok_or_else(|| Refusal::not_yet(Some(p), "an untyped value"))?,
            );
        }
        let scoped = self.instantiate(mc, body, &entries, &tys)?;
        let mut ctx = FnCtx {
            scoped,
            inline_only: true,
            ..Default::default()
        };
        ctx.locals.insert(state.name, PREV.to_string());
        if let Some(p) = p {
            let r = entries[1].1.name;
            if matches!(self.ty(p), Some(Type::Record(_))) {
                let m = self.record_map(p)?;
                ctx.records.insert(r, m);
            } else {
                let e = self.expr(p)?;
                ctx.locals.insert(r, e.at(P_ATOM));
            }
        }
        self.mark_boundary(&mut ctx, mc, body, &entries)?;
        let law = self.resolve(body);
        let Some(lc) = self.call(law) else {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain kernel of this form",
            ));
        };
        let mut selects = Vec::new();
        for a in lc
            .args
            .iter()
            .chain(lc.named.iter().map(|na| &na.value))
            .copied()
        {
            if !mentions(self, a, state) {
                continue;
            }
            let ok = self.head(a) == Some("ifelse")
                && matches!(self.call(a).unwrap().args[..], [cond, x, y]
                    if !mentions(self, x, state)
                        && !mentions(self, y, state)
                        && self.data_condition(cond, state))
                && matches!(self.call(a).unwrap().args[..], [_, x, y]
                    if matches!(ctx.scoped.get(&x), Some(Type::Scalar(_)))
                        && matches!(ctx.scoped.get(&y), Some(Type::Scalar(_))));
            if !ok {
                return Err(Refusal::not_yet(
                    Some(a),
                    "a markovchain kernel argument of this form",
                ));
            }
            selects.push(a);
        }
        let outer = self.fctx.replace(ctx);
        let printed = self.kernel_law_text(mc, law, &selects);
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        self.merge_declared(&ctx);
        if !ctx.lines.is_empty() {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain kernel that needs statements",
            ));
        }
        let (call, parts) = printed?;

        // The lag: the initial state, then each row's observed history.
        let lag = self.fresh(&format!("{lhs}_previous"));
        // The row length's linear form decides the parentheses: a data
        // size is still a placeholder token here.
        let hi = if row_len.as_ref().is_some_and(|l| l.atomic()) {
            cols.clone()
        } else {
            format!("({cols})")
        };
        self.tdata.push(format!("array[{rows}, {cols}] int {lag};"));
        self.tdata
            .push(format!("{lag}[:, 1] = rep_array({}, {rows});", init_x.s));
        self.tdata
            .push(format!("{lag}[:, 2:{hi}] = {lhs}[:, 1:({cols} - 1)];"));
        self.record_decl(
            &lag,
            StanType {
                array: vec![rows.clone(), cols.clone()],
                base: "int",
                dims: vec![],
                bounds: Default::default(),
            },
        );
        // One data index array per select: 2 where the condition holds.
        let mut texts = Vec::new();
        for (ph, choices, cond) in parts {
            let sel = self.fresh(&format!("{lhs}_select"));
            let (i, j) = (self.loop_var(), self.loop_var());
            let at = cond.replace(PREV, &format!("{lag}[{i}, {j}]"));
            self.tdata.push(format!("array[{rows}, {cols}] int {sel};"));
            self.tdata.push(format!(
                "for ({i} in 1:{rows}) {{\n  for ({j} in 1:{cols}) {{\n    {sel}[{i}, {j}] = {at} ? 2 : 1;\n  }}\n}}"
            ));
            self.used.remove(&i);
            self.used.remove(&j);
            // Every atom row: the select at `[init, atom, …, atom]`.
            let atom_sel = match atom {
                Some(v) => {
                    let s = self.fresh(&format!("{lhs}_atom_select"));
                    let first = cond.replace(PREV, &format!("({})", init_x.s));
                    let later = cond.replace(PREV, &format!("({v})"));
                    self.tdata.push(format!("array[{cols}] int {s};"));
                    self.tdata.push(format!("{s}[1] = {first} ? 2 : 1;"));
                    self.tdata.push(format!(
                        "{s}[2:{hi}] = rep_array({later} ? 2 : 1, {cols} - 1);"
                    ));
                    Some(s)
                }
                None => None,
            };
            texts.push((ph, choices, sel, atom_sel));
        }
        let open = call.find('(').unwrap_or(call.len());
        let name = call[..open].to_string();
        let inner = call[open..]
            .strip_prefix('(')
            .and_then(|x| x.strip_suffix(')'))
            .unwrap_or("");
        let mut out = Vec::new();
        for a in super::mixture::split_args(inner) {
            out.push(match texts.iter().find(|(ph, _, _, _)| *ph == a) {
                Some((_, choices, sel, atom_sel)) => RowArg::PerRow {
                    row: format!("{choices}[{sel}[{ROW}]]"),
                    rest: format!("{choices}[to_array_1d({sel}[{ROW}])]"),
                    atom: atom_sel.as_ref().map(|s| format!("{choices}[{s}]")),
                },
                None if a.contains(PREV) => {
                    return Err(Refusal::not_yet(
                        Some(mc),
                        "a markovchain kernel argument of this form",
                    ));
                }
                None => RowArg::Shared(a),
            });
        }
        Ok(RowLaw {
            name: format!("{name}_lpmf"),
            args: out,
        })
    }

    /// The kernel's law as Stan text, each select argument replaced by a
    /// placeholder, with its choices `[b, a]'` and its condition.
    #[allow(clippy::type_complexity)]
    fn kernel_law_text(
        &mut self,
        mc: NodeId,
        law: NodeId,
        selects: &[NodeId],
    ) -> Result<(String, Vec<(String, String, String)>)> {
        let l = self.law_any(law)?;
        if !l.discrete
            || !l.vectorized
            || l.trunc.is_some()
            || l.support.is_some()
            || l.declared
            || l.lhs("{y}") != "{y}"
        {
            return Err(Refusal::not_yet(
                Some(mc),
                "a markovchain kernel of this law",
            ));
        }
        let open = l.call.find('(').unwrap_or(l.call.len());
        let name = &l.call[..open];
        let inner = l.call[open..]
            .strip_prefix('(')
            .and_then(|x| x.strip_suffix(')'))
            .unwrap_or("");
        let mut args = super::mixture::split_args(inner);
        let mut parts = Vec::new();
        for (k, &a) in selects.iter().enumerate() {
            let [cond, x, y] = self.call(a).unwrap().args[..] else {
                unreachable!()
            };
            let text = self.expr(a)?.s;
            let ph = format!("\u{6}{k}\u{6}");
            let wrapped = format!("({text})");
            let Some(slot) = args.iter_mut().find(|s| **s == text || **s == wrapped) else {
                return Err(Refusal::not_yet(
                    Some(a),
                    "a markovchain kernel argument of this form",
                ));
            };
            *slot = ph.clone();
            // Stan indexes a parenthesized literal, not `[b, a]'[i]`.
            let choices = format!("([{}, {}]')", self.real(y)?.s, self.real(x)?.s);
            let cond = self.expr(cond)?.s;
            parts.push((ph, choices, cond));
        }
        let call = format!("{name}({})", args.join(", "));
        Ok((call, parts))
    }

    /// `true` when `cond` reads only the chain's state and fixed values.
    fn data_condition(&self, cond: NodeId, state: Ref) -> bool {
        let mut stack = vec![cond];
        while let Some(id) = stack.pop() {
            match self.m.node(id) {
                Node::Ref(r) if *r == state => {}
                Node::Ref(r) if r.ns == RefNs::SelfMod => {
                    let fixed = self.m.binding_by_name(r.name).is_some_and(|b| {
                        self.m.phase_of(self.m.binding(b).rhs) == Some(Phase::Fixed)
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
}

/// `true` when `p` occurs under `root`.
fn mentions(e: &Emitter<'_>, root: NodeId, p: Ref) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if matches!(e.m.node(id), Node::Ref(r) if *r == p) {
            return true;
        }
        e.m.for_each_child(id, |c| stack.push(c));
    }
    false
}
