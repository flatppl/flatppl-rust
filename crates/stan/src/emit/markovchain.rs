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
use std::collections::HashMap;

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
                support: None,
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

/// Stands for the chain's state, then its per-step input, in a support
/// bound: a bound that holds one depends on the trajectory.
const STATE: &str = "\u{6}";
const STEP_INPUT: &str = "\u{8}";

/// A bare `markovchain(κ, init, n[, p])` or `kscan(κ, init, xs[, p])`.
struct Chain {
    init: NodeId,
    /// `n` of a markovchain.
    n: Option<NodeId>,
    /// `xs` of a kscan.
    xs: Option<NodeId>,
    p: Option<NodeId>,
    body: NodeId,
    entries: Vec<(flatppl_core::Symbol, Ref)>,
}

impl<'m> Emitter<'m> {
    /// The parts of a bare chain, refused by what is missing.
    fn chain(&self, m: NodeId) -> Result<Chain> {
        let c = self.call(m).unwrap();
        let kscan = self.head(m) == Some("kscan");
        let (kernel, init, third, p) = match c.args[..] {
            [k, i, t] => (k, i, t, None),
            [k, i, t, p] => (k, i, t, Some(p)),
            _ => return Err(Refusal::not_yet(Some(m), "a chain with this arity")),
        };
        if !c.named.is_empty() {
            return Err(Refusal::not_yet(Some(m), "a chain with keyword arguments"));
        }
        if matches!(self.ty(init), Some(Type::Record(_))) {
            return Err(Refusal::not_yet(Some(m), "a record-state markovchain"));
        }
        let fun = self.user_callee(kernel).ok_or_else(|| {
            Refusal::not_yet(Some(m), "a chain of a kernel that is not a user law")
        })?;
        let (body, entries) = self.callable_parts(fun)?;
        let head = self.head(self.resolve(body)).unwrap_or("");
        let rows = crate::tables::law_rows_by_head(head, None);
        if rows.is_empty() {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain step that is not a table law",
            ));
        }
        let scalar = |v: &str| v.starts_with("real") || v.starts_with("int");
        if !rows.iter().any(|r| scalar(&r.variate)) {
            return Err(Refusal::not_yet(
                Some(m),
                format!("a chain step of the multivariate law `{head}`"),
            ));
        }
        if !matches!(self.ty(init), Some(Type::Scalar(_))) {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain whose state is not a scalar",
            ));
        }
        let want = 1 + usize::from(kscan) + usize::from(p.is_some());
        if entries.len() != want {
            return Err(Refusal::not_yet(Some(m), "a chain kernel of another arity"));
        }
        Ok(Chain {
            init,
            n: (!kscan).then_some(third),
            xs: kscan.then_some(third),
            p,
            body,
            entries,
        })
    }

    /// The kernel inputs of `ch` bound: the state and per-step input as
    /// `state` and `input` (whole arrays when `full`), the parameter held.
    #[allow(clippy::type_complexity)]
    fn chain_leaves(
        &mut self,
        ch: &Chain,
        state: (String, bool, Type),
        input: Option<(String, bool, Type)>,
        full: bool,
    ) -> Result<(
        Vec<Type>,
        HashMap<flatppl_core::Symbol, (flatppl_core::Symbol, super::aggregate::Lifted)>,
        HashMap<flatppl_core::Symbol, NodeId>,
        HashMap<flatppl_core::Symbol, HashMap<String, String>>,
    )> {
        let mut tys = Vec::new();
        let mut leaves = HashMap::new();
        let mut nodes = HashMap::new();
        let mut records = HashMap::new();
        let mut leaf = |e: &(flatppl_core::Symbol, Ref),
                        (text, array, ty): (String, bool, Type),
                        tys: &mut Vec<Type>| {
            tys.push(ty);
            leaves.insert(
                e.1.name,
                (
                    e.0,
                    super::aggregate::Lifted {
                        ex: super::Ex::atom(text),
                        full,
                        array,
                    },
                ),
            );
        };
        leaf(&ch.entries[0], state, &mut tys);
        let mut k = 1;
        if let Some(input) = input {
            leaf(&ch.entries[1], input, &mut tys);
            k = 2;
        }
        if let Some(p) = ch.p {
            let r = ch.entries[k].1.name;
            let t = self
                .ty(p)
                .ok_or_else(|| Refusal::not_yet(Some(p), "an untyped value"))?;
            if matches!(t, Type::Record(_)) {
                records.insert(r, self.record_map(p)?);
            } else {
                let e = self.expr(p)?;
                leaves.insert(
                    r,
                    (
                        ch.entries[k].0,
                        super::aggregate::Lifted {
                            ex: e,
                            full: false,
                            array: false,
                        },
                    ),
                );
                nodes.insert(r, p);
            }
            tys.push(t);
        }
        Ok((tys, leaves, nodes, records))
    }

    /// The element type of a kscan's inputs `xs`.
    fn step_input_type(&self, m: NodeId, xs: NodeId) -> Result<Type> {
        match self.ty(xs) {
            Some(Type::Array { elem, shape }) if shape.len() == 1 => Ok(*elem),
            _ => Err(Refusal::not_yet(
                Some(m),
                "a kscan over inputs that are not one array",
            )),
        }
    }

    /// The support bounds a chain's step law gives each trajectory element,
    /// for a parameter trajectory's declaration. A bound that reads the
    /// state or the per-step input differs per element: it refuses.
    pub(super) fn chain_support(
        &mut self,
        m: NodeId,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        let ch = self.chain(m)?;
        let real = Type::Scalar(ScalarType::Real);
        // The per-step input has the element type of `xs` (`Binomial(n, …)`
        // takes an integer `n`).
        let input = match ch.xs {
            Some(xs) => Some((STEP_INPUT.to_string(), false, self.step_input_type(m, xs)?)),
            None => None,
        };
        let state_ty = self.ty(ch.init).unwrap_or(real);
        let (tys, leaves, _, records) =
            self.chain_leaves(&ch, (STATE.to_string(), false, state_ty), input, false)?;
        let scoped = self.instantiate(m, ch.body, &ch.entries, &tys)?;
        let mut ctx = FnCtx {
            scoped,
            inline_only: true,
            records,
            ..Default::default()
        };
        for (k, (_, l)) in &leaves {
            ctx.locals.insert(*k, l.ex.s.clone());
        }
        self.mark_boundary(&mut ctx, m, ch.body, &ch.entries)?;
        let outer = self.fctx.replace(ctx);
        let support = self.support(ch.body);
        self.fctx = outer;
        let support = support?;
        let reads = |b: &Option<String>| {
            b.as_ref()
                .is_some_and(|s| s.contains(STATE) || s.contains(STEP_INPUT))
        };
        if let Some((lo, hi)) = &support
            && (reads(lo) || reads(hi))
        {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain step whose support depends on the state",
            ));
        }
        Ok(support)
    }

    /// `y ~ markovchain(κ, init, n[, p])` or `kscan(κ, init, xs[, p])` at
    /// one trajectory `y` (spec §06 density rules: the product of the step
    /// densities, traj₀ = init). The state is the lag `[init, y₁, …,
    /// y_{n−1}]` and the per-step input `xs`, both whole arrays, so the step
    /// law is one vectorized statement through the elementwise lift.
    pub(super) fn chain_statement(&mut self, lhs: &str, m: NodeId) -> Result<Vec<String>> {
        let ch = self.chain(m)?;
        let t = self.decls.get(lhs).cloned();
        let Some(t) = t.filter(|t| t.rank() == 1) else {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain over a variate that is not one trajectory",
            ));
        };
        // The trajectory's length is the chain's.
        let length = match (ch.n, ch.xs) {
            (Some(n), _) => self.lin_int(n),
            (_, Some(xs)) => self.sym_extent(xs, 0, 0),
            _ => None,
        };
        let have = self.ext_lin.get(lhs).and_then(|e| e.first()).cloned();
        let same = match (&length, &have) {
            (Some(a), Some(b)) => self.same_size(a, b),
            _ => false,
        };
        if !same {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain whose length is not its trajectory's length",
            ));
        }
        // Stan's integer arguments take an integer array or a held value as
        // is: the lift computes over real vectors, so an integer argument
        // computed from the state or the per-step input refuses.
        let state = ch.entries[0].1;
        let input = ch.xs.map(|_| ch.entries[1].1);
        let computed = self.integer_law_args(ch.body).into_iter().any(|a| {
            let bare_input = matches!(self.m.node(a), Node::Ref(r) if Some(*r) == input);
            !bare_input && (mentions(self, a, state) || input.is_some_and(|i| mentions(self, a, i)))
        });
        if computed {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain step whose integer argument is computed from the state or the per-step input",
            ));
        }
        // Every trajectory element has the step law's support: one bound
        // for a parameter's declaration, one for the data's density.
        self.chain_support(m)?;
        let data = self.is_data(lhs);
        let n = t.array.first().or(t.dims.first()).cloned().unwrap();
        let int = t.is_int();
        // The lag `[init, y₁, …, y_{n−1}]` is a real vector, the state of a
        // real law argument.
        let init = self.real(ch.init)?.s;
        // A data size is still a placeholder token in `n`: the chain's
        // length says whether it is a literal.
        let literal = [&length, &have]
            .into_iter()
            .find_map(|l| l.as_ref().and_then(|l| l.as_const()));
        let hi = match literal {
            Some(k) => (k - 1).to_string(),
            None => format!("({n} - 1)"),
        };
        let history = if int {
            format!("to_vector({lhs}[1:{hi}])")
        } else {
            format!("{lhs}[1:{hi}]")
        };
        let lag_text = format!("append_row({init}, {history})");
        // A step that never reads the state needs no lag.
        let reads_state = mentions(self, ch.body, state);
        let lag = if !reads_state {
            lag_text
        } else if data && self.m.phase_of(ch.init) == Some(Phase::Fixed) {
            let name = self.fresh_local(&format!("{lhs}_lag"));
            let lt = StanType {
                array: vec![],
                base: "vector",
                dims: vec![n.clone()],
                bounds: Default::default(),
                support: None,
            };
            self.tdata
                .push(format!("{} = {lag_text};", lt.render(&name)));
            self.record_decl(&name, lt);
            name
        } else {
            lag_text
        };
        // The state keeps its type for the kernel's instantiation.
        let elem = if int {
            Type::Scalar(ScalarType::Integer)
        } else {
            Type::Scalar(ScalarType::Real)
        };
        let input = match ch.xs {
            Some(xs) => {
                let elem_x = self.step_input_type(m, xs)?;
                let x_int = self.shape(xs).is_some_and(|s| s.int);
                Some((self.expr(xs)?.s, x_int, elem_x))
            }
            None => None,
        };
        let (tys, leaves, nodes, records) =
            self.chain_leaves(&ch, (lag, false, elem), input, true)?;
        let Some((body, ctx)) =
            self.vectorized_law_body(m, ch.body, &ch.entries, &tys, leaves, nodes, records)?
        else {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain step whose argument does not lift over the trajectory",
            ));
        };
        let outer = self.fctx.replace(ctx);
        let law = self.law(body, false);
        self.fctx = outer;
        let law = law?;
        if !law.vectorized || law.multivariate {
            return Err(Refusal::not_yet(
                Some(m),
                "a chain step law that does not vectorize",
            ));
        }
        self.integer_data(m, &law, lhs)?;
        let mut out = vec![law.statement(m, &law.lhs(lhs), &law.call_text(None))?];
        out.extend(law.extra.iter().cloned());
        self.unchecked_guard(lhs, true, &law, out)
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
