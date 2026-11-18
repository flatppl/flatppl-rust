//! `scan(f, init, xs)` and `scan(f, init, xs, p)` (spec §04 "Folds and
//! scans").
//!
//! - An additive step from 0 is `cumulative_sum`.
//! - A step that never reads its accumulator is a map: the broadcast of its
//!   body over `xs`, one whole-array expression when the body lifts, else an
//!   elementwise loop with no running state.
//! - A step that reads its accumulator is a recurrence: one `for` loop. Each
//!   accumulator field the step reads is a running local in a nested block.
//!   Each field read after the scan is a column. When only the last entry is
//!   read, the running state itself is the value and no column is kept.
//!
//! A record accumulator gives a table, held as one array per column. A row
//! function broadcast over it reads columns, and a user function called with
//! a record argument is inlined with the record's fields bound to Stan names.

use super::extents::Lin;
use super::functions::FnCtx;
use super::{Emitter, Ex, P_ATOM};
use crate::refuse::{Refusal, Result};
use crate::stan_type::StanType;
use flatppl_core::{Dim, Node, NodeId, Phase, Ref, RefNs, Scalar, Symbol, Type};
use std::collections::{BTreeSet, HashMap, HashSet};

/// The Stan names a scan was emitted under. A scalar or vector scan uses the
/// field name `""`.
#[derive(Clone, Default)]
pub(crate) struct ScanOut {
    /// Field → its column array.
    columns: HashMap<String, String>,
    /// Field → its running state, when only the last entry is read.
    last: HashMap<String, String>,
}

/// Accumulator field → the accumulator fields its step body reads.
type FieldReads = HashMap<String, BTreeSet<String>>;

/// How the value of a scan is read after it.
enum Uses {
    /// Fields read row by row (`""` for a scalar or vector scan).
    Columns(BTreeSet<String>),
    /// Fields of the last entry only.
    Last(BTreeSet<String>),
}

/// A scan's step function and arguments.
struct Step {
    at: NodeId,
    body: NodeId,
    entries: Vec<(Symbol, Ref)>,
    init: NodeId,
    xs: NodeId,
    param: Option<NodeId>,
    /// The step body's fields, in order, for a record accumulator.
    fields: Option<Vec<(String, NodeId)>>,
}

impl<'m> Emitter<'m> {
    /// `scan(…)` as a value: `cumulative_sum`, a map, or the column of a
    /// scalar or vector recurrence.
    pub(super) fn scan_expr(&mut self, id: NodeId, args: &[NodeId]) -> Result<Ex> {
        let (f, init, xs) = match args {
            [f, init, xs] | [f, init, xs, _] => (*f, *init, *xs),
            _ => return Err(Refusal::not_yet(Some(id), "scan with this arity")),
        };
        if args.len() == 3
            && let Some(e) = self.additive_scan(id, f, init, xs)?
        {
            return Ok(e);
        }
        if matches!(self.ty(init), Some(Type::Record(_))) {
            return Err(Refusal::not_yet(Some(id), "a record scan used whole"));
        }
        let step = self.step(id)?;
        // An array of vectors has no Stan whole-value readers (`sum` of it
        // is a vector in FlatPPL, ill-typed in Stan): rows only.
        if self.shape(init).is_some_and(|s| s.rank >= 1) && self.scan_read_whole(id) {
            return Err(Refusal::not_yet(
                Some(id),
                "a scan of vectors read whole (only its rows are supported)",
            ));
        }
        if !self.reads_accumulator(&step) {
            let out = self.map_scan(&step, &BTreeSet::from([String::new()]))?;
            let col = out.columns[""].clone();
            self.remember(id, &col);
            return Ok(Ex::atom(col));
        }
        let out = self.recurrence(&step, &Uses::Columns(BTreeSet::from([String::new()])))?;
        let col = out.columns[""].clone();
        self.remember(id, &col);
        Ok(Ex::atom(col))
    }

    /// A `scan` whose step adds: `cumulative_sum`, from `init = 0`.
    fn additive_scan(
        &mut self,
        id: NodeId,
        f: NodeId,
        init: NodeId,
        xs: NodeId,
    ) -> Result<Option<Ex>> {
        let zero = matches!(self.m.node(self.resolve(init)), Node::Lit(Scalar::Int(0)))
            || matches!(self.m.node(self.resolve(init)), Node::Lit(Scalar::Real(x)) if *x == 0.0);
        let adds = match self.m.node(f) {
            Node::Const(c) => self.name(*c) == "add",
            _ => match self.user_callee(f) {
                Some(fun) => {
                    let (body, entries) = self.callable_parts(fun)?;
                    let refs: Vec<Ref> = entries.iter().map(|(_, r)| *r).collect();
                    let args = self.call(body).map(|c| c.args.to_vec()).unwrap_or_default();
                    self.head(body) == Some("add")
                        && refs.len() == 2
                        && args.len() == 2
                        && args
                            .iter()
                            .all(|&a| matches!(self.m.node(a), Node::Ref(r) if refs.contains(r)))
                        && !matches!(
                            (self.m.node(args[0]), self.m.node(args[1])),
                            (Node::Ref(a), Node::Ref(b)) if a == b
                        )
                }
                None => false,
            },
        };
        if !(zero && adds) {
            return Ok(None);
        }
        let x = self.expr(xs)?;
        self.table_fn(id, "cumsum", None, vec![x]).map(Some)
    }

    /// The scan node a value denotes, if any.
    pub(super) fn scan_of(&self, x: NodeId) -> Option<NodeId> {
        let r = self.resolve(x);
        (self.head(r) == Some("scan")).then_some(r)
    }

    /// `true` for a scan whose accumulator is a record (its value is a table).
    pub(super) fn record_scan(&self, scan: NodeId) -> bool {
        self.call(scan)
            .and_then(|c| c.args.get(1).copied())
            .is_some_and(|init| matches!(self.ty(init), Some(Type::Record(_))))
    }

    fn step(&mut self, id: NodeId) -> Result<Step> {
        let args = self.call(id).unwrap().args.to_vec();
        let (f, init, xs, param) = (args[0], args[1], args[2], args.get(3).copied());
        let fun = self
            .user_callee(f)
            .ok_or_else(|| Refusal::not_yet(Some(id), "scan of a builtin step"))?;
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != args.len() - 1 {
            return Err(Refusal::not_yet(Some(id), "a scan step of another arity"));
        }
        // `functionof(body, acc = a0, …)` binds its inputs to module values:
        // the body reads them through other bindings, which this emitter
        // does not substitute.
        if entries.iter().any(|(_, r)| r.ns == RefNs::SelfMod) {
            return Err(Refusal::not_yet(
                Some(id),
                "a scan step whose inputs are module values",
            ));
        }
        let fields = if matches!(self.ty(init), Some(Type::Record(_))) {
            let rec = self.resolve(body);
            if self.head(rec) != Some("record") {
                return Err(Refusal::not_yet(
                    Some(id),
                    "a record scan step that is not a record expression",
                ));
            }
            Some(
                self.call(rec)
                    .unwrap()
                    .named
                    .iter()
                    .map(|na| (self.name(na.name).to_string(), na.value))
                    .collect(),
            )
        } else {
            None
        };
        Ok(Step {
            at: id,
            body,
            entries,
            init,
            xs,
            param,
            fields,
        })
    }

    fn reads_accumulator(&self, step: &Step) -> bool {
        mentions(self, step.body, step.entries[0].1)
    }

    /// The fields of `param` read under `root`: `get(param, "f")`, also
    /// through user functions it is passed to. `None` when it is used whole.
    fn fields_read(&self, root: NodeId, param: Ref, depth: u32) -> Option<BTreeSet<String>> {
        if depth > 16 {
            return None;
        }
        let mut out = BTreeSet::new();
        let mut stack = vec![(root, None::<NodeId>)];
        while let Some((id, parent)) = stack.pop() {
            if matches!(self.m.node(id), Node::Ref(r) if *r == param) {
                let p = parent?;
                let pc = self.call(p)?;
                if self.head(p) == Some("get")
                    && pc.args.len() == 2
                    && pc.args[0] == id
                    && let Node::Lit(Scalar::Str(f)) = self.m.node(pc.args[1])
                {
                    out.insert(f.to_string());
                    continue;
                }
                let fun = self.is_user_call(p)?;
                let (body, entries) = self.callable_parts(fun).ok()?;
                for (k, &a) in pc.args.iter().enumerate() {
                    if a == id {
                        out.extend(self.fields_read(body, entries.get(k)?.1, depth + 1)?);
                    }
                }
                continue;
            }
            self.m.for_each_child(id, |c| stack.push((c, Some(id))));
        }
        Some(out)
    }

    // ------------------------------------------------------------- uses

    /// Every node with `child` among its children, over all bindings.
    fn parents_of(&self, child: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (_, b) in self.m.bindings() {
            let mut stack = vec![b.rhs];
            while let Some(id) = stack.pop() {
                if !seen.insert(id) {
                    continue;
                }
                self.m.for_each_child(id, |c| {
                    if c == child && !out.contains(&id) {
                        out.push(id);
                    }
                    stack.push(c);
                });
            }
        }
        out
    }

    /// The nodes that denote the scan: itself and references to it.
    fn scan_refs(&self, scan: NodeId) -> Vec<NodeId> {
        let mut out = vec![scan];
        let mut seen = HashSet::new();
        for (_, b) in self.m.bindings() {
            let mut stack = vec![b.rhs];
            while let Some(id) = stack.pop() {
                if !seen.insert(id) {
                    continue;
                }
                if matches!(self.m.node(id), Node::Ref(r) if r.ns == RefNs::SelfMod)
                    && id != scan
                    && self.resolve(id) == scan
                {
                    out.push(id);
                }
                self.m.for_each_child(id, |c| stack.push(c));
            }
        }
        out
    }

    /// `true` when some reader of the scan's value is not a row `get`.
    fn scan_read_whole(&self, scan: NodeId) -> bool {
        self.scan_refs(scan).into_iter().any(|r| {
            self.parents_of(r)
                .into_iter()
                .any(|p| self.head(p) != Some("get") || self.call(p).is_none_or(|c| c.args[0] != r))
        })
    }

    /// `true` when `sel` selects the last entry of the scan.
    fn is_last(&mut self, scan: NodeId, sel: NodeId) -> bool {
        let xs = self.call(scan).unwrap().args[2];
        match (self.lin_int(sel), self.sym_extent(xs, 0, 0)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// How the scan's value is read: by rows (columns), or the last entry.
    fn scan_uses(&mut self, scan: NodeId) -> Result<Uses> {
        let record = self.record_scan(scan);
        let mut columns = BTreeSet::new();
        let mut last = BTreeSet::new();
        let mut any_last = false;
        let whole = || Refusal::not_yet(Some(scan), "a record scan used whole");
        for r in self.scan_refs(scan) {
            for p in self.parents_of(r) {
                let pc = self.call(p).unwrap();
                let pargs = pc.args.to_vec();
                match self.head(p) {
                    Some("get") if pargs[0] == r && pargs.len() == 2 => {
                        if let Node::Lit(Scalar::Str(f)) = self.m.node(pargs[1]) {
                            if !record {
                                return Err(whole());
                            }
                            columns.insert(f.to_string());
                        } else if self.is_last(scan, pargs[1]) {
                            any_last = true;
                            if record {
                                last.extend(self.last_entry_fields(scan, p)?);
                            } else {
                                last.insert(String::new());
                            }
                        } else if record {
                            return Err(Refusal::not_yet(
                                Some(p),
                                "a row of a record scan other than the last",
                            ));
                        } else {
                            columns.insert(String::new());
                        }
                    }
                    Some("broadcast") if record => {
                        let fun = self
                            .user_callee(pargs[0])
                            .ok_or_else(|| Refusal::not_yet(Some(p), "a broadcast over a table"))?;
                        let (body, entries) = self.callable_parts(fun)?;
                        for (k, &a) in pargs[1..].iter().enumerate() {
                            if a == r {
                                let read = entries
                                    .get(k)
                                    .and_then(|e| self.fields_read(body, e.1, 0))
                                    .ok_or_else(|| {
                                        Refusal::not_yet(
                                            Some(p),
                                            "a table used other than through field reads",
                                        )
                                    })?;
                                columns.extend(read);
                            }
                        }
                    }
                    _ if record => return Err(whole()),
                    _ => {
                        columns.insert(String::new());
                    }
                }
            }
        }
        if any_last && !columns.is_empty() {
            if record {
                return Err(Refusal::not_yet(
                    Some(scan),
                    "a record scan read by rows and as its last entry",
                ));
            }
            return Ok(Uses::Columns(columns));
        }
        if any_last {
            Ok(Uses::Last(last))
        } else {
            Ok(Uses::Columns(columns))
        }
    }

    /// The fields of a record scan's last entry `get_node` its reader uses.
    fn last_entry_fields(&self, scan: NodeId, get_node: NodeId) -> Result<BTreeSet<String>> {
        let mut out = BTreeSet::new();
        let unsupported =
            || Refusal::not_yet(Some(scan), "the last entry of a record scan used whole");
        for p in self.parents_of(get_node) {
            let pc = self.call(p).unwrap();
            if self.head(p) == Some("get")
                && let [_, f] = pc.args[..]
                && let Node::Lit(Scalar::Str(f)) = self.m.node(f)
            {
                out.insert(f.to_string());
                continue;
            }
            let fun = self.is_user_call(p).ok_or_else(unsupported)?;
            let (body, entries) = self.callable_parts(fun)?;
            for (k, &a) in pc.args.iter().enumerate() {
                if a == get_node {
                    let e = entries.get(k).ok_or_else(unsupported)?;
                    out.extend(self.fields_read(body, e.1, 0).ok_or_else(unsupported)?);
                }
            }
        }
        Ok(out)
    }

    // --------------------------------------------------------- readers

    /// The columns of a record scan, emitted on first use.
    pub(super) fn scan_table(&mut self, scan: NodeId) -> Result<HashMap<String, String>> {
        if let Some(out) = self.recall_scan(scan) {
            return Ok(out.columns);
        }
        let uses = self.scan_uses(scan)?;
        let Uses::Columns(_) = uses else {
            return Err(Refusal::not_yet(Some(scan), "a record scan used whole"));
        };
        let step = self.step(scan)?;
        let out = if self.reads_accumulator(&step) {
            self.recurrence(&step, &uses)?
        } else {
            let Uses::Columns(fields) = &uses else {
                unreachable!()
            };
            self.map_scan(&step, fields)?
        };
        self.remember_scan(scan, out.clone());
        Ok(out.columns)
    }

    /// The last entry `scan[n]`: the running state, when only the last
    /// entry is read. `None` when the scan keeps columns instead.
    pub(super) fn scan_last(&mut self, scan: NodeId) -> Result<Option<HashMap<String, String>>> {
        if let Some(out) = self.recall_scan(scan) {
            return Ok((!out.last.is_empty()).then_some(out.last));
        }
        let uses = self.scan_uses(scan)?;
        if !matches!(uses, Uses::Last(_)) {
            return Ok(None);
        }
        let step = self.step(scan)?;
        if !self.reads_accumulator(&step) {
            return Ok(None);
        }
        let out = self.recurrence(&step, &uses)?;
        self.remember_scan(scan, out.clone());
        Ok(Some(out.last))
    }

    /// `x[sel]` when `x` is a scan read only at its last entry.
    pub(super) fn scan_last_value(&mut self, x: NodeId, sel: NodeId) -> Result<Option<Ex>> {
        let Some(scan) = self.scan_of(x) else {
            return Ok(None);
        };
        if self.record_scan(scan) || !self.is_last(scan, sel) {
            return Ok(None);
        }
        Ok(self
            .scan_last(scan)?
            .and_then(|m| m.get("").cloned())
            .map(Ex::atom))
    }

    fn recall_scan(&self, scan: NodeId) -> Option<ScanOut> {
        match &self.fctx {
            Some(c) => c.scans.get(&scan).cloned(),
            None => self.scans.get(&scan).cloned(),
        }
    }

    fn remember_scan(&mut self, scan: NodeId, out: ScanOut) {
        match &mut self.fctx {
            Some(c) => c.scans.insert(scan, out),
            None => self.scans.insert(scan, out),
        };
    }

    // --------------------------------------------------------- emission

    /// The base name of a scan's columns: its binding, else its owner.
    fn scan_base(&self, scan: NodeId) -> String {
        if let Some((_, b)) = self.m.bindings().find(|(_, b)| b.rhs == scan) {
            return self.name(b.name).to_string();
        }
        match &self.owner {
            Some((name, root)) if *root == scan => name.clone(),
            Some((name, _)) => format!("{name}_scan"),
            None => "scan".to_string(),
        }
    }

    /// The owner's name, for a scalar running state `<owner>_state`.
    fn state_base(&self, scan: NodeId) -> String {
        match &self.owner {
            Some((name, _)) => name.clone(),
            None => self.scan_base(scan),
        }
    }

    /// `true` when the owner being bound is the scan's last entry itself.
    fn binds_last(&self, scan: NodeId) -> bool {
        self.owner.as_ref().is_some_and(|(_, root)| {
            self.head(*root) == Some("get")
                && self
                    .call(*root)
                    .is_some_and(|c| c.args.len() == 2 && self.scan_of(c.args[0]) == Some(scan))
        })
    }

    /// The loop over `xs`: its bounds, the step's `next`, and the row number
    /// a column is indexed by.
    fn scan_loop(&mut self, xs: NodeId, var: &str) -> Result<(String, String, Ex, String)> {
        if let Some(r) = self.range(xs)
            && let Some((a, b, start)) = r.unit_loop()
        {
            // Over `lo:hi` the row number is `var - lo + 1`.
            let off = start.add(-1);
            let row = match off.as_const() {
                Some(0) => var.to_string(),
                Some(k) if k < 0 => format!("{var} + {}", -k),
                _ if off.atomic() => format!("{var} - {}", off.print()),
                _ => format!("{var} - ({})", off.print()),
            };
            return Ok((a, b, Ex::atom(var.to_string()), row));
        }
        let n = self.scan_length(xs)?;
        let x = self.elem(xs, &[var.to_string()])?;
        let hi = if n.atomic() {
            n.print()
        } else {
            format!("({})", n.print())
        };
        Ok(("1".into(), hi, x, var.to_string()))
    }

    fn scan_length(&mut self, xs: NodeId) -> Result<Lin> {
        if let Some(l) = self.sym_extent(xs, 0, 0) {
            return Ok(l);
        }
        match self.ty(xs) {
            Some(Type::Array { shape, .. }) => match shape.first() {
                Some(Dim::Static(n)) => Ok(Lin::konst(i64::from(*n))),
                _ => Err(Refusal::not_yet(Some(xs), "a dynamic extent")),
            },
            _ => Err(Refusal::not_yet(Some(xs), "a scan over this value")),
        }
    }

    /// The element type of `xs` and the step's argument types.
    fn step_types(&mut self, step: &Step) -> Result<HashMap<NodeId, Type>> {
        let elem = match self.ty(step.xs) {
            Some(Type::Table { .. }) => {
                return Err(Refusal::not_yet(Some(step.at), "scan over a table"));
            }
            Some(Type::Array { shape, elem }) if shape.len() == 1 => *elem,
            Some(Type::Array { .. }) => {
                return Err(Refusal::not_yet(Some(step.at), "scan over a matrix"));
            }
            _ => return Err(Refusal::not_yet(Some(step.at), "scan over this value")),
        };
        let mut tys = vec![
            self.ty(step.init)
                .ok_or_else(|| Refusal::not_yet(Some(step.init), "an untyped value"))?,
            elem,
        ];
        if let Some(p) = step.param {
            tys.push(
                self.ty(p)
                    .ok_or_else(|| Refusal::not_yet(Some(p), "an untyped value"))?,
            );
        }
        self.instantiate(step.at, step.body, &step.entries, &tys)
    }

    /// The step's parameter argument `p`: a record's fields by name, or one
    /// expression.
    fn bind_param(&mut self, step: &Step, ctx: &mut FnCtx) -> Result<()> {
        let Some(p) = step.param else {
            return Ok(());
        };
        let pr = step.entries[2].1.name;
        if matches!(self.ty(p), Some(Type::Record(_))) {
            let map = self.record_map(p)?;
            ctx.records.insert(pr, map);
        } else {
            let e = self.expr(p)?;
            ctx.locals.insert(pr, e.at(P_ATOM));
        }
        Ok(())
    }

    /// The init value of accumulator field `f` (`""` for the whole init).
    fn init_of(&self, step: &Step, f: &str) -> Result<NodeId> {
        if f.is_empty() {
            return Ok(step.init);
        }
        let rec = self.resolve(step.init);
        self.call(rec)
            .filter(|_| self.head(rec) == Some("record"))
            .and_then(|c| {
                c.named
                    .iter()
                    .find(|na| self.name(na.name) == f)
                    .map(|na| na.value)
            })
            .ok_or_else(|| {
                Refusal::not_yet(
                    Some(step.init),
                    "a record scan whose init is not a record expression",
                )
            })
    }

    /// The Stan type of accumulator field `f`: the step's result type,
    /// sized from the init value.
    fn field_type(
        &mut self,
        step: &Step,
        types: &HashMap<NodeId, Type>,
        f: &str,
        owner: &str,
    ) -> Result<StanType> {
        let init = self.init_of(step, f)?;
        let value = match &step.fields {
            Some(fs) => fs.iter().find(|(n, _)| n == f).map(|(_, v)| *v),
            None => Some(step.body),
        };
        let ty = value
            .and_then(|v| types.get(&v).cloned())
            .filter(|t| matches!(t, Type::Scalar(_) | Type::Array { .. }))
            .or_else(|| self.ty(init))
            .ok_or_else(|| Refusal::not_yet(Some(init), "an untyped value"))?;
        self.declare_ty(&ty, None, false, init, owner, &[])
    }

    /// A recurrence: one `for` loop over `xs`, running locals for the
    /// accumulator fields the step reads, columns or last-entry state as
    /// `uses` asks.
    fn recurrence(&mut self, step: &Step, uses: &Uses) -> Result<ScanOut> {
        let types = self.step_types(step)?;
        let acc = step.entries[0].1;
        // Each field's body and the accumulator fields it reads.
        let (bodies, reads): (Vec<(String, NodeId)>, FieldReads) = match &step.fields {
            Some(fs) => {
                let mut reads = HashMap::new();
                for (f, v) in fs {
                    let r = self.fields_read(*v, acc, 0).ok_or_else(|| {
                        Refusal::not_yet(Some(*v), "a step that uses its record accumulator whole")
                    })?;
                    reads.insert(f.clone(), r);
                }
                (fs.clone(), reads)
            }
            None => (
                vec![(String::new(), step.body)],
                HashMap::from([(String::new(), BTreeSet::from([String::new()]))]),
            ),
        };
        let running: BTreeSet<String> = reads.values().flatten().cloned().collect();
        let (wanted, last_mode) = match uses {
            Uses::Columns(c) => (c.clone(), false),
            Uses::Last(c) => (c.clone(), true),
        };
        for f in &wanted {
            if !bodies.iter().any(|(n, _)| n == f) {
                return Err(Refusal::not_yet(
                    Some(step.at),
                    format!("the scan field `{f}`, which the step does not return"),
                ));
            }
        }
        let fixed = self.m.phase_of(step.at) == Some(Phase::Fixed);
        let base = self.scan_base(step.at);
        let scalar = step.fields.is_none();

        // Running locals: the fields the step reads, plus the last entry's
        // fields when it is the value.
        let mut state: Vec<String> = running.iter().cloned().collect();
        if last_mode {
            for f in &wanted {
                if !state.contains(f) {
                    state.push(f.clone());
                }
            }
        }
        state.sort_by_key(|f| bodies.iter().position(|(n, _)| n == f));
        let mut names: HashMap<String, String> = HashMap::new();
        let mut decls = Vec::new();
        for f in &state {
            let name = if scalar && last_mode && self.binds_last(step.at) {
                // `name = scan(…)[n]`: the running state is the binding.
                self.fresh(&self.state_base(step.at))
            } else if scalar {
                self.fresh(&format!("{}_state", self.state_base(step.at)))
            } else {
                self.fresh(f)
            };
            let t = self.field_type(step, &types, f, &name)?;
            let init = self.init_of(step, f)?;
            let v = self.expr(init)?;
            decls.push(format!("{} = {};", t.render(&name), v.s));
            self.record_decl(&name, t);
            names.insert(f.clone(), name);
        }
        // Columns of the fields read by rows.
        let mut columns: HashMap<String, String> = HashMap::new();
        let mut column_decls = Vec::new();
        if !last_mode {
            let n_lin = self.scan_length(step.xs)?;
            let n = n_lin.print();
            for f in &wanted {
                let name = if f.is_empty() {
                    self.fresh(&base)
                } else {
                    self.fresh(&format!("{base}_{f}"))
                };
                let elem = self.field_type(step, &types, f, &name)?;
                let t = column_type(elem, &n);
                column_decls.push(format!("{};", t.render(&name)));
                let mut lins = vec![n_lin.clone()];
                if let Some(inner) = self.ext_lin.get(&name).cloned() {
                    lins.extend(inner);
                }
                self.ext_lin.insert(name.clone(), lins);
                self.record_decl(&name, t);
                columns.insert(f.clone(), name);
            }
        }

        let module_t = self
            .m
            .bindings()
            .any(|(_, b)| super::sanitize(self.name(b.name)) == "t");
        let var = if self.used.contains("t") || module_t {
            self.loop_var()
        } else {
            self.used.insert("t".to_string());
            "t".to_string()
        };
        let body = self.recurrence_body(
            step, types, &var, &bodies, &reads, &names, &columns, &wanted, last_mode,
        );
        self.used.remove(&var);
        let (lo, hi, body) = body?;

        let mut loop_text = format!("for ({var} in {lo}:{hi}) {{\n");
        for l in &body {
            for part in l.lines() {
                loop_text.push_str(&format!("  {part}\n"));
            }
        }
        loop_text.push('}');
        let mut lines = column_decls;
        if last_mode {
            lines.extend(decls);
            lines.push(loop_text);
        } else {
            let mut block = String::from("{\n");
            for l in decls.iter().chain(std::iter::once(&loop_text)) {
                for part in l.lines() {
                    block.push_str(&format!("  {part}\n"));
                }
            }
            block.push('}');
            lines.push(block);
        }
        self.put(fixed, lines);
        Ok(ScanOut {
            columns,
            last: if last_mode {
                wanted
                    .iter()
                    .map(|f| (f.clone(), names[f].clone()))
                    .collect()
            } else {
                HashMap::new()
            },
        })
    }

    /// The statements of one step (rule order: unread columns, running
    /// locals in dependency order, read columns).
    #[allow(clippy::too_many_arguments)]
    fn recurrence_body(
        &mut self,
        step: &Step,
        types: HashMap<NodeId, Type>,
        var: &str,
        bodies: &[(String, NodeId)],
        reads: &HashMap<String, BTreeSet<String>>,
        names: &HashMap<String, String>,
        columns: &HashMap<String, String>,
        wanted: &BTreeSet<String>,
        last_mode: bool,
    ) -> Result<(String, String, Vec<String>)> {
        let (lo, hi, x, row) = self.scan_loop(step.xs, var)?;
        let mut ctx = FnCtx {
            scoped: types,
            ..Default::default()
        };
        ctx.locals.insert(step.entries[1].1.name, x.at(P_ATOM));
        self.bind_param(step, &mut ctx)?;
        let acc = step.entries[0].1.name;
        if step.fields.is_some() {
            ctx.records.insert(acc, names.clone());
        } else {
            ctx.locals.insert(acc, names[""].clone());
        }
        let base = self.scan_base(step.at);
        let outer = self.fctx.replace(ctx);
        let outer_owner = self.owner.replace((base, step.body));
        let out = self.step_statements(&row, bodies, reads, names, columns, wanted, last_mode);
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        self.owner = outer_owner;
        self.merge_declared(&ctx);
        out.map(|lines| (lo, hi, lines))
    }

    #[allow(clippy::too_many_arguments)]
    fn step_statements(
        &mut self,
        row: &str,
        bodies: &[(String, NodeId)],
        reads: &HashMap<String, BTreeSet<String>>,
        names: &HashMap<String, String>,
        columns: &HashMap<String, String>,
        wanted: &BTreeSet<String>,
        last_mode: bool,
    ) -> Result<Vec<String>> {
        let mut out = Vec::new();
        // 1. Fields the step does not read: their columns, or their last-entry
        //    state, from the old running values.
        for (f, v) in bodies {
            if names.contains_key(f) && reads.values().any(|r| r.contains(f)) {
                continue;
            }
            let target = if last_mode {
                names.get(f).cloned()
            } else {
                columns.get(f).map(|c| format!("{c}[{row}]"))
            };
            if let Some(target) = target {
                let e = self.step_value(*v, &mut out)?;
                out.push(format!("{target} = {e};"));
            }
        }
        // 2. Running locals, a field before the fields it reads; temporaries
        //    when the reads form a cycle.
        let run: Vec<&(String, NodeId)> = bodies
            .iter()
            .filter(|(f, _)| names.contains_key(f) && reads.values().any(|r| r.contains(f)))
            .collect();
        match order(&run, reads) {
            Some(order) => {
                for (f, v) in order {
                    let e = self.step_value(v, &mut out)?;
                    out.push(format!("{} = {e};", names[&f]));
                }
            }
            None => {
                let mut assigns = Vec::new();
                for (f, v) in &run {
                    let name = &names[f];
                    let tmp = self.fresh(&format!("{name}_next"));
                    let t = self.decls[name].clone();
                    let e = self.step_value(*v, &mut out)?;
                    out.push(format!("{} = {e};", t.render(&tmp)));
                    assigns.push(format!("{name} = {tmp};"));
                    self.used.remove(&tmp);
                }
                out.extend(assigns);
            }
        }
        // 3. Columns of the fields the step reads, from the new values.
        if !last_mode {
            for (f, _) in bodies {
                if let (Some(c), Some(n)) = (columns.get(f), names.get(f))
                    && wanted.contains(f)
                {
                    out.push(format!("{c}[{row}] = {n};"));
                }
            }
        }
        Ok(out)
    }

    /// A step field's value, with the statements it needs placed before it.
    fn step_value(&mut self, v: NodeId, out: &mut Vec<String>) -> Result<String> {
        let e = self.expr(v);
        out.append(&mut self.fctx.as_mut().unwrap().lines);
        Ok(e?.s)
    }

    /// A map scan: the step never reads its accumulator, so each entry is the
    /// step's body at one element of `xs`. A body that lifts over the whole
    /// array is one expression; any other fills its column elementwise.
    fn map_scan(&mut self, step: &Step, wanted: &BTreeSet<String>) -> Result<ScanOut> {
        let types = self.step_types(step)?;
        let fixed = self.m.phase_of(step.at) == Some(Phase::Fixed);
        let base = self.scan_base(step.at);
        let bodies: Vec<(String, NodeId)> = match &step.fields {
            Some(fs) => fs.clone(),
            None => vec![(String::new(), step.body)],
        };
        let n_lin = self.scan_length(step.xs)?;
        let n = n_lin.print();
        let mut columns = HashMap::new();
        let mut lines = Vec::new();
        let mut looped = Vec::new();
        for (f, v) in &bodies {
            if !wanted.contains(f) {
                continue;
            }
            let name = if f.is_empty() {
                self.fresh(&base)
            } else {
                self.fresh(&format!("{base}_{f}"))
            };
            let elem = self.field_type(step, &types, f, &name)?;
            let t = column_type(elem, &n);
            match self.lift_map(step, &types, *v)? {
                Some(e) => lines.push(format!("{} = {};", t.render(&name), e.s)),
                None => {
                    lines.push(format!("{};", t.render(&name)));
                    looped.push((name.clone(), *v));
                }
            }
            let mut lins = vec![n_lin.clone()];
            if let Some(inner) = self.ext_lin.get(&name).cloned() {
                lins.extend(inner);
            }
            self.ext_lin.insert(name.clone(), lins);
            self.record_decl(&name, t);
            columns.insert(f.clone(), name);
        }
        if !looped.is_empty() {
            let var = self.loop_var();
            let out = self.map_loop(step, types, &var, &looped);
            self.used.remove(&var);
            lines.push(out?);
        }
        self.put(fixed, lines);
        let out = ScanOut {
            columns,
            last: HashMap::new(),
        };
        self.remember_scan(step.at, out.clone());
        Ok(out)
    }

    /// The step body over the whole `xs`, when it lifts.
    fn lift_map(
        &mut self,
        step: &Step,
        types: &HashMap<NodeId, Type>,
        v: NodeId,
    ) -> Result<Option<Ex>> {
        let xs = self.expr(step.xs)?;
        let int = self.shape(step.xs).is_some_and(|s| s.int);
        let mut ctx = FnCtx {
            scoped: types.clone(),
            ..Default::default()
        };
        self.bind_param(step, &mut ctx)?;
        let x = step.entries[1];
        let outer = self.fctx.replace(ctx);
        let outer_v = self.vlocals.replace(HashMap::from([(
            x.1.name,
            super::aggregate::Lifted {
                ex: xs,
                full: true,
                array: int,
            },
        )]));
        let lifted = self.lift(v, &[x.0], &HashMap::new(), &HashMap::new());
        self.vlocals = outer_v;
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        match lifted {
            Ok(l) if l.full && ctx.lines.is_empty() => {
                self.merge_declared(&ctx);
                Ok(Some(if l.array {
                    self.to_real(&l.ex, 1)
                } else {
                    l.ex
                }))
            }
            _ => Ok(None),
        }
    }

    fn map_loop(
        &mut self,
        step: &Step,
        types: HashMap<NodeId, Type>,
        var: &str,
        looped: &[(String, NodeId)],
    ) -> Result<String> {
        let (lo, hi, x, row) = self.scan_loop(step.xs, var)?;
        let mut ctx = FnCtx {
            scoped: types,
            ..Default::default()
        };
        ctx.locals.insert(step.entries[1].1.name, x.at(P_ATOM));
        self.bind_param(step, &mut ctx)?;
        let base = self.scan_base(step.at);
        let outer = self.fctx.replace(ctx);
        let outer_owner = self.owner.replace((base, step.body));
        let mut body = Vec::new();
        let mut failed = None;
        for (name, v) in looped {
            match self.step_value(*v, &mut body) {
                Ok(e) => body.push(format!("{name}[{row}] = {e};")),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        self.owner = outer_owner;
        self.merge_declared(&ctx);
        if let Some(e) = failed {
            return Err(e);
        }
        let mut text = format!("for ({var} in {lo}:{hi}) {{\n");
        for l in &body {
            for part in l.lines() {
                text.push_str(&format!("  {part}\n"));
            }
        }
        text.push('}');
        Ok(text)
    }

    /// Names declared in an inner context belong to the enclosing body too,
    /// so a Stan function never takes them as captured arguments.
    pub(super) fn merge_declared(&mut self, inner: &FnCtx) {
        if let Some(c) = &mut self.fctx {
            c.declared.extend(inner.declared.iter().cloned());
        }
    }

    // ------------------------------------------------------ table broadcast

    /// `f.(…, T, …)` over the table a record scan returns: a row field read
    /// `row.f` is column `T_f`. A projection `row.f` is the column itself;
    /// any other body fills its value row by row.
    pub(super) fn table_broadcast(&mut self, base: &str, id: NodeId) -> Result<String> {
        let c = self.call(id).unwrap();
        let args = c.args[1..].to_vec();
        let over_table = || Refusal::not_yet(Some(id), "a broadcast over a table");
        let tables: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(self.ty(**a), Some(Type::Table { .. })))
            .map(|(k, _)| k)
            .collect();
        let [k] = tables[..] else {
            return Err(over_table());
        };
        let scan = self
            .scan_of(args[k])
            .filter(|&s| self.record_scan(s))
            .ok_or_else(over_table)?;
        let fun = self.user_callee(c.args[0]).ok_or_else(over_table)?;
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != args.len() {
            return Err(Refusal::not_yet(
                Some(id),
                "a user function called with another arity",
            ));
        }
        let columns = self.scan_table(scan)?;
        let row = entries[k].1;
        // A projection: the column itself.
        let b = self.resolve(body);
        if self.head(b) == Some("get")
            && let [x, f] = self.call(b).unwrap().args[..]
            && matches!(self.m.node(x), Node::Ref(r) if *r == row)
            && let Node::Lit(Scalar::Str(f)) = self.m.node(f)
        {
            let col = columns
                .get(&f.to_string())
                .cloned()
                .ok_or_else(over_table)?;
            self.remember(id, &col);
            return Ok(col);
        }
        // Any other body: filled row by row.
        let Some(Type::Table { columns: ctys, .. }) = self.ty(args[k]) else {
            return Err(over_table());
        };
        let row_ty = Type::Record(ctys.clone());
        let n = self.scan_length(self.call(scan).unwrap().args[2])?;
        let n = if n.atomic() {
            n.print()
        } else {
            format!("({})", n.print())
        };
        let var = self.loop_var();
        let mut tys = Vec::new();
        let mut ctx = FnCtx::default();
        let mut failed = None;
        for (j, &a) in args.iter().enumerate() {
            let r = entries[j].1.name;
            if j == k {
                tys.push(row_ty.clone());
                let map = columns
                    .iter()
                    .map(|(f, c)| (f.clone(), format!("{c}[{var}]")))
                    .collect();
                ctx.records.insert(r, map);
                continue;
            }
            let ty = self.ty(a);
            match ty {
                Some(Type::Array { shape, elem }) if shape.len() == 1 => {
                    tys.push(*elem);
                    match self.elem(a, std::slice::from_ref(&var)) {
                        Ok(e) => {
                            ctx.locals.insert(r, e.at(P_ATOM));
                        }
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                Some(t) => {
                    tys.push(t);
                    match self.expr(a) {
                        Ok(e) => {
                            ctx.locals.insert(r, e.at(P_ATOM));
                        }
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                None => {
                    failed = Some(Refusal::not_yet(Some(a), "an untyped argument"));
                    break;
                }
            }
        }
        if let Some(e) = failed {
            self.used.remove(&var);
            return Err(e);
        }
        let out = self
            .instantiate(id, body, &entries, &tys)
            .and_then(|scoped| {
                ctx.scoped = scoped;
                let outer = self.fctx.replace(ctx);
                let mut lines = Vec::new();
                let v = self.step_value(body, &mut lines);
                let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
                self.merge_declared(&ctx);
                v.map(|v| (v, lines))
            });
        self.used.remove(&var);
        let (v, mut lines) = out?;
        let stan = self.fresh(base);
        let t = self.declare(id, None, false, &stan, &[])?;
        lines.push(format!("{stan}[{var}] = {v};"));
        let mut text = format!("for ({var} in 1:{n}) {{\n");
        for l in &lines {
            for part in l.lines() {
                text.push_str(&format!("  {part}\n"));
            }
        }
        text.push('}');
        let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
        self.put(fixed, vec![format!("{};", t.render(&stan)), text]);
        self.record_decl(&stan, t);
        self.remember(id, &stan);
        Ok(stan)
    }

    // ------------------------------------------------- record arguments

    /// The Stan names of a record value's fields: a record expression, a
    /// record argument already bound, a log weight's record argument, or
    /// the last entry of a record scan.
    pub(super) fn record_map(&mut self, a: NodeId) -> Result<HashMap<String, String>> {
        if let Node::Ref(r) = self.m.node(a)
            && r.ns == RefNs::Local
        {
            if let Some(m) = self.fctx.as_ref().and_then(|c| c.records.get(&r.name)) {
                return Ok(m.clone());
            }
            if self.fctx.as_ref().and_then(|c| c.record_local) == Some(r.name) {
                return Ok(self
                    .fields
                    .iter()
                    .map(|(k, v)| (self.name(*k).to_string(), v.clone()))
                    .collect());
            }
        }
        let rec = self.resolve(a);
        if self.head(rec) == Some("record") {
            let named: Vec<(String, NodeId)> = self
                .call(rec)
                .unwrap()
                .named
                .iter()
                .map(|na| (self.name(na.name).to_string(), na.value))
                .collect();
            let mut out = HashMap::new();
            for (f, v) in named {
                let e = self.expr(v)?;
                out.insert(f, e.at(P_ATOM));
            }
            return Ok(out);
        }
        if self.head(rec) == Some("get")
            && let [x, sel] = self.call(rec).unwrap().args[..]
            && let Some(scan) = self.scan_of(x)
            && self.record_scan(scan)
            && self.is_last(scan, sel)
            && let Some(m) = self.scan_last(scan)?
        {
            return Ok(m);
        }
        Err(Refusal::not_yet(Some(a), "a record argument of this form"))
    }

    /// A user function called with a record argument, inlined: Stan functions
    /// take no records. Each record parameter reads the argument's fields.
    pub(super) fn inline_call(&mut self, id: NodeId, fun: NodeId, args: &[NodeId]) -> Result<Ex> {
        let (body, entries) = self.callable_parts(fun)?;
        if entries.len() != args.len() {
            return Err(Refusal::not_yet(
                Some(id),
                "a user function called with another arity",
            ));
        }
        let mut ctx = FnCtx::default();
        let mut tys = Vec::new();
        for ((_, r), &a) in entries.iter().zip(args) {
            let t = self
                .ty(a)
                .ok_or_else(|| Refusal::not_yet(Some(a), "an untyped argument"))?;
            if matches!(t, Type::Record(_)) {
                let m = self.record_map(a)?;
                ctx.records.insert(r.name, m);
            } else {
                let e = self.expr(a)?;
                ctx.locals.insert(r.name, e.at(P_ATOM));
            }
            tys.push(t);
        }
        ctx.scoped = self.instantiate(id, body, &entries, &tys)?;
        let name = self
            .m
            .bindings()
            .find(|(_, b)| b.rhs == fun)
            .map(|(_, b)| self.name(b.name).to_string())
            .unwrap_or_else(|| "inline".to_string());
        let outer = self.fctx.replace(ctx);
        let outer_owner = self.owner.replace((name, body));
        let v = self.expr(body);
        let ctx = std::mem::replace(&mut self.fctx, outer).unwrap();
        self.owner = outer_owner;
        self.merge_declared(&ctx);
        if !ctx.lines.is_empty() {
            let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
            self.put(fixed, ctx.lines);
        }
        v
    }

    /// `true` when a call passes a record argument.
    pub(super) fn has_record_argument(&self, args: &[NodeId]) -> bool {
        args.iter()
            .any(|&a| matches!(self.ty(a), Some(Type::Record(_))))
    }
}

/// A field before the fields it reads; `None` on a cycle.
fn order(
    run: &[&(String, NodeId)],
    reads: &HashMap<String, BTreeSet<String>>,
) -> Option<Vec<(String, NodeId)>> {
    let mut done: Vec<(String, NodeId)> = Vec::new();
    let mut left: Vec<&(String, NodeId)> = run.to_vec();
    while !left.is_empty() {
        // A field goes once no other remaining field reads it.
        let pos = left.iter().position(|(f, _)| {
            !left
                .iter()
                .any(|(g, _)| g != f && reads.get(g).is_some_and(|r| r.contains(f)))
        })?;
        let (f, v) = left.remove(pos);
        done.push((f.clone(), *v));
    }
    Some(done)
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

/// An array of `n` values of `elem`: a `vector` of reals, else an array.
fn column_type(mut t: StanType, n: &str) -> StanType {
    t.bounds = Default::default();
    if t.base == "real" && t.array.is_empty() && t.dims.is_empty() {
        t.base = "vector";
        t.dims = vec![n.to_string()];
    } else {
        t.array.insert(0, n.to_string());
    }
    t
}
