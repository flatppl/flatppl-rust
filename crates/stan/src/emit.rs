//! The structural walk: target → prior and kernel → Stan blocks.

use crate::data::{self, DataSource};
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::{StanType, declare};
use crate::tables;
use crate::{EmitOptions, ManifestParam, Placement};
use flatppl_core::{
    CallHead, Dim, Module, Node, NodeId, Phase, Ref, RefNs, Scalar, ScalarType, Symbol, Type,
    ValueSet,
};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

mod aggregate;
mod disintegrate;
mod elementwise;
mod extents;
mod filter;
mod functions;
mod interval;
mod markovchain;
mod mixture;
mod pattern;
mod pushfwd;
mod ranges;
mod scan;
mod statistics;
mod structure;
mod weight;

/// Expression precedence, loosest first, as Stan parses it.
const P_TERNARY: u8 = 0;
const P_CMP: u8 = 3;
const P_ADD: u8 = 4;
const P_MUL: u8 = 5;
const P_UNARY: u8 = 6;
const P_POW: u8 = 7;
/// A postfix transpose `x'`: an operand of every operator, but Stan parses
/// no index after it (`(x')[i]`).
const P_POSTFIX: u8 = 8;
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
    /// The expression as a stored operand that a reader puts in any context:
    /// an atom or a postfix transpose keeps its precedence, so a product reads
    /// `x'` and an index `(x')[i]`; anything looser is parenthesized.
    pub(crate) fn operand(&self) -> Ex {
        if self.prec >= P_POSTFIX {
            self.clone()
        } else {
            Ex::atom(self.at(P_ATOM))
        }
    }
    /// The transpose: an atom `x'` or a literal `[a, b]'` gives back `x` or
    /// `[a, b]`, not `x''`.
    fn transposed(&self) -> String {
        match self.s.strip_suffix('\'') {
            Some(x) if matches!(self.prec, P_ATOM | P_POSTFIX) || row_literal(x).is_some() => {
                x.to_string()
            }
            _ => format!("{}'", self.at(P_ATOM)),
        }
    }
}

/// The slice `s` of an array of matrices (`array[S] matrix[K, B]`) as a
/// FlatPPL vector or matrix, by which of its three positions are kept:
/// `x[i, j, :]` is a row_vector, `x[:, j, k]` an array of reals, `x[:, j, :]`
/// an array of row_vectors. `None` for `x[:, :, k]`, an array of vectors,
/// which Stan cannot make a matrix of.
pub(super) fn matrices_slice(kept: &[bool], s: String) -> Option<String> {
    match kept {
        [false, false, true] => Some(format!("{s}'")),
        [true, false, false] => Some(format!("to_vector({s})")),
        [true, false, true] => Some(format!("to_matrix({s})")),
        [true, true, false] => None,
        _ => Some(s),
    }
}

/// The elements of a row literal `[a, b]`: the text inside one bracket pair
/// that spans all of `x`.
fn row_literal(x: &str) -> Option<&str> {
    let mut depth = 0i32;
    let whole = x.starts_with('[')
        && x.char_indices().all(|(i, c)| {
            depth += i32::from(c == '[') - i32::from(c == ']');
            depth > 0 || i == x.len() - 1
        });
    whole.then(|| &x[1..x.len() - 1])
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

fn is_complex(ty: &Type) -> bool {
    match ty {
        Type::Scalar(ScalarType::Complex) => true,
        Type::Array { elem, .. } | Type::TVector { elem, .. } => is_complex(elem),
        _ => false,
    }
}

/// How a rank-2 variate and its rank-2 law arguments flatten for one
/// vectorized statement: both orders must agree element by element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flat {
    /// A matrix variate: `to_vector` is column-major.
    Col,
    /// A two-dimensional array variate: `to_array_1d` is row-major.
    Row,
}

/// A table law's variate, from its rows' `variate` and `sizes` columns.
struct RowVariate {
    /// The constrained Stan type of the law's support, if the variate names one.
    support: Option<&'static str>,
    /// The variate axes the size spans (2 for a square matrix type).
    axes: usize,
    size: RowSize,
    /// The `sizes` expression as written, for a refusal.
    expr: String,
}

/// The size of a law's variate, by the argument it reads.
enum RowSize {
    /// The argument itself (`LKJ(n, eta)`).
    Arg(NodeId),
    /// `lengthof(a)`.
    Length(NodeId),
    /// `sizeof(a)[i]`, 1-based.
    Extent(NodeId, usize),
}

/// One prior field of a posterior target.
struct PriorField {
    field: Symbol,
    /// The draw or elementof binding the field denotes, when it is a binding.
    binding: Option<Symbol>,
    /// The law the field is drawn from; `None` for a flat (Lebesgue) part.
    law: Option<NodeId>,
    node: NodeId,
    /// The sets a record-level `truncate(Lebesgue(…), …)` gives the field,
    /// in order. When present they declare it (see `declare_params`).
    sets: Vec<NodeId>,
}

pub(crate) struct Emitter<'m> {
    m: &'m Module,
    opts: &'m EmitOptions,
    used: HashSet<String>,
    /// The module's binding and record field names: loop variables and
    /// block locals avoid them, so a value declared later under one of
    /// these names never meets a loop or local of the same name.
    reserved: HashSet<String>,
    /// FlatPPL binding → the Stan name it is bound to.
    env: HashMap<Symbol, String>,
    /// Prior field or kernel input name → its Stan parameter.
    fields: HashMap<Symbol, String>,
    /// Prior fields a query point sets (`logdensityof(post, record(…))`):
    /// locals with no declared bounds.
    point_fields: HashSet<Symbol>,
    /// The bounds of a point field's declared set its input's declaration
    /// does not state: the support test checks them.
    point_guards: HashMap<Symbol, (Option<String>, Option<String>)>,
    /// Set by a `tilde` that printed a truncation without `T[]` (the
    /// interval-mass form), whose line relies on the variate's declared bounds.
    relies_on_bounds: bool,
    /// The target is `outputs` of a declared signature (§13).
    abi_target: bool,
    /// Stan name → its declaration.
    decls: HashMap<String, StanType>,
    /// Each observed draw → its observed value, the node its data name is
    /// declared from, filled before any observation's statement.
    observed_at: HashMap<Symbol, NodeId>,
    /// Kernel inputs inside the prior, bound by name to prior fields.
    links: Vec<(Symbol, Symbol)>,
    /// Derived prior fields `v = f(x)` of one draw x, by f(x)'s node: the
    /// map's body, x, and x's law (§06: the law is `pushfwd(f, law(x))`).
    derived: HashMap<NodeId, (NodeId, Ref, NodeId)>,
    /// `functions` block sources, deduplicated.
    functions: Vec<String>,
    /// Stan name → its extents as linear forms, axis by axis.
    ext_lin: HashMap<String, Vec<extents::Lin>>,
    /// Size parameter substitutions while sizing a user function's body.
    esubst: Vec<HashMap<Symbol, (NodeId, usize)>>,
    /// The module-binding inputs among each `esubst` frame's keys.
    esubst_self: Vec<HashSet<Symbol>>,
    /// Unsourced data sizes, in union-find classes.
    svars: Vec<extents::SizeVar>,
    /// Each (variate axis, size the law gives) pair that can differ for some
    /// data: [`Self::resolve_sizes`] decides and prints its check.
    size_checks: Vec<extents::SizeCheck>,
    /// Observed values whose `Multinomial` count is their own sum: their
    /// support guard always holds.
    own_sum: HashSet<String>,
    /// The value an observation statement is being written for.
    observing: Option<NodeId>,
    /// Node types of a user function's body being sized.
    sized_scope: Vec<HashMap<NodeId, Type>>,
    /// Set while declaring a `data` value: an extent with no source becomes
    /// a fresh data size.
    data_sizes: bool,
    /// Stan name → its extents and their size names, axis by axis.
    extents: HashMap<String, Vec<(u32, String)>>,
    /// (record binding, field) → data leaf name.
    leaves: HashMap<(Symbol, String), String>,
    in_progress: HashSet<Symbol>,
    /// The binding whose value is being computed, and its root node.
    owner: Option<(String, NodeId)>,
    /// Set while a law fills its flattened arguments.
    flat: Option<Flat>,
    /// Restrict law rows to each law's general pattern (a `joint`'s parts).
    general_rows: bool,
    /// Set while printing an observed value: `splitblocks(A, [1, n])` is
    /// then A, its rows in order.
    rows_ok: bool,
    /// `filter` count names, by the `filter` node.
    filter_counts: HashMap<NodeId, String>,
    /// `ifelse` index arrays already filled, by the `ifelse` node.
    selects: HashMap<NodeId, String>,
    /// Parameters declared from their law's sets (`ordered[K]`): only these
    /// may take a law whose restriction the declaration carries.
    set_declared: HashSet<String>,
    /// `logweighted` log weights of the prior, with their base measures.
    weights: Vec<(NodeId, NodeId)>,
    /// Conditions known to hold while `normalize(weighted(c, M))` builds M
    /// (ruling R1): an `ifelse` on one of them prints its first branch.
    true_conds: Vec<NodeId>,
    /// Values placed under their own name, by node: a law filled twice
    /// (whole and per element) computes each once.
    placed: HashMap<NodeId, String>,
    /// Non-scalar inlined calls placed at module level (see [`FnCtx::inlined`]).
    inlined: Vec<(NodeId, Vec<NodeId>, String)>,
    /// Stan function bodies being emitted: inside one, a data size has no
    /// name. An inlined body at model scope is not one.
    function_depth: usize,
    /// The user function body being emitted.
    fctx: Option<functions::FnCtx>,
    /// Scans emitted at module scope, by node.
    scans: HashMap<NodeId, scan::ScanOut>,
    /// A copy of the module for instantiating function bodies.
    scratch: Option<Module>,
    /// (function, argument types) → Stan function and its captured names.
    fn_memo: HashMap<(NodeId, String), (String, Vec<String>)>,
    /// Function text (name left out) → its Stan name.
    fn_texts: HashMap<String, String>,
    /// Function parameters lifted over whole arrays (see `aggregate::lift`).
    vlocals: Option<HashMap<Symbol, aggregate::Lifted>>,
    sizes: HashMap<u32, Vec<Symbol>>,
    /// The printed name of a `load_module` import (`m_theta` for
    /// `__import_theta_0`; see [`ImportNames`]).
    display: HashMap<Symbol, Symbol>,
    /// Each binding [`Emitter::name`] named: what the program prints, from
    /// which `emit_with_bundle` decides import names.
    named: RefCell<HashSet<Symbol>>,
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
    /// The module binding each manifest parameter stands for, when it has one.
    manifest_bindings: Vec<Option<Symbol>>,
}

/// Every binding name and record field name of `m`, as Stan names.
fn module_names(m: &Module) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut stack = Vec::new();
    for (_, b) in m.bindings() {
        out.insert(sanitize(m.resolve(b.name)));
        stack.push(b.rhs);
    }
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Node::Call(c) = m.node(id) {
            for na in c.named.iter() {
                out.insert(sanitize(m.resolve(na.name)));
            }
        }
        m.for_each_child(id, |k| stack.push(k));
    }
    out
}

/// Node `n` of `work`, the host with its `load_module` members imported, as a
/// node of `host`: a host node is itself; a node of an import is the host
/// node that reads the import; a host value rebuilt around an import is its
/// binding's own value. So a refusal names what the user wrote in the host.
pub(crate) fn host_node(
    host: &Module,
    work: &Module,
    grafted: &flatppl_determinizer::Grafted,
    n: NodeId,
) -> NodeId {
    use flatppl_core::Idx;
    let mut n = n;
    // A reader can itself be a rebuilt host node; each step moves to an
    // older node, and the bound guards the walk.
    for _ in 0..=grafted.readers.len() {
        if n.index() < grafted.host_nodes {
            return n;
        }
        let imported = |root: NodeId| {
            let mut stack = vec![root];
            let mut seen = HashSet::new();
            while let Some(id) = stack.pop() {
                if id == n {
                    return true;
                }
                if !seen.insert(id) {
                    continue;
                }
                if let Node::Ref(r) = work.node(id)
                    && r.ns == RefNs::SelfMod
                    && let Some(b) = work.binding_by_name(r.name)
                    && b.index() >= grafted.host_bindings
                {
                    stack.push(work.binding(b).rhs);
                }
                work.for_each_child(id, |c| stack.push(c));
            }
            false
        };
        if let Some(&(_, reader)) = grafted.readers.iter().find(|(root, _)| imported(*root)) {
            n = reader;
            continue;
        }
        // A host value rebuilt around an import has the shape of the host's
        // own value: walk both together to the node `n` was rebuilt from.
        for (b, binding) in work.bindings() {
            if b.index() >= grafted.host_bindings {
                continue;
            }
            let mut stack = vec![(binding.rhs, host.binding(b).rhs)];
            let mut seen = HashSet::new();
            while let Some((w, h)) = stack.pop() {
                if w == n {
                    return h;
                }
                if !seen.insert(w) {
                    continue;
                }
                let (wc, hc) = (work.node(w).children(), host.node(h).children());
                if wc.len() == hc.len() {
                    stack.extend(wc.into_iter().zip(hc));
                }
            }
        }
        return n;
    }
    n
}

pub(crate) fn sanitize(name: &str) -> String {
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
            reserved: module_names(m),
            env: HashMap::new(),
            fields: HashMap::new(),
            point_fields: HashSet::new(),
            point_guards: HashMap::new(),
            relies_on_bounds: false,
            abi_target: false,
            decls: HashMap::new(),
            observed_at: HashMap::new(),
            links: vec![],
            derived: HashMap::new(),
            functions: vec![],
            extents: HashMap::new(),
            ext_lin: HashMap::new(),
            esubst: vec![],
            esubst_self: vec![],
            sized_scope: vec![],
            svars: vec![],
            size_checks: vec![],
            own_sum: HashSet::new(),
            observing: None,
            display: HashMap::new(),
            named: RefCell::new(HashSet::new()),
            data_sizes: false,
            leaves: HashMap::new(),
            in_progress: HashSet::new(),
            owner: None,
            flat: None,
            general_rows: false,
            rows_ok: false,
            selects: HashMap::new(),
            filter_counts: HashMap::new(),
            set_declared: HashSet::new(),
            weights: vec![],
            true_conds: vec![],
            placed: HashMap::new(),
            inlined: Vec::new(),
            function_depth: 0,
            fctx: None,
            scans: HashMap::new(),
            scratch: None,
            fn_memo: HashMap::new(),
            fn_texts: HashMap::new(),
            vlocals: None,
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
            manifest_bindings: vec![],
        }
    }

    fn record_decl(&mut self, name: &str, t: StanType) {
        if let Some(ctx) = &mut self.fctx {
            ctx.declared.insert(name.to_string());
        }
        self.decls.insert(name.to_string(), t);
    }

    /// The type of a node: its instantiated type inside a function body.
    fn ty(&self, id: NodeId) -> Option<Type> {
        self.sized_scope
            .last()
            .and_then(|s| s.get(&id).cloned())
            .or_else(|| self.fctx.as_ref().and_then(|c| c.scoped.get(&id).cloned()))
            .or_else(|| self.m.type_of(id).cloned())
    }

    fn name(&self, s: Symbol) -> &'m str {
        if self.m.binding_by_name(s).is_some() {
            self.named.borrow_mut().insert(s);
        }
        self.m.resolve(self.display.get(&s).copied().unwrap_or(s))
    }

    /// `true` when the Stan name `name` holds data: a `data` entry (literal or
    /// loaded) or a value declared in `transformed data`, such as a computed
    /// observation.
    fn is_data(&self, name: &str) -> bool {
        let declares = |line: &str| {
            let first = line.lines().next().unwrap_or("");
            let head = first
                .split(" = ")
                .next()
                .unwrap_or("")
                .trim_end_matches(';');
            !first.starts_with(['{', ' ']) && head.rsplit(' ').next() == Some(name)
        };
        self.json.contains_key(name)
            || self.deferred.iter().any(|d| d.stan == name)
            || self.data.iter().any(|l| declares(l))
            || self.tdata.iter().any(|l| declares(l))
    }

    /// A loop variable or block local: a fresh name that is no module
    /// binding or record field name either.
    fn fresh_local(&mut self, base: &str) -> String {
        let base = sanitize(base);
        let mut name = base.clone();
        let mut k = 2;
        while self.used.contains(&name) || self.reserved.contains(&name) {
            name = format!("{base}_{k}");
            k += 1;
        }
        self.used.insert(name.clone());
        name
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

    fn declare(
        &mut self,
        node: NodeId,
        vs: Option<&ValueSet>,
        bounded: bool,
        owner: &str,
        hints: &[Option<NodeId>],
    ) -> Result<StanType> {
        let ty = self
            .ty(node)
            .ok_or_else(|| Refusal::not_yet(Some(node), "an untyped value"))?;
        self.declare_ty(&ty, vs, bounded, node, owner, hints)
    }

    /// [`Self::declare`] for a `data` value: an extent with no symbolic
    /// source is a data size of its own.
    // A set written on the value (`elementof(S)`, `load_data`) is its domain, so it is declared;
    // a law's support is not: observed data stays plain and scores -inf outside it.
    fn declare_data(
        &mut self,
        node: NodeId,
        vs: Option<&ValueSet>,
        bounded: bool,
        owner: &str,
        hints: &[Option<NodeId>],
    ) -> Result<StanType> {
        let outer = std::mem::replace(&mut self.data_sizes, true);
        let t = self.declare(node, vs, bounded, owner, hints);
        self.data_sizes = outer;
        t
    }

    fn declare_ty(
        &mut self,
        ty: &Type,
        vs: Option<&ValueSet>,
        bounded: bool,
        node: NodeId,
        owner: &str,
        hints: &[Option<NodeId>],
    ) -> Result<StanType> {
        // Collect the extents first, then name them, so the size closure does
        // not borrow `self` while `declare` runs.
        if self
            .fctx
            .as_ref()
            .is_some_and(|ctx| ctx.plan_arrays.is_some())
        {
            return declare(ty, vs, bounded, &mut |n| n.to_string(), node);
        }
        // A dynamic extent is declared only by a symbolic size.
        let marked = crate::stan_type::mark_dynamic(ty);
        let ty = &marked;
        let mut extents = Vec::new();
        declare(
            ty,
            vs,
            bounded,
            &mut |n| {
                extents.push(n);
                String::new()
            },
            node,
        )?;
        let naxes = extents.len();
        let lins: Vec<extents::Lin> = extents
            .iter()
            .enumerate()
            .map(|(a, &n)| {
                self.axis_name(n, hints.get(a).copied().flatten(), node, a, naxes, owner)
            })
            .collect();
        for (&n, l) in extents.iter().zip(&lins) {
            if n == crate::stan_type::DYNAMIC && (l.as_const().is_some() || l.terms.is_empty()) {
                return Err(Refusal::not_yet(Some(node), "a dynamic extent"));
            }
        }
        let names: Vec<String> = lins.iter().map(extents::Lin::print).collect();
        self.ext_lin.insert(owner.to_string(), lins);
        self.extents.insert(
            owner.to_string(),
            extents.iter().copied().zip(names.iter().cloned()).collect(),
        );
        let mut k = 0;
        declare(
            ty,
            vs,
            bounded,
            &mut |_| {
                k += 1;
                names[k - 1].clone()
            },
            node,
        )
    }

    /// The axis-size expressions a set names: `cartpow(S, [m, n])` gives
    /// `[m, n]` then S's own axes.
    pub(super) fn set_hints(&self, set: NodeId) -> Vec<Option<NodeId>> {
        let s = self.resolve(set);
        let Some(c) = self.call(s) else {
            return vec![];
        };
        match self.head(s) {
            Some("cartpow") if c.args.len() == 2 => {
                let n = c.args[1];
                let mut out = self.extent_list(n);
                out.extend(self.set_hints(c.args[0]));
                out
            }
            Some("stdsimplex" | "ordered" | "zerosum") if c.args.len() == 1 => {
                vec![Some(c.args[0])]
            }
            _ => vec![],
        }
    }

    /// `n` or `[m, n]` as per-axis size expressions.
    fn extent_list(&self, n: NodeId) -> Vec<Option<NodeId>> {
        let r = self.resolve(n);
        if self.head(r) == Some("vector") {
            self.call(r)
                .unwrap()
                .args
                .iter()
                .map(|&a| Some(a))
                .collect()
        } else {
            vec![Some(n)]
        }
    }

    /// The axis sizes a prior law names: `iid` counts, then the law's own
    /// variate length (`Dirichlet(ones(K))`).
    pub(super) fn law_hints(&self, law: NodeId) -> Vec<Option<NodeId>> {
        let m = self.resolve(law);
        let Some(c) = self.call(m) else {
            return vec![];
        };
        match self.head(m) {
            Some("iid") if c.args.len() == 2 => {
                let mut out = self.extent_list(c.args[1]);
                out.extend(self.law_hints(c.args[0]));
                out
            }
            Some("normalize" | "truncate") => self.law_hints(c.args[0]),
            Some(head) => match self.row_variate(head, m) {
                Some(v) => {
                    let hint = match v.size {
                        RowSize::Arg(a) => Some(a),
                        RowSize::Length(a) => self.length_hint(a),
                        RowSize::Extent(..) => None,
                    };
                    vec![hint; v.axes]
                }
                None => vec![],
            },
            None => vec![],
        }
    }

    /// A table law's variate as its rows state it (`cholesky_factor_corr[{n}]`
    /// with `n` from `sizes`): `None` when the rows of `head` disagree or
    /// the variate names no sized base.
    fn row_variate(&self, head: &str, m: NodeId) -> Option<RowVariate> {
        let rows = tables::law_rows_by_head(head, None);
        let first = rows.first()?;
        if rows
            .iter()
            .any(|r| r.variate != first.variate || r.sizes != first.sizes)
        {
            return None;
        }
        let (base, rest) = first.variate.split_once('[')?;
        if base == "array" || base.contains('<') {
            return None;
        }
        let key = rest.strip_prefix('{')?.strip_suffix("}]")?;
        let expr = first.sizes.iter().find(|(k, _)| k == key)?.1.clone();
        let support = crate::stan_type::support_type(base);
        let axes = match support {
            Some(s) => 1 + usize::from(crate::stan_type::square_support(s)),
            None if base == "vector" => 1,
            None => return None,
        };
        let params = flatppl_infer::distribution_param_names(head)?;
        let args = self.law_args(head, m).ok()?;
        let arg = |p: &str| {
            params
                .iter()
                .position(|q| q == p)
                .and_then(|i| args.get(i).copied())
        };
        // The three `sizes` forms: an argument, `lengthof(a)`, `sizeof(a)[i]`.
        let size = if let Some(p) = expr
            .strip_prefix("lengthof(")
            .and_then(|s| s.strip_suffix(')'))
        {
            RowSize::Length(arg(p)?)
        } else if let Some((p, i)) = expr
            .strip_prefix("sizeof(")
            .and_then(|s| s.split_once(")["))
        {
            RowSize::Extent(arg(p)?, i.strip_suffix(']')?.parse().ok()?)
        } else {
            RowSize::Arg(arg(&expr)?)
        };
        Some(RowVariate {
            support,
            axes,
            size,
            expr,
        })
    }

    /// The length expression of `ones(K)`, `zeros(K)` or `fill(x, K)`.
    fn length_hint(&self, e: NodeId) -> Option<NodeId> {
        let r = self.resolve(e);
        let c = self.call(r)?;
        match (self.head(r)?, c.args.as_ref()) {
            ("ones" | "zeros", [n]) | ("fill", [_, n]) => Some(*n),
            _ => None,
        }
    }

    /// The declared set of a `load_data` call (its second argument).
    fn load_set(&self, call: NodeId) -> Option<NodeId> {
        let c = self.call(call)?;
        c.args.get(1).copied().or_else(|| {
            c.named
                .iter()
                .find(|na| self.name(na.name) == "valueset")
                .map(|na| na.value)
        })
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
        let rhs = self.m.binding(bid).rhs;
        let out = self.at_module(|me| me.bind_new(sym, rhs));
        self.in_progress.remove(&sym);
        let out = out?;
        self.env.insert(sym, out.clone());
        Ok(out)
    }

    fn bind_new(&mut self, sym: Symbol, rhs: NodeId) -> Result<String> {
        // An alias prints as what it names, so it is named only after.
        if let Some(other) = self.self_ref(rhs) {
            return self.bind(other);
        }
        let name = self.name(sym);
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
                let hints = self
                    .load_set(rhs)
                    .map(|s| self.set_hints(s))
                    .unwrap_or_default();
                let t = self.declare_data(rhs, self.m.valueset_of(rhs), true, &stan, &hints)?;
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
            let mut t = self.declare_data(rhs, None, false, &stan, &[])?;
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
        let outer = self.owner.replace((base.to_string(), rhs));
        // `base = scan(…)[n]` read only there: the running state is `base`.
        if self.head(rhs) == Some("get")
            && let [x, s] = self.call(rhs).unwrap().args[..]
        {
            let last = self.scan_last_value(x, s);
            if !matches!(last, Ok(None)) {
                self.owner = outer;
                let e = last?.unwrap();
                self.remember(rhs, &e.s);
                return Ok(e.s);
            }
        }
        // A filter bound whole: its kept elements are the binding.
        if self.head(rhs) == Some("filter") {
            let out = self.filter_value(base, rhs);
            self.owner = outer;
            return out;
        }
        // `rowstack` of a map scan of vectors: its rows filled in place.
        if self.head(rhs) == Some("rowstack")
            && let Some(scan) = self.scan_of(self.call(rhs).unwrap().args[0])
            && let Some(name) = self.stacked_map_scan(base, rhs, scan)?
        {
            self.owner = outer;
            return Ok(name);
        }
        // A user call inlined to a name (a record argument's field): the
        // binding is that name, with no copy.
        if self.is_user_call(rhs).is_some()
            && self.has_record_argument(&self.call(rhs).unwrap().args)
        {
            let e = self.expr(rhs);
            self.owner = outer;
            let e = e?;
            if structure::is_identifier(&e.s) && self.decls.contains_key(&e.s) {
                self.remember(rhs, &e.s);
                return Ok(e.s);
            }
            let outer = self.owner.replace((base.to_string(), rhs));
            let out = self.place(base, rhs, Ok(e));
            self.owner = outer;
            return out;
        }
        // A scan bound whole: its column is the binding.
        if self.head(rhs) == Some("scan") {
            let e = self.expr(rhs);
            self.owner = outer;
            let e = e?;
            if structure::is_identifier(&e.s) {
                self.remember(rhs, &e.s);
                return Ok(e.s);
            }
            let outer = self.owner.replace((base.to_string(), rhs));
            let out = self.place(base, rhs, Ok(e));
            self.owner = outer;
            return out;
        }
        let out = if self.head(rhs) == Some("aggregate") {
            self.aggregate_value(base, rhs)
        } else if self.user_broadcast(rhs) {
            self.user_broadcast_value(base, rhs)
        } else if self.head(rhs) == Some("rowstack")
            && self.user_broadcast(self.call(rhs).unwrap().args[0])
            && self
                .vectorized_broadcast(self.call(rhs).unwrap().args[0])?
                .is_err()
        {
            self.stacked_broadcast(base, rhs)
        } else {
            let attempt = self.expr(rhs);
            self.place(base, rhs, attempt)
        };
        self.owner = outer;
        out
    }

    /// A value without a Stan whole-array expression, nested in a larger
    /// expression: its own variable, filled by loops, keeps the rest
    /// vectorized.
    fn hoist(&mut self, id: NodeId, what: &str, refused: Refusal) -> Result<Ex> {
        if let Some((_, expr)) = self
            .fctx
            .as_ref()
            .and_then(|ctx| ctx.plan_arrays.as_ref())
            .and_then(|arrays| {
                arrays
                    .iter()
                    .find(|(old, _)| self.m.structural_eq(*old, id))
            })
        {
            return Ok(expr.clone());
        }
        match &self.owner {
            Some((_, root)) if *root == id => Err(refused),
            owner => {
                let base = match owner {
                    Some((name, _)) => format!("{name}_{what}"),
                    None => what.to_string(),
                };
                let expr = Ex::atom(self.place(&base, id, Err(refused))?);
                if let Some(arrays) = self.fctx.as_mut().and_then(|ctx| ctx.plan_arrays.as_mut()) {
                    arrays.push((id, expr.clone()));
                }
                Ok(expr)
            }
        }
    }

    /// The name a node's value was placed under, in this context.
    fn recall(&self, id: NodeId) -> Option<String> {
        match &self.fctx {
            Some(c) => c.placed.get(&id).cloned(),
            None => self.placed.get(&id).cloned(),
        }
    }

    fn remember(&mut self, id: NodeId, name: &str) {
        match &mut self.fctx {
            Some(c) => c.placed.insert(id, name.to_string()),
            None => self.placed.insert(id, name.to_string()),
        };
    }

    /// `f.(…)` of a user function.
    fn user_broadcast(&self, id: NodeId) -> bool {
        self.head(id) == Some("broadcast")
            && self.user_callee(self.call(id).unwrap().args[0]).is_some()
    }

    /// Statements of a computed value, in `transformed data` when it is
    /// fixed, else per [`Placement`].
    fn put(&mut self, fixed: bool, lines: Vec<String>) {
        if let Some(ctx) = &mut self.fctx {
            ctx.lines.extend(lines);
        } else if fixed {
            self.tdata.extend(lines);
        } else {
            match self.opts.derived {
                Placement::ModelLocals => self.locals.extend(lines),
                Placement::TransformedParameters => self.tparams.extend(lines),
            }
        }
    }

    /// Place a value whose whole-array expression is `attempt`, falling back
    /// to elementwise loops.
    fn place(&mut self, base: &str, rhs: NodeId, attempt: Result<Ex>) -> Result<String> {
        let fixed = self.m.phase_of(rhs) == Some(Phase::Fixed);
        // Arrays of matrices can be constructed or copied. Stan has no
        // arithmetic on these arrays.
        if self.shape(rhs).is_some_and(|s| s.rank == 3)
            && !matches!(self.head(rhs), Some("array" | "fill" | "vector"))
            && !attempt
                .as_ref()
                .is_ok_and(|e| self.decls.contains_key(&e.s))
        {
            return Err(Refusal::not_yet(Some(rhs), "arithmetic on a rank-3 value"));
        }
        if self.shape(rhs).is_some_and(|s| s.rank == 3)
            && let Err(r) = attempt
        {
            return Err(r);
        }
        let lines = match attempt {
            Ok(e) => {
                let stan = self.fresh(base);
                let t = self.declare(rhs, None, false, &stan, &[])?;
                let line = format!("{} = {};", t.render(&stan), e.s);
                self.record_decl(&stan, t);
                (stan, vec![line])
            }
            Err(r) if r.kind == RefusalKind::NotYetSupported && self.elementwise(rhs) => {
                let stan = self.fresh(base);
                let t = self.declare(rhs, None, false, &stan, &[])?;
                let lines = self.fill_loops(&stan, &t, rhs)?;
                self.record_decl(&stan, t);
                (stan, lines)
            }
            Err(r) => return Err(r),
        };
        let (stan, lines) = lines;
        self.put(fixed, lines);
        self.remember(rhs, &stan);
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
        // The field's declared set names its axes.
        let hints = match self.head(rhs) {
            Some("load_data") => self
                .load_set(rhs)
                .map(|s| self.resolve(s))
                .and_then(|s| self.call(s))
                .and_then(|c| {
                    c.named
                        .iter()
                        .find(|na| self.name(na.name) == fname)
                        .map(|na| na.value)
                })
                .map(|s| self.set_hints(s))
                .unwrap_or_default(),
            _ => vec![],
        };
        let t = self.declare_data(node, vs.as_ref(), true, &stan, &hints)?;
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
        self.ty(id).as_ref().and_then(shape_of)
    }

    /// An arithmetic expression, promoting integer arrays to vectors or matrices.
    fn real(&mut self, id: NodeId) -> Result<Ex> {
        // An integer vector as one column or row of a real matrix.
        let r = self.resolve(id);
        if self.head(r) == Some("addaxes")
            && let [x, lead, trail] = self.call(r).unwrap().args[..]
            && self.shape(x).is_some_and(|s| s.rank == 1 && s.int)
        {
            match (self.m.node(lead), self.m.node(trail)) {
                (Node::Lit(Scalar::Int(0)), Node::Lit(Scalar::Int(1))) => {
                    let e = self.expr(x)?;
                    return Ok(Ex::atom(format!("to_matrix(to_vector({}))", e.s)));
                }
                (Node::Lit(Scalar::Int(1)), Node::Lit(Scalar::Int(0))) => {
                    let e = self.expr(x)?;
                    return Ok(Ex::atom(format!("to_matrix(to_vector({})')", e.s)));
                }
                _ => {}
            }
        }
        let e = self.expr(id)?;
        match self.shape(id) {
            Some(Shape { rank: 1, int: true }) => Ok(Ex::atom(format!("to_vector({})", e.s))),
            Some(Shape { rank: 2, int: true }) => Ok(Ex::atom(format!("to_matrix({})", e.s))),
            // An integer rank-3 value is `array[S, K, B] int`; §09 promotes its
            // entries where a helper takes real matrices.
            Some(Shape { rank: 3, int: true }) => {
                if let Some(core) = tables::core("matrices") {
                    self.add_functions(&core.functions);
                }
                Ok(Ex::atom(format!("flatppl_matrices({})", e.s)))
            }
            _ => Ok(e),
        }
    }

    /// Element `i` of a gather `x[idx]` (one integer-array selector) as
    /// `x[idx[i]]`: indexing the gathered array would copy it per element.
    pub(super) fn gather_at(&mut self, node: NodeId, i: &str) -> Result<Option<String>> {
        if self.head(node) != Some("get") {
            return Ok(None);
        }
        let c = self.call(node).unwrap();
        let (x, sels) = (c.args[0], c.args[1..].to_vec());
        let gathers: Vec<usize> = sels
            .iter()
            .enumerate()
            .filter(|(_, s)| self.shape(**s).is_some_and(|sh| sh.rank == 1 && sh.int))
            .map(|(k, _)| k)
            .collect();
        let [g] = gathers.as_slice() else {
            return Ok(None);
        };
        if sels
            .iter()
            .any(|s| matches!(self.m.node(*s), Node::Const(_) | Node::Lit(Scalar::Str(_))))
        {
            return Ok(None);
        }
        let base = self.expr(x)?;
        let mut parts = Vec::new();
        for (k, s) in sels.iter().enumerate() {
            let e = self.expr(*s)?;
            parts.push(if k == *g {
                format!("{}[{i}]", e.at(P_ATOM))
            } else {
                e.s
            });
        }
        Ok(Some(format!("{}[{}]", base.at(P_ATOM), parts.join(", "))))
    }

    /// A refusal inside the parser-made binding `name` (the tuple of
    /// `a, b = f(…)`) moves to `reader`, the reference the user wrote: the
    /// user never sees a parser-made name.
    fn out_of_synthetic(&self, mut e: Refusal, name: Symbol, reader: NodeId) -> Refusal {
        let Some(b) = self.m.binding_by_name(name) else {
            return e;
        };
        let binding = self.m.binding(b);
        if !binding.synthetic {
            return e;
        }
        let mut stack = vec![binding.rhs];
        while let Some(id) = stack.pop() {
            if Some(id) == e.node {
                e.node = Some(reader);
                return e;
            }
            self.m.for_each_child(id, |k| stack.push(k));
        }
        e
    }

    fn expr(&mut self, id: NodeId) -> Result<Ex> {
        if let Some(e) = self.fctx.as_ref().and_then(|c| c.lifted.get(&id)) {
            return Ok(e.clone());
        }
        if let Some(n) = self.recall(id) {
            return Ok(Ex::atom(n));
        }
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
                "im" => Ok(Ex::atom("1i")),
                other => Err(Refusal::not_yet(
                    Some(id),
                    format!("the constant `{other}`"),
                )),
            },
            Node::Ref(r) => match r.ns {
                RefNs::SelfMod => match self.boundary_ref(id, r.name)? {
                    Some(e) => Ok(e),
                    None => match self.bind(r.name) {
                        Ok(s) => Ok(Ex::atom(s)),
                        Err(e) => Err(self.out_of_synthetic(e, r.name, id)),
                    },
                },
                RefNs::Local => self.local_ref(id, r.name),
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
            ("builtin_logdensityof", [law, parameters, point])
                if matches!(self.m.node(*law), Node::Const(s) if self.name(*s) == "Normal")
                    && self.head(*parameters) == Some("record") =>
            {
                let parameters = self.law_args("Normal", *parameters)?;
                let &[mu, sigma] = parameters.as_slice() else {
                    return Err(Refusal::not_yet(Some(id), "Normal density parameters"));
                };
                let y = self.expr(*point)?;
                let mu = self.expr(mu)?;
                let sigma = self.expr(sigma)?;
                Ok(Ex::atom(format!(
                    "normal_lpdf({} | {}, {})",
                    y.s, mu.s, sigma.s
                )))
            }
            ("add", [a, b]) => bin(self, "+", P_ADD, *a, *b),
            ("sub", [a, b]) => bin(self, "-", P_ADD, *a, *b),
            ("mul", [a, b])
                if matches!(self.ty(*a), Some(Type::TVector { .. }))
                    && matches!(self.ty(*b), Some(Type::Array { shape, elem })
                        if shape.len() == 1 && matches!(*elem, Type::Scalar(_)))
                    && [*a, *b]
                        .iter()
                        .any(|&n| self.ty(n).as_ref().is_some_and(is_complex)) =>
            {
                // Stan's row-times-column overload conjugates its left input.
                // FlatPPL multiplies bilinearly, even when the row is complex.
                let row = self.real(*a)?;
                let col = self.real(*b)?;
                Ok(Ex::atom(format!(
                    "(to_matrix({}) * {})[1]",
                    row.s,
                    col.at(P_MUL + 1),
                )))
            }
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
            ("ifelse", [c0, a, _])
                if self
                    .true_conds
                    .iter()
                    .any(|&t| self.m.structural_eq(t, *c0)) =>
            {
                self.expr(*a)
            }
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
            ("get", [s, i]) if let Some(entry) = self.outer_size_entry(*s, *i) => {
                self.outer_size_at(*i, entry)
            }
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
                if matches!(self.ty(*x), Some(Type::TVector { .. })) {
                    return Err(Refusal::not_yet(Some(id), "a `fill` of a row vector"));
                }
                let v = self.expr(*x)?;
                // `fill(x, k)` of a real vector is k copies of x (§07), typed
                // as an array of vectors: `rep_matrix(x, k)` would give k
                // columns of x.
                if self.shape(*x).is_some_and(|s| s.rank == 1 && !s.int)
                    && self.shape(*n).is_some_and(|s| s.rank == 0)
                {
                    let k = self.expr(*n)?;
                    return Ok(Ex::atom(format!("rep_array({}, {})", v.s, k.s)));
                }
                self.filled(id, &v.at(P_ATOM), *n)
            }
            ("array", [v, size, order]) => self.rank3_array(id, *v, *size, *order),
            // A real vector as one matrix column (`addaxes(u, 0, 1)`) or row
            // (`addaxes(v, 1, 0)`). An integer vector has no Stan integer
            // matrix: it is filled elementwise (or read through `real`).
            ("addaxes", [x, lead, trail])
                if self.shape(*x).is_some_and(|s| s.rank == 1 && !s.int)
                    && matches!(
                        (self.m.node(*lead), self.m.node(*trail)),
                        (Node::Lit(Scalar::Int(0)), Node::Lit(Scalar::Int(1)))
                            | (Node::Lit(Scalar::Int(1)), Node::Lit(Scalar::Int(0)))
                    ) =>
            {
                let e = self.real(*x)?;
                if matches!(self.m.node(*lead), Node::Lit(Scalar::Int(0))) {
                    Ok(Ex::atom(format!("to_matrix({})", e.s)))
                } else {
                    Ok(Ex::atom(format!("to_matrix({}')", e.at(P_ATOM))))
                }
            }
            // `splitblocks(A, [1, n])` observed: the 1×n blocks are A's rows
            // in order, so the observation is A itself (spec §07 "Array and
            // table operations"). Elsewhere it stays refused.
            ("splitblocks", _) if self.rows_ok && self.row_blocks(id).is_some() => {
                let a = self.row_blocks(id).unwrap();
                self.expr(a)
            }
            ("colstack", [cols]) => self.stack(id, *cols, "append_col", false),
            ("rowstack", [rows]) => self.stack(id, *rows, "append_row", true),
            ("%call", _) => match self.is_user_call(id) {
                Some(fun) => self.user_call(id, fun, &args),
                None => self.module_call(id),
            },
            ("scan", _) => self.scan_expr(id, &args),
            ("filter", _) => {
                let base = match &self.owner {
                    Some((name, _)) => format!("{name}_kept"),
                    None => "kept".to_string(),
                };
                Ok(Ex::atom(self.filter_value(&base, id)?))
            }
            ("logdensityof", [m, x]) => self.log_density(id, *m, *x),
            ("sum", [a]) if self.density_broadcast(*a).is_some() => {
                let (m, x) = self.density_broadcast(*a).unwrap();
                self.log_density(id, m, x)
            }
            ("indicesof", [_]) if let Some(r) = self.range(id) => Ok(Ex::atom(r.array())),
            ("sizeof", [x]) if let Some(form) = self.sizeof_form(*x) => {
                let v = self.expr(*x)?;
                let (k, from) = form;
                Ok(Ex::atom(format!(
                    "dims({})[{}:{}]",
                    v.s,
                    from + 1,
                    from + k
                )))
            }
            ("lengthof" | "indicesof" | "indicesof0", [x])
                if self.outer_axes(*x).is_some() || head == "lengthof" && self.is_matrix(*x) =>
            {
                self.outer_size(id, head, *x)
            }
            ("transpose" | "adjoint", [a]) => {
                let e = self.real(*a)?;
                let transposed = e.transposed();
                if head == "adjoint" && self.ty(*a).as_ref().is_some_and(is_complex) {
                    return Ok(Ex::atom(format!("conj({transposed})")));
                }
                // An index onto `x'` needs `(x')[i]`.
                let prec = if transposed.ends_with('\'') {
                    P_POSTFIX
                } else {
                    P_ATOM
                };
                Ok(Ex {
                    s: transposed,
                    prec,
                })
            }
            // `cat(x)` is x; of one scalar, a vector of length 1 (spec §07).
            ("cat", [x]) if self.shape(*x).is_some_and(|s| s.rank == 0) => self.vector(id, &[*x]),
            ("cat", [x]) => self.expr(*x),
            // `cat(1, 2, 3)` of scalars is `vector(1, 2, 3)` (spec §07).
            ("cat", elems)
                if elems.len() >= 2
                    && elems
                        .iter()
                        .all(|&a| self.shape(a).is_some_and(|s| s.rank == 0)) =>
            {
                self.vector(id, elems)
            }
            // functions.ron `cat`: integer arrays of any rank append with
            // append_array (append_row takes vectors and matrices only),
            // left to right.
            ("cat", [first, rest @ ..])
                if !rest.is_empty() && self.shape(id).is_some_and(|s| s.int && s.rank >= 1) =>
            {
                let mut out = self.expr(*first)?.s;
                for a in rest {
                    let r = self.expr(*a)?;
                    out = format!("append_array({out}, {})", r.s);
                }
                Ok(Ex::atom(out))
            }
            // `cat` of real vectors: one `append_row` chain, adjacent vector
            // literals joined (`[h]', [t]'` is `[h, t]'`). Scalars beside
            // vectors are a static error (§06); all scalars are `vector` above.
            ("cat", [_, rest @ ..])
                if !rest.is_empty()
                    && matches!(self.ty(id), Some(Type::Array { shape, elem })
                        if shape.len() == 1 && matches!(*elem, Type::Scalar(ScalarType::Real))) =>
            {
                let mut parts: Vec<String> = Vec::new();
                for &a in &args {
                    let p = self.real(a)?.s;
                    let lit = |t: &str| {
                        t.strip_suffix('\'')
                            .and_then(row_literal)
                            .map(str::to_string)
                    };
                    match (parts.last().and_then(|l| lit(l)), lit(&p)) {
                        (Some(l), Some(r)) => *parts.last_mut().unwrap() = format!("[{l}, {r}]'"),
                        _ => parts.push(p),
                    }
                }
                let mut out = parts[0].clone();
                for p in &parts[1..] {
                    out = format!("append_row({out}, {p})");
                }
                Ok(Ex::atom(out))
            }
            // `cat` appends along the outer axis, left to right (§07): an
            // array of vectors with `append_array` (append_row takes vectors
            // and matrices only), vectors and matrices with `append_row` past
            // the two-argument table template.
            ("cat", [first, rest @ ..])
                if !rest.is_empty()
                    && self.shape(id).is_some_and(|s| !s.int)
                    && let Some(join) = match self.ty(id) {
                        Some(Type::Array { shape, elem }) if shape.len() == 1 => match *elem {
                            Type::Array { .. } => Some("append_array"),
                            Type::Scalar(_) if rest.len() >= 2 => Some("append_row"),
                            _ => None,
                        },
                        Some(Type::Array { shape, .. }) if shape.len() == 2 && rest.len() >= 2 => {
                            Some("append_row")
                        }
                        _ => None,
                    } =>
            {
                let mut out = self.real(*first)?.s;
                for a in rest {
                    let r = self.real(*a)?;
                    out = format!("{join}({out}, {})", r.s);
                }
                Ok(Ex::atom(out))
            }
            ("aggregate", _) => {
                let base = match &self.owner {
                    Some((name, _)) => format!("{name}_agg"),
                    None => "agg".to_string(),
                };
                Ok(Ex::atom(self.aggregate_value(&base, id)?))
            }
            _ => {
                if !c.named.is_empty() {
                    return Err(Refusal::not_yet(
                        Some(id),
                        format!("{head} with keyword arguments"),
                    ));
                }
                let row = self.fn_row(id, head, None)?;
                if row.data_only
                    && args
                        .iter()
                        .any(|&a| self.m.phase_of(a) != Some(Phase::Fixed))
                {
                    return Err(Refusal::not_yet(
                        Some(id),
                        format!("`{head}` of a parameter (Stan needs data arguments)"),
                    ));
                }
                // An integer result (`sum` of an integer array) keeps its
                // integer arguments.
                let int = self.shape(id).is_some_and(|s| s.int);
                let mut parts = Vec::new();
                for a in &args {
                    parts.push(if int { self.expr(*a)? } else { self.real(*a)? });
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
        let row = self.fn_row(id, name, module)?;
        if row.args.len() != parts.len() || row.args.iter().any(|a| a == "...") {
            return Err(Refusal::not_yet(
                Some(id),
                format!("{name} with this arity"),
            ));
        }
        if module.is_none()
            && matches!(name, "real" | "conj" | "abs2")
            && let [arg] = parts.as_slice()
            && let Some(input) = self.call(id).and_then(|call| {
                call.args
                    .get(usize::from(self.head(id) == Some("broadcast")))
            })
            && let Some(ty) = self.ty(*input)
        {
            if is_complex(&ty) {
                if name == "abs2" {
                    // The scalar-only Stan norm overload is not vectorized.
                    return Ok(if matches!(ty, Type::Scalar(_)) {
                        Ex::atom(format!("norm({})", arg.s))
                    } else {
                        Ex {
                            s: format!("square(get_real({0})) + square(get_imag({0}))", arg.s),
                            prec: P_ADD,
                        }
                    });
                }
            } else if name != "abs2" {
                return Ok(
                    if matches!(ty, Type::Scalar(ScalarType::Integer | ScalarType::Boolean)) {
                        Ex::atom(format!("get_real({})", arg.s))
                    } else {
                        arg.clone()
                    },
                );
            }
        }
        // `fmax`/`fmin` return real; `max`/`min` of two integers is an integer.
        if matches!(name, "max" | "min")
            && module.is_none()
            && self.shape(id).is_some_and(|s| s.int)
        {
            return Ok(Ex::atom(format!("{name}({}, {})", parts[0].s, parts[1].s)));
        }
        if row.kind == "helper" {
            for u in &row.uses {
                if let Some(core) = tables::core(u) {
                    self.add_functions(&core.functions);
                }
            }
            self.add_functions(&row.functions);
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

    /// A row's `functions` text, one top-level definition at a time: a
    /// program defines each once.
    pub(super) fn add_functions(&mut self, text: &str) {
        for d in tables::definitions(text) {
            if !self.functions.contains(&d) {
                self.functions.push(d);
            }
        }
    }

    /// `array(v, [S, K, B], [1, 2, 3])`: `S` matrices filled row-major from
    /// consecutive segments of `v`, as `array[S] matrix[K, B]`.
    fn rank3_array(&mut self, id: NodeId, v: NodeId, size: NodeId, order: NodeId) -> Result<Ex> {
        let ints = |me: &Self, n: NodeId| -> Option<Vec<i64>> {
            let r = me.resolve(n);
            if me.head(r) != Some("vector") {
                return None;
            }
            me.call(r)?
                .args
                .iter()
                .map(|&a| match me.m.node(me.resolve(a)) {
                    Node::Lit(Scalar::Int(k)) => Some(*k),
                    _ => None,
                })
                .collect()
        };
        let Some(dims) = ints(self, size) else {
            return Err(Refusal::not_yet(
                Some(id),
                "`array` with a non-literal size",
            ));
        };
        if dims.len() != 3 {
            return Err(Refusal::not_yet(
                Some(id),
                format!("`array` of rank {}", dims.len()),
            ));
        }
        if ints(self, order).as_deref() != Some(&[1, 2, 3][..]) {
            return Err(Refusal::not_yet(Some(id), "`array` with another dimorder"));
        }
        if self.shape(v).is_none_or(|s| s.rank != 1 || s.int) {
            return Err(Refusal::not_yet(
                Some(id),
                "`array` of a value that is not a real vector",
            ));
        }
        let (s, k, b) = (dims[0], dims[1], dims[2]);
        let ve = self.real(v)?;
        // One sample reads the vector whole; more read segments of one name.
        let name = if s == 1 || structure::is_identifier(&ve.s) {
            ve.s.clone()
        } else {
            self.place("flat", v, Ok(ve.clone()))?
        };
        let mats: Vec<String> = (0..s)
            .map(|i| {
                let seg = if s == 1 {
                    name.clone()
                } else {
                    format!("segment({name}, {}, {})", i * k * b + 1, k * b)
                };
                format!("to_matrix({seg}, {k}, {b}, 0)")
            })
            .collect();
        Ok(Ex::atom(format!("{{{}}}", mats.join(", "))))
    }

    /// The matrix A of `splitblocks(A, [1, n])` when n is A's whole row,
    /// one block per row. The inferred type carries spec §07's quotient:
    /// blocks of one row each give an outer shape `[M, 1]` of `[1, n]`
    /// blocks. Any other block shape is not A's rows.
    pub(super) fn row_blocks(&self, id: NodeId) -> Option<NodeId> {
        let r = self.resolve(id);
        if self.head(r) != Some("splitblocks") {
            return None;
        }
        let [a, _] = self.call(r)?.args[..] else {
            return None;
        };
        if self.shape(a).is_none_or(|s| s.rank != 2) {
            return None;
        }
        let Some(Type::Array { shape, elem }) = self.ty(r) else {
            return None;
        };
        let one = |d: &Dim| matches!(d, Dim::Static(1));
        let rows = shape.len() == 2 && one(&shape[1]);
        let block =
            matches!(elem.as_ref(), Type::Array { shape: b, .. } if b.len() == 2 && one(&b[0]));
        (rows && block).then_some(a)
    }

    /// The `functions.ron` row of a call Stan can evaluate: `builtin` or
    /// `helper`. `none` refuses for good, `todo` and unverified rows for now.
    fn fn_row(
        &self,
        id: NodeId,
        name: &str,
        module: Option<&str>,
    ) -> Result<&'static tables::FnRow> {
        let Some(row) = tables::function(name, module) else {
            return Err(Refusal::not_yet(Some(id), name.to_string()));
        };
        match row.kind.as_str() {
            "none" => Err(Refusal::new(
                RefusalKind::NoStanFunction,
                Some(id),
                name.to_string(),
                format!("`{name}` has no Stan form: {}", row.caveat),
            )),
            "builtin" | "helper" if row.status.starts_with("unverified") => Err(Refusal::not_yet(
                Some(id),
                format!("`{name}` (its Stan mapping is unverified)"),
            )),
            "builtin" | "helper" => Ok(row),
            "operator"
                if row.status == "verified"
                    && matches!(name, "isfinite" | "iszero" | "land" | "lall") =>
            {
                Ok(row)
            }
            _ => Err(Refusal::not_yet(Some(id), name.to_string())),
        }
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
        let mut parts = Vec::new();
        for a in &args {
            parts.push(self.real(*a)?);
        }
        self.table_fn(id, fname, Some(&module), parts).map_err(|e| {
            if e.kind.is_never() {
                e
            } else {
                Refusal::not_yet(Some(id), format!("the call `{member}(…)`"))
            }
        })
    }

    /// The standard-module name a callee reference resolves into.
    pub(super) fn standard_module(&self, callee: NodeId) -> Option<String> {
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

    /// The number of outer axes of an array whose elements are arrays (a
    /// vector of vectors), `None` for any other value.
    fn outer_axes(&self, x: NodeId) -> Option<usize> {
        match self.ty(x)? {
            Type::Array { shape, elem }
                if matches!(*elem, Type::Array { .. } | Type::TVector { .. }) =>
            {
                Some(shape.len())
            }
            _ => None,
        }
    }

    /// A rank-2 array of scalars.
    fn is_matrix(&self, x: NodeId) -> bool {
        matches!(self.ty(x), Some(Type::Array { ref shape, ref elem })
            if shape.len() == 2 && matches!(**elem, Type::Scalar(_)))
    }

    /// Where Stan's `dims(x)` is not §07's `sizeof(x)`: the number of
    /// entries and the first `dims` entry (0-based) they start at. A vector of
    /// vectors, which §03 does not read as a matrix, has its outer extents
    /// (`dims` counts through the elements); a `vector[n]` has {n} (`dims`
    /// gives {n, 1}); a `row_vector[n]` has {n} (`dims` gives {1, n}). An
    /// integer array, a matrix and a real array keep `dims`.
    fn sizeof_form(&self, x: NodeId) -> Option<(usize, usize)> {
        if let Some(k) = self.outer_axes(x) {
            return Some((k, 0));
        }
        match self.ty(x)? {
            Type::Array { shape, elem }
                if shape.len() == 1
                    && matches!(*elem, Type::Scalar(ScalarType::Real | ScalarType::Complex)) =>
            {
                Some((1, 0))
            }
            Type::TVector { .. } => Some((1, 1)),
            _ => None,
        }
    }

    /// The number of entries of `sizeof(x)` where the type fixes it: one per
    /// axis of an array of scalars, or of a [`Self::sizeof_form`] value.
    fn sizeof_rank(&self, x: NodeId) -> Option<usize> {
        if let Some((k, _)) = self.sizeof_form(x) {
            return Some(k);
        }
        match self.ty(x)? {
            Type::Array { shape, elem } if matches!(*elem, Type::Scalar(_)) => Some(shape.len()),
            _ => None,
        }
    }

    /// `lengthof`, `indicesof` and `indicesof0` of a vector of vectors, which
    /// §03 does not read as a matrix: they count its elements, where Stan's
    /// `num_elements` counts through them. §07 defines `lengthof` for
    /// vectors and tables only, so a matrix refuses.
    fn outer_size(&mut self, id: NodeId, head: &str, x: NodeId) -> Result<Ex> {
        let Some(k) = self.outer_axes(x) else {
            return Err(Refusal::new(
                RefusalKind::IllFormedModule,
                Some(id),
                "`lengthof` of a matrix".to_string(),
                "§07 defines `lengthof` for vectors and tables, not for a matrix; \
                 `sizeof` gives a matrix's dimensions"
                    .to_string(),
            ));
        };
        let v = self.expr(x)?;
        let n = format!("size({})", v.s);
        Ok(Ex::atom(match (head, k) {
            ("lengthof", 1) => n,
            ("indicesof", 1) => format!("linspaced_int_array({n}, 1, {n})"),
            ("indicesof0", 1) => format!("linspaced_int_array({n}, 0, {n} - 1)"),
            _ => {
                return Err(Refusal::not_yet(
                    Some(id),
                    format!("`{head}` of an array of arrays with more than one outer axis"),
                ));
            }
        }))
    }

    /// `sizeof(x)[n]` with a literal n, where the type fixes the entries of
    /// `sizeof(x)`: x, n, their number and the first `dims` entry (see
    /// [`Self::sizeof_form`]; 0 where `dims` is already `sizeof`). `None` for
    /// any other `get`.
    fn outer_size_entry(&self, s: NodeId, i: NodeId) -> Option<(NodeId, i64, (usize, usize))> {
        let r = self.resolve(s);
        if self.head(r) != Some("sizeof") {
            return None;
        }
        let &[x] = &self.call(r)?.args[..] else {
            return None;
        };
        let form = self
            .sizeof_form(x)
            .or_else(|| self.sizeof_rank(x).map(|k| (k, 0)))?;
        let Node::Lit(Scalar::Int(n)) = *self.m.node(self.resolve(i)) else {
            return None;
        };
        Some((x, n, form))
    }

    /// Entry n of `sizeof(x)`, `dims(x)[from + n]`. An n past its entries is
    /// out of range, which the program can see.
    fn outer_size_at(
        &mut self,
        i: NodeId,
        (x, n, (k, from)): (NodeId, i64, (usize, usize)),
    ) -> Result<Ex> {
        if n < 1 || n > k as i64 {
            let entries = if k == 1 { "entry" } else { "entries" };
            let reason = if self.outer_axes(x).is_some() {
                format!(
                    "index {n} into `sizeof` of a vector of vectors, which has {k} {entries}: \
                     §03 does not read a vector of vectors as a matrix, so its `sizeof` gives \
                     its outer extents only"
                )
            } else {
                let what = if k == 1 {
                    "a vector".to_string()
                } else {
                    format!("a rank-{k} array")
                };
                format!(
                    "index {n} into `sizeof` of {what}, which has {k} {entries}: §07 \
                     `sizeof` gives one entry per axis"
                )
            };
            return Err(Refusal::new(
                RefusalKind::IllFormedModule,
                Some(i),
                "`sizeof` index".to_string(),
                reason,
            ));
        }
        let v = self.expr(x)?;
        Ok(Ex::atom(format!("dims({})[{}]", v.s, from as i64 + n)))
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
        // An integer literal under a real result reads as a real literal.
        let value = if !sh.int && value.parse::<i64>().is_ok() {
            format!("{value}.0")
        } else {
            value.to_string()
        };
        // Rank 3 is `array[S] matrix[K, B]`, as `array(v, [S, K, B])` is.
        if let (3, false, [s, k, b]) = (sh.rank, sh.int, extents.as_slice()) {
            return Ok(Ex::atom(format!(
                "rep_array(rep_matrix({value}, {k}, {b}), {s})"
            )));
        }
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
            let refused = Refusal::not_yet(Some(id), "a stacked array");
            if matches!(&self.owner, Some((_, root)) if *root == id) {
                return Err(refused);
            }
            // A declared `array[K] vector[M]`: one row (column) per step.
            if let Ok(e) = self.expr(original)
                && let Some(t) = self.decls.get(&e.s).cloned()
                && t.array.len() == 1
                && matches!(t.base, "vector" | "complex_vector")
                && t.dims.len() == 1
            {
                let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
                let stan = self.fresh(&base);
                let decl = self.declare(id, None, false, &stan, &[])?;
                let i = self.loop_var();
                let step = if transpose {
                    format!("{stan}[{i}] = {}[{i}]';", e.s)
                } else {
                    format!("{stan}[:, {i}] = {}[{i}];", e.s)
                };
                let k = &t.array[0];
                self.put(
                    fixed,
                    vec![
                        format!("{};", decl.render(&stan)),
                        format!("for ({i} in 1:{k}) {{\n  {step}\n}}"),
                    ],
                );
                self.used.remove(&i);
                self.record_decl(&stan, decl);
                self.remember(id, &stan);
                return Ok(Ex::atom(stan));
            }
            return Ok(Ex::atom(self.place(&base, id, Err(refused))?));
        }
        let items = self.call(list).unwrap().args.to_vec();
        let mut parts = Vec::new();
        for item in items {
            let e = self.real(item)?;
            parts.push(if transpose { e.transposed() } else { e.s });
        }
        // One row or column is still a matrix (`[1, n]`, `[n, 1]`).
        let mut acc = if parts.len() == 1 {
            format!("to_matrix({})", parts[0])
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
            // Postfix `'` binds looser than an index: `([a, b]')[i]`.
            Ok(Ex {
                s: format!("[{}]'", parts.join(", ")),
                prec: P_POW,
            })
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
                let f = f.to_string();
                return Ok(Ex::atom(self.at_module(|me| me.leaf(rec, &f, id))?));
            }
            // A field of a log weight's record argument: its parameter.
            let input = |me: &Self, rf: &flatppl_core::Ref| {
                rf.ns == RefNs::Local
                    || me
                        .fctx
                        .as_ref()
                        .is_some_and(|c| c.self_inputs.contains(&rf.name))
            };
            if let Node::Ref(rf) = self.m.node(x)
                && input(self, rf)
                && self.fctx.as_ref().and_then(|c| c.record_local) == Some(rf.name)
            {
                let f = f.to_string();
                return self.record_field(id, &f);
            }
            // A field of a record argument bound by name (a scan accumulator,
            // a scan parameter, an inlined call's record parameter).
            if let Node::Ref(rf) = self.m.node(x)
                && input(self, rf)
                && let Some(m) = self.fctx.as_ref().and_then(|c| c.records.get(&rf.name))
            {
                return m
                    .get(&**f)
                    .cloned()
                    .ok_or_else(|| Refusal::not_yet(Some(id), format!("the field access `.{f}`")));
            }
            // A column of the table a record scan returns.
            if let Some(scan) = self.scan_of(x)
                && self.record_scan(scan)
            {
                let f = f.to_string();
                let cols = self.scan_table(scan)?;
                return cols
                    .get(&f)
                    .map(|s| Ex::atom(s.clone()))
                    .ok_or_else(|| Refusal::not_yet(Some(id), format!("the field access `.{f}`")));
            }
            // A field of a literal record, or a column of a literal table.
            let r = self.resolve(x);
            if matches!(self.head(r), Some("record" | "table"))
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
        // The last entry of a scan read only there: its running state.
        if let [s] = sel
            && let Some(e) = self.scan_last_value(x, *s)?
        {
            return Ok(e);
        }
        let base = self.expr(x)?;
        let mut idx = Vec::new();
        for s in sel {
            match self.m.node(*s) {
                Node::Const(c) if self.name(*c) == "all" => idx.push(":".to_string()),
                _ => {
                    let computed = matches!(self.head(*s), Some("broadcast" | "indicesof"))
                        && !self.user_broadcast(*s);
                    if computed && let Some(r) = self.range(*s) {
                        // An affine range: a slice, or linspaced_int_array.
                        idx.push(r.index());
                    } else if computed
                        && self.head(*s) == Some("broadcast")
                        && self.shape(*s).is_some_and(|sh| sh.int && sh.rank >= 1)
                    {
                        // Any other computed integer index array is named as one.
                        let refused = Refusal::not_yet(Some(*s), "a computed index array");
                        idx.push(self.hoist(*s, "idx", refused)?.s);
                    } else {
                        idx.push(self.expr(*s)?.s);
                    }
                }
            }
        }
        let s = format!("{}[{}]", base.at(P_ATOM), idx.join(", "));
        if sel.len() == 3 && self.shape(x).is_some_and(|sh| sh.rank == 3) {
            return self.rank3_index(id, x, sel, s);
        }
        // A matrix row is a Stan row_vector; a FlatPPL vector is a Stan vector.
        let matrix = matches!(self.ty(x), Some(Type::Array { shape, elem })
            if shape.len() == 2 && matches!(elem.as_ref(), Type::Scalar(ScalarType::Real | ScalarType::Complex)));
        let row = sel.len() == 2
            && self.shape(sel[0]).is_some_and(|s| s.rank == 0)
            && self.shape(id).is_some_and(|s| s.rank == 1);
        if matrix && row {
            return Ok(Ex {
                s: format!("{s}'"),
                prec: P_POSTFIX,
            });
        }
        Ok(Ex::atom(s))
    }

    /// `x[i, j, k]` of a rank-3 `x` as the Stan type FlatPPL means: a kept
    /// axis gives a vector or a matrix. A real `x` (`array[S] matrix[K, B]`)
    /// yields a row_vector for `x[i, j, :]` and arrays across its array axis;
    /// an integer `x` (`array[S, K, B] int`) yields integer arrays, as
    /// FlatPPL integer arrays print.
    fn rank3_index(&mut self, id: NodeId, x: NodeId, sel: &[NodeId], s: String) -> Result<Ex> {
        let kept: Vec<bool> = sel
            .iter()
            .map(|&k| {
                matches!(self.m.node(k), Node::Const(c) if self.name(*c) == "all")
                    || self.shape(k).is_some_and(|sh| sh.rank >= 1)
            })
            .collect();
        // A real rank-3 value is an array of matrices, an integer one an array.
        let int = self.shape(x).is_some_and(|sh| sh.int);
        let matrices = !int;
        let n = kept.iter().filter(|&&k| k).count();
        // `x[i, j, :]` of an array of matrices is a row transposed.
        let transposed = matches!(kept.as_slice(), [false, false, true]);
        let text = if n == 0 || n == 3 || !matrices {
            // An integer slice is an integer array, as FlatPPL integer arrays print.
            Some(s)
        } else {
            matrices_slice(&kept, s)
        };
        let Some(text) = text else {
            return Err(Refusal::not_yet(
                Some(id),
                "`x[:, :, k]` of a rank-3 array of matrices",
            ));
        };
        let prec = if transposed && matrices && n == 1 {
            P_POSTFIX
        } else {
            P_ATOM
        };
        Ok(Ex { s: text, prec })
    }

    /// `f.(args…)` as one Stan expression over the whole arrays, per the
    /// `functions.ron` row: a `vectorized` row takes the arrays themselves,
    /// a `broadcast` template (`.*`, `./`, `.^`) gives the elementwise
    /// operator. Anything else refuses here and is filled by loops instead.
    fn broadcast(&mut self, id: NodeId, f: NodeId, args: &[NodeId]) -> Result<Ex> {
        // A standard-module function: one call of its row's vector overload
        // when the argument ranks match it.
        if let (Node::Ref(r), Some(module)) = (self.m.node(f), self.standard_module(f)) {
            let fname = self.name(r.name).to_string();
            let row = self.fn_row(id, &fname, Some(&module))?;
            // `overload` lists the array signatures, `;`-separated. A `column`
            // is `addaxes(u, 0, 1)`, passed to Stan as the vector `u`.
            let column = |me: &Self, a: NodeId| -> Option<NodeId> {
                let r = me.resolve(a);
                let c = me.call(r).filter(|_| me.head(r) == Some("addaxes"))?;
                matches!(
                    (me.m.node(c.args[1]), me.m.node(c.args[2])),
                    (Node::Lit(Scalar::Int(0)), Node::Lit(Scalar::Int(1)))
                )
                .then_some(c.args[0])
                .filter(|&u| me.shape(u).is_some_and(|s| s.rank == 1))
            };
            // An exact signature first. Then a scalar may fill a `vector` slot
            // as `rep_vector`, when the result length has a Stan size.
            let len = self
                .ty(id)
                .and_then(|t| elementwise::dims_of(&t).first().copied().flatten());
            let can_promote = self.shape(id).is_some_and(|s| s.rank == 1)
                && (len.is_some() || self.sym_extent(id, 0, 0).is_some());
            let fit = [false, true]
                .into_iter()
                .filter(|&p| !p || can_promote)
                .find_map(|promote| {
                    row.overload
                        .split("; ")
                        .filter(|o| !o.is_empty())
                        .find_map(|o| {
                            let kinds: Vec<&str> = o.split(", ").collect();
                            let ok = kinds.len() == args.len()
                                && args.iter().zip(&kinds).all(|(&a, k)| {
                                    let rank = self.shape(a).map(|s| s.rank);
                                    match (*k, rank) {
                                        ("column", _) => column(self, a).is_some(),
                                        ("vector", Some(0)) => promote,
                                        _ => matches!(
                                            (*k, rank),
                                            ("vector", Some(1))
                                                | ("real", Some(0))
                                                | ("matrix", Some(2))
                                        ),
                                    }
                                });
                            ok.then_some(kinds)
                        })
                });
            let Some(kinds) = fit else {
                let form = if row.overload.is_empty() {
                    "no vector form".to_string()
                } else {
                    format!("vector forms ({})", row.overload)
                };
                return Err(Refusal::not_yet(
                    Some(id),
                    format!("`{fname}.(…)` with these argument shapes ({form})"),
                ));
            };
            let mut parts = Vec::new();
            for (&a, k) in args.iter().zip(&kinds) {
                let a = if *k == "column" {
                    column(self, a).unwrap()
                } else {
                    a
                };
                let e = self.real(a)?;
                if *k == "vector" && self.shape(a).is_some_and(|s| s.rank == 0) {
                    let n = self.size_text(id, 0, len.unwrap_or(0));
                    if len.is_none() && n == "0" {
                        return Err(Refusal::not_yet(
                            Some(id),
                            format!(
                                "`{fname}.(…)` with a scalar where its length has no Stan size"
                            ),
                        ));
                    }
                    parts.push(Ex::atom(format!("rep_vector({}, {n})", e.s)));
                } else {
                    parts.push(e);
                }
            }
            return self.table_fn(id, &fname, Some(&module), parts);
        }
        let Node::Const(fs) = self.m.node(f) else {
            if self.user_callee(f).is_none() {
                return Err(Refusal::not_yet(Some(id), "broadcast of a user function"));
            }
            if let Some(e) = self.row_selector_broadcast(id)? {
                return Ok(e);
            }
            let base = match &self.owner {
                Some((name, _)) => format!("{name}_values"),
                None => "values".to_string(),
            };
            if let Ok(e) = self.vectorized_broadcast(id)? {
                return Ok(e);
            }
            return Ok(Ex::atom(self.looped_broadcast(&base, id)?));
        };
        let fname = self.name(*fs);
        // Stan has no integer-array arithmetic: an integer array is filled by
        // loops.
        if self.shape(id).is_some_and(|s| s.int && s.rank >= 1) {
            if let Some(r) = self.range(id) {
                return Ok(Ex::atom(r.array()));
            }
            let refused = Refusal::not_yet(Some(id), format!("{fname}.(…) of integers"));
            return self.hoist(id, fname, refused);
        }
        let row = tables::function(fname, None)
            .ok_or_else(|| Refusal::not_yet(Some(id), format!("{fname}.(…)")))?;
        // An outer broadcast through `addaxes` (a column against a row): each
        // operand repeated to the full matrix, then the matrix operation.
        if let Some(e) = self.outer_broadcast(id, fname, row, args)? {
            return Ok(e);
        }
        if fname == "ifelse"
            && let Some(e) = self.ifelse_select(id, args)?
        {
            return Ok(e);
        }
        let scalar = |me: &Self, a: NodeId| me.shape(a).is_some_and(|s| s.rank == 0);
        match (fname, args) {
            // The operators keep their precedence-aware printing.
            ("add", [a, b]) if row.vectorized => self.call_expr_bin("+", P_ADD, *a, *b),
            ("sub", [a, b]) if row.vectorized => self.call_expr_bin("-", P_ADD, *a, *b),
            ("neg", [a]) if row.vectorized => {
                let e = self.real(*a)?;
                Ok(Ex {
                    s: format!("-{}", e.at(P_UNARY)),
                    prec: P_UNARY,
                })
            }
            // The broadcast templates' `.*`, `./` and `.^`. A scalar operand
            // takes the scalar operator where Stan's scalar-by-array form is
            // the same elementwise product (`.*` with a scalar side, `./` with
            // a scalar divisor); `.^` stays dotted unless both are scalar.
            ("mul", [a, b]) if row.broadcast.contains(".*") => {
                let op = if scalar(self, *a) || scalar(self, *b) {
                    "*"
                } else {
                    ".*"
                };
                self.call_expr_bin(op, P_MUL, *a, *b)
            }
            ("divide", [a, b]) if row.broadcast.contains("./") => {
                // Stan's int / int truncates; FlatPPL divides as reals.
                // Integer arrays already become vectors through `real`.
                let int_scalar =
                    |me: &Self, x: NodeId| me.shape(x) == Some(Shape { rank: 0, int: true });
                if int_scalar(self, *a) && int_scalar(self, *b) {
                    let l = self.real(*a)?;
                    let r = self.real(*b)?;
                    return Ok(Ex {
                        s: format!("1.0 * {} / {}", l.at(P_MUL), r.at(P_MUL + 1)),
                        prec: P_MUL,
                    });
                }
                let op = if scalar(self, *b) { "/" } else { "./" };
                self.call_expr_bin(op, P_MUL, *a, *b)
            }
            ("pow", [a, b]) if row.broadcast.contains(".^") => {
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
            (name, _) if row.vectorized && row.kind != "operator" => {
                let mut parts = Vec::new();
                for &a in args {
                    parts.push(self.real(a)?);
                }
                self.table_fn(id, name, None, parts)
            }
            (name, _) => {
                let refused = Refusal::not_yet(Some(id), format!("{name}.(…)"));
                if self.elementwise(id) {
                    self.hoist(id, name, refused)
                } else {
                    Err(refused)
                }
            }
        }
    }

    /// `f.(a, b)` of rank 2 where an operand has a singleton axis the result
    /// does not (`addaxes(u, 0, 1)`, a column, or `addaxes(v, 1, 0)`, a row):
    /// that operand is repeated to the full matrix (`rep_matrix(u, n)`,
    /// `rep_matrix(v', m)`, or `rep_matrix(to_vector(X), n)` for any other
    /// column), then the elementwise matrix operation. `None` when no operand
    /// has such an axis or the operation has no matrix form; the caller then
    /// fills by loops.
    fn outer_broadcast(
        &mut self,
        id: NodeId,
        fname: &str,
        row: &tables::FnRow,
        args: &[NodeId],
    ) -> Result<Option<Ex>> {
        let dims = |me: &Self, x: NodeId| -> Option<(u32, u32)> {
            match me.ty(x)? {
                Type::Array { shape, elem }
                    if shape.len() == 2 && matches!(*elem, Type::Scalar(_)) =>
                {
                    match (&shape[0], &shape[1]) {
                        (Dim::Static(a), Dim::Static(b)) => Some((*a, *b)),
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        // A real matrix result: `rep_matrix` and the matrix operations are real.
        let real_matrix = matches!(self.ty(id), Some(Type::Array { shape, elem })
            if shape.len() == 2 && matches!(*elem, Type::Scalar(ScalarType::Real)));
        let Some((r0, r1)) = dims(self, id).filter(|_| real_matrix) else {
            return Ok(None);
        };
        // How each operand reaches the full matrix.
        enum Side {
            Scalar,
            Full,
            Column,
            Row,
        }
        let mut sides = Vec::new();
        for &a in args {
            let rank = self.shape(a).map(|s| s.rank);
            sides.push(match (rank, dims(self, a)) {
                (Some(0), _) => Side::Scalar,
                (_, Some((d0, d1))) if d0 == r0 && d1 == r1 => Side::Full,
                (_, Some((d0, 1))) if d0 == r0 && r1 != 1 => Side::Column,
                (_, Some((1, d1))) if d1 == r1 && r0 != 1 => Side::Row,
                _ => return Ok(None),
            });
        }
        if !sides.iter().any(|s| matches!(s, Side::Column | Side::Row)) {
            return Ok(None);
        }
        let supported = match fname {
            "add" | "sub" | "mul" | "divide" | "pow" => args.len() == 2,
            "neg" => args.len() == 1,
            _ => row.vectorized && row.kind != "operator",
        };
        if !supported {
            return Ok(None);
        }
        let (rows, cols) = (self.size_text(id, 0, r0), self.size_text(id, 1, r1));
        // The vector inside `addaxes(u, …)`, when the operand is one.
        let addaxes_vector = |me: &Self, a: NodeId| -> Option<NodeId> {
            let r = me.resolve(a);
            (me.head(r) == Some("addaxes"))
                .then(|| me.call(r).unwrap().args[0])
                .filter(|&u| me.shape(u).is_some_and(|s| s.rank == 1))
        };
        let mut parts = Vec::new();
        let mut full = Vec::new();
        let mut outer: [Option<Ex>; 2] = [None, None];
        for (&a, side) in args.iter().zip(&sides) {
            let printed = match side {
                Side::Column | Side::Row => match addaxes_vector(self, a) {
                    Some(u) => self.real(u).map(|e| (e, true)),
                    None => self.real(a).map(|e| (e, false)),
                },
                _ => self.real(a).map(|e| (e, false)),
            };
            let (e, vector) = match printed {
                Ok(p) => p,
                Err(r) if r.kind.is_never() => return Err(r),
                Err(_) => return Ok(None),
            };
            // One column times one row is the outer product `u * v'`.
            match side {
                Side::Column if vector => outer[0] = Some(e.clone()),
                Side::Column => outer[0] = Some(Ex::atom(format!("to_vector({})", e.s))),
                Side::Row if vector => outer[1] = Some(Ex::atom(e.transposed())),
                Side::Row => outer[1] = Some(Ex::atom(format!("to_row_vector({})", e.s))),
                _ => {}
            }
            parts.push(match (side, vector) {
                (Side::Column, true) => Ex::atom(format!("rep_matrix({}, {cols})", e.s)),
                (Side::Column, false) => {
                    Ex::atom(format!("rep_matrix(to_vector({}), {cols})", e.s))
                }
                (Side::Row, true) => Ex::atom(format!("rep_matrix({}', {rows})", e.at(P_ATOM))),
                (Side::Row, false) => {
                    Ex::atom(format!("rep_matrix(to_row_vector({}), {rows})", e.s))
                }
                _ => e,
            });
            full.push(!matches!(side, Side::Scalar));
        }
        if fname == "mul"
            && let [Some(u), Some(v)] = outer
        {
            return Ok(Some(binary(&[u, v], "*", P_MUL)));
        }
        let both = full.iter().all(|f| *f);
        Ok(Some(match fname {
            "add" => binary(&parts, "+", P_ADD),
            "sub" => binary(&parts, "-", P_ADD),
            "mul" if both => binary(&parts, ".*", P_MUL),
            "mul" => binary(&parts, "*", P_MUL),
            "divide" if full[1] => binary(&parts, "./", P_MUL),
            "divide" => binary(&parts, "/", P_MUL),
            "pow" => Ex {
                s: format!("{} .^ {}", parts[0].at(P_POW + 1), parts[1].at(P_POW)),
                prec: P_POW,
            },
            "neg" => Ex {
                s: format!("-{}", parts[0].at(P_UNARY)),
                prec: P_UNARY,
            },
            name => self.table_fn(id, name, None, parts)?,
        }))
    }

    /// `ifelse.(cond, a, b)` with a data condition and scalar branches, one
    /// of them not data: `[b, a]'[sel]`, the branch picked by a data index
    /// array filled once in `transformed data`. With data branches the
    /// value is data too and keeps its `transformed data` fill.
    fn ifelse_select(&mut self, id: NodeId, args: &[NodeId]) -> Result<Option<Ex>> {
        let [cond, a, b] = args[..] else {
            return Ok(None);
        };
        let fixed = |me: &Self, x: NodeId| me.m.phase_of(x) == Some(Phase::Fixed);
        let scalar = |me: &Self, x: NodeId| me.shape(x).is_some_and(|s| s.rank == 0);
        if self.fctx.is_some()
            || self.shape(id)
                != Some(Shape {
                    rank: 1,
                    int: false,
                })
            || !fixed(self, cond)
            || self.shape(cond).is_none_or(|s| s.rank != 1)
            || !scalar(self, a)
            || !scalar(self, b)
            || (fixed(self, a) && fixed(self, b))
        {
            return Ok(None);
        }
        let sel = match self.selects.get(&id) {
            Some(sel) => sel.clone(),
            None => {
                let Some(n) = self.sym_extent(id, 0, 0) else {
                    return Ok(None);
                };
                let n = n.print();
                let base = match &self.owner {
                    Some((name, _)) => format!("{name}_select"),
                    None => "select".to_string(),
                };
                let sel = self.fresh(&base);
                let i = self.loop_var();
                let at = self.at_module(|me| me.elem(cond, std::slice::from_ref(&i)));
                self.used.remove(&i);
                let at = at?;
                self.tdata.push(format!("array[{n}] int {sel};"));
                self.tdata.push(format!(
                    "for ({i} in 1:{n}) {{\n  {sel}[{i}] = {} ? 2 : 1;\n}}",
                    at.at(P_CMP)
                ));
                self.selects.insert(id, sel.clone());
                sel
            }
        };
        let (ra, rb) = (self.real(a)?, self.real(b)?);
        // Stan indexes a parenthesized literal, not `[b, a]'[i]`.
        Ok(Some(Ex::atom(format!("([{}, {}]')[{sel}]", rb.s, ra.s))))
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

/// `a op b` at precedence `prec`, left-associative.
fn binary(parts: &[Ex], op: &str, prec: u8) -> Ex {
    Ex {
        s: format!("{} {op} {}", parts[0].at(prec), parts[1].at(prec + 1)),
        prec,
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

/// The target binding: the named one, or the module's single posterior (a
/// `bayesupdate`, or a `restrict` that splits along observed leaves).
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
    let probe = Emitter::new(m, opts);
    let posts: Vec<Symbol> = m
        .bindings()
        .filter(|(_, b)| match probe.head(b.rhs) {
            Some("bayesupdate") => true,
            Some("restrict") => probe.restrict_split(b.rhs).is_ok(),
            _ => false,
        })
        .map(|(_, b)| b.name)
        .collect();
    // With no candidate, a lone `restrict` is what the user meant: its own
    // refusal says why it does not split.
    let restricts: Vec<NodeId> = m
        .bindings()
        .map(|(_, b)| b.rhs)
        .filter(|&rhs| probe.head(rhs) == Some("restrict"))
        .collect();
    if posts.is_empty()
        && let [r] = restricts[..]
        && let Err(refusal) = probe.restrict_split(r)
    {
        return Err(refusal);
    }
    match posts.as_slice() {
        [one] => Ok(*one),
        [] => Err(Refusal::new(
            RefusalKind::NoTarget,
            None,
            "the target",
            "the module has no `bayesupdate` and no `restrict` to observed fields; pass `--target NAME` naming a posterior or likelihood",
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

/// How the bindings a `load_module` import added print: a renamed one under
/// its `display` name, and a positive integer member with a public name as a
/// size (`int<lower=0> len`), as the same binding written in the host does.
#[derive(Default)]
pub(crate) struct ImportNames {
    pub display: HashMap<Symbol, Symbol>,
    pub public: HashSet<Symbol>,
}

pub(crate) fn program(
    m: &Module,
    opts: &EmitOptions,
    rules: &crate::RulePolicy,
) -> Result<crate::StanProgram> {
    let (target, optimised) = prepare(m, opts, &flatppl_infer::ModuleBundle::new(), rules)?;
    program_printed(
        optimised.as_ref().unwrap_or(m),
        opts,
        &ImportNames::default(),
        target,
    )
    .0
}

pub(crate) fn prepare(
    m: &Module,
    opts: &EmitOptions,
    bundle: &flatppl_infer::ModuleBundle,
    rules: &crate::RulePolicy,
) -> Result<(Symbol, Option<Module>)> {
    let target = select_target(m, opts)?;
    let optimised = flatppl_optimise::optimise_with_bundle_and_rules(
        m,
        Some(&[target]),
        flatppl_optimise::Profile::Stan,
        bundle,
        rules,
    );
    Ok((target, optimised))
}

/// What an emission printed: each binding it named and each name it made.
#[derive(Default)]
pub(crate) struct Printed {
    pub named: HashSet<Symbol>,
    pub used: HashSet<String>,
}

/// Emit a selected target and report which names the emission printed.
pub(crate) fn program_printed(
    m: &Module,
    opts: &EmitOptions,
    imports: &ImportNames,
    target: Symbol,
) -> (Result<crate::StanProgram>, Printed) {
    let mut e = Emitter::new(m, opts);
    e.display.clone_from(&imports.display);
    for (_, b) in m.bindings() {
        if imports.public.contains(&b.name)
            && let Node::Lit(Scalar::Int(n)) = m.node(b.rhs)
            && *n > 0
            && let Ok(n) = u32::try_from(*n)
        {
            e.sizes.entry(n).or_default().push(b.name);
        }
    }
    let out = e.finish(target);
    let printed = Printed {
        named: e.named.take(),
        used: std::mem::take(&mut e.used),
    };
    (out, printed)
}

impl Emitter<'_> {
    fn finish(&mut self, target: Symbol) -> Result<crate::StanProgram> {
        self.emit_target(target)?;
        self.query_inputs(target);
        self.abi_parameters()?;
        self.resolve_sizes()?;
        let mut model = self.locals.clone();
        if !model.is_empty() && !(self.prior.is_empty() && self.lik.is_empty()) {
            model.push(String::new());
        }
        model.extend(self.prior.iter().cloned());
        model.extend(self.lik.iter().cloned());
        let data_inputs = self.data_inputs();
        let mut code = String::new();
        block(&mut code, "functions", &self.functions);
        block(&mut code, "data", &self.data);
        block(&mut code, "transformed data", &self.tdata);
        block(&mut code, "parameters", &self.params);
        block(&mut code, "transformed parameters", &self.tparams);
        block(&mut code, "model", &model);
        Ok(crate::StanProgram {
            code,
            data: std::mem::take(&mut self.json),
            deferred_data: std::mem::take(&mut self.deferred),
            manifest: crate::Manifest {
                target: self.m.resolve(target).to_string(),
                data: data_inputs,
                parameters: std::mem::take(&mut self.manifest),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transposed() -> Ex {
        Ex {
            s: "x'".into(),
            prec: P_POSTFIX,
        }
    }

    /// Stan 2.39 parses `x'` as an operand of every operator and of a second
    /// transpose; only an index after it needs `(x')`.
    #[test]
    fn a_postfix_transpose_is_parenthesized_only_under_an_index() {
        let t = transposed();
        assert_eq!(format!("{} * y", t.at(P_MUL)), "x' * y");
        assert_eq!(format!("{} ^ 2", t.at(P_POW + 1)), "x' ^ 2");
        assert_eq!(format!("{} .^ 2", t.at(P_POW + 1)), "x' .^ 2");
        assert_eq!(format!("-{}", t.at(P_UNARY)), "-x'");
        assert_eq!(t.transposed(), "x");
        assert_eq!(format!("{}[i]", t.at(P_ATOM)), "(x')[i]");
    }

    /// A stored local keeps a postfix transpose, so each reader asks its own
    /// level; a looser expression is stored parenthesized, as before.
    #[test]
    fn a_stored_operand_keeps_a_postfix_transpose() {
        let t = transposed().operand();
        assert_eq!(t.at(P_MUL), "x'");
        assert_eq!(t.at(P_ATOM), "(x')");
        let sum = Ex {
            s: "a + b".into(),
            prec: P_ADD,
        }
        .operand();
        assert_eq!(sum.at(P_MUL), "(a + b)");
        assert_eq!(sum.at(P_ATOM), "(a + b)");
    }
}
