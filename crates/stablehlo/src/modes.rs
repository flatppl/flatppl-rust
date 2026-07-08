//! The mode builders — `emit_logdensity` (this task) and, from Task 6,
//! `emit_sample` — that turn a determinized FlatPDL [`Module`] into one
//! complete `func.func` StableHLO module.
//!
//! **Free parameters vs. fixed data.** A determinized module's top-level
//! bindings carry a `Phase` (spec §04): `elementof(...)`-declared parameters
//! are `Phase::Parameterized`, everything derived from them (including the
//! logdensity query itself) is *also* `Parameterized` (phase is a taint over
//! the whole dependent subtree, not a leaf marker), and already-pinned
//! observed data is `Phase::Fixed`. [`emit_logdensity`] therefore cannot use
//! "phase is `Parameterized`" alone to find the free parameters — it also
//! checks that the binding's RHS is *structurally* a bare `elementof(...)`
//! call, i.e. a parameter *declaration* rather than a computation that
//! merely depends on one (see [`is_free_param`]).
//!
//! Each free parameter becomes a fresh `func.func` argument
//! (`%argN : tensor<...>`, in top-level binding/source order — a
//! deterministic order derived from the module itself), and its RHS
//! `NodeId` is [`Emitter::bind`]-seeded to that argument's [`Value`] *before*
//! the query is walked. This is essential, not cosmetic: `elementof` has no
//! op-map lowering (it is a declaration, not a computation), so if the query
//! walk ever reached an unbound `elementof(...)` node directly, it would
//! refuse. Pre-binding means a `Ref` back to the parameter resolves straight
//! to the pre-allocated `Value` via `Emitter::lower_node`'s memo, and the
//! `elementof` node itself is never visited.
//!
//! Fixed data needs no special handling here: `Emitter::lower_node`'s
//! ordinary `Lit` dispatch already turns a fixed scalar leaf into a
//! `stablehlo.constant` when the query walk reaches it.
//!
//! **Finding the query.** Nothing in FlatPDL marks a binding as "the"
//! logdensity output — constructing the query is a step upstream of this
//! crate (the CLI verb / the testsuite harness, per the design doc). Every
//! `flatppl-determinizer` density fixture and golden test follows the same
//! shape, though: the density expression (`lp = logdensityof(...)`, or
//! equivalent) is the LAST public top-level binding in source order. This
//! module relies on that convention rather than re-deriving one.
//!
//! That convention is silent-wrong-result-capable: [`Module`]'s own doc
//! disclaims that binding order carries spec meaning, so a module with any
//! public binding *after* the density expression (a diagnostic/auxiliary
//! value) would otherwise have [`emit_logdensity`] lower that trailing
//! binding instead — producing a well-formed `tensor<f32>` module with wrong
//! semantics, no refusal. [`emit_logdensity`] therefore guards the selected
//! output with a cheap structural check ([`contains_logdensityof_call`]):
//! the binding's RHS subtree must contain at least one `builtin_logdensityof`
//! call, or it refuses rather than mis-lower. [`emit_sample`] applies the
//! analogous guard ([`contains_sample_call`]) over `builtin_sample` — but,
//! unlike [`contains_logdensityof_call`], that guard must also follow
//! `(%ref self x)` leaves to `x`'s bound RHS, TRANSITIVELY: a record/
//! hierarchical `@sample` forward model's query is a `record(...)` whose
//! fields are bare refs to bindings the determiniser has rewritten in place
//! (`flatppl_determinizer::sample::lower_shared_record_sample`), with the
//! actual `builtin_sample` call sitting one or more binding-hops away on
//! each ref's resolved RHS — `Node::for_each_child` does not descend
//! through a `Ref` at all, so a purely structural walk never reaches it.
//! See [`contains_sample_call`]'s own doc comment for the walk.
//!
//! **`@sample`.** [`emit_sample`] mirrors [`emit_logdensity`]'s structure
//! exactly — same free-parameter/fixed-data binding loop, same
//! last-public-binding query convention, an analogous (but ref-following,
//! see above) query-output guard — but the query's RHS is not itself a bare
//! `builtin_sample` call: a value-terminal `rand(rng, lawof(x))`
//! (`flatppl_determinizer::sample`) lowers to
//! `get0(builtin_sample(rng, ctor, kernel_input), 0)`, projecting
//! the drawn-value slot of the sampled `(value, new_rngstate)` pair. Rather
//! than special-casing that shape here, [`Emitter::lower_node`]'s dispatch
//! (`emitter.rs`) recognizes a `get0`/`get` projection of a `builtin_sample`
//! call structurally and reads the registry's already-computed drawn value
//! straight through — see `Emitter::sample_tuple_slot`'s doc comment — so
//! [`emit_sample`] can lower its query the same generic way
//! [`emit_logdensity`] does.

use std::collections::HashSet;

use flatppl_core::{CallHead, Module, Node, NodeId, Phase, Ref, RefNs, Scalar};

use crate::EmitOptions;
use crate::emitter::Emitter;
use crate::mlir::{MlirTy, Value};
use crate::refuse::EmitError;
use crate::types::mlir_type_of;

/// Emit `@logdensity` for a determinized module `m` (see the module doc
/// comment for the free-param/fixed-data/query-finding rules). `m` is
/// assumed already FlatPDL-conformant — [`crate::emit`] (the mode router)
/// checks that once, up front.
pub fn emit_logdensity(m: &Module, opts: &EmitOptions) -> Result<String, EmitError> {
    let mut e = Emitter::new(m, opts.dtype);

    // Free parameters, in binding (source) order: bind each BEFORE the query
    // is walked (see module doc comment).
    let mut args: Vec<(String, MlirTy)> = Vec::new();
    for (_, binding) in m.bindings() {
        if !is_free_param(m, binding.rhs) {
            continue;
        }
        let name = format!("%arg{}", args.len());
        let ty = mlir_type_of(m, binding.rhs, opts.dtype)?;
        e.bind(
            binding.rhs,
            Value {
                ssa: name.clone(),
                ty: ty.clone(),
            },
        );
        args.push((name, ty));
    }

    let query = m.public_bindings().last().ok_or_else(|| {
        EmitError::whole("module has no public binding to emit as the logdensity query")
    })?;
    let query_rhs = query.1.rhs;

    // Guard the "last public binding" convention (see module doc comment):
    // refuse rather than silently lower a trailing non-density binding.
    if !contains_logdensityof_call(m, query_rhs) {
        return Err(EmitError::at(
            query_rhs,
            "selected query output contains no density term (builtin_logdensityof); \
             FlatPDL has no query marker — cannot identify the logdensity output",
        ));
    }

    let result = e.lower_node(query_rhs)?;
    Ok(e.finish("logdensity", &args, &[&result]))
}

/// Whether the subtree rooted at `id` (the node itself, or any descendant
/// reached via [`Module::for_each_child`]) contains a `Call` whose head is
/// the builtin `builtin_logdensityof` — the structural signal that `id` is
/// actually a density term. See the module doc comment on why
/// [`emit_logdensity`] cannot trust binding order alone.
fn contains_logdensityof_call(m: &Module, root: NodeId) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if is_builtin_call(m, id, "builtin_logdensityof") {
            return true;
        }
        m.for_each_child(id, |c| stack.push(c));
    }
    false
}

/// Emit `@sample` for a determinized module `m` — see the module doc comment
/// for how this mirrors [`emit_logdensity`] (free-param/fixed-data binding
/// loop, last-public-binding query convention, structural query-output
/// guard) and how its query's `get0(builtin_sample(...), 0)` shape is
/// resolved generically via [`Emitter::lower_node`]'s dispatch. `m` is
/// assumed already FlatPDL-conformant — [`crate::emit`] (the mode router)
/// checks that once, up front.
pub fn emit_sample(m: &Module, opts: &EmitOptions) -> Result<String, EmitError> {
    let mut e = Emitter::new(m, opts.dtype);

    let query = m.public_bindings().last().ok_or_else(|| {
        EmitError::whole("module has no public binding to emit as the sample query")
    })?;
    let query_rhs = query.1.rhs;

    // Guard the "last public binding" convention (see the module doc
    // comment): refuse rather than silently lower a trailing non-sample
    // binding. Checked BEFORE `find_rng_source` so a non-sample query gets
    // the precise "no sample term" refusal, not "no rng source".
    if !contains_sample_call(m, query_rhs) {
        return Err(EmitError::at(
            query_rhs,
            "selected query output contains no sample term (builtin_sample); \
             FlatPDL has no query marker — cannot identify the @sample output",
        ));
    }

    // Bind the FlatPDL rng source to `%key` (spec §07 rng ABI: `rnginit`'s
    // seed→state math is NOT lowered — the source binds directly to the
    // threaded key). `%key` is func arg 0.
    let key_ty = MlirTy::Key;
    let key_name = "%key".to_string();
    let src = find_rng_source(m, query_rhs).ok_or_else(|| {
        EmitError::at(
            query_rhs,
            "no rng source to bind to %key: every builtin_sample's rng arg \
             resolves to another sample's advanced key, so there is no \
             rnginit/external(rngstates) source to thread from",
        )
    })?;
    e.bind(
        src,
        Value {
            ssa: key_name.clone(),
            ty: key_ty.clone(),
        },
    );
    let mut args: Vec<(String, MlirTy)> = vec![(key_name, key_ty)];

    // Free parameters, in binding (source) order — identical to
    // `emit_logdensity`'s loop (see the module doc comment): a `@sample`
    // forward model can still have `elementof`-declared hyperparameters, in
    // which case they become `%argN` func args (numbered independently of
    // `%key`) just as they do for `@logdensity`. A fixed-hyperparameter prior
    // (the common case) simply yields no extra args.
    let mut nfree = 0;
    for (_, binding) in m.bindings() {
        if !is_free_param(m, binding.rhs) {
            continue;
        }
        let name = format!("%arg{nfree}");
        nfree += 1;
        let ty = mlir_type_of(m, binding.rhs, opts.dtype)?;
        e.bind(
            binding.rhs,
            Value {
                ssa: name.clone(),
                ty: ty.clone(),
            },
        );
        args.push((name, ty));
    }

    // Lower the query's value component (spec §07: `rand` yields
    // `(value, new_rstate)`; a destructured query is a bare `tuple(v, r)`, of
    // which only `v` is the drawn value), then thread out the final advanced
    // key (`Emitter::cur_key` after the whole draw chain) as the second result.
    let value = e.lower_node(query_value_component(m, query_rhs))?;
    let final_key = e.cur_key();
    Ok(e.finish("sample", &args, &[&value, &final_key]))
}

/// The value component of a `@sample` query. A destructured `rand` whose
/// binding is used directly as the output is a bare `tuple(value, advanced_rng)`
/// (the determiniser's [`flatppl_determinizer::sample`] shape) — lower only the
/// `value` slot; every other query shape (a value-terminal `get0(sample, 0)`,
/// a `record(...)`, a bare ref) is already the value and is returned unchanged.
fn query_value_component(m: &Module, query_rhs: NodeId) -> NodeId {
    if let Node::Call(c) = m.node(query_rhs) {
        if let CallHead::Builtin(sym) = c.head {
            if m.resolve(sym) == "tuple" && c.args.len() == 2 {
                return c.args[0];
            }
        }
    }
    query_rhs
}

/// Find the FlatPDL rng SOURCE reachable from the `@sample` query — the
/// `builtin_sample` whose rng argument does NOT (transitively) resolve to
/// another sample's advanced-key slot, i.e. the `rnginit(...)`/
/// `external(rngstates)` that seeds the whole threaded chain (spec §07). The
/// returned [`NodeId`] is that source sample's rng-argument node, which
/// [`emit_sample`] binds to `%key` so [`crate::registry::lower_sample`]'s
/// `e.lower_node(rng)` resolves straight to the func argument (the `rnginit`
/// node itself is never lowered — its seed→state math is out of scope).
///
/// `None` when no such source exists (every sample's rng arg is another
/// sample's advanced key — a cycle, or a model whose only rng comes from a
/// slot with no root): [`emit_sample`] then refuses rather than silently
/// dropping the key. In a well-formed threaded chain there is exactly one
/// source; the first found (in reachability-walk order) is returned.
fn find_rng_source(m: &Module, query_rhs: NodeId) -> Option<NodeId> {
    for sample in collect_sample_calls(m, query_rhs) {
        if let Node::Call(c) = m.node(sample) {
            if let Some(&rng_arg) = c.args.first() {
                if !derives_from_sample(m, rng_arg) {
                    return Some(rng_arg);
                }
            }
        }
    }
    None
}

/// Collect every `builtin_sample` [`NodeId`] reachable from `root`, following
/// `(%ref self x)` leaves to their bound RHS (transitively) as well as
/// [`Module::for_each_child`] — the same reach as [`contains_sample_call`]
/// (a record/hierarchical query's samples sit one or more binding-hops away on
/// ref-resolved RHSs). Deduplicated (a sample projected as both
/// `get0(s, 0)` and `get0(s, 1)` is one node) via the visited set.
fn collect_sample_calls(m: &Module, root: NodeId) -> Vec<NodeId> {
    let mut stack = vec![root];
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut samples = Vec::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if is_builtin_call(m, id, "builtin_sample") {
            samples.push(id);
        }
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            if let Some(bid) = m.binding_by_name(*name) {
                stack.push(m.binding(bid).rhs);
            }
        }
        m.for_each_child(id, |c| stack.push(c));
    }
    samples
}

/// Whether `id` — resolved through `(%ref self x)` hops and literal `tuple`/
/// `get`/`get0` projections — ultimately derives from a `builtin_sample`
/// (i.e. is some sample's drawn value or advanced key). Used by
/// [`find_rng_source`] to distinguish a chained sample's rng arg (a prior
/// sample's advanced-key slot, `get0(sample, 1)` — possibly via a
/// `tuple(...)` the determiniser built and a 1-based `get(_, 2)`) from a true
/// source (`rnginit`/`external`, which is not sample-derived).
fn derives_from_sample(m: &Module, id: NodeId) -> bool {
    let id = resolve_self_ref(m, id);
    let Node::Call(c) = m.node(id) else {
        return false;
    };
    let head = match c.head {
        CallHead::Builtin(sym) => m.resolve(sym),
        CallHead::User(_) => return false,
    };
    let base = match head {
        "get0" => 0,
        "get" => 1,
        _ => return false,
    };
    let [container, index] = match <[NodeId; 2]>::try_from(&c.args[..]) {
        Ok(pair) => pair,
        Err(_) => return false,
    };
    let container = resolve_self_ref(m, container);
    if is_builtin_call(m, container, "builtin_sample") {
        return true;
    }
    // `get`/`get0` of a literal `tuple(...)` → recurse into the projected slot.
    if let Node::Call(tc) = m.node(container) {
        if let CallHead::Builtin(sym) = tc.head {
            if m.resolve(sym) == "tuple" {
                if let Node::Lit(Scalar::Int(sel)) = m.node(index) {
                    let idx = sel - base;
                    if idx >= 0 && (idx as usize) < tc.args.len() {
                        return derives_from_sample(m, tc.args[idx as usize]);
                    }
                }
            }
        }
    }
    false
}

/// Resolve `id` through `(%ref self x)` hops transitively (a cycle-guarded
/// generalization of the emitter's one-hop `resolve_ref_one`), returning the
/// first non-`SelfMod`-ref node. Used by [`derives_from_sample`] to see
/// through the determiniser's binding chains (`s2 = get(__0x1, 2)`, etc.).
fn resolve_self_ref(m: &Module, id: NodeId) -> NodeId {
    let mut cur = id;
    let mut seen: HashSet<NodeId> = HashSet::new();
    while seen.insert(cur) {
        match m.node(cur) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => match m.binding_by_name(*name) {
                Some(bid) => cur = m.binding(bid).rhs,
                None => return cur,
            },
            _ => return cur,
        }
    }
    cur
}

/// Whether the subtree rooted at `id` contains a `Call` whose head is the
/// builtin `builtin_sample` — the [`emit_sample`] analogue of
/// [`contains_logdensityof_call`].
///
/// Unlike [`contains_logdensityof_call`], this walk also follows
/// `(%ref self x)` leaves to `x`'s bound RHS (mirroring the ref-resolution
/// rule in [`crate::emitter::Emitter::resolves_to_builtin_sample`]),
/// TRANSITIVELY rather than one hop. A record/hierarchical `@sample` forward
/// model's query is a `record(...)` whose fields are bare `(%ref self mu)`
/// leaves — `Node::for_each_child` does not descend through a `Ref` at all,
/// and the rewritten `builtin_sample` sits one OR MORE binding-hops away on
/// `mu`'s (and, for a shared/hierarchical latent, `mu`'s own dependency's)
/// RHS (`flatppl_determinizer::sample::lower_shared_record_sample`), so a
/// single-hop resolution is not enough. A `HashSet` of already-visited
/// `NodeId`s guards against a reference cycle (none should arise from a
/// well-formed FlatPDL module — bindings form a DAG — but the guard costs
/// nothing and this walk has no other termination proof).
fn contains_sample_call(m: &Module, root: NodeId) -> bool {
    let mut stack = vec![root];
    let mut seen: HashSet<NodeId> = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if is_builtin_call(m, id, "builtin_sample") {
            return true;
        }
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        {
            if let Some(bid) = m.binding_by_name(*name) {
                stack.push(m.binding(bid).rhs);
            }
        }
        m.for_each_child(id, |c| stack.push(c));
    }
    false
}

/// A free-parameter declaration: `Phase::Parameterized` (spec §04 "Phase of
/// an expression") AND structurally a bare `elementof(...)` call. The phase
/// check alone is not enough — see the module doc comment on why phase is a
/// taint over the whole dependent subtree, not a parameter-leaf marker.
fn is_free_param(m: &Module, rhs: NodeId) -> bool {
    m.phase_of(rhs) == Some(Phase::Parameterized) && is_builtin_call(m, rhs, "elementof")
}

/// Whether `id` is (structurally) a `Call` whose head is the builtin named
/// `name` — shared by [`is_free_param`]'s `elementof(...)` check and
/// [`contains_logdensityof_call`]'s `builtin_logdensityof` check.
fn is_builtin_call(m: &Module, id: NodeId, name: &str) -> bool {
    matches!(
        m.node(id),
        Node::Call(c) if matches!(
            c.head,
            CallHead::Builtin(sym) if m.resolve(sym) == name
        )
    )
}
