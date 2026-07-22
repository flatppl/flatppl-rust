//! The mode builders — [`emit_logdensity_abi`] and [`emit_sample_abi`] — that
//! turn a determinized FlatPDL [`Module`] into one complete `func.func`
//! StableHLO module, driven by the model's explicit `inputs`/`outputs`
//! compilation ABI (see [`Abi`]/[`read_abi`]).
//!
//! **Free parameters vs. fixed data.** A determinized module's top-level
//! bindings carry a `Phase` (spec §04): `elementof(...)`-declared parameters
//! are `Phase::Parameterized`, everything derived from them (including the
//! query itself) is *also* `Parameterized` (phase is a taint over the whole
//! dependent subtree, not a leaf marker), and already-pinned observed data is
//! `Phase::Fixed`. The phase alone therefore cannot identify the free
//! parameters — [`is_free_param`] also checks that the binding's RHS is
//! *structurally* a bare `elementof(...)` call, i.e. a parameter *declaration*
//! rather than a computation that merely depends on one.
//!
//! Each ABI input becomes a fresh `func.func` argument (`%argN : tensor<...>`,
//! in the declared `inputs` order — no source-order dependence), and its RHS
//! `NodeId` is [`Emitter::bind`]-seeded to that argument's [`Value`] *before*
//! the query is walked. This is essential, not cosmetic: `elementof` has no
//! op-map lowering (it is a declaration, not a computation), so if the query
//! walk ever reached an unbound `elementof(...)` node directly, it would
//! refuse. Pre-binding means a `Ref` back to the parameter resolves straight
//! to the pre-allocated `Value` via `Emitter::lower_node`'s memo, and the
//! `elementof` node itself is never visited. Fixed data needs no special
//! handling: `Emitter::lower_node`'s ordinary `Lit` dispatch turns a fixed
//! scalar leaf into a `stablehlo.constant` when the query walk reaches it.
//!
//! **Designating the query.** Nothing in FlatPDL marks a binding as "the"
//! output; the query is designated *explicitly* by the reserved
//! `outputs = (…)` binding ([`read_abi`]). The last-public-binding heuristic
//! this module once used — which was silent-wrong-result-capable, since
//! [`Module`]'s own doc disclaims that binding order carries spec meaning —
//! has been removed: a module declaring neither `inputs` nor `outputs` is
//! refused by [`crate::emit`] rather than guessed at. [`emit_logdensity_abi`]
//! lowers each declared output; `inputs` is exhaustive over the module's
//! `elementof` parameters (a parameter missing from `inputs` refuses).
//!
//! **`@sample`.** [`emit_sample_abi`] mirrors [`emit_logdensity_abi`] but
//! threads the rng key: the leading `%key : tensor<2xui64>` argument (spec §07
//! `rand(rstate, m) -> (value, new_rstate)`) is bound from the rng source
//! reachable from the declared output ([`find_rng_source`]) — NOT drawn from
//! `inputs`, and the rng-source binding is exempt from the inputs-exhaustiveness
//! check — and the function returns the two-result `(value, new_key)` pair. The
//! declared sample output's value component (a value-terminal `rand(rng,
//! lawof(x))` lowers to `get0(builtin_sample(rng, ctor, kernel_input), 0)`) is
//! projected by [`Emitter::lower_node`]'s dispatch, which recognizes a
//! `get0`/`get` projection of a `builtin_sample` call structurally and reads
//! the registry's already-computed drawn value straight through (see
//! `Emitter::sample_tuple_slot`).

use std::collections::HashSet;

use flatppl_core::{CallHead, Module, Node, NodeId, Phase, Ref, RefNs, Scalar, Symbol};

use crate::EmitOptions;
use crate::emitter::Emitter;
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::refuse::EmitError;
use crate::types::mlir_type_of;

/// The compilation ABI declared by the reserved `inputs = …` / `outputs = …`
/// top-level bindings (design doc
/// `docs/superpowers/specs/2026-07-17-inputs-outputs-abi-design.md`): an
/// explicit, ordered argument/result list for the emitted `func.func`. This
/// is the sole way a query and its arguments are designated — the legacy
/// source-order / last-public-binding heuristic has been removed. `inputs` are
/// resolved to the referenced binding's [`Symbol`] (in declared order: each
/// must be an `elementof` parameter or fixed input — see
/// [`emit_logdensity_abi`]);
/// `outputs` are the declared query [`NodeId`]s (in declared order — already
/// reduced to deterministic expressions by determinization, per the module doc
/// comment's "Designating the query").
pub(crate) struct Abi {
    pub inputs: Vec<Symbol>,
    pub outputs: Vec<NodeId>,
}

/// Read the `inputs`/`outputs` ABI off `m`, if declared. Returns `None` when
/// NEITHER reserved binding is present — [`crate::emit`] then refuses (the
/// last-public-binding/source-order heuristic has been removed, so there is no
/// fallback). `inputs`/`outputs` survive determinization (they are the DCE
/// roots, design doc "Dead-code elimination"), so this reads them straight off
/// the determinized module by binding name; no new IR field is needed.
///
/// Each reserved binding's RHS is a single value or a `tuple(...)` call (the
/// surface `(v1, v2, …)` sugar) — [`tuple_elems`] normalizes both to a
/// `Vec<NodeId>` in source order. `inputs` elements are further resolved
/// through `(%ref self x)` to the referenced binding's [`Symbol`]; a
/// non-ref element (a malformed `inputs` entry) is dropped rather than
/// guessed at — [`emit_logdensity_abi`]'s exhaustiveness check (every
/// `elementof` binding must appear in `inputs`) then refuses rather than
/// silently mis-binding an argument.
pub(crate) fn read_abi(m: &Module) -> Option<Abi> {
    let inputs_binding = m.bindings().find(|(_, b)| m.resolve(b.name) == "inputs");
    let outputs_binding = m.bindings().find(|(_, b)| m.resolve(b.name) == "outputs");
    if inputs_binding.is_none() && outputs_binding.is_none() {
        return None;
    }
    let inputs = inputs_binding
        .map(|(_, b)| tuple_elems(m, b.rhs))
        .unwrap_or_default()
        .into_iter()
        .filter_map(|elem| match m.node(elem) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => Some(*name),
            _ => None,
        })
        .collect();
    let outputs = outputs_binding
        .map(|(_, b)| tuple_elems(m, b.rhs))
        .unwrap_or_default();
    Some(Abi { inputs, outputs })
}

/// Normalize a reserved `inputs`/`outputs` binding's RHS — a single value, or
/// a `tuple(...)` call (the surface `(v1, v2, …)` sugar, see
/// [`crate::modes`]'s design-doc reference) — to its element [`NodeId`]s in
/// declared order.
fn tuple_elems(m: &Module, rhs: NodeId) -> Vec<NodeId> {
    if is_builtin_call(m, rhs, "tuple") {
        if let Node::Call(c) = m.node(rhs) {
            return c.args.to_vec();
        }
    }
    vec![rhs]
}

/// Emit `@logdensity` for a determinized module `m` that declares the
/// `inputs`/`outputs` ABI (see [`Abi`]/[`read_abi`]) — the `LogDensity`-mode
/// ABI path. Replaces the removed source-order free-param loop and
/// last-public-binding query convention: arguments are built from `abi.inputs`
/// in declared order and results from `abi.outputs` in declared order
/// (multi-result via [`Emitter::finish`], already multi-result for
/// [`emit_sample_abi`]'s two rets).
///
/// Scope: an ABI input is either an `elementof` parameter or a
/// fixed-phase input construct — `external(S)` (a scalar/shaped runtime arg
/// typed from `S`) or `load_data(...)` (a `tensor<N×f32>` whose length `N` is
/// pinned from a compile-time file read, threaded via
/// [`EmitOptions::input_shapes`]; values are never baked). Any other binding
/// named in `inputs` (a literal, a computed value) refuses. `inputs` is
/// authoritative and exhaustive for `elementof`: every `elementof` binding in
/// `m` must appear in `abi.inputs`, else this refuses naming the missing
/// parameter. A fixed-phase binding (`external`/`load_data`) that an output
/// reaches but that is NOT listed in `inputs` also refuses, pointing at the
/// ABI — data is passed as a runtime argument, never baked (design doc
/// phase→ABI table).
pub(crate) fn emit_logdensity_abi(
    m: &Module,
    abi: &Abi,
    opts: &EmitOptions,
) -> Result<String, EmitError> {
    let mut e = Emitter::new(m, opts.dtype);

    // Exhaustiveness: every `elementof` parameter in the module must be
    // listed in `inputs` (design doc: "inputs ... is authoritative and
    // exhaustive"). Checked before building any args — a missing parameter
    // is a malformed ABI, refuse rather than emit a partial signature.
    for (_, binding) in m.bindings() {
        if is_free_param(m, binding.rhs) && !abi.inputs.contains(&binding.name) {
            return Err(EmitError::at(
                binding.rhs,
                format!(
                    "elementof parameter `{}` is not listed in `inputs`; the inputs \
                     ABI is exhaustive — every elementof parameter must appear in `inputs`",
                    m.resolve(binding.name)
                ),
            ));
        }
    }

    // A fixed-phase input construct (`external`/`load_data`) that survived
    // root-DCE (i.e. an output reaches it) but is NOT declared in `inputs`
    // refuses, pointing at the ABI: fixed data becomes a runtime argument only
    // by being listed in `inputs`; its values are never baked (design doc
    // phase→ABI table + "load_data — shape, not values"). A fixed binding no
    // output reaches was already pruned by DCE, so it never gets here — the
    // refusal fires exactly when the value would actually be needed.
    for (_, binding) in m.bindings() {
        if is_fixed_input(m, binding.rhs) && !abi.inputs.contains(&binding.name) {
            return Err(EmitError::at(
                binding.rhs,
                format!(
                    "fixed-phase binding `{}` is reached by an output but is not listed in \
                     `inputs`; list it in inputs to pass it as a runtime argument (its shape \
                     is pinned at compile time; its values are never baked)",
                    m.resolve(binding.name)
                ),
            ));
        }
    }

    if abi.outputs.is_empty() {
        return Err(EmitError::whole(
            "`outputs` ABI binding is missing or empty; at least one output is required",
        ));
    }

    let mut args: Vec<(String, MlirTy, ElemKind)> = Vec::with_capacity(abi.inputs.len());
    for &sym in &abi.inputs {
        let bid = m.binding_by_name(sym).ok_or_else(|| {
            EmitError::whole(format!(
                "`inputs` names `{}`, which is not a binding of the determinized module",
                m.resolve(sym)
            ))
        })?;
        let binding = m.binding(bid);
        // Accept `elementof` (parameterized) and the fixed-phase input
        // constructs `external`/`load_data`; anything else (a literal, a
        // computed value) cannot be an ABI argument and refuses. The message
        // keeps "not an elementof parameter" for the literal/computed case.
        if !is_free_param(m, binding.rhs) && !is_fixed_input(m, binding.rhs) {
            return Err(EmitError::at(
                binding.rhs,
                format!(
                    "`inputs` entry `{}` is not an elementof parameter, external, or \
                     load_data input — only these constructs can be ABI arguments",
                    m.resolve(sym)
                ),
            ));
        }
        let name = format!("%arg{}", args.len());
        let (mut ty, elem) = mlir_type_of(m, binding.rhs, opts.dtype)?;
        // Shape-pin a fixed-phase input whose FlatPDL type carries a dynamic
        // dim (`load_data` → `tensor<?×f32>`) from the compile-time length map
        // (design doc "load_data — shape, not values"): `tensor<N×f32>`. A `?`
        // dim would be unusable downstream. `elementof`/statically-shaped
        // inputs need no pin and keep their inferred type.
        if let Some(shape) = opts.input_shapes.get(m.resolve(sym)) {
            ty = MlirTy::Ranked(shape.iter().map(|&n| Some(n)).collect());
        }
        // Use the inferred element kind (not a hardcoded `Real`): an integer /
        // boolean `elementof` (or int `load_data`) input must arrive as an
        // int/bool tensor arg so the value-path widening reconciles correctly.
        // For a real input this is `ElemKind::Real` — byte-identical to before.
        e.bind(
            binding.rhs,
            Value {
                ssa: name.clone(),
                ty: ty.clone(),
                elem,
            },
        );
        args.push((name, ty, elem));
    }

    let mut rets_vals: Vec<Value> = Vec::with_capacity(abi.outputs.len());
    for &node in &abi.outputs {
        rets_vals.push(e.lower_node(node)?);
    }
    let rets: Vec<&Value> = rets_vals.iter().collect();
    Ok(e.finish("logdensity", &args, &rets))
}

/// Emit `@sample` for a determinized module `m` that declares the
/// `inputs`/`outputs` ABI — the Sample-mode analogue of
/// [`emit_logdensity_abi`]. The single declared output is the sample query
/// (a value-terminal `rand(rstate, M)` / `builtin_sample`-bearing binding).
/// `%key` (the threaded rng state, spec §07's `rand(rstate, m) -> (value,
/// new_rstate)`) is arg 0, found from the output via [`find_rng_source`] —
/// NOT drawn from `inputs`; the rng-source binding is exempt from the
/// inputs-exhaustiveness check. `abi.inputs` supplies the additional free
/// params (`elementof`) / fixed inputs as `%arg0..` (numbered independently of
/// `%key`), in declared order. Returns the two-result `(value, new_key)`
/// function.
pub(crate) fn emit_sample_abi(
    m: &Module,
    abi: &Abi,
    opts: &EmitOptions,
) -> Result<String, EmitError> {
    let mut e = Emitter::new(m, opts.dtype);

    // Exactly one sample output.
    let output = match abi.outputs.as_slice() {
        [one] => *one,
        _ => {
            return Err(EmitError::whole(
                "`outputs` for a sample query must name exactly one output (the \
                 sampled value)",
            ));
        }
    };

    // %key = arg 0, bound from the rng source reachable from the output.
    let src = find_rng_source(m, output).ok_or_else(|| {
        EmitError::at(
            output,
            "no rng source to bind to %key: the declared sample output reaches \
             no rnginit/external(rngstates) source to thread from",
        )
    })?;
    let key_name = "%key".to_string();
    e.bind(
        src,
        Value {
            ssa: key_name.clone(),
            ty: MlirTy::Key,
            elem: ElemKind::Real,
        },
    );
    let mut args: Vec<(String, MlirTy, ElemKind)> = vec![(key_name, MlirTy::Key, ElemKind::Real)];

    // Exhaustiveness over the ABI inputs, mirroring emit_logdensity_abi — with
    // ONE exemption: the rng-source binding is bound to `%key` (arg 0), not a
    // listed input. `src` is that source's rng-argument node; when it is a
    // `(%ref self s)` the exempt binding is `s` (a bare `external(rngstates)`
    // rng source would otherwise trip the fixed-input check below).
    let rng_src_sym = match m.node(src) {
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) => Some(*name),
        _ => None,
    };
    for (_, binding) in m.bindings() {
        if Some(binding.name) == rng_src_sym {
            continue;
        }
        // Every `elementof` parameter must be listed in `inputs`.
        if is_free_param(m, binding.rhs) && !abi.inputs.contains(&binding.name) {
            return Err(EmitError::at(
                binding.rhs,
                format!(
                    "elementof parameter `{}` is not listed in `inputs`; the inputs \
                     ABI is exhaustive",
                    m.resolve(binding.name)
                ),
            ));
        }
        // A fixed-phase input (`external`/`load_data`) reached by the sample
        // output but not listed in `inputs` refuses, pointing at the ABI: data
        // is passed as a runtime argument, never baked.
        if is_fixed_input(m, binding.rhs) && !abi.inputs.contains(&binding.name) {
            return Err(EmitError::at(
                binding.rhs,
                format!(
                    "fixed-phase binding `{}` is reached by the sample output but is not \
                     listed in `inputs`; list it in inputs to pass it as a runtime argument \
                     (its shape is pinned at compile time; its values are never baked)",
                    m.resolve(binding.name)
                ),
            ));
        }
    }

    // Bind declared inputs (in declared order), numbered `%arg0..`
    // INDEPENDENTLY of the leading `%key` — reproducing the pre-purge
    // `emit_sample` signature exactly (`%key`, `%arg0`, `%arg1`, …) so the
    // frozen sample goldens stay byte-identical: the purge removes the query
    // heuristic, it does not renumber the emitted arguments.
    for (nfree, &sym) in abi.inputs.iter().enumerate() {
        let bid = m.binding_by_name(sym).ok_or_else(|| {
            EmitError::whole(format!(
                "`inputs` names `{}`, which is not a binding of the determinized module",
                m.resolve(sym)
            ))
        })?;
        let binding = m.binding(bid);
        if !is_free_param(m, binding.rhs) && !is_fixed_input(m, binding.rhs) {
            return Err(EmitError::at(
                binding.rhs,
                format!(
                    "`inputs` entry `{}` is not an elementof parameter, external, or \
                     load_data input",
                    m.resolve(sym)
                ),
            ));
        }
        let name = format!("%arg{nfree}");
        let (mut ty, elem) = mlir_type_of(m, binding.rhs, opts.dtype)?;
        if let Some(shape) = opts.input_shapes.get(m.resolve(sym)) {
            ty = MlirTy::Ranked(shape.iter().map(|&n| Some(n)).collect());
        }
        e.bind(
            binding.rhs,
            Value {
                ssa: name.clone(),
                ty: ty.clone(),
                elem,
            },
        );
        args.push((name, ty, elem));
    }

    // The declared output is a `(%ref self draws)` node (ABI outputs are refs,
    // unlike the legacy path's binding RHS); resolve it to the underlying
    // `rand`/tuple before extracting the value component, so `@sample` returns
    // the drawn value — not the `tuple(value, advanced_rng)` — as its first
    // result (spec §07 `rand -> (value, new_rstate)`; keeps parity with
    // `emit_sample`'s `query.1.rhs`).
    let value = e.lower_node(query_value_component(m, resolve_self_ref(m, output)))?;
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
/// [`emit_sample_abi`] binds to `%key` so [`crate::registry::lower_sample`]'s
/// `e.lower_node(rng)` resolves straight to the func argument (the `rnginit`
/// node itself is never lowered — its seed→state math is out of scope).
///
/// `None` when no such source exists (every sample's rng arg is another
/// sample's advanced key — a cycle, or a model whose only rng comes from a
/// slot with no root): [`emit_sample_abi`] then refuses rather than silently
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
/// [`Module::for_each_child`] — a record/hierarchical query's samples sit one
/// or more binding-hops away on ref-resolved RHSs, so a purely structural walk
/// that did not follow refs would miss them. Deduplicated (a sample projected
/// as both `get0(s, 0)` and `get0(s, 1)` is one node) via the visited set.
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

/// A free-parameter declaration: `Phase::Parameterized` (spec §04 "Phase of
/// an expression") AND structurally a bare `elementof(...)` call. The phase
/// check alone is not enough — see the module doc comment on why phase is a
/// taint over the whole dependent subtree, not a parameter-leaf marker.
fn is_free_param(m: &Module, rhs: NodeId) -> bool {
    m.phase_of(rhs) == Some(Phase::Parameterized) && is_builtin_call(m, rhs, "elementof")
}

/// A fixed-phase input construct: structurally a bare `external(...)` or
/// `load_data(...)` call (spec §04 "fixed" phase — set at initialization,
/// immutable after). Listed in `inputs`, such a binding becomes a runtime
/// argument (the values are NOT baked; `load_data`'s shape is pinned from a
/// compile-time file read — design doc "load_data — shape, not values"); NOT
/// listed, [`emit_logdensity_abi`] refuses. A purely structural check (like
/// [`is_free_param`]'s `elementof` test) — the phase taint is not needed to
/// distinguish these declarations.
fn is_fixed_input(m: &Module, rhs: NodeId) -> bool {
    is_builtin_call(m, rhs, "external") || is_builtin_call(m, rhs, "load_data")
}

/// Whether `id` is (structurally) a `Call` whose head is the builtin named
/// `name` — shared by [`is_free_param`]'s `elementof(...)` check,
/// [`is_fixed_input`]'s `external`/`load_data` check, and [`tuple_elems`]'s
/// `tuple(...)` check.
fn is_builtin_call(m: &Module, id: NodeId, name: &str) -> bool {
    matches!(
        m.node(id),
        Node::Call(c) if matches!(
            c.head,
            CallHead::Builtin(sym) if m.resolve(sym) == name
        )
    )
}
