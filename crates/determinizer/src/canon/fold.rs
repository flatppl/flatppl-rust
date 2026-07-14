//! Pass 1: constant folding, trivial-alias ref resolution, and dead-binding
//! elimination — the canonicalization foundation (Buffy #263).

use std::collections::HashMap;

use flatppl_core::{CallHead, Module, Node, NodeId, Scalar};

use crate::driver::map_tree;

/// Fold builtin arithmetic on literal operands. Two-phase: `map_tree`'s closure
/// only gets `&Module` (no `alloc`), so phase 1 walks each binding's RHS
/// bottom-up allocating the literal replacement for every foldable node
/// (`collect_folds`, needs `&mut Module`); phase 2 applies those replacements
/// via a `map_tree` identity-by-id closure. One sweep folds a whole already-
/// literal subtree bottom-up (recursion in `collect_folds` sees an inner fold's
/// result within the same sweep); the driver's fixpoint re-runs this after
/// `resolve_alias_refs`/`sweep_dead_bindings` may have exposed further literal
/// operands.
pub(crate) fn const_fold(m: &mut Module) -> bool {
    let mut replacements: HashMap<NodeId, NodeId> = HashMap::new();
    let roots: Vec<NodeId> = m.bindings().map(|(_, b)| b.rhs).collect();
    for root in &roots {
        collect_folds(m, *root, &mut replacements);
    }
    if replacements.is_empty() {
        return false;
    }
    let mut changed = false;
    let pairs: Vec<(flatppl_core::BindingId, NodeId)> =
        m.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    for (bid, root) in pairs {
        let new = map_tree(m, root, &mut |_m, id| replacements.get(&id).copied());
        if new != root {
            m.set_binding_rhs(bid, new);
            changed = true;
        }
    }
    changed
}

/// Walk `root` bottom-up; for each foldable node, alloc its literal result and
/// record `node -> replacement`. Recurses so an inner fold's literal result is
/// visible to its parent within the SAME sweep (faster convergence than relying
/// only on the driver fixpoint, and keeps the map_tree apply purely by-id).
fn collect_folds(m: &mut Module, id: NodeId, out: &mut HashMap<NodeId, NodeId>) -> Option<Scalar> {
    let children: Vec<NodeId> = m.node(id).children();
    let child_scalars: Vec<Option<Scalar>> =
        children.iter().map(|&c| collect_folds(m, c, out)).collect();

    let Node::Call(c) = m.node(id) else {
        // A literal node reports its own value so a parent can fold; a
        // just-folded child (recorded in `out`) also needs to report its value,
        // but `out` only maps node -> replacement id, not id -> Scalar, so we
        // read the replacement's own Lit value back out (it was just alloc'd
        // as a `Node::Lit`, so this always succeeds).
        if let Node::Lit(s) = m.node(id) {
            return Some(s.clone());
        }
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    let op = m.resolve(sym).to_string();
    let nargs = c.args.len();
    // Binary real arithmetic on two literal reals. Restricted to the four basic
    // ops: IEEE-754 mandates these be correctly rounded, so `f64` arithmetic is
    // bit-identical across any conforming implementation (Rust's compiled
    // arithmetic vs. the JS engine's) — required for the det-js numeric
    // equivalence gate (Buffy #263 verification). `pow` is deliberately
    // EXCLUDED: general real exponentiation is not rounding-mandated by
    // IEEE-754 (it is transcendental for non-integer exponents) and libm
    // implementations are known to disagree in the last bit; extend to `pow`
    // only if a corpus model needs it AND the numeric gate stays green.
    if nargs == 2 && c.named.is_empty() {
        if let (Some(Scalar::Real(a)), Some(Scalar::Real(b))) =
            (&child_scalars[0], &child_scalars[1])
        {
            let r = match op.as_str() {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "divide" => a / b,
                _ => return None,
            };
            let lit = m.alloc(Node::Lit(Scalar::Real(r)));
            out.insert(id, lit);
            return Some(Scalar::Real(r));
        }
    }
    // Unary ops on one literal real. `neg` is exact (sign flip only, no
    // rounding); `sqrt` is IEEE-754-mandated correctly rounded, so it is also
    // bit-identical across conforming implementations. `log`/`exp` are
    // deliberately EXCLUDED for the same reason `pow` is above: they are not
    // rounding-mandated and can diverge in the last bit between Rust's libm and
    // the JS engine's Math.log/Math.exp — folding them here would risk breaking
    // the bit-identical det-js equivalence gate. Leaving `log`/`exp` unevaluated
    // also matches the "flatppl-rust is not a density engine" boundary: the
    // determiniser normalizes the IR, it does not pre-compute transcendental
    // function values that belong to the evaluating engine. `abs` is ALSO
    // deliberately excluded despite being exact: `log(abs(k))` is the
    // established canonical symbolic form for a pushfwd/locscale log-volume
    // term (see `pushfwd_golden.rs`/`density_golden.rs`, five existing goldens
    // pin `(abs <k>)` unevaluated as "logvol = log|k|") — folding it away would
    // erase that documented, human-legible structure for no numeric benefit.
    if nargs == 1 && c.named.is_empty() {
        if let Some(Scalar::Real(a)) = &child_scalars[0] {
            let r = match op.as_str() {
                "neg" => -a,
                "sqrt" => a.sqrt(),
                _ => return None,
            };
            let lit = m.alloc(Node::Lit(Scalar::Real(r)));
            out.insert(id, lit);
            return Some(Scalar::Real(r));
        }
    }
    None
}

/// Replace `(%ref self x)` where `x` binds a trivial value — another ref leaf,
/// a `Const`, or a `Lit` — with that value. Non-trivial bindings are left
/// (inlining them would duplicate shared subterms; leave that to CSE / the
/// rewriter).
pub(crate) fn resolve_alias_refs(m: &mut Module) -> bool {
    // Map: binding-name Symbol -> trivial replacement NodeId.
    let mut trivial: HashMap<flatppl_core::Symbol, NodeId> = HashMap::new();
    for (_, b) in m.bindings() {
        match m.node(b.rhs) {
            Node::Lit(_) | Node::Const(_) | Node::Ref(_) => {
                trivial.insert(b.name, b.rhs);
            }
            _ => {}
        }
    }
    if trivial.is_empty() {
        return false;
    }
    let mut changed = false;
    let pairs: Vec<(flatppl_core::BindingId, NodeId, flatppl_core::Symbol)> =
        m.bindings().map(|(bid, b)| (bid, b.rhs, b.name)).collect();
    for (bid, root, self_name) in pairs {
        let new = map_tree(m, root, &mut |m, id| {
            if let Node::Ref(flatppl_core::Ref {
                ns: flatppl_core::RefNs::SelfMod,
                name,
            }) = m.node(id)
            {
                // Do not rewrite a binding's own self-reference into itself
                // (a trivial binding whose RHS is literally `%ref self <name>`
                // referring to itself would otherwise loop forever across
                // sweeps; that shape shouldn't occur, but the guard is free).
                if *name != self_name {
                    return trivial.get(name).copied();
                }
            }
            None
        });
        if new != root {
            m.set_binding_rhs(bid, new);
            changed = true;
        }
    }
    changed
}
/// Zero any unreferenced, engine-generated (`synthetic`) binding to a
/// fixpoint — a generalization of `driver::sweep_dead_measure_bindings`
/// (which only sweeps measure/likelihood-typed dead bindings during the
/// greedy legalizer loop) to ANY value-typed dead SYNTHETIC binding orphaned
/// by `const_fold`/`resolve_alias_refs`.
///
/// **Deliberately requires `synthetic`, not just `!public`.** `Binding::public`
/// is purely a name-shape convention ("does not start with `_`") — it says
/// nothing about whether a binding is a meaningful, externally-queryable
/// value. A scoring harness's `__score__` binding (the flatppl-testsuite/CLI
/// convention: append `__score__ = logdensityof(...)` and query it by name)
/// is `!public` under that convention yet is exactly the value an external
/// caller wants read back. An earlier cut of this sweep used `!public` alone
/// as the eligibility guard and silently zeroed `__score__` to `0.0` — caught
/// by the Buffy #263 numeric det-js equivalence gate (canon vs. no-canon
/// scores diverged), not by any existing golden, because no prior test named
/// a binding with a leading underscore. `synthetic` (set only by the parser,
/// for a lifted anon / `%mlhs` split, and by determinizer-internal scaffolding
/// — never by a user-chosen or harness-chosen name) is the correct predicate:
/// no external caller can be depending on a name it never chose. Never
/// touches a `%public` binding either, for the same reason `sweep_dead_measure_bindings`
/// doesn't: a public name is part of the model's declared interface even if
/// nothing internal refers to it. Mirrors the existing sweep's convention of
/// zeroing the RHS to `Lit(Real(0.0))` rather than removing the binding:
/// `Module` has no binding-removal API, and downstream code / golden tests
/// already expect a swept binding to survive as `(%bind name 0.0)`.
pub(crate) fn sweep_dead_bindings(m: &mut Module) -> bool {
    let mut changed = false;
    loop {
        // `is_zeroed_sentinel` excludes a binding already swept: without it, a
        // zeroed synthetic/unreferenced binding is STILL synthetic and STILL
        // unreferenced (zeroing its RHS doesn't change either), so it would
        // match the filter again next iteration and get re-zeroed forever —
        // an infinite loop (unlike `driver::sweep_dead_measure_bindings`, whose
        // eligibility is keyed on the RHS being a combinator op or
        // Measure/Likelihood/Kernel-*typed*; a zeroed `Lit` fails that check on
        // its own, so that sweep's loop naturally drops it and terminates).
        let dead: Vec<flatppl_core::BindingId> = m
            .bindings()
            .filter(|(bid, b)| {
                !b.public
                    && b.synthetic
                    && !is_zeroed_sentinel(m, b.rhs)
                    && !binding_is_referenced(m, *bid, b.name)
            })
            .map(|(bid, _)| bid)
            .collect();
        if dead.is_empty() {
            break;
        }
        for bid in dead {
            let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
            m.set_binding_rhs(bid, zero);
        }
        changed = true;
    }
    changed
}

/// True iff `rhs` is already the sweep's own zero-literal sentinel
/// (`Lit(Real(0.0))`) — i.e. this binding was already swept (or happens to be
/// a source-level `= 0.0`, which is output-identical either way).
fn is_zeroed_sentinel(m: &Module, rhs: NodeId) -> bool {
    matches!(m.node(rhs), Node::Lit(Scalar::Real(z)) if *z == 0.0)
}

/// True iff any binding OTHER than `bid` contains a `(%ref self name_sym)` —
/// as a body sub-node OR as a `functionof`/`kernelof` reification INPUT (the
/// `Inputs` bucket is invisible to `children()`/`for_each_child`, so a binding
/// referenced only through a reification boundary would otherwise look dead).
/// Mirrors `driver::binding_is_referenced`/`driver::subtree_contains_ref` for
/// this general (non-measure-typed) sweep.
fn binding_is_referenced(
    m: &Module,
    bid: flatppl_core::BindingId,
    name_sym: flatppl_core::Symbol,
) -> bool {
    m.bindings()
        .filter(|(other_bid, _)| *other_bid != bid)
        .any(|(_, binding)| subtree_contains_ref(m, binding.rhs, name_sym))
}

/// BFS subtree search: true iff the subtree at `root` contains a
/// `Ref(SelfMod, name_sym)` node, as a body sub-node or a reification `Inputs`
/// boundary entry.
fn subtree_contains_ref(m: &Module, root: NodeId, name_sym: flatppl_core::Symbol) -> bool {
    let mut queue = vec![root];
    let mut qi = 0;
    while qi < queue.len() {
        let id = queue[qi];
        qi += 1;
        match m.node(id) {
            Node::Ref(flatppl_core::Ref {
                ns: flatppl_core::RefNs::SelfMod,
                name,
            }) if *name == name_sym => return true,
            Node::Call(c) => {
                if let Some(flatppl_core::Inputs::Spec(entries)) = &c.inputs {
                    for (_, r) in entries.iter() {
                        if r.ns == flatppl_core::RefNs::SelfMod && r.name == name_sym {
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
        m.for_each_child(id, |c| queue.push(c));
    }
    false
}
