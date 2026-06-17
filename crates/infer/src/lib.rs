//! Type, shape, and phase inference for [`flatppl_core`] modules.
//!
//! The **type-domain trace** (engine-concepts §17): a memoised walk of the IR
//! in dependency order, dispatching each call to a per-op rule and filling the
//! module's type/phase side-tables. Inference is **structure-preserving** —
//! it reads and annotates, never rewrites ("resolve, don't rewrite",
//! engine-concepts §17.1); the annotated module projects to spec-§11
//! `(%meta <type> <phase>)` on FlatPIR write.
//!
//! **Honesty over coverage.** The op catalogue ([`ops`]) grows with the
//! toolchain. An op without a rule yields `%deferred` plus a once-per-op
//! note diagnostic — never a guess; genuine type errors (reference cycles,
//! unresolved names) yield `(%failed …)` plus an error diagnostic.
//! Cross-module references (`load_module` / `standard_module`) are resolved
//! against the [`ModuleBundle`] supplied to [`infer_module`]; a missing
//! dependency or an unknown binding yields `(%failed …)` plus an error
//! diagnostic, never a silent `%deferred`.
//!
//! Phases follow the spec-§04 ancestor rule: `stochastic > parameterized >
//! fixed`, with `elementof` introducing *parameterized*, `draw` *stochastic*,
//! and `external` / loaded data / reifications *fixed*.

mod modules;
mod ops;
mod trace;

use crate::modules::InferSession;

use flatppl_core::{Module, NodeId};

pub use modules::ModuleBundle;

/// A message produced during inference. `Error` marks an ill-formed module
/// (the offending nodes carry `(%failed …)` types); `Note` records an
/// honest gap (op not yet in the catalogue, deferred feature).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    /// The IR node the diagnostic is about, when known. `None` for module-level
    /// diagnostics with no single offending node.
    pub node: Option<NodeId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Note,
}

impl Diagnostic {
    pub(crate) fn error(message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
            node: None,
        }
    }
    pub(crate) fn note(message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Note,
            message: message.into(),
            node: None,
        }
    }
    /// An error anchored to the offending node.
    pub(crate) fn error_at(node: NodeId, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
            node: Some(node),
        }
    }
}

/// How far to take the trace. The levels form a hierarchy — each includes
/// everything below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Phases only (the spec-§04 ancestor classification). Types are not
    /// annotated — `%meta` carries `%deferred` in the type slot.
    Phase,
    /// Phases + structural types. Array dimensions are static only where
    /// they are syntactically literal; measure/kernel masses stay
    /// `%deferred`.
    Type,
    /// Phases + types + value sets: the strongest statically known set
    /// containing each node's value (a measure node's support), from the
    /// spec-§03 set vocabulary — the third `%meta` slot. Producers: the §08
    /// Domain/Support catalogue, `elementof`/`truncate` set arguments,
    /// normalization functions (`softmax`, guarded `l1unit`).
    Valueset,
    /// Phases + types + value sets + total-mass classes on measure/kernel
    /// types (spec §11 "Total-mass classes"), composed per op — and static
    /// rejection of `normalize` on measures with known zero/infinite mass.
    Normalization,
    /// Everything + shape resolution: fixed-phase integer expressions at
    /// shape positions (`iid` counts, `cartpow` sizes, distribution dims)
    /// are resolved demand-driven (engine-concepts §17.1 — "resolve, don't
    /// rewrite"; the source IR is never modified).
    Shape,
}

/// Infer at [`Level::Shape`] (everything) — see [`infer_with`].
pub fn infer(module: &mut Module) -> Vec<Diagnostic> {
    infer_with(module, Level::Shape)
}

/// Infer types and phases for every binding of `module` at the given
/// [`Level`], resolving cross-module (`load_module`) references against
/// `bundle`. The host supplies parsed dependencies in `bundle`; the engine
/// does no file I/O.
pub fn infer_module(module: &mut Module, bundle: &ModuleBundle, level: Level) -> Vec<Diagnostic> {
    let session = InferSession::new(bundle);
    trace::Inferencer::new(module, level, &session).run()
}

/// Infer types and phases for every binding of `module` at the given
/// [`Level`], filling its type/phase side-tables in place. Best-effort:
/// always annotates as much as it can; returned diagnostics report errors
/// (ill-formed module) and notes (honest `%deferred` gaps).
pub fn infer_with(module: &mut Module, level: Level) -> Vec<Diagnostic> {
    let bundle = ModuleBundle::new();
    infer_module(module, &bundle, level)
}

#[cfg(test)]
mod infer_module_entry_tests {
    use super::*;

    #[test]
    fn infer_module_empty_bundle_matches_infer_with() {
        let mut a = flatppl_syntax::parse("x = add(1, 2)").expect("parses");
        let mut b = a.clone();
        let bundle = ModuleBundle::new();
        let da = infer_with(&mut a, Level::Type);
        let db = infer_module(&mut b, &bundle, Level::Type);
        assert_eq!(da.len(), db.len());
        let (_id, ba) = a.bindings().next().unwrap();
        let (_id2, bb) = b.bindings().next().unwrap();
        assert_eq!(a.type_of(ba.rhs), b.type_of(bb.rhs));
    }
}

#[cfg(test)]
mod cross_module_value_tests {
    use super::*;

    fn bundle_with(path: &str, src: &str) -> ModuleBundle {
        let dep = flatppl_syntax::parse(src).expect("dep parses");
        let mut b = ModuleBundle::new();
        b.insert(path, dep);
        b
    }

    #[test]
    fn cross_module_value_ref_resolves_to_dep_type() {
        let bundle = bundle_with(
            "helpers.flatppl",
            "center = elementof(reals)\nshifted_value = add(center, 1.0)",
        );
        let mut model = flatppl_syntax::parse(
            "helpers = load_module(\"helpers.flatppl\")\nv = add(helpers.shifted_value, 2.0)",
        )
        .expect("model parses");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            !diags.iter().any(|d| d.message.contains("deferred")),
            "no deferred note for a resolved cross-module ref; got {diags:?}"
        );
        let (_id, vb) = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "v")
            .expect("binding `v` not found");
        assert_eq!(
            model.type_of(vb.rhs),
            Some(&flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real))
        );
    }

    #[test]
    fn missing_dependency_is_an_error() {
        let mut model =
            flatppl_syntax::parse("helpers = load_module(\"absent.flatppl\")\nv = helpers.x")
                .expect("parses");
        let bundle = ModuleBundle::new();
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(diags.iter().any(|d| d.message.contains("not found")));
    }

    #[test]
    fn private_binding_reference_is_an_error() {
        let bundle = bundle_with("h.flatppl", "_secret = 1.0\npublic_v = 2.0");
        let mut model =
            flatppl_syntax::parse("h = load_module(\"h.flatppl\")\nv = h._secret").expect("parses");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(diags.iter().any(|d| d.message.contains("private")));
    }

    /// The `not-a-module` path in `load_directive` is defensive: the parser's
    /// name pre-pass only lowers `m.x` to `RefNs::Module` when `m` is a
    /// `load_module` / `standard_module` binding, so well-formed surface syntax
    /// cannot produce a `RefNs::Module` ref where the alias is a plain value.
    /// (Writing `helpers = 1.0; v = helpers.x` makes `helpers.x` desugar to
    /// `get(helpers, "x")`, NOT a cross-module ref — the dot becomes field
    /// access.) This test confirms the hand-built IR path that could reach it.
    #[test]
    fn not_a_module_path_is_defensive_unreachable_from_surface_syntax() {
        // Surface syntax: `helpers` is a plain value, not a load_module call.
        // The parser will lower `helpers.x` as field access `get(helpers, "x")`,
        // never as a cross-module ref — so no "not a module" error is produced.
        let mut model = flatppl_syntax::parse("helpers = 1.0\nv = helpers.x").expect("parses");
        let diags = infer_module(&mut model, &ModuleBundle::new(), Level::Type);
        // No "not a module" error — instead `get(1.0, "x")` is a type gap.
        assert!(
            !diags.iter().any(|d| d.message.contains("is not a module")),
            "surface syntax cannot trigger not-a-module; got {diags:?}"
        );
    }

    #[test]
    fn unknown_binding_in_dep_is_an_error() {
        let bundle = bundle_with("h.flatppl", "x = 1.0");
        let mut model = flatppl_syntax::parse("h = load_module(\"h.flatppl\")\nv = h.nonexistent")
            .expect("parses");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            diags.iter().any(|d| d.message.contains("has no binding")),
            "expected unknown-binding error; got {diags:?}"
        );
    }
}

#[cfg(test)]
mod substitution_tests {
    use super::*;

    #[test]
    fn substitution_flows_into_dependency() {
        // Unsubstituted, the dep gives `out: Integer` (add(Integer, Integer)).
        // Seeding `p` with a Real-typed argument promotes `out` to Real, so this
        // assertion fails if seeding is disabled.
        let dep = flatppl_syntax::parse("p = elementof(integers)\nout = add(p, 1)").expect("dep");
        let mut bundle = ModuleBundle::new();
        bundle.insert("d.flatppl", dep);
        let mut model = flatppl_syntax::parse(
            "a = add(2.0, 3.0)\nd = load_module(\"d.flatppl\", p = a)\nv = d.out",
        )
        .expect("model");
        let _ = infer_module(&mut model, &bundle, Level::Type);
        let (_id, vb) = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "v")
            .expect("binding `v` not found");
        assert_eq!(
            model.type_of(vb.rhs),
            Some(&flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real)),
            "out promotes to real because substituted p is real (Integer otherwise)"
        );
    }

    #[test]
    fn same_dep_two_substitutions_memoize_separately() {
        let dep = flatppl_syntax::parse("p = elementof(reals)\nout = p").expect("dep");
        let mut bundle = ModuleBundle::new();
        bundle.insert("d.flatppl", dep);
        let mut model = flatppl_syntax::parse(
            "a = 1\nb = 2.0\nd1 = load_module(\"d.flatppl\", p = a)\nd2 = load_module(\"d.flatppl\", p = b)\nv1 = d1.out\nv2 = d2.out",
        ).expect("model");
        let _ = infer_module(&mut model, &bundle, Level::Type);
        let v1 = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "v1")
            .expect("binding `v1` not found")
            .1
            .rhs;
        let v2 = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "v2")
            .expect("binding `v2` not found")
            .1
            .rhs;
        assert_eq!(
            model.type_of(v1),
            Some(&flatppl_core::Type::Scalar(
                flatppl_core::ScalarType::Integer
            ))
        );
        assert_eq!(
            model.type_of(v2),
            Some(&flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real))
        );
    }
}

#[cfg(test)]
mod unknown_input_tests {
    use super::*;
    #[test]
    fn substitution_for_unknown_input_is_an_error() {
        let dep = flatppl_syntax::parse("p = elementof(reals)\nout = p").expect("dep");
        let mut bundle = ModuleBundle::new();
        bundle.insert("d.flatppl", dep);
        let mut model =
            flatppl_syntax::parse("a = 1.0\nd = load_module(\"d.flatppl\", q = a)\nv = d.out")
                .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            diags.iter().any(|d| d.message.contains("has no input `q`")),
            "expected unknown-input error; got {diags:?}"
        );
    }
}

#[cfg(test)]
mod cycle_tests {
    use super::*;

    #[test]
    fn import_cycle_is_reported_not_infinite() {
        // a.flatppl loads b.flatppl which loads a.flatppl.
        let a = flatppl_syntax::parse("bb = load_module(\"b.flatppl\")\nx = bb.y").expect("a");
        let b = flatppl_syntax::parse("aa = load_module(\"a.flatppl\")\ny = aa.x").expect("b");
        let mut bundle = ModuleBundle::new();
        bundle.insert("a.flatppl", a.clone());
        bundle.insert("b.flatppl", b);
        // Infer `a` as root; resolving bb.y → b, whose aa.x → a closes the loop.
        let mut root = a;
        let diags = infer_module(&mut root, &bundle, Level::Type);
        assert!(
            diags.iter().any(|d| d.message.contains("cycle")),
            "expected a module cycle diagnostic; got {diags:?}"
        );
    }
}

#[cfg(test)]
mod cross_module_callable_tests {
    use super::*;

    #[test]
    fn cross_module_kernel_likelihood_infers() {
        let helpers = flatppl_syntax::parse(
            "center = elementof(reals)\nspread = elementof(posreals)\n\
             obs_kernel = functionof(Normal(mu = add(center, _x_), sigma = spread), center = center, spread = spread, x = _x_)",
        ).expect("helpers");
        let mut bundle = ModuleBundle::new();
        bundle.insert("helpers.flatppl", helpers);
        let mut model = flatppl_syntax::parse(
            "a = elementof(reals)\nhelpers = load_module(\"helpers.flatppl\", center = a)\n\
             input_data = 2.5\nL = likelihoodof(helpers.obs_kernel, input_data)",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Normalization);
        let lb = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "L")
            .expect("binding `L` not found")
            .1
            .rhs;
        let ty = model.type_of(lb);
        // Pin the concrete result: a `%likelihood` whose observation domain is
        // the kernel body's measure domain — `Normal(...)` is real-valued, so
        // `obstype` is `(%scalar real)`. A bare `Kernel`, a wrong obstype, or
        // `Any` would all fail here (the negative `!Deferred` check would not).
        assert!(
            matches!(
                ty,
                Some(flatppl_core::Type::Likelihood { obstype, .. })
                    if **obstype == flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real)
            ),
            "L should be a Likelihood over (%scalar real); got {ty:?}; diags {diags:?}"
        );
    }

    /// Gold-standard parity guard: applying a cross-module reified callable must
    /// produce an IDENTICAL result type to applying the same callable defined
    /// locally. This pins that approach (a) — riding the dependency's
    /// body-result type over `Resolved` — never diverges from the single-module
    /// `reified_result_type` path. (Substituting `center = a` cross-module
    /// changes the value's *location*, not its TYPE, so the types must match.)
    #[test]
    fn cross_module_callable_matches_local_application() {
        // Local: the kernel and its likelihood inlined in one module.
        let mut local = flatppl_syntax::parse(
            "center = elementof(reals)\nspread = elementof(posreals)\n\
             obs_kernel = functionof(Normal(mu = add(center, _x_), sigma = spread), center = center, spread = spread, x = _x_)\n\
             input_data = 2.5\nL = likelihoodof(obs_kernel, input_data)",
        )
        .expect("local");
        let _ = infer_module(&mut local, &ModuleBundle::new(), Level::Normalization);
        let l_local = local
            .bindings()
            .find(|(_, b)| local.resolve(b.name) == "L")
            .expect("binding `L` not found in local")
            .1
            .rhs;
        let local_ty = local.type_of(l_local).cloned();

        // Cross-module: the same kernel in a loaded dependency.
        let helpers = flatppl_syntax::parse(
            "center = elementof(reals)\nspread = elementof(posreals)\n\
             obs_kernel = functionof(Normal(mu = add(center, _x_), sigma = spread), center = center, spread = spread, x = _x_)",
        )
        .expect("helpers");
        let mut bundle = ModuleBundle::new();
        bundle.insert("helpers.flatppl", helpers);
        let mut model = flatppl_syntax::parse(
            "a = elementof(reals)\nhelpers = load_module(\"helpers.flatppl\", center = a)\n\
             input_data = 2.5\nL = likelihoodof(helpers.obs_kernel, input_data)",
        )
        .expect("model");
        let _ = infer_module(&mut model, &bundle, Level::Normalization);
        let l_cross = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "L")
            .expect("binding `L` not found in cross-module model")
            .1
            .rhs;
        let cross_ty = model.type_of(l_cross).cloned();

        assert!(local_ty.is_some(), "local infer failed");
        assert!(cross_ty.is_some(), "cross infer failed");
        assert_eq!(
            local_ty, cross_ty,
            "cross-module callable application must match local application"
        );
    }
}

#[cfg(test)]
mod diag_anchor_tests {
    use super::*;
    use flatppl_core::{Binding, Module, Node, Ref, RefNs, Span};

    /// Build a module programmatically: one binding `y` whose RHS is a
    /// `SelfMod` reference to the undeclared name `nope`.  The parser can
    /// never produce this configuration (it only emits `SelfMod` refs for
    /// names on some LHS, which always get a corresponding binding), so we
    /// construct the IR directly via `flatppl_core`.
    fn module_with_dangling_ref() -> (Module, Span) {
        let mut m = Module::default();
        let nope_sym = m.intern("nope");
        let y_sym = m.intern("y");
        // The ref node — this is the "offending node" the diagnostic should anchor to.
        let ref_span = Span { start: 8, end: 12 }; // offset into "y = add(nope, 1)"
        let ref_id = m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: nope_sym,
        }));
        m.set_span(ref_id, ref_span);
        m.add_binding(Binding {
            name: y_sym,
            rhs: ref_id,
            doc: None,
            public: true,
            synthetic: false,
        });
        (m, ref_span)
    }

    #[test]
    fn unresolved_reference_diagnostic_is_anchored() {
        let (mut m, ref_span) = module_with_dangling_ref();
        let diags = infer_with(&mut m, Level::Type);
        let unresolved = diags
            .iter()
            .find(|d| d.message.contains("unresolved reference"))
            .expect("an unresolved-reference diagnostic");
        let node = unresolved.node.expect("diagnostic carries a node anchor");
        let span = m.span_of(node).expect("anchored node has a span");
        assert_eq!(span, ref_span, "anchored span matches the ref node's span");
    }
}
