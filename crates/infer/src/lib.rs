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

mod catalogue;
mod modules;
mod ops;
mod trace;

use crate::modules::InferSession;

use flatppl_core::{Module, NodeId};

pub use catalogue::{Catalogue, parse_catalogue};
pub use modules::ModuleBundle;

/// Return the process-global built-in catalogue (parsed once from the bundled
/// `catalogue.ron`).
///
/// Exposes the same singleton as the crate-internal `catalogue::builtin()` but
/// makes it accessible to crates that drive the LSP — e.g. `flatppl-lsp` uses
/// this to enumerate base distribution/function names and standard-module
/// binding names for completion suggestions without duplicating the catalogue.
pub fn builtin_catalogue() -> &'static Catalogue {
    catalogue::builtin()
}

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

/// Like [`infer_module`], but also resolves `standard_module` references
/// against host-supplied external catalogues (merged with the built-in
/// catalogue).
///
/// The host pre-parses each external `.ron` catalogue file via
/// [`parse_catalogue`] and passes the slice here. If any external catalogue
/// introduces a module name that already exists in the built-in catalogue (or
/// in an earlier external catalogue in the slice), one module-level
/// `Severity::Error` diagnostic is emitted per collision; inference then
/// proceeds using the built-in-only catalogue for colliding names (the
/// external entry is shadowed, never silently merged).
///
/// Passing an empty `catalogues` slice is equivalent to calling
/// [`infer_module`] directly.
pub fn infer_module_ext(
    module: &mut Module,
    bundle: &ModuleBundle,
    catalogues: &[Catalogue],
    level: Level,
) -> Vec<Diagnostic> {
    let session = InferSession::with_external_catalogues(bundle, catalogues);
    // Surface collision errors before (or alongside) the trace run so the host
    // always sees them regardless of whether the colliding module is referenced.
    let mut diags = match session.catalogues.check_collisions() {
        Ok(()) => Vec::new(),
        Err(msg) => msg.lines().map(Diagnostic::error).collect::<Vec<_>>(),
    };
    diags.extend(trace::Inferencer::new(module, level, &session).run());
    diags
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
        b.insert(path, std::sync::Arc::new(dep));
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

    /// Spec §04 stochastic boundary: a `stochastic`-phase binding in the loaded
    /// module is invisible to the importer. A `draw` that has not been reified
    /// via `lawof`/`kernelof` may not be referenced across the module boundary.
    #[test]
    fn stochastic_dep_binding_is_invisible() {
        let bundle = bundle_with("h.flatppl", "x ~ Normal(0.0, 1.0)\nmu = 1.0");
        let mut model =
            flatppl_syntax::parse("h = load_module(\"h.flatppl\")\nv = h.x").expect("parses");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            diags
                .iter()
                .any(|d| d.severity == Severity::Error && d.message.contains("stochastic")),
            "a stochastic dep binding must be invisible; got {diags:?}"
        );
    }

    /// A `fixed`/`parameterized` binding right beside the stochastic one stays
    /// visible — the boundary hides only stochastic bindings.
    #[test]
    fn non_stochastic_dep_binding_stays_visible() {
        let bundle = bundle_with("h.flatppl", "x ~ Normal(0.0, 1.0)\nmu = 1.0");
        let mut model =
            flatppl_syntax::parse("h = load_module(\"h.flatppl\")\nv = h.mu").expect("parses");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            !diags
                .iter()
                .any(|d| d.severity == Severity::Error && d.message.contains("stochastic")),
            "a fixed dep binding must stay visible; got {diags:?}"
        );
    }

    /// Spec §04: access to loaded modules is NOT transitive. The importer reaches
    /// names in `mid`, but not names in `leaf` that `mid` loads but does not
    /// re-export — `m.seed` (seed lives in leaf) is "has no binding", not a
    /// silent two-hop reach.
    #[test]
    fn access_is_not_transitive() {
        let mut bundle = bundle_with("leaf.flatppl", "seed = 5");
        bundle.insert(
            "mid.flatppl",
            std::sync::Arc::new(
                flatppl_syntax::parse("leaf = load_module(\"leaf.flatppl\")").expect("mid parses"),
            ),
        );
        let mut model =
            flatppl_syntax::parse("m = load_module(\"mid.flatppl\")\nv = m.seed").expect("parses");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("has no binding `seed`")),
            "leaf's `seed` must not be reachable through mid; got {diags:?}"
        );
    }
}

#[cfg(test)]
mod substitution_tests {
    use super::*;

    #[test]
    fn substitution_flows_into_dependency() {
        // Unsubstituted, the dep gives `out: Real` (add(Real, Integer)). Seeding
        // `p` with an Integer-typed argument narrows `out` to Integer, so this
        // assertion fails if seeding is disabled. The source `a` is parameterized
        // and its set (`integers`) lies within the input's declared `reals` —
        // spec §04 compatible (phase + value set).
        let dep = flatppl_syntax::parse("p = elementof(reals)\nout = add(p, 1)").expect("dep");
        let mut bundle = ModuleBundle::new();
        bundle.insert("d.flatppl", std::sync::Arc::new(dep));
        let mut model = flatppl_syntax::parse(
            "a = elementof(integers)\nd = load_module(\"d.flatppl\", p = a)\nv = d.out",
        )
        .expect("model");
        let _ = infer_module(&mut model, &bundle, Level::Type);
        let (_id, vb) = model
            .bindings()
            .find(|(_, b)| model.resolve(b.name) == "v")
            .expect("binding `v` not found");
        assert_eq!(
            model.type_of(vb.rhs),
            Some(&flatppl_core::Type::Scalar(
                flatppl_core::ScalarType::Integer
            )),
            "out narrows to integer because substituted p is integer (Real otherwise)"
        );
    }

    #[test]
    fn same_dep_two_substitutions_memoize_separately() {
        let dep = flatppl_syntax::parse("p = elementof(reals)\nout = p").expect("dep");
        let mut bundle = ModuleBundle::new();
        bundle.insert("d.flatppl", std::sync::Arc::new(dep));
        // Both substitution sources are parameterized (spec §04: `elementof`
        // input ← parameterized) but carry distinct types, so the two loads
        // memoize separately (Integer vs Real).
        let mut model = flatppl_syntax::parse(
            "a = elementof(integers)\nb = elementof(reals)\nd1 = load_module(\"d.flatppl\", p = a)\nd2 = load_module(\"d.flatppl\", p = b)\nv1 = d1.out\nv2 = d2.out",
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
mod substitution_phase_tests {
    //! Spec §04 (Module composition → Load-time substitution): the phase of a
    //! loaded module's input governs what it may be bound to —
    //! `external` ← **fixed** only, `elementof` ← **parameterized** only.
    use super::*;

    fn bundle_with(path: &str, src: &str) -> ModuleBundle {
        let dep = flatppl_syntax::parse(src).expect("dep parses");
        let mut b = ModuleBundle::new();
        b.insert(path, std::sync::Arc::new(dep));
        b
    }

    /// `external` input ← fixed value: the allowed case, no diagnostic.
    /// (Mirrors the corrected `bayesian_inference_common`: `c = external(reals)`
    /// bound to a fixed `c_scaling = 5`.)
    #[test]
    fn external_input_bound_to_fixed_is_ok() {
        let bundle = bundle_with("d.flatppl", "c = external(reals)\nout = mul(c, 2.0)");
        let mut model = flatppl_syntax::parse(
            "c_scaling = 5\nd = load_module(\"d.flatppl\", c = c_scaling)\nv = d.out",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            !diags
                .iter()
                .any(|d| d.severity == Severity::Error && d.message.contains("may only be bound")),
            "external ← fixed is valid; got {diags:?}"
        );
    }

    /// `elementof` input ← fixed value: the spec violation (this was the bug in
    /// the pre-fix `bayesian_inference_common`). Must be an anchored Error.
    #[test]
    fn elementof_input_bound_to_fixed_is_an_error() {
        let bundle = bundle_with("d.flatppl", "c = elementof(reals)\nout = mul(c, 2.0)");
        let mut model = flatppl_syntax::parse(
            "c_scaling = 5\nd = load_module(\"d.flatppl\", c = c_scaling)\nv = d.out",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        let err = diags
            .iter()
            .find(|d| d.severity == Severity::Error && d.message.contains("may only be bound"));
        let err = err.unwrap_or_else(|| {
            panic!("elementof ← fixed must be an error; got {diags:?}");
        });
        assert!(
            err.message.contains("parameterized") && err.node.is_some(),
            "error must cite the parameterized requirement and anchor to the value node; got {err:?}"
        );
    }

    /// `external` input ← parameterized value: also a violation (external is a
    /// load-time-fixed hyperparameter).
    #[test]
    fn external_input_bound_to_parameterized_is_an_error() {
        let bundle = bundle_with("d.flatppl", "c = external(reals)\nout = mul(c, 2.0)");
        let mut model = flatppl_syntax::parse(
            "p = elementof(reals)\nd = load_module(\"d.flatppl\", c = p)\nv = d.out",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            diags.iter().any(|d| d.severity == Severity::Error
                && d.message.contains("may only be bound")
                && d.message.contains("fixed")),
            "external ← parameterized must be an error; got {diags:?}"
        );
    }

    /// `elementof` input ← parameterized value: the allowed case (the
    /// `load_module/` fixture's `center = a` shape), no phase diagnostic.
    #[test]
    fn elementof_input_bound_to_parameterized_is_ok() {
        let bundle = bundle_with("d.flatppl", "c = elementof(reals)\nout = mul(c, 2.0)");
        let mut model = flatppl_syntax::parse(
            "p = elementof(reals)\nd = load_module(\"d.flatppl\", c = p)\nv = d.out",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Type);
        assert!(
            !diags
                .iter()
                .any(|d| d.severity == Severity::Error && d.message.contains("may only be bound")),
            "elementof ← parameterized is valid; got {diags:?}"
        );
    }

    /// Spec §04: "Value sets must be compatible." Binding an input declared over
    /// `posreals` with a value whose set is `reals` (a proven strict superset)
    /// is incompatible — the value can take points outside the input's domain.
    #[test]
    fn substitution_value_set_wider_than_input_is_an_error() {
        let bundle = bundle_with("d.flatppl", "c = elementof(posreals)\nout = mul(c, 2.0)");
        let mut model = flatppl_syntax::parse(
            "p = elementof(reals)\nd = load_module(\"d.flatppl\", c = p)\nv = d.out",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Valueset);
        assert!(
            diags.iter().any(|d| d.severity == Severity::Error
                && d.message.contains("value set")
                && d.message.contains("posreals")),
            "reals ⊋ posreals substitution must be flagged; got {diags:?}"
        );
    }

    /// The compatible direction (`posreals` value into a `reals` input) is fine —
    /// and the conservative check must NOT false-positive when it cannot prove a
    /// strict superset.
    #[test]
    fn substitution_value_set_within_input_is_ok() {
        let bundle = bundle_with("d.flatppl", "c = elementof(reals)\nout = mul(c, 2.0)");
        let mut model = flatppl_syntax::parse(
            "p = elementof(posreals)\nd = load_module(\"d.flatppl\", c = p)\nv = d.out",
        )
        .expect("model");
        let diags = infer_module(&mut model, &bundle, Level::Valueset);
        assert!(
            !diags.iter().any(|d| d.message.contains("value set")),
            "posreals ⊆ reals is compatible; got {diags:?}"
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
        bundle.insert("d.flatppl", std::sync::Arc::new(dep));
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
        bundle.insert("a.flatppl", std::sync::Arc::new(a.clone()));
        bundle.insert("b.flatppl", std::sync::Arc::new(b));
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
        bundle.insert("helpers.flatppl", std::sync::Arc::new(helpers));
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
        bundle.insert("helpers.flatppl", std::sync::Arc::new(helpers));
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

#[cfg(test)]
mod infer_module_ext_tests {
    use super::*;
    use flatppl_core::{Mass, ScalarType, Type, ValueSet};

    /// RON source for a minimal external catalogue that defines one module
    /// `"myext"` with a single normalized real distribution `"MyDist"`.
    const MYEXT_RON: &str = r#"Catalogue(
        base: [],
        modules: [
            Module(name: "myext", version: "0.1", bindings: [
                Binding(name: "MyDist", sig: Distribution(
                    domain: Scalar(Real),
                    support: Reals,
                    mass: Normalized,
                )),
            ]),
        ],
    )"#;

    /// RON source for an external catalogue whose module name collides with
    /// the built-in `"particle-physics"` module.
    const COLLIDING_RON: &str = r#"Catalogue(
        base: [],
        modules: [
            Module(name: "particle-physics", version: "9.9", bindings: [
                Binding(name: "Dummy", sig: Distribution(
                    domain: Scalar(Real),
                    support: Reals,
                    mass: Normalized,
                )),
            ]),
        ],
    )"#;

    fn errors(diags: &[Diagnostic]) -> Vec<&Diagnostic> {
        diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect()
    }

    fn binding_ty<'m>(module: &'m flatppl_core::Module, name: &str) -> Option<&'m Type> {
        let rhs = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == name)?
            .1
            .rhs;
        module.type_of(rhs)
    }

    /// An external module's distribution applied via `standard_module` infers
    /// to a normalized real measure — proving a third-party module resolves
    /// end-to-end without built-in catalogue changes.
    #[test]
    fn external_distribution_resolves_and_infers_measure() {
        let ext = parse_catalogue(MYEXT_RON).expect("myext parses");
        let bundle = ModuleBundle::new();
        let src = r#"
e = standard_module("myext", "0.1")
x = e.MyDist(0.0, 1.0)
"#;
        let mut module = flatppl_syntax::parse(src).expect("source parses");
        let diags = infer_module_ext(&mut module, &bundle, &[ext], Level::Normalization);

        assert!(
            errors(&diags).is_empty(),
            "no errors expected for a valid external module; got {diags:?}"
        );

        let ty = binding_ty(&module, "x");
        assert!(
            matches!(
                ty,
                Some(Type::Measure { domain, mass: Mass::Normalized })
                    if **domain == Type::Scalar(ScalarType::Real)
            ),
            "x should be Measure(Real, Normalized); got {ty:?}"
        );

        let rhs = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == "x")
            .unwrap()
            .1
            .rhs;
        let vset = module.valueset_of(rhs).cloned();
        assert_eq!(
            vset,
            Some(ValueSet::Reals),
            "x's support should be Reals; got {vset:?}"
        );
    }

    /// An external catalogue whose module name collides with a built-in module
    /// surfaces a `"duplicate standard module"` error diagnostic.  The
    /// built-in wins: the colliding external entry is shadowed, so a reference
    /// to the built-in binding (`hepphys.CrystalBall`) still resolves to the
    /// built-in type (`Measure{real, Normalized}`).
    #[test]
    fn collision_with_builtin_emits_error_diagnostic() {
        let colliding = parse_catalogue(COLLIDING_RON).expect("colliding parses");
        let bundle = ModuleBundle::new();
        let src = r#"
hepphys = standard_module("particle-physics", "0.1")
y = hepphys.CrystalBall(0.0, 1.0, 1.5, 2.0)
"#;
        let mut module = flatppl_syntax::parse(src).expect("source parses");
        let diags = infer_module_ext(&mut module, &bundle, &[colliding], Level::Normalization);

        let collision_errors: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.severity == Severity::Error && d.message.contains("duplicate standard module")
            })
            .collect();
        assert!(
            !collision_errors.is_empty(),
            "expected a 'duplicate standard module' error; got {diags:?}"
        );
        assert!(
            collision_errors
                .iter()
                .any(|d| d.message.contains("particle-physics")),
            "collision error should name the colliding module; got {collision_errors:?}"
        );

        // Built-in wins: despite the collision, `hepphys.CrystalBall` must
        // still resolve to the built-in type — the external entry is shadowed,
        // never merged.  A regression that ejects the catalogue on collision
        // would leave `y` as Deferred or Failed.
        let ty = binding_ty(&module, "y");
        assert!(
            matches!(
                ty,
                Some(Type::Measure { domain, mass: Mass::Normalized })
                    if **domain == Type::Scalar(ScalarType::Real)
            ),
            "built-in wins: y should be Measure(Real, Normalized) despite the collision; got {ty:?}"
        );
    }

    /// `infer_module_ext` with an empty external slice produces identical
    /// results to `infer_module` on the same input.
    #[test]
    fn empty_external_is_equivalent_to_infer_module() {
        let src = r#"
hepphys = standard_module("particle-physics", "0.1")
y = hepphys.CrystalBall(0.0, 1.0, 1.5, 2.0)
"#;
        let bundle = ModuleBundle::new();

        let mut m_base = flatppl_syntax::parse(src).expect("parses");
        let diags_base = infer_module(&mut m_base, &bundle, Level::Normalization);

        let mut m_ext = flatppl_syntax::parse(src).expect("parses");
        let diags_ext = infer_module_ext(&mut m_ext, &bundle, &[], Level::Normalization);

        let ty_base = binding_ty(&m_base, "y").cloned();
        let ty_ext = binding_ty(&m_ext, "y").cloned();

        assert_eq!(
            ty_base, ty_ext,
            "empty-external infer_module_ext must match infer_module"
        );
        assert_eq!(
            errors(&diags_base).len(),
            errors(&diags_ext).len(),
            "error counts must match"
        );
    }
}

/// `div` / `mod` are integer-domain (spec §07): a real or complex operand is a
/// static error, integer operands type to `integer`, and real division stays
/// available as the separate `divide` op.
#[cfg(test)]
mod int_domain_div_mod_tests {
    use super::*;
    use flatppl_core::{ScalarType, Type};

    fn infer_src(src: &str) -> (flatppl_core::Module, Vec<Diagnostic>) {
        let mut module = flatppl_syntax::parse(src).expect("source parses");
        let diags = infer_with(&mut module, Level::Normalization);
        (module, diags)
    }

    fn errs(diags: &[Diagnostic]) -> Vec<&Diagnostic> {
        diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect()
    }

    fn ty<'m>(module: &'m flatppl_core::Module, name: &str) -> Option<&'m Type> {
        let rhs = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == name)?
            .1
            .rhs;
        module.type_of(rhs)
    }

    #[test]
    fn real_operand_to_div_is_an_integer_domain_error() {
        // One real operand (arg 1), one integer (arg 2): exactly one error,
        // anchored to the real operand — pins the per-operand check.
        let (_m, diags) = infer_src("a = 7.0\nb = 2\nq = div(a, b)\n");
        let e = errs(&diags);
        assert_eq!(e.len(), 1, "one error expected; got {diags:?}");
        assert!(
            e[0].message.contains("`div` is integer-domain")
                && e[0].message.contains("argument 1 is real")
                && e[0].message.contains("use `divide` for real division"),
            "message should name argument 1 (real) + the divide hint; got {:?}",
            e[0].message
        );
    }

    #[test]
    fn real_operand_to_mod_is_an_integer_domain_error() {
        let (_m, diags) = infer_src("a = 7\nb = 2.0\nr = mod(a, b)\n");
        let e = errs(&diags);
        assert_eq!(e.len(), 1, "one error expected; got {diags:?}");
        assert!(
            e[0].message.contains("`mod` is integer-domain")
                && e[0].message.contains("argument 2 is real"),
            "message should name argument 2 (real); got {:?}",
            e[0].message
        );
        // `mod` is not real division, so it carries no `divide` hint.
        assert!(!e[0].message.contains("use `divide`"));
    }

    #[test]
    fn both_real_operands_flag_each_argument() {
        // Both operands real → one diagnostic per offending operand.
        let (_m, diags) = infer_src("a = 7.0\nb = 2.0\nq = div(a, b)\n");
        assert_eq!(
            errs(&diags).len(),
            2,
            "both operands flagged; got {diags:?}"
        );
    }

    #[test]
    fn integer_operands_to_div_mod_type_to_integer_with_no_error() {
        let (m, diags) = infer_src("a = 7\nb = 2\nq = div(a, b)\nr = mod(a, b)\n");
        assert!(errs(&diags).is_empty(), "no errors expected; got {diags:?}");
        assert_eq!(ty(&m, "q"), Some(&Type::Scalar(ScalarType::Integer)));
        assert_eq!(ty(&m, "r"), Some(&Type::Scalar(ScalarType::Integer)));
    }

    #[test]
    fn divide_keeps_real_division_no_integer_domain_check() {
        let (m, diags) = infer_src("a = 7.0\nb = 2.0\nq = divide(a, b)\n");
        assert!(errs(&diags).is_empty(), "no errors expected; got {diags:?}");
        assert_eq!(ty(&m, "q"), Some(&Type::Scalar(ScalarType::Real)));
    }
}
