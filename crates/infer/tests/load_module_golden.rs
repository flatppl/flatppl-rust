//! Golden integration test for the spec §11 two-module example.
//!
//! `helpers.flatppl` defines `center`, `spread`, `obs_kernel`, and
//! `shifted_value`. `model.flatppl` loads it with a substitution
//! (`center = a`), draws `b`, and forms a likelihood `L`.
//!
//! With a populated bundle the cross-module reference resolves cleanly and `L`
//! must infer to a concrete `Likelihood` type — not `%deferred` or `%failed`.

use std::path::PathBuf;

use flatppl_core::{Mass, ScalarType, Type};
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

fn read_fixture(name: &str) -> String {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "../../fixtures/flatppl/load_module",
        name,
    ]
    .iter()
    .collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading fixture {name}: {e}"))
}

#[test]
fn spec_example_infers_cross_module() {
    let helpers_src = read_fixture("helpers.flatppl");
    let model_src = read_fixture("model.flatppl");

    let helpers = flatppl_syntax::parse(&helpers_src).expect("helpers.flatppl parses");
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", std::sync::Arc::new(helpers));

    let mut model = flatppl_syntax::parse(&model_src).expect("model.flatppl parses");
    let diags = infer_module(&mut model, &bundle, Level::Shape);

    // No "not found" or "deferred" errors — the bundle satisfies the dependency.
    let unexpected: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.severity == Severity::Error
                && (d.message.contains("not found") || d.message.contains("deferred"))
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "unexpected error diagnostics: {unexpected:?}"
    );

    // `L` must resolve to a concrete Likelihood type — not Deferred or Failed.
    let lb = model
        .bindings()
        .find(|(_, b)| model.resolve(b.name) == "L")
        .expect("model has an `L` binding")
        .1
        .rhs;
    let ty = model.type_of(lb);
    assert!(
        matches!(
            ty,
            Some(flatppl_core::Type::Likelihood { obstype, .. })
                if **obstype == flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real)
        ),
        "L should be Likelihood over (%scalar real); got {ty:?}; diags: {diags:?}"
    );

    // Also assert the types of helpers' own bindings as standalone inference.
    // Run helpers standalone at Level::Shape so obs_kernel and shifted_value
    // are fully annotated.
    let helpers_src = read_fixture("helpers.flatppl");
    let mut helpers_anno = flatppl_syntax::parse(&helpers_src).expect("helpers parses");
    let _ = infer_module(&mut helpers_anno, &ModuleBundle::new(), Level::Shape);

    // `obs_kernel = functionof(Normal(…), center=…, spread=…, x=…)`
    // Normal(…) is a Measure, so functionof-of-measure is a Kernel.
    let obs_kernel_rhs = helpers_anno
        .bindings()
        .find(|(_, b)| helpers_anno.resolve(b.name) == "obs_kernel")
        .expect("helpers has `obs_kernel`")
        .1
        .rhs;
    let obs_kernel_ty = helpers_anno.type_of(obs_kernel_rhs);
    assert!(
        matches!(obs_kernel_ty, Some(Type::Kernel { .. })),
        "obs_kernel should be Type::Kernel; got {obs_kernel_ty:?}"
    );

    // `shifted_value = center + 1.0` where center = elementof(reals) → Real.
    let shifted_value_rhs = helpers_anno
        .bindings()
        .find(|(_, b)| helpers_anno.resolve(b.name) == "shifted_value")
        .expect("helpers has `shifted_value`")
        .1
        .rhs;
    let shifted_value_ty = helpers_anno.type_of(shifted_value_rhs);
    assert_eq!(
        shifted_value_ty,
        Some(&Type::Scalar(ScalarType::Real)),
        "shifted_value should be (%scalar real); got {shifted_value_ty:?}"
    );

    // Confirm obs_kernel's mass class: at Level::Normalization, Normal is
    // a normalized measure, so the kernel's mass is Normalized.
    // (helpers is re-inferred at Normalization level to check mass.)
    let mut helpers_norm = flatppl_syntax::parse(&helpers_src).expect("helpers parses");
    let _ = infer_module(
        &mut helpers_norm,
        &ModuleBundle::new(),
        Level::Normalization,
    );
    let obs_kernel_norm_rhs = helpers_norm
        .bindings()
        .find(|(_, b)| helpers_norm.resolve(b.name) == "obs_kernel")
        .expect("helpers has `obs_kernel`")
        .1
        .rhs;
    let obs_kernel_norm_ty = helpers_norm.type_of(obs_kernel_norm_rhs);
    assert!(
        matches!(
            obs_kernel_norm_ty,
            Some(Type::Kernel {
                mass: Mass::Normalized,
                ..
            })
        ),
        "obs_kernel at Normalization level should be Kernel{{mass: Normalized}}; got {obs_kernel_norm_ty:?}"
    );
}

/// Cross-module callable APPLICATION (load-module flavor b): module B imports
/// A's reified callable and APPLIES it — concrete argument types from B flow
/// through A's callable body to produce a typed result.
///
/// Setup:
///   A (helpers) defines `obs_kernel = functionof(Normal(...), center=..., x=...)`
///   B (model)   loads A with `center = a` and does `L = likelihoodof(helpers.obs_kernel, obs)`
///
/// The applied result `L` must infer to a concrete `Likelihood` whose obstype is
/// `(%scalar real)` — i.e., the real-valued argument `obs` flowed through
/// `Normal`'s domain inside A's callable body and produced a concrete type.
///
/// If the engine does NOT yet flow concrete args through a cross-module callable
/// body, `L` would be `%deferred` or `%failed` rather than a concrete
/// `Likelihood`; in that case this test is `#[ignore]`'d with a clear reason.
#[test]
fn cross_module_callable_application_yields_concrete_result() {
    // Module A: the kernel is a reified callable over Normal, parameterized by
    // `center` (substituted via load_module) and free variable `x`.
    let a_src = "center = elementof(reals)\nspread = elementof(posreals)\n\
         obs_kernel = functionof(Normal(mu = add(center, _x_), sigma = spread), \
         center = center, spread = spread, x = _x_)";

    // Module B: loads A with a concrete `center` substitution, then applies the
    // imported callable to a concrete scalar observation.
    let b_src = "a = elementof(reals)\n\
         helpers = load_module(\"helpers.flatppl\", center = a)\n\
         obs = 2.5\n\
         L = likelihoodof(helpers.obs_kernel, obs)";

    let helpers = flatppl_syntax::parse(a_src).expect("helpers (module A) parses");
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", std::sync::Arc::new(helpers));

    let mut model = flatppl_syntax::parse(b_src).expect("model (module B) parses");
    let diags = infer_module(&mut model, &bundle, Level::Normalization);

    // No resolution errors — the bundle satisfies the dependency.
    let unresolved: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.severity == flatppl_infer::Severity::Error
                && (d.message.contains("not found") || d.message.contains("deferred"))
        })
        .collect();
    assert!(
        unresolved.is_empty(),
        "unexpected resolution errors: {unresolved:?}"
    );

    // The applied result `L` must be a concrete Likelihood — NOT Deferred/Failed.
    // Concrete arg types (real obs) must have flowed through A's callable body.
    let lb = model
        .bindings()
        .find(|(_, b)| model.resolve(b.name) == "L")
        .expect("model has an `L` binding")
        .1
        .rhs;
    let ty = model.type_of(lb);
    assert!(
        matches!(
            ty,
            Some(Type::Likelihood { obstype, .. })
                if **obstype == Type::Scalar(ScalarType::Real)
        ),
        "L = likelihoodof(helpers.obs_kernel, obs) must infer to \
         Likelihood{{obstype: (%scalar real)}} after concrete args flow \
         through A's callable body; got {ty:?}; diags: {diags:?}"
    );
}
