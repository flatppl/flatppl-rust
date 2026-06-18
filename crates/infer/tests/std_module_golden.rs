//! Integration tests for §09 standard-module references in inference.
//!
//! `hepphys = standard_module("particle-physics", "0.1")` followed by a
//! reference to `hepphys.CrystalBall(...)` (a distribution) or
//! `specfns.erf(x)` (a function) must INFER by resolving the reference against
//! the built-in catalogue (Task 6 added the data + `Catalogue::module`).
//!
//! These models are built from the real surface form so the parser lowering
//! to `RefNs::Module` is exercised end to end. No fixtures are read.

use flatppl_core::{Dim, Mass, ScalarType, Type, ValueSet};
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

/// Parse `src`, infer at `level` with an empty bundle (standard modules are
/// resolved from the built-in catalogue, not the host bundle), and return the
/// annotated module plus diagnostics.
fn infer_src(src: &str, level: Level) -> (flatppl_core::Module, Vec<flatppl_infer::Diagnostic>) {
    let mut module = flatppl_syntax::parse(src).expect("source parses");
    let diags = infer_module(&mut module, &ModuleBundle::new(), level);
    (module, diags)
}

fn binding_ty<'m>(module: &'m flatppl_core::Module, name: &str) -> Option<&'m Type> {
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)?
        .1
        .rhs;
    module.type_of(rhs)
}

fn binding_vset(module: &flatppl_core::Module, name: &str) -> Option<ValueSet> {
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)?
        .1
        .rhs;
    module.valueset_of(rhs).cloned()
}

fn errors(diags: &[flatppl_infer::Diagnostic]) -> Vec<&flatppl_infer::Diagnostic> {
    diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

/// A standard-module distribution applied (`hepphys.CrystalBall(...)`) infers a
/// normalized real measure with support over the reals.
#[test]
fn std_distribution_applied_infers_measure() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
y = hepphys.CrystalBall(0.0, 1.0, 1.5, 2.0)
"#;
    let (module, diags) = infer_src(src, Level::Normalization);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "y");
    assert!(
        matches!(
            ty,
            Some(Type::Measure { domain, mass: Mass::Normalized })
                if **domain == Type::Scalar(ScalarType::Real)
        ),
        "y should be Measure over (%scalar real) with normalized mass; got {ty:?}"
    );

    let vset = binding_vset(&module, "y");
    assert_eq!(
        vset,
        Some(ValueSet::Reals),
        "y's support should be Reals; got {vset:?}"
    );
}

/// A §09 distribution that is NOT a probability measure (`ContinuedPoisson`,
/// spec §09: "not normalized, and so not a probability measure"; catalogue
/// `mass: Finite`) keeps its catalogue mass — it must NOT be forced to
/// Normalized like the probability distributions.
#[test]
fn std_non_probability_distribution_keeps_finite_mass() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
y = hepphys.ContinuedPoisson(2.0)
"#;
    let (module, diags) = infer_src(src, Level::Normalization);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "y");
    assert!(
        matches!(
            ty,
            Some(Type::Measure {
                mass: Mass::Finite,
                ..
            })
        ),
        "ContinuedPoisson should be a Finite (non-probability) measure; got {ty:?}"
    );
}

/// `broadcast(hepphys.ContinuedPoisson, rates)` over a §09 standard-module
/// distribution head is an independent product over the array — a measure,
/// exactly like a built-in distribution head (`broadcast(Poisson, …)`), and it
/// keeps the catalogue's `Finite` mass rather than being forced to Normalized.
#[test]
fn broadcast_of_std_distribution_infers_array_measure() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
rates = [2.0, 3.0]
y = broadcast(hepphys.ContinuedPoisson, rates)
"#;
    let (module, diags) = infer_src(src, Level::Normalization);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "y");
    assert!(
        matches!(
            ty,
            Some(Type::Measure { domain, mass: Mass::Finite })
                if matches!(
                    domain.as_ref(),
                    Type::Array { elem, .. } if **elem == Type::Scalar(ScalarType::Real)
                )
        ),
        "broadcast of ContinuedPoisson should be a Finite measure over an array of reals; got {ty:?}"
    );
}

/// `broadcast(hepphys.interp_poly6_exp, …)` over a §09 standard-module
/// *function* head maps elementwise into an array of the per-cell result type
/// (here real → real), just like the built-in deterministic-op path
/// (`broadcast(add, …)`) — not a measure, and never `%deferred`. The per-bin
/// templates (lo/nom/hi) are arrays while the nuisance parameter `alpha` is a
/// single scalar that rides along — exactly the histfactory normsys shape.
#[test]
fn broadcast_of_std_function_infers_array_value() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
lo = [0.9, 0.8]
nom = [1.0, 1.0]
hi = [1.1, 1.2]
alpha = elementof(reals)
y = broadcast(hepphys.interp_poly6_exp, lo, nom, hi, alpha)
"#;
    let (module, diags) = infer_src(src, Level::Type);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "y");
    assert!(
        matches!(
            ty,
            Some(Type::Array { shape, elem })
                if shape.len() == 1 && **elem == Type::Scalar(ScalarType::Real)
        ),
        "broadcast of interp_poly6_exp should be an array of reals (a value, not a measure); got {ty:?}"
    );
}

/// A §09 function whose result is a different scalar kind than its inputs
/// (`hepphys.resonance_breitwigner(…) → Complex`) broadcasts to an array of that
/// result kind — the cell type comes from the catalogue sig, not the inputs.
#[test]
fn broadcast_of_std_function_uses_catalogue_result_kind() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
sigma = [1.0, 2.0]
m = [3.0, 4.0]
width = [0.1, 0.2]
ma = [0.0, 0.0]
mb = [0.0, 0.0]
l = [1, 1]
d = [1.0, 1.0]
y = broadcast(hepphys.resonance_breitwigner, sigma, m, width, ma, mb, l, d)
"#;
    let (module, diags) = infer_src(src, Level::Type);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "y");
    assert!(
        matches!(
            ty,
            Some(Type::Array { elem, .. }) if **elem == Type::Scalar(ScalarType::Complex)
        ),
        "broadcast of resonance_breitwigner should be an array of complex; got {ty:?}"
    );
}

/// End-to-end of the converter's constraint chain (the reported inference gap):
/// a boundary-less `functionof` over a `broadcast` of a §09 distribution is a
/// kernel over its single parametric leaf, and `likelihoodof` of it carries that
/// input — the whole chain types, no silent `%deferred`.
#[test]
fn functionof_of_std_distribution_broadcast_chains_to_likelihood() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
g = elementof(cartpow(posreals, 2))
tau = [4.0, 9.0]
constraint = functionof(broadcast(hepphys.ContinuedPoisson, broadcast(mul, g, tau)))
L = likelihoodof(constraint, tau)
"#;
    let (module, diags) = infer_src(src, Level::Type);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let constraint_ty = binding_ty(&module, "constraint");
    assert!(
        matches!(constraint_ty, Some(Type::Kernel { inputs, .. }) if inputs.len() == 1),
        "constraint should be a Kernel over its one parametric leaf; got {constraint_ty:?}"
    );
    let l_ty = binding_ty(&module, "L");
    assert!(
        matches!(l_ty, Some(Type::Likelihood { inputs, .. }) if inputs.len() == 1),
        "L should be a Likelihood carrying the auto-traced input; got {l_ty:?}"
    );
}

/// A standard-module function applied (`specfns.erf(x)`) follows the argument's
/// scalar kind: real in, real out.
#[test]
fn std_function_applied_infers_scalar() {
    let src = r#"
specfns = standard_module("special-functions", "0.1")
x = elementof(reals)
y = specfns.erf(x)
"#;
    let (module, diags) = infer_src(src, Level::Valueset);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "y");
    assert_eq!(
        ty,
        Some(&Type::Scalar(ScalarType::Real)),
        "erf(real) should be (%scalar real); got {ty:?}"
    );
}

/// `linalg.kron(A, B)` infers a (dynamic-dim) matrix type.
#[test]
fn std_function_matrix_result() {
    let src = r#"
linalg = standard_module("ext-linear-algebra", "0.1")
A = elementof(reals)
B = elementof(reals)
M = linalg.kron(A, B)
"#;
    let (module, diags) = infer_src(src, Level::Type);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "M");
    assert!(
        matches!(
            ty,
            Some(Type::Array { shape, elem })
                if shape.len() == 2
                    && **elem == Type::Scalar(ScalarType::Real)
        ),
        "kron should infer a 2-D real matrix type; got {ty:?}"
    );
}

/// A degraded distribution (`hepphys.Argus(...)`) lowers its support to the
/// degraded value (PosReals) and emits a `Severity::Note` explaining the
/// honest-degrade approximation.
#[test]
fn std_degraded_distribution_support() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
y = hepphys.Argus(1.0, 2.0, 3.0)
"#;
    let (module, diags) = infer_src(src, Level::Valueset);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let vset = binding_vset(&module, "y");
    assert_eq!(
        vset,
        Some(ValueSet::PosReals),
        "Argus support should degrade to PosReals; got {vset:?}"
    );

    // The degraded note must be surfaced so the user knows why the support is
    // approximate (spec policy: honest-degrade notification).
    let notes: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Note)
        .collect();
    assert!(
        !notes.is_empty(),
        "applying a degraded §09 binding must emit a Severity::Note; got {diags:?}"
    );
    assert!(
        notes
            .iter()
            .any(|d| d.message.contains("interval(0, resonance)")),
        "degraded note should mention the exact spec support; got {notes:?}"
    );
}

/// A bare standard-module distribution reference (not applied) types like a
/// bare base distribution name (`Type::Any`).
#[test]
fn std_bare_distribution_ref() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
cb = hepphys.CrystalBall
"#;
    let (module, diags) = infer_src(src, Level::Type);
    assert!(errors(&diags).is_empty(), "unexpected errors: {diags:?}");

    let ty = binding_ty(&module, "cb");
    assert_eq!(
        ty,
        Some(&Type::Any),
        "bare distribution ref should be Type::Any (matching a bare base name); got {ty:?}"
    );
}

/// An unknown module name resolves to a "not found" error.
#[test]
fn std_unknown_module_errors_not_found() {
    let src = r#"
m = standard_module("no-such-module", "0.1")
y = m.Whatever(1.0)
"#;
    let (_module, diags) = infer_src(src, Level::Type);
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("not found")),
        "expected a `not found` error; got {diags:?}"
    );
}

/// A reference to a non-existent binding of a real module errors with "has no
/// binding".
#[test]
fn std_missing_binding_errors() {
    let src = r#"
hepphys = standard_module("particle-physics", "0.1")
y = hepphys.NotABinding(1.0)
"#;
    let (_module, diags) = infer_src(src, Level::Type);
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("has no binding")),
        "expected a `has no binding` error; got {diags:?}"
    );
}

/// A wrong requested version errors with "unknown version".
#[test]
fn std_wrong_version_errors() {
    let src = r#"
hepphys = standard_module("particle-physics", "9.9")
y = hepphys.CrystalBall(0.0, 1.0, 1.5, 2.0)
"#;
    let (_module, diags) = infer_src(src, Level::Type);
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("unknown version")),
        "expected an `unknown version` error; got {diags:?}"
    );
}

/// Sanity: the dynamic-dim kron matrix is not accidentally a fixed shape at a
/// non-shape level (guards against threading a stale dim).
#[test]
fn std_kron_dims_dynamic_without_shape_level() {
    let src = r#"
linalg = standard_module("ext-linear-algebra", "0.1")
A = elementof(reals)
B = elementof(reals)
M = linalg.kron(A, B)
"#;
    let (module, _diags) = infer_src(src, Level::Type);
    if let Some(Type::Array { shape, .. }) = binding_ty(&module, "M") {
        assert_eq!(shape.as_ref(), &[Dim::Dynamic, Dim::Dynamic]);
    } else {
        panic!("kron should be an array type");
    }
}
