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
