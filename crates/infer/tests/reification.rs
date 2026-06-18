//! Boundary-less reification auto-trace (spec §04, FlatPIR `%autoinputs`).
//!
//! A `functionof` / `kernelof` written with no explicit boundary discovers its
//! inputs as the `elementof` parametric-phase leaves of the body's ancestor
//! subgraph, in **canonical order (sorted by name)** — so a converter's
//! incidental build order never leaks into the input list. These tests pin that
//! discovery (the hs3/pyhf importers emit bare `functionof(model)` and rely on
//! it; before this landed their reifications stayed `%deferred`).

use flatppl_core::{Mass, Type};
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

/// Parse + infer with an empty bundle; assert no errors; return the module.
fn infer_src(src: &str, level: Level) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(src).expect("source parses");
    let diags = infer_module(&mut module, &ModuleBundle::new(), level);
    assert!(
        diags.iter().all(|d| d.severity != Severity::Error),
        "unexpected errors: {diags:?}"
    );
    module
}

fn binding_ty<'m>(module: &'m flatppl_core::Module, name: &str) -> Option<&'m Type> {
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)?
        .1
        .rhs;
    module.type_of(rhs)
}

fn input_names<'m>(
    module: &'m flatppl_core::Module,
    inputs: &[flatppl_core::Symbol],
) -> Vec<&'m str> {
    inputs.iter().map(|s| module.resolve(*s)).collect()
}

/// A boundary-less `functionof` over a measure body is a KERNEL whose inputs are
/// the body's `elementof` leaves, **sorted by name** — not the build order.
#[test]
fn auto_inputs_are_elementof_leaves_sorted_by_name() {
    // `zeta` is bound and used before `alpha`; the input list must still be
    // [alpha, zeta], proving discovery order does not leak.
    let src = r#"
zeta = elementof(reals)
alpha = elementof(posreals)
expected = add(zeta, alpha)
model = functionof(Normal(mu = expected, sigma = alpha))
"#;
    // Normalization level so the kernel's mass slot is filled (Normal ⇒ a
    // Markov kernel) alongside the input list.
    let module = infer_src(src, Level::Normalization);
    match binding_ty(&module, "model") {
        Some(Type::Kernel {
            inputs,
            mass: Mass::Normalized,
        }) => assert_eq!(
            input_names(&module, inputs),
            ["alpha", "zeta"],
            "auto-inputs must be canonical (name-sorted)"
        ),
        other => panic!("model should be a normalized Kernel; got {other:?}"),
    }
}

/// A boundary-less `functionof` over a VALUE body is a Function (not a kernel);
/// its inputs are still the body's `elementof` leaves, deduped across reuse.
#[test]
fn auto_inputs_value_body_is_function_deduped() {
    let src = r#"
a = elementof(reals)
b = elementof(reals)
y = functionof(add(mul(a, a), b))
"#;
    let module = infer_src(src, Level::Type);
    match binding_ty(&module, "y") {
        Some(Type::Function { inputs }) => assert_eq!(
            input_names(&module, inputs),
            ["a", "b"],
            "`a` used twice must appear once"
        ),
        other => panic!("y should be a Function; got {other:?}"),
    }
}

/// A fixed-phase ancestor (no `elementof` under it) is closed over, not an
/// input: only the genuine parametric leaf becomes an input.
#[test]
fn auto_inputs_close_over_fixed_ancestors() {
    let src = r#"
nominal = [5.0, 10.0]
mu = elementof(reals)
expected = broadcast(mul, nominal, mu)
model = functionof(broadcast(Poisson, expected))
"#;
    let module = infer_src(src, Level::Type);
    match binding_ty(&module, "model") {
        Some(Type::Kernel { inputs, .. }) => assert_eq!(
            input_names(&module, inputs),
            ["mu"],
            "the fixed `nominal` data is closed over, not an input"
        ),
        other => panic!("model should be a Kernel; got {other:?}"),
    }
}

/// A leaf reached by two distinct paths (a shared sub-expression / diamond) is
/// recorded once: the ancestor walk's visited-set both dedupes the input and
/// stops the second traversal from re-descending the shared subgraph.
#[test]
fn auto_inputs_dedupe_shared_subexpression() {
    let src = r#"
a = elementof(reals)
shared = mul(a, a)
model = functionof(add(shared, shared))
"#;
    let module = infer_src(src, Level::Type);
    match binding_ty(&module, "model") {
        Some(Type::Function { inputs }) => assert_eq!(
            input_names(&module, inputs),
            ["a"],
            "the diamond's shared leaf must appear once"
        ),
        other => panic!("model should be a Function; got {other:?}"),
    }
}

/// The walk descends through an *alias* binding (`b = a`, an RHS that is a bare
/// reference, not a call) to reach the genuine `elementof` leaf behind it.
#[test]
fn auto_inputs_descend_through_alias() {
    let src = r#"
a = elementof(reals)
b = a
model = functionof(add(b, b))
"#;
    let module = infer_src(src, Level::Type);
    match binding_ty(&module, "model") {
        Some(Type::Function { inputs }) => assert_eq!(
            input_names(&module, inputs),
            ["a"],
            "the alias `b` resolves to its leaf `a`"
        ),
        other => panic!("model should be a Function; got {other:?}"),
    }
}

/// The walk descends through a *user-callable application* in the body — the
/// callee is followed and the argument's `elementof` leaf is discovered — so a
/// reification over a body that calls a helper binding still types its inputs.
#[test]
fn auto_inputs_descend_through_user_call() {
    let src = r#"
a = elementof(reals)
helper = functionof(mul(a, a))
applied = helper(a)
model = functionof(add(applied, a))
"#;
    let module = infer_src(src, Level::Type);
    match binding_ty(&module, "model") {
        Some(Type::Function { inputs }) => assert_eq!(
            input_names(&module, inputs),
            ["a"],
            "the user-call's leaf must be discovered through callee + args"
        ),
        other => panic!("model should be a Function; got {other:?}"),
    }
}
