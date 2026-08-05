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

/// Parse + infer, returning the module and the diagnostics unfiltered.
fn infer_diags(src: &str) -> (flatppl_core::Module, Vec<flatppl_infer::Diagnostic>) {
    let mut module = flatppl_syntax::parse(src).expect("source parses");
    let diags = infer_module(&mut module, &ModuleBundle::new(), Level::Type);
    (module, diags)
}

/// Spec §04 *Placeholders and holes*: "All placeholders must appear both in the
/// expression to be reified and the boundary input keyword arguments." An
/// `%autoinputs` boundary declares NO placeholder — the auto-trace records
/// `elementof` leaves only — so this module violates the rule, and before this
/// check it inferred with ZERO diagnostics and scored with a dangling
/// `(%ref %local _v_)` inside `builtin_logdensityof`.
#[test]
fn a_placeholder_no_boundary_declares_is_a_static_error() {
    let src = "F = functionof(Normal(mu = _v_, sigma = 1.0))\n\
               lp = logdensityof(lawof(F), 0.5)";
    let (module, diags) = infer_diags(src);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "exactly one error: {diags:?}");
    let d = errors[0];
    assert!(
        d.message.contains("placeholder `_v_`")
            && d.message.contains("no boundary input declares it")
            && d.message.contains(
                "All placeholders must appear both in the expression to be reified and the \
                 boundary input keyword arguments"
            )
            && d.message.contains("declare it as `v = _v_`"),
        "the message names the placeholder, quotes §04 and gives the fix: {}",
        d.message
    );
    // Anchored at the placeholder occurrence itself, not the reification.
    let span = module
        .span_of(d.node.expect("anchored"))
        .expect("the placeholder ref carries a span");
    assert_eq!(
        &src[span.start as usize..span.end as usize],
        "_v_",
        "the position is the placeholder occurrence"
    );
}

/// The same body with the placeholder DECLARED is legal and keeps inferring: the
/// error is about the missing declaration, not about placeholders.
#[test]
fn a_declared_placeholder_still_infers() {
    let src = "F = functionof(Normal(mu = _v_, sigma = 1.0), v = _v_)";
    let module = infer_src(src, Level::Type);
    match binding_ty(&module, "F") {
        Some(Type::Kernel { inputs, .. }) => assert_eq!(input_names(&module, inputs), ["v"]),
        other => panic!("F should be a Kernel over `v`; got {other:?}"),
    }
}

/// A lambda's placeholder is declared by the `%specinputs` boundary the sugar
/// emits (`v -> …` becomes `functionof(…, v = _v_)`), so lambdas are untouched.
#[test]
fn a_lambda_placeholder_is_declared_by_its_own_boundary() {
    let src = "M = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 1.0, sigma = 2.0))\n\
               P = pushfwd(v -> get(v, [\"a\"]), M)\n\
               q = logdensityof(P, 0.5)";
    let _ = infer_src(src, Level::Type);
}

/// §04's own DISALLOWED nesting example. The INNER reification must be the one
/// that errors: "A placeholder in an inner `functionof` or `kernelof` **must**
/// be bound there", so `_c_` being declared by the OUTER boundary does not
/// rescue it. `_d_` IS declared by the boundary that reaches it, and is silent.
#[test]
fn an_inner_reification_must_declare_its_own_placeholder() {
    let src = "b = 2.0\n\
               some_value = 3.0\n\
               g = functionof(functionof(_a_ * b + _c_, a = _a_)(a = some_value) + _d_, \
               c = _c_, d = _d_)";
    let (_, diags) = infer_diags(src);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "one error, at the inner scope: {diags:?}");
    assert!(
        errors[0].message.contains("placeholder `_c_`"),
        "`_c_` is the one the inner boundary fails to declare: {}",
        errors[0].message
    );
}

/// §04's LEGAL nesting example, the control for the test above: the same
/// placeholder name in two scopes, each bound where it occurs.
#[test]
fn the_same_placeholder_name_in_two_scopes_is_legal() {
    let src = "b = 2.0\n\
               some_value = 3.0\n\
               g = functionof(functionof(_a_ * b, a = _a_)(a = some_value) + _a_, a = _a_)";
    let _ = infer_src(src, Level::Type);
}

/// The walk follows self-refs, so hiding the placeholder one binding away does
/// not evade the check (§04 forbids such a binding outright — "An expression
/// with placeholders … must *not* appear outside of a `functionof(...)` or
/// `kernelof(...)`" — which nothing enforces yet; this catches it at the use).
#[test]
fn a_placeholder_reached_through_a_self_ref_is_caught() {
    let src = "expr = _v_ * 2.0\n\
               F = functionof(expr)";
    let (_, diags) = infer_diags(src);
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("placeholder `_v_`")),
        "the ref must be followed: {diags:?}"
    );
}
