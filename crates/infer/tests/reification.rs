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

#[test]
fn auto_inputs_follow_long_cached_alias_chains() {
    let mut src = String::from("x0 = elementof(reals)\n");
    for i in 1..=12000 {
        src.push_str(&format!("x{i} = x{}\n", i - 1));
    }
    src.push_str("model = functionof(x12000)\n");
    let module = infer_src(&src, Level::Type);
    match binding_ty(&module, "model") {
        Some(Type::Function { inputs }) => assert_eq!(input_names(&module, inputs), ["x0"]),
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
/// expression to be reified and the boundary input keyword arguments." The
/// auto-trace declares NO placeholder — it records `elementof` leaves only — so
/// this module violates the rule, and before this check it inferred with ZERO
/// diagnostics and scored with a dangling `(%ref %local _v_)` inside
/// `builtin_logdensityof`.
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

/// FlatPIR may carry an explicit entry list under `%autoinputs` (the reader
/// accepts one there), and an entry targeting a placeholder declares it exactly
/// as a `%specinputs` entry does — §04 asks only that the placeholder "appear …
/// in the boundary input keyword arguments", not which origin tag records them.
/// No workspace producer emits this shape, hence the hand-written FlatPIR.
#[test]
fn an_autoinputs_entry_declares_its_placeholder() {
    let pir = "(%module\n  \
       (%public F)\n  \
       (%bind F (functionof (Normal (%kwarg mu (%ref %local _v_)) (%kwarg sigma 1.0)) \
       %autoinputs ((v (%ref %local _v_))))))";
    let mut module = flatppl_flatpir::read(pir).expect("hand-written FlatPIR reads");
    let diags = infer_module(&mut module, &ModuleBundle::new(), Level::Type);
    assert!(
        diags.iter().all(|d| d.severity != Severity::Error),
        "the `%autoinputs` entry declares `_v_`: {diags:?}"
    );
}

/// Spec §04 *Specifying reification boundaries*: "Boundary input names must be
/// distinct — a repeated name is a static error, which likewise forbids a lambda
/// or named function from repeating an argument name." Every §05 sugar lowers to
/// the same `%specinputs` boundary, so all three spellings must error. Before
/// this check each inferred with ZERO diagnostics as a function over `(a, a)`.
#[test]
fn a_repeated_boundary_input_name_is_a_static_error() {
    for src in [
        "f(a, a) = add(a, a)",
        "g = (a, a) -> add(a, a)",
        "c = 1.0\nh = functionof(add(c, c), a = c, a = c)",
        "m = elementof(reals)\nx ~ Normal(mu = m, sigma = 1.0)\nk = kernelof(x, a = m, a = m)",
    ] {
        let (_, diags) = infer_diags(src);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert_eq!(errors.len(), 1, "exactly one error for `{src}`: {diags:?}");
        assert!(
            errors[0]
                .message
                .contains("boundary input `a` is declared more than once")
                && errors[0].message.contains(
                    "Boundary input names must be distinct — a repeated name is a static error"
                ),
            "the message names the input and quotes §04: {}",
            errors[0].message
        );
    }
}

/// The diagnostic is anchored at the reification node. Boundary entries are plain
/// `(Symbol, Ref)` data with no node of their own, so the repeated name has no
/// span of its own to carry; for the `f(a, a) = …` sugar the reification's span is
/// the body, since the surface argument list sits at the binding name and is
/// recoverable from the source text only (`crates/lsp/src/names.rs`).
#[test]
fn a_repeated_boundary_input_is_anchored_at_the_reification() {
    let src = "f(a, a) = add(a, a)";
    let (module, diags) = infer_diags(src);
    let d = diags
        .iter()
        .find(|d| d.severity == Severity::Error)
        .expect("one error");
    let span = module
        .span_of(d.node.expect("anchored"))
        .expect("the reification carries a span");
    assert_eq!(&src[span.start as usize..span.end as usize], "add(a, a)");
}

/// The control: distinct names are silent, and one repeated name is reported
/// once however many times it repeats.
#[test]
fn distinct_boundary_input_names_are_silent() {
    let src = "c = 1.0\nd = 2.0\nh = functionof(add(c, d), p = c, q = d)";
    let (_, diags) = infer_diags(src);
    assert!(
        diags.iter().all(|d| d.severity != Severity::Error),
        "distinct names are legal: {diags:?}"
    );
}

#[test]
fn a_thrice_repeated_boundary_input_reports_once() {
    let src = "c = 1.0\nh = functionof(add(c, c), a = c, a = c, a = c)";
    let (_, diags) = infer_diags(src);
    assert_eq!(
        diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count(),
        1,
        "one error per repeated name: {diags:?}"
    );
}

/// An explicit `%autoinputs` entry list can repeat a name too, and §04's rule is
/// about the boundary, not about which origin tag records it.
#[test]
fn a_repeated_autoinputs_name_is_a_static_error() {
    let pir = "(%module\n  \
       (%public other F)\n  \
       (%bind other (elementof reals))\n  \
       (%bind F (functionof (add (%ref self other) (%ref self other)) \
       %autoinputs ((a (%ref self other)) (a (%ref self other))))))";
    let mut module = flatppl_flatpir::read(pir).expect("hand-written FlatPIR reads");
    let diags = infer_module(&mut module, &ModuleBundle::new(), Level::Type);
    assert!(
        diags.iter().any(|d| d.severity == Severity::Error
            && d.message
                .contains("boundary input `a` is declared more than once")),
        "the `%autoinputs` boundary repeats `a`: {diags:?}"
    );
}

/// The control for the test above: an `%autoinputs` list that does NOT target
/// the placeholder leaves it undeclared, so the §04 check still fires.
#[test]
fn an_autoinputs_list_that_misses_the_placeholder_still_errors() {
    let pir = "(%module\n  \
       (%public other F)\n  \
       (%bind other (elementof reals))\n  \
       (%bind F (functionof (Normal (%kwarg mu (%ref %local _v_)) (%kwarg sigma 1.0)) \
       %autoinputs ((w (%ref self other))))))";
    let mut module = flatppl_flatpir::read(pir).expect("hand-written FlatPIR reads");
    let diags = infer_module(&mut module, &ModuleBundle::new(), Level::Type);
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("placeholder `_v_`")),
        "an entry for `other` declares no placeholder: {diags:?}"
    );
}

// ---------------------------------------------------------------------------
// §04 reification: the PHASE of a reified callable follows its CAPTURED
// ancestors.
//
// A `functionof` may reference `draw` nodes of the enclosing graph. They stay
// shared ancestors with a single realization, so the callable is conditional on
// that realization — never resampled per call, never marginalized. What that
// costs is determinism, and the phase is where it is paid: a reification that
// captures a draw is itself `%stochastic`. §04's "FlatPPL has no closures"
// narrows to the deterministic ancestors it names ("a fixed ancestor is
// resolved to its value").
//
// Before this, EVERY reification typed `%fixed` by construction, so the §02
// overview's `model_R` claimed determinism while its density depended on the
// drawn `raw_syst`. The reified callable's own annotation is now the record of
// what it is conditional on.
// ---------------------------------------------------------------------------

/// The inferred phase of binding `name`'s right-hand side.
fn binding_phase(module: &flatppl_core::Module, name: &str) -> Option<flatppl_core::Phase> {
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)?
        .1
        .rhs;
    module.phase_of(rhs)
}

/// Infer and assert the module is error-free; return it. `Level::Type` rather
/// than `Level::Phase` because a `Phase` run annotates no types, and these tests
/// read the input list alongside the phase.
fn infer_phases(src: &str) -> flatppl_core::Module {
    infer_src(src, Level::Type)
}

/// The §02 overview shape: a boundary-less `functionof` over a measure whose
/// intensity descends from a drawn systematic. Accepted, and STOCHASTIC — the
/// kernel is conditional on `raw_syst`'s single realization.
#[test]
fn a_boundary_less_functionof_that_captures_a_draw_is_stochastic() {
    let src = "\
n_sig = elementof(reals)
raw_syst ~ Normal(mu = 0.0, sigma = 1.0)
resolution = mul(2.5, exp(mul(0.12, raw_syst)))
intensity = weighted(n_sig, Normal(mu = 125.0, sigma = resolution))
model_R = functionof(PoissonProcess(intensity = truncate(intensity, interval(2.0, 8.0))))";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "model_R"),
        Some(flatppl_core::Phase::Stochastic),
        "capturing the drawn `raw_syst` makes the reification stochastic"
    );
    // The captured draw is NOT an input: only the `elementof` leaf is.
    match binding_ty(&module, "model_R") {
        Some(Type::Kernel { inputs, .. }) => assert_eq!(
            input_names(&module, inputs),
            ["n_sig"],
            "a captured draw is an ancestor, not a boundary input"
        ),
        other => panic!("model_R should be a Kernel; got {other:?}"),
    }
}

/// Naming the draw as a boundary input is the CONDITIONAL kernel: §04
/// substitutes it with a fresh `elementof(valueset(a))` input before the trace
/// runs, so nothing is captured and the callable is `%fixed`.
#[test]
fn a_boundary_input_over_the_draw_is_a_fixed_conditional_kernel() {
    let src = "\
n_sig = elementof(reals)
raw_syst ~ Normal(mu = 0.0, sigma = 1.0)
resolution = mul(2.5, exp(mul(0.12, raw_syst)))
intensity = weighted(n_sig, Normal(mu = 125.0, sigma = resolution))
model_R = functionof(PoissonProcess(intensity = truncate(intensity, interval(2.0, 8.0))),
    n_sig = n_sig, raw_syst = raw_syst)";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "model_R"),
        Some(flatppl_core::Phase::Fixed),
        "a boundary-named draw is substituted away, so nothing is captured"
    );
    match binding_ty(&module, "model_R") {
        Some(Type::Kernel { inputs, .. }) => assert_eq!(
            input_names(&module, inputs),
            ["n_sig", "raw_syst"],
            "both boundary inputs ride into the kernel type"
        ),
        other => panic!("model_R should be a Kernel; got {other:?}"),
    }
}

/// `lawof` absorbs, so a body over a reified measure captures nothing and stays
/// `%fixed`. §04 "Phase of the reified law": "`lawof` absorbs stochasticity into
/// the reified law rather than propagating it outward."
#[test]
fn a_functionof_over_a_lawof_reified_measure_is_fixed() {
    let src = "\
m = elementof(reals)
x ~ Normal(mu = m, sigma = 1.0)
f = functionof(lawof(x))";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "f"),
        Some(flatppl_core::Phase::Fixed),
        "the draw is absorbed by `lawof`, not captured"
    );
    assert!(
        matches!(binding_ty(&module, "f"), Some(Type::Kernel { .. })),
        "f should be a Kernel; got {:?}",
        binding_ty(&module, "f")
    );
}

/// `kernelof` is never stochastic. §04 "Kernels and `kernelof`" makes
/// `kernelof(x, kwargs…)` equivalent to `functionof(lawof(x), kwargs…)`, so its
/// whole body sits under an implicit `lawof`. Both the fully-cut and the
/// marginalizing spelling are `%fixed` — the second is the `prior_predictive`
/// case §04 blesses ("they are internal stochastic nodes in the traced sub-DAG,
/// not boundary inputs, so `lawof` integrates them out").
#[test]
fn kernelof_is_fixed_whatever_its_boundary_leaves_uncut() {
    for src in [
        "m = elementof(reals)\nx ~ Normal(mu = m, sigma = 1.0)\nk = kernelof(x)",
        "m = elementof(reals)\nx ~ Normal(mu = m, sigma = 1.0)\nk = kernelof(x, m = m)",
        "mu ~ Normal(mu = 0.0, sigma = 1.0)\n\
         y ~ Normal(mu = mu, sigma = 1.0)\n\
         k = kernelof(record(y = y))",
    ] {
        let module = infer_phases(src);
        assert_eq!(
            binding_phase(&module, "k"),
            Some(flatppl_core::Phase::Fixed),
            "`kernelof` reifies the law, so it captures no draw: `{src}`"
        );
    }
}

/// A draw written INLINE in the body is captured just as a bound one is.
#[test]
fn an_inline_draw_in_the_body_is_captured() {
    let src = "f = functionof(mul(2.0, draw(Normal(mu = 0.0, sigma = 1.0))))";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "f"),
        Some(flatppl_core::Phase::Stochastic),
        "an anonymous draw is still a captured stochastic ancestor"
    );
}

/// A boundary that cuts one draw and leaves another is still stochastic: the
/// remaining capture is what decides. Substitution runs before the trace, so the
/// cut draw is out of the subgraph entirely.
#[test]
fn a_partial_boundary_stays_stochastic_on_the_uncut_draw() {
    let src = "\
a ~ Normal(mu = 0.0, sigma = 1.0)
b ~ Normal(mu = 0.0, sigma = 1.0)
f = functionof(add(a, b), a = a)";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "f"),
        Some(flatppl_core::Phase::Stochastic),
        "`b` is still captured"
    );
    // The control: cutting BOTH leaves nothing captured.
    let both = "\
a ~ Normal(mu = 0.0, sigma = 1.0)
b ~ Normal(mu = 0.0, sigma = 1.0)
f = functionof(add(a, b), a = a, b = b)";
    assert_eq!(
        binding_phase(&infer_phases(both), "f"),
        Some(flatppl_core::Phase::Fixed),
        "both draws substituted away"
    );
}

/// A nested reification carries its OWN phase. The walk cuts at the inner
/// `functionof`, so the outer one is stochastic only through what IT captures —
/// a callable value is not itself a stochastic node.
#[test]
fn a_nested_reification_carries_its_own_phase() {
    let src = "\
c = elementof(reals)
a ~ Normal(mu = 0.0, sigma = 1.0)
inner = functionof(add(a, c))
outer = functionof(inner(c))";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "inner"),
        Some(flatppl_core::Phase::Stochastic),
        "`inner` captures `a`"
    );
    assert_eq!(
        binding_phase(&module, "outer"),
        Some(flatppl_core::Phase::Fixed),
        "`outer` captures no draw of its own; it applies a callable"
    );
}

/// A fully-cut nesting is fixed at both levels.
#[test]
fn a_fully_cut_nesting_is_fixed_at_both_levels() {
    let src = "\
c = elementof(reals)
a ~ Normal(mu = 0.0, sigma = 1.0)
inner = functionof(add(a, c), a = a, c = c)
outer = functionof(inner(a, c), a = a, c = c)";
    let module = infer_phases(src);
    for name in ["inner", "outer"] {
        assert_eq!(
            binding_phase(&module, name),
            Some(flatppl_core::Phase::Fixed),
            "`{name}` captures nothing"
        );
    }
}

/// A lambda is `functionof` with placeholders (§04 *Lambda notation*), so it
/// captures the same way: a lambda over a drawn value is stochastic. This is the
/// shape both `flatppl-examples` models use, and it now infers.
#[test]
fn a_lambda_closing_over_a_draw_is_stochastic() {
    let src = "\
s ~ normalize(truncate(Cauchy(0.0, 1.0), interval(0.0, inf)))
step = prev -> Normal(mu = prev, sigma = s)";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "step"),
        Some(flatppl_core::Phase::Stochastic),
        "the lambda is conditional on `s`'s realization"
    );
}

/// The control: a lambda over parameterized and fixed ancestors only stays
/// `%fixed`. A parameterized ancestor is not a draw.
#[test]
fn a_lambda_over_a_parametric_ancestor_is_fixed() {
    let src = "\
s = elementof(posreals)
step = prev -> Normal(mu = prev, sigma = s)";
    assert_eq!(
        binding_phase(&infer_phases(src), "step"),
        Some(flatppl_core::Phase::Fixed),
        "a parameterized ancestor is not a captured draw"
    );
}

/// The phase rides outward through an ordinary consumer: a value derived from a
/// stochastic reification is stochastic too, by §04's ancestor rule. This is
/// what makes the capture visible to everything downstream rather than stopping
/// at the callable.
#[test]
fn the_captured_phase_propagates_to_a_consumer() {
    let src = "\
s ~ Normal(mu = 0.0, sigma = 1.0)
f = functionof(mul(2.0, s))
y = f()
z = add(y, 1.0)";
    let module = infer_phases(src);
    for name in ["f", "y", "z"] {
        assert_eq!(
            binding_phase(&module, name),
            Some(flatppl_core::Phase::Stochastic),
            "`{name}` inherits the capture"
        );
    }
}

/// A stochastic reification captured by `markovchain` is ONE shared value across
/// every step, not a fresh draw per step. The chain's step kernel is the §04
/// lambda that captures it, so this pins the ruling's "single realisation" at the
/// place it matters most.
#[test]
fn a_markovchain_step_kernel_captures_one_shared_draw() {
    let src = "\
sigma_step ~ normalize(truncate(Cauchy(0.0, 1.0), interval(0.0, inf)))
step_kernel = prev -> Normal(mu = prev, sigma = sigma_step)
x ~ markovchain(step_kernel, 0.0, 120)";
    let module = infer_phases(src);
    assert_eq!(
        binding_phase(&module, "step_kernel"),
        Some(flatppl_core::Phase::Stochastic),
        "the step kernel captures `sigma_step`"
    );
    assert_eq!(
        binding_phase(&module, "x"),
        Some(flatppl_core::Phase::Stochastic),
        "the chain is stochastic through both its own draw and the capture"
    );
    // The capture is recorded ONCE, as a single ancestor binding — the property
    // that makes it one realization shared by all 120 steps rather than 120
    // independent draws. `sigma_step` appears once in the step kernel's trace.
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "step_kernel")
        .expect("step_kernel is bound")
        .1
        .rhs;
    let mut refs = 0;
    let mut pending = vec![rhs];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let flatppl_core::Node::Ref(r) = module.node(id) {
            if module.resolve(r.name) == "sigma_step" {
                refs += 1;
            }
        }
        pending.extend(module.node(id).children());
    }
    assert_eq!(
        refs, 1,
        "one ref node to the captured draw, not one per step"
    );
}
