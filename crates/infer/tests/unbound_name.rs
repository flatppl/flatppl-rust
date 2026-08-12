//! Spec §04 "Name resolution": an unqualified name resolves to a current-module
//! binding, else to a built-in, and "Unresolvable names are static errors".
//!
//! The parser turns every unqualified name it does not find among the module's
//! bindings into a bare `base` atom (`Node::Const`). Before this gate, an atom
//! naming no built-in was typed `Type::Any` / `%fixed` and absorbed silently, so
//! `infer` exited 0 and the determiniser lowered a FREE VARIABLE into FlatPDL —
//! the shape below, found by the testsuite's shared-latent sweep. The
//! `self.`-qualified spelling of the same mistake always errored.

use flatppl_infer::{Diagnostic, Severity};

fn errors(src: &str) -> Vec<Diagnostic> {
    let mut m = flatppl_syntax::parse(src).expect("fixture must parse");
    flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn assert_unresolvable(src: &str, name: &str) {
    let errs = errors(src);
    let hit = errs
        .iter()
        .find(|d| d.message.contains(&format!("unresolvable name `{name}`")));
    let Some(d) = hit else {
        let msgs: Vec<_> = errs.iter().map(|d| &d.message).collect();
        panic!("expected an unresolvable-name error for `{name}`; got: {msgs:?}");
    };
    // Located, so an editor and the CLI can point at the offending name.
    assert!(
        d.node.is_some(),
        "the diagnostic must carry its node: {d:?}"
    );
}

fn assert_unresolvable_call(src: &str, name: &str) {
    let errs = errors(src);
    let hit = errs.iter().find(|d| {
        d.message
            .contains(&format!("unresolvable call to `{name}`"))
    });
    let Some(d) = hit else {
        let msgs: Vec<_> = errs.iter().map(|d| &d.message).collect();
        panic!("expected an unresolvable-call error for `{name}`; got: {msgs:?}");
    };
    assert!(
        d.node.is_some(),
        "the diagnostic must carry its node: {d:?}"
    );
}

fn assert_clean(src: &str) {
    let msgs: Vec<String> = errors(src).into_iter().map(|d| d.message).collect();
    assert!(msgs.is_empty(), "expected no errors; got: {msgs:?}");
}

/// The reproducer. `f1` names a joint COMPONENT, which binds nothing in scope
/// (§04: "Record field names and table column names are local to their object
/// and not part of the global module namespace"), so `Normal(mu = f1, …)`
/// references an unresolvable name.
#[test]
fn joint_component_name_is_not_in_scope_keyword_spelling() {
    assert_unresolvable(
        "z = draw(Normal(mu = 0.4, sigma = 1.0))\n\
         q = logdensityof(joint(f1 = Normal(mu = z, sigma = 0.5), \
         f2 = Normal(mu = f1, sigma = 1.5)), record(f1 = 0.3, f2 = 0.7))\n",
        "f1",
    );
}

/// The positional `joint` spelling reaches the same check.
#[test]
fn joint_component_name_is_not_in_scope_positional_spelling() {
    assert_unresolvable(
        "z = draw(Normal(mu = 0.4, sigma = 1.0))\n\
         q = logdensityof(joint(Normal(mu = z, sigma = 0.5), \
         Normal(mu = f1, sigma = 1.5)), [0.3, 0.7])\n",
        "f1",
    );
}

/// Nothing about the defect is `joint`-specific: any unqualified name with no
/// binding is unresolvable, including in a plain arithmetic call.
#[test]
fn bare_unbound_name_in_value_position_is_an_error() {
    assert_unresolvable("y = add(nosuchname, 1.0)\n", "nosuchname");
}

/// A record FIELD name is local to its record, so reusing one as a value
/// reference does not resolve either.
#[test]
fn record_field_name_is_not_a_binding() {
    assert_unresolvable("r = record(a = 1.0)\ny = add(a, 2.0)\n", "a");
}

// --- Legitimate late resolution: everything below must stay clean. -----------
//
// The gate needs the whole `base` namespace, not just the rows in
// `catalogue.ron`, so these pin the categories the catalogue does not carry.

/// A `functionof` placeholder is a `%local` ref, not a bare atom — a different
/// `infer_node` arm, untouched by the gate.
#[test]
fn reification_placeholders_resolve() {
    assert_clean("g = functionof(add(_x_, 1.0), x = _x_)\ny = g(2.0)\n");
}

/// A `fn` hole desugars to the same `%local` placeholder.
#[test]
fn fn_holes_resolve() {
    assert_clean("f = fn(add(_, 1.0))\ny = f(2.0)\n");
}

/// §03 set names and §07 constants are bare atoms with no `catalogue.ron` row.
#[test]
fn set_names_and_constants_resolve() {
    assert_clean(
        "p = elementof(posreals)\n\
         q = elementof(interval(0.0, 1.0))\n\
         s = external(reals)\n\
         y = mul(pi, inf)\n\
         z = im\n",
    );
}

/// A built-in used as a VALUE — the `sum` in `reduce(sum, …)`, the `Poisson` in
/// `broadcast(Poisson, …)` — is a bare atom in an ordinary argument position.
#[test]
fn builtins_as_values_resolve() {
    assert_clean(
        "v = [1.0, 2.0, 3.0]\n\
         y = reduce(sum, v)\n\
         m = broadcast(Poisson, v)\n",
    );
}

/// `log2` has a `catalogue.ron` row but is absent from the keyword list, so the
/// union in `builtins::is_base_name` is load-bearing. Removing either half of it
/// reddens this test or `set_names_and_constants_resolve`.
#[test]
fn catalogue_only_builtins_resolve() {
    assert_clean("y = log2(8.0)\n");
}

/// Shadowing keeps working: §04 makes built-in names bindable, and a binding
/// wins over the built-in at step 1 of resolution.
#[test]
fn a_binding_shadowing_a_builtin_resolves() {
    assert_clean("sum = 3.0\ny = add(sum, 1.0)\n");
}

/// The metric of a `metricsum` binding (§05: "The marker form `metric: C[...] :=
/// expr`") is an ordinary reference — resolvable when bound, and reported when
/// not.
#[test]
fn metricsum_metric_resolves_when_bound() {
    assert_clean(
        "g = [[1.0, 0.0], [0.0, -1.0]]\n\
         L1 = [[1.0, 2.0], [3.0, 4.0]]\n\
         L2 = [[1.0, 0.0], [0.0, 1.0]]\n\
         g: C[.mu^, .nu_] := L1[.mu^, .beta_] * L2[.beta^, .nu_]\n",
    );
}

#[test]
fn metricsum_metric_is_reported_when_unbound() {
    assert_unresolvable(
        "L1 = [[1.0, 2.0], [3.0, 4.0]]\n\
         L2 = [[1.0, 0.0], [0.0, 1.0]]\n\
         g: C[.mu^, .nu_] := L1[.mu^, .beta_] * L2[.beta^, .nu_]\n",
        "g",
    );
}

// --- The CALL-HEAD half of the same §04 rule. -------------------------------
//
// A bare `name(...)` call parses to `CallHead::Builtin(name)`, not to a
// `Node::Const`, so the bare-atom arm never sees it. `ops::call_rule`'s
// catalogue-dispatch fallthrough used to answer every unrecognised head with a
// `%deferred` note, which is honest for a builtin awaiting a type rule and wrong
// for a head that names nothing: `y = nromal(1.0)` inferred with only a note and
// the determiniser emitted the free call verbatim.

/// A typo'd distribution name is a static error, not a `%deferred` note.
#[test]
fn unknown_call_head_is_an_error() {
    assert_unresolvable_call("y = nromal(1.0)\n", "nromal");
}

/// The same for a head that looks like a function.
#[test]
fn unknown_function_call_head_is_an_error() {
    assert_unresolvable_call("v = [1.0, 2.0]\ny = nosuchfn(v)\n", "nosuchfn");
}

/// A builtin that HAS no type rule keeps its `%deferred` note — the gate
/// separates "no rule yet" from "no such name", and must not collapse the two.
/// `Lebesgue` is a §08 distribution with no catalogue row of any kind.
#[test]
fn a_rowless_builtin_head_still_defers_rather_than_erroring() {
    assert_clean("m = Lebesgue(dims = 1)\n");
}

/// A user-defined callable is `CallHead::User`, resolved through the callee, and
/// never reaches the builtin-head arm.
#[test]
fn user_defined_call_heads_resolve() {
    assert_clean("f(x) = add(x, 1.0)\ny = f(2.0)\n");
}

// --- §09 members: bare is unresolvable, alias-qualified is not. -------------
//
// A §09 standard-module member has a row in `catalogues/particle-physics.ron`
// but lives in that MODULE's namespace, not in `base`. §09 gives no unqualified
// spelling, so a bare occurrence is the unresolvable-name case. Before this,
// `y = add(kallen, 1.0)` determinized to `y = kallen + 1.0` with `kallen`
// unbound — a wrong lowering, not merely a missed error.

#[test]
fn a_bare_module_member_is_unresolvable() {
    assert_unresolvable("y = add(kallen, 1.0)\n", "kallen");
}

#[test]
fn a_bare_module_member_call_is_unresolvable() {
    assert_unresolvable_call("y = kallen(1.0, 2.0, 3.0)\n", "kallen");
}

/// Including the eight §09 DISTRIBUTION constructors, which the keyword list
/// listed alongside the base ones.
#[test]
fn a_bare_module_distribution_constructor_is_unresolvable() {
    assert_unresolvable_call("y = CrystalBall(0.0, 1.0, 1.5, 2.0)\n", "CrystalBall");
}

/// The accept side: behind its alias the same member resolves, because that is a
/// `RefNs::Module` ref checked against the module catalogue — a path that never
/// consults `is_base_name`.
#[test]
fn an_alias_qualified_module_function_resolves() {
    assert_clean(
        "pp = standard_module(\"particle-physics\", \"0.1\")\n\
         y = pp.kallen(1.0, 2.0, 3.0)\n",
    );
}

#[test]
fn an_alias_qualified_module_distribution_resolves() {
    assert_clean(
        "pp = standard_module(\"particle-physics\", \"0.1\")\n\
         m = pp.CrystalBall(m0 = 0.0, sigma = 1.0, alpha = 1.5, n = 2.0)\n\
         lp = logdensityof(m, 0.5)\n",
    );
}

/// The five §08 distributions with no catalogue row still resolve bare. This is
/// the pair to `a_bare_module_distribution_constructor_is_unresolvable`: both
/// sets were called "§08 distributions with no row" before the review, and only
/// these five actually are.
#[test]
fn rowless_base_distributions_resolve_bare() {
    assert_clean("m = Dirac(x = 1.0)\nlp = logdensityof(m, 1.0)\n");
}

// --- The kernel-tag exemption is a SLOT, not a name. -------------------------
//
// The determiniser emits a bare §09 constructor as a kernel tag
// (`broadcast(builtin_logdensityof, ContinuedPoisson, …)`) and re-runs inference
// over its own output, so a constructor in the TAG SLOT must resolve. An earlier
// cut exempted every argument of any `builtin_*` / `broadcast` call and narrowed
// only by name, which let a constructor pass in three slots that are not the tag.
// `builtins::kernel_tag_node` now decides the slot from the §07 signatures.

/// The observed-value argument of `builtin_logdensityof` (§07: `kernel,
/// kernel_input, x`) is not the tag slot.
#[test]
fn a_constructor_in_the_observed_value_slot_is_unresolvable() {
    assert_unresolvable(
        "y = builtin_logdensityof(Normal, record(mu = 0.0, sigma = 1.0), CrystalBall)\n",
        "CrystalBall",
    );
}

/// Nor is the kernel-input (params) argument.
#[test]
fn a_constructor_in_the_params_slot_is_unresolvable() {
    assert_unresolvable(
        "y = builtin_logdensityof(Normal, CrystalBall, 0.5)\n",
        "CrystalBall",
    );
}

/// Nor the rngstate argument of `builtin_sample` (§07: `rngstate, kernel,
/// kernel_input, n, m, …`), whose tag is the SECOND argument, not the first.
#[test]
fn a_constructor_in_the_rngstate_slot_is_unresolvable() {
    assert_unresolvable(
        "y = builtin_sample(CrystalBall, Normal, record(mu = 0.0, sigma = 1.0))\n",
        "CrystalBall",
    );
}

/// The accept control for the two positional tag slots — index 0 for
/// `builtin_logdensityof`, index 1 for `builtin_sample`. These are the shapes the
/// determiniser emits, so narrowing must not touch them.
#[test]
fn a_constructor_in_the_tag_slot_resolves() {
    assert_clean(
        "y = builtin_logdensityof(CrystalBall, \
         record(m0 = 0.0, sigma = 1.0, alpha = 1.5, n = 2.0), 0.5)\n",
    );
    assert_clean(
        "s = builtin_sample(rnginit(0), CrystalBall, \
         record(m0 = 0.0, sigma = 1.0, alpha = 1.5, n = 2.0))\n",
    );
}

/// The tag slot resolves through the keyword spelling too — §07 names the
/// parameter `kernel`, and `arity_check` accepts keyword calls to the primitives.
#[test]
fn a_constructor_in_the_keyword_tag_slot_resolves() {
    assert_clean(
        "y = builtin_logdensityof(kernel = CrystalBall, \
         kernel_input = record(m0 = 0.0, sigma = 1.0, alpha = 1.5, n = 2.0), x = 0.5)\n",
    );
}

/// The exemption also requires the name to BE a constructor, so a §09 function is
/// unresolvable even sitting in the tag slot.
#[test]
fn a_module_function_in_the_tag_slot_is_still_unresolvable() {
    assert_unresolvable(
        "y = builtin_logdensityof(kallen, record(a = 1.0), 0.5)\n",
        "kallen",
    );
}
