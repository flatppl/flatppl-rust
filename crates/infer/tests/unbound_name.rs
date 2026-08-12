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

/// A §09 standard-module member is a `RefNs::Module` ref resolved against the
/// catalogue, never a bare atom.
#[test]
fn standard_module_members_resolve() {
    assert_clean(
        "pp = standard_module(\"particle-physics\", \"0.1\")\n\
         y = pp.kallen(1.0, 2.0, 3.0)\n",
    );
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
