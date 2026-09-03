//! §04 special-operation arity, and §09 standard-module member arity and names.
//!
//! §04 "Calling conventions" declares each special operation's input count in
//! prose rather than in the catalogue, so the catalogue-driven `arity_check`
//! answered `None` for every one of them and a nullary or over-supplied call
//! typed at exit 0. §09 members had no arity or name check at all:
//! `user_arity_check` returned early for a module ref.
//!
//! These tests check COUNTS, not spellings. §04 contradicts itself on whether a
//! distinguished input may be passed by keyword; see `ops::SpecialArity`'s doc
//! comment and `flatppl-dev/audit-fix-infer.md`.
//!
//! Assertions read the DIAGNOSTIC SET, not the rendered module: a substring of
//! the printed IR can be satisfied by a child node, and a refusal that produced
//! no diagnostic would pass such a check silently. Every refusal is also checked
//! to carry a source location, since a diagnostic with none is unusable in an
//! editor and unreportable by the linter.

use flatppl_infer::{Diagnostic, Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d: &Diagnostic| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// Every error diagnostic carries a node to anchor it at. A refusal with no
/// location is not usable in an editor and not reportable by the linter.
fn errors_are_located(src: &str) -> bool {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    let errs: Vec<&Diagnostic> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    !errs.is_empty() && errs.iter().all(|d| d.node.is_some())
}

fn assert_refuses(src: &str, needle: &str) {
    let errs = errors(src);
    assert!(
        errs.iter().any(|e| e.contains(needle)),
        "`{src}` must refuse mentioning {needle}; got: {errs:?}"
    );
    assert!(errors_are_located(src), "`{src}` refused with no location");
}

fn assert_clean(src: &str) {
    let errs = errors(src);
    assert!(errs.is_empty(), "`{src}` must type cleanly; got: {errs:?}");
}

// ---- §04: special-operation arity -------------------------------------------

/// §04 "Calling conventions": "Nullary calls (`f()`) are not allowed", and for
/// the special operations specifically "The total number of inputs is never
/// zero." Seven nullary calls typed at exit 0 before this — `record()` as an
/// empty record and both loaders as `%module`.
#[test]
fn nullary_special_operations_are_refused() {
    for src in [
        "a = aggregate()\n",
        "a = metricsum()\n",
        "a = load_module()\n",
        "a = standard_module()\n",
        "a = record()\n",
        "a = table()\n",
        "a = broadcast()\n",
        "a = draw()\n",
        "a = elementof()\n",
    ] {
        assert_refuses(src, "spec §04 \"Calling conventions\"");
    }
}

/// §04's nullary rule reaches the four variadic PRODUCT and CHAIN heads too, with
/// no exemption (owner ruling). An earlier revision exempted them because three
/// `spec_coverage_measures.rs` tests reached design-PR #73's no-laundering rider
/// through `joint()`, the only measure reachable from source whose mass is
/// genuinely `%deferred`. That coverage moved to `ops.rs`
/// `mod no_laundering_tests`, asserted against `product_mass` and `lawof_type`
/// directly, so the rider no longer depends on `joint()` being spellable and §04
/// applies to all four.
#[test]
fn the_nullary_product_and_chain_heads_are_refused() {
    for src in [
        "e = joint()\n",
        "e = jointchain()\n",
        "e = cartprod()\n",
        "e = superpose()\n",
    ] {
        assert_refuses(src, "takes at least 1 input");
    }
}

/// Each of the four still types with components, so the rule is about the nullary
/// call and not about the head.
#[test]
fn the_product_and_chain_heads_type_with_components() {
    assert_clean("m = joint(Normal(0.0, 1.0), Normal(2.0, 3.0))\n");
    assert_clean("s = cartprod(reals, integers)\n");
    assert_clean("m = superpose(Normal(0.0, 1.0), Normal(2.0, 3.0))\n");
}

/// `vector` is the one head whose catalogue row admits zero arguments, and the
/// case is real: §04 "Multi-axis aggregation" says "The empty axis list `[]` is
/// legal and denotes full reduction to a scalar", and an empty axis list lowers
/// to `(vector)`. The nullary rule must not reach a head whose row has ruled.
///
/// `[]` on its own is a DIFFERENT question and stays refused: §05's
/// `ArrayLiteral` admits no empty form, so a bare `[]` is an axis list outside
/// its one legal position. This test pins the full-contraction spelling, which is
/// what `flatppl-examples/examples/dminus-to-3pi-amplitude.flatppl` writes.
#[test]
fn a_full_contraction_metricsum_still_types() {
    assert_clean(
        "g = rowstack([[1.0, 0.0], [0.0, -1.0]])\nv = [1.0, 2.0]\n\
         g: s[] := v[.mu^] * v[.mu_]\n",
    );
}

/// §04 "Calling conventions": "`standard_module`: Two distinguished inputs", and
/// §04 "Standard modules": "`standard_module` only accepts positional arguments".
/// Under- and over-supply both typed `%module` before, the third argument silently
/// discarded.
#[test]
fn standard_module_takes_exactly_two_positional_inputs() {
    assert_refuses(
        "m = standard_module(\"particle-physics\")\n",
        "takes 2 input",
    );
    assert_refuses(
        "m = standard_module(\"particle-physics\", \"0.1\", \"ignored\")\n",
        "takes 2 input",
    );
    assert_refuses(
        "m = standard_module(name = \"particle-physics\", compat = \"0.1\")\n",
        "no keyword arguments",
    );
}

/// §04: "`aggregate`, `metricsum`, `markovchain`, `kscan`: Three distinguished
/// inputs". The COUNT only — see `SpecialArity`'s doc comment for why the keyword
/// spelling is left alone, and `axis_position.rs` for the test that accepts it.
#[test]
fn three_input_special_operations_take_three_inputs() {
    assert_refuses("A = [1, 2]\ns = aggregate(sum, [])\n", "takes 3 input");
    assert_refuses(
        "A = [1, 2]\ns = aggregate(sum, [], A[.i], 4)\n",
        "takes 3 input",
    );
}

/// The keyword `aggregate` spelling `axis_position.rs` accepts stays accepted:
/// this change checks counts, not spellings.
#[test]
fn the_keyword_aggregate_spelling_is_untouched() {
    assert_clean(
        "A = rowstack([[1.0, 0.0], [0.0, 1.0]])\n\
         x = aggregate(f_reduction = sum, output_axes = [.i], expr = A[.i, .j])\n",
    );
}

/// §04: "`tuple`: Unnamed variadic inputs with significant order (minimum two)".
#[test]
fn a_one_element_tuple_is_refused() {
    assert_refuses("t = tuple(1)\n", "at least 2 input");
}

/// §04: "`load_module`: One distinguished input plus optional variadic named
/// inputs with no significant order" — so the substitution keywords stay legal
/// alongside the one positional source.
#[test]
fn load_module_keeps_its_named_substitutions() {
    let errs = errors("m = load_module(\"nope.flatppl\", x = 1.0)\n");
    assert!(
        !errs.iter().any(|e| e.contains("distinguished input")),
        "a named substitution is not an arity error: {errs:?}"
    );
}

// ---- §09: standard-module member arity and names ----------------------------

/// §09's Parameters column declares a member's arity exactly as §08 does for a
/// base distribution, and §04's binding rule governs the application either way.
/// `user_arity_check` returned early for every module ref before this, so
/// `pp.CrystalBall` typed identically on 0, 1, 4 and 5 arguments.
#[test]
fn a_module_member_is_checked_against_its_catalogue_row() {
    let pp = "pp = standard_module(\"particle-physics\", \"0.1\")\n";
    for (n, call) in [
        (0, "pp.CrystalBall()"),
        (1, "pp.CrystalBall(1.0)"),
        (5, "pp.CrystalBall(1.0, 2.0, 3.0, 4.0, 5.0)"),
    ] {
        let src = format!("{pp}c = {call}\n");
        assert_refuses(&src, "takes 4 arguments (spec §09)");
        let _ = n;
    }
    assert_clean(&format!("{pp}c = pp.CrystalBall(1.0, 2.0, 3.0, 4.0)\n"));
    assert_clean(&format!(
        "{pp}c = pp.CrystalBall(m0 = 1.0, sigma = 2.0, alpha = 3.0, n = 4.0)\n"
    ));
}

/// §04 "Calling conventions": "A call with field or column names that do not
/// match the callable's argument names is a static error." The §09 function rows
/// carried no `names` at all, so `sf.erf(nope = true)` typed the same real scalar
/// as `sf.erf(x)`; the names now come from §09's Arguments column.
#[test]
fn a_module_function_checks_its_argument_names() {
    let sf = "sf = standard_module(\"special-functions\", \"0.1\")\n";
    assert_refuses(
        &format!("{sf}e = sf.erf(nope = true)\n"),
        "has no parameter `nope`",
    );
    assert_clean(&format!("{sf}e = sf.erf(x = 1.0)\n"));
    assert_clean(&format!("{sf}e = sf.erf(1.0)\n"));
    assert_clean(&format!("{sf}b = sf.bessel_j(v = 1.0, z = 2.0)\n"));
}
