//! §03 well-formedness of the aggregate constructors: duplicate field names, a
//! mixed positional-and-keyword set or measure product, and an array literal that
//! mixes a string element with a value element.
//!
//! Each let an ill-formed model type cleanly: `record(a = 1, a = 2)` produced a
//! record with two fields both named `a`, `cartprod(reals, a = integers)` dropped
//! its positional component from the value set, and `[1, "x"]` typed an integer
//! array of integers.
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

// ---- §03: aggregate constructors --------------------------------------------

/// §03 "Records": "Fields are accessed by name, not by position". A repeated name
/// is therefore unreachable. All three constructors kept the duplicate before:
/// the record type carried two `a` fields, and so did the joint's domain.
#[test]
fn a_duplicate_field_name_is_refused() {
    assert_refuses("r = record(a = 1, a = 2)\n", "declares `a` twice");
    assert_refuses("t = table(a = [1], a = [2])\n", "declares `a` twice");
    assert_refuses(
        "j = joint(a = Normal(0, 1), a = Normal(2, 3))\n",
        "declares `a` twice",
    );
    assert_refuses(
        "s = cartprod(a = reals, a = integers)\n",
        "declares `a` twice",
    );
}

/// Distinct names are the working case and must not regress.
#[test]
fn distinct_field_names_still_type() {
    assert_clean("r = record(a = 1, b = 2)\n");
    assert_clean("t = table(a = [1], b = [2])\n");
    assert_clean("s = cartprod(a = reals, b = integers)\n");
}

/// §03 "Cartesian product" defines the positional form ("a set of arrays, not a
/// set of tuples") and the keyword form ("produces a set of records") separately,
/// and gives no meaning to a call writing both. `set_call_valueset_at` took the
/// named branch whenever any named argument existed and dropped every positional
/// one, so `cartprod(reals, a = integers)` had value set `record(a = integers)`.
#[test]
fn a_mixed_cartprod_spelling_is_refused() {
    assert_refuses(
        "s = cartprod(reals, a = integers)\nv = elementof(s)\n",
        "mixes positional and keyword components",
    );
}

/// Both unmixed spellings still type, each to its own member kind.
#[test]
fn unmixed_cartprod_spellings_still_type() {
    assert_clean("s = cartprod(reals, integers)\nv = elementof(s)\n");
    assert_clean("s = cartprod(a = reals, b = integers)\nv = elementof(s)\n");
}

/// §03 "Arrays" admits "scalar values (real, integer, boolean and complex values)
/// or arrays" as elements. A string is neither, so an array holding one beside a
/// value is neither an array of values nor a field-name selector list. All three
/// spellings typed `(%array 1 (2) (%scalar integer))` before — an integer element
/// type over two elements, one of which is not a number.
#[test]
fn an_array_mixing_strings_and_values_is_refused() {
    for src in [
        "a = [1, \"x\"]\n",
        "b = [\"x\", 1]\n",
        "c = [true, \"x\"]\n",
    ] {
        assert_refuses(src, "an array element is a string");
    }
}

/// A HOMOGENEOUS string array is the field-name selector list the spec writes
/// throughout — §06 `disintegrate(["obs"], …)`, §04 `relabel(x, ["x", "y",
/// "z"])`, §07 `get(_, ["a", "c"])`. Refusing it would refuse the spec's own
/// examples, so this is the boundary of the rule above.
#[test]
fn a_homogeneous_string_selector_list_still_types() {
    assert_clean("s = [\"obs\"]\n");
    assert_clean("s = [\"a\", \"c\"]\n");
    assert_clean("r = record(a = 1.0, c = 2.0)\nx = get(r, [\"a\", \"c\"])\n");
}
