//! A `load_module` literal resolves against the file that declares it, so the
//! same spelling in two importers denotes two different files (spec §04 "Path
//! resolution"):
//!
//! > Relative file paths in `load_module(...)` are resolved relative to the
//! > directory of the FlatPPL file containing that `load_module(...)` call
//!
//! The bundle is therefore keyed by resolved identity, and each dependency
//! carries the `(declaring file, literal)` pair that reached it. Keyed by the
//! literal alone, `a/common.flatppl` and `b/common.flatppl` collapse into one
//! entry and every reference reads the second one's type.

use std::sync::Arc;

use flatppl_core::{Module, Type};
use flatppl_infer::{Diagnostic, Level, ModuleBundle, infer_module};

fn parse(src: &str) -> Module {
    flatppl_syntax::parse(src).expect("source parses")
}

/// The `a`/`b` graph of the audit repro: a top model importing two
/// intermediates that each import their own `common.flatppl` under that one
/// spelling. `val` is a real in `a`, a boolean in `b`.
fn colliding_bundle() -> ModuleBundle {
    let mut bundle = ModuleBundle::new();
    bundle.set_root("/ws/top.flatppl");
    bundle.insert_resolved(
        "/ws/top.flatppl",
        "a/model.flatppl",
        "/ws/a/model.flatppl",
        Arc::new(parse("c = load_module(\"common.flatppl\")\nx = c.val\n")),
    );
    bundle.insert_resolved(
        "/ws/top.flatppl",
        "b/model.flatppl",
        "/ws/b/model.flatppl",
        Arc::new(parse("c = load_module(\"common.flatppl\")\nx = c.val\n")),
    );
    bundle.insert_resolved(
        "/ws/a/model.flatppl",
        "common.flatppl",
        "/ws/a/common.flatppl",
        Arc::new(parse("val = 1.5\n")),
    );
    bundle.insert_resolved(
        "/ws/b/model.flatppl",
        "common.flatppl",
        "/ws/b/common.flatppl",
        Arc::new(parse("val = true\n")),
    );
    bundle
}

fn type_of_binding(m: &mut Module, name: &str) -> Type {
    let sym = m.intern(name);
    let bid = m
        .binding_by_name(sym)
        .unwrap_or_else(|| panic!("no `{name}` binding"));
    m.type_of(m.binding(bid).rhs)
        .cloned()
        .unwrap_or_else(|| panic!("`{name}` has no inferred type"))
}

fn errors(diags: &[Diagnostic]) -> Vec<&Diagnostic> {
    diags
        .iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect()
}

#[test]
fn two_importer_local_commons_resolve_to_their_own_file() {
    let bundle = colliding_bundle();
    let mut top = parse(
        "ma = load_module(\"a/model.flatppl\")\n\
         mb = load_module(\"b/model.flatppl\")\n\
         ya = ma.x\n\
         yb = mb.x\n",
    );
    let diags = infer_module(&mut top, &bundle, Level::Shape);
    assert!(
        errors(&diags).is_empty(),
        "the graph is valid; got {:?}",
        errors(&diags)
    );
    assert_eq!(
        type_of_binding(&mut top, "ya"),
        Type::Scalar(flatppl_core::ScalarType::Real),
        "`ya` reads `a/common.flatppl`'s real `val`"
    );
    assert_eq!(
        type_of_binding(&mut top, "yb"),
        Type::Scalar(flatppl_core::ScalarType::Boolean),
        "`yb` reads `b/common.flatppl`'s boolean `val`"
    );
}

#[test]
fn a_literal_keyed_bundle_still_resolves() {
    // A host with no path resolution of its own inserts under the literal.
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(parse("val = 1.5\n")));
    let mut m = parse("h = load_module(\"helpers.flatppl\")\ny = h.val\n");
    let diags = infer_module(&mut m, &bundle, Level::Shape);
    assert!(errors(&diags).is_empty(), "got {:?}", errors(&diags));
    assert_eq!(
        type_of_binding(&mut m, "y"),
        Type::Scalar(flatppl_core::ScalarType::Real)
    );
}

#[test]
fn an_ambiguous_literal_has_no_importer_free_identity() {
    let bundle = colliding_bundle();
    // `a/model.flatppl` is spelled once, so it keeps a literal alias.
    assert!(bundle.get("a/model.flatppl").is_some());
    // `common.flatppl` denotes two files. A lookup with no importer context
    // must refuse rather than return one of them.
    assert!(bundle.get("common.flatppl").is_none());
    assert!(bundle.identity_of("", "common.flatppl").is_none());
    assert_eq!(
        bundle.identity_of("/ws/a/model.flatppl", "common.flatppl"),
        Some("/ws/a/common.flatppl")
    );
    assert_eq!(
        bundle.identity_of("/ws/b/model.flatppl", "common.flatppl"),
        Some("/ws/b/common.flatppl")
    );
}

#[test]
fn two_spellings_of_one_file_share_the_memo() {
    // `./helpers.flatppl` and `helpers.flatppl` resolve to one identity, so the
    // dependency is inferred once and both refs read the same annotated module.
    let mut bundle = ModuleBundle::new();
    bundle.set_root("/ws/top.flatppl");
    let dep = Arc::new(parse("val = 1.5\n"));
    bundle.insert_resolved(
        "/ws/top.flatppl",
        "helpers.flatppl",
        "/ws/helpers.flatppl",
        dep.clone(),
    );
    bundle.insert_resolved(
        "/ws/top.flatppl",
        "./helpers.flatppl",
        "/ws/helpers.flatppl",
        dep,
    );
    let mut m = parse(
        "h1 = load_module(\"helpers.flatppl\")\n\
         h2 = load_module(\"./helpers.flatppl\")\n\
         y1 = h1.val\n\
         y2 = h2.val\n",
    );
    let diags = infer_module(&mut m, &bundle, Level::Shape);
    assert!(errors(&diags).is_empty(), "got {:?}", errors(&diags));
    assert_eq!(type_of_binding(&mut m, "y1"), type_of_binding(&mut m, "y2"));
}
