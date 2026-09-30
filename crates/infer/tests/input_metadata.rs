use std::sync::Arc;

use flatppl_core::Module;
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module_with_inputs};

fn parse(source: &str) -> Module {
    flatppl_syntax::parse(source).unwrap()
}

fn bundle(source: &str) -> ModuleBundle {
    let mut bundle = ModuleBundle::new();
    bundle.set_root("query.flatppl");
    bundle.insert_resolved(
        "query.flatppl",
        "dep.flatppl",
        "models/dep.flatppl",
        Arc::new(parse(source)),
    );
    bundle
}

fn inputs(bundle: &ModuleBundle, name: &str) -> Option<Vec<(String, String)>> {
    let identity = bundle.identity_of(bundle.root(), "dep.flatppl")?;
    let source = bundle.get_by_id(identity)?;
    let node = source
        .bindings()
        .find(|(_, b)| source.resolve(b.name) == name)?
        .1
        .rhs;
    bundle.auto_inputs_of(identity, node).map(|entries| {
        entries
            .iter()
            .map(|(label, target)| {
                (
                    source.resolve(*label).to_string(),
                    source.resolve(target.name).to_string(),
                )
            })
            .collect()
    })
}

#[test]
fn input_metadata_tracks_the_bundle_snapshot() {
    let source = "a = elementof(reals)\nb = elementof(reals)\nf = functionof(a)\nk = functionof(Normal(0, 1))";
    let query = "h = load_module(\"dep.flatppl\")\nf = h.f\nk = h.k";
    let (diagnostics, prepared) =
        infer_module_with_inputs(&mut parse(query), bundle(source), Level::Shape);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "{diagnostics:?}"
    );
    assert_eq!(inputs(&prepared, "f"), Some(vec![("a".into(), "a".into())]));
    assert_eq!(inputs(&prepared, "k"), Some(vec![]));

    let mut edited = prepared.clone();
    edited.insert_resolved(
        "query.flatppl",
        "dep.flatppl",
        "models/dep.flatppl",
        Arc::new(parse(&source.replace("functionof(a)", "functionof(b)"))),
    );
    assert_eq!(inputs(&edited, "f"), None);
    assert_eq!(inputs(&prepared, "f"), Some(vec![("a".into(), "a".into())]));
    let (diagnostics, edited) = infer_module_with_inputs(&mut parse(query), edited, Level::Shape);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "{diagnostics:?}"
    );
    assert_eq!(inputs(&edited, "f"), Some(vec![("b".into(), "b".into())]));

    let mut rerooted = prepared.clone();
    rerooted.set_root("another-query.flatppl");
    assert_eq!(inputs(&rerooted, "f"), None);
    assert_eq!(inputs(&prepared, "f"), Some(vec![("a".into(), "a".into())]));
}

#[test]
fn incomplete_inference_keeps_the_input_fallback() {
    let source = "a = elementof(reals)\nf = functionof(a)";
    let query = "h = load_module(\"dep.flatppl\")\nf = h.f";
    let (_, prepared) = infer_module_with_inputs(&mut parse(query), bundle(source), Level::Type);
    assert_eq!(inputs(&prepared, "f"), Some(vec![("a".into(), "a".into())]));
    for (level, suffix, failed) in [
        (Level::Phase, "", false),
        (Level::Type, "\nbad = missing", true),
    ] {
        let (diagnostics, prepared) = infer_module_with_inputs(
            &mut parse(&format!("{query}{suffix}")),
            prepared.clone(),
            level,
        );
        assert_eq!(
            diagnostics.iter().any(|d| d.severity == Severity::Error),
            failed
        );
        assert_eq!(inputs(&prepared, "f"), None);
    }
}

#[test]
fn substituted_sources_keep_the_input_fallback() {
    let source = "a = elementof(reals)\nf = functionof(a)";
    let query = "x = elementof(integers)\nh = load_module(\"dep.flatppl\", a = x)\nf = h.f";
    let (diagnostics, prepared) =
        infer_module_with_inputs(&mut parse(query), bundle(source), Level::Shape);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "{diagnostics:?}"
    );
    assert_eq!(inputs(&prepared, "f"), None);
}
