//! A loaded module must not reset the active inference depth budget.

use std::sync::Arc;

use flatppl_core::depth::DEFAULT_MAX_DEPTH;
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

#[test]
fn deep_module_references_return_a_resource_error() {
    let mut bundle = ModuleBundle::new();
    for i in 0..=DEFAULT_MAX_DEPTH {
        let source = if i == DEFAULT_MAX_DEPTH {
            "value = 1\n".to_string()
        } else {
            format!(
                "dep = load_module(\"m{}.flatppl\")\nvalue = dep.value\n",
                i + 1
            )
        };
        bundle.insert(
            format!("m{i}.flatppl"),
            Arc::new(flatppl_syntax::parse(&source).unwrap()),
        );
    }
    let mut root =
        flatppl_syntax::parse("dep = load_module(\"m0.flatppl\")\nvalue = dep.value\n").unwrap();
    let diags = infer_module(&mut root, &bundle, Level::Type);
    assert!(
        diags.iter().any(|d| d.severity == Severity::Error
            && d.message.contains("inference graph")
            && d.message.contains("resource guard")),
        "{diags:?}"
    );
}
