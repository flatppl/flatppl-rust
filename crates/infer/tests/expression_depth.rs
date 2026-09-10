//! Published pyhf models exceed the former 128-level inference budget.

use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

#[test]
fn expression_beyond_the_former_budget_infers() {
    let terms = std::iter::repeat_n("1.0", 160)
        .collect::<Vec<_>>()
        .join(" + ");
    let mut module = flatppl_syntax::parse(&format!("value = {terms}\n")).unwrap();
    let diags = infer_module(&mut module, &ModuleBundle::new(), Level::Type);
    assert!(
        !diags.iter().any(|d| d.severity == Severity::Error),
        "{diags:?}"
    );
}
