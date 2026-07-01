//! Cross-module reification (spec §04).
//!
//! §04 reifies the **combined DAG**: a boundary-less `functionof` traces across
//! loaded-module boundaries to the dependency's own `elementof` leaves, so
//! splitting a model across files does not change what it reifies. That
//! trace-across is realized by the module linker (engine-concepts §7) and is not
//! yet implemented here — so the engine **refuses** the boundary-less
//! cross-module case (loud, not a silent mis-reification) and directs to the
//! explicit boundary form, which is also the encapsulation opt-out for reusable
//! dependencies. Single-module auto-trace is covered in `reification.rs`.

use std::sync::Arc;

use flatppl_infer::{Diagnostic, Level, ModuleBundle, infer_module};

fn xmod(dep_path: &str, dep_src: &str, model_src: &str) -> (String, Vec<Diagnostic>) {
    let dep = flatppl_syntax::parse(dep_src).expect("dependency parses");
    let mut bundle = ModuleBundle::new();
    bundle.insert(dep_path, Arc::new(dep));
    let mut model = flatppl_syntax::parse(model_src).expect("model parses");
    let diags = infer_module(&mut model, &bundle, Level::Shape);
    (flatppl_flatpir::write(&model), diags)
}

/// Boundary-less reification whose body depends on a cross-module value is
/// refused — a loud diagnostic + a `%failed` reification, not the old silent
/// atomic-leaf mis-reification.
#[test]
fn boundary_less_cross_module_reification_is_refused() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "theta = elementof(reals)\nscaled = mul(theta, 2.0)",
        "dep = load_module(\"dep.flatppl\")\nf = functionof(dep.scaled)",
    );
    assert!(
        diags
            .iter()
            .any(|d| d.message.contains("loaded module") && d.message.contains("boundary")),
        "should refuse with a cross-module diagnostic, got: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let f = out.lines().find(|l| l.contains("%bind f")).unwrap_or("");
    assert!(f.contains("%failed"), "f should be %failed, got:\n{out}");
}

/// The explicit boundary form is the cross-module / encapsulation opt-out: naming
/// the boundary at `dep.scaled` reifies a function with the chosen input name
/// (`k`), decoupled from the dependency's internals — no refusal.
#[test]
fn explicit_boundary_reifies_across_a_module() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "theta = elementof(reals)\nscaled = mul(theta, 2.0)",
        "dep = load_module(\"dep.flatppl\")\n\
         f = functionof(add(dep.scaled, 1.0), k = dep.scaled)",
    );
    assert!(
        !diags.iter().any(|d| d.message.contains("loaded module")),
        "explicit boundary must NOT be refused, got: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let f = out.lines().find(|l| l.contains("%bind f")).unwrap_or("");
    assert!(
        f.contains("(%function (%inputs k))"),
        "explicit boundary should reify a function of `k`, got:\n{out}"
    );
}
