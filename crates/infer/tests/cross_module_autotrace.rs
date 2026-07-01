//! Cross-module boundary-less reification auto-trace (spec §04 + §11).
//!
//! Under **per-module inference** (§11: *"Each module is annotated independently
//! … from its own perspective"*) the loading module B never traces into a
//! dependency — it sees only the dependency's public interface. So a parameterized
//! cross-module reference (`dep.scaled`) is an **atomic parametric leaf** from B's
//! view — the cross-module analogue of a local `elementof`. A boundary-less
//! `functionof` therefore records such a reference as one of its inputs (a fixed
//! cross-module ref is closed over). The module boundary is a reification boundary:
//! B does NOT reach `dep`'s internal `elementof` leaves.

use std::sync::Arc;

use flatppl_infer::{Level, ModuleBundle, infer_module};

fn ir_xmod(dep_path: &str, dep_src: &str, model_src: &str) -> String {
    let dep = flatppl_syntax::parse(dep_src).expect("dependency parses");
    let mut bundle = ModuleBundle::new();
    bundle.insert(dep_path, Arc::new(dep));
    let mut model = flatppl_syntax::parse(model_src).expect("model parses");
    let _ = infer_module(&mut model, &bundle, Level::Shape);
    flatppl_flatpir::write(&model)
}

/// A boundary-less `functionof` over a parameterized cross-module value records
/// that reference as an atomic input (per-module: the module boundary stops the
/// trace, `dep.scaled` is a leaf named `scaled`). This is NOT the old empty-input
/// (invalid nullary) result.
#[test]
fn cross_module_param_ref_is_an_atomic_auto_input() {
    let out = ir_xmod(
        "dep.flatppl",
        "theta = elementof(reals)\nscaled = mul(theta, 2.0)",
        "dep = load_module(\"dep.flatppl\")\nf = functionof(dep.scaled)",
    );
    let f = out.lines().find(|l| l.contains("%bind f")).unwrap_or("");
    assert!(
        f.contains("(%function (%inputs scaled))") && f.contains("(scaled (%ref dep scaled))"),
        "functionof(dep.scaled) should have the cross-module ref as its sole atomic input `scaled`, got:\n{out}"
    );
}

/// A cross-module parametric leaf combines with LOCAL `elementof` leaves:
/// `functionof(add(dep.scaled, a))` (local `a = elementof`) reifies both — inputs
/// `a` and `scaled`, in canonical (name-sorted) order.
#[test]
fn cross_module_leaf_combines_with_local_leaves() {
    let out = ir_xmod(
        "dep.flatppl",
        "theta = elementof(reals)\nscaled = mul(theta, 2.0)",
        "a = elementof(reals)\n\
         dep = load_module(\"dep.flatppl\")\n\
         f = functionof(add(dep.scaled, a))",
    );
    let f = out.lines().find(|l| l.contains("%bind f")).unwrap_or("");
    assert!(
        f.contains("(%function (%inputs a scaled))")
            && f.contains("(a (%ref self a))")
            && f.contains("(scaled (%ref dep scaled))"),
        "functionof(add(dep.scaled, a)) should reify local `a` + cross-module `scaled`, got:\n{out}"
    );
}
