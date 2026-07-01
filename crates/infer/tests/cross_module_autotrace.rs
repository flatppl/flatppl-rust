//! Reification is module-local (spec §04).
//!
//! `functionof`/`kernelof` reify within the current module only: a parameterized
//! value reached through a loaded-module reference cannot become an input —
//! neither auto-collected nor named as an explicit boundary — so such a
//! reification is a static error. A loaded module's callables and values are
//! still *used* normally (applied or referenced); only taking one as a reified
//! input is disallowed. Single-module auto-trace is covered in `reification.rs`.

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

fn refused(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.message.contains("module-local"))
}

/// A boundary-less reification whose trace reaches a cross-module parameterized
/// value is a static error (`%failed` + a diagnostic) — never a silent
/// mis-reification.
#[test]
fn boundary_less_cross_module_reification_is_refused() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "theta = elementof(reals)\nscaled = mul(theta, 2.0)",
        "dep = load_module(\"dep.flatppl\")\nf = functionof(dep.scaled)",
    );
    assert!(refused(&diags), "should refuse, got: {diags:?}");
    let f = out.lines().find(|l| l.contains("%bind f")).unwrap_or("");
    assert!(f.contains("%failed"), "f should be %failed, got:\n{out}");
}

/// Reification never crosses a module boundary — not even with an EXPLICIT
/// boundary. Naming a loaded-module binding as a boundary node is a static error.
#[test]
fn explicit_boundary_onto_cross_module_node_is_refused() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "theta = elementof(reals)\nscaled = mul(theta, 2.0)",
        "dep = load_module(\"dep.flatppl\")\n\
         f = functionof(add(dep.scaled, 1.0), k = dep.scaled)",
    );
    assert!(
        refused(&diags),
        "explicit cross-module boundary should refuse, got: {diags:?}"
    );
    let f = out.lines().find(|l| l.contains("%bind f")).unwrap_or("");
    assert!(f.contains("%failed"), "f should be %failed, got:\n{out}");
}

/// A reification may USE a cross-module callable as a black box: `functionof`
/// over `dep.f(a)` reifies a function of the LOCAL input `a` (`dep.f` is closed
/// over) — no refusal.
#[test]
fn reification_may_use_a_cross_module_callable() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "f = x -> x * 2.0",
        "dep = load_module(\"dep.flatppl\")\na = elementof(reals)\ng = functionof(dep.f(a))",
    );
    assert!(
        !refused(&diags),
        "using a cross-module callable must NOT refuse, got: {diags:?}"
    );
    let g = out.lines().find(|l| l.contains("%bind g")).unwrap_or("");
    assert!(
        g.contains("(%function (%inputs a))"),
        "g should reify a function of the local input `a`, got:\n{out}"
    );
}

/// `lawof` only *references* — it produces a measure, not a fixed input list — so
/// a measure over a cross-module value is fine (no refusal); the module-local
/// rule constrains input collection, which `lawof` does not do.
#[test]
fn lawof_may_reference_a_cross_module_value() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "scaled = elementof(reals)",
        "dep = load_module(\"dep.flatppl\")\nz ~ Normal(dep.scaled, 1.0)\nm = lawof(z)",
    );
    assert!(
        !refused(&diags),
        "lawof over a cross-module value must NOT refuse, got: {diags:?}"
    );
    let m = out.lines().find(|l| l.contains("%bind m")).unwrap_or("");
    assert!(
        m.contains("(%measure"),
        "m should be a measure, got:\n{out}"
    );
}

/// The spec's explicit consequence (§04): a measure that carries a parametric
/// dependency on another module's node is a legal *measure* (`lawof` above), but
/// it "cannot then be reified to a kernel" — `functionof` over such a measure
/// would auto-collect the cross-module parameterized node as an input, so it is
/// refused, exactly like a direct value reification.
#[test]
fn a_measure_with_a_cross_module_parametric_dep_cannot_be_reified_to_a_kernel() {
    let (out, diags) = xmod(
        "dep.flatppl",
        "scaled = elementof(reals)",
        "dep = load_module(\"dep.flatppl\")\n\
         z ~ Normal(dep.scaled, 1.0)\n\
         m = lawof(z)\n\
         k = functionof(m)",
    );
    assert!(
        refused(&diags),
        "reifying a measure with a cross-module parametric dep to a kernel should refuse, got: {diags:?}"
    );
    let k = out.lines().find(|l| l.contains("%bind k")).unwrap_or("");
    assert!(k.contains("%failed"), "k should be %failed, got:\n{out}");
}
