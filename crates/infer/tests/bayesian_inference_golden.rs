//! Integration tests over the `bayesian_inference_{1..4}` example family
//! (copied from `flatppl-examples`). The four variants express the *same*
//! Bayesian model four ways:
//!
//! - **1** — domain declarations (`elementof`) + explicit `joint` prior;
//! - **2** — stochastic `~` draws + explicit `lawof` / `kernelof` reification;
//! - **3** — joint law + structural `disintegrate(["obs"], …)`;
//! - **4** — joint law + `restrict(joint_model, record(obs = …))`.
//!
//! Variants 3 and 4 additionally exercise **two-level nested module loading**:
//! the model loads `bayesian_inference_common.flatppl` (with a `c` substitution),
//! which itself loads `bayesian_inference_priors.flatppl`. This pins that the
//! cross-module bundle resolves transitively and that members reached across two
//! module hops type concretely.

use std::path::PathBuf;
use std::sync::Arc;

use flatppl_core::{Module, Type};
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

fn read_fixture(name: &str) -> String {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "../../fixtures/flatppl/bayesian_inference",
        name,
    ]
    .iter()
    .collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading fixture {name}: {e}"))
}

fn parse_fixture(name: &str) -> Module {
    flatppl_syntax::parse(&read_fixture(name)).unwrap_or_else(|e| panic!("{name} parses: {e}"))
}

/// The inferred type of the RHS of the top-level binding named `name`.
fn ty_of<'m>(module: &'m Module, name: &str) -> Option<&'m Type> {
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)?
        .1
        .rhs;
    module.type_of(rhs)
}

/// Diagnostics that signal a *cross-module resolution* failure specifically
/// (as opposed to an as-yet-unimplemented type rule, which surfaces as a
/// `Note`/`%deferred`, not an `Error`).
fn resolution_errors(diags: &[flatppl_infer::Diagnostic]) -> Vec<&flatppl_infer::Diagnostic> {
    diags
        .iter()
        .filter(|d| {
            d.severity == Severity::Error
                && (d.message.contains("not found")
                    || d.message.contains("has no binding")
                    || d.message.contains("has no input")
                    || d.message.contains("cycle")
                    || d.message.contains("private")
                    || d.message.contains("deferred"))
        })
        .collect()
}

/// Variants 1 and 2 are self-contained (no `load_module`): they must infer with
/// no errors, and the analysis pipeline (`likelihoodof` → `bayesupdate`) must
/// produce a concrete `Likelihood` for `L` and a `Measure` posterior.
#[test]
fn single_file_variants_infer_clean() {
    for name in [
        "bayesian_inference_1.flatppl",
        "bayesian_inference_2.flatppl",
    ] {
        let mut module = parse_fixture(name);
        let diags = infer_module(&mut module, &ModuleBundle::new(), Level::Shape);

        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "{name}: unexpected errors: {errors:?}");

        assert!(
            matches!(ty_of(&module, "L"), Some(Type::Likelihood { .. })),
            "{name}: L should be a Likelihood; got {:?}",
            ty_of(&module, "L")
        );
        assert!(
            matches!(ty_of(&module, "posterior"), Some(Type::Measure { .. })),
            "{name}: posterior should be a Measure; got {:?}",
            ty_of(&module, "posterior")
        );
    }
}

/// Variants 3 and 4 load a common module that itself loads a priors module.
/// With both transitive dependencies in the bundle, inference must resolve the
/// nested chain with **no resolution errors**, and the members reached across
/// the two hops must type concretely (not `Deferred`/`Failed`).
fn nested_bundle() -> ModuleBundle {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "bayesian_inference_common.flatppl",
        Arc::new(parse_fixture("bayesian_inference_common.flatppl")),
    );
    bundle.insert(
        "bayesian_inference_priors.flatppl",
        Arc::new(parse_fixture("bayesian_inference_priors.flatppl")),
    );
    bundle
}

#[test]
fn nested_module_loading_resolves_transitively() {
    for name in [
        "bayesian_inference_3.flatppl",
        "bayesian_inference_4.flatppl",
    ] {
        let mut module = parse_fixture(name);
        let diags = infer_module(&mut module, &nested_bundle(), Level::Shape);

        let res_errs = resolution_errors(&diags);
        assert!(
            res_errs.is_empty(),
            "{name}: nested module resolution must be clean; got {res_errs:?}"
        );

        // `a = common.f_a(theta2)` — a cross-module function application whose
        // result must be a concrete scalar, proving the two-hop member ref
        // (model → common → priors) typed end to end.
        assert!(
            matches!(ty_of(&module, "a"), Some(Type::Scalar(_))),
            "{name}: `a = common.f_a(theta2)` must type to a scalar; got {:?}",
            ty_of(&module, "a")
        );
        // `posterior` is the analysis endpoint of both variants (bayesupdate in
        // 3, restrict in 4) — a Measure either way.
        assert!(
            matches!(ty_of(&module, "posterior"), Some(Type::Measure { .. })),
            "{name}: posterior should be a Measure; got {:?}",
            ty_of(&module, "posterior")
        );
    }
}
