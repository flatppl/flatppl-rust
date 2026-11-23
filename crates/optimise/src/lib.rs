//! Typed rewrites of an inferred FlatPPL model or a determinized query.

mod concat;
mod conjugacy;
mod demand;
mod engine;
mod fixed_poisson;
mod gaussian;
mod matrix_projection;
mod ordered_scan;
mod outputs;
mod project;
mod selectors;
mod types;

pub use demand::{ScanOutputDemand, scan_output_demand};
use flatppl_core::{Module, Symbol};
pub use matrix_projection::{MatrixProjection, MatrixProjectionFact, plan_matrix_projection};
pub use ordered_scan::ordered_scan_width;

/// The shared language and each emitter have separate rule schedules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Portable,
    Stan,
    StableHlo { target: Target, dtype: Dtype },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F64,
}

/// Optimise reachable definitions, retaining binding names, query inputs and
/// callable boundaries. With no roots, public bindings are observable and all
/// definitions remain in the module.
/// Returns a freshly annotated module only when a cheaper expression wins.
pub fn optimise(module: &Module, roots: Option<&[Symbol]>, profile: Profile) -> Option<Module> {
    optimise_with_bundle(module, roots, profile, &flatppl_infer::ModuleBundle::new())
}

/// Optimise a module with the same dependency resolution used for inference.
pub fn optimise_with_bundle(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
) -> Option<Module> {
    optimise_query(module, roots, profile, bundle, true)
}

/// Apply the StableHLO rules with the emitter's likelihood-reduction setting.
pub fn optimise_stablehlo(
    module: &Module,
    roots: Option<&[Symbol]>,
    target: Target,
    dtype: Dtype,
    sufficient_statistics: bool,
) -> Option<Module> {
    optimise_query(
        module,
        roots,
        Profile::StableHlo { target, dtype },
        &flatppl_infer::ModuleBundle::new(),
        sufficient_statistics,
    )
}

fn optimise_query(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
    sufficient_statistics: bool,
) -> Option<Module> {
    let pruned = demand::prune(module, roots, profile, bundle);
    project::optimise(
        pruned.as_ref().unwrap_or(module),
        roots,
        profile,
        bundle,
        sufficient_statistics,
    )
    .or(pruned)
}
