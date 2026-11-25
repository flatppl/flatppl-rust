//! Typed rewrites of an inferred FlatPPL model or a determinized query.

mod boundary;
mod categorical_counts;
mod concat;
mod conjugacy;
mod demand;
mod engine;
mod fixed_poisson;
mod gaussian;
mod lowered;
mod matrix_projection;
mod ordered_scan;
mod outputs;
mod policy;
mod project;
mod selectors;
mod types;

pub use categorical_counts::{SimplexRowCheck, plan_simplex_rows, plan_simplex_rows_with_rules};
pub use demand::{ScanOutputDemand, scan_output_demand, scan_output_demand_with_rules};
use flatppl_core::{Module, Symbol};
pub use matrix_projection::{
    MatrixProjection, MatrixProjectionFact, plan_matrix_projection,
    plan_matrix_projection_with_rules,
};
pub use ordered_scan::{
    ordered_scan_width, ordered_scan_width_with_rules, plan_ordered_scan_index,
    plan_ordered_scan_index_with_rules,
};
pub use policy::{
    RuleInfo, RulePolicy, RuleSelector, UnknownRuleSelector, available_passes, available_rules,
    available_rulesets, available_stages,
};

/// The shared language and each emitter have separate rule schedules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Portable,
    Stan,
    StableHlo { target: Target, dtype: Dtype },
}

/// Where in the pipeline a profile's rules run: on the source measure graph,
/// or on the module that lowering produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Source,
    Lowered,
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

/// Optimise a self-contained module with a caller-selected rule policy.
pub fn optimise_with_rules(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    rules: &RulePolicy,
) -> Option<Module> {
    optimise_with_bundle_and_rules(
        module,
        roots,
        profile,
        &flatppl_infer::ModuleBundle::new(),
        rules,
    )
}

/// Optimise a module with the same dependency resolution used for inference.
pub fn optimise_with_bundle(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
) -> Option<Module> {
    optimise_with_bundle_and_rules(module, roots, profile, bundle, &RulePolicy::default())
}

/// Optimise a module with its dependency bundle and a caller-selected rule policy.
pub fn optimise_with_bundle_and_rules(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
    rules: &RulePolicy,
) -> Option<Module> {
    optimise_query(module, roots, profile, bundle, true, rules)
}

/// Apply the portable rules of the lowered stage to a determinized module.
/// The run starts only when a loaded lowered ruleset can match a term.
pub fn optimise_lowered(
    module: &Module,
    roots: Option<&[Symbol]>,
    rules: &RulePolicy,
) -> Option<Module> {
    if rules.disables_stage(Stage::Lowered) || !lowered::trigger(module, rules) {
        return None;
    }
    lowered_query(module, roots, rules)
}

/// The lowered stage without its trigger, for measuring the run itself.
#[doc(hidden)]
pub fn optimise_lowered_forced(
    module: &Module,
    roots: Option<&[Symbol]>,
    rules: &RulePolicy,
) -> Option<Module> {
    lowered_query(module, roots, rules)
}

/// Project a determinized module as the lowered stage does and rebuild every
/// projected call from its operands. Returns the rebuilt module and the
/// number of calls rebuilt; a faithful projection prints the input again.
#[doc(hidden)]
pub fn lowered_roundtrip(module: &Module, roots: Option<&[Symbol]>) -> Option<(Module, usize)> {
    project::roundtrip(module, roots)
}

// A lowered module is already reduced to the requested roots, so the demand
// prune of the source stage does not run.
fn lowered_query(module: &Module, roots: Option<&[Symbol]>, rules: &RulePolicy) -> Option<Module> {
    project::optimise(
        module,
        roots,
        Profile::Portable,
        Stage::Lowered,
        &flatppl_infer::ModuleBundle::new(),
        false,
        rules,
    )
}

/// Apply the StableHLO rules with the emitter's likelihood-reduction setting.
pub fn optimise_stablehlo(
    module: &Module,
    roots: Option<&[Symbol]>,
    target: Target,
    dtype: Dtype,
    sufficient_statistics: bool,
) -> Option<Module> {
    optimise_stablehlo_with_rules(
        module,
        roots,
        target,
        dtype,
        sufficient_statistics,
        &RulePolicy::default(),
    )
}

/// Optimise a StableHLO query with its likelihood-reduction setting and rule policy.
pub fn optimise_stablehlo_with_rules(
    module: &Module,
    roots: Option<&[Symbol]>,
    target: Target,
    dtype: Dtype,
    sufficient_statistics: bool,
    rules: &RulePolicy,
) -> Option<Module> {
    optimise_query(
        module,
        roots,
        Profile::StableHlo { target, dtype },
        &flatppl_infer::ModuleBundle::new(),
        sufficient_statistics,
        rules,
    )
}

fn optimise_query(
    module: &Module,
    roots: Option<&[Symbol]>,
    profile: Profile,
    bundle: &flatppl_infer::ModuleBundle,
    sufficient_statistics: bool,
    rules: &RulePolicy,
) -> Option<Module> {
    let stage = match profile {
        Profile::StableHlo { .. } => Stage::Lowered,
        Profile::Portable | Profile::Stan => Stage::Source,
    };
    if rules.disables_stage(stage) {
        return None;
    }
    let pruned = if rules.disables_pass("demand-prune") {
        None
    } else {
        demand::prune(module, roots, profile, bundle)
    };
    project::optimise(
        pruned.as_ref().unwrap_or(module),
        roots,
        profile,
        stage,
        bundle,
        sufficient_statistics,
        rules,
    )
    .or(pruned)
}
