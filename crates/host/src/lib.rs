//! Immutable source contexts and typed StableHLO query exports for native hosts.
//!
//! A [`Context`] loads files or inline source and registers modules under explicit
//! import names. Loaded modules retain their source and import closure. Create a
//! new context to replace a registered definition or reload changed files.
//!
//! [`LoadedModule::compile`] lowers explicit FlatPPL `inputs` and `outputs` to
//! StableHLO plus a tensor ABI. The host chooses execution and differentiation.

mod analysis;
mod compiler;
mod constants;
mod data;
mod export;
mod formula;
mod incremental;
mod inspect;
mod normal_coordinates;
mod posterior;
mod readers;
mod transforms;

pub use analysis::QueryAnalysis;
pub use compiler::{
    BindingInfo, Context, Diagnostic, LoadedModule, MathFormat, MathRow, Position,
    PreparedLikelihood,
};
pub use constants::Constant;
pub use data::{DataError, load_fixed_data, resolve_data_sources};
pub use export::{Export, Field, RuntimeStatus, Schema};
pub use flatppl_determinizer::PreparationOptions;
pub use flatppl_determinizer::{
    AnalysisError, InputArray, InputCell, InputPath, ModelAnalysis, NormalCoordinate, OutputTerm,
    ReductionOutcome,
};
pub use flatppl_formula;
pub use flatppl_hs3::{Auxiliary, Observations, PyhfOptions};
pub use flatppl_stablehlo::{BatchSpec, Dtype, EmitOptions, IntegrationOptions, Target};
pub use formula::{FormulaModel, FormulaTableSource, PreparedFormulaPrediction};
pub use incremental::{IncrementalScoreQueries, IncrementalTermQuery, ScoreRevisions, ScoreTicket};
pub use inspect::{BindingKind, Bound, Domain, DomainField, InputInfo};
pub use normal_coordinates::{
    NormalCoordinateExports, PreparedNormalCoordinate, PreparedNormalCoordinates,
};
pub use posterior::{PosteriorSampling, PosteriorUpdate};
pub use transforms::TransformedQueries;
