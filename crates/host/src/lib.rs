//! Immutable source contexts and typed StableHLO query exports for native hosts.
//!
//! A [`Context`] loads files or inline source and registers modules under explicit
//! import names. Loaded modules retain their source and import closure. Create a
//! new context to replace a registered definition or reload changed files.
//!
//! [`LoadedModule::compile`] lowers explicit FlatPPL `inputs` and `outputs` to
//! StableHLO plus a tensor ABI. The host chooses execution and differentiation.

mod compiler;
mod constants;
mod data;
mod export;
mod formula;
mod inspect;
mod transforms;

pub use compiler::{BindingInfo, Context, Diagnostic, LoadedModule, MathFormat, MathRow, Position};
pub use constants::Constant;
pub use data::{load_fixed_data, resolve_data_sources};
pub use export::{Export, Field, Schema};
pub use flatppl_formula;
pub use flatppl_hs3::{Auxiliary, Observations, PyhfOptions};
pub use flatppl_stablehlo::{BatchSpec, Dtype, EmitOptions, IntegrationOptions, Target};
pub use formula::FormulaModel;
pub use inspect::{BindingKind, Bound, Domain, DomainField};
pub use transforms::TransformedQueries;
