//! FlatPPL → FlatPDL determiniser: a greedy directional legalizer that eliminates
//! the measure layer, leaving deterministic ops + the six `builtin_*` primitives.
//! Type-level transform — flatppl-rust does not evaluate densities.
mod conformance;
mod density;
mod driver;
mod marginal;
mod refuse;
mod sample;
pub use conformance::is_flatpdl;
pub use driver::determinize;
pub use refuse::{NonConformKind, NonConformance, RefuseError};
