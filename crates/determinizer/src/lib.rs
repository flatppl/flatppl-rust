//! FlatPPL → FlatPDL determiniser: a greedy directional legalizer that eliminates
//! the measure layer, leaving deterministic ops + the six `builtin_*` primitives.
//! Type-level transform — flatppl-rust does not evaluate densities.
mod conformance;
mod crossmodule;
mod density;
mod disintegrate;
mod driver;
mod invert;
mod jointchain;
mod kernel;
mod marginal;
mod refuse;
mod sample;
pub use conformance::is_flatpdl;
pub use disintegrate::split_disintegrate;
pub use driver::{determinize, determinize_with};
pub use refuse::{NonConformKind, NonConformance, RefuseError};
