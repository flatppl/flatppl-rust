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
mod export;

pub use compiler::{BindingInfo, Context, Diagnostic, LoadedModule};
pub use constants::Constant;
pub use export::{Export, Field, Schema};
pub use flatppl_stablehlo::{Dtype, EmitOptions};
