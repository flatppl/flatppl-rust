//! `flatppl-syntax` — canonical FlatPPL surface syntax (spec §05).
//!
//! Parses surface FlatPPL into the [`flatppl_core`] model (stripping syntactic
//! sugar) and pretty-prints a module back to canonical FlatPPL. FlatPIR
//! S-expression I/O lives in `flatppl-flatpir`.
//!
//! **Scope.** Lowering here is *syntactic only* — operators → `add`/…,
//! indexing/field → `get`, `~` → `draw`, `[…]` → `vector`, etc., with
//! built-in-vs-user call resolution by a binding-name pre-pass. Reification
//! (`functionof` / `kernelof` / `lawof`) is kept as an *un-traced construct
//! call*; the ancestor-slice reification is a Phase-2 semantic pass. The
//! printer re-applies the sugar ([`Syntax::Full`], the default) or emits the
//! spec §04 lowered linear form ([`Syntax::Minimal`]); see [`print_with`].

mod error;
mod parser;
mod printer;
mod token;

pub use error::{Error, Result};
pub use parser::parse;
pub use printer::{Syntax, print, print_with};
