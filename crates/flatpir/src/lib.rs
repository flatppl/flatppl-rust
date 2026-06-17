//! `flatppl-flatpir` — FlatPIR, the canonical S-expression intermediate
//! representation (spec §11).
//!
//! Reads and writes the [`flatppl_core`] in-memory model from/to FlatPIR's
//! canonical S-expression syntax. Surface FlatPPL I/O lives in `flatppl-syntax`.
//!
//! ```
//! let src = "(%module (%public y) (%bind x 1) (%bind y (add (%ref self x) 2.0)))";
//! let module = flatppl_flatpir::read(src).unwrap();
//! let back = flatppl_flatpir::write(&module);
//! // Canonical, not byte-preserving — but stable under re-reading:
//! assert_eq!(back, flatppl_flatpir::write(&flatppl_flatpir::read(&back).unwrap()));
//! ```
//!
//! **Round-trip contract.** `core → FlatPIR → core` is structure-preserving;
//! `FlatPIR → core → FlatPIR` reaches a canonical fixpoint after the first
//! write (whitespace/ordering normalized, semantics preserved). Annotations
//! (the `%meta` type / phase / value-set wrapper) live in side-tables and are
//! emitted only when present.

mod error;
mod json;
mod reader;
mod writer;

pub mod sexpr;

pub use error::{Error, Result};
pub use json::{from_json, to_json, try_to_json};
pub use reader::read;
pub use writer::write;
