//! `flatppl-mathdoc` — FlatPPL rendered as mathematics.
//!
//! Builds a target-independent math AST ([`ast`]) from a typed and phased
//! [`flatppl_core::Module`] and prints it as MathML, TeX, or Typst. The notation — which
//! FlatPPL construct becomes which mathematical form, how names render, where
//! the measure/density distinction shows — is specified in `NOTATION.md`
//! (reproduced below) and is the contract all printers implement.
//!
//! Binary-free: the CLI (`flatppl convert model.flatppl model.html`) and the
//! wasm API (`render_math`) are thin adapters in their own crates.
#![doc = include_str!("../NOTATION.md")]

pub mod ast;
#[cfg(feature = "document")]
pub mod document;
#[cfg(feature = "document")]
pub mod export;
#[cfg(feature = "json")]
pub mod json;
pub mod lower;
pub mod mathml;
pub mod names;
pub mod notation;
pub mod render;
pub mod tex;
pub mod typst;

pub use render::{BindingRender, Diagnostic, Rendering, render, render_source, render_with_source};
