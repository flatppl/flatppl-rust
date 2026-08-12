//! The built-in name set, for the `shadows-builtin` rule.
//!
//! The roster itself lives in `flatppl_infer::builtins` — spec-§04 name
//! resolution needs it there, and one list cannot drift from itself. Still kept
//! in sync with `flatppl-grammars/keyword-lists.json` by
//! `crates/lint/tests/builtins_sync.rs`.

pub(crate) use flatppl_infer::builtins::BUILTINS;
