//! The FlatPPL language server library.
//!
//! **Crate role:** `flatppl-lsp` is a binary tool crate — a peer of
//! `flatppl-cli` — and is NOT part of the wasm32-linkable library set. Its
//! `[lib]` target exists solely to let integration tests drive the server
//! in-process without spawning a subprocess.
//!
//! Houses the salsa database that backs incremental analysis. Tasks downstream
//! add `#[salsa::tracked]` queries (parse, analyze) and tracked structs over
//! the [`db::Database`] established here.

pub mod capabilities;
pub mod db;
pub mod line_index;
pub mod queries;
pub mod server;
