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
//!
//! Archive-aware clients send `flatppl/bundleSources` with `sources[id]` text,
//! `resolutions[id].modules[source]` target identities, and `uris[id]` document
//! URIs. The client validates archive paths and obtains remote trust approval.
//! Data entries stay outside `sources`. The server performs no archive I/O.
//! Members with `flatppl-zip:` identities remain read-only and mounted after
//! document close. New graph notifications update their text and module edges.

pub mod capabilities;
pub mod db;
pub mod line_index;
pub mod names;
pub mod outbound;
pub mod pool;
pub mod queries;
pub mod rename;
pub mod server;
pub mod signature;
mod text;
