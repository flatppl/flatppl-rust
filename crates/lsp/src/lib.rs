//! The FlatPPL language server library.
//!
//! Houses the salsa database that backs incremental analysis. Tasks downstream
//! add `#[salsa::tracked]` queries (parse, analyze) and tracked structs over
//! the [`db::Database`] established here.

pub mod capabilities;
pub mod db;
pub mod line_index;
pub mod queries;
pub mod server;
