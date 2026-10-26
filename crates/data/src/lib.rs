//! Readers for the data formats every FlatPPL engine must support (spec §07
//! `load_data`): CSV and WSV, Arrow IPC, and Safetensors. JSON needs no reader
//! of its own beyond `serde_json`. Each format has exactly one implementation
//! here; the `load_data` host and the formula frontend interpret its output.

#[cfg(feature = "arrow")]
mod arrow;
mod delimited;
#[cfg(feature = "safetensors")]
pub mod safetensors;
mod table;

pub use delimited::{TextTable, read_csv, read_wsv};
pub use table::{Column, ColumnValues, Table};

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
