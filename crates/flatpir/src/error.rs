//! Errors raised while reading FlatPIR text.
//!
//! Reading is fail-fast: the first malformed form aborts with an [`Error`]
//! carrying a human-readable message and, where known, the 1-based source line.
//! (Writing is infallible — a well-formed [`Module`](flatppl_core::Module)
//! always renders.)

use std::fmt;

/// A FlatPIR read error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub message: String,
    /// 1-based source line, or `0` when the error is not tied to a position.
    pub line: usize,
}

impl Error {
    /// An error with no source position.
    pub fn new(message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
            line: 0,
        }
    }

    /// An error at a known 1-based source line.
    pub fn at(line: usize, message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
            line,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line > 0 {
            write!(
                f,
                "FlatPIR read error (line {}): {}",
                self.line, self.message
            )
        } else {
            write!(f, "FlatPIR read error: {}", self.message)
        }
    }
}

impl std::error::Error for Error {}

/// Result type for FlatPIR reading.
pub type Result<T> = std::result::Result<T, Error>;
