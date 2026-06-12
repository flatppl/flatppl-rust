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
    /// Byte span `[start, end)` into the source, when known. Renderers fall
    /// back to highlighting all of `line` when absent.
    pub span: Option<(u32, u32)>,
}

impl Error {
    /// An error with no source position.
    pub fn new(message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
            line: 0,
            span: None,
        }
    }

    /// An error at a known 1-based source line.
    pub fn at(line: usize, message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
            line,
            span: None,
        }
    }

    /// An error at a known 1-based source line and byte span.
    pub fn at_span(line: usize, span: (u32, u32), message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
            line,
            span: Some(span),
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
