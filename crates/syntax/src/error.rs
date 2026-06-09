//! Errors raised while tokenizing or parsing canonical FlatPPL surface text.

use std::fmt;

/// A FlatPPL parse/tokenize error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub message: String,
    /// 1-based source line, or `0` when not localized.
    pub line: u32,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
            line: 0,
        }
    }

    pub fn at(line: u32, message: impl Into<String>) -> Self {
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
                "FlatPPL parse error (line {}): {}",
                self.line, self.message
            )
        } else {
            write!(f, "FlatPPL parse error: {}", self.message)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
