//! Converter error type.
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Malformed or schema-invalid JSON.
    Json(serde_json::Error),
    /// A distribution `type` with no mapping.
    UnknownDistType(String),
    /// A histfactory modifier `type` with no mapping.
    UnknownModifier(String),
    /// A string reference that names no parameter/function/distribution.
    UnresolvedRef(String),
    /// A construct outside this importer's supported subset.
    Unsupported(String),
    /// Same-named modifiers declare incompatible constraint types.
    IncompatibleConstraint { parameter: String },
    /// No observation data found for the named channel.
    NoObservation(String),
    /// The built module failed to print-then-reparse cleanly: the importer
    /// produced surface text that `flatppl_syntax` cannot parse back. Carries
    /// the parser's error message.
    RoundTrip(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Json(e) => write!(f, "invalid HS3 JSON: {e}"),
            Error::UnknownDistType(t) => write!(f, "unsupported HS3 distribution type: {t}"),
            Error::UnknownModifier(t) => write!(f, "unsupported histfactory modifier: {t}"),
            Error::UnresolvedRef(n) => write!(f, "unresolved reference: {n}"),
            Error::Unsupported(w) => write!(f, "unsupported HS3 construct: {w}"),
            Error::IncompatibleConstraint { parameter } => {
                write!(
                    f,
                    "modifier `{parameter}` used with incompatible constraint types"
                )
            }
            Error::NoObservation(ch) => {
                write!(f, "no observation data for channel `{ch}`")
            }
            Error::RoundTrip(msg) => {
                write!(f, "imported module failed to re-parse: {msg}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Json(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_error_wraps() {
        let e: Error = serde_json::from_str::<serde_json::Value>("{")
            .unwrap_err()
            .into();
        assert!(matches!(e, Error::Json(_)));
    }
}
