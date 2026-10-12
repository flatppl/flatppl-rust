//! Converter-neutral IR construction, with the HS3 diagnostic wrapper.
pub(crate) use flatppl_convert::{Builder, sanitize_ident};

pub(crate) fn check_binding_name(name: &str, what: &str) -> crate::error::Result<()> {
    flatppl_convert::check_binding_name(name, what).map_err(crate::error::Error::Unsupported)
}
