//! HS3 (HEP Statistics Serialization Standard) → FlatPPL importer.
//!
//! Parses an HS3 JSON document and builds a `flatppl_core::Module`, following the
//! HS³/RooFit profile in `flatppl-design/docs/12-profiles.md`. Import only.
//!
//! Accepts two JSON formats:
//! - Native HS3: top-level `distributions`, `likelihoods`, etc.
//! - pyhf workspace: top-level `channels` key triggers the pyhf lift path.
mod error;
pub use error::{Error, Result};
pub(crate) mod builder;
mod convert;
pub(crate) mod dist_spec;
pub(crate) mod distribution;
pub(crate) mod expr;
pub(crate) mod histfactory;
pub(crate) mod likelihood;
pub(crate) mod model;
pub(crate) mod presets;
pub(crate) mod pyhf;

use flatppl_core::Module;
use flatppl_syntax::{Syntax, parse, print_with};

/// Validate that a freshly built module survives a print-then-reparse round
/// trip: the importer's output is printed in canonical (`Minimal`) surface text
/// and parsed back by `flatppl_syntax`. A parse failure means the importer
/// emitted syntactically-corrupt text, so the module is rejected with
/// [`Error::RoundTrip`] rather than returned silently. On success the original
/// module is returned unchanged (the reparse result is discarded — it only
/// serves as the validity check).
fn validate_round_trip(module: Module) -> Result<Module> {
    let text = print_with(&module, Syntax::Minimal);
    match parse(&text) {
        Ok(_) => Ok(module),
        Err(e) => Err(Error::RoundTrip(e.to_string())),
    }
}

/// Parse an HS3 or pyhf JSON document into a FlatPPL module.
///
/// Dispatch: if the top-level JSON object has a `"channels"` key, the pyhf
/// workspace lift path is taken; otherwise, the native HS3 path is used.
pub fn read(json: &str) -> Result<Module> {
    validate_round_trip(read_unchecked(json)?)
}

/// Like [`read`] but without the print→reparse self-check.
pub fn read_unchecked(json: &str) -> Result<Module> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    if value.get("channels").is_some() {
        let doc: model::PyhfDocument = serde_json::from_value(value)?;
        pyhf::pyhf_to_module(&doc)
    } else {
        let doc: model::Document = serde_json::from_value(value)?;
        convert::document_to_module(&doc)
    }
}

/// Parse a pyhf workspace JSON document into a FlatPPL module.
///
/// Requires the top-level `"channels"` key that identifies a pyhf workspace.
/// Returns [`Error::Unsupported`] if the document lacks `"channels"`, with a
/// hint to use the native HS3 path instead.
pub fn read_pyhf(json: &str) -> Result<Module> {
    validate_round_trip(read_pyhf_unchecked(json)?)
}

/// Like [`read_pyhf`] but without the print→reparse self-check.
pub fn read_pyhf_unchecked(json: &str) -> Result<Module> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    if value.get("channels").is_none() {
        return Err(Error::Unsupported(
            "expected a pyhf workspace with top-level `channels`; \
             this looks like a native HS3 document — use --from hs3 instead"
                .to_owned(),
        ));
    }
    let doc: model::PyhfDocument = serde_json::from_value(value)?;
    pyhf::pyhf_to_module(&doc)
}

/// Whether the document carries a *non-empty* `analyses` block. That block is
/// not imported by `read_hs3`/`read_pyhf` (it is inference configuration, out of
/// scope for model conversion); this lets a CLI surface a note without the
/// library printing. An absent, null, or empty `analyses` returns `false` —
/// nothing was actually dropped.
pub fn document_has_analyses(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| v.get("analyses").cloned())
        .is_some_and(|a| match a {
            serde_json::Value::Null => false,
            serde_json::Value::Array(items) => !items.is_empty(),
            other => !other.is_null(),
        })
}

/// Parse a native HS3 JSON document into a FlatPPL module.
///
/// If the document has a top-level `"channels"` key (which identifies a pyhf
/// workspace, not native HS3), returns [`Error::Unsupported`] with a hint to
/// use `--from pyhf`.
pub fn read_hs3(json: &str) -> Result<Module> {
    validate_round_trip(read_hs3_unchecked(json)?)
}

/// Like [`read_hs3`] but without the print→reparse self-check. Lower latency
/// for callers (e.g. language bindings) that don't need the importer's output
/// re-validated on every call.
pub fn read_hs3_unchecked(json: &str) -> Result<Module> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    if value.get("channels").is_some() {
        return Err(Error::Unsupported(
            "this looks like a pyhf workspace (top-level `channels` present); \
             use --from pyhf instead"
                .to_owned(),
        ));
    }
    let doc: model::Document = serde_json::from_value(value)?;
    convert::document_to_module(&doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal pyhf workspace fixture (single channel, two bins, one sample).
    const PYHF_JSON: &str = r#"{
        "channels": [{"name": "ch", "samples": [{"name": "sig",
            "data": [10.0, 20.0],
            "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]}]}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
        "observations": [{"name": "ch", "data": [12.0, 18.0]}],
        "version": "1.0.0"
    }"#;

    // Minimal native HS3 fixture (single Gaussian distribution).
    const HS3_JSON: &str = r#"{
        "distributions": [{"name": "obs", "type": "gaussian_dist",
            "mean": "mu", "sigma": "s", "x": "x_obs"}],
        "parameter_points": [{"name": "nom",
            "entries": [{"name": "mu", "value": 0.0}, {"name": "s", "value": 1.0}]}]
    }"#;

    #[test]
    fn read_pyhf_on_pyhf_workspace_succeeds() {
        read_pyhf(PYHF_JSON).expect("read_pyhf should parse a pyhf workspace");
    }

    #[test]
    fn read_pyhf_on_native_hs3_errors() {
        let err = read_pyhf(HS3_JSON).expect_err("read_pyhf should reject native HS3");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got: {err}"
        );
        assert!(
            err.to_string().contains("channels"),
            "error should mention `channels`: {err}"
        );
    }

    #[test]
    fn read_hs3_on_native_doc_succeeds() {
        read_hs3(HS3_JSON).expect("read_hs3 should parse a native HS3 document");
    }

    #[test]
    fn read_hs3_on_pyhf_workspace_errors_with_hint() {
        let err = read_hs3(PYHF_JSON).expect_err("read_hs3 should reject pyhf workspace");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got: {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("pyhf") && msg.contains("channels"),
            "error should hint pyhf/channels: {msg}"
        );
    }
}
