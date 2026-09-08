//! `flatppl-wasm-api` — the WebAssembly / JavaScript binding surface for
//! FlatPPL.
//!
//! A thin `wasm-bindgen` wrapper over the pure library crates (the
//! `flatppl-hs3` importers + the `flatppl-syntax` printer), built to wasm
//! and consumed in-browser by the web gallery's "Convert to FlatPPL"
//! command. The conversion logic lives in the libraries; this crate only
//! adapts string-in / string-out to the JS boundary.
//!
//! It is the **browser member of the host-binding family** — the same
//! library crates also link into PyO3 / jlrs / cxx hosts (see the workspace
//! ARCHITECTURE). It is NOT part of the wasm32-linkable *library* set: it is
//! the crate that *produces* the wasm artifact, linking those libraries in.
//!
//! The `#[wasm_bindgen]` entry point compiles only for `wasm32`; on the host
//! just [`convert_str`] (and its tests) build, so `wasm-bindgen` never enters
//! a host build.

/// Convert a model document to FlatPPL source text.
///
/// `from` selects the importer (`"pyhf"` or `"hs3"`); `to` must be
/// `"flatppl"` (the only target for now). Returns the rendered FlatPPL text,
/// or an error message (the importer/printer diagnostic). This is the
/// plain-Rust core — host-testable, no wasm — that the wasm entry point wraps.
pub fn convert_str(input: &str, from: &str, to: &str) -> Result<String, String> {
    if to != "flatppl" {
        return Err(format!(
            "unsupported target '{to}' (only 'flatppl' for now)"
        ));
    }
    let module = match from {
        "pyhf" => flatppl_hs3::read_pyhf(input).map_err(|e| format!("pyhf import: {e}"))?,
        "hs3" => flatppl_hs3::read_hs3(input).map_err(|e| format!("hs3 import: {e}"))?,
        other => {
            return Err(format!(
                "unsupported source '{other}' (expected 'pyhf' or 'hs3')"
            ));
        }
    };
    Ok(flatppl_syntax::print(&module))
}

/// The wasm/JS boundary. `convert(input, from, to) -> string`; a returned
/// `Err` becomes a JavaScript `Error` object carrying the message.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub fn convert(input: &str, from: &str, to: &str) -> Result<String, JsError> {
        super::convert_str(input, from, to).map_err(|e| JsError::new(&e))
    }
}

#[cfg(test)]
mod tests {
    use super::convert_str;

    // A real pyhf workspace, shared with the hs3 crate's own tests.
    const PYHF_2BIN: &str = include_str!("../../hs3/tests/fixtures/2bin_1channel.json");

    #[test]
    fn pyhf_to_flatppl_preserves_the_model() {
        let out = convert_str(PYHF_2BIN, "pyhf", "flatppl").expect("pyhf → flatppl");
        // Expected quantities come from the fixture, not from another importer
        // run. Compare parsed expressions so whitespace and printer sugar are
        // irrelevant. shapesys has rate factor (nominal / uncertainty)^2.
        let expected = [
            ("singlechannel_observed", "[50.0, 60.0]"),
            ("singlechannel_signal_nominal", "[5.0, 10.0]"),
            ("singlechannel_background_nominal", "[50.0, 60.0]"),
            ("uncorr_bkguncrt_sigma", "[5.0, 12.0]"),
            (
                "uncorr_bkguncrt_tau",
                "(singlechannel_background_nominal ./ uncorr_bkguncrt_sigma) .^ 2",
            ),
            (
                "singlechannel_likelihood",
                "likelihoodof(singlechannel_model, singlechannel_observed)",
            ),
            (
                "likelihood",
                "joint_likelihood(singlechannel_likelihood, uncorr_bkguncrt_constraint_likelihood)",
            ),
        ];
        let mut source = out;
        for (name, expression) in expected {
            source.push_str(&format!("\n_expected_{name} = {expression}\n"));
        }
        let module = flatppl_syntax::parse(&source).expect("returned source must parse");
        let rhs = |name: &str| {
            module
                .bindings()
                .find(|(_, binding)| module.resolve(binding.name) == name)
                .unwrap_or_else(|| panic!("missing binding {name}"))
                .1
                .rhs
        };
        for (name, _) in expected {
            assert!(
                module.structural_eq(rhs(name), rhs(&format!("_expected_{name}"))),
                "converted expression changed for {name}"
            );
        }
    }

    #[test]
    fn rejects_unknown_source_and_target() {
        assert!(convert_str("{}", "stan", "flatppl").is_err());
        assert!(convert_str("{}", "pyhf", "json").is_err());
    }
}
