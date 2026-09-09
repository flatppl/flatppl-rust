//! The string-in / string-out JSON contract (`flatppl-dev/math-view-design.md`
//! §4), used by `flatppl-wasm-api::render_math` and available to any host.
//!
//! Request:
//!
//! ```json
//! { "source": "<module text>", "path": "models/model.flatppl",
//!   "bundle": { "<resolved path>": "<text>" }, "formats": ["mathml"] }
//! ```
//!
//! Response:
//!
//! ```json
//! { "order": ["mu", "tau"],
//!   "bindings": [ { "name": "mu", "names": ["mu"], "kind": "draw",
//!                   "mathml": "<math …>…</math>", "refs": ["mu"],
//!                   "loc": { "start": 12, "end": 30 }, "annotation": null } ],
//!   "diagnostics": [ { "binding": "", "message": "…" } ] }
//! ```
//!
//! `bundle` is keyed by each dependency's **resolved** path — the directive
//! joined to its importer's directory per spec §04 (`/` separator, `..`
//! allowed, an `http`/`https` URL as is) — exactly what the host resolved to
//! load the text; transitive directives resolve against their own importer.
//! `order` is the rows in **source order** (`NOTATION.md`: no topological
//! sort). `names` lists every binding a row binds: a decomposition `a, b ~ M`
//! is one row named `a` with `names: ["a", "b"]`, and `order` lists row names
//! only. `refs` lists the *other* rows a row refers to, in order of
//! appearance, its own names excluded. An empty `formats` means `["mathml"]`;
//! formats other than `mathml` are not produced yet, and asking for one adds a
//! module-level diagnostic rather than failing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::render::{self, Rendering};

#[derive(Deserialize)]
struct Request {
    source: String,
    #[serde(default = "default_path")]
    path: String,
    #[serde(default)]
    bundle: HashMap<String, String>,
    #[serde(default = "default_formats")]
    formats: Vec<String>,
}

fn default_path() -> String {
    "model.flatppl".to_string()
}

fn default_formats() -> Vec<String> {
    vec!["mathml".to_string()]
}

#[derive(Serialize)]
struct Response {
    order: Vec<String>,
    bindings: Vec<BindingJson>,
    diagnostics: Vec<DiagnosticJson>,
}

#[derive(Serialize)]
struct BindingJson {
    name: String,
    names: Vec<String>,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mathml: Option<String>,
    refs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loc: Option<Loc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotation: Option<String>,
}

#[derive(Serialize)]
struct Loc {
    start: u32,
    end: u32,
}

#[derive(Serialize)]
struct DiagnosticJson {
    binding: String,
    message: String,
}

/// Render the request in `input` (JSON) to the response (JSON). A malformed
/// request or an unparsable primary module is an `Err` with the message.
pub fn render_math(input: &str) -> Result<String, String> {
    let req: Request = serde_json::from_str(input).map_err(|e| format!("request: {e}"))?;
    let rendering = render::render_source(&req.source, &req.path, &req.bundle)?;
    let want_mathml = req.formats.is_empty() || req.formats.iter().any(|f| f == "mathml");
    let response = to_response(rendering, want_mathml, &req.formats);
    serde_json::to_string(&response).map_err(|e| format!("response: {e}"))
}

fn to_response(rendering: Rendering, want_mathml: bool, formats: &[String]) -> Response {
    let Rendering {
        order,
        bindings,
        diagnostics,
    } = rendering;
    let mut diagnostics: Vec<DiagnosticJson> = diagnostics
        .into_iter()
        .map(|d| DiagnosticJson {
            binding: d.binding,
            message: d.message,
        })
        .collect();
    for f in formats {
        if f != "mathml" {
            diagnostics.push(DiagnosticJson {
                binding: String::new(),
                message: format!("format `{f}` is not available yet; only `mathml` is produced"),
            });
        }
    }
    Response {
        order,
        bindings: bindings
            .into_iter()
            .map(|b| BindingJson {
                name: b.name,
                names: b.names,
                kind: b.kind.as_str(),
                mathml: want_mathml.then_some(b.mathml),
                refs: b.refs,
                loc: b.loc.map(|(start, end)| Loc { start, end }),
                annotation: b.annotation,
            })
            .collect(),
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_round_trips() {
        let input = serde_json::json!({
            "source": "mu ~ Normal(0, 5)\nx = 2 * mu\na, b ~ MvNormal(mu = [0.0, 0.0], cov = eye(2))",
            "path": "m.flatppl",
            "bundle": {},
            "formats": ["mathml", "typst"]
        })
        .to_string();
        let out = render_math(&input).expect("renders");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["order"], serde_json::json!(["mu", "x", "a"]));
        assert_eq!(v["bindings"][0]["kind"], "draw");
        assert!(
            v["bindings"][0]["mathml"]
                .as_str()
                .unwrap()
                .starts_with("<math ")
        );
        assert_eq!(v["bindings"][2]["names"], serde_json::json!(["a", "b"]));
        assert_eq!(v["bindings"][0]["refs"], serde_json::json!([]));
        assert_eq!(v["bindings"][1]["refs"], serde_json::json!(["mu"]));
        assert!(v["bindings"][0]["loc"]["start"].is_number());
        assert!(
            v["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|d| d["message"].as_str().unwrap().contains("typst"))
        );
    }

    #[test]
    fn defaults_fill_in_and_bad_requests_error() {
        let out = render_math(r#"{"source": "x = 1"}"#).expect("renders");
        assert!(out.contains("\"mathml\":\"<math"));
        let out = render_math(r#"{"source": "x = 1", "formats": []}"#).expect("renders");
        assert!(out.contains("\"mathml\":\"<math"), "{out}");
        assert!(render_math("not json").is_err());
        assert!(render_math(r#"{"source": "x = ("}"#).is_err());
    }
}
