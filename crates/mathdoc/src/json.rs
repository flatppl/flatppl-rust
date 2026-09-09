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
//!   "doc": { "title": "Eight Schools", "html": "<p>…</p>" },
//!   "bindings": [ { "name": "mu", "names": ["mu"], "kind": "draw",
//!                   "mathml": "<math …>…</math>", "refs": [],
//!                   "loc": { "start": 12, "end": 30 }, "annotation": null,
//!                   "doc": { "html": "<p>the mean</p>", "block": false } } ],
//!   "diagnostics": [ { "binding": "", "message": "…" } ] }
//! ```
//!
//! `doc` is a doc-comment rendered to an HTML fragment by [`crate::document`]
//! (Markdown with `$…$` math as MathML, sanitised, headings shifted one level
//! down); `block` marks a multi-line `%%%` comment. The module-level `doc` is
//! the doc-comment on `flatppl_compat`, its leading heading split off as
//! `title`. Both are absent when there is no such comment. Math the converter
//! refused shows as `<span class="math-error">` and adds a diagnostic on the
//! binding (module-level for the module doc).
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
//! `tex` and `typst` give native math source without delimiters. Unknown
//! formats add a module-level diagnostic rather than failing. The notation
//! key uses the same requested formats as the binding rows; its `note` is an
//! HTML fragment (text escaped, parameter letters as inline MathML).
//! Set `document: true` to also receive `document: { html, css }`: the full
//! article and scoped stylesheet shared with the standalone HTML export.
//! Binding metadata remains available for host selection and source navigation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document;
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
    #[serde(default)]
    document: bool,
}

fn default_path() -> String {
    "model.flatppl".to_string()
}

fn default_formats() -> Vec<String> {
    vec!["mathml".to_string()]
}

#[derive(Serialize)]
struct Response {
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<DocumentJson>,
    order: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    doc: Option<ModuleDocJson>,
    bindings: Vec<BindingJson>,
    diagnostics: Vec<DiagnosticJson>,
    notation: Vec<NotationJson>,
}

#[derive(Serialize)]
struct DocumentJson {
    html: String,
    css: &'static str,
}

#[derive(Serialize)]
struct NotationJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    mathml: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    typst: Option<String>,
    source: String,
    note: String,
}

#[derive(Serialize)]
struct ModuleDocJson {
    title: Option<String>,
    html: String,
}

#[derive(Serialize)]
struct DocJson {
    html: String,
    block: bool,
}

#[derive(Serialize)]
struct BindingJson {
    name: String,
    names: Vec<String>,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mathml: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    typst: Option<String>,
    refs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loc: Option<Loc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    doc: Option<DocJson>,
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
    let (module, rendering) =
        render::render_source_with_module(&req.source, &req.path, &req.bundle)?;
    let document = req.document.then(|| {
        let title = req.path.rsplit('/').next().unwrap_or(&req.path);
        let title = title.strip_suffix(".flatppl").unwrap_or(title);
        DocumentJson {
            html: document::fragment(&module, &rendering, title).html,
            css: document::CSS,
        }
    });
    let want_mathml = req.formats.is_empty() || req.formats.iter().any(|f| f == "mathml");
    let mut response = to_response(rendering, want_mathml, &req.formats);
    response.document = document;
    serde_json::to_string(&response).map_err(|e| format!("response: {e}"))
}

fn to_response(rendering: Rendering, want_mathml: bool, formats: &[String]) -> Response {
    let want_tex = formats.iter().any(|f| f == "tex");
    let want_typst = formats.iter().any(|f| f == "typst");
    let Rendering {
        order,
        bindings,
        diagnostics,
        module_doc,
        notation,
    } = rendering;
    let mut diagnostics: Vec<DiagnosticJson> = diagnostics
        .into_iter()
        .map(|d| DiagnosticJson {
            binding: d.binding,
            message: d.message,
        })
        .collect();
    let doc = module_doc.as_ref().map(|d| {
        let rendered = document::module_doc_html(d);
        diagnostics.extend(rendered.errors.into_iter().map(|message| DiagnosticJson {
            binding: String::new(),
            message,
        }));
        ModuleDocJson {
            title: rendered.title,
            html: rendered.html,
        }
    });
    for f in formats {
        if !matches!(f.as_str(), "mathml" | "tex" | "typst") {
            diagnostics.push(DiagnosticJson {
                binding: String::new(),
                message: format!("unsupported format `{f}`; use `mathml`, `tex`, or `typst`"),
            });
        }
    }
    let mut rows = Vec::with_capacity(bindings.len());
    for b in bindings {
        let doc = b.doc.as_ref().map(|d| {
            let rendered = document::doc_html(d);
            diagnostics.extend(rendered.errors.into_iter().map(|message| DiagnosticJson {
                binding: b.name.clone(),
                message,
            }));
            DocJson {
                html: rendered.html,
                block: rendered.block,
            }
        });
        rows.push(BindingJson {
            name: b.name,
            names: b.names,
            kind: b.kind.as_str(),
            mathml: want_mathml.then_some(b.mathml),
            tex: want_tex.then(|| crate::tex::statement(&b.statement)),
            typst: want_typst.then(|| crate::typst::statement(&b.statement)),
            refs: b.refs,
            loc: b.loc.map(|(start, end)| Loc { start, end }),
            annotation: b.annotation,
            doc,
        });
    }
    Response {
        document: None,
        notation: notation
            .into_iter()
            .map(|n| NotationJson {
                mathml: want_mathml.then(|| crate::mathml::expr(&n.form)),
                tex: want_tex.then(|| crate::tex::expr(&n.form)),
                typst: want_typst.then(|| crate::typst::expr(&n.form)),
                note: n.note_html(),
                source: n.source,
            })
            .collect(),
        order,
        doc,
        bindings: rows,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_round_trips() {
        let input = serde_json::json!({
            "source": "%%%\n# Title\n\nAbstract with $\\mu$.\n%%%\nflatppl_compat = \"0.1\"\n% the mean, $\\bad{x}$\nmu ~ Normal(0, 5)\nx = 2 * mu\na, b ~ MvNormal(mu = [0.0, 0.0], cov = eye(2))",
            "path": "m.flatppl",
            "bundle": {},
            "formats": ["mathml", "tex", "typst", "svg"]
        })
        .to_string();
        let out = render_math(&input).expect("renders");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["order"], serde_json::json!(["mu", "x", "a"]));
        assert_eq!(v["bindings"][0]["kind"], "draw");
        let tex = v["bindings"][0]["tex"].as_str().expect("plain TeX");
        assert!(tex.contains(r"\mathcal{N}\left(0, {5}^{2}\right)"), "{tex}");
        assert!(!tex.contains("htmlData"));
        assert!(
            v["bindings"][0]["typst"]
                .as_str()
                .unwrap()
                .contains("attach(5, tr: 2)")
        );
        assert!(
            v["bindings"][0]["mathml"]
                .as_str()
                .unwrap()
                .starts_with("<math ")
        );
        assert_eq!(v["bindings"][2]["names"], serde_json::json!(["a", "b"]));
        assert_eq!(v["bindings"][0]["refs"], serde_json::json!([]));
        assert_eq!(v["bindings"][1]["refs"], serde_json::json!(["mu"]));
        let normal = v["notation"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["source"] == "Normal(mu, sigma)")
            .unwrap();
        assert!(
            normal["mathml"]
                .as_str()
                .unwrap()
                .contains("<msup><mi>σ</mi><mn>2</mn></msup>")
        );
        assert!(normal["note"].as_str().unwrap().contains("variance"));
        assert!(v["bindings"][0]["loc"]["start"].is_number());
        assert_eq!(v["doc"]["title"], "Title");
        assert!(
            v["doc"]["html"]
                .as_str()
                .unwrap()
                .starts_with("<p>Abstract with <math>")
        );
        assert_eq!(v["bindings"][0]["doc"]["block"], false);
        assert!(
            v["bindings"][0]["doc"]["html"]
                .as_str()
                .unwrap()
                .contains("math-error")
        );
        assert!(v["bindings"][1].get("doc").is_none());
        assert!(
            v["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|d| d["binding"] == "mu" && d["message"].as_str().unwrap().contains("bad"))
        );
        assert!(
            v["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|d| d["message"].as_str().unwrap().contains("svg"))
        );
    }

    #[test]
    fn embedded_document_matches_the_standalone_article() {
        let source =
            "% $\\alpha$\nx ~ Normal(0, 2)\ndata = [1,2,3,4,5,6,7,8,9,10,11,12,13]\n_ = Normal(0)";
        let input = serde_json::json!({ "source": source, "document": true });
        let out: serde_json::Value =
            serde_json::from_str(&render_math(&input.to_string()).unwrap()).unwrap();
        let (module, rendering) =
            render::render_source_with_module(source, "model.flatppl", &HashMap::new()).unwrap();
        let page = document::html(&module, &rendering, "model");
        let article = out["document"]["html"].as_str().unwrap();
        assert!(page.contains(article));
        assert!(page.contains(out["document"]["css"].as_str().unwrap()));
        assert!(article.contains("flatppl-data-grid"));
        assert!(article.contains("<mn>13</mn>"));
        assert!(article.contains("Random variables"));
        assert!(article.contains("data-flatppl-binding=\"x\""));
        assert!(!article.contains("$\\alpha$"));
        assert!(article.contains(out["bindings"][0]["doc"]["html"].as_str().unwrap()));
        for diagnostic in out["diagnostics"].as_array().unwrap() {
            assert!(article.contains(&crate::mathml::escape(
                diagnostic["message"].as_str().unwrap()
            )));
        }
        let legacy: serde_json::Value = serde_json::from_str(
            &render_math(&serde_json::json!({ "source": source }).to_string()).unwrap(),
        )
        .unwrap();
        assert!(legacy.get("document").is_none());
        assert_eq!(out["bindings"], legacy["bindings"]);
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
