//! Per-binding rendering: the module as a list of fragments (the viewer's
//! unit, contract §4 of `flatppl-dev/math-view-design.md`).
//!
//! [`render`] takes an already-parsed module — typed and phased if the caller
//! ran inference, bare otherwise — and produces one [`BindingRender`] per row
//! in source order. [`render_source`] is the host-facing convenience that
//! parses the primary module and a pre-resolved `{path: text}` bundle, runs
//! inference, and renders; parse failures are errors, inference failures are
//! diagnostics on a module that still renders.

use std::collections::HashMap;
use std::sync::Arc;

use flatppl_core::{Doc, Module};
use flatppl_infer::{Level, ModuleBundle};

use crate::ast::Statement;
use crate::lower::{self, Kind, Row};
use crate::mathml;

/// One rendered row.
#[derive(Clone, Debug, PartialEq)]
pub struct BindingRender {
    /// The row's primary name (the first of `names`).
    pub name: String,
    /// Every binding the row binds — one, or a decomposition's targets.
    pub names: Vec<String>,
    pub kind: Kind,
    /// The statement as a `<math display="block">` fragment.
    pub mathml: String,
    /// The statement tree, for document assembly.
    pub statement: Statement,
    /// Bindings the fragment refers to, left-hand side first.
    pub refs: Vec<String>,
    /// Source byte range of the binding's right-hand side, when known.
    pub loc: Option<(u32, u32)>,
    pub annotation: Option<String>,
    /// The binding's doc-comment, when it has one.
    pub doc: Option<Doc>,
}

/// A construct a row could not render, or an inference message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// The row the message belongs to; empty for a module-level message.
    pub binding: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rendering {
    /// Row names in source order.
    pub order: Vec<String>,
    pub bindings: Vec<BindingRender>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Render every binding of `module` in source order.
pub fn render(module: &Module) -> Rendering {
    let rows = lower::lower_module(module);
    let mut bindings = Vec::with_capacity(rows.len());
    let mut diagnostics = Vec::new();
    for row in rows {
        let Row {
            binding,
            names,
            kind,
            statement,
            annotation,
            diagnostics: row_diags,
        } = row;
        let name = names[0].clone();
        for message in row_diags {
            diagnostics.push(Diagnostic {
                binding: name.clone(),
                message,
            });
        }
        let b = module.binding(binding);
        bindings.push(BindingRender {
            mathml: mathml::fragment(&name, &statement),
            refs: statement.refs(),
            loc: module.span_of(b.rhs).map(|s| (s.start, s.end)),
            doc: b.doc.clone(),
            name,
            names,
            kind,
            statement,
            annotation,
        });
    }
    Rendering {
        order: bindings.iter().map(|b| b.name.clone()).collect(),
        bindings,
        diagnostics,
    }
}

/// Parse `source` (the module at `path`) and the pre-resolved dependency
/// texts in `bundle`, infer, and render. Dependencies are keyed by the
/// `load_module` directive string as the importer spells it.
pub fn render_source(
    source: &str,
    path: &str,
    bundle: &HashMap<String, String>,
) -> Result<Rendering, String> {
    let mut module = flatppl_syntax::parse(source).map_err(|e| format!("{path}: {e}"))?;
    let mut deps = ModuleBundle::new();
    deps.set_root(path.to_string());
    for (dep_path, text) in bundle {
        let dep = flatppl_syntax::parse(text).map_err(|e| format!("{dep_path}: {e}"))?;
        deps.insert(dep_path.clone(), Arc::new(dep));
    }
    let infer_diags = flatppl_infer::infer_module(&mut module, &deps, Level::Shape);
    let mut rendering = render(&module);
    for d in infer_diags {
        if d.severity != flatppl_infer::Severity::Error {
            continue;
        }
        let binding = d
            .node
            .and_then(|n| owning_binding(&module, n))
            .unwrap_or_default();
        rendering.diagnostics.push(Diagnostic {
            binding,
            message: d.message,
        });
    }
    Ok(rendering)
}

/// The binding whose right-hand side contains `node`, by name.
fn owning_binding(module: &Module, node: flatppl_core::NodeId) -> Option<String> {
    fn contains(module: &Module, root: flatppl_core::NodeId, node: flatppl_core::NodeId) -> bool {
        if root == node {
            return true;
        }
        let mut found = false;
        module.for_each_child(root, |c| {
            if !found && contains(module, c, node) {
                found = true;
            }
        });
        found
    }
    module
        .bindings()
        .find(|(_, b)| contains(module, b.rhs, node))
        .map(|(_, b)| module.resolve(b.name).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_lists_rows_in_source_order_with_refs_and_kinds() {
        let src = "%%%\n# Title\n%%%\nflatppl_compat = \"0.1\"\n% the mean\nmu ~ Normal(0, 5)\nx = 2 * mu\nK = kernelof(x, mu = mu)";
        let r = render_source(src, "m.flatppl", &HashMap::new()).expect("renders");
        assert_eq!(r.order, vec!["mu", "x", "K"]);
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        let mu = &r.bindings[0];
        assert_eq!(mu.kind, Kind::Draw);
        assert!(
            mu.mathml
                .starts_with("<math display=\"block\" data-flatppl-binding=\"mu\">")
        );
        assert_eq!(mu.doc.as_ref().map(|d| d.lines.len()), Some(1));
        assert!(mu.loc.is_some());
        let x = &r.bindings[1];
        assert_eq!(x.refs, vec!["x", "mu"]);
        assert_eq!(r.bindings[2].kind, Kind::Callable);
    }

    #[test]
    fn inference_errors_become_diagnostics_and_the_module_still_renders() {
        let src = "y = Normal(0, 1, 2)\nz = y";
        let r = render_source(src, "m.flatppl", &HashMap::new()).expect("renders");
        assert_eq!(r.order, vec!["y", "z"]);
        assert!(
            r.diagnostics.iter().any(|d| d.binding == "y"),
            "{:?}",
            r.diagnostics
        );
        assert!(r.bindings[0].mathml.contains("<mi>Normal</mi>"));
    }

    #[test]
    fn a_parse_error_is_an_error() {
        assert!(render_source("x = (", "m.flatppl", &HashMap::new()).is_err());
    }

    #[test]
    fn dependencies_resolve_from_the_bundle() {
        let mut bundle = HashMap::new();
        bundle.insert(
            "helpers.flatppl".to_string(),
            "f(x) = 2 * x\nc = elementof(reals)".to_string(),
        );
        let src = "h = load_module(\"helpers.flatppl\")\ny = h.f(3.0)";
        let r = render_source(src, "m.flatppl", &bundle).expect("renders");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        assert_eq!(r.bindings[0].kind, Kind::Module);
        assert!(
            r.bindings[1]
                .mathml
                .contains("<mi data-flatppl-ref=\"h\">h</mi><mo>.</mo><mi>f</mi>")
        );
    }
}
