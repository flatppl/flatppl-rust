//! Per-binding rendering: the module as a list of fragments (the viewer's
//! unit, contract §4 of `flatppl-dev/math-view-design.md`).
//!
//! [`render`] takes an already-parsed module — typed and phased if the caller
//! ran inference, bare otherwise — and produces one [`BindingRender`] per row
//! in source order. [`render_source`] is the host-facing convenience that
//! parses the primary module and a pre-resolved `{resolved path: text}`
//! bundle, runs inference, and renders; parse failures are errors, inference
//! failures are diagnostics on a module that still renders.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use flatppl_core::{CallHead, Doc, Idx, Module, Node, NodeId, Scalar};
use flatppl_fileaccess::Location;
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

/// Parse `source` (the module at `path`) and the dependency texts in
/// `bundle`, infer, and render.
///
/// `bundle` is keyed by each dependency's **resolved** path — the
/// `load_module` directive joined to its importer's directory per spec §04
/// ("relative to the directory of the file containing the call", `/`
/// separator, `..` allowed; an `http`/`https` URL as is), which is what a host
/// resolved to load the text in the first place. Directives are walked from
/// `path` through the bundle, so a transitive dependency's directive resolves
/// against *its* importer. A key spelled exactly as a directive is accepted
/// too, for a one-level bundle that did no resolution of its own.
pub fn render_source(
    source: &str,
    path: &str,
    bundle: &HashMap<String, String>,
) -> Result<Rendering, String> {
    let mut module = flatppl_syntax::parse(source).map_err(|e| format!("{path}: {e}"))?;
    let deps = resolve_bundle(&module, path, bundle)?;
    let infer_diags = flatppl_infer::infer_module(&mut module, &deps, Level::Shape);
    let mut rendering = render(&module);
    attach_inference_diagnostics(&mut rendering, &module, &infer_diags);
    Ok(rendering)
}

/// Add inference ERRORS to `rendering` as diagnostics on the binding whose
/// right-hand side holds the offending node (module-level when there is
/// none). Notes — honest `%deferred` gaps — are not rendering defects and are
/// left out.
pub fn attach_inference_diagnostics(
    rendering: &mut Rendering,
    module: &Module,
    diags: &[flatppl_infer::Diagnostic],
) {
    for d in diags {
        if d.severity != flatppl_infer::Severity::Error {
            continue;
        }
        let binding = d
            .node
            .and_then(|n| owning_binding(module, n))
            .unwrap_or_default();
        rendering.diagnostics.push(Diagnostic {
            binding,
            message: d.message.clone(),
        });
    }
}

/// Parse the bundle and record every `load_module` resolution reachable from
/// the root, keyed by the bundle's own (resolved) paths.
fn resolve_bundle(
    root: &Module,
    root_path: &str,
    bundle: &HashMap<String, String>,
) -> Result<ModuleBundle, String> {
    // Parsed once each; matched by lexically normalised location so `a/./b`,
    // `a/b` and `x/../a/b` denote the same key.
    let mut by_norm: HashMap<String, (String, Arc<Module>)> = HashMap::new();
    for (key, text) in bundle {
        let parsed = flatppl_syntax::parse(text).map_err(|e| format!("{key}: {e}"))?;
        by_norm.insert(
            Location::parse(key).normalized().display(),
            (key.clone(), Arc::new(parsed)),
        );
    }
    let mut deps = ModuleBundle::new();
    deps.set_root(root_path.to_string());
    // Every bundled module stays reachable under its own key.
    for (key, arc) in by_norm.values() {
        deps.insert(key.clone(), arc.clone());
    }
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, Arc<Module>)> = VecDeque::new();
    for literal in load_module_literals(root) {
        link(
            root_path,
            &literal,
            &by_norm,
            &mut deps,
            &mut visited,
            &mut queue,
        );
    }
    while let Some((importer, importer_module)) = queue.pop_front() {
        for literal in load_module_literals(&importer_module) {
            link(
                &importer,
                &literal,
                &by_norm,
                &mut deps,
                &mut visited,
                &mut queue,
            );
        }
    }
    Ok(deps)
}

/// Record that `literal`, spelled in `importer`, denotes the bundle entry it
/// resolves to (resolved path first, bare directive second), and queue that
/// entry for its own directives.
fn link(
    importer: &str,
    literal: &str,
    by_norm: &HashMap<String, (String, Arc<Module>)>,
    deps: &mut ModuleBundle,
    visited: &mut HashSet<String>,
    queue: &mut VecDeque<(String, Arc<Module>)>,
) {
    let resolved = Location::parse(importer)
        .join(literal)
        .normalized()
        .display();
    let entry = by_norm
        .get(&resolved)
        .or_else(|| by_norm.get(&Location::parse(literal).normalized().display()));
    let Some((key, arc)) = entry else {
        return;
    };
    deps.insert_resolved(
        importer.to_string(),
        literal.to_string(),
        key.clone(),
        arc.clone(),
    );
    if visited.insert(key.clone()) {
        queue.push_back((key.clone(), arc.clone()));
    }
}

/// The `load_module` source literals of `module` (positional or `source =`).
fn load_module_literals(module: &Module) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..module.node_count() {
        let Node::Call(call) = module.node(NodeId::from_usize(i)) else {
            continue;
        };
        let CallHead::Builtin(head) = call.head else {
            continue;
        };
        if module.resolve(head) != "load_module" {
            continue;
        }
        let source = call
            .args
            .first()
            .copied()
            .or_else(|| {
                call.named
                    .iter()
                    .find(|n| module.resolve(n.name) == "source")
                    .map(|n| n.value)
            })
            .and_then(|id| match module.node(id) {
                Node::Lit(Scalar::Str(s)) => Some(s.to_string()),
                _ => None,
            });
        if let Some(s) = source {
            out.push(s);
        }
    }
    out
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
    fn bundle_keys_are_resolved_paths_and_transitive_directives_resolve_against_their_importer() {
        let mut bundle = HashMap::new();
        bundle.insert(
            "lib/h.flatppl".to_string(),
            "u = load_module(\"util.flatppl\")\ng(x) = u.twice(x)".to_string(),
        );
        bundle.insert(
            "lib/util.flatppl".to_string(),
            "twice(x) = 2 * x".to_string(),
        );
        let src = "h = load_module(\"../lib/h.flatppl\")\ny = h.g(1.0)";
        let r = render_source(src, "models/m.flatppl", &bundle).expect("renders");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        assert_eq!(r.bindings[0].kind, Kind::Module);
        assert!(
            r.bindings[1]
                .mathml
                .contains("<mi data-flatppl-ref=\"h\">h</mi><mo>.</mo><mi>g</mi>")
        );
        // `./` and `..` spellings of the same key still match.
        let mut spelled = HashMap::new();
        spelled.insert(
            "./lib/./h.flatppl".to_string(),
            bundle["lib/h.flatppl"].clone(),
        );
        spelled.insert(
            "lib/x/../util.flatppl".to_string(),
            bundle["lib/util.flatppl"].clone(),
        );
        let r = render_source(src, "models/m.flatppl", &spelled).expect("renders");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
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
