//! Capability helpers: the owned diagnostic type the queries carry, mapping
//! from parse errors and inference diagnostics, and the public `diagnostics`,
//! `hover`, `document_symbols`, `workspace_symbols`, and `inlay_hints` functions
//! that convert internal types to LSP protocol values.

use std::str::FromStr;

use crate::db::{Catalogues, Database, FileSet, SourceFile};
use crate::queries::{
    SpanIndex, analyze, line_index, node_at_offset_indexed, parse, parsed_catalogues, resolve_path,
};
use flatppl_core::{CallHead, Node, RefNs, Scalar};

/// Test-only counter incremented each time the static completion set is built.
/// Used by `static_completion_items_built_once` to prove the `OnceLock` cache is
/// populated exactly once across calls.
#[cfg(test)]
static STATIC_COMPLETION_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// An owned, salsa-friendly diagnostic: a byte range into the source, a severity,
/// and a message. The LSP `Range` (UTF-16) is computed at emit time from these
/// byte offsets via the line index — kept out of here so this stays salsa-storable.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LspDiag {
    pub start: u32, // byte offset
    pub end: u32,   // byte offset
    pub severity: DiagSeverity,
    pub message: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DiagSeverity {
    Error,
    Hint,
}

impl LspDiag {
    /// Map a fail-fast parse error to a single diagnostic over its span.
    ///
    /// `flatppl_syntax::Error` carries a public `span: Option<(u32, u32)>` field
    /// (byte offsets `[start, end)`) and a `message: String` field. When `span`
    /// is `None` (error is not spatially localised) the range collapses to
    /// `(0, 0)` so the caller can still emit a module-level squiggle.
    pub fn from_parse_error(e: &flatppl_syntax::Error) -> Self {
        let (start, end) = e.span.unwrap_or((0, 0));
        LspDiag {
            start,
            end,
            severity: DiagSeverity::Error,
            message: e.message.clone(),
        }
    }

    /// Map an inference diagnostic, anchoring its range to the offending node's
    /// span when known (else (0,0)). `Error` → `Error`, `Note` → `Hint`.
    pub fn from_infer(d: &flatppl_infer::Diagnostic, module: &flatppl_core::Module) -> Self {
        let (start, end) = d
            .node
            .and_then(|n| module.span_of(n))
            .map(|s| (s.start, s.end))
            .unwrap_or((0, 0));
        let severity = match d.severity {
            flatppl_infer::Severity::Error => DiagSeverity::Error,
            flatppl_infer::Severity::Note => DiagSeverity::Hint,
        };
        LspDiag {
            start,
            end,
            severity,
            message: d.message.clone(),
        }
    }
}

// ── LSP protocol output ──────────────────────────────────────────────────────

/// Map all `LspDiag`s from an analyzed file to `lsp_types::Diagnostic` values.
///
/// Byte offsets are converted to UTF-16 (line, character) positions via
/// [`crate::line_index::LineIndex`] so that LSP clients receive correctly positioned ranges.
pub fn diagnostics(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
) -> Vec<lsp_types::Diagnostic> {
    let analyzed = analyze(db, file, fs, cats);
    let li = line_index(db, file);
    analyzed
        .diagnostics(db)
        .iter()
        .map(|d| {
            let start = li.position(d.start);
            let end = li.position(d.end);
            let range = lsp_types::Range::new(
                lsp_types::Position::new(start.line, start.character),
                lsp_types::Position::new(end.line, end.character),
            );
            let severity = Some(match d.severity {
                DiagSeverity::Error => lsp_types::DiagnosticSeverity::ERROR,
                DiagSeverity::Hint => lsp_types::DiagnosticSeverity::HINT,
            });
            lsp_types::Diagnostic::new(range, severity, None, None, d.message.clone(), None, None)
        })
        .collect()
}

/// Return all top-level bindings in `file` as LSP `DocumentSymbol`s.
///
/// Each binding is emitted as a `VARIABLE` symbol whose range and selection
/// range are derived from the byte span of its RHS node. Bindings whose RHS
/// node has no recorded span are silently skipped. Returns an empty vec when
/// the file fails to parse or contains no spanned bindings.
pub fn document_symbols(
    db: &dyn salsa::Database,
    file: SourceFile,
) -> Vec<lsp_types::DocumentSymbol> {
    let li = line_index(db, file);
    let parsed = parse(db, file);
    let Some(module) = parsed.module(db) else {
        return vec![];
    };
    let mut syms = Vec::new();
    for (_, binding) in module.bindings() {
        let Some(span) = module.span_of(binding.rhs) else {
            continue;
        };
        let start = li.position(span.start);
        let end = li.position(span.end);
        let range = lsp_types::Range::new(
            lsp_types::Position::new(start.line, start.character),
            lsp_types::Position::new(end.line, end.character),
        );
        #[allow(deprecated)]
        syms.push(lsp_types::DocumentSymbol {
            name: module.resolve(binding.name).to_string(),
            kind: lsp_types::SymbolKind::VARIABLE,
            range,
            selection_range: range,
            detail: None,
            tags: None,
            deprecated: None,
            children: None,
        });
    }
    syms
}

/// Return all top-level bindings across every file in `fs` as LSP
/// [`SymbolInformation`] values, filtered by `query`.
///
/// `query` is matched case-insensitively against each binding name; an empty
/// `query` matches every binding.  Bindings whose RHS node has no recorded span
/// are silently skipped.  Files that fail to parse produce no symbols.
///
/// The `location.uri` is derived from the [`SourceFile::path`] stored in the
/// salsa database: paths that already look like `file://…` URIs are used as-is;
/// bare filesystem paths are prefixed with `file://`.
#[allow(deprecated)] // SymbolInformation.deprecated field is deprecated in LSP 3.16
pub fn workspace_symbols(
    db: &Database,
    fs: FileSet,
    query: &str,
) -> Vec<lsp_types::SymbolInformation> {
    let query_lower = query.to_lowercase();
    let mut syms = Vec::new();

    for &file in fs.files(db) {
        let path = file.path(db);
        let uri_str = if path.starts_with("file://") {
            path.clone()
        } else {
            crate::server::path_to_file_uri(&path)
        };
        let Ok(uri) = lsp_types::Uri::from_str(&uri_str) else {
            continue;
        };

        let li = line_index(db, file);
        let parsed = parse(db, file);
        let Some(module) = parsed.module(db) else {
            continue;
        };

        for (_, binding) in module.bindings() {
            let name = module.resolve(binding.name).to_string();
            if !query_lower.is_empty() && !name.to_lowercase().contains(&query_lower) {
                continue;
            }
            let Some(span) = module.span_of(binding.rhs) else {
                continue;
            };
            let start = li.position(span.start);
            let end = li.position(span.end);
            let range = lsp_types::Range::new(
                lsp_types::Position::new(start.line, start.character),
                lsp_types::Position::new(end.line, end.character),
            );
            let location = lsp_types::Location {
                uri: uri.clone(),
                range,
            };
            #[allow(deprecated)]
            syms.push(lsp_types::SymbolInformation {
                name,
                kind: lsp_types::SymbolKind::VARIABLE,
                tags: None,
                deprecated: None,
                location,
                container_name: None,
            });
        }
    }

    syms
}

/// Return a markdown hover string for the node at `byte_offset` in `file`, or
/// `None` when the cursor is not over any typed node.
///
/// The string is a fenced markdown block beginning with `**type:**` (and
/// optionally `**phase:**`) derived from the inferred [`flatppl_core::Type`]
/// and [`flatppl_core::Phase`] at the node. If analysis fails (parse error)
/// or no node spans the cursor, returns `None`.
pub fn hover(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
    index: &SpanIndex,
) -> Option<String> {
    let analyzed = analyze(db, file, fs, cats);
    let module = analyzed.module(db)?;
    let node_id = node_at_offset_indexed(index, byte_offset)?;
    let ty = module.type_of(node_id)?;
    let mut parts = vec![format!("**type:** `{}`", module.display_type(ty))];
    if let Some(phase) = module.phase_of(node_id) {
        parts.push(format!("**phase:** `{phase}`"));
    }
    if let Some(vs) = module.valueset_of(node_id) {
        parts.push(format!("**value-set:** `{vs}`"));
    }
    Some(parts.join("  \n"))
}

/// The resolved location of a definition: a file path (as stored in the salsa
/// database, i.e. a bare filesystem path or workspace-relative path) and a
/// half-open byte range `[start, end)` of the definition's RHS node.
pub struct DefLoc {
    pub path: String,
    pub start: u32,
    pub end: u32,
}

/// Return the definition location for the symbol under `byte_offset` in `file`,
/// or `None` when the cursor is not over a resolvable reference.
///
/// - [`RefNs::SelfMod`]: resolve to the binding in the same module whose RHS
///   span is returned.
/// - [`RefNs::Module`]: cross-file goto; find the alias binding's directive
///   (`load_module` or `standard_module`), resolve its path to a dep
///   [`SourceFile`], analyze it, and search its bindings by name string for the
///   binding named `r.name`. A `standard_module` reference has no workspace file
///   to navigate to, so `resolve_path` finds nothing and this returns `None`.
/// - [`RefNs::Local`]: not navigable from the surface; returns `None`.
pub fn goto_definition(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
    index: &SpanIndex,
) -> Option<DefLoc> {
    let analyzed = analyze(db, file, fs, cats);
    let module = analyzed.module(db)?;
    let node_id = node_at_offset_indexed(index, byte_offset)?;
    let Node::Ref(r) = module.node(node_id) else {
        return None;
    };
    match r.ns {
        RefNs::SelfMod => {
            let bid = module.binding_by_name(r.name)?;
            let rhs = module.binding(bid).rhs;
            let span = module.span_of(rhs)?;
            Some(DefLoc {
                path: file.path(db).clone(),
                start: span.start,
                end: span.end,
            })
        }
        RefNs::Module(alias) => {
            // Find the binding named `alias` in `module` and extract its
            // load_module/standard_module directive path from the call's first arg.
            let alias_name = module.resolve(alias);
            let (_, alias_binding) = module
                .bindings()
                .find(|(_, b)| module.resolve(b.name) == alias_name)?;
            let Node::Call(call) = module.node(alias_binding.rhs) else {
                return None;
            };
            let CallHead::Builtin(_) = call.head else {
                return None;
            };
            let directive_path = match call.args.first().map(|&a| module.node(a)) {
                Some(Node::Lit(Scalar::Str(s))) => s.to_string(),
                _ => return None,
            };
            // Resolve the directive path to a workspace SourceFile.
            let dep_file = resolve_path(db, file, &directive_path, fs)?;
            // Analyze the dependency (uses salsa cache).
            let dep_analyzed = analyze(db, dep_file, fs, cats);
            let dep_mod = dep_analyzed.module(db)?;
            // Cross-interner name match: compare by string, not by Symbol.
            let want_name = module.resolve(r.name);
            let (_, dep_binding) = dep_mod
                .bindings()
                .find(|(_, b)| dep_mod.resolve(b.name) == want_name)?;
            let span = dep_mod.span_of(dep_binding.rhs)?;
            Some(DefLoc {
                path: dep_file.path(db).clone(),
                start: span.start,
                end: span.end,
            })
        }
        RefNs::Local => None,
    }
}

/// Return inlay type hints for all bindings in `file` whose RHS span falls
/// within `[start_byte, end_byte)` and whose type is known.
///
/// For each qualifying binding a single `InlayHint` is emitted at the end of
/// the RHS span (i.e. after the last character of the expression). The label
/// is `: <type>` formatted via [`flatppl_core::Module::display_type`], the
/// same readable code-like notation used by hover (e.g. `integer`, `real`,
/// `measure<real> · normalized`). Bindings with no RHS span or no inferred
/// type are silently skipped.
pub fn inlay_hints(
    db: &Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    start_byte: u32,
    end_byte: u32,
) -> Vec<lsp_types::InlayHint> {
    let li = line_index(db, file);
    let analyzed = analyze(db, file, fs, cats);
    let Some(module) = analyzed.module(db) else {
        return vec![];
    };
    let mut hints = Vec::new();
    for (_, binding) in module.bindings() {
        let Some(span) = module.span_of(binding.rhs) else {
            continue;
        };
        // Filter to bindings whose RHS is fully contained in the requested
        // range. Containment (not intersection) is deliberate: it can never
        // place a hint outside the client's visible window. A binding whose RHS
        // straddles a range boundary gets no hint until the range covers it.
        if span.start < start_byte || span.end > end_byte {
            continue;
        }
        // The inlay carries the node's full inferred specification — type,
        // value-set, and phase — via `display_meta`, e.g.
        // `sqrt(x): real {nonnegreals, stochastic}`.
        let Some(meta) = module.display_meta(binding.rhs) else {
            continue;
        };
        let pos = li.position(span.end);
        hints.push(lsp_types::InlayHint {
            position: lsp_types::Position::new(pos.line, pos.character),
            label: lsp_types::InlayHintLabel::String(format!(": {meta}")),
            kind: Some(lsp_types::InlayHintKind::TYPE),
            text_edits: None,
            tooltip: None,
            padding_left: Some(false),
            padding_right: Some(false),
            data: None,
        });
    }
    hints
}

/// Return completion items for the given position in `file`.
///
/// `ctx: crate::server::CompletionContext` determines the completion mode:
///
/// **`Member(alias)`**: Finds the binding named `alias` in the analyzed module,
/// reads its `standard_module` call's module name, then lists all matching
/// binding names from the built-in catalogue and any external catalogues.
/// Returns only those items (kind `FUNCTION`).
///
/// **`AfterTilde`**: Returns the full general set (keywords, built-in base
/// names, external catalogue base names, in-scope bindings), with `sort_text`
/// buckets that float catalogue distributions to the top (`"0_<label>"` for
/// distributions, `"1_<label>"` for others). Nothing is hidden.
///
/// **`Other`**: Returns the full general set without setting `sort_text`, so
/// the client uses its default ordering. The set includes language keywords,
/// built-in base names, external catalogue base names, and in-scope binding
/// names.
///
/// When `lead_space` is true the cursor sits tight against a `~`/`=` operator
/// (no space typed yet); every returned item's `insert_text` gains a leading
/// space so accepting one yields `x ~ Normal` rather than `x ~Normal`. The
/// display `label` stays clean. Member completion never sees this (it returns
/// before the leading-space pass).
///
/// All items are deduplicated by label.
pub(crate) fn completion(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    ctx: crate::server::CompletionContext,
    lead_space: bool,
) -> Vec<lsp_types::CompletionItem> {
    use lsp_types::{CompletionItem, CompletionItemKind};

    // Obtain the external catalogues from the tracked `parsed_catalogues` query
    // (parsed once per `Catalogues` revision; failures silently skipped — the
    // server already emits diagnostics for them via `analyze`).
    let external_cats = parsed_catalogues(db, cats);

    if let crate::server::CompletionContext::Member(alias) = ctx {
        let builtin = flatppl_infer::builtin_catalogue();
        // Member completion: find the alias binding, read its standard_module name,
        // list that module's bindings from built-in + external catalogues.
        let module_name = find_standard_module_name(db, file, fs, cats, &alias);
        let mut items: Vec<CompletionItem> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        if let Some(mod_name) = module_name {
            // Built-in catalogue first.
            if let Some(names) = builtin.module_binding_names(&mod_name) {
                for name in names {
                    if seen.insert(name.to_string()) {
                        items.push(CompletionItem {
                            label: name.to_string(),
                            kind: Some(CompletionItemKind::FUNCTION),
                            ..Default::default()
                        });
                    }
                }
            }
            // External catalogues in order.
            for ext in external_cats.as_slice() {
                if let Some(names) = ext.module_binding_names(&mod_name) {
                    for name in names {
                        if seen.insert(name.to_string()) {
                            items.push(CompletionItem {
                                label: name.to_string(),
                                kind: Some(CompletionItemKind::FUNCTION),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }
        return items;
    }

    // General completion.
    //
    // The keyword + built-in base-name set is process-constant: it never depends
    // on the file, file-set, or catalogue inputs. Build it exactly once and reuse
    // the cached vector on every subsequent call.
    let static_items = static_completion_items();

    // Start from the cached static set; pre-populate `seen` from its labels so the
    // per-call external/in-scope passes do not re-add a static item.
    let mut items: Vec<CompletionItem> = static_items.clone();
    let mut seen: std::collections::HashSet<String> =
        static_items.iter().map(|i| i.label.clone()).collect();

    // External catalogue base names (per-call: depend on the `cats` input).
    for ext in external_cats.as_slice() {
        for name in ext.base_names() {
            if seen.insert(name.to_string()) {
                items.push(CompletionItem {
                    label: name.to_string(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    ..Default::default()
                });
            }
        }
    }

    // In-scope binding names from the analyzed module.
    let analyzed = analyze(db, file, fs, cats);
    if let Some(module) = analyzed.module(db) {
        for (_, binding) in module.bindings() {
            let name = module.resolve(binding.name).to_string();
            if seen.insert(name.clone()) {
                items.push(CompletionItem {
                    label: name,
                    kind: Some(CompletionItemKind::VARIABLE),
                    ..Default::default()
                });
            }
        }
    }

    // AfterTilde: float distributions to the top via sortText buckets, hiding
    // nothing. Distributions (catalogue Sig::Distribution, built-in or external)
    // get bucket "0_", everything else "1_". `Other` leaves sortText unset so
    // the client uses its default ordering.
    if matches!(ctx, crate::server::CompletionContext::AfterTilde) {
        let builtin = flatppl_infer::builtin_catalogue();
        for it in &mut items {
            let is_dist = builtin.base_is_distribution(&it.label)
                || external_cats
                    .as_slice()
                    .iter()
                    .any(|c| c.base_is_distribution(&it.label));
            let bucket = if is_dist { '0' } else { '1' };
            it.sort_text = Some(format!("{bucket}_{}", it.label));
        }
    }

    // Tight against a `~`/`=` operator: prepend a space to the inserted text so
    // accepting an item produces `x ~ Normal` (the idiomatic spacing) without
    // the user typing the space. The `label` is left clean for display and
    // filtering. Skipped once a space already separates operator and cursor, so
    // an existing space is never doubled.
    //
    // `filter_text` is intentionally left unset: clients default it to `label`
    // (LSP §3.3.1), so typed-prefix matching uses the clean label, not the
    // space-prefixed `insert_text`.
    if lead_space {
        for it in &mut items {
            it.insert_text = Some(format!(" {}", it.label));
        }
    }

    items
}

/// Build the process-constant portion of the general completion set: the
/// language keywords and every built-in base name, deduplicated by label.
///
/// This never depends on any request input, so it is built once and reused.
fn build_static_completion_items() -> Vec<lsp_types::CompletionItem> {
    #[cfg(test)]
    STATIC_COMPLETION_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    use lsp_types::{CompletionItem, CompletionItemKind};
    let mut items: Vec<CompletionItem> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for kw in &["self", "base", "in", "all", "only", "true", "false"] {
        if seen.insert((*kw).to_string()) {
            items.push(CompletionItem {
                label: (*kw).to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                ..Default::default()
            });
        }
    }
    let builtin = flatppl_infer::builtin_catalogue();
    for name in builtin.base_names() {
        if seen.insert(name.to_string()) {
            items.push(CompletionItem {
                label: name.to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                ..Default::default()
            });
        }
    }
    items
}

/// Return the cached static completion set, building it on first use.
///
/// In production the cache is a process-global `OnceLock`: the set is built once
/// for the life of the process. Under `cfg(test)` the cache is resettable (via
/// [`reset_static_completion_items`]) so the build-once contract can be asserted
/// deterministically regardless of which test populates the cache first.
#[cfg(not(test))]
fn static_completion_items() -> Vec<lsp_types::CompletionItem> {
    use std::sync::OnceLock;
    static STATIC_ITEMS: OnceLock<Vec<lsp_types::CompletionItem>> = OnceLock::new();
    STATIC_ITEMS
        .get_or_init(build_static_completion_items)
        .clone()
}

#[cfg(test)]
fn static_completion_items() -> Vec<lsp_types::CompletionItem> {
    let mut guard = STATIC_ITEMS_TEST_CACHE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(build_static_completion_items());
    }
    guard.as_ref().unwrap().clone()
}

#[cfg(test)]
static STATIC_ITEMS_TEST_CACHE: std::sync::Mutex<Option<Vec<lsp_types::CompletionItem>>> =
    std::sync::Mutex::new(None);

/// Drop the test-only static-completion cache so the next call rebuilds it.
#[cfg(test)]
fn reset_static_completion_items() {
    *STATIC_ITEMS_TEST_CACHE.lock().unwrap() = None;
}

/// Attempt to resolve the `standard_module` module name for the binding named
/// `alias` in `file`. Returns `None` when the binding is absent, is not a
/// `standard_module` call, or the first argument is not a string literal.
///
/// When the file fails to parse (e.g. because the cursor is mid-expression on
/// the last line), this function retries with the last newline-delimited line
/// stripped, allowing completion to work even while the user is still typing.
fn find_standard_module_name(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    alias: &str,
) -> Option<String> {
    // First attempt: analyze the file as-is.
    let analyzed = analyze(db, file, fs, cats);
    if let Some(module) = analyzed.module(db) {
        return extract_standard_module_name(module, alias);
    }

    // Second attempt: the file may fail to parse because the cursor is in the
    // middle of an incomplete expression (e.g. `x = alias.`). Strip the last
    // non-empty line and try parsing the remainder directly.
    let text = file.text(db);
    let repaired = strip_last_nonempty_line(text);
    if repaired.is_empty() || repaired == text {
        return None;
    }
    let module = flatppl_syntax::parse(repaired).ok()?;
    extract_standard_module_name(&module, alias)
}

/// Extract the `standard_module` first-arg string from the binding named `alias`.
fn extract_standard_module_name(module: &flatppl_core::Module, alias: &str) -> Option<String> {
    // Find the binding whose name equals `alias` (string compare across the
    // module's interner).
    let (_, binding) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == alias)?;
    let Node::Call(call) = module.node(binding.rhs) else {
        return None;
    };
    let CallHead::Builtin(head_sym) = call.head else {
        return None;
    };
    if module.resolve(head_sym) != "standard_module" {
        return None;
    }
    // First arg is the module name string.
    match call.args.first().map(|&a| module.node(a)) {
        Some(Node::Lit(Scalar::Str(s))) => Some(s.to_string()),
        _ => None,
    }
}

/// Strip the last non-empty line from `text` (newline-separated), returning
/// the remainder. If `text` has no newline or is entirely blank, returns an
/// empty string.
fn strip_last_nonempty_line(text: &str) -> &str {
    // Walk backwards, skipping trailing whitespace / empty lines, then find the
    // newline that ends the preceding line.
    let bytes = text.as_bytes();
    let mut end = bytes.len();
    // Skip trailing blank content.
    while end > 0 && (bytes[end - 1] == b'\n' || bytes[end - 1] == b'\r') {
        end -= 1;
    }
    // Now find the last newline before `end`.
    let nl = bytes[..end].iter().rposition(|&b| b == b'\n' || b == b'\r');
    match nl {
        Some(pos) => &text[..=pos],
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the general-completion tests, which share the resettable
    /// static-items cache and its build counter. Without this guard they could
    /// interleave (one test's `reset` racing another's `get_or_build`) and skew
    /// the build count. The member-completion path does not touch the cache, so
    /// it intentionally does not take this lock.
    static COMPLETION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ── parse error tests ────────────────────────────────────────────────────

    #[test]
    fn from_parse_error_covers_span() {
        let src = "x = (((";
        let e = flatppl_syntax::parse(src).expect_err("must fail to parse");
        let diag = LspDiag::from_parse_error(&e);
        let src_len = src.len() as u32;
        // The range must fit inside the source and the message must be non-empty.
        assert!(
            diag.start <= src_len,
            "start={} out of source length {src_len}",
            diag.start
        );
        assert!(
            diag.end <= src_len,
            "end={} out of source length {src_len}",
            diag.end
        );
        assert!(
            !diag.message.is_empty(),
            "diagnostic message must not be empty"
        );
        assert_eq!(diag.severity, DiagSeverity::Error);
    }

    // ── inference diagnostic tests ───────────────────────────────────────────

    /// Build a minimal module (hand-constructed IR, not parsed) containing a
    /// dangling `%ref self.nope` anchored at bytes 8..12. When inferred, the
    /// engine emits an unresolved-reference error anchored to that ref node.
    /// After mapping with `from_infer`, the resulting `LspDiag` should carry
    /// the same byte range.
    #[test]
    fn from_infer_anchors_to_node_span() {
        use flatppl_core::{Binding, Module, Node, Ref, RefNs, Span};

        let mut m = Module::default();
        let nope_sym = m.intern("nope");
        let y_sym = m.intern("y");

        let ref_span = Span { start: 8, end: 12 };
        let ref_id = m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: nope_sym,
        }));
        m.set_span(ref_id, ref_span);
        m.add_binding(Binding {
            name: y_sym,
            rhs: ref_id,
            doc: None,
            public: true,
            synthetic: false,
        });

        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Type);
        let anchored = diags
            .iter()
            .find(|d| d.node.is_some() && d.severity == flatppl_infer::Severity::Error)
            .expect("expected an anchored Error diagnostic from the dangling ref");

        let lsp_diag = LspDiag::from_infer(anchored, &m);
        assert_eq!(lsp_diag.start, ref_span.start, "start must match ref_span");
        assert_eq!(lsp_diag.end, ref_span.end, "end must match ref_span");
        assert_eq!(lsp_diag.severity, DiagSeverity::Error);
    }

    #[test]
    fn note_maps_to_hint() {
        // An unrecognised op (`foo(1)`) produces a `Note` (honest %deferred gap).
        let mut m = flatppl_syntax::parse("x = foo(1)").expect("parses");
        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Type);
        let note_diag = diags
            .iter()
            .find(|d| d.severity == flatppl_infer::Severity::Note)
            .expect("expected at least one Note diagnostic for unknown op foo");

        let lsp_diag = LspDiag::from_infer(note_diag, &m);
        assert_eq!(
            lsp_diag.severity,
            DiagSeverity::Hint,
            "Note must map to Hint; got {:?}",
            lsp_diag.severity
        );
    }

    // ── diagnostics() capability tests ──────────────────────────────────────

    use super::workspace_symbols;
    use crate::db::{Catalogues, Database, FileSet, SourceFile};

    #[test]
    fn diagnostics_maps_byte_range_to_lsp_range() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = (((".to_string());
        let fs = FileSet::new(&db, vec![f]);
        let ds = diagnostics(&db, f, fs, cats);
        assert!(
            !ds.is_empty(),
            "parse error must produce at least one diagnostic"
        );
        assert!(
            ds[0].severity.is_some(),
            "severity must be populated; got {:?}",
            ds[0].severity
        );
        assert_eq!(
            ds[0].severity,
            Some(lsp_types::DiagnosticSeverity::ERROR),
            "parse errors must map to ERROR severity"
        );
    }

    // ── hover() capability tests ─────────────────────────────────────────────

    #[test]
    fn hover_reports_type_at_cursor() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        // "x = add(1, 2)" — offset 0 is 'x' (the binding name node), which
        // should carry the inferred type of the whole expression.
        let src = "x = add(1, 2)";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        // Try a few offsets; at least one inside the expression should return Some.
        let offsets: &[u32] = &[0, 4, 9];
        let index = crate::queries::node_span_index(&db, f, fs, cats);
        let found = offsets
            .iter()
            .find_map(|&off| hover(&db, f, fs, cats, off, &index));
        let s = found.expect("at least one offset inside 'x = add(1, 2)' must yield hover info");
        assert!(
            s.to_lowercase().contains("type"),
            "hover string must mention 'type'; got: {s:?}"
        );
    }

    #[test]
    fn hover_none_off_any_node() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = add(1, 2)".to_string());
        let fs = FileSet::new(&db, vec![f]);
        let index = crate::queries::node_span_index(&db, f, fs, cats);
        assert!(
            hover(&db, f, fs, cats, 9999, &index).is_none(),
            "offset past end of source must yield None"
        );
    }

    // ── value-set in hover ───────────────────────────────────────────────────

    /// `c = elementof(reals)` — the `elementof(reals)` call node carries the
    /// value-set `Reals` after inference. Hovering anywhere inside the call
    /// expression must produce a hover string that includes "value-set".
    #[test]
    fn hover_reports_value_set() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        // "c = elementof(reals)"
        //  0123456789...
        // 'e' of 'elementof' is at offset 4; the call spans offsets 4..20.
        let src = "c = elementof(reals)";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        // Try several offsets inside the elementof(...) call; at least one must
        // report a value-set (the infer engine annotates the call node).
        let offsets: &[u32] = &[4, 9, 14, 18];
        let index = crate::queries::node_span_index(&db, f, fs, cats);
        let found = offsets
            .iter()
            .find_map(|&off| hover(&db, f, fs, cats, off, &index));
        let s = found.expect("at least one offset inside 'elementof(reals)' must yield hover info");
        assert!(
            s.contains("value-set"),
            "hover string must contain 'value-set'; got: {s:?}"
        );
        // The value-set for elementof(reals) renders in the spec surface
        // vocabulary as `reals` (lowercase), not the Debug `Reals`.
        assert!(
            s.contains("reals"),
            "hover string must mention 'reals' for elementof(reals); got: {s:?}"
        );
    }

    // ── document_symbols() capability tests ─────────────────────────────────

    #[test]
    fn document_symbols_lists_bindings() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let f = SourceFile::new(
            &db,
            "m.flatppl".to_string(),
            "x = 1\ny = add(x, 2)".to_string(),
        );
        let fs = FileSet::new(&db, vec![f]);
        let _ = (fs, cats);
        let syms = document_symbols(&db, f);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"x") && names.contains(&"y"),
            "document_symbols must include both 'x' and 'y'; got: {names:?}"
        );
    }

    #[test]
    fn document_symbols_does_not_run_inference() {
        use crate::queries::ANALYZE_RUNS;
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let f = SourceFile::new(
            &db,
            "m.flatppl".to_string(),
            "x = 1\ny = add(x, 2)".to_string(),
        );
        let fs = FileSet::new(&db, vec![f]);
        let _ = (fs, cats);
        // Reset the counter, call document_symbols, then verify no analyze ran.
        ANALYZE_RUNS.with(|c| c.set(0));
        let syms = document_symbols(&db, f);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"x") && names.contains(&"y"),
            "document_symbols must still include 'x' and 'y' after switching to parse; got: {names:?}"
        );
        let runs = ANALYZE_RUNS.with(|c| c.get());
        assert_eq!(
            runs, 0,
            "document_symbols must not run inference (ANALYZE_RUNS should be 0, got {runs})"
        );
    }

    #[test]
    fn hover_none_on_parse_error() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let f = SourceFile::new(&db, "bad.flatppl".to_string(), "x = (((".to_string());
        let fs = FileSet::new(&db, vec![f]);
        // Parse failure means no module, so hover must always return None.
        let index = crate::queries::node_span_index(&db, f, fs, cats);
        assert!(
            hover(&db, f, fs, cats, 0, &index).is_none(),
            "hover on a parse-error file must return None"
        );
    }

    // ── workspace_symbols() capability tests ────────────────────────────────

    #[test]
    fn workspace_symbols_span_files_and_filter() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let a = SourceFile::new(&db, "a.flatppl".to_string(), "alpha = 1".to_string());
        let b = SourceFile::new(&db, "b.flatppl".to_string(), "beta = 2".to_string());
        let fs = FileSet::new(&db, vec![a, b]);
        let _ = cats;
        let all = workspace_symbols(&db, fs, "");
        assert!(
            all.iter().any(|s| s.name == "alpha") && all.iter().any(|s| s.name == "beta"),
            "workspace_symbols must include 'alpha' and 'beta'; got: {:?}",
            all.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        let filtered = workspace_symbols(&db, fs, "alph");
        assert!(
            !filtered.is_empty()
                && filtered
                    .iter()
                    .all(|s| s.name.to_lowercase().contains("alph")),
            "filtered workspace_symbols must contain only 'alph' matches; got: {:?}",
            filtered.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    // ── inlay_hints() capability tests ──────────────────────────────────────

    #[test]
    fn inlay_hints_label_binding_types() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        // `add(1, 2)` infers to integer at fixed phase. The hint renders the
        // full spec via `display_meta`: type + phase (the integer value-set is
        // the natural extent of `integer`, so it is omitted as redundant).
        let src = "x = add(1, 2)";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let hints = inlay_hints(&db, f, fs, cats, 0, src.len() as u32);
        let labels: Vec<String> = hints
            .iter()
            .filter_map(|h| match &h.label {
                lsp_types::InlayHintLabel::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(
            labels.iter().any(|l| l == ": integer {fixed}"),
            "inlay label must be the readable spec `: integer {{fixed}}`; got: {labels:?}"
        );
        assert!(
            !labels
                .iter()
                .any(|l| l.contains("Scalar(") || l.contains("Symbol(")),
            "inlay labels must not contain Debug struct syntax; got: {labels:?}"
        );
    }

    /// An iid count derived from a fixed `lengthof` of a literal array must show
    /// a CONCRETE dim in the inlay hint, not `?`. This is what `Level::Shape`
    /// (vs `Normalization`) buys: shape resolution folds `lengthof([…5…])` → 5,
    /// so `y ~ iid(Poisson(λ), n)` is `…[5]`, not `…[?]`.
    #[test]
    fn inlay_hints_resolve_iid_count_dim() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let src = "counts = [2, 3, 7, 6, 4]\n\
                   n = lengthof(counts)\n\
                   lambda ~ Gamma(2.0, 1.0)\n\
                   y ~ iid(Poisson(lambda), n)\n";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let hints = inlay_hints(&db, f, fs, cats, 0, src.len() as u32);
        let labels: Vec<String> = hints
            .iter()
            .filter_map(|h| match &h.label {
                lsp_types::InlayHintLabel::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(
            labels.iter().any(|l| l.contains("[5]")),
            "y's iid count should resolve to dim 5 ([5], not [?]); got: {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l.contains("[?]")),
            "no shape should be left dynamic for this fully-fixed model; got: {labels:?}"
        );
    }

    /// A tightened value-set (here `sqrt` of a real → `nonnegreals`) must
    /// surface in the inlay's spec brace, since it is tighter than `real`'s
    /// natural extent and is what distinguishes the binding's domain.
    #[test]
    fn inlay_hints_show_tightened_value_set() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let src = "x ~ Normal(0.0, 1.0)\nr = sqrt(x)\n";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let hints = inlay_hints(&db, f, fs, cats, 0, src.len() as u32);
        let labels: Vec<String> = hints
            .iter()
            .filter_map(|h| match &h.label {
                lsp_types::InlayHintLabel::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(
            labels.iter().any(|l| l.contains("nonnegreals")),
            "sqrt's nonnegreals value-set must show in the inlay spec; got: {labels:?}"
        );
    }

    // ── goto_definition() capability tests ──────────────────────────────────

    #[test]
    fn goto_same_module_binding() {
        use crate::queries::node_span_index;
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let src = "x = 1\ny = add(x, 2)";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let off = "x = 1\ny = add(".len() as u32; // on the `x` argument
        let index = node_span_index(&db, f, fs, cats);
        let loc = goto_definition(&db, f, fs, cats, off, &index).expect("definition");
        assert_eq!(loc.path, "m.flatppl");
        // x's binding rhs is the literal `1` at byte 4
        assert!(loc.start <= 4 && 4 < loc.end);
    }

    #[test]
    fn goto_cross_file_load_module_binding() {
        use crate::queries::node_span_index;
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "shifted = 1.0".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = h.shifted".to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model]);
        let off = "h = load_module(\"helpers.flatppl\")\nv = h.".len() as u32; // on `shifted`
        let index = node_span_index(&db, model, fs, cats);
        let loc = goto_definition(&db, model, fs, cats, off, &index).expect("cross-file def");
        assert_eq!(loc.path, "helpers.flatppl");
    }

    // ── completion() capability tests ────────────────────────────────────────

    #[test]
    fn after_tilde_sorts_distributions_first_without_hiding() {
        use crate::db::{Catalogues, Database, FileSet, SourceFile};
        use crate::server::CompletionContext;
        let db = Database::default();
        let file = SourceFile::new(&db, "m.flatppl".to_string(), "x ~ ".to_string());
        let fs = FileSet::new(&db, vec![file]);
        let cats = Catalogues::new(&db, Vec::<String>::new());

        let items = completion(&db, file, fs, cats, CompletionContext::AfterTilde, false);

        let normal = items
            .iter()
            .find(|i| i.label == "Normal")
            .expect("Normal present");
        let kw = items
            .iter()
            .find(|i| i.label == "self")
            .expect("keyword present");
        let n_sort = normal
            .sort_text
            .as_deref()
            .expect("distribution has sort_text");
        let kw_sort = kw.sort_text.as_deref().expect("keyword has sort_text");
        assert!(
            n_sort < kw_sort,
            "distribution must sort before keyword: {n_sort} !< {kw_sort}"
        );
        // Nothing hidden: the keyword is still in the list.
        assert!(
            items.iter().any(|i| i.label == "self"),
            "keyword must still be present"
        );
    }

    #[test]
    fn other_context_leaves_default_order() {
        use crate::db::{Catalogues, Database, FileSet, SourceFile};
        use crate::server::CompletionContext;
        let db = Database::default();
        let file = SourceFile::new(&db, "m.flatppl".to_string(), "x = ".to_string());
        let fs = FileSet::new(&db, vec![file]);
        let cats = Catalogues::new(&db, Vec::<String>::new());

        let items = completion(&db, file, fs, cats, CompletionContext::Other, false);
        // Full set still present; no AfterTilde bias applied.
        assert!(items.iter().any(|i| i.label == "Normal"));
        assert!(
            items.iter().all(|i| i.sort_text.is_none()),
            "Other context sets no sort_text"
        );
        // lead_space == false: inserted text is untouched (no leading space).
        assert!(
            items.iter().all(|i| i.insert_text.is_none()),
            "lead_space=false must leave insert_text unset"
        );
    }

    #[test]
    fn lead_space_prepends_space_to_insert_text_without_touching_label() {
        use crate::db::{Catalogues, Database, FileSet, SourceFile};
        use crate::server::CompletionContext;
        let db = Database::default();
        // "mu =" — cursor tight against `=`, value position (Other).
        let file = SourceFile::new(&db, "m.flatppl".to_string(), "mu =".to_string());
        let fs = FileSet::new(&db, vec![file]);
        let cats = Catalogues::new(&db, Vec::<String>::new());

        let items = completion(&db, file, fs, cats, CompletionContext::Other, true);
        let normal = items
            .iter()
            .find(|i| i.label == "Normal")
            .expect("Normal present");
        // Display label stays clean; only the inserted text gains the space.
        assert_eq!(normal.label, "Normal");
        assert_eq!(
            normal.insert_text.as_deref(),
            Some(" Normal"),
            "tight lead_space must prepend a single leading space to insert_text"
        );
        assert!(
            items
                .iter()
                .all(|i| i.insert_text.as_deref() == Some(&format!(" {}", i.label))),
            "every item must carry a leading-space insert_text when lead_space=true"
        );
    }

    #[test]
    fn after_tilde_with_lead_space_keeps_sort_and_adds_space() {
        use crate::db::{Catalogues, Database, FileSet, SourceFile};
        use crate::server::CompletionContext;
        let db = Database::default();
        // "x ~" — tight against `~`, distribution position.
        let file = SourceFile::new(&db, "m.flatppl".to_string(), "x ~".to_string());
        let fs = FileSet::new(&db, vec![file]);
        let cats = Catalogues::new(&db, Vec::<String>::new());

        let items = completion(&db, file, fs, cats, CompletionContext::AfterTilde, true);
        let normal = items
            .iter()
            .find(|i| i.label == "Normal")
            .expect("Normal present");
        // Distribution bias and leading-space insert coexist.
        assert_eq!(normal.sort_text.as_deref(), Some("0_Normal"));
        assert_eq!(normal.insert_text.as_deref(), Some(" Normal"));
    }

    #[test]
    fn completion_includes_keywords_builtins_and_scope() {
        let _guard = COMPLETION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let src = "alpha = 1\nbeta = 2";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let items = completion(
            &db,
            f,
            fs,
            cats,
            crate::server::CompletionContext::Other,
            false,
        );
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        // Built-in base distribution must be present.
        assert!(
            labels.contains(&"Normal"),
            "completion must include built-in base name 'Normal'; got: {labels:?}"
        );
        // In-scope bindings must be present.
        assert!(
            labels.contains(&"alpha"),
            "completion must include in-scope binding 'alpha'; got: {labels:?}"
        );
        // At least one keyword must be present.
        assert!(
            labels.iter().any(|l| *l == "self" || *l == "base"),
            "completion must include a keyword such as 'self' or 'base'; got: {labels:?}"
        );
    }

    #[test]
    fn completion_after_module_alias_dot_lists_module_bindings() {
        let db = Database::default();
        let ron = r#"Catalogue(base: [], modules: [Module(name:"myext",version:"0.1",bindings:[Binding(name:"MyDist", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized))])])"#;
        let cats = Catalogues::new(&db, vec![ron.to_string()]);
        let src = "e = standard_module(\"myext\",\"0.1\")\nx = e.";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let items = completion(
            &db,
            f,
            fs,
            cats,
            crate::server::CompletionContext::Member("e".to_string()),
            false,
        );
        assert!(
            items.iter().any(|i| i.label == "MyDist"),
            "member completion must list 'MyDist' from the external module; got: {:?}",
            items.iter().map(|i| &i.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn static_completion_items_built_once() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = COMPLETION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Drop any cache another test may have populated, then reset the counter,
        // so the build-once contract is measured deterministically here.
        reset_static_completion_items();
        STATIC_COMPLETION_BUILDS.store(0, Relaxed);

        let db1 = Database::default();
        let cats1 = Catalogues::new(&db1, vec![]);
        let f1 = SourceFile::new(&db1, "a.flatppl".to_string(), "alpha = 1".to_string());
        let fs1 = FileSet::new(&db1, vec![f1]);

        let db2 = Database::default();
        let cats2 = Catalogues::new(&db2, vec![]);
        let f2 = SourceFile::new(&db2, "b.flatppl".to_string(), "beta = 2".to_string());
        let fs2 = FileSet::new(&db2, vec![f2]);

        let items1 = completion(
            &db1,
            f1,
            fs1,
            cats1,
            crate::server::CompletionContext::Other,
            false,
        );
        let items2 = completion(
            &db2,
            f2,
            fs2,
            cats2,
            crate::server::CompletionContext::Other,
            false,
        );
        let items3 = completion(
            &db1,
            f1,
            fs1,
            cats1,
            crate::server::CompletionContext::Other,
            false,
        );

        // The static portion must have been built exactly once across all calls.
        let builds = STATIC_COMPLETION_BUILDS.load(Relaxed);
        assert_eq!(
            builds, 1,
            "static completion items must be built once, not per call; got {builds} builds"
        );

        // The returned sets must still include the expected items.
        let labels1: Vec<&str> = items1.iter().map(|i| i.label.as_str()).collect();
        assert!(
            labels1.contains(&"Normal"),
            "completion must include built-in 'Normal'; got: {labels1:?}"
        );
        assert!(
            labels1.contains(&"alpha"),
            "completion must include in-scope 'alpha'; got: {labels1:?}"
        );
        assert!(
            labels1.iter().any(|l| *l == "self" || *l == "base"),
            "completion must include keywords; got: {labels1:?}"
        );
        let labels2: Vec<&str> = items2.iter().map(|i| i.label.as_str()).collect();
        assert!(
            labels2.contains(&"beta"),
            "second db completion must include 'beta'; got: {labels2:?}"
        );
        let _ = items3; // third call must reuse the cache
    }
}
