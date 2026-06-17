//! Capability helpers: the owned diagnostic type the queries carry, mapping
//! from parse errors and inference diagnostics, and the public `diagnostics`
//! and `hover` functions that convert internal types to LSP protocol values.

use crate::db::{Catalogues, FileSet, SourceFile};
use crate::line_index::LineIndex;
use crate::queries::analyze;

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
/// [`LineIndex`] so that LSP clients receive correctly positioned ranges.
pub fn diagnostics(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
) -> Vec<lsp_types::Diagnostic> {
    let analyzed = analyze(db, file, fs, cats);
    let text = file.text(db);
    let li = LineIndex::new(text);
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
) -> Option<String> {
    let analyzed = analyze(db, file, fs, cats);
    let module = analyzed.module(db)?;
    let node_id = module.node_at_offset(byte_offset)?;
    let ty = module.type_of(node_id)?;
    let mut parts = vec![format!("**type:** `{ty:?}`")];
    if let Some(phase) = module.phase_of(node_id) {
        parts.push(format!("**phase:** `{phase:?}`"));
    }
    if let Some(vs) = module.valueset_of(node_id) {
        parts.push(format!("**value-set:** `{vs:?}`"));
    }
    Some(parts.join("  \n"))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let found = offsets.iter().find_map(|&off| hover(&db, f, fs, cats, off));
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
        assert!(
            hover(&db, f, fs, cats, 9999).is_none(),
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
        let found = offsets.iter().find_map(|&off| hover(&db, f, fs, cats, off));
        let s = found.expect("at least one offset inside 'elementof(reals)' must yield hover info");
        assert!(
            s.contains("value-set"),
            "hover string must contain 'value-set'; got: {s:?}"
        );
        // The value-set for elementof(reals) should be Reals.
        assert!(
            s.contains("Reals"),
            "hover string must mention 'Reals' for elementof(reals); got: {s:?}"
        );
    }

    #[test]
    fn hover_none_on_parse_error() {
        let db = Database::default();
        let cats = Catalogues::new(&db, vec![]);
        let f = SourceFile::new(&db, "bad.flatppl".to_string(), "x = (((".to_string());
        let fs = FileSet::new(&db, vec![f]);
        // Parse failure means no module, so hover must always return None.
        assert!(
            hover(&db, f, fs, cats, 0).is_none(),
            "hover on a parse-error file must return None"
        );
    }
}
