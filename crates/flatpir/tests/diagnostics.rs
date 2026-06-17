//! Reader diagnostics carry source spans (spec §11; Phase-1 residue).
//!
//! Structurally-invalid-but-lexically-valid FlatPIR must report its errors with
//! a source line and byte span, so the CLI can render a caret under the
//! offending form — the same treatment the S-expression lexer already gives to
//! unbalanced parens and bad string escapes. Before this, the reader's
//! structural/semantic errors were all unpositioned (`line 0`, no span) and
//! degraded to a bare one-line message.

use flatppl_flatpir::read;

/// Slice the reported byte span out of the source, asserting it is in bounds.
fn span_text(src: &str, span: Option<(u32, u32)>) -> &str {
    let (start, end) = span.expect("a structural reader error must carry a byte span");
    let (start, end) = (start as usize, end as usize);
    assert!(
        start < end && end <= src.len(),
        "span in bounds: {start}..{end}"
    );
    &src[start..end]
}

#[test]
fn arity_error_points_at_the_offending_form() {
    // `(%bind)` is a well-formed S-expression but malformed FlatPIR.
    let src = "(%module\n  (%bind))\n";
    let err = read(src).unwrap_err();

    assert_eq!(err.line, 2, "error should be on the line of `(%bind)`");
    assert_eq!(
        span_text(src, err.span),
        "%bind",
        "span covers the offending form"
    );
}

#[test]
fn unknown_call_head_is_localized() {
    let src = "(%module\n  (%bind x (%bogus 1)))\n";
    let err = read(src).unwrap_err();

    assert!(err.line >= 1, "error carries a line");
    assert!(
        span_text(src, err.span).contains("%bogus"),
        "span should cover the bogus head"
    );
}

#[test]
fn unknown_type_in_meta_is_localized() {
    let src = "(%module\n  (%bind x (add (%meta %bogustype %fixed reals) 1 2)))\n";
    let err = read(src).unwrap_err();

    assert!(
        span_text(src, err.span).contains("%bogustype"),
        "span should cover the unknown type"
    );
}
