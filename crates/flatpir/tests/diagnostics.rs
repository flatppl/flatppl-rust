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
fn repeated_public_interface_is_localized() {
    let src = "(%module\n  (%public x)\n  (%public y)\n  (%bind x 1)\n  (%bind y 2))";
    let err = read(src).unwrap_err();
    assert_eq!(span_text(src, err.span), "(%public y)");
}

#[test]
fn duplicate_binding_is_localized() {
    let src = "(%module\n  (%bind x 1)\n  (%bind x 2))";
    let err = read(src).unwrap_err();
    assert_eq!(span_text(src, err.span), "(%bind x 2)");
    assert!(err.message.contains("duplicate binding `x`"));
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

// §11 "Literal values": a scalar literal carries no leading sign; a negated
// literal is the call `(neg 1.0)`. A signed atom is what the writer never
// emits, so the reader must refuse it rather than silently accept it.

#[test]
fn signed_real_atom_is_refused() {
    let src = "(%module\n  (%bind x -1.0))\n";
    let err = read(src).unwrap_err();

    assert!(
        span_text(src, err.span).contains("-1.0"),
        "span should cover the signed atom"
    );
    assert!(
        err.message.contains("no leading sign") && err.message.contains("(neg 1.0)"),
        "error should quote §11's rule: {}",
        err.message
    );
}

#[test]
fn signed_integer_atom_is_refused() {
    let src = "(%module\n  (%bind x -42))\n";
    let err = read(src).unwrap_err();

    assert!(
        span_text(src, err.span).contains("-42"),
        "span should cover the signed atom"
    );
    assert!(
        err.message.contains("no leading sign") && err.message.contains("(neg 42)"),
        "error should quote §11's rule: {}",
        err.message
    );
}

#[test]
fn signed_atom_in_nested_position_is_refused() {
    // The signed atom is not the top-level bind expression but an argument
    // buried inside a call — the refusal must still fire and localize on it.
    let src = "(%module\n  (%bind x (add 1.0 (mul 2.0 -3.0))))\n";
    let err = read(src).unwrap_err();

    assert!(
        span_text(src, err.span).contains("-3.0"),
        "span should cover the nested signed atom, got: {:?}",
        err.span
    );
    assert!(
        err.message.contains("no leading sign"),
        "error should quote §11's rule: {}",
        err.message
    );
}

#[test]
fn canonical_neg_call_still_reads() {
    // The form the writer actually emits for a negated literal must keep
    // parsing: `(neg 1.0)`, not a signed atom.
    let src = "(%module (%public x) (%bind x (neg 1.0)))\n";
    let m = read(src).expect("(neg 1.0) must still parse");
    let text = flatppl_flatpir::write(&m);
    assert!(
        text.contains("(neg 1.0)"),
        "round-trip should preserve the canonical neg call: {text}"
    );
}

#[test]
fn malformed_numeric_atoms_are_refused() {
    for atom in ["1__2", "1_", "0x_FF", "0xFF_", "1._2", "1e_2"] {
        let src = format!("(%module (%bind x {atom}))\n");
        let err = read(&src).expect_err("malformed numeric atoms must not become constants");
        assert!(
            err.message.contains("malformed numeric atom"),
            "unexpected error for {atom}: {}",
            err.message
        );
    }

    for atom in ["0xF_F", "1.2_5", "1e1_0"] {
        let src = format!("(%module (%bind x {atom}))\n");
        read(&src).unwrap_or_else(|err| panic!("valid numeric atom {atom} failed: {err}"));
    }
}

#[test]
fn numeric_module_names_are_refused() {
    for src in [
        "(%module (%bind 1 0))\n",
        "(%module (%public 1) (%bind x 0))\n",
    ] {
        let err = read(src).expect_err("module binding and public names must be names");
        assert!(
            err.message.contains("invalid FlatPPL") && err.message.contains("name"),
            "unexpected error: {}",
            err.message
        );
    }
}

#[test]
fn an_empty_nested_type_list_is_an_error_not_a_panic() {
    // `()` in a `%meta` type slot indexed `items[0]` on an empty list and
    // aborted the process (exit 101). It must report like any other malformed
    // form.
    let src = "(%module\n  (%public x)\n  (%bind x (%meta (() %fixed reals) 1)))\n";
    let err = read(src).unwrap_err();
    assert!(
        err.message.contains("empty `()` is not a type"),
        "got: {}",
        err.message
    );
    assert_eq!(span_text(src, err.span), "()");
}

#[test]
fn an_empty_nested_valueset_list_is_an_error_not_a_panic() {
    let src = "(%module\n  (%public x)\n  (%bind x (%meta ((%scalar real) %fixed ()) 1)))\n";
    let err = read(src).unwrap_err();
    assert!(
        err.message.contains("empty `()` is not a value set"),
        "got: {}",
        err.message
    );
}
