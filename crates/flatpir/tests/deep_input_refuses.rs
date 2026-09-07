//! Pathologically deep FlatPIR refuses instead of aborting the process.
//!
//! Both readers are covered. The s-expression reader had NO guard and aborted
//! above about 8000 levels; the JSON reader had a local `MAX_DEPTH = 128` that
//! happened to equal the shared default and now uses it, so the two agree on
//! the depth and derive the number in one place.

use flatppl_core::DEFAULT_MAX_DEPTH;

fn nested_sexpr(depth: usize) -> String {
    format!(
        "(%module (%bind x {}1.0{}))\n",
        "(neg ".repeat(depth),
        ")".repeat(depth)
    )
}

#[test]
fn deep_sexpr_refuses_rather_than_overflowing() {
    let err = flatppl_flatpir::read(&nested_sexpr(60_000))
        .expect_err("deep nesting must refuse, not overflow the stack");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("nesting is deeper") && msg.contains(&DEFAULT_MAX_DEPTH.to_string()),
        "the refusal must name the limit: {msg}"
    );
}

#[test]
fn a_sexpr_within_the_limit_still_reads() {
    assert!(
        flatppl_flatpir::read(&nested_sexpr(100)).is_ok(),
        "a depth of 100 is inside the limit and must read"
    );
}

#[test]
fn deep_json_refuses_cleanly_at_some_layer() {
    // Built iteratively; a recursive builder would hit the test's own stack.
    let mut expr = String::from("1.0");
    for _ in 0..1000 {
        expr = format!("[\"neg\", {expr}]");
    }
    let doc = format!("{{\"%module\": [{{\"%bind\": [\"x\", {expr}]}}]}}");
    // `serde_json` enforces its OWN recursion limit while parsing the text, so
    // it refuses before this crate's reader is reached. That is a real guard
    // and the reason the JSON path never crashed; the reader's own `MAX_DEPTH`
    // (now the shared constant) is the second line of defence for a `Value`
    // built programmatically rather than parsed. What matters either way is
    // that a deep document produces an error rather than an abort.
    let parsed = serde_json::from_str::<serde_json::Value>(&doc);
    match parsed {
        Err(e) => assert!(
            e.to_string().contains("recursion limit"),
            "serde_json should be the layer that refuses here: {e}"
        ),
        Ok(v) => {
            let err = flatppl_flatpir::from_json(&v)
                .expect_err("if serde_json admits it, the reader must refuse it");
            assert!(format!("{err:?}").contains("nesting is deeper"));
        }
    }
}

#[test]
fn json_within_the_limit_still_reads() {
    // Round-trip a real module rather than hand-building the encoding, so the
    // shape is whatever `to_json` actually emits. Built with this crate's own
    // s-expression reader, so the test adds no dependency.
    let m = flatppl_flatpir::read(&nested_sexpr(20)).expect("depth 20 reads");
    let v = flatppl_flatpir::to_json(&m);
    assert!(
        flatppl_flatpir::from_json(&v).is_ok(),
        "a depth of 20 is inside every limit and must read back"
    );
}
