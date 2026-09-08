//! Pathologically deep source refuses instead of aborting the process.
//!
//! Before the shared depth budget, `x = ((((…1.0…))))` past about 5500 levels
//! killed the process with `fatal runtime error: stack overflow`, which no
//! caller can catch. The guard sits on the single re-entry point of the
//! expression grammar, so every nested construct is covered by one check.

use flatppl_core::DEFAULT_MAX_DEPTH;

fn nested_parens(depth: usize) -> String {
    format!("x = {}1.0{}\n", "(".repeat(depth), ")".repeat(depth))
}

#[test]
fn nesting_past_the_limit_is_a_parse_error_not_an_abort() {
    // Far past the old crash floor: this used to abort, so reaching the
    // assertion at all is part of what is being tested.
    let err = flatppl_syntax::parse(&nested_parens(50_000))
        .expect_err("deep nesting must refuse, not overflow the stack");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("nesting is deeper") && msg.contains(&DEFAULT_MAX_DEPTH.to_string()),
        "the refusal must name the limit: {msg}"
    );
}

#[test]
fn nesting_within_the_limit_still_parses() {
    // The deepest model in any workspace corpus is 14 levels; 100 is already
    // seven times that and must be accepted.
    assert!(
        flatppl_syntax::parse(&nested_parens(100)).is_ok(),
        "a depth of 100 is far inside the limit and must parse"
    );
}

#[test]
fn breadth_is_not_charged_as_depth() {
    // 5000 siblings at depth 2. A guard that counted every step rather than
    // nesting would refuse this, and it is a shape real models have.
    let elems = (0..5000)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    assert!(
        flatppl_syntax::parse(&format!("x = [{elems}]\n")).is_ok(),
        "a wide flat array is not deep and must parse"
    );
}

fn flat_sum(terms: usize) -> String {
    format!(
        "x = {}\n",
        std::iter::repeat_n("1", terms)
            .collect::<Vec<_>>()
            .join(" + ")
    )
}

#[test]
fn a_flat_operator_chain_is_guarded_before_recursive_consumers() {
    assert!(flatppl_syntax::parse(&flat_sum(1_000)).is_ok());
    let err = flatppl_syntax::parse(&flat_sum(50_000))
        .expect_err("a left-deep constructed tree must be bounded");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("constructed expression depth") && msg.contains("4096"),
        "the refusal must name the constructed-tree limit: {msg}"
    );
}

#[test]
fn punctuation_breadth_is_bounded_before_parser_allocation() {
    let source = format!("x = [{}]\n", ",".repeat(300_000));
    let err =
        flatppl_syntax::parse(&source).expect_err("a punctuation stream must hit the token budget");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("surface token count") && msg.contains("262144"),
        "the refusal must name the token limit: {msg}"
    );
}
