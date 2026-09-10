//! Expression breadth, nesting, and the remaining constructed-tree guard.

fn nested_parens(depth: usize) -> String {
    format!("x = {}1.0{}\n", "(".repeat(depth), ")".repeat(depth))
}

#[test]
fn nesting_beyond_the_former_budget_parses() {
    // Published pyhf models exceed the former 128-level expression budget.
    flatppl_syntax::parse(&nested_parens(160)).unwrap();
}

#[test]
fn breadth_is_not_charged_as_depth() {
    // A wide array exceeds the former 262144-token cap without deep nesting.
    let elems = (0..132_000)
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
