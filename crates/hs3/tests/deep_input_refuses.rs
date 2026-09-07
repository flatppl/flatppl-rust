//! Pathologically deep HS3 input refuses instead of aborting the process.

use flatppl_core::DEFAULT_MAX_DEPTH;

fn doc_with_expression(expr: &str) -> String {
    format!(
        r#"{{"distributions":[{{"name":"d","type":"generic_dist","expression":"{expr}"}}],
            "parameter_points":[]}}"#
    )
}

#[test]
fn a_deeply_nested_expression_refuses_rather_than_overflowing() {
    // Past the old crash floor of about 10000 levels, so reaching the
    // assertion at all is part of what is being tested.
    let expr = format!("{}x{}", "(".repeat(60_000), ")".repeat(60_000));
    let err = flatppl_hs3::read_hs3(&doc_with_expression(&expr))
        .expect_err("deep nesting must refuse, not overflow the stack");
    let msg = err.to_string();
    assert!(
        msg.contains("nesting is deeper") && msg.contains(&DEFAULT_MAX_DEPTH.to_string()),
        "the refusal must name the limit: {msg}"
    );
}

#[test]
fn an_expression_within_the_limit_still_converts() {
    let expr = format!("{}x{}", "(".repeat(50), ")".repeat(50));
    assert!(
        flatppl_hs3::read_hs3(&doc_with_expression(&expr)).is_ok(),
        "a depth of 50 is inside the limit and must convert"
    );
}

/// A SHALLOW document that builds a DEEP tree.
///
/// `fold_function` reduces an operand array into a left-nested chain, so N
/// summands become a tree N deep from a document of nesting depth three. That
/// used to abort the process.
///
/// The depth is CREATED at the fold, so it is checked there, against the same
/// shared limit. Relying on the importer's re-parse gate instead was not
/// enough: the release binary happened to survive printing a 10000-deep chain
/// and refuse at the gate, but a debug build overflows first, so the refusal
/// has to happen before the tree is built. Making the fold not build the depth
/// at all (an n-ary node, or a balanced tree) is a separate card.
#[test]
fn ten_thousand_flat_summands_refuse_cleanly() {
    let summands = (0..10_000)
        .map(|i| format!("{i}.0"))
        .collect::<Vec<_>>()
        .join(", ");
    let doc = format!(
        r#"{{"functions":[{{"name":"s","type":"sum","summands":[{summands}]}}],
             "distributions":[{{"name":"d","type":"gaussian_dist","mean":"mu","sigma":"sg","x":"x"}}],
             "parameter_points":[{{"name":"p","parameters":[
               {{"name":"mu","value":0.0}},{{"name":"sg","value":1.0}}]}}]}}"#
    );
    let err = flatppl_hs3::read_hs3(&doc)
        .expect_err("a 10000-deep constructed chain must refuse, not overflow");
    let msg = err.to_string();
    assert!(
        msg.contains("nesting is deeper") && msg.contains(&DEFAULT_MAX_DEPTH.to_string()),
        "the refusal must name the limit: {msg}"
    );
}

#[test]
fn a_short_summand_array_still_converts() {
    let summands = (0..20)
        .map(|i| format!("{i}.0"))
        .collect::<Vec<_>>()
        .join(", ");
    let doc = format!(
        r#"{{"functions":[{{"name":"s","type":"sum","summands":[{summands}]}}],
             "distributions":[{{"name":"d","type":"gaussian_dist","mean":"mu","sigma":"sg","x":"x"}}],
             "parameter_points":[{{"name":"p","parameters":[
               {{"name":"mu","value":0.0}},{{"name":"sg","value":1.0}}]}}]}}"#
    );
    assert!(
        flatppl_hs3::read_hs3(&doc).is_ok(),
        "20 summands is a 20-deep chain, well inside the limit"
    );
}
