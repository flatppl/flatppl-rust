//! An all-dotted comparison chain is broadcast sugar for the `land` chain (spec
//! §05): `a .< b .<= c` lowers to `land.(a .< b, b .<= c)`. A chain that mixes
//! dotted and plain comparisons has no such reading and is refused at the first
//! operator that differs from the chain's first.

use flatppl_syntax::{Syntax, parse, print_with};

fn assert_equivalent(sugared: &str, explicit: &str) {
    let module = parse(sugared).unwrap();
    let expected = flatppl_flatpir::write(&parse(explicit).unwrap());
    assert_eq!(flatppl_flatpir::write(&module), expected);
    for syntax in [Syntax::Full, Syntax::Minimal] {
        let reparsed = parse(&print_with(&module, syntax)).unwrap();
        assert_eq!(flatppl_flatpir::write(&reparsed), expected);
    }
}

const ARRAYS: &str =
    "a = [1.0, 2.0, 3.0]\nb = [2.0, 2.0, 2.0]\nc = [3.0, 2.0, 1.0]\nd = [4.0, 1.0, 0.0]\n";

#[test]
fn a_three_operand_dotted_chain_is_the_broadcast_land_chain() {
    assert_equivalent(
        &format!("{ARRAYS}r = a .< b .<= c"),
        &format!("{ARRAYS}r = land.(a .< b, b .<= c)"),
    );
}

#[test]
fn a_four_operand_dotted_chain_nests_left() {
    assert_equivalent(
        &format!("{ARRAYS}r = a .< b .<= c .> d"),
        &format!("{ARRAYS}r = land.(land.(a .< b, b .<= c), c .> d)"),
    );
}

#[test]
fn a_mixed_chain_is_refused_at_the_differing_operator() {
    for (chain, operator) in [
        ("a .< b <= c", "<="),
        ("a < b .<= c", ".<="),
        ("a .< b .<= c < d", "<"),
    ] {
        let source = format!("{ARRAYS}r = {chain}\n");
        let error = parse(&source).unwrap_err();
        assert!(
            error
                .message
                .contains("a comparison chain mixes dotted and plain comparisons"),
            "{chain}: {}",
            error.message
        );
        let (start, end) = error.span.expect("located");
        let line = source.lines().nth(4).unwrap();
        let line_start = source.find(line).unwrap();
        let at = line_start + line.rfind(&format!(" {operator} ")).unwrap() + 1;
        assert_eq!(
            (start as usize, end as usize),
            (at, at + operator.len()),
            "{chain}"
        );
    }
}
