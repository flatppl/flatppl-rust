use flatppl_formula::parse_formula;
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
    errors: Vec<String>,
}

#[derive(Deserialize)]
struct Case {
    formula: String,
    intercept: u8,
    variables: Vec<String>,
    terms: Vec<String>,
}

#[test]
fn expanded_terms_match_r_with_brms_fixed_formula_order() {
    let oracle: Oracle = serde_json::from_str(include_str!("fixtures/formula-terms.json")).unwrap();
    for case in oracle.cases {
        let formula = parse_formula(&case.formula).unwrap();
        assert_eq!(formula.intercept, case.intercept == 1, "{}", case.formula);
        assert_eq!(formula.variables, case.variables, "{}", case.formula);
        assert_eq!(
            formula
                .fixed
                .iter()
                .map(|term| term.factors.join(":"))
                .collect::<Vec<_>>(),
            case.terms,
            "{}",
            case.formula
        );
    }
    for formula in oracle.errors {
        assert!(parse_formula(&formula).is_err(), "{formula}");
    }
}

#[test]
fn random_intercept_removal_composes_with_fixed_expansion() {
    assert_eq!(
        parse_formula("y ~ (a + b)^2 + (1 | h) + (1 | g) - (1 | h)").unwrap(),
        parse_formula("y ~ a*b + (1 | g)").unwrap()
    );
}

#[test]
fn combinatorial_expansion_and_syntax_depth_are_bounded() {
    let terms = (0..13)
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join("+");
    assert!(
        parse_formula(&format!("y ~ ({terms})^13"))
            .unwrap_err()
            .0
            .contains("term limit")
    );
    let nested = format!("y ~ {}x{}", "(".repeat(128), ")".repeat(128));
    assert!(
        parse_formula(&nested)
            .unwrap_err()
            .0
            .contains("nesting limit")
    );
}
