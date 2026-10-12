use flatppl_formula::{DataPlacement, build, read};
use serde_json::json;

fn spec(priors: serde_json::Value) -> flatppl_formula::Spec {
    let mut spec = read(include_str!("fixtures/gaussian.formula.json")).unwrap();
    spec.formula = "y ~ 0 + x + z".into();
    spec.data
        .as_mut()
        .unwrap()
        .columns
        .push(flatppl_formula::Column {
            name: "z".into(),
            values: flatppl_formula::ColumnValues::Real64(vec![2.0, 3.0, 5.0]),
            valid: None,
        });
    spec.options.data_placement = DataPlacement::Inline;
    spec.priors = serde_json::from_value(priors).unwrap();
    spec
}

#[test]
fn coefficient_overrides_remove_fixed_values_from_sampled_coordinates() {
    let mut spec = spec(json!([
        {"class": "b", "prior": "normal(0, 2)"},
        {"class": "b", "coef": "x", "prior": "student_t(4, 1, 3)"},
        {"class": "b", "coef": "z", "prior": "constant(0.5)"},
        {"class": "sigma", "prior": "constant(1)"}
    ]));
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let coordinates: Vec<_> = converted
        .info
        .parameters
        .iter()
        .filter(|p| !p.derived)
        .map(|p| (p.name.as_str(), p.binding.as_str(), p.index))
        .collect();
    assert_eq!(coordinates, [("b_x", "b_1", None)]);
    assert!(!converted.info.has_improper_prior);
    // Selector precedence is independent of the order in which rows arrive.
    spec.priors.reverse();
    let reversed = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(converted.info).unwrap(),
        serde_json::to_value(reversed.info).unwrap()
    );
}

#[test]
fn scoped_prior_expressions_use_the_parameter_graph() {
    let mut spec = spec(json!([
        {"class": "b", "prior": "normal(0, sigma)", "lb": 0, "ub": "sigma"},
        {"class": "b", "coef": "z", "prior": "Normal(1, 2)"},
        {"class": "sigma", "prior": "exponential(2)"}
    ]));
    let conversion = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        conversion
            .info
            .parameters
            .iter()
            .filter(|p| !p.derived)
            .count(),
        3
    );
    // Both malformed text and a well-formed unknown constructor fail the same
    // public conversion boundary; a valid scoped expression above succeeds.
    for prior in [
        "Normal(0, 1))\ninjected = (1",
        "NotADistribution(0, 1)",
        "Normal(0, missing)",
    ] {
        spec.priors[0].prior = prior.into();
        assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
    }
}
