use flatppl_formula::{build, read};
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
    spec.priors = serde_json::from_str(&priors.to_string()).unwrap();
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
        {"class": "b", "prior": "normal(0, sigma)", "lb": 0.25, "ub": "sigma"},
        {"class": "b", "coef": "z", "prior": "Normal(1, 2)"},
        {"class": "sigma", "prior": "exponential(2)"}
    ]));
    let conversion = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        conversion.recipe.priors.b[0].lb,
        Some(flatppl_formula::PriorBound::Number(0.25))
    );
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

#[test]
fn group_prior_selectors_follow_brms_coefficient_and_alias_precedence() {
    let mut spec = read(include_str!("fixtures/group-priors.formula.json")).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    // Actual brms 2.23.2 make_stancode(normalize=TRUE): g has a half-Cauchy
    // intercept sd, a Student-t slope sd restricted above zero, and LKJ(3).
    // h inherits both half-Normal sd laws and LKJ(2) from the class priors.
    let priors = serde_json::to_value(&converted.recipe.mu.priors).unwrap();
    assert_eq!(
        priors["sd"],
        json!([
            [{"prior": "cauchy(0, 1)"}, {"prior": "student_t(4, 1, 3)"}],
            [{"prior": "normal(0, 2)"}, {"prior": "normal(0, 2)"}]
        ])
    );
    assert_eq!(
        priors["cor"],
        json!([{"prior": "lkj_corr_cholesky(3)"}, {"prior": "lkj(2)"}])
    );
    assert!(!converted.info.has_improper_prior);
    let sd: Vec<_> = converted
        .info
        .parameters
        .iter()
        .filter(|p| p.name.starts_with("sd_"))
        .map(|p| (p.name.as_str(), p.binding.as_str(), p.index, p.derived))
        .collect();
    assert_eq!(
        sd,
        [
            ("sd_g__Intercept", "sd_g_1", None, false),
            ("sd_g__x", "sd_g_2", None, false),
            ("sd_h__Intercept", "sd_h", Some(1), false),
            ("sd_h__x", "sd_h", Some(2), false)
        ]
    );
    spec.priors.reverse();
    let reversed = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(converted.info).unwrap(),
        serde_json::to_value(reversed.info).unwrap()
    );
    for prior in &mut spec.priors {
        if prior.class == "cor" {
            prior.class = "L".into();
        } else if prior.class == "L" {
            prior.class = "cor".into();
        }
    }
    let aliases = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        flatppl_syntax::print(&reversed.module),
        flatppl_syntax::print(&aliases.module)
    );
    // cor/L share a selector identity; spelling cannot hide a duplicate row.
    let mut duplicate = spec
        .priors
        .iter()
        .find(|prior| prior.class == "L")
        .unwrap()
        .clone();
    duplicate.class = "cor".into();
    spec.priors.push(duplicate);
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
}

#[test]
fn fixed_group_scales_leave_only_sampled_components_in_coordinates() {
    let mut spec = read(include_str!("fixtures/group-priors.formula.json")).unwrap();
    spec.priors.extend(
        serde_json::from_value::<Vec<flatppl_formula::Prior>>(json!([
            {"class": "sd", "group": "g", "coef": "Intercept", "prior": "constant(0.7)"},
            {"class": "sd", "group": "h", "prior": "constant(0)"}
        ]))
        .unwrap(),
    );
    // A later group's sanitized name collides with the first group's scalar
    // component. Metadata must use the allocated binding, not reconstruct it.
    spec.formula = "y ~ x + (1 + x | g) + (1 + x | g_1)".into();
    spec.data.as_mut().unwrap().columns[3].name = "g_1".into();
    for prior in &mut spec.priors {
        if prior.group.as_deref() == Some("h") {
            prior.group = Some("g_1".into());
        }
    }
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let sd: Vec<_> = converted
        .info
        .parameters
        .iter()
        .filter(|p| p.name.starts_with("sd_"))
        .map(|p| (p.name.as_str(), p.binding.as_str(), p.index, p.derived))
        .collect();
    assert_eq!(
        sd,
        [
            ("sd_g__Intercept", "sd_g", Some(1), true),
            ("sd_g__x", "sd_g_2", None, false),
            ("sd_g_1__Intercept", "sd_g_1_2", Some(1), true),
            ("sd_g_1__x", "sd_g_1_2", Some(2), true)
        ]
    );
    assert!(!converted.info.has_improper_prior);
}
