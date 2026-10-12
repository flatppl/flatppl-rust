use flatppl_formula::{DataMode, DataValue, Recipe, ThinPlate, build, prepare_data, read};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Oracle {
    x: Vec<f64>,
    rank: Option<usize>,
    penalty_scale: f64,
    basis: Vec<Vec<f64>>,
    prediction: Vec<f64>,
    predicted: Vec<Vec<f64>>,
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-7 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn fitted_smooth_and_retained_prediction_match_mgcv() {
    let cases: BTreeMap<String, Oracle> =
        serde_json::from_str(include_str!("fixtures/splines.json")).unwrap();
    for (name, oracle) in cases {
        let formula = match oracle.rank {
            Some(rank) => format!("y ~ s(x, k={rank})"),
            None => "y ~ s(x)".into(),
        };
        let spec = read(
            &serde_json::json!({
                "formula": formula,
                "priors": [
                    {"class":"b", "prior":"normal(0,2)"},
                    {"class":"Intercept", "prior":"normal(0,1)"},
                    {"class":"sds", "prior":"exponential(1)"},
                    {"class":"sigma", "prior":"exponential(1)"}
                ],
                "data":{"nrows":oracle.x.len(),"columns":[
                    {"name":"x", "values":{"kind":"real64", "values":oracle.x}},
                    {"name":"y", "values":{"kind":"real64", "values":oracle.x}}
                ]}
            })
            .to_string(),
        )
        .unwrap();
        let result = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        assert!(!result.info.has_improper_prior);
        let recipe: Recipe =
            serde_json::from_str(&serde_json::to_string(&result.recipe).unwrap()).unwrap();
        let smooth = &recipe.predictors["mu"].smooths[0];
        let fitted = &smooth.basis;
        let rows = |data: &flatppl_formula::DataDocument| {
            (0..data.rows.len())
                .map(|i| {
                    smooth
                        .inputs
                        .iter()
                        .map(|name| {
                            let DataValue::Real64(values) = &data.constants[name] else {
                                panic!("expected numeric spline data")
                            };
                            values[i]
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let mut values = rows(&result.data);
        let mut table = spec.data.unwrap();
        table.columns.retain(|column| column.name == "x");
        table.nrows = oracle.prediction.len();
        table.columns[0].values = flatppl_formula::ColumnValues::Real64(oracle.prediction.clone());
        values.extend(rows(
            &prepare_data(&recipe, &table, DataMode::Prediction).unwrap(),
        ));
        let mut expected = oracle.basis;
        expected.extend(oracle.predicted);
        close(fitted.penalty_scale, oracle.penalty_scale);
        assert_eq!(values.len(), expected.len(), "{name}");
        assert_eq!(values[0].len(), expected[0].len(), "{name}");
        // Independent eigensolvers may rotate the penalized coefficients.
        // The affine scale and full covariance preserve the coefficient prior.
        for (x, y) in values.iter().zip(&expected) {
            close(x[0], y[0]);
        }
        for i in 0..values.len() {
            for j in 0..values.len() {
                let dot = |rows: &[Vec<f64>]| {
                    rows[i][1..]
                        .iter()
                        .zip(&rows[j][1..])
                        .map(|(x, y)| x * y)
                        .sum()
                };
                close(dot(&values), dot(&expected));
            }
        }
        let rank = values[0].len();
        for j in 0..rank {
            close(values[..oracle.x.len()].iter().map(|row| row[j]).sum(), 0.0);
            for k in 0..rank {
                let penalty: f64 = fitted
                    .centers
                    .iter()
                    .enumerate()
                    .map(|(a, x)| {
                        fitted
                            .centers
                            .iter()
                            .enumerate()
                            .map(|(b, y)| {
                                fitted.transform[a][j] * (x - y).abs().powi(3) / 12.0
                                    * fitted.transform[b][k]
                            })
                            .sum::<f64>()
                    })
                    .sum();
                close(
                    penalty / fitted.penalty_scale,
                    if j == k && j > 0 { 1.0 } else { 0.0 },
                );
            }
        }
    }
}

#[test]
fn prediction_validates_retained_numeric_state() {
    let state = ThinPlate::fit(&[-2.0, -1.0, 0.0, 1.0, 2.0], Some(3)).unwrap();
    let mut invalid = state.clone();
    invalid.transform[0][1] = f64::NAN;
    assert!(invalid.evaluate(&[0.5]).is_err());
    let mut document = serde_json::to_value(&state).unwrap();
    document["penalty_scale"] = 0.0.into();
    let invalid: ThinPlate = serde_json::from_value(document).unwrap();
    assert!(invalid.evaluate(&[0.5]).is_err());
    assert!(
        state.evaluate(&[0.5]).unwrap()[0]
            .iter()
            .all(|v| v.is_finite())
    );
}
