use flatppl_formula::{DataMode, DataValue, Recipe, ThinPlate, build, prepare_data, read};
use serde::Deserialize;
use serde_json::{Value, json};
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

#[derive(Deserialize)]
struct FactorOracle {
    x: Vec<f64>,
    levels: Vec<String>,
    groups: Vec<String>,
    new_x: Vec<f64>,
    new_groups: Vec<String>,
    cases: Vec<FactorCase>,
}

#[derive(Deserialize)]
struct FactorCase {
    ordered: bool,
    curves: Vec<FactorCurve>,
}

#[derive(Deserialize)]
struct FactorCurve {
    level: String,
    basis: Vec<Vec<f64>>,
}

fn factor_column(groups: &[String], levels: &[String], ordered: bool) -> Value {
    let codes: Vec<_> = groups
        .iter()
        .map(|group| levels.iter().position(|level| level == group).unwrap())
        .collect();
    json!({"name":"f","values":{"kind":"factor","values":{"codes":codes,"levels":levels,"ordered":ordered}}})
}

#[test]
fn factor_smooths_preserve_mgcv_curve_covariance_and_prediction() {
    let oracle: FactorOracle =
        serde_json::from_str(include_str!("fixtures/factor-smooths.json")).unwrap();
    for case in oracle.cases {
        let spec = read(
            &json!({
                "formula": "y ~ s(x, by=f, k=4)",
                "priors": [
                    {"class":"sds", "coef":"s(x, k = 4, by = f):fA", "prior":"exponential(2)"}
                ],
                "data": {"nrows": oracle.x.len(), "columns": [
                    {"name":"x", "values":{"kind":"real64", "values":oracle.x}},
                    {"name":"y", "values":{"kind":"real64", "values":oracle.x}},
                    factor_column(&oracle.groups, &oracle.levels, case.ordered)
                ]}
            })
            .to_string(),
        )
        .unwrap();
        let model = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        let factor = &model.info.factor_levels["f"];
        assert_eq!(factor.levels, oracle.levels);
        assert_eq!(factor.ordered, case.ordered);
        let exported = serde_json::to_value(&model.recipe).unwrap();
        let recipe: Recipe = serde_json::from_value(exported).unwrap();
        let predictor = &recipe.predictors["mu"];
        let slopes: Vec<_> = model
            .info
            .parameters
            .iter()
            .filter(|parameter| parameter.binding == "bs")
            .map(|parameter| parameter.name.as_str())
            .collect();
        let expected_slopes: Vec<_> = case
            .curves
            .iter()
            .map(|curve| format!("b_sx:f{}_1", curve.level))
            .collect();
        assert_eq!(slopes, expected_slopes);
        assert_eq!(predictor.priors.sds.last().unwrap().prior, "exponential(2)");

        let mut levels = oracle.levels.clone();
        levels.reverse();
        let new_data = serde_json::from_value(json!({"nrows": oracle.new_x.len(), "columns": [
            {"name":"x", "values":{"kind":"real64", "values":oracle.new_x}},
            factor_column(&oracle.new_groups, &levels, case.ordered)
        ]}))
        .unwrap();
        let prediction = prepare_data(&recipe, &new_data, DataMode::Prediction).unwrap();
        for curve in case.curves {
            let level = oracle
                .levels
                .iter()
                .position(|x| x == &curve.level)
                .unwrap()
                + 1;
            let actual: Vec<_> = [&model.data, &prediction]
                .into_iter()
                .flat_map(|data| {
                    let DataValue::Integers(codes) = &data.constants["smooth_1_by"] else {
                        panic!("expected retained level codes")
                    };
                    let inputs = &predictor.smooths[0].inputs;
                    codes.iter().enumerate().map(move |(i, &code)| {
                        inputs
                            .iter()
                            .map(|name| match &data.constants[name] {
                                DataValue::Real64(x) if code as usize == level => x[i],
                                DataValue::Real64(_) => 0.0,
                                _ => panic!("expected numeric spline data"),
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for (i, row) in actual.iter().enumerate() {
                close(row[0], curve.basis[i][0]);
                for j in 0..actual.len() {
                    let dot = |basis: &[Vec<f64>]| {
                        basis[i][1..]
                            .iter()
                            .zip(&basis[j][1..])
                            .map(|(a, b)| a * b)
                            .sum()
                    };
                    close(dot(&actual), dot(&curve.basis));
                }
            }
        }
    }
}

#[test]
fn tensor_smooth_preserves_block_gram_matrices_for_new_rows() {
    let spec = serde_json::from_value(json!({
        "formula": "y ~ t2(x, z, k=c(3,4))",
        "priors": [
            {"class": "sds", "coef": "t2(x, z, k = c(3, 4))", "prior": "exponential(2)"}
        ]
    }))
    .unwrap();
    let training = flatppl_formula::read_json_table(
        r#"{"x":[-2,-1,0,1,2,3,1,3,null],"z":[0,1,3,1,-1,2,3,0,0],"y":[1,2,1,3,4,2,1,2,3]}"#,
    )
    .unwrap();
    let result = build(&spec, &training).unwrap();
    assert_eq!(result.data.rows, (0..8).collect::<Vec<_>>());
    let recipe: Recipe =
        serde_json::from_value(serde_json::to_value(&result.recipe).unwrap()).unwrap();
    assert_eq!(recipe.predictors["mu"].priors.bs.len(), 3);
    assert_eq!(recipe.predictors["mu"].priors.sds.len(), 3);
    assert!(
        recipe.predictors["mu"]
            .priors
            .sds
            .iter()
            .all(|p| p.prior == "exponential(2)")
    );
    let new = flatppl_formula::read_json_table(r#"{"x":[-3,3.5],"z":[0.5,4]}"#).unwrap();
    let held = prepare_data(&recipe, &new, DataMode::Prediction).unwrap();
    // Independent mgcv 1.9.4 cubic margins, normalized with a constant and a
    // centered unit-RMS linear null basis. The three entries per block are
    // train/train, train/prediction and prediction/prediction Gram entries.
    let checks = [
        (
            "fixed",
            3,
            [2.176582192409661, -0.171849103262374, -0.886816843058109],
        ),
        (
            "rr",
            2,
            [-0.004672947707001, 0.044424213899920, -0.119362290605944],
        ),
        (
            "nr",
            4,
            [0.033658399670824, -0.033268829465305, 0.100842277114581],
        ),
        (
            "rn",
            2,
            [-0.003191363059250, -0.007993367162465, 0.000753642199716],
        ),
    ];
    for (block, width, expected) in checks {
        let mut gram = [0.0; 3];
        for j in 1..=width {
            let name = format!("tensor_1_{block}_{j}");
            let DataValue::Real64(train) = &result.data.constants[&name] else {
                panic!("expected real training column");
            };
            let DataValue::Real64(new) = &held.constants[&name] else {
                panic!("expected real prediction column");
            };
            if block == "fixed" {
                close(train.iter().sum(), 0.0);
            }
            gram[0] += train[0] * train[1];
            gram[1] += train[2] * new[0];
            gram[2] += new[0] * new[1];
        }
        for (actual, expected) in gram.into_iter().zip(expected) {
            close(actual, expected);
        }
    }
}
