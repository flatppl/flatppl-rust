use flatppl_formula::ThinPlate;
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
        let fitted = ThinPlate::fit(&oracle.x, oracle.rank).unwrap();
        let fitted: ThinPlate =
            serde_json::from_str(&serde_json::to_string(&fitted).unwrap()).unwrap();
        let mut values = fitted.evaluate(&oracle.x).unwrap();
        values.extend(fitted.evaluate(&oracle.prediction).unwrap());
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
