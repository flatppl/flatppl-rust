use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Contrast {
    Treatment,
    Sum,
    Helmert,
    Polynomial,
}

pub(crate) fn matrix(kind: Contrast, levels: &[String]) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let n = levels.len();
    if n < 2 {
        return Err(Error(
            "contrasts require at least two observed levels".into(),
        ));
    }
    if kind == Contrast::Polynomial && n > 16 {
        return Err(Error(
            "polynomial contrasts support at most 16 observed levels; use another contrast for larger factors".into(),
        ));
    }
    let mut values = vec![vec![0.0; n - 1]; n];
    let names = match kind {
        Contrast::Treatment => {
            for (i, row) in values.iter_mut().enumerate().skip(1) {
                row[i - 1] = 1.0;
            }
            levels[1..].to_vec()
        }
        Contrast::Sum => {
            for (i, row) in values.iter_mut().enumerate().take(n - 1) {
                row[i] = 1.0;
            }
            values[n - 1].fill(-1.0);
            (1..n).map(|i| i.to_string()).collect()
        }
        Contrast::Helmert => {
            for column in 0..n - 1 {
                for row in values.iter_mut().take(column + 1) {
                    row[column] = -1.0;
                }
                values[column + 1][column] = (column + 1) as f64;
            }
            (1..n).map(|i| i.to_string()).collect()
        }
        Contrast::Polynomial => {
            values = polynomial(n);
            (1..n)
                .map(|degree| match degree {
                    1 => ".L".into(),
                    2 => ".Q".into(),
                    3 => ".C".into(),
                    _ => format!("^{degree}"),
                })
                .collect()
        }
    };
    Ok((names, values))
}

fn polynomial(n: usize) -> Vec<Vec<f64>> {
    let mut basis = vec![vec![1.0 / (n as f64).sqrt(); n]];
    for degree in 1..n {
        // Multiplying the preceding orthogonal polynomial by x supplies the
        // next degree with positive leading coefficient, as in R's contr.poly.
        let mut column: Vec<_> = basis[degree - 1]
            .iter()
            .enumerate()
            .map(|(i, value)| (i as f64 - (n - 1) as f64 / 2.0) * value)
            .collect();
        // Reorthogonalization prevents the high-degree recurrence from
        // amplifying rounding error near the ends of the equally spaced grid.
        for _ in 0..2 {
            for previous in &basis {
                let projection: f64 = column.iter().zip(previous).map(|(x, y)| x * y).sum();
                for (x, y) in column.iter_mut().zip(previous) {
                    *x -= projection * y;
                }
            }
        }
        let norm = column.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut column {
            *x /= norm;
        }
        basis.push(column);
    }
    (0..n)
        .map(|row| basis[1..].iter().map(|column| column[row]).collect())
        .collect()
}
