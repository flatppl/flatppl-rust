use crate::{Error, Result, Table, data::ColumnModel, emit};
use nalgebra::{DMatrix, linalg::SymmetricEigen};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct TensorTerm {
    pub columns: [String; 2],
    pub ranks: Option<[usize; 2]>,
}

impl TensorTerm {
    pub(crate) fn label(&self) -> String {
        let [x, z] = &self.columns;
        match self.ranks {
            None => format!("t2({x}, {z})"),
            Some([a, b]) if a == b => format!("t2({x}, {z}, k = {a})"),
            Some([a, b]) => format!("t2({x}, {z}, k = c({a}, {b}))"),
        }
    }

    pub(crate) fn name(&self) -> String {
        let [x, z] = &self.columns;
        emit::coefficient_name(&format!("t2({x},{z})"))
    }
}

pub(crate) const FIXED_COLS: usize = 3;
pub(crate) const BLOCKS: [&str; 3] = ["rr", "nr", "rn"];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TensorSmooth {
    margins: [CubicMargin; 2],
    // Absorb one sum-to-zero constraint over the unpenalized tensor block.
    fixed_transform: [[f64; FIXED_COLS]; 4],
    penalty_scale: f64,
    pub fixed: Vec<String>,
    pub blocks: [TensorBlock; 3],
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct TensorBlock {
    pub inputs: Vec<String>,
    pub sds: String,
    pub z: String,
}

impl TensorSmooth {
    pub(crate) fn fit(term: &TensorTerm, table: &Table, rows: &[usize]) -> Result<Self> {
        let [kx, kz] = term.ranks.unwrap_or([5; 2]);
        if kx.checked_mul(kz).is_none_or(|size| size > 4096) {
            return Err(Error(
                "tensor smooth supports at most 4096 basis columns".into(),
            ));
        }
        let [x, z] = tensor_columns(term, table, rows)?;
        let x_margin = CubicMargin::fit(&x, kx)?;
        let z_margin = CubicMargin::fit(&z, kz)?;
        let x_design = x_margin.values(&x)?;
        let z_design = z_margin.values(&z)?;
        let mut null_sums = [0.0; 4];
        let mut norm: f64 = 0.0;
        for (x, z) in x_design.iter().zip(&z_design) {
            for (total, value) in null_sums
                .iter_mut()
                .zip(products(&x[kx - 2..], &z[kz - 2..]))
            {
                *total += value;
            }
            let x_norm = x.iter().map(|x| x.abs()).sum::<f64>();
            let z_norm = z.iter().map(|z| z.abs()).sum::<f64>();
            norm = norm.max(x_norm * z_norm);
        }
        if norm <= 0.0 || !norm.is_finite() || null_sums.iter().any(|x| !x.is_finite()) {
            return Err(Error("tensor smooth lost numerical rank".into()));
        }
        // The norm of each disjoint indicator penalty is one. mgcv's
        // smoothCon divides it by 1 / ||X||_infinity^2 before centering.
        let penalty_scale = 1.0 / norm;
        let q = crate::splines::complete_q(DMatrix::from_column_slice(4, 1, &null_sums));
        let fixed_transform = std::array::from_fn(|i| std::array::from_fn(|j| q[(i, j + 1)]));
        Ok(Self {
            margins: [x_margin, z_margin],
            fixed_transform,
            penalty_scale,
            fixed: Vec::new(),
            blocks: std::array::from_fn(|_| TensorBlock::default()),
        })
    }

    pub(crate) fn block_sizes(&self) -> [usize; 3] {
        let [a, b] = self.margins.each_ref().map(|margin| margin.knots.len() - 2);
        [a * b, 2 * b, 2 * a]
    }

    pub(crate) fn values(
        &self,
        term: &TensorTerm,
        table: &Table,
        rows: &[usize],
    ) -> Result<Vec<Vec<f64>>> {
        if !self.penalty_scale.is_finite() || self.penalty_scale <= 0.0 {
            return Err(Error("invalid retained tensor penalty scale".into()));
        }
        let [x, z] = tensor_columns(term, table, rows)?;
        let [x_margin, z_margin] = &self.margins;
        let x_design = x_margin.values(&x)?;
        let z_design = z_margin.values(&z)?;
        let kx = x_margin.knots.len() - 2;
        let kz = z_margin.knots.len() - 2;
        x_design
            .iter()
            .zip(&z_design)
            .map(|(x, z)| {
                let (xr, xn) = x.split_at(kx);
                let (zr, zn) = z.split_at(kz);
                let null: Vec<_> = products(xn, zn).collect();
                let fixed = (0..FIXED_COLS).map(|j| {
                    null.iter()
                        .zip(&self.fixed_transform)
                        .map(|(v, row)| v * row[j])
                        .sum()
                });
                let random = products(xr, zr)
                    .chain(products(xn, zr))
                    .chain(products(xr, zn))
                    .map(|value| value * self.penalty_scale);
                let values: Vec<f64> = fixed.chain(random).collect();
                if values.iter().any(|x| !x.is_finite()) {
                    return Err(Error("tensor design values must be finite".into()));
                }
                Ok(values)
            })
            .collect()
    }
}

fn products<'a>(left: &'a [f64], right: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    left.iter().flat_map(|a| right.iter().map(move |b| a * b))
}

fn tensor_columns(term: &TensorTerm, table: &Table, rows: &[usize]) -> Result<[Vec<f64>; 2]> {
    let [x, z] = &term.columns;
    Ok([
        table.column(x)?.numeric(rows)?,
        table.column(z)?.numeric(rows)?,
    ])
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CubicMargin {
    knots: Vec<f64>,
    // Values and second derivatives of the reparameterized basis at the knots.
    values: Vec<Vec<f64>>,
    curvature: Vec<Vec<f64>>,
}

impl CubicMargin {
    fn fit(x: &[f64], rank: usize) -> Result<Self> {
        if x.iter().any(|x| !x.is_finite()) {
            return Err(Error("tensor margins need finite training values".into()));
        }
        let mut distinct = x.to_vec();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        if rank < 3 || rank > distinct.len() {
            return Err(Error(format!(
                "tensor marginal rank must be between 3 and the {} distinct training values",
                distinct.len()
            )));
        }
        let knots: Vec<_> = (0..rank)
            .map(|i| {
                let index = i as f64 * (distinct.len() - 1) as f64 / (rank - 1) as f64;
                let lower = index.floor() as usize;
                let fraction = index.fract();
                (1.0 - fraction) * distinct[lower] + fraction * distinct[index.ceil() as usize]
            })
            .collect();
        let h: Vec<_> = knots.windows(2).map(|knots| knots[1] - knots[0]).collect();
        if h.iter().any(|h| !h.is_finite() || *h <= 0.0) {
            return Err(Error(
                "tensor marginal knots must be finite and distinct".into(),
            ));
        }
        // Natural cubic interpolation: T M = D maps knot values to the
        // interior second derivatives. D' T^-1 D is the roughness penalty.
        let penalized = rank - 2;
        let mut t = DMatrix::zeros(penalized, penalized);
        let mut d = DMatrix::zeros(penalized, rank);
        for j in 0..penalized {
            let left = h[j];
            let right = h[j + 1];
            t[(j, j)] = (left + right) / 3.0;
            if j + 1 < penalized {
                t[(j, j + 1)] = right / 6.0;
                t[(j + 1, j)] = right / 6.0;
            }
            d[(j, j)] = 1.0 / left;
            d[(j, j + 1)] = -(1.0 / left + 1.0 / right);
            d[(j, j + 2)] = 1.0 / right;
        }
        let interior = t
            .cholesky()
            .ok_or_else(|| Error("tensor marginal penalty lost numerical rank".into()))?
            .solve(&d);
        let penalty = d.transpose() * &interior;
        let curvature = DMatrix::from_fn(rank, rank, |i, j| {
            if i == 0 || i == rank - 1 {
                0.0
            } else {
                interior[(i - 1, j)]
            }
        });
        let basis = Self {
            knots,
            values: matrix_rows(&DMatrix::identity(rank, rank)),
            curvature: matrix_rows(&curvature),
        };
        let raw = basis.values(x)?;
        let design = DMatrix::from_fn(x.len(), rank, |i, j| raw[i][j]);
        let mut rotation = natural_parameterization(&design, penalty, penalized)?;
        // The two null eigenvectors can rotate freely. Fix the affine basis
        // before tensoring so its scale does not depend on the eigensolver.
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let deviation = (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        if !deviation.is_finite() || deviation <= 0.0 {
            return Err(Error(
                "tensor marginal nullspace lost numerical rank".into(),
            ));
        }
        rotation.column_mut(penalized).fill(1.0);
        for (i, &knot) in basis.knots.iter().enumerate() {
            rotation[(i, penalized + 1)] = (knot - mean) / deviation;
        }
        let mut curvature = curvature * &rotation;
        curvature.columns_mut(penalized, 2).fill(0.0);
        Ok(Self {
            values: matrix_rows(&rotation),
            curvature: matrix_rows(&curvature),
            ..basis
        })
    }

    fn values(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        let k = self.knots.len();
        if k < 3
            || self.values.len() != k
            || self.curvature.len() != k
            || self.knots.iter().any(|knot| !knot.is_finite())
            || self
                .values
                .iter()
                .chain(&self.curvature)
                .any(|r| r.len() != k)
            || self.knots.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error("invalid retained tensor margin".into()));
        }
        x.iter()
            .map(|&x| {
                if !x.is_finite() {
                    return Err(Error("tensor prediction needs finite values".into()));
                }
                let i = self
                    .knots
                    .partition_point(|knot| *knot <= x)
                    .saturating_sub(1)
                    .min(k - 2);
                let lo = self.knots[i];
                let hi = self.knots[i + 1];
                let h = hi - lo;
                let b = (x - lo) / h;
                let a = 1.0 - b;
                let (c, d) = if x < self.knots[0] {
                    (0.0, -b * h * h / 6.0)
                } else if x > self.knots[k - 1] {
                    (-a * h * h / 6.0, 0.0)
                } else {
                    ((a * a * a - a) * h * h / 6.0, (b * b * b - b) * h * h / 6.0)
                };
                let row: Vec<_> = (0..k)
                    .map(|j| {
                        a * self.values[i][j]
                            + b * self.values[i + 1][j]
                            + c * self.curvature[i][j]
                            + d * self.curvature[i + 1][j]
                    })
                    .collect();
                if row.iter().any(|x| !x.is_finite()) {
                    return Err(Error("tensor marginal values must be finite".into()));
                }
                Ok(row)
            })
            .collect()
    }
}

fn natural_parameterization(
    design: &DMatrix<f64>,
    penalty: DMatrix<f64>,
    penalized: usize,
) -> Result<DMatrix<f64>> {
    let k = design.ncols();
    // Whiten the penalized basis as in mgcv's type 3 parameterization.
    let eig = SymmetricEigen::try_new((&penalty + penalty.transpose()) * 0.5, f64::EPSILON, 10000)
        .ok_or_else(|| Error("tensor marginal eigendecomposition did not converge".into()))?;
    let mut order: Vec<_> = (0..k).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let mut rotation = DMatrix::from_fn(k, k, |i, j| eig.eigenvectors[(i, order[j])]);
    for (j, &index) in order.iter().take(penalized).enumerate() {
        let value = eig.eigenvalues[index];
        if value <= 0.0 || !value.is_finite() {
            return Err(Error("tensor marginal penalty lost numerical rank".into()));
        }
        rotation.column_mut(j).scale_mut(1.0 / value.sqrt());
    }
    let rough = rotation.columns(0, penalized);
    let rms = (design * rough).norm() / ((design.nrows() * penalized) as f64).sqrt();
    if !rms.is_finite() || rms <= 0.0 {
        return Err(Error("tensor marginal basis lost numerical rank".into()));
    }
    rotation.columns_mut(0, penalized).scale_mut(1.0 / rms);
    Ok(rotation)
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    matrix
        .row_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}
