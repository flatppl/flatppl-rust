use crate::{Error, Result};
use nalgebra::{DMatrix, DVector, linalg::SymmetricEigen};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Smooth {
    pub basis: ThinPlate,
    pub inputs: Vec<String>,
    pub sds: String,
    pub z: String,
}

/// Fitted one-dimensional thin-plate basis, with second-derivative penalty.
/// The first output column is unpenalized; the remaining columns have unit
/// penalty after dividing integrated squared curvature by `penalty_scale`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThinPlate {
    pub shift: f64,
    pub centers: Vec<f64>,
    /// Rows multiply `[abs(x - center)^3 / 12, ..., 1, x]` at `x - shift`.
    pub transform: Vec<Vec<f64>>,
    pub penalty_scale: f64,
}

impl ThinPlate {
    /// Fit the mgcv/brms default thin-plate regression basis to retained rows.
    pub fn fit(x: &[f64], rank: Option<usize>) -> Result<Self> {
        let rank = rank.unwrap_or(10);
        if x.is_empty() || x.iter().any(|x| !x.is_finite()) {
            return Err(Error(
                "thin-plate spline needs finite training values".into(),
            ));
        }
        let shift = x.iter().sum::<f64>() / x.len() as f64;
        let mut centers: Vec<_> = x.iter().map(|x| x - shift).collect();
        centers.sort_by(f64::total_cmp);
        centers.dedup();
        let n = centers.len();
        if rank < 3 || rank > n {
            return Err(Error(format!(
                "thin-plate rank must be between 3 and the {n} distinct training values"
            )));
        }
        if n > 2000 {
            return Err(Error(
                "thin-plate splines with more than 2000 distinct training values require unsupported knot subsampling".into(),
            ));
        }
        let kernel = DMatrix::from_fn(n, n, |i, j| radial(centers[i], centers[j]));
        let basis = if rank == n {
            DMatrix::identity(n, n)
        } else {
            let eig = eigen(kernel.clone())?;
            let mut order: Vec<_> = (0..n).collect();
            order.sort_by(|&i, &j| {
                eig.eigenvalues[j]
                    .abs()
                    .total_cmp(&eig.eigenvalues[i].abs())
            });
            order.truncate(rank);
            // mgcv truncates by magnitude, then lists the retained eigenpairs
            // algebraically. This order matters for the later penalty scale.
            order.sort_by(|&i, &j| eig.eigenvalues[j].total_cmp(&eig.eigenvalues[i]));
            DMatrix::from_fn(n, rank, |i, j| eig.eigenvectors[(i, order[j])])
        };
        let polynomials = DMatrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { centers[i] });
        let constraints = basis.transpose() * polynomials;
        let reversed = DMatrix::from_fn(rank, 2, |i, j| constraints[(rank - i - 1, j)]);
        let q = complete_q(reversed);
        let null = DMatrix::from_fn(rank, rank - 2, |i, j| q[(rank - i - 1, rank - j - 1)]);
        let mut transform = DMatrix::zeros(n + 2, rank);
        transform
            .view_mut((0, 0), (n, rank - 2))
            .copy_from(&(basis * null));
        transform[(n, rank - 2)] = 1.0;
        transform[(n + 1, rank - 1)] = 1.0;
        let unique = raw_basis(&centers, 0.0, &centers) * &transform;
        let rows: Vec<_> = x
            .iter()
            .map(|x| {
                centers
                    .binary_search_by(|c| c.partial_cmp(&(x - shift)).unwrap())
                    .unwrap()
            })
            .collect();
        let mut design = DMatrix::from_fn(x.len(), rank, |i, j| unique[(rows[i], j)]);
        for j in 0..rank {
            let rms = (design.column(j).norm_squared() / x.len() as f64).sqrt();
            if rms == 0.0 || !rms.is_finite() {
                return Err(Error("thin-plate basis lost numerical rank".into()));
            }
            transform.column_mut(j).scale_mut(1.0 / rms);
            design.column_mut(j).scale_mut(1.0 / rms);
        }
        let radial_transform = transform.rows(0, n);
        let mut penalty = radial_transform.transpose() * kernel * radial_transform;
        let penalty_norm = penalty
            .column_iter()
            .map(|c| c.abs().sum())
            .fold(0.0, f64::max);
        let design_norm = design.row_iter().map(|r| r.abs().sum()).fold(0.0, f64::max);
        let penalty_scale = penalty_norm / design_norm.powi(2);
        if !penalty_scale.is_finite() || penalty_scale <= 0.0 {
            return Err(Error(
                "thin-plate penalty scale is not positive and finite".into(),
            ));
        }
        penalty /= penalty_scale;

        let constraint = DMatrix::from_fn(rank, 1, |j, _| design.column(j).sum());
        let q = complete_q(constraint).columns(1, rank - 1).into_owned();
        transform *= &q;
        design *= &q;
        penalty = q.transpose() * penalty * q;
        let eig = eigen(penalty)?;
        let mut order: Vec<_> = (0..rank - 1).collect();
        order.sort_by(|&i, &j| eig.eigenvalues[j].total_cmp(&eig.eigenvalues[i]));
        let rotation = DMatrix::from_fn(rank - 1, rank - 1, |i, j| eig.eigenvectors[(i, order[j])]);
        transform *= &rotation;
        design *= rotation;
        let mut average_norm = 0.0;
        for (j, &index) in order.iter().take(rank - 2).enumerate() {
            let value = eig.eigenvalues[index];
            if value <= 0.0 || !value.is_finite() {
                return Err(Error("thin-plate penalty lost numerical rank".into()));
            }
            transform.column_mut(j).scale_mut(1.0 / value.sqrt());
            average_norm += design.column(j).norm_squared() / value;
        }
        average_norm /= (rank - 2) as f64;
        let fixed_scale = (average_norm / design.column(rank - 2).norm_squared()).sqrt();
        if !fixed_scale.is_finite() || fixed_scale <= 0.0 {
            return Err(Error(
                "thin-plate nullspace scale is not positive and finite".into(),
            ));
        }
        transform.column_mut(rank - 2).scale_mut(fixed_scale);
        // Give the one-dimensional affine nullspace its natural positive slope.
        if transform[(n + 1, rank - 2)] < 0.0 {
            transform.column_mut(rank - 2).scale_mut(-1.0);
        }
        let order = std::iter::once(rank - 2)
            .chain(0..rank - 2)
            .collect::<Vec<_>>();
        let transform = (0..n + 2)
            .map(|i| order.iter().map(|&j| transform[(i, j)]).collect())
            .collect();
        Ok(Self {
            shift,
            centers,
            transform,
            penalty_scale,
        })
    }

    /// Evaluate on new rows without centering or fitting them again.
    pub fn evaluate(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n = self.centers.len() + 2;
        let rank = self.transform.first().map_or(0, Vec::len);
        if rank < 2
            || rank >= self.centers.len()
            || self.transform.len() != n
            || self.transform.iter().any(|row| row.len() != rank)
            || !self.shift.is_finite()
            || !self.penalty_scale.is_finite()
            || self.penalty_scale <= 0.0
            || self
                .centers
                .iter()
                .chain(self.transform.iter().flatten())
                .any(|x| !x.is_finite())
        {
            return Err(Error("invalid retained thin-plate transform".into()));
        }
        if x.iter().any(|x| !x.is_finite()) {
            return Err(Error("thin-plate prediction needs finite values".into()));
        }
        let transform = DMatrix::from_fn(n, rank, |i, j| self.transform[i][j]);
        // Prediction needs only one raw basis row, not an observations-by-knots
        // temporary. The returned fitted design has just rank-1 columns.
        let mut row = DVector::from_element(n, 1.0);
        Ok(x.iter()
            .map(|x| {
                let x = x - self.shift;
                for (value, center) in row.iter_mut().zip(&self.centers) {
                    *value = radial(x, *center);
                }
                row[n - 1] = x;
                transform.tr_mul(&row).as_slice().to_vec()
            })
            .collect())
    }
}

fn radial(x: f64, center: f64) -> f64 {
    (x - center).abs().powi(3) / 12.0
}

fn raw_basis(x: &[f64], shift: f64, centers: &[f64]) -> DMatrix<f64> {
    DMatrix::from_fn(x.len(), centers.len() + 2, |i, j| {
        if j < centers.len() {
            radial(x[i] - shift, centers[j])
        } else if j == centers.len() {
            1.0
        } else {
            x[i] - shift
        }
    })
}

fn complete_q(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut transpose = DMatrix::identity(matrix.nrows(), matrix.nrows());
    matrix.qr().q_tr_mul(&mut transpose);
    transpose.transpose()
}

fn eigen(matrix: DMatrix<f64>) -> Result<SymmetricEigen<f64, nalgebra::Dyn>> {
    SymmetricEigen::try_new(matrix, f64::EPSILON, 10000)
        .ok_or_else(|| Error("thin-plate eigendecomposition did not converge".into()))
}
