//! Fused §09 regression densities. Linear predictors stay in link scale until
//! the row likelihood needs them; shared normalizers are computed once.

use super::*;

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Bernoulli,
    Binomial,
    Categorical,
    Normal,
    Poisson,
    NegativeBinomial,
    StudentT,
    Gamma,
    Ordered,
    Beta,
}

macro_rules! entry {
    ($constant:ident, $name:literal, $score:ident, $sample:ident, $family:ident, [$($field:literal),*]) => {
        pub(super) const $constant: (&str, DistLowering) = ($name, DistLowering {
            logpdf: $score, sample: Some($sample), touniform: None,
            broadcast: Broadcast::PerCell(&[$($field),*]),
        });
        fn $score(e: &mut Emitter, p: &Params, y: &Value) -> Result<Value, EmitError> {
            score(e, p, y, Family::$family)
        }
        fn $sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
            sample(e, p, Family::$family)
        }
    };
}

entry!(
    BERNOULLI,
    "BernoulliLogitGLM",
    bernoulli_score,
    bernoulli_draw,
    Bernoulli,
    ["x", "alpha", "beta"]
);
entry!(
    BINOMIAL,
    "BinomialLogitGLM",
    binomial_score,
    binomial_draw,
    Binomial,
    ["x", "n", "alpha", "beta"]
);
entry!(
    CATEGORICAL,
    "CategoricalLogitGLM",
    categorical_score,
    categorical_draw,
    Categorical,
    ["x", "alpha", "beta"]
);
entry!(
    NORMAL,
    "NormalGLM",
    normal_score,
    normal_draw,
    Normal,
    ["x", "alpha", "beta", "sigma"]
);
entry!(
    POISSON,
    "PoissonLogGLM",
    poisson_score,
    poisson_draw,
    Poisson,
    ["x", "alpha", "beta"]
);
entry!(
    NEGATIVE_BINOMIAL,
    "NegativeBinomial2LogGLM",
    negative_binomial_score,
    negative_binomial_draw,
    NegativeBinomial,
    ["x", "alpha", "beta", "psi"]
);
entry!(
    STUDENT_T,
    "StudentTGLM",
    student_score,
    student_draw,
    StudentT,
    ["x", "nu", "alpha", "beta", "sigma"]
);
entry!(
    GAMMA,
    "GammaLogGLM",
    gamma_score,
    gamma_draw,
    Gamma,
    ["x", "alpha", "beta", "shape"]
);
entry!(
    ORDERED,
    "OrderedLogitGLM",
    ordered_score,
    ordered_draw,
    Ordered,
    ["x", "beta", "cutpoints"]
);
entry!(
    BETA,
    "BetaLogitGLM",
    beta_score,
    beta_draw,
    Beta,
    ["x", "alpha", "beta", "precision"]
);

fn dimensions(
    e: &Emitter,
    p: &Params,
    v: &Value,
    rank: usize,
    name: &str,
) -> Result<Vec<u64>, EmitError> {
    match e.cell_ty(v) {
        MlirTy::Ranked(d) if d.len() == rank && d.iter().all(Option::is_some) => {
            Ok(d.iter().map(|n| n.unwrap()).collect())
        }
        _ => Err(EmitError::at(
            p.kernel_input,
            format!("regression parameter '{name}' requires a fixed rank-{rank} shape"),
        )),
    }
}

fn scalar_param(e: &mut Emitter, p: &Params, name: &str) -> Result<Value, EmitError> {
    let value = p.get(e, name)?;
    if e.cell_ty(&value) != MlirTy::Scalar {
        return Err(EmitError::at(
            p.kernel_input,
            format!("regression parameter '{name}' must be scalar"),
        ));
    }
    if name != "nu" {
        e.check_finite_parameter(p.kernel_input, name, &value)?;
    }
    Ok(value)
}

fn predictors(e: &mut Emitter, p: &Params, family: Family) -> Result<(Value, u64), EmitError> {
    let x = p.get(e, "x")?;
    let xd = dimensions(e, p, &x, 2, "x")?;
    let beta = p.get(e, "beta")?;
    e.check_finite_parameter(p.kernel_input, "x", &x)?;
    e.check_finite_parameter(p.kernel_input, "beta", &beta)?;
    let bd = dimensions(
        e,
        p,
        &beta,
        if family == Family::Categorical { 2 } else { 1 },
        "beta",
    )?;
    if xd[1] != bd[0] {
        return Err(EmitError::at(
            p.kernel_input,
            "regression x columns must match beta rows",
        ));
    }
    if family == Family::Categorical {
        let alpha = p.get(e, "alpha")?;
        e.check_finite_parameter(p.kernel_input, "alpha", &alpha)?;
        let ad = dimensions(e, p, &alpha, 1, "alpha")?;
        if ad[0] != bd[1] || ad[0] == 0 {
            return Err(EmitError::at(
                p.kernel_input,
                "categorical alpha and beta class dimensions differ",
            ));
        }
        let xb = e.matmat(&x, &beta);
        let alpha = e.reshape_cell(&alpha, MlirTy::Ranked(vec![Some(1), Some(ad[0])]));
        return Ok((e.add(&xb, &alpha), xd[0]));
    }
    let xb = e.matvec(&x, &beta);
    if family == Family::Ordered {
        let cuts = p.get(e, "cutpoints")?;
        let cd = dimensions(e, p, &cuts, 1, "cutpoints")?;
        if cd[0] == 0 {
            return Err(EmitError::at(
                p.kernel_input,
                "ordered regression needs at least one cutpoint",
            ));
        }
        let inf = e.inf_like(&cuts);
        let magnitude = e.abs(&cuts);
        let finite = e.compare("LT", &magnitude, &inf);
        let mut valid = e.reduce_all(&finite);
        if cd[0] > 1 {
            let left = vector_slice(e, &cuts, 0, cd[0] - 1);
            let right = vector_slice(e, &cuts, 1, cd[0]);
            let increasing = e.compare("LT", &left, &right);
            let increasing = e.reduce_all(&increasing);
            valid = e.and(&valid, &increasing);
        }
        e.record_runtime_error(&valid, 9);
        return Ok((xb, xd[0]));
    }
    let alpha = scalar_param(e, p, "alpha")?;
    Ok((e.add(&xb, &alpha), xd[0]))
}

/// max(z,0) + log1p(exp(-abs(z))) avoids overflow in either tail.
fn softplus(e: &mut Emitter, z: &Value) -> Value {
    let abs = e.abs(z);
    let negative = e.neg(&abs);
    let ex = e.exp(&negative);
    let tail = e.log1p(&ex);
    let zero = e.scalar(0.0);
    let positive = e.max(z, &zero);
    e.add(&positive, &tail)
}

fn log_sigmoid(e: &mut Emitter, z: &Value) -> Value {
    let negative = e.neg(z);
    let sp = softplus(e, &negative);
    e.neg(&sp)
}

/// Use log1p near the mean and log differences when subtraction loses the tail.
fn log_density_ratio(e: &mut Emitter, relative: &Value, log_y: &Value, log_mean: &Value) -> Value {
    let cutoff = e.scalar(-0.5);
    let far = e.compare("LT", relative, &cutoff);
    let zero = e.scalar(0.0);
    let safe = e.select(&far, &zero, relative);
    let near = e.log1p(&safe);
    let tail = e.sub(log_y, log_mean);
    e.select(&far, &tail, &near)
}

fn student_normalizer(e: &mut Emitter, nu: &Value, log_sigma: &Value) -> Value {
    let half = e.scalar(0.5);
    if matches!(e.target(), crate::Target::Cpu) {
        let half_nu = e.mul(&half, nu);
        let log_beta = e.log_beta(&half_nu, &half);
        let negative_log_beta = e.neg(&log_beta);
        let log_nu = e.log(nu);
        let half_log_nu = e.mul(&half, &log_nu);
        let normalizer = e.sub(&negative_log_beta, &half_log_nu);
        return e.sub(&normalizer, log_sigma);
    }
    let a = e.mul(&half, nu);
    let eight = e.scalar(8.0);
    let large = e.compare("GE", &a, &eight);
    let small_a = e.select(&large, &eight, &a);
    let small_plus_half = e.add(&small_a, &half);
    let numerator = e.lgamma(&small_plus_half);
    let denominator = e.lgamma(&small_a);
    let small = e.sub(&numerator, &denominator);
    let log_nu = e.log(nu);
    let half_log_nu = e.mul(&half, &log_nu);
    let small = e.sub(&small, &half_log_nu);
    let half_log_pi = e.scalar(0.5 * std::f64::consts::PI.ln());
    let small = e.sub(&small, &half_log_pi);
    // Cancel the large Stirling terms before evaluation. The fixed half-step
    // avoids the general log-Beta branches and keeps large-nu gradients.
    let big_a = e.select(&large, &a, &eight);
    let big_plus_half = e.add(&big_a, &half);
    let correction_a = e.log_gamma_correction(&big_a);
    let correction_plus_half = e.log_gamma_correction(&big_plus_half);
    let correction = e.sub(&correction_plus_half, &correction_a);
    let ratio = e.div(&half, &big_a);
    let log_ratio = e.log1p(&ratio);
    let scaled = e.mul(&big_a, &log_ratio);
    let big = e.sub(&scaled, &half);
    let big = e.add(&big, &correction);
    let constant = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
    let big = e.sub(&big, &constant);
    let normalizer = e.select(&large, &big, &small);
    e.sub(&normalizer, log_sigma)
}

fn finite_response(e: &mut Emitter, y: &Value) -> Value {
    let magnitude = e.abs(y);
    let inf = e.inf_like(y);
    e.compare("LT", &magnitude, &inf)
}

fn finish(e: &mut Emitter, supported: &Value, rows: &Value) -> Value {
    let rows = outside_count_support(e, supported, rows);
    e.reduce_sum_last_axis(&rows)
}

fn score(e: &mut Emitter, p: &Params, y: &Value, family: Family) -> Result<Value, EmitError> {
    let (eta, n) = predictors(e, p, family)?;
    let yd = dimensions(e, p, y, 1, "response")?;
    if yd != [n] {
        return Err(EmitError::at(
            p.kernel_input,
            "regression response length must match x rows",
        ));
    }
    if family == Family::Categorical {
        return categorical_density(e, p, y, &eta, n);
    }
    if family == Family::Ordered {
        return ordered_density(e, p, y, &eta);
    }
    if family == Family::Bernoulli {
        if let Some(score) = e.reduce_regression_design(&eta, y) {
            return Ok(score);
        }
        // The arithmetic of the broadcast law `Bernoulli.(invlogit.(eta))`.
        let one = e.scalar(1.0);
        let zero = e.scalar(0.0);
        let success = e.compare("EQ", y, &one);
        let failure = e.compare("EQ", y, &zero);
        let supported = e.or(&success, &failure);
        let rows = super::bernoulli_logit_score(e, &eta, &success, &supported);
        return Ok(finish(e, &supported, &rows));
    }
    let one = e.scalar(1.0);
    let zero = e.scalar(0.0);
    let y = e.convert(y, ElemKind::Real);
    match family {
        Family::Binomial => {
            let trials = scalar_param(e, p, "n")?;
            let (supported, y) = count_support(e, &y, Some(&trials));
            let negative_eta = e.neg(&eta);
            let yes = softplus(e, &negative_eta);
            let no = softplus(e, &eta);
            let successes = e.mul(&y, &yes);
            let failures = e.sub(&trials, &y);
            let failures = e.mul(&failures, &no);
            let penalty = e.add(&successes, &failures);
            let rows = e.neg(&penalty);
            let n_minus_y = e.sub(&trials, &y);
            let a = e.add(&y, &one);
            let b = e.add(&n_minus_y, &one);
            let lb = e.log_beta(&a, &b);
            let log_n = e.log1p(&trials);
            let choose = e.add(&log_n, &lb);
            let rows = e.sub(&rows, &choose);
            Ok(finish(e, &supported, &rows))
        }
        Family::Poisson => {
            let (supported, y) = count_support(e, &y, None);
            let cutoff = e.scalar(16.0);
            let large = e.compare("GE", &y, &cutoff);
            let small_y = e.select(&large, &zero, &y);
            let small_eta = e.select(&large, &zero, &eta);
            let ye = e.mul(&small_y, &small_eta);
            let rate = e.exp(&small_eta);
            let yp1 = e.add(&small_y, &one);
            let factorial = e.lgamma(&yp1);
            let rows = e.sub(&ye, &rate);
            let rows = e.sub(&rows, &factorial);
            // Loader's existing deviance/Stirling core avoids cancellation
            // between y*eta, exp(eta), and lgamma(y+1) for ordinary large counts.
            let large_y = e.select(&large, &y, &cutoff);
            let log_cutoff = e.log(&cutoff);
            let large_eta = e.select(&large, &eta, &log_cutoff);
            let large_rate = e.exp(&large_eta);
            let log_y = e.log(&large_y);
            let log_ratio = e.sub(&log_y, &large_eta);
            let stable = poisson_large_logpdf(e, &large_y, &large_rate, &log_y, &log_ratio);
            let rows = e.select(&large, &stable, &rows);
            Ok(finish(e, &supported, &rows))
        }
        Family::NegativeBinomial => {
            let psi = scalar_param(e, p, "psi")?;
            let (supported, y) = count_support(e, &y, None);
            let log_psi = e.log(&psi);
            let gap = e.sub(&eta, &log_psi);
            let negative_gap = e.neg(&gap);
            let a = softplus(e, &negative_gap);
            let b = softplus(e, &gap);
            let ya = e.mul(&y, &a);
            let pb = e.mul(&psi, &b);
            let penalty = e.add(&ya, &pb);
            let choose = log_count_choose(e, &y, &psi);
            let at_zero = e.compare("EQ", &y, &zero);
            let choose = e.select(&at_zero, &zero, &choose);
            let rows = e.sub(&choose, &penalty);
            Ok(finish(e, &supported, &rows))
        }
        Family::Normal | Family::StudentT => {
            let sigma = scalar_param(e, p, "sigma")?;
            let supported = finite_response(e, &y);
            let safe = e.select(&supported, &y, &eta);
            let diff = e.sub(&safe, &eta);
            let student_nu = if family == Family::StudentT {
                let nu = scalar_param(e, p, "nu")?;
                let inf = e.inf_like(&nu);
                let finite = e.compare("LT", &nu, &inf);
                Some((nu, finite))
            } else {
                None
            };
            let normal_diff = match &student_nu {
                Some((_, finite)) => e.select(finite, &zero, &diff),
                None => diff.clone(),
            };
            let normal_sigma = match &student_nu {
                Some((_, finite)) => e.select(finite, &one, &sigma),
                None => sigma.clone(),
            };
            let z = e.div(&normal_diff, &normal_sigma);
            let square = e.mul(&z, &z);
            let half = e.scalar(0.5);
            let log_sigma = e.log(&sigma);
            let constant = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
            let normal_constant = e.add(&log_sigma, &constant);
            let normal_quad = e.mul(&half, &square);
            let normal_quad = e.neg(&normal_quad);
            let count = e.scalar(n as f64);
            if let Some((nu, finite_nu)) = student_nu {
                let safe_nu = e.select(&finite_nu, &nu, &one);
                let nu_plus_one = e.add(&safe_nu, &one);
                let a = e.mul(&half, &nu_plus_one);
                let norm = student_normalizer(e, &safe_nu, &log_sigma);
                let log_nu = e.log(&safe_nu);
                let abs_diff = e.abs(&diff);
                let stable_tail = |e: &mut Emitter| {
                    let nonzero = e.compare("GT", &abs_diff, &zero);
                    let safe_abs = e.select(&nonzero, &abs_diff, &one);
                    let log_abs = e.log(&safe_abs);
                    let log_z = e.sub(&log_abs, &log_sigma);
                    let two = e.scalar(2.0);
                    let log_z_squared = e.mul(&two, &log_z);
                    let log_ratio = e.sub(&log_z_squared, &log_nu);
                    let large = e.compare("GT", &log_ratio, &zero);
                    let large = e.and(&large, &nonzero);
                    let safe_diff = e.select(&large, &zero, &diff);
                    let sigma = e.select(&large, &one, &sigma);
                    let nu = e.select(&large, &one, &safe_nu);
                    let small_z = e.div(&safe_diff, &sigma);
                    let small_square = e.mul(&small_z, &small_z);
                    let small_ratio = e.div(&small_square, &nu);
                    let small = e.log1p(&small_ratio);
                    let tail = softplus(e, &log_ratio);
                    e.select(&large, &tail, &small)
                };
                // CPU benefits from skipping the tail math for ordinary data.
                // GPU benefits from one fused vector path without a case boundary.
                let q = if matches!(e.target(), crate::Target::Gpu) {
                    stable_tail(e)
                } else {
                    let peak = e.reduce_max_last_axis(&abs_diff);
                    let root_nu = e.sqrt(&safe_nu);
                    let scale = e.min(&root_nu, &one);
                    let scale = e.mul(&sigma, &scale);
                    let bound = e.scalar(match e.dtype() {
                        Dtype::F32 => (f32::MAX as f64).sqrt() * 0.25,
                        Dtype::F64 => f64::MAX.sqrt() * 0.25,
                    });
                    let limit = e.mul(&bound, &scale);
                    let overflow = e.compare("GT", &peak, &limit);
                    // Guard unused divisions and squares before their derivatives.
                    let safe_diff = e.select(&overflow, &zero, &diff);
                    let fast_sigma = e.select(&overflow, &one, &sigma);
                    let fast_nu = e.select(&overflow, &one, &safe_nu);
                    let small_z = e.div(&safe_diff, &fast_sigma);
                    let small_square = e.mul(&small_z, &small_z);
                    let small_ratio = e.div(&small_square, &fast_nu);
                    let small = e.log1p(&small_ratio);
                    e.matrix_if_for_reverse(&overflow, &small, stable_tail)
                };
                let q = e.mul(&a, &q);
                let q = e.neg(&q);
                if matches!(e.target(), crate::Target::Gpu) {
                    let q = e.select(&finite_nu, &q, &normal_quad);
                    let q = finish(e, &supported, &q);
                    let normal_norm = e.neg(&normal_constant);
                    let norm = e.select(&finite_nu, &norm, &normal_norm);
                    let norm = e.mul(&norm, &count);
                    Ok(e.add(&norm, &q))
                } else {
                    // CPU reverse mode fuses better when the Normal limit is
                    // selected after each scalar reduction.
                    let q = finish(e, &supported, &q);
                    let norm = e.mul(&norm, &count);
                    let student = e.add(&norm, &q);
                    let normal_quad = finish(e, &supported, &normal_quad);
                    let normal_constant = e.mul(&normal_constant, &count);
                    let normal = e.sub(&normal_quad, &normal_constant);
                    Ok(e.select(&finite_nu, &student, &normal))
                }
            } else {
                let normal_constant = e.mul(&normal_constant, &count);
                let normal_quad = finish(e, &supported, &normal_quad);
                Ok(e.sub(&normal_quad, &normal_constant))
            }
        }
        Family::Gamma => {
            let shape = scalar_param(e, p, "shape")?;
            let positive = e.compare("GT", &y, &zero);
            let nonnegative = e.compare("GE", &y, &zero);
            let finite = finite_response(e, &y);
            let supported = e.and(&nonnegative, &finite);
            let safe = e.select(&positive, &y, &one);
            let log_y = e.log(&safe);
            let log_shape = e.log(&shape);
            let log_rate = e.sub(&log_shape, &eta);
            let exponent = e.sub(&log_y, &eta);
            let delta = e.expm1(&exponent);
            let deviance = e.sub(&exponent, &delta);
            let deviance = e.mul(&shape, &deviance);
            let half = e.scalar(0.5);
            let half_log_shape = e.mul(&half, &log_shape);
            let half_log_two_pi = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
            let norm = e.sub(&half_log_shape, &half_log_two_pi);
            let correction = e.stirling_remainder(&shape);
            let norm = e.sub(&norm, &correction);
            let rows = e.add(&deviance, &norm);
            let rows = e.sub(&rows, &log_y);
            let smaller = e.compare("LT", &shape, &one);
            let equal = e.compare("EQ", &shape, &one);
            let infinity = e.inf_like(&rows);
            let negative_infinity = e.neg(&infinity);
            let limit = e.select(&smaller, &infinity, &negative_infinity);
            let limit = e.select(&equal, &log_rate, &limit);
            let rows = e.select(&positive, &rows, &limit);
            Ok(finish(e, &supported, &rows))
        }
        Family::Beta => {
            let precision = scalar_param(e, p, "precision")?;
            let positive = e.compare("GT", &y, &zero);
            let below = e.compare("LT", &y, &one);
            let interior = e.and(&positive, &below);
            let nonnegative = e.compare("GE", &y, &zero);
            let bounded = e.compare("LE", &y, &one);
            let supported = e.and(&nonnegative, &bounded);
            let half = e.scalar(0.5);
            let safe = e.select(&interior, &y, &half);
            let log_y = e.log(&safe);
            let negative_y = e.neg(&safe);
            let log_complement = e.log1p(&negative_y);
            let lp = e.log(&precision);
            let lm = log_sigmoid(e, &eta);
            let ne = e.neg(&eta);
            let lc = log_sigmoid(e, &ne);
            let la = e.add(&lp, &lm);
            let lb = e.add(&lp, &lc);
            let mean = e.invlogit(&eta);
            let complement = e.invlogit(&ne);
            let a = e.mul(&precision, &mean);
            let b = e.mul(&precision, &complement);
            let am1 = e.sub(&a, &one);
            let bm1 = e.sub(&b, &one);
            let ay = e.mul(&am1, &log_y);
            let by = e.mul(&bm1, &log_complement);
            // Below seven, both shifted shapes are below eight. The recurrence
            // then needs only two row log-Gammas and one shared log-Gamma.
            let seven = e.scalar(7.0);
            let general = e.compare("GE", &precision, &seven);
            let small_precision = e.select(&general, &one, &precision);
            let small_a = e.select(&general, &one, &a);
            let small_b = e.select(&general, &one, &b);
            let small_ap1 = e.add(&small_a, &one);
            let small_bp1 = e.add(&small_b, &one);
            let small_gamma_phi = e.lgamma(&small_precision);
            let small_gamma_a = e.lgamma(&small_ap1);
            let small_gamma_b = e.lgamma(&small_bp1);
            let two = e.scalar(2.0);
            let small_norm = e.mul(&two, &lp);
            let small_norm = e.add(&small_norm, &small_gamma_phi);
            let small_norm = e.sub(&small_norm, &small_gamma_a);
            let small_norm = e.sub(&small_norm, &small_gamma_b);
            let small_norm = e.add(&small_norm, &lm);
            let small_norm = e.add(&small_norm, &lc);
            let powers = e.add(&ay, &by);
            let small_rows = e.add(&small_norm, &powers);
            let general_rows = |e: &mut Emitter| {
                // Shifted Beta recurrence retains log-shape tails without
                // subtracting three large log-Gamma values.
                let ap1 = e.add(&a, &one);
                let bp1 = e.add(&b, &one);
                let shared_denominator_rows = match e.dtype() {
                    Dtype::F32 => 16384,
                    Dtype::F64 => 8192,
                };
                let shifted_beta =
                    if matches!(e.target(), crate::Target::Cpu) && n >= shared_denominator_rows {
                        // Large CPU gradients benefit from sharing the denominator.
                        // Both shifted shapes are below eight only when phi<14.
                        let fourteen = e.scalar(14.0);
                        let small_phi = e.min(&precision, &fourteen);
                        let total = e.add(&small_phi, &two);
                        let log_gamma_sum = e.lgamma(&total);
                        e.log_beta_with_log_gamma_sum(&ap1, &bp1, Some(&log_gamma_sum))
                    } else {
                        e.log_beta(&ap1, &bp1)
                    };
                let norm = e.add(&la, &lb);
                let norm = e.sub(&norm, &lp);
                let log_phi_plus_one = e.log1p(&precision);
                let norm = e.sub(&norm, &log_phi_plus_one);
                let norm = e.sub(&norm, &shifted_beta);
                let rows = e.add(&norm, &ay);
                let rows = e.add(&rows, &by);
                // At high precision, normalizer and row powers are both O(phi).
                // Center the powers on the mean before reducing their difference.
                // If phi<16, no row can have both shapes>=8. Keep this entire
                // concentration calculation out of ordinary low-precision scoring.
                let concentrated = |e: &mut Emitter| {
                    let threshold = e.scalar(8.0);
                    let large_a = e.compare("GE", &a, &threshold);
                    let large_b = e.compare("GE", &b, &threshold);
                    let large = e.and(&large_a, &large_b);
                    let safe_a = e.select(&large, &a, &threshold);
                    let safe_b = e.select(&large, &b, &threshold);
                    let safe_mean = e.select(&large, &mean, &half);
                    let safe_complement = e.select(&large, &complement, &half);
                    let safe_large_y = e.select(&large, &safe, &half);
                    let gap = e.sub(&safe_large_y, &safe_mean);
                    let ratio_a = e.div(&gap, &safe_mean);
                    let negative_gap = e.neg(&gap);
                    let ratio_b = e.div(&negative_gap, &safe_complement);
                    let log_ratio_a = log_density_ratio(e, &ratio_a, &log_y, &lm);
                    let log_ratio_b = log_density_ratio(e, &ratio_b, &log_complement, &lc);
                    let term_a = e.mul(&safe_a, &log_ratio_a);
                    let term_b = e.mul(&safe_b, &log_ratio_b);
                    let deviance = e.add(&term_a, &term_b);
                    let normalizer = e.add(&lp, &lm);
                    let normalizer = e.add(&normalizer, &lc);
                    let log_two_pi = e.scalar((2.0 * std::f64::consts::PI).ln());
                    let normalizer = e.sub(&normalizer, &log_two_pi);
                    let normalizer = e.mul(&half, &normalizer);
                    let correction_a = e.log_gamma_correction(&safe_a);
                    let correction_b = e.log_gamma_correction(&safe_b);
                    let sixteen = e.scalar(16.0);
                    let safe_phi = e.max(&precision, &sixteen);
                    let correction_phi = e.log_gamma_correction(&safe_phi);
                    let normalizer = e.sub(&normalizer, &correction_a);
                    let normalizer = e.sub(&normalizer, &correction_b);
                    let normalizer = e.add(&normalizer, &correction_phi);
                    let centered = e.add(&deviance, &normalizer);
                    let centered = e.sub(&centered, &log_y);
                    let centered = e.sub(&centered, &log_complement);
                    e.select(&large, &centered, &rows)
                };
                if matches!(e.target(), crate::Target::Gpu) {
                    concentrated(e)
                } else {
                    let sixteen = e.scalar(16.0);
                    let concentration = e.compare("GE", &precision, &sixteen);
                    e.matrix_if(&concentration, &rows, concentrated)
                }
            };
            let rows = if matches!(e.target(), crate::Target::Gpu) {
                general_rows(e)
            } else {
                e.matrix_if(&general, &small_rows, general_rows)
            };
            let at_zero = e.compare("EQ", &y, &zero);
            let endpoint_shape = e.select(&at_zero, &a, &b);
            let endpoint_log_other = e.select(&at_zero, &lb, &la);
            let smaller = e.compare("LT", &endpoint_shape, &one);
            let equal = e.compare("EQ", &endpoint_shape, &one);
            let infinity = e.inf_like(&rows);
            let negative_infinity = e.neg(&infinity);
            let limit = e.select(&smaller, &infinity, &negative_infinity);
            let limit = e.select(&equal, &endpoint_log_other, &limit);
            let rows = e.select(&interior, &rows, &limit);
            Ok(finish(e, &supported, &rows))
        }
        _ => unreachable!(),
    }
}

fn row_column(e: &mut Emitter, matrix: &Value, rows: u64, column: u64) -> Value {
    let MlirTy::Ranked(dims) = &matrix.ty else {
        unreachable!()
    };
    let rank = dims.len();
    let mut starts = vec![0; rank];
    let mut limits = dims.iter().map(|n| n.unwrap()).collect::<Vec<_>>();
    starts[rank - 1] = column;
    limits[rank - 1] = column + 1;
    let slice = e.slice(matrix, &starts, &limits, &vec![1; rank]);
    e.reshape_cell(&slice, MlirTy::Ranked(vec![Some(rows)]))
}

fn vector_slice(e: &mut Emitter, vector: &Value, from: u64, to: u64) -> Value {
    let MlirTy::Ranked(dims) = &vector.ty else {
        unreachable!()
    };
    let rank = dims.len();
    let mut starts = vec![0; rank];
    let mut limits = dims.iter().map(|n| n.unwrap()).collect::<Vec<_>>();
    starts[rank - 1] = from;
    limits[rank - 1] = to;
    e.slice(vector, &starts, &limits, &vec![1; rank])
}

fn categorical_density(
    e: &mut Emitter,
    p: &Params,
    y: &Value,
    eta: &Value,
    rows: u64,
) -> Result<Value, EmitError> {
    let dims = dimensions(e, p, eta, 2, "logits")?;
    let classes = dims[1];
    let one = e.scalar(1.0);
    let upper = e.scalar(classes as f64);
    let y = e.convert(y, ElemKind::Real);
    let (supported, y) = count_support(e, &y, Some(&upper));
    let nonzero = e.compare("GE", &y, &one);
    let supported = e.and(&supported, &nonzero);
    let maxima = e.reduce_max_last_axis(eta);
    let maxima_matrix = e.reshape_cell(&maxima, MlirTy::Ranked(vec![Some(rows), Some(1)]));
    let centered = e.sub(eta, &maxima_matrix);
    let ex = e.exp(&centered);
    let sums = e.reduce_sum_last_axis(&ex);
    let logs = e.log(&sums);
    // An integer row gather below avoids a dense one-hot/category reduction.
    let selected = if e.batch_rank(&centered) == 0 && e.batch_rank(&y) == 0 {
        let row_ids = e.index_vector(rows, true);
        let width = e.int_value_const(classes as i64);
        let row_ids = e.mul(&row_ids, &width);
        let safe = e.select(&supported, &y, &one);
        let category = e.sub(&safe, &one);
        let category = e.convert(&category, ElemKind::Int);
        let indices = e.add(&row_ids, &category);
        let flat = e.reshape(&centered, MlirTy::Ranked(vec![Some(rows * classes)]));
        e.gather(&flat, &indices, 0)
    } else {
        let mut selected = e.constant_like(0.0, &logs);
        for class in 0..classes {
            let column = row_column(e, &centered, rows, class);
            let label = e.scalar((class + 1) as f64);
            let hit = e.compare("EQ", &y, &label);
            selected = e.select(&hit, &column, &selected);
        }
        selected
    };
    let score = e.sub(&selected, &logs);
    Ok(finish(e, &supported, &score))
}

fn ordered_density(
    e: &mut Emitter,
    p: &Params,
    y: &Value,
    eta: &Value,
) -> Result<Value, EmitError> {
    let cuts = p.get(e, "cutpoints")?;
    let n = dimensions(e, p, &cuts, 1, "cutpoints")?[0];
    if e.batch_rank(&cuts) > 0 || e.batch_rank(eta) > 0 || e.batch_rank(y) > 0 {
        return ordered_density_batched(e, p, y, eta, &cuts, n);
    }
    let one = e.scalar(1.0);
    let upper = e.scalar((n + 1) as f64);
    let y = e.convert(y, ElemKind::Real);
    let (valid, y) = count_support(e, &y, Some(&upper));
    let nonzero = e.compare("GE", &y, &one);
    let valid = e.and(&valid, &nonzero);
    let first = e.compare("EQ", &y, &one);
    let last = e.compare("EQ", &y, &upper);
    let middle0 = e.not(&first);
    let middle1 = e.not(&last);
    let middle = e.and(&middle0, &middle1);
    let middle = e.and(&middle, &valid);
    let lower_index = e.sub(&y, &one);
    let last_index = e.scalar((n - 1) as f64);
    let upper_index = e.min(&lower_index, &last_index);
    let zero = e.scalar(0.0);
    let upper_index = e.max(&upper_index, &zero);
    let two = e.scalar(2.0);
    let lower_index = e.sub(&y, &two);
    let lower_index = e.max(&lower_index, &zero);
    let lower_index = e.min(&lower_index, &last_index);
    let upper_index = e.convert(&upper_index, ElemKind::Int);
    let lower_index = e.convert(&lower_index, ElemKind::Int);
    let hi = e.gather(&cuts, &upper_index, 0);
    let lo = e.gather(&cuts, &lower_index, 0);
    let a = e.sub(&lo, eta);
    let b = e.sub(&hi, eta);
    let negative_b = e.neg(&b);
    let lb = softplus(e, &negative_b);
    let la = softplus(e, &a);
    let gap = e.sub(&lo, &hi);
    let safe_gap = e.scalar(-1.0);
    let gap = e.select(&middle, &gap, &safe_gap);
    let gap = e.expm1(&gap);
    let gap = e.neg(&gap);
    let gap = e.log(&gap);
    let penalties = e.add(&lb, &la);
    let interior = e.sub(&gap, &penalties);
    let first_score = e.neg(&lb);
    let last_score = e.neg(&la);
    let score = e.select(&first, &first_score, &interior);
    let score = e.select(&last, &last_score, &score);
    Ok(finish(e, &valid, &score))
}

fn expand_event(e: &mut Emitter, value: &Value, rows: u64, batch: Option<u64>) -> Value {
    match batch {
        None => value.clone(),
        Some(b) => {
            let mut dims = vec![Some(b)];
            let MlirTy::Ranked(event) = &value.ty else {
                return value.clone();
            };
            dims.extend(event);
            let axes = (1..=event.len() as u64).collect::<Vec<_>>();
            let expanded = e.broadcast_in_dim(value, &axes, MlirTy::Ranked(dims));
            let mut flat = event.clone();
            flat[0] = Some(b * rows);
            e.reshape(&expanded, MlirTy::Ranked(flat))
        }
    }
}

fn ordered_density_batched(
    e: &mut Emitter,
    _p: &Params,
    y: &Value,
    eta: &Value,
    cuts: &Value,
    count: u64,
) -> Result<Value, EmitError> {
    let one = e.scalar(1.0);
    let upper = e.scalar((count + 1) as f64);
    let y = e.convert(y, ElemKind::Real);
    let (supported, y) = count_support(e, &y, Some(&upper));
    let positive = e.compare("GE", &y, &one);
    let supported = e.and(&supported, &positive);
    let mut selected = e.constant_like(0.0, eta);
    for class in 0..=count {
        let score = if class == 0 {
            let cut = vector_elem(e, cuts, 0);
            let diff = e.sub(&cut, eta);
            log_sigmoid(e, &diff)
        } else if class == count {
            let cut = vector_elem(e, cuts, count - 1);
            let diff = e.sub(eta, &cut);
            log_sigmoid(e, &diff)
        } else {
            let lower = vector_elem(e, cuts, class - 1);
            let upper = vector_elem(e, cuts, class);
            let a = e.sub(&lower, eta);
            let negative_b = e.sub(eta, &upper);
            let a = softplus(e, &a);
            let b = softplus(e, &negative_b);
            let gap = e.sub(&lower, &upper);
            let gap = e.expm1(&gap);
            let gap = e.neg(&gap);
            let gap = e.log(&gap);
            let score = e.sub(&gap, &a);
            e.sub(&score, &b)
        };
        let label = e.scalar((class + 1) as f64);
        let hit = e.compare("EQ", &y, &label);
        selected = e.select(&hit, &score, &selected);
    }
    Ok(finish(e, &supported, &selected))
}

fn sample(e: &mut Emitter, p: &Params, family: Family) -> Result<Value, EmitError> {
    let (eta, rows) = predictors(e, p, family)?;
    let outer = e.batch_shape();
    let batch = outer.as_ref().map(|s| s.iter().product::<u64>());
    let lanes = rows * batch.unwrap_or(1);
    let eta = expand_event(e, &eta, rows, batch);
    let ty = MlirTy::Ranked(vec![Some(lanes)]);
    // Rejection samplers already own a flat independent lane mask.
    e.set_batch_shape(vec![lanes]);
    let result = sample_flat(e, p, family, &eta, lanes, &ty);
    match &outer {
        Some(shape) => e.set_batch_shape(shape.clone()),
        None => e.clear_batch_shape(),
    }
    let result = result?;
    let mut shape = outer.unwrap_or_default();
    shape.push(rows);
    Ok(e.reshape(
        &result,
        MlirTy::Ranked(shape.into_iter().map(Some).collect()),
    ))
}

fn sample_flat(
    e: &mut Emitter,
    p: &Params,
    family: Family,
    eta: &Value,
    lanes: u64,
    ty: &MlirTy,
) -> Result<Value, EmitError> {
    let one = e.scalar(1.0);
    let zero = e.scalar(0.0);
    match family {
        Family::Normal | Family::StudentT => {
            let sigma = scalar_param(e, p, "sigma")?;
            let normal = e.rng("NORMAL", ty);
            let z = if family == Family::StudentT {
                let nu = scalar_param(e, p, "nu")?;
                let inf = e.inf_like(&nu);
                let finite = e.compare("LT", &nu, &inf);
                let safe_nu = e.select(&finite, &nu, &one);
                let half = e.scalar(0.5);
                let shape = e.mul(&safe_nu, &half);
                let chi = draw_gamma(e, &shape, &half);
                let ratio = e.div(&chi, &safe_nu);
                let denominator = e.sqrt(&ratio);
                let student = e.div(&normal, &denominator);
                e.select(&finite, &student, &normal)
            } else {
                normal
            };
            let z = e.mul(&sigma, &z);
            Ok(e.add(eta, &z))
        }
        Family::Bernoulli => {
            let prob = e.invlogit(eta);
            let u = e.rng("UNIFORM", ty);
            let success = e.compare("LT", &u, &prob);
            Ok(e.select(&success, &one, &zero))
        }
        Family::Binomial => {
            let trials = literal_fixed_positive_int(e, p, "n", "BinomialLogitGLM", "sample")?;
            let draw_ty = MlirTy::Ranked(vec![Some(lanes), Some(trials)]);
            e.set_batch_shape(vec![lanes, trials]);
            let u = e.rng("UNIFORM", &draw_ty);
            e.set_batch_shape(vec![lanes]);
            let prob = e.invlogit(eta);
            let prob = e.broadcast_in_dim(&prob, &[0], draw_ty);
            let success = e.compare("LT", &u, &prob);
            let indicators = e.select(&success, &one, &zero);
            Ok(e.reduce_sum_last_axis(&indicators))
        }
        Family::Poisson => {
            let rate = e.exp(eta);
            Ok(draw_poisson(e, &rate))
        }
        Family::NegativeBinomial => {
            let psi = scalar_param(e, p, "psi")?;
            let log_gamma = draw_gamma_scaled(e, &psi, &one, GammaScale::Log);
            let log_psi = e.log(&psi);
            let rate_log = e.add(&log_gamma, eta);
            let rate_log = e.sub(&rate_log, &log_psi);
            let rate = e.exp(&rate_log);
            Ok(draw_poisson(e, &rate))
        }
        Family::Gamma => {
            let shape = scalar_param(e, p, "shape")?;
            let log_draw = draw_gamma_scaled(e, &shape, &shape, GammaScale::Log);
            let log_draw = e.add(&log_draw, eta);
            Ok(e.exp(&log_draw))
        }
        Family::Beta => {
            let precision = scalar_param(e, p, "precision")?;
            let mean = e.invlogit(eta);
            let negative = e.neg(eta);
            let complement = e.invlogit(&negative);
            let a = e.mul(&precision, &mean);
            let b = e.mul(&precision, &complement);
            let ga = draw_gamma_scaled(e, &a, &one, GammaScale::Log);
            let gb = draw_gamma_scaled(e, &b, &one, GammaScale::Log);
            let gap = e.sub(&ga, &gb);
            Ok(e.invlogit(&gap))
        }
        Family::Categorical => {
            let dims = dimensions(e, p, eta, 2, "logits")?;
            let classes = dims[1];
            let maxima = e.reduce_max_last_axis(eta);
            let maxima = e.broadcast_in_dim(&maxima, &[0], eta.ty.clone());
            let centered = e.sub(eta, &maxima);
            let weights = e.exp(&centered);
            let totals = e.reduce_sum_last_axis(&weights);
            let u = e.rng("UNIFORM", ty);
            let target = e.mul(&u, &totals);
            let mut cumulative = e.constant(0.0, ty.clone());
            let mut result = e.constant(1.0, ty.clone());
            for class in 0..classes - 1 {
                let column = row_column(e, &weights, lanes, class);
                cumulative = e.add(&cumulative, &column);
                let above = e.compare("GE", &target, &cumulative);
                let increment = e.select(&above, &one, &zero);
                result = e.add(&result, &increment);
            }
            Ok(result)
        }
        Family::Ordered => {
            let cuts = p.get(e, "cutpoints")?;
            let count = dimensions(e, p, &cuts, 1, "cutpoints")?[0];
            let u = e.rng("UNIFORM", ty);
            let mut result = e.constant(1.0, ty.clone());
            for c in 0..count {
                let cut = vector_elem(e, &cuts, c);
                let arg = e.sub(&cut, eta);
                let cumulative = e.invlogit(&arg);
                let above = e.compare("GE", &u, &cumulative);
                let increment = e.select(&above, &one, &zero);
                result = e.add(&result, &increment);
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_normalizer_retains_large_nu_precision() {
        // For even nu=2m, Gamma(m+.5)/Gamma(m) is an exact central-binomial
        // coefficient times sqrt(pi). Independent log-factorial evaluation.
        for dtype in [Dtype::F32, Dtype::F64] {
            let module = flatppl_core::Module::new();
            let mut e = Emitter::new(&module, dtype);
            for (nu, expected) in [(1000.0, -0.9191885331630055), (10000.0, -0.918963533204631)] {
                let nu = e.scalar(nu);
                let log_sigma = e.scalar(0.0);
                let value = student_normalizer(&mut e, &nu, &log_sigma);
                let value = e.constant_real(&value).expect("constant normalizer");
                let tolerance = if matches!(dtype, Dtype::F32) {
                    1e-6
                } else {
                    1e-12
                };
                assert!(
                    (value - expected).abs() < tolerance,
                    "{value} != {expected}"
                );
            }
        }
    }
}
