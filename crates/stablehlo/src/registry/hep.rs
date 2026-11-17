//! Normalized particle-physics densities and direct sampling transforms.

use super::*;

mod landau;

fn coordinate_batch(e: &mut Emitter, value: &Value) -> (Value, Axes) {
    let axes = e.axes_of(value);
    let rank = match &value.ty {
        MlirTy::Ranked(d) => d.len(),
        _ => 0,
    };
    (
        e.axes_view(
            value,
            Axes {
                batch: rank,
                layers: vec![],
            },
        ),
        axes,
    )
}

fn integral(
    e: &mut Emitter,
    upper: &Value,
    template: &Value,
    maximum_panels: u64,
    mut f: impl FnMut(&mut Emitter, &Value) -> Value,
) -> Value {
    let mut previous = None;
    let mut chosen = e.scalar(f64::NAN);
    let mut accepted = e.bool_value_const(false);
    let eps = e.scalar(match e.dtype() {
        Dtype::F32 => 32.0 * f32::EPSILON as f64,
        Dtype::F64 => 128.0 * f64::EPSILON,
    });
    for panels in [32_u64, 64, 128, 256]
        .into_iter()
        .take_while(|&n| n <= maximum_panels)
    {
        let count = e.scalar(panels as f64);
        let (value, magnitude) = e.gauss_legendre16(upper, template, &count, panels * 16, &mut f);
        if let Some(previous) = previous {
            let error = e.sub(&value, &previous);
            let error = e.abs(&error);
            let tolerance = e.mul(&eps, &magnitude);
            let passed = e.compare("LE", &error, &tolerance);
            let pending = e.not(&accepted);
            let take = e.and(&pending, &passed);
            chosen = e.select(&take, &value, &chosen);
            accepted = e.or(&accepted, &passed);
        }
        previous = Some(value);
    }
    e.record_runtime_error(&accepted, 10);
    chosen
}

pub(super) fn landau_logpdf(e: &mut Emitter, p: &Params, x: &Value) -> Result<Value, EmitError> {
    landau::logpdf(e, p, x)
}

// Weideman, SIAM J. Numer. Anal. 31 (1994), doi:10.1137/0731077.
const FADDEEVA_L: f64 = 6.727171322029716;
const FADDEEVA_A: [f64; 64] = [
    0.0,
    -9.055037244324897e-17,
    -1.1653209979568358e-16,
    -1.1716769801049914e-16,
    1.2109620530784368e-16,
    -1.2605218122250793e-17,
    -2.18212715384761e-16,
    -2.2104253016558195e-16,
    -1.0824968888340203e-16,
    -2.2944355346394034e-16,
    -8.964015185669267e-17,
    -2.124476077622484e-17,
    -3.357657519909759e-17,
    2.3581813318296436e-17,
    -6.8871462743752275e-18,
    -4.376637422574319e-17,
    7.417563524085893e-17,
    5.389074992350545e-17,
    7.793604873008101e-17,
    -5.193078949160759e-17,
    2.384529504876383e-16,
    6.186473856582468e-16,
    -7.657088729755898e-16,
    -4.2506362862487974e-15,
    -2.0190295016094956e-16,
    3.299454337156331e-14,
    5.909242235017462e-14,
    -1.549385220783347e-13,
    -7.920822247734917e-13,
    -3.939344590984745e-13,
    5.832585737473097e-12,
    1.7501502908887984e-11,
    -6.4707349345185925e-12,
    -1.7560607958960022e-10,
    -4.533915068936149e-10,
    2.4434786854343034e-10,
    5.186955616565486e-09,
    1.592681388741325e-08,
    7.435710708457646e-09,
    -1.3610261250028284e-07,
    -6.650424120776997e-07,
    -1.5547722782660322e-06,
    -7.564244111822494e-08,
    1.7901801586212798e-05,
    0.00010227006798923578,
    0.0003962745103981355,
    0.0012549788049982572,
    0.003460207948107515,
    0.008565381413176008,
    0.019380399024538295,
    0.040552846529580175,
    0.07911655067602572,
    0.14477859973586424,
    0.24963969994535562,
    0.4070443030398735,
    0.6293868343374367,
    0.9249760252638086,
    1.2944377517175158,
    1.727506085787117,
    2.20125657128641,
    2.680732639559084,
    3.1224481894020366,
    3.4804961039850415,
    3.7141697931977027,
];

type Pair = (Value, Value);

fn complex_product(e: &mut Emitter, a: &Pair, b: &Pair) -> Pair {
    let rr = e.mul(&a.0, &b.0);
    let ii = e.mul(&a.1, &b.1);
    let ri = e.mul(&a.0, &b.1);
    let ir = e.mul(&a.1, &b.0);
    (e.sub(&rr, &ii), e.add(&ri, &ir))
}

fn faddeeva_rational(e: &mut Emitter, x: &Value, y: &Value) -> Pair {
    let l = e.scalar(FADDEEVA_L);
    let dr = e.add(&l, y);
    let dr2 = e.mul(&dr, &dr);
    let x2 = e.mul(x, x);
    let norm2 = e.add(&dr2, &x2);
    let inverse = (e.div(&dr, &norm2), e.div(x, &norm2));
    let twice_l = e.scalar(2.0 * FADDEEVA_L);
    let one = e.scalar(1.0);
    let zr = e.mul(&twice_l, &inverse.0);
    let zr = e.sub(&zr, &one);
    let zi = e.mul(&twice_l, &inverse.1);
    let z = (zr, zi);
    let mut p = (e.scalar(FADDEEVA_A[0]), e.scalar(0.0));
    for coefficient in &FADDEEVA_A[1..] {
        p = complex_product(e, &p, &z);
        let c = e.scalar(*coefficient);
        p.0 = e.add(&p.0, &c);
    }
    let square = complex_product(e, &inverse, &inverse);
    let product = complex_product(e, &p, &square);
    let two = e.scalar(2.0);
    let re = e.mul(&two, &product.0);
    let im = e.mul(&two, &product.1);
    let constant = e.scalar(1.0 / std::f64::consts::PI.sqrt());
    let direct_re = e.mul(&constant, &inverse.0);
    let direct_im = e.mul(&constant, &inverse.1);
    (e.add(&re, &direct_re), e.add(&im, &direct_im))
}

fn faddeeva_real(e: &mut Emitter, x: &Value, y: &Value) -> Value {
    let x = e.abs(x);
    let maximum = e.max(&x, y);
    let eight = e.scalar(8.0);
    let large = e.compare("GT", &maximum, &eight);
    let threshold = e.scalar(0.01);
    let near_axis = e.compare("LT", y, &threshold);
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let regular_x = e.select(&large, &zero, &x);
    let regular_y = e.select(&large, &one, y);
    let rational = faddeeva_rational(e, &regular_x, &regular_y).0;

    // Near the real axis, direct real-part subtraction loses the small Voigt
    // wing. Taylor-expand in Im(z), starting with the exact exp(-x²) real part.
    let axis_x = e.select(&large, &zero, &x);
    let axis_y = e.select(&near_axis, y, &zero);
    let wi = faddeeva_rational(e, &axis_x, &zero).1;
    let square = e.mul(&axis_x, &axis_x);
    let negative = e.neg(&square);
    let mut w0 = (e.exp(&negative), wi);
    let minus_two = e.scalar(-2.0);
    let coefficient = e.mul(&minus_two, &axis_x);
    let source = e.scalar(2.0 / std::f64::consts::PI.sqrt());
    let dr = e.mul(&coefficient, &w0.0);
    let di = e.mul(&coefficient, &w0.1);
    let mut w1 = (dr, e.add(&di, &source));
    let mut power = (zero.clone(), axis_y.clone());
    let term = complex_product(e, &w1, &power);
    let mut series = e.add(&w0.0, &term.0);
    for n in 1..12 {
        let cr = e.mul(&coefficient, &w1.0);
        let ci = e.mul(&coefficient, &w1.1);
        let two_re = e.mul(&minus_two, &w0.0);
        let two_im = e.mul(&minus_two, &w0.1);
        let re = e.add(&cr, &two_re);
        let im = e.add(&ci, &two_im);
        let denominator = e.scalar((n + 1) as f64);
        let next = (e.div(&re, &denominator), e.div(&im, &denominator));
        let pr = e.mul(&power.1, &axis_y);
        let pr = e.neg(&pr);
        let pi = e.mul(&power.0, &axis_y);
        power = (pr, pi);
        let term = complex_product(e, &next, &power);
        series = e.add(&series, &term.0);
        w0 = w1;
        w1 = next;
    }
    let small = e.select(&near_axis, &series, &rational);

    // The algebraic upper-half-plane expansion retains relative accuracy in
    // far Lorentzian wings. Its Gaussian term matters on the real boundary.
    let asym_x = e.select(&large, &x, &one);
    let asym_y = e.select(&large, y, &one);
    let scale = e.max(&asym_x, &asym_y);
    let xr = e.div(&asym_x, &scale);
    let yi = e.div(&asym_y, &scale);
    let xx = e.mul(&xr, &xr);
    let yy = e.mul(&yi, &yi);
    let norm = e.add(&xx, &yy);
    let ir = e.div(&xr, &norm);
    let ii = e.div(&yi, &norm);
    let ir = e.div(&ir, &scale);
    let ii = e.div(&ii, &scale);
    let inverse = (ir, e.neg(&ii));
    let inverse_square = complex_product(e, &inverse, &inverse);
    let mut term = inverse.clone();
    let mut sum = inverse;
    for n in 1..=32 {
        term = complex_product(e, &term, &inverse_square);
        let c = e.scalar(n as f64 - 0.5);
        term = (e.mul(&c, &term.0), e.mul(&c, &term.1));
        sum = (e.add(&sum.0, &term.0), e.add(&sum.1, &term.1));
    }
    let constant = e.scalar(-1.0 / std::f64::consts::PI.sqrt());
    let algebraic = e.mul(&constant, &sum.1);
    let thirty_two = e.scalar(32.0);
    let gx = e.min(&x, &thirty_two);
    let gy = e.select(&near_axis, y, &zero);
    let gx2 = e.mul(&gx, &gx);
    let gy2 = e.mul(&gy, &gy);
    let exponent = e.sub(&gy2, &gx2);
    let phase = e.mul(&gx, &gy);
    let phase = e.mul(&minus_two, &phase);
    let cosine = e.cos(&phase);
    let exponential = e.exp(&exponent);
    let gaussian = e.mul(&exponential, &cosine);
    let gaussian = e.select(&near_axis, &gaussian, &zero);
    let asymptotic = e.add(&algebraic, &gaussian);
    e.select(&large, &asymptotic, &small)
}

pub(super) fn voigtian_logpdf(e: &mut Emitter, p: &Params, x: &Value) -> Result<Value, EmitError> {
    let mean = p.get(e, "mean")?;
    let width = p.get(e, "width")?;
    let sigma = p.get(e, "sigma")?;
    let difference = e.sub(x, &mean);
    let magnitude = e.abs(&difference);
    let cauchy_scale = e.max(&magnitude, &width);
    let threshold = e.scalar(match e.dtype() {
        Dtype::F32 => 0.02,
        Dtype::F64 => 0.001,
    });
    let narrow = e.mul(&threshold, &cauchy_scale);
    let lorentzian = e.compare("LT", &sigma, &narrow);
    let safe_sigma = e.select(&lorentzian, &cauchy_scale, &sigma);
    let inv_root_two = e.scalar(std::f64::consts::FRAC_1_SQRT_2);
    let relative_x = e.div(&difference, &safe_sigma);
    let zr = e.mul(&relative_x, &inv_root_two);
    let relative_width = e.div(&width, &safe_sigma);
    let half_inv_root_two = e.scalar(0.5 * std::f64::consts::FRAC_1_SQRT_2);
    let zi = e.mul(&relative_width, &half_inv_root_two);
    let profile = faddeeva_real(e, &zr, &zi);
    let log_profile = e.log(&profile);
    let log_sigma = e.log(&safe_sigma);
    let log_root_two_pi = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
    let regular = e.sub(&log_profile, &log_sigma);
    let regular = e.sub(&regular, &log_root_two_pi);

    // Gaussian convolution of the Cauchy profile: exp(σ² ∂²/2) f_Cauchy.
    // Four even moments retain the small-sigma derivatives without subtracting
    // normalization terms of order 1/sigma. The ratios are bounded by `narrow`.
    let dx = e.div(&difference, &cauchy_scale);
    let relative_width = e.div(&width, &cauchy_scale);
    let half = e.scalar(0.5);
    let dg = e.mul(&relative_width, &half);
    let dx2 = e.mul(&dx, &dx);
    let dg2 = e.mul(&dg, &dg);
    let norm = e.add(&dx2, &dg2);
    let norm2 = e.mul(&norm, &norm);
    let a = e.div(&dx2, &norm2);
    let b = e.div(&dg2, &norm2);
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let guarded_sigma = e.select(&lorentzian, &sigma, &zero);
    let relative_sigma = e.div(&guarded_sigma, &cauchy_scale);
    let q = e.mul(&relative_sigma, &relative_sigma);
    let mut ap = vec![one.clone()];
    let mut bp = vec![one.clone()];
    for n in 1..=4 {
        let an = e.mul(&ap[n - 1], &a);
        let bn = e.mul(&bp[n - 1], &b);
        ap.push(an);
        bp.push(bn);
    }
    let polynomials: &[&[f64]] = &[
        &[3.0, -1.0],
        &[5.0, -10.0, 1.0],
        &[7.0, -35.0, 21.0, -1.0],
        &[9.0, -84.0, 126.0, -36.0, 1.0],
    ];
    let mut qn = one;
    let mut correction = zero;
    for (index, coefficients) in polynomials.iter().enumerate() {
        let n = index + 1;
        let mut polynomial = e.scalar(0.0);
        for (k, coefficient) in coefficients.iter().enumerate() {
            let term = e.mul(&ap[n - k], &bp[k]);
            let constant = e.scalar(*coefficient);
            let term = e.mul(&term, &constant);
            polynomial = e.add(&polynomial, &term);
        }
        qn = e.mul(&qn, &q);
        let moment = e.scalar([1.0, 3.0, 15.0, 105.0][index]);
        let term = e.mul(&qn, &moment);
        let term = e.mul(&term, &polynomial);
        correction = e.add(&correction, &term);
    }
    let correction = e.log1p(&correction);
    let log_width = e.log(&width);
    let log_two_pi = e.scalar((2.0 * std::f64::consts::PI).ln());
    let log_norm = e.log(&norm);
    let log_scale = e.log(&cauchy_scale);
    let two = e.scalar(2.0);
    let log_scale = e.mul(&two, &log_scale);
    let cauchy = e.sub(&log_width, &log_two_pi);
    let cauchy = e.sub(&cauchy, &log_scale);
    let cauchy = e.sub(&cauchy, &log_norm);
    let cauchy = e.add(&cauchy, &correction);
    Ok(e.select(&lorentzian, &cauchy, &regular))
}

fn expm1_minus_x(e: &mut Emitter, x: &Value) -> Value {
    let magnitude = e.abs(x);
    let threshold = e.scalar(0.125);
    let small = e.compare("LT", &magnitude, &threshold);
    let zero = e.scalar(0.0);
    let safe = e.select(&small, x, &zero);
    let mut factorial = 1.0;
    let coefficients: Vec<_> = (1..=24)
        .filter_map(|n| {
            factorial *= n as f64;
            (n >= 2).then_some(1.0 / factorial)
        })
        .collect();
    let polynomial = horner(e, &safe, &coefficients);
    let square = e.mul(&safe, &safe);
    let series = e.mul(&square, &polynomial);
    let safe_direct = e.select(&small, &zero, x);
    let exponential = e.expm1(&safe_direct);
    let direct = e.sub(&exponential, &safe_direct);
    e.select(&small, &series, &direct)
}

/// log(∫₀¹ t^power exp(slope*t)dt) - slope. The coordinate s=-rate*log(t)
/// keeps the endpoint maximum resolved for both large positive slopes and
/// negative slopes approaching -power. Strong negative slopes use Gamma P.
fn argus_endpoint_integral(e: &mut Emitter, slope: &Value, power: &Value) -> Value {
    let (slope, power) = e.broadcast_pair(slope, power);
    let (slope, axes) = coordinate_batch(e, &slope);
    let (power, _) = coordinate_batch(e, &power);
    let one = e.scalar(1.0);
    let shape = e.add(&power, &one);
    let linear = e.add(&shape, &slope);
    let root = e.sqrt(&shape);
    let rate = e.add(&linear, &root);
    let kernel = |e: &mut Emitter, s: &Value| {
        let q = e.div(s, &rate);
        let minus_q = e.neg(&q);
        let remainder = expm1_minus_x(e, &minus_q);
        let linear_term = e.mul(&linear, &q);
        let nonlinear = e.mul(&slope, &remainder);
        e.sub(&nonlinear, &linear_term)
    };
    let mut upper = e.constant_like(1.0, &slope);
    let cutoff = e.scalar(-50.0);
    let two = e.scalar(2.0);
    for _ in 0..12 {
        let value = kernel(e, &upper);
        let extend = e.compare("GT", &value, &cutoff);
        let doubled = e.mul(&upper, &two);
        upper = e.select(&extend, &doubled, &upper);
    }
    let value = kernel(e, &upper);
    let bounded = e.compare("LE", &value, &cutoff);
    e.record_runtime_error(&bounded, 10);
    let value = integral(e, &upper, &slope, 256, |e, s| {
        let exponent = kernel(e, s);
        e.exp(&exponent)
    });
    let log_value = e.log(&value);
    let log_rate = e.log(&rate);
    let result = e.sub(&log_value, &log_rate);
    e.axes_view(&result, axes)
}

pub(super) fn argus_logpdf(e: &mut Emitter, p: &Params, x: &Value) -> Result<Value, EmitError> {
    let resonance = p.get(e, "resonance")?;
    let slope = p.get(e, "slope")?;
    let power = p.get(e, "power")?;
    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let half = e.scalar(0.5);
    let positive = e.compare("GT", x, &zero);
    let below = e.compare("LT", x, &resonance);
    let supported = e.and(&positive, &below);
    let safe = e.mul(&half, &resonance);
    let x = e.select(&supported, x, &safe);
    let ratio = e.div(&x, &resonance);
    let square = e.mul(&ratio, &ratio);
    let gap = e.sub(&resonance, &x);
    let gap = e.div(&gap, &resonance);
    let sum = e.add(&one, &ratio);
    let u = e.mul(&gap, &sum);
    let log_u = e.log(&u);
    let negative_slope = e.neg(&slope);
    let negative = e.compare("LT", &slope, &zero);
    let gamma_region = e.compare("LT", &power, &negative_slope);
    let gamma_region = e.and(&negative, &gamma_region);
    let quad_slope = e.select(&gamma_region, &zero, &slope);
    let quad_power = e.select(&gamma_region, &half, &power);
    let log_scaled = argus_endpoint_integral(e, &quad_slope, &quad_power);
    let u_minus_one = e.neg(&square);
    let shifted_exponent = e.mul(&slope, &u_minus_one);
    let log_power = e.mul(&power, &log_u);
    let direct_kernel = e.add(&shifted_exponent, &log_power);
    let minus_half = e.scalar(-0.5);
    let near_one = e.compare("GT", &u_minus_one, &minus_half);
    let safe_delta = e.select(&near_one, &u_minus_one, &zero);
    let remainder = e.special_log1pmx(&safe_delta);
    let remainder = e.mul(&power, &remainder);
    let linear = e.add(&power, &slope);
    let linear = e.mul(&linear, &safe_delta);
    let stable_kernel = e.add(&remainder, &linear);
    let endpoint = e.select(&near_one, &stable_kernel, &direct_kernel);
    let endpoint = e.sub(&endpoint, &log_scaled);
    let shape = e.add(&power, &one);
    let shape = e.select(&gamma_region, &shape, &one);
    let rate = e.select(&gamma_region, &negative_slope, &one);
    let argument = e.mul(&rate, &u);
    let gamma_prefactor = e.gamma_log_prefactor(&shape, &argument);
    let cdf = e.regularized_gamma(&shape, &rate);
    let log_cdf = e.log(&cdf);
    let gamma = e.sub(&gamma_prefactor, &log_u);
    let gamma = e.sub(&gamma, &log_cdf);
    let kernel = e.select(&gamma_region, &gamma, &endpoint);
    let log_x = e.log(&x);
    let log_resonance = e.log(&resonance);
    let two = e.scalar(2.0);
    let log_resonance2 = e.mul(&two, &log_resonance);
    let log_two = e.scalar(std::f64::consts::LN_2);
    let jacobian = e.add(&log_two, &log_x);
    let jacobian = e.sub(&jacobian, &log_resonance2);
    let value = e.add(&kernel, &jacobian);
    let outside = e.constant_like(f64::NEG_INFINITY, &value);
    Ok(e.select(&supported, &value, &outside))
}

#[derive(Clone, Copy)]
enum ParameterDomain {
    Real,
    Positive,
    TailExponent,
}

fn parameter(
    e: &mut Emitter,
    p: &Params,
    name: &str,
    domain: ParameterDomain,
) -> Result<Value, EmitError> {
    let value = p.get(e, name)?;
    let value = e.convert(&value, ElemKind::Real);
    let absolute = e.abs(&value);
    let infinity = e.inf_like(&value);
    let finite = e.compare("LT", &absolute, &infinity);
    let (valid, fallback) = match domain {
        ParameterDomain::Real => (finite, 0.0),
        ParameterDomain::Positive | ParameterDomain::TailExponent => {
            let boundary = e.scalar(if matches!(domain, ParameterDomain::Positive) {
                0.0
            } else {
                1.0
            });
            let positive = e.compare("GT", &value, &boundary);
            (
                e.and(&finite, &positive),
                if matches!(domain, ParameterDomain::Positive) {
                    1.0
                } else {
                    2.0
                },
            )
        }
    };
    e.record_runtime_error(&valid, 9);
    let fallback = e.scalar(fallback);
    Ok(e.select(&valid, &value, &fallback))
}

fn open_uniform(e: &mut Emitter, template: &Value) -> Value {
    let u = e.rng("UNIFORM", &template.ty);
    let half_epsilon = match e.dtype() {
        Dtype::F32 => 0.5 * f32::EPSILON as f64,
        Dtype::F64 => 0.5 * f64::EPSILON,
    };
    let lower = e.scalar(half_epsilon);
    let upper = e.scalar(1.0 - half_epsilon);
    let u = e.max(&u, &lower);
    e.min(&u, &upper)
}

fn softplus(e: &mut Emitter, x: &Value) -> Value {
    let threshold = e.scalar(30.0);
    let large = e.compare("GT", x, &threshold);
    let zero = e.scalar(0.0);
    let direct_x = e.select(&large, &zero, x);
    let exponential = e.exp(&direct_x);
    let direct = e.log1p(&exponential);
    let negative = e.neg(x);
    let negative = e.select(&large, &negative, &zero);
    let exponential = e.exp(&negative);
    let correction = e.log1p(&exponential);
    let upper = e.add(x, &correction);
    e.select(&large, &upper, &direct)
}

fn logadd(e: &mut Emitter, a: &Value, b: &Value) -> Value {
    let difference = e.sub(a, b);
    let threshold = e.scalar(30.0);
    let large = e.compare("GT", &difference, &threshold);
    let zero = e.scalar(0.0);
    let direct_x = e.select(&large, &zero, &difference);
    let exponential = e.exp(&direct_x);
    let direct = e.log1p(&exponential);
    let direct = e.add(b, &direct);
    let negative = e.neg(&difference);
    let negative = e.select(&large, &negative, &zero);
    let exponential = e.exp(&negative);
    let correction = e.log1p(&exponential);
    let upper = e.add(a, &correction);
    e.select(&large, &upper, &direct)
}

mod crystal_ball;
mod sampling;
pub(super) use crystal_ball::{
    crystal_ball_logpdf, crystal_ball_sample, double_sided_crystal_ball_logpdf,
    double_sided_crystal_ball_sample,
};
pub(super) use sampling::{
    argus_sample, bifurcated_normal_sample, landau_sample, relativistic_breit_wigner_sample,
    voigtian_sample,
};
