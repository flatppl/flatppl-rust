//! Log density from piecewise Chebyshev expansions of §09's contour integral.
//! Center coefficients use the saddle contour / negative-axis branch cut;
//! the tails retain their analytic exponential and x⁻² factors. This keeps
//! event arrays pointwise, with no per-event quadrature workspace.

use super::*;

const C0: &[f64] = &[
    0.0003790068145087726,
    0.00037815893696633223,
    -8.428752146536323e-07,
    4.953022947491763e-09,
    -4.862285377377624e-11,
    6.697421150686729e-13,
    -1.1869890562432189e-14,
    2.4249083428607375e-16,
    -1.7172886266853787e-17,
    -8.953686834526861e-17,
    3.450421999889727e-17,
    4.091228741012911e-17,
    -1.197611918405868e-16,
    -3.844101329129859e-17,
    -9.475021721060931e-17,
    2.087089182034596e-17,
];
const C1: &[f64] = &[
    -16.752500800284864,
    22.520467732760405,
    -10.176057666947349,
    3.145178546990003,
    -0.7494853169314034,
    0.14521354395166441,
    -0.023650187282707066,
    0.0033192065903898307,
    -0.00040933909746776394,
    4.50024850412868e-05,
    -4.457162414041176e-06,
    4.019571094703048e-07,
    -3.338166207440302e-08,
    2.5537300911686902e-09,
    -1.784972848496734e-10,
    1.2037462534276547e-11,
    -8.406053269548649e-13,
    3.913478337687811e-14,
    2.8171909249863347e-15,
    -2.609024107869118e-15,
    1.5543122344752192e-15,
    -1.4802973661668755e-15,
    4.144832625267251e-15,
    -4.736951571734001e-15,
];
const C2: &[f64] = &[
    -1.8129389630688115,
    -0.03766461691332183,
    -0.09376144438124148,
    0.016607709119119567,
    -0.0020234461451885226,
    0.0001986714823802417,
    -1.6488283275832718e-05,
    1.1804279660199641e-06,
    -7.370742869382809e-08,
    4.0552420314604414e-09,
    -1.9995486162145562e-10,
    9.096446103415597e-12,
    -3.9314298344583776e-13,
    1.5803330866148713e-14,
    -5.759281940243e-16,
    0.0,
];
const C3: &[f64] = &[
    -2.59055345861203,
    -0.6631166591641323,
    0.007830666430405888,
    0.007525183578676325,
    -0.0020224958240225685,
    0.0003759926810021574,
    -5.8512653865345276e-05,
    8.03357776200477e-06,
    -9.926745476525124e-07,
    1.1118450283751885e-07,
    -1.1288966546695603e-08,
    1.0338060225587031e-09,
    -8.460921763556173e-11,
    6.125050987237923e-12,
    -3.93123901487602e-13,
    2.3869795029440866e-14,
];
const C4: &[f64] = &[
    -4.6939164854993365,
    -1.2582351902814652,
    0.16681914204597673,
    -0.02529661902383597,
    0.0034373807250621564,
    -0.00025731604088221146,
    -6.852793340180622e-05,
    4.658269061553149e-05,
    -1.80212352118411e-05,
    5.825036282035422e-06,
    -1.7110053877768626e-06,
    4.715666487480484e-07,
    -1.23823110129348e-07,
    3.122905028980231e-08,
    -7.599141615125381e-09,
    1.7882620208352943e-09,
    -4.0730882555402634e-10,
    8.975923364220196e-11,
    -1.9107955908545585e-11,
    3.918017060962342e-12,
    -7.699116228813511e-13,
    1.4397510961217108e-13,
    -2.531308496145357e-14,
    4.070817756958907e-15,
];
const C5: &[f64] = &[
    0.12728681550720689,
    -0.056672312845742506,
    0.0017118117092438856,
    0.0013147256349073375,
    -0.00032745352596508324,
    4.4906349703491296e-05,
    -3.839484029704904e-06,
    8.385490941639898e-08,
    3.887759602707133e-08,
    -8.335783452309462e-09,
    9.496340877476513e-10,
    -4.728797446323994e-11,
    -6.109761161625718e-12,
    1.8852057501878017e-12,
    -2.518702477485868e-13,
    1.6493750809587482e-14,
];
const C6: &[f64] = &[
    0.036618195562376,
    -0.030010240090478667,
    0.006027018882425187,
    -0.0006812984664827254,
    2.652005997395123e-05,
    6.4020694197550776e-06,
    -1.7785961235381192e-06,
    2.8381958533974236e-07,
    -3.5770635993231234e-08,
    3.810215743329557e-09,
    -3.3584208642363825e-10,
    2.0555220820348913e-11,
    1.756062082449968e-13,
    -3.2248151631691924e-13,
    6.686318165805005e-14,
    -9.63291946209921e-15,
];
const C7: &[f64] = &[
    0.005650177787327348,
    -0.004988085641962577,
    0.001152603222129216,
    -0.00017404786629167648,
    1.865996035373853e-05,
    -1.4296973645703198e-06,
    6.778793812846574e-08,
    4.0662762678097056e-10,
    -5.124808476715638e-10,
    7.137267745521243e-11,
    -7.277903736204999e-12,
    6.4948877869305e-13,
    -5.3936374680776944e-14,
    4.5867172906999265e-15,
    -9.725293487194975e-17,
    3.946495907847236e-17,
];
const C8: &[f64] = &[
    0.0005037238560514634,
    -0.0007138933855335174,
    0.0003143668547514475,
    -9.777060155713967e-05,
    2.310619671211258e-05,
    -4.3330246365637045e-06,
    6.60373567618907e-07,
    -8.234584254599802e-08,
    8.247626287379196e-09,
    -6.091750853241718e-10,
    1.9733878436806703e-11,
    3.2302590367214157e-12,
    -8.410915244312775e-13,
    1.2729451688694646e-13,
    -1.5973686132500498e-14,
    2.334449907687164e-15,
];
const C9: &[f64] = &[
    7.647516546027688e-06,
    -1.1010840120178135e-05,
    5.000582581091671e-06,
    -1.6231167878056395e-06,
    4.056562823085075e-07,
    -8.185589586669334e-08,
    1.3768344308820402e-08,
    -1.9735799117394557e-09,
    2.4478526477109394e-10,
    -2.6528323647680047e-11,
    2.5214547533401968e-12,
    -2.0999580602962575e-13,
    1.53496421260283e-14,
    -3.5566796988177824e-16,
    3.693135647829266e-16,
    2.7669178255008975e-16,
];

struct Region {
    lower: f64,
    upper: f64,
    logarithmic: bool,
    coefficients: &'static [f64],
}
const REGIONS: &[Region] = &[
    Region {
        lower: -5.0,
        upper: -1.0,
        logarithmic: false,
        coefficients: C1,
    },
    Region {
        lower: -1.0,
        upper: 1.0,
        logarithmic: false,
        coefficients: C2,
    },
    Region {
        lower: 1.0,
        upper: 5.0,
        logarithmic: false,
        coefficients: C3,
    },
    Region {
        lower: 5.0,
        upper: 20.0,
        logarithmic: false,
        coefficients: C4,
    },
    Region {
        lower: 2.995732273553991,
        upper: 4.605170185988092,
        logarithmic: true,
        coefficients: C5,
    },
    Region {
        lower: 4.605170185988092,
        upper: 6.907755278982137,
        logarithmic: true,
        coefficients: C6,
    },
    Region {
        lower: 6.907755278982137,
        upper: 9.210340371976184,
        logarithmic: true,
        coefficients: C7,
    },
    Region {
        lower: 9.210340371976184,
        upper: 13.815510557964274,
        logarithmic: true,
        coefficients: C8,
    },
    Region {
        lower: 13.815510557964274,
        upper: 18.420680743952367,
        logarithmic: true,
        coefficients: C9,
    },
];

fn chebyshev(e: &mut Emitter, x: &Value, coefficients: &[f64]) -> Value {
    let mut b1 = e.scalar(0.0);
    let mut b2 = e.scalar(0.0);
    let two = e.scalar(2.0);
    let twice_x = e.mul(&two, x);
    for coefficient in coefficients[1..].iter().rev() {
        let product = e.mul(&twice_x, &b1);
        let difference = e.sub(&product, &b2);
        let c = e.scalar(*coefficient);
        let b0 = e.add(&difference, &c);
        b2 = b1;
        b1 = b0;
    }
    let product = e.mul(x, &b1);
    let value = e.sub(&product, &b2);
    let c = e.scalar(coefficients[0]);
    e.add(&value, &c)
}

pub(super) fn logpdf(e: &mut Emitter, p: &Params, x: &Value) -> Result<Value, EmitError> {
    let loc = p.get(e, "loc")?;
    let scale = p.get(e, "scale")?;
    let difference = e.sub(x, &loc);
    let z = e.div(&difference, &scale);
    let magnitude = e.abs(&z);
    let infinity = e.inf_like(&z);
    let finite = e.compare("LT", &magnitude, &infinity);
    let zero = e.scalar(0.0);
    let safe_z = e.select(&finite, &z, &zero);
    let minus_five = e.scalar(-5.0);
    let left = e.compare("LE", &safe_z, &minus_five);
    let left_x = e.select(&left, &safe_z, &minus_five);
    let one = e.scalar(1.0);
    let exponent = e.add(&left_x, &one);
    let u = e.exp(&exponent);
    let negative = e.neg(&exponent);
    let c = e.exp(&negative);
    let coefficient = e.scalar(2.0 / (-4.0_f64).exp());
    let coordinate = e.mul(&coefficient, &u);
    let coordinate = e.sub(&coordinate, &one);
    let correction = chebyshev(e, &coordinate, C0);
    let half = e.scalar(0.5);
    let half_exponent = e.mul(&half, &exponent);
    let left_base = e.neg(&c);
    let left_base = e.sub(&left_base, &half_exponent);
    let constant = e.scalar(0.5 * (2.0 * std::f64::consts::PI).ln());
    let left_base = e.sub(&left_base, &constant);
    let mut result = e.add(&left_base, &correction);
    let twenty = e.scalar(20.0);
    let right = e.compare("GE", &safe_z, &twenty);
    let right_x = e.select(&right, &safe_z, &twenty);
    let log_right = e.log(&right_x);
    let two = e.scalar(2.0);
    let twice_log_right = e.mul(&two, &log_right);
    for region in REGIONS {
        let raw = if region.logarithmic {
            &log_right
        } else {
            &safe_z
        };
        let lower = e.scalar(region.lower);
        let upper = e.scalar(region.upper);
        let above = e.compare("GE", raw, &lower);
        let below = e.compare("LE", raw, &upper);
        let within = e.and(&above, &below);
        let within = if region.logarithmic {
            e.and(&within, &right)
        } else {
            within
        };
        let midpoint = e.scalar(0.5 * (region.lower + region.upper));
        let input = e.select(&within, raw, &midpoint);
        let distance = e.sub(&input, &midpoint);
        let multiplier = e.scalar(2.0 / (region.upper - region.lower));
        let coordinate = e.mul(&multiplier, &distance);
        let value = chebyshev(e, &coordinate, region.coefficients);
        let value = if region.logarithmic {
            e.sub(&value, &twice_log_right)
        } else {
            value
        };
        result = e.select(&within, &value, &result);
    }
    // Watson expansion of the branch-cut integral, through x⁻⁴. The first
    // omitted relative term is O((log(x)/x)³), negligible from 10^8 upward.
    let threshold = e.scalar(1e8);
    let far = e.compare("GE", &safe_z, &threshold);
    let far_x = e.select(&far, &safe_z, &threshold);
    let log_x = e.log(&far_x);
    let inverse = e.div(&one, &far_x);
    let twice_log = e.mul(&two, &log_x);
    let euler = std::f64::consts::EULER_GAMMA;
    let constant = e.scalar(2.0 * euler - 3.0);
    let first = e.add(&twice_log, &constant);
    let constant = e.scalar(euler - 11.0 / 6.0);
    let shifted = e.add(&log_x, &constant);
    let square = e.mul(&shifted, &shifted);
    let three = e.scalar(3.0);
    let second = e.mul(&three, &square);
    let constant = e.scalar(std::f64::consts::PI.powi(2) / 2.0 + 49.0 / 12.0);
    let second = e.sub(&second, &constant);
    let second = e.mul(&second, &inverse);
    let relative = e.add(&first, &second);
    let relative = e.mul(&relative, &inverse);
    let correction = e.log1p(&relative);
    let far_value = e.sub(&correction, &twice_log);
    result = e.select(&far, &far_value, &result);
    let negative_infinity = e.constant_like(f64::NEG_INFINITY, &result);
    result = e.select(&finite, &result, &negative_infinity);
    let nan = e.compare("NE", &z, &z);
    let nan_value = e.constant_like(f64::NAN, &result);
    result = e.select(&nan, &nan_value, &result);
    let log_scale = e.log(&scale);
    Ok(e.sub(&result, &log_scale))
}
