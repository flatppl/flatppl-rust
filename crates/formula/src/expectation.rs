use crate::{Family, Recipe, emit::broadcast, family::Observation, response};
use flatppl_convert::Builder;
use flatppl_core::{Node, NodeId, Scalar};

pub(crate) fn emit(
    b: &mut Builder<'_>,
    recipe: &Recipe,
    observation: Observation<'_>,
) -> Option<NodeId> {
    use Family::*;
    let Observation {
        mu,
        trials,
        scale,
        auxiliary,
        cuts,
    } = observation;
    let auxiliary_value = |name| {
        auxiliary
            .iter()
            .find(|(key, _)| *key == name)
            .expect("validated family parameter")
            .1
    };
    if let Some(truncation) = &recipe.formula.truncation {
        let sigma = scale.unwrap_or_else(|| auxiliary_value("sigma"));
        return Some(response::truncated_normal_rows(
            b,
            mu,
            sigma,
            truncation,
            "truncated_mean",
            truncated_normal_mean,
        ));
    }
    let mean = match recipe.family {
        Gaussian
        | Bernoulli
        | Poisson
        | Negbinomial
        | Geometric
        | Gamma
        | Exgaussian
        | Exponential
        | Weibull
        | Beta
        | ZeroInflatedPoisson
        | ZeroInflatedNegbinomial
        | ZeroInflatedBeta
        | HurdleGamma => mu,
        Student => {
            let nu = auxiliary_value("nu");
            let one = b.lit_real(1.0);
            let has_mean = broadcast(b, "gt", &[nu, one]);
            let undefined = b.lit_real(f64::NAN);
            broadcast(b, "ifelse", &[has_mean, mu, undefined])
        }
        Lognormal | HurdleLognormal => {
            let sigma = auxiliary_value("sigma");
            let variance = broadcast(b, "mul", &[sigma, sigma]);
            let half = b.lit_real(0.5);
            let shift = broadcast(b, "mul", &[half, variance]);
            let location = broadcast(b, "add", &[mu, shift]);
            broadcast(b, "exp", &[location])
        }
        Binomial | BetaBinomial | ZeroInflatedBinomial => {
            broadcast(b, "mul", &[trials.expect("validated trial count"), mu])
        }
        HurdlePoisson | HurdleNegbinomial => {
            let log_zero = if recipe.family == HurdleNegbinomial {
                let shape = auxiliary_value("shape");
                let ratio = broadcast(b, "divide", &[mu, shape]);
                let log = broadcast(b, "log1p", &[ratio]);
                let magnitude = broadcast(b, "mul", &[shape, log]);
                broadcast(b, "neg", &[magnitude])
            } else {
                broadcast(b, "neg", &[mu])
            };
            let neg_mass = broadcast(b, "expm1", &[log_zero]);
            let mass = broadcast(b, "neg", &[neg_mass]);
            broadcast(b, "divide", &[mu, mass])
        }
        Cumulative => {
            let disc = auxiliary
                .iter()
                .find_map(|&(name, value)| (name == "disc").then_some(value));
            crate::ordinal::probabilities(b, mu, cuts, disc)
        }
        Acat => crate::adjacent::probabilities(b, mu, cuts),
        Categorical => crate::categorical::probabilities(b, auxiliary),
        Wiener => crate::wiener::expected_time(
            b,
            mu,
            auxiliary_value("bs"),
            auxiliary_value("ndt"),
            auxiliary_value("bias"),
        ),
        VonMises => return None,
    };
    let zero_probability = match recipe.family {
        HurdlePoisson | HurdleNegbinomial | HurdleGamma | HurdleLognormal => "hu",
        ZeroInflatedPoisson | ZeroInflatedNegbinomial | ZeroInflatedBinomial | ZeroInflatedBeta => {
            "zi"
        }
        _ => return Some(mean),
    };
    let probability = auxiliary_value(zero_probability);
    let one = b.lit_real(1.0);
    let nonzero_weight = broadcast(b, "sub", &[one, probability]);
    Some(broadcast(b, "mul", &[nonzero_weight, mean]))
}

fn truncated_normal_mean(
    b: &mut Builder<'_>,
    mean: NodeId,
    scale: NodeId,
    bounds: [NodeId; 2],
) -> NodeId {
    if bounds
        .iter()
        .any(|&bound| matches!(b.m.node(bound), Node::Lit(Scalar::Real(x)) if x.is_infinite()))
    {
        return truncated_normal_flux_mean(b, mean, scale, bounds);
    }
    let [lower, upper] = bounds;
    let zero = b.lit_real(0.0);
    let one = b.lit_real(1.0);
    let half = b.lit_real(0.5);
    let width = b.call("sub", &[upper, lower]);
    let delta = b.call("sub", &[lower, mean]);
    let scaled_width = b.call("divide", &[width, scale]);
    let scaled_delta = b.call("divide", &[delta, scale]);
    let slope = b.call("mul", &[scaled_width, scaled_delta]);
    let square = b.call("mul", &[scaled_width, scaled_width]);
    let curvature = b.call("mul", &[half, square]);
    // In local coordinates x = lower + width * u, the log density relative
    // to the lower endpoint is -u * (slope + curvature * u). With total
    // variation bounded by one, eight-point quadrature resolves both moments
    // without subtracting nearly equal endpoint fluxes.
    let abs_slope = b.call("abs", &[slope]);
    let span = b.call("add", &[abs_slope, curvature]);
    let small_span = b.call("le", &[span, one]);
    let positive_scale = b.call("gt", &[scale, zero]);
    let finite_scale = b.call("isfinite", &[scale]);
    let valid_scale = b.call("land", &[positive_scale, finite_scale]);
    let positive_width = b.call("gt", &[width, zero]);
    let valid = b.call("land", &[positive_width, valid_scale]);
    let local = b.call("land", &[small_span, valid]);
    let slope = b.call("ifelse", &[local, slope, zero]);
    let curvature = b.call("ifelse", &[local, curvature, zero]);
    let fraction = local_mean_fraction(b, slope, curvature);
    let local_mean = b.call("mul", &[width, fraction]);
    let local_mean = b.call("add", &[lower, local_mean]);

    // Keep the inactive flux branch away from collapsed intervals and
    // ill-conditioned ratios before reverse-mode differentiation.
    let negative_one = b.lit_real(-1.0);
    let flux_mean = b.call("ifelse", &[local, zero, mean]);
    let flux_scale = b.call("ifelse", &[local, one, scale]);
    let flux_lower = b.call("ifelse", &[local, negative_one, lower]);
    let flux_upper = b.call("ifelse", &[local, one, upper]);
    let flux = truncated_normal_flux_mean(b, flux_mean, flux_scale, [flux_lower, flux_upper]);
    b.call("ifelse", &[local, local_mean, flux])
}

fn local_mean_fraction(b: &mut Builder<'_>, slope: NodeId, curvature: NodeId) -> NodeId {
    // Eight-point Gauss–Legendre rule on [-1, 1]. Rescale the nodes to
    // [0, 1]. The common weight factor of 1/2 cancels from the ratio.
    const PAIRS: [(f64, f64); 4] = [
        (0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
        (0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
    ];
    let nodes: Vec<_> = PAIRS
        .into_iter()
        .flat_map(|(node, _)| {
            [
                b.lit_real(0.5 * (1.0 - node)),
                b.lit_real(0.5 * (1.0 + node)),
            ]
        })
        .collect();
    let nodes = b.array(&nodes);
    let weights: Vec<_> = PAIRS
        .into_iter()
        .flat_map(|(_, weight)| [b.lit_real(weight); 2])
        .collect();
    let weights = b.array(&weights);
    let quadratic = broadcast(b, "mul", &[curvature, nodes]);
    let linear = broadcast(b, "add", &[slope, quadratic]);
    let exponent = broadcast(b, "mul", &[nodes, linear]);
    let exponent = broadcast(b, "neg", &[exponent]);
    let density = broadcast(b, "exp", &[exponent]);
    let weights = broadcast(b, "mul", &[weights, density]);
    let numerator = broadcast(b, "mul", &[nodes, weights]);
    let numerator = b.call("sum", &[numerator]);
    let denominator = b.call("sum", &[weights]);
    b.call("divide", &[numerator, denominator])
}

fn truncated_normal_flux_mean(
    b: &mut Builder<'_>,
    mean: NodeId,
    scale: NodeId,
    bounds: [NodeId; 2],
) -> NodeId {
    let normal = b.call("Normal", &[mean, scale]);
    let restricted = response::truncate(b, normal, bounds);
    let mass = response::log_mass(b, restricted);
    let zero = b.lit_real(0.0);
    let half_negative = b.lit_real(-0.5);
    let log_sqrt_two_pi = b.lit_real(0.5 * std::f64::consts::TAU.ln());
    let minus_infinity = b.lit_real(f64::NEG_INFINITY);
    let [lower, upper] = bounds.map(|bound| {
        if matches!(b.m.node(bound), Node::Lit(Scalar::Real(x)) if x.is_infinite()) {
            return zero;
        }
        // Remove inactive infinite endpoints before scaling and exponentiation.
        let finite = b.call("isfinite", &[bound]);
        let safe_bound = b.call("ifelse", &[finite, bound, mean]);
        let delta = b.call("sub", &[safe_bound, mean]);
        let z = b.call("divide", &[delta, scale]);
        let squared = b.call("mul", &[z, z]);
        let log_shape = b.call("mul", &[half_negative, squared]);
        let log_density = b.call("sub", &[log_shape, log_sqrt_two_pi]);
        let ratio = b.call("sub", &[log_density, mass]);
        let log_flux = b.call("ifelse", &[finite, ratio, minus_infinity]);
        b.call("exp", &[log_flux])
    });
    let shift = b.call("sub", &[lower, upper]);
    let delta = b.call("mul", &[scale, shift]);
    b.call("add", &[mean, delta])
}
