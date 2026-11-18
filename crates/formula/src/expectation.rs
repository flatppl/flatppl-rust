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
        Binomial | ZeroInflatedBinomial => {
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
        Cumulative => crate::ordinal::probabilities(b, mu, cuts),
        Categorical => crate::categorical::probabilities(b, auxiliary),
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
