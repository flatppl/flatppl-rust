use crate::{emit::broadcast, family::Observation};
use flatppl_convert::Builder;
use flatppl_core::NodeId;

fn parameters(observation: Observation<'_>) -> (NodeId, NodeId) {
    let auxiliary = |name| {
        observation
            .auxiliary
            .iter()
            .find(|(key, _)| *key == name)
            .expect("ex-Gaussian auxiliary parameter")
            .1
    };
    let sigma = observation.scale.unwrap_or_else(|| auxiliary("sigma"));
    (sigma, auxiliary("beta"))
}

fn finite_positive(b: &mut Builder<'_>, value: NodeId) -> NodeId {
    let zero = b.lit_real(0.0);
    let positive = b.call("gt", &[value, zero]);
    let finite = b.call("isfinite", &[value]);
    b.call("land", &[positive, finite])
}

pub(crate) fn log_density(b: &mut Builder<'_>, observation: Observation<'_>, y: NodeId) -> NodeId {
    let (sigma, beta) = parameters(observation);
    let sigma_valid = finite_positive(b, sigma);
    let beta_valid = finite_positive(b, beta);
    let one = b.lit_real(1.0);
    let sigma = b.call("ifelse", &[sigma_valid, sigma, one]);
    let beta = b.call("ifelse", &[beta_valid, beta, one]);
    let rate = b.call("divide", &[one, beta]);
    let location = b.call("sub", &[observation.mu, beta]);
    let residual = b.call("sub", &[y, location]);
    let score =
        flatppl_determinizer::exponentially_modified_normal_logpdf(b.m, rate, sigma, residual);
    let valid = b.call("land", &[sigma_valid, beta_valid]);
    let zero_mass = b.lit_real(f64::NEG_INFINITY);
    b.call("ifelse", &[valid, score, zero_mass])
}

pub(crate) fn conditional_observation(
    b: &mut Builder<'_>,
    observation: Observation<'_>,
    rows: &str,
) -> NodeId {
    let (sigma, beta) = parameters(observation);
    let n = b.self_ref(rows);
    let one = b.lit_real(1.0);
    // Expand a constant scale to one independent exponential draw per row.
    let ones = b.call("fill", &[one, n]);
    let rates = broadcast(b, "divide", &[ones, beta]);
    let latent_law = broadcast(b, "Exponential", &[rates]);
    let latent = b.call("draw", &[latent_law]);
    let name = b.alloc_name("exgaussian_delay");
    b.bind(&name, latent);
    let delay = b.self_ref(&name);
    let location = broadcast(b, "sub", &[observation.mu, beta]);
    let mean = broadcast(b, "add", &[location, delay]);
    broadcast(b, "Normal", &[mean, sigma])
}
