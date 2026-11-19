use crate::{emit::broadcast, family::Observation};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Node, NodeId, Ref, RefNs, graph::special_functions_alias};

pub(crate) fn draw(b: &mut Builder<'_>, observation: Observation<'_>, rows: &str) -> NodeId {
    let trials = observation.trials.expect("validated trial count");
    let phi = observation.parameter("phi");
    let one = b.lit_real(1.0);
    let zero = b.lit_int(0);
    let row_count = b.self_ref(rows);
    let zeros = b.call("zeros", &[row_count]);
    // Even a constant mean and precision need a separate latent probability per row.
    let mu = broadcast(b, "add", &[observation.mu, zeros]);
    let complement = broadcast(b, "sub", &[one, mu]);
    let alpha = broadcast(b, "mul", &[mu, phi]);
    let beta = broadcast(b, "mul", &[complement, phi]);
    let probabilities = broadcast(b, "Beta", &[alpha, beta]);
    let draw = b.call("draw", &[probabilities]);
    let name = b.alloc_name("beta_binomial_probabilities");
    b.bind(&name, draw);
    let probability = b.self_ref(&name);

    // The Binomial constructor takes positive trials. Zero trials produce zero.
    let has_trials = broadcast(b, "gt", &[trials, zero]);
    let one = b.lit_int(1);
    let positive_trials = broadcast(b, "max", &[trials, one]);
    let law = broadcast(b, "Binomial", &[positive_trials, probability]);
    let count = b.call("draw", &[law]);
    let name = b.alloc_name("beta_binomial_counts");
    b.bind(&name, count);
    let count = b.self_ref(&name);
    broadcast(b, "ifelse", &[has_trials, count, zero])
}

pub(crate) fn log_mass(b: &mut Builder<'_>, observation: Observation<'_>, count: NodeId) -> NodeId {
    let trials = observation.trials.expect("validated trial count");
    let phi = observation.parameter("phi");
    let mu = observation.mu;
    let zero = b.lit_int(0);
    let one = b.lit_real(1.0);
    let complement = b.call("sub", &[one, mu]);
    let alpha = b.call("mul", &[mu, phi]);
    let beta = b.call("mul", &[complement, phi]);

    let alpha_positive = b.call("gt", &[alpha, zero]);
    let beta_positive = b.call("gt", &[beta, zero]);
    let positive = b.call("land", &[alpha_positive, beta_positive]);
    let finite = b.call("isfinite", &[phi]);
    let valid_shapes = b.call("land", &[positive, finite]);
    let count_nonnegative = b.call("ge", &[count, zero]);
    let count_within_trials = b.call("le", &[count, trials]);
    let valid_count = b.call("land", &[count_nonnegative, count_within_trials]);

    let [phi, alpha, beta] = [phi, alpha, beta].map(|x| b.call("ifelse", &[valid_shapes, x, one]));
    let count = b.call("ifelse", &[valid_count, count, zero]);
    let trials = b.call("ifelse", &[valid_count, trials, zero]);
    let failures = b.call("sub", &[trials, count]);

    // Combine factorials with the rising factors before taking logs.
    // This avoids subtracting two log-beta values on the scale of phi.
    let alias = special_functions_alias(b.m);
    let name = b.sym("logbeta");
    let logbeta = b.m.alloc(Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }));
    let total = log_factorial_ratio(b, logbeta, phi, trials);
    let successes = log_factorial_ratio(b, logbeta, alpha, count);
    let failures = log_factorial_ratio(b, logbeta, beta, failures);
    let score = b.call("sub", &[total, successes]);
    let score = b.call("sub", &[score, failures]);
    let valid = b.call("land", &[valid_shapes, valid_count]);
    let impossible = b.lit_real(f64::NEG_INFINITY);
    b.call("ifelse", &[valid, score, impossible])
}

/// Log of `j! / (shape)_j`, with `(shape)_j` the rising factorial.
fn log_factorial_ratio(
    b: &mut Builder<'_>,
    logbeta: NodeId,
    shape: NodeId,
    count: NodeId,
) -> NodeId {
    let zero = b.lit_real(0.0);
    let one = b.lit_real(1.0);
    let positive = b.call("gt", &[count, zero]);
    let safe_count = b.call("ifelse", &[positive, count, one]);
    let log_count = b.call("log", &[safe_count]);
    let mass = b.m.alloc(Node::Call(Call {
        head: CallHead::User(logbeta),
        args: Box::new([shape, safe_count]),
        named: Box::new([]),
        inputs: None,
    }));
    let ratio = b.call("add", &[log_count, mass]);
    b.call("ifelse", &[positive, ratio, zero])
}
