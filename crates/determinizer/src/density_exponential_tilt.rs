//! Exponential tilts that stay in an existing distribution family.
//! Recognition precedes measure elimination, so scoring and sampling agree.

use super::*;

pub(crate) struct Law {
    pub measure: NodeId,
    pub valid: NodeId,
    pub integer: bool,
    pub log_beta: Option<NodeId>,
    score: Score,
}

#[derive(Clone, Copy)]
enum Score {
    Primitive,
    Binomial {
        n: NodeId,
        p: NodeId,
        theta: NodeId,
    },
    Poisson {
        rate: NodeId,
        theta: NodeId,
        log_rate: NodeId,
    },
    Geometric {
        p: NodeId,
        theta: NodeId,
        log_failure: NodeId,
    },
    NegativeBinomial {
        source: NodeId,
        theta: NodeId,
        shape: NodeId,
        log_ratio: NodeId,
        log_beta: NodeId,
    },
}

/// Match either normalized log weighting or its equivalent `tilt` spelling.
pub(crate) fn recognize(m: &mut Module, node: NodeId) -> Option<Law> {
    let node = resolve_ref_chain(m, node).0;
    let (weight, base, reference, parameter) = match builtin_name(m, node)? {
        "normalize" => {
            let normalized = expect_builtin_call(m, node, "normalize")?;
            let [weighted] = *normalized.args else {
                return None;
            };
            let weighted = resolve_ref_chain(m, weighted).0;
            let weighted = expect_builtin_call(m, weighted, "logweighted")?;
            if !matches!(weighted.args.len(), 2 | 3) {
                return None;
            }
            (
                weighted.args[0],
                weighted.args[1],
                None,
                weighted.args.get(2).copied(),
            )
        }
        "tilt" => {
            let call = expect_builtin_call(m, node, "tilt")?;
            let [source, base] = *call.args else {
                return None;
            };
            let source = resolve_ref_chain(m, source).0;
            let source = expect_builtin_call(m, source, "logweighted")?;
            if !matches!(source.args.len(), 2 | 3) {
                return None;
            }
            (
                source.args[0],
                base,
                Some(source.args[1]),
                source.args.get(2).copied(),
            )
        }
        _ => return None,
    };
    let base = resolve_ref_chain(m, base).0;
    let measure_type = m.type_of(base).cloned();
    let inherited = recognize(m, base);
    let base = inherited.as_ref().map_or(base, |law| law.measure);
    let (ctor, parameters) = split_kernel_constructor(m, base)?;
    let name = m.resolve(ctor).to_owned();
    if !matches!(
        name.as_str(),
        "Normal"
            | "Gamma"
            | "Exponential"
            | "ChiSquared"
            | "Bernoulli"
            | "Binomial"
            | "Poisson"
            | "Geometric"
            | "NegativeBinomial"
            | "NegativeBinomial2"
            | "MvNormal"
    ) {
        return None;
    }
    let vector = name == "MvNormal";
    let integer = matches!(
        name.as_str(),
        "Bernoulli"
            | "Binomial"
            | "Poisson"
            | "Geometric"
            | "NegativeBinomial"
            | "NegativeBinomial2"
    );
    if let Some(reference) = reference
        && !unit_reference(m, reference, base, &name, vector)
    {
        return None;
    }
    let (body, input) = weight_function_body(m, weight, parameter)?;
    let affine = affine(m, body, input, vector)?;
    let mut valid = build_call(m, "isfinite", &[affine.offset]);
    if let Some(law) = &inherited {
        valid = build_call(m, "land", &[valid, law.valid]);
    }
    let Some(theta) = affine.slope else {
        return Some(Law {
            measure: base,
            valid,
            integer,
            log_beta: inherited.as_ref().and_then(|law| law.log_beta),
            score: inherited.map_or(Score::Primitive, |law| law.score),
        });
    };
    let theta = if vector {
        let n = m.alloc(Node::Lit(Scalar::Int(iid_static_size(m, base)? as i64)));
        let zero = real(m, 0.0);
        let zeros = build_call(m, "fill", &[zero, n]);
        tilt::elementwise(m, "add", &[theta, zeros])
    } else {
        theta
    };
    let finite = if vector {
        let entries = tilt::elementwise(m, "isfinite", &[theta]);
        build_call(m, "lall", &[entries])
    } else {
        build_call(m, "isfinite", &[theta])
    };
    valid = build_call(m, "land", &[valid, finite]);
    let param = |m: &Module, name| find_kwarg(m, &parameters, name);
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let mut score = Score::Primitive;
    let mut log_beta = None;
    let (family, args) = match name.as_str() {
        "Normal" => {
            let mu = param(m, "mu")?;
            let sigma = param(m, "sigma")?;
            require_finite(m, &mut valid, mu);
            require_positive(m, &mut valid, sigma);
            let shift = build_call(m, "mul", &[theta, sigma]);
            let shift = build_call(m, "mul", &[shift, sigma]);
            let mean = build_call(m, "add", &[mu, shift]);
            require_finite(m, &mut valid, mean);
            ("Normal", vec![mean, sigma])
        }
        "Gamma" | "Exponential" | "ChiSquared" => {
            let rate = if name == "ChiSquared" {
                real(m, 0.5)
            } else {
                param(m, "rate")?
            };
            require_positive(m, &mut valid, rate);
            let tilted_rate = build_call(m, "sub", &[rate, theta]);
            require_positive(m, &mut valid, tilted_rate);
            let safe_rate = build_call(m, "ifelse", &[valid, tilted_rate, one]);
            if name == "Exponential" {
                ("Exponential", vec![safe_rate])
            } else {
                let shape = if name == "ChiSquared" {
                    let k = param(m, "k")?;
                    let half = real(m, 0.5);
                    build_call(m, "mul", &[k, half])
                } else {
                    param(m, "shape")?
                };
                require_positive(m, &mut valid, shape);
                let safe_shape = build_call(m, "ifelse", &[valid, shape, one]);
                ("Gamma", vec![safe_shape, safe_rate])
            }
        }
        "Bernoulli" | "Binomial" => {
            let (p, theta) = match inherited.as_ref().map(|law| law.score) {
                Some(Score::Binomial {
                    p, theta: previous, ..
                }) => (p, build_call(m, "add", &[previous, theta])),
                _ => (param(m, "p")?, theta),
            };
            require_probability(m, &mut valid, p);
            // Never differentiate log(0) in an inactive endpoint branch.
            let at_zero = build_call(m, "iszero", &[p]);
            let at_one = equal(m, p, one);
            let endpoint = build_call(m, "lor", &[at_zero, at_one]);
            let half = real(m, 0.5);
            let interior = build_call(m, "ifelse", &[endpoint, half, p]);
            let odds = build_call(m, "logit", &[interior]);
            let odds = build_call(m, "add", &[odds, theta]);
            let probability = build_call(m, "invlogit", &[odds]);
            let probability = build_call(m, "ifelse", &[endpoint, p, probability]);
            let probability = build_call(m, "ifelse", &[valid, probability, zero]);
            if name == "Bernoulli" {
                score = Score::Binomial { n: one, p, theta };
                ("Bernoulli", vec![probability])
            } else {
                let n = param(m, "n")?;
                let nonnegative = build_call(m, "ge", &[n, zero]);
                valid = build_call(m, "land", &[valid, nonnegative]);
                score = Score::Binomial { n, p, theta };
                ("Binomial", vec![n, probability])
            }
        }
        "Poisson" => {
            let (rate, theta) = match inherited.as_ref().map(|law| law.score) {
                Some(Score::Poisson {
                    rate,
                    theta: previous,
                    ..
                }) => (rate, build_call(m, "add", &[previous, theta])),
                _ => (param(m, "rate")?, theta),
            };
            require_finite(m, &mut valid, rate);
            let nonnegative = build_call(m, "ge", &[rate, zero]);
            valid = build_call(m, "land", &[valid, nonnegative]);
            let at_zero = build_call(m, "iszero", &[rate]);
            let safe = build_call(m, "ifelse", &[at_zero, one, rate]);
            let log_rate = build_call(m, "log", &[safe]);
            let log_rate = build_call(m, "add", &[log_rate, theta]);
            score = Score::Poisson {
                rate,
                theta,
                log_rate,
            };
            let rate = build_call(m, "exp", &[log_rate]);
            let rate = build_call(m, "ifelse", &[at_zero, zero, rate]);
            require_finite(m, &mut valid, rate);
            let rate = build_call(m, "ifelse", &[valid, rate, zero]);
            ("Poisson", vec![rate])
        }
        "Geometric" => {
            let (p, theta) = match inherited.as_ref().map(|law| law.score) {
                Some(Score::Geometric {
                    p, theta: previous, ..
                }) => (p, build_call(m, "add", &[previous, theta])),
                _ => (param(m, "p")?, theta),
            };
            require_probability(m, &mut valid, p);
            require_positive(m, &mut valid, p);
            let at_one = equal(m, p, one);
            let half = real(m, 0.5);
            let interior = build_call(m, "ifelse", &[at_one, half, p]);
            let negative = build_call(m, "neg", &[interior]);
            let log_failure = build_call(m, "log1p", &[negative]);
            let log_failure = build_call(m, "add", &[log_failure, theta]);
            score = Score::Geometric {
                p,
                theta,
                log_failure,
            };
            let mass_finite = build_call(m, "lt", &[log_failure, zero]);
            let mass_finite = build_call(m, "lor", &[at_one, mass_finite]);
            valid = build_call(m, "land", &[valid, mass_finite]);
            let failure_minus_one = build_call(m, "expm1", &[log_failure]);
            let p = build_call(m, "neg", &[failure_minus_one]);
            let p = build_call(m, "ifelse", &[at_one, one, p]);
            let p = build_call(m, "ifelse", &[valid, p, one]);
            ("Geometric", vec![p])
        }
        "NegativeBinomial" | "NegativeBinomial2" => {
            let (source, theta) = match inherited.as_ref().map(|law| law.score) {
                Some(Score::NegativeBinomial {
                    source,
                    theta: previous,
                    ..
                }) => (source, build_call(m, "add", &[previous, theta])),
                _ => (base, theta),
            };
            let (ctor, parameters) = split_kernel_constructor(m, source)?;
            let (shape, original_ratio, mean) = if m.resolve(ctor) == "NegativeBinomial" {
                let beta = find_kwarg(m, &parameters, "beta")?;
                require_positive(m, &mut valid, beta);
                (
                    find_kwarg(m, &parameters, "alpha")?,
                    build_call(m, "log1p", &[beta]),
                    None,
                )
            } else {
                let mu = find_kwarg(m, &parameters, "mu")?;
                let psi = find_kwarg(m, &parameters, "psi")?;
                require_positive(m, &mut valid, mu);
                let log_mu = build_call(m, "log", &[mu]);
                let log_psi = build_call(m, "log", &[psi]);
                let ratio = build_call(m, "sub", &[log_psi, log_mu]);
                (psi, softplus(m, ratio), Some((mu, log_mu)))
            };
            require_positive(m, &mut valid, shape);
            let log_ratio = build_call(m, "sub", &[original_ratio, theta]);
            let positive_ratio = build_call(m, "gt", &[log_ratio, zero]);
            let safe_ratio = build_call(m, "ifelse", &[positive_ratio, log_ratio, one]);
            let negative = build_call(m, "neg", &[safe_ratio]);
            let log_success = log1mexp(m, negative);
            let mut tilted_log_beta = build_call(m, "add", &[safe_ratio, log_success]);
            let mass_finite = if let Some((mu, log_mu)) = mean {
                // psi/mu may underflow even at the identity. The numerator
                // retains the law and its live derivative without that ratio.
                let tiny = build_call(m, "iszero", &[original_ratio]);
                let safe_theta = build_call(m, "ifelse", &[tiny, theta, zero]);
                let shifted = build_call(m, "expm1", &[safe_theta]);
                let delta = build_call(m, "neg", &[shifted]);
                let negative = build_call(m, "gt", &[delta, zero]);
                let numerator_theta = build_call(m, "ifelse", &[negative, zero, safe_theta]);
                let shifted = build_call(m, "expm1", &[numerator_theta]);
                let shifted = build_call(m, "mul", &[mu, shifted]);
                let numerator = build_call(m, "sub", &[shape, shifted]);
                let positive = build_call(m, "gt", &[numerator, zero]);
                let finite = build_call(m, "isfinite", &[numerator]);
                let proper = build_call(m, "land", &[positive, finite]);
                let safe = build_call(m, "ifelse", &[proper, numerator, one]);
                let log = build_call(m, "log", &[safe]);
                let log = build_call(m, "sub", &[log, log_mu]);
                let log = build_call(m, "sub", &[log, numerator_theta]);
                // Away from zero, scaling the reverse seed by the large
                // numerator first can underflow. Add the positive terms in
                // log space, retaining beta even when its ratio flushes.
                let safe_delta = build_call(m, "ifelse", &[negative, delta, one]);
                let log_delta = build_call(m, "log", &[safe_delta]);
                let log_shape = build_call(m, "log", &[shape]);
                let log_beta = build_call(m, "sub", &[log_shape, log_mu]);
                let difference = build_call(m, "sub", &[log_beta, log_delta]);
                let correction = softplus(m, difference);
                let negative_log = build_call(m, "add", &[log_delta, correction]);
                let negative_log = build_call(m, "sub", &[negative_log, safe_theta]);
                let log = build_call(m, "ifelse", &[negative, negative_log, log]);
                tilted_log_beta = build_call(m, "ifelse", &[tiny, log, tilted_log_beta]);
                build_call(m, "ifelse", &[tiny, proper, positive_ratio])
            } else {
                positive_ratio
            };
            valid = build_call(m, "land", &[valid, mass_finite]);
            score = Score::NegativeBinomial {
                source,
                theta,
                shape,
                log_ratio,
                log_beta: tilted_log_beta,
            };
            log_beta = Some(tilted_log_beta);
            let beta = build_call(m, "expm1", &[log_ratio]);
            let shape = build_call(m, "ifelse", &[valid, shape, one]);
            let beta = build_call(m, "ifelse", &[valid, beta, one]);
            ("NegativeBinomial", vec![shape, beta])
        }
        "MvNormal" => {
            let mu = param(m, "mu")?;
            let cov = param(m, "cov")?;
            let shift = build_call(m, "mul", &[cov, theta]);
            let mu = build_call(m, "add", &[mu, shift]);
            ("MvNormal", vec![mu, cov])
        }
        _ => return None,
    };
    let measure = build_call(m, family, &args);
    if let Some(ty) = measure_type {
        m.set_type(measure, ty);
    }
    Some(Law {
        measure,
        valid,
        integer,
        log_beta,
        score,
    })
}

pub(crate) fn density(m: &mut Module, law: Law, point: NodeId) -> Result<NodeId, RefuseError> {
    let score = match law.score {
        Score::Primitive => lower_measure_density(m, law.measure, point)?,
        Score::Binomial { n, p, theta } => binomial_density(m, n, p, theta, point),
        Score::Poisson {
            rate,
            theta,
            log_rate,
        } => {
            let at_zero = build_call(m, "iszero", &[rate]);
            let zero = real(m, 0.0);
            let log_rate = build_call(m, "ifelse", &[at_zero, zero, log_rate]);
            let weighted = count_log(m, point, log_rate);
            let tilted_rate = build_call(m, "exp", &[log_rate]);
            let factorial = log_factorial(m, point);
            let score = build_call(m, "sub", &[weighted, tilted_rate]);
            let score = build_call(m, "sub", &[score, factorial]);
            let weight = endpoint_weight(m, theta, at_zero);
            let limit = build_call(m, "mul", &[rate, weight]);
            let limit = build_call(m, "neg", &[limit]);
            let supported = build_call(m, "iszero", &[point]);
            let limit = gate_density(m, supported, limit);
            build_call(m, "ifelse", &[at_zero, limit, score])
        }
        Score::Geometric {
            p,
            theta,
            log_failure,
        } => {
            let one = real(m, 1.0);
            let at_one = equal(m, p, one);
            let safe = real(m, -std::f64::consts::LN_2);
            let log_failure = build_call(m, "ifelse", &[at_one, safe, log_failure]);
            let success = log1mexp(m, log_failure);
            let failures = count_log(m, point, log_failure);
            let score = build_call(m, "add", &[success, failures]);
            let weight = endpoint_weight(m, theta, at_one);
            let delta = build_call(m, "sub", &[p, one]);
            let limit = build_call(m, "mul", &[delta, weight]);
            let supported = build_call(m, "iszero", &[point]);
            let limit = gate_density(m, supported, limit);
            build_call(m, "ifelse", &[at_one, limit, score])
        }
        Score::NegativeBinomial {
            shape,
            log_ratio,
            log_beta,
            ..
        } => {
            let log_failure = build_call(m, "neg", &[log_ratio]);
            let success = negative_binomial_success(m, shape, log_ratio, log_beta);
            let failures = count_log(m, point, log_failure);
            let choose = negative_binomial_choose(m, shape, point);
            let score = build_call(m, "add", &[choose, success]);
            build_call(m, "add", &[score, failures])
        }
    };
    let score = if law.integer {
        let zero = real(m, 0.0);
        let supported = build_call(m, "ge", &[point, zero]);
        gate_density(m, supported, score)
    } else {
        score
    };
    Ok(guard(m, score, law.valid))
}

fn binomial_density(m: &mut Module, n: NodeId, p: NodeId, theta: NodeId, point: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let half = real(m, 0.5);
    let at_zero = build_call(m, "iszero", &[p]);
    let at_one = equal(m, p, one);
    let endpoint = build_call(m, "lor", &[at_zero, at_one]);
    let interior = build_call(m, "ifelse", &[endpoint, half, p]);
    let odds = build_call(m, "logit", &[interior]);
    let odds = build_call(m, "add", &[odds, theta]);
    let negative = build_call(m, "neg", &[odds]);
    let success = softplus(m, negative);
    let failure = softplus(m, odds);
    let failures = build_call(m, "sub", &[n, point]);
    let successes = count_log(m, point, success);
    let failures_term = count_log(m, failures, failure);
    let penalty = build_call(m, "add", &[successes, failures_term]);
    let nf = log_factorial(m, n);
    let xf = log_factorial(m, point);
    let ff = log_factorial(m, failures);
    let choose = build_call(m, "sub", &[nf, xf]);
    let choose = build_call(m, "sub", &[choose, ff]);
    let score = build_call(m, "sub", &[choose, penalty]);

    // The supported point mass has score zero, but its one-sided derivative
    // with respect to the original p is -n exp(theta) or n exp(-theta).
    let weight_zero = endpoint_weight(m, theta, at_zero);
    let p_zero = build_call(m, "ifelse", &[at_zero, p, zero]);
    let p_zero = build_call(m, "mul", &[p_zero, weight_zero]);
    let p_zero = build_call(m, "mul", &[n, p_zero]);
    let score_zero = build_call(m, "neg", &[p_zero]);
    let no_success = build_call(m, "iszero", &[point]);
    let score_zero = gate_density(m, no_success, score_zero);
    let negative = build_call(m, "neg", &[theta]);
    let weight_one = endpoint_weight(m, negative, at_one);
    let p_one = build_call(m, "sub", &[p, one]);
    let p_one = build_call(m, "ifelse", &[at_one, p_one, zero]);
    let p_one = build_call(m, "mul", &[p_one, weight_one]);
    let score_one = build_call(m, "mul", &[n, p_one]);
    let no_failure = equal(m, point, n);
    let score_one = gate_density(m, no_failure, score_one);
    let score = build_call(m, "ifelse", &[at_zero, score_zero, score]);
    let score = build_call(m, "ifelse", &[at_one, score_one, score]);
    let upper = build_call(m, "le", &[point, n]);
    gate_density(m, upper, score)
}

// Preserve finite one-sided derivatives while keeping the point-mass value
// defined when that derivative overflows. Mask before exp for inactive AD.
fn endpoint_weight(m: &mut Module, theta: NodeId, active: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let theta = build_call(m, "ifelse", &[active, theta, zero]);
    let weight = build_call(m, "exp", &[theta]);
    let finite = build_call(m, "isfinite", &[weight]);
    let safe = build_call(m, "ifelse", &[finite, theta, zero]);
    let weight = build_call(m, "exp", &[safe]);
    build_call(m, "ifelse", &[finite, weight, zero])
}

fn log_factorial(m: &mut Module, n: NodeId) -> NodeId {
    let one = real(m, 1.0);
    let next = build_call(m, "add", &[n, one]);
    build_call(m, "loggamma", &[next])
}

fn negative_binomial_choose(m: &mut Module, shape: NodeId, point: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let empty = build_call(m, "iszero", &[point]);
    let single = equal(m, point, one);
    let small = build_call(m, "lor", &[empty, single]);
    let b = build_call(m, "max", &[shape, point]);
    let s = build_call(m, "min", &[shape, point]);
    let cutoff = real(m, 16.0);
    let large = build_call(m, "ge", &[b, cutoff]);
    let special = build_call(m, "lor", &[small, large]);
    let safe_shape = build_call(m, "ifelse", &[special, one, shape]);
    let safe_point = build_call(m, "ifelse", &[special, one, point]);
    let total = build_call(m, "add", &[safe_shape, safe_point]);
    let total = build_call(m, "loggamma", &[total]);
    let gamma = build_call(m, "loggamma", &[safe_shape]);
    let factorial = log_factorial(m, safe_point);
    let choose = build_call(m, "sub", &[total, gamma]);
    let choose = build_call(m, "sub", &[choose, factorial]);
    // logΓ(b+s)-logΓ(b)-logΓ(s)-log(k), with b>=s. This
    // avoids cancellation against either the shape or the count factorial.
    let safe_b = build_call(m, "max", &[b, cutoff]);
    let safe_s = build_call(m, "ifelse", &[small, one, s]);
    let increment = log_gamma_increment(m, safe_b, safe_s);
    let gamma_s = build_call(m, "loggamma", &[safe_s]);
    let large_choose = build_call(m, "sub", &[increment, gamma_s]);
    let safe_k = build_call(m, "ifelse", &[small, one, point]);
    let log_k = build_call(m, "log", &[safe_k]);
    let large_choose = build_call(m, "sub", &[large_choose, log_k]);
    let choose = build_call(m, "ifelse", &[large, large_choose, choose]);
    let log_shape = build_call(m, "log", &[shape]);
    let choose = build_call(m, "ifelse", &[single, log_shape, choose]);
    build_call(m, "ifelse", &[empty, zero, choose])
}

// DLMF5.11.1, differenced before evaluation. Five correction terms leave
// an absolute remainder below1.1e-16 per logΓ for a>=16.
fn log_gamma_increment(m: &mut Module, a: NodeId, k: NodeId) -> NodeId {
    let log_a = build_call(m, "log", &[a]);
    let leading = build_call(m, "mul", &[k, log_a]);
    let ratio = build_call(m, "divide", &[k, a]);
    let log_ratio = build_call(m, "log1p", &[ratio]);
    let total = build_call(m, "add", &[a, k]);
    let half = real(m, 0.5);
    let factor = build_call(m, "sub", &[total, half]);
    let term = build_call(m, "mul", &[factor, log_ratio]);
    let term = build_call(m, "sub", &[term, k]);
    // XLA may flush k/a to zero. Its correction tends to zero, not -k.
    let vanished = build_call(m, "iszero", &[ratio]);
    let zero = real(m, 0.0);
    let term = build_call(m, "ifelse", &[vanished, zero, term]);
    let result = build_call(m, "add", &[leading, term]);
    let before = stirling_correction(m, a);
    let after = stirling_correction(m, total);
    let correction = build_call(m, "sub", &[after, before]);
    build_call(m, "add", &[result, correction])
}

fn stirling_correction(m: &mut Module, x: NodeId) -> NodeId {
    let one = real(m, 1.0);
    let inverse = build_call(m, "divide", &[one, x]);
    let square = build_call(m, "mul", &[inverse, inverse]);
    let mut polynomial = real(m, 1.0 / 1188.0);
    for coefficient in [-1.0 / 1680.0, 1.0 / 1260.0, -1.0 / 360.0, 1.0 / 12.0] {
        polynomial = build_call(m, "mul", &[polynomial, square]);
        let coefficient = real(m, coefficient);
        polynomial = build_call(m, "add", &[polynomial, coefficient]);
    }
    build_call(m, "mul", &[inverse, polynomial])
}

fn negative_binomial_success(
    m: &mut Module,
    shape: NodeId,
    ratio: NodeId,
    log_beta: NodeId,
) -> NodeId {
    let cutoff = real(m, std::f64::consts::LN_2);
    let small = build_call(m, "lt", &[ratio, cutoff]);
    let safe_ratio = build_call(m, "ifelse", &[small, cutoff, ratio]);
    let negative = build_call(m, "neg", &[safe_ratio]);
    let log_success = log1mexp(m, negative);
    let difference = build_call(m, "sub", &[log_beta, ratio]);
    let log_success = build_call(m, "ifelse", &[small, difference, log_success]);
    let ordinary = build_call(m, "mul", &[shape, log_success]);
    // At R>40, -log(1-exp(-R))/exp(-R) differs from1 by <3e-18.
    // Scale in log space before exp so a representable aggregate survives.
    let limit = real(m, 40.0);
    let large = build_call(m, "gt", &[ratio, limit]);
    let log_shape = build_call(m, "log", &[shape]);
    let exponent = build_call(m, "sub", &[log_shape, ratio]);
    let scaled = build_call(m, "exp", &[exponent]);
    let scaled = build_call(m, "neg", &[scaled]);
    build_call(m, "ifelse", &[large, scaled, ordinary])
}

fn log1mexp(m: &mut Module, x: NodeId) -> NodeId {
    let cutoff = real(m, -std::f64::consts::LN_2);
    let near_zero = build_call(m, "gt", &[x, cutoff]);
    let near_x = build_call(m, "ifelse", &[near_zero, x, cutoff]);
    let expm1 = build_call(m, "expm1", &[near_x]);
    let negative = build_call(m, "neg", &[expm1]);
    let near = build_call(m, "log", &[negative]);
    let far_x = build_call(m, "ifelse", &[near_zero, cutoff, x]);
    let exp = build_call(m, "exp", &[far_x]);
    let negative = build_call(m, "neg", &[exp]);
    let far = build_call(m, "log1p", &[negative]);
    build_call(m, "ifelse", &[near_zero, near, far])
}

fn equal(m: &mut Module, left: NodeId, right: NodeId) -> NodeId {
    let difference = build_call(m, "sub", &[left, right]);
    build_call(m, "iszero", &[difference])
}

fn softplus(m: &mut Module, x: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let positive = build_call(m, "max", &[x, zero]);
    let abs = build_call(m, "abs", &[x]);
    let negative = build_call(m, "neg", &[abs]);
    let exp = build_call(m, "exp", &[negative]);
    let log = build_call(m, "log1p", &[exp]);
    build_call(m, "add", &[positive, log])
}

fn count_log(m: &mut Module, count: NodeId, log: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let empty = build_call(m, "iszero", &[count]);
    let log = build_call(m, "ifelse", &[empty, zero, log]);
    build_call(m, "mul", &[count, log])
}

/// An undefined normalizer gives an undefined score/value, not a different law.
pub(crate) fn guard(m: &mut Module, value: NodeId, valid: NodeId) -> NodeId {
    let nan = real(m, f64::NAN);
    build_call(m, "ifelse", &[valid, value, nan])
}

pub(crate) fn guard_sample(m: &mut Module, value: NodeId, valid: NodeId, integer: bool) -> NodeId {
    if integer {
        // These count laws have nonnegative support. Return an out-of-support
        // sentinel for an undefined law, regardless of the target real dtype.
        let invalid = m.alloc(Node::Lit(Scalar::Int(-1)));
        build_call(m, "ifelse", &[valid, value, invalid])
    } else {
        guard(m, value, valid)
    }
}

fn real(m: &mut Module, value: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(value)))
}

fn require_finite(m: &mut Module, valid: &mut NodeId, value: NodeId) {
    let finite = build_call(m, "isfinite", &[value]);
    *valid = build_call(m, "land", &[*valid, finite]);
}

fn require_positive(m: &mut Module, valid: &mut NodeId, value: NodeId) {
    require_finite(m, valid, value);
    let zero = real(m, 0.0);
    let positive = build_call(m, "gt", &[value, zero]);
    *valid = build_call(m, "land", &[*valid, positive]);
}

fn require_probability(m: &mut Module, valid: &mut NodeId, value: NodeId) {
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let lower = build_call(m, "ge", &[value, zero]);
    let upper = build_call(m, "le", &[value, one]);
    let inside = build_call(m, "land", &[lower, upper]);
    *valid = build_call(m, "land", &[*valid, inside]);
}

/// Only discard a reference whose density is one on the whole base support.
fn unit_reference(m: &Module, reference: NodeId, base: NodeId, family: &str, vector: bool) -> bool {
    let reference = resolve_ref_chain(m, reference).0;
    let discrete = matches!(
        family,
        "Bernoulli"
            | "Binomial"
            | "Poisson"
            | "Geometric"
            | "NegativeBinomial"
            | "NegativeBinomial2"
    );
    let Some(call) =
        expect_builtin_call(m, reference, if discrete { "Counting" } else { "Lebesgue" })
    else {
        return false;
    };
    let Some(support) = named_or_positional(m, call, "support") else {
        return false;
    };
    let support = resolve_ref_chain(m, support).0;
    if vector {
        let Some(power) = expect_builtin_call(m, support, "cartpow") else {
            return false;
        };
        let [set, n] = *power.args else { return false };
        let set = resolve_ref_chain(m, set).0;
        let expected = if discrete { "nonnegintegers" } else { "reals" };
        return matches!(m.node(set), Node::Const(sym) if m.resolve(*sym) == expected)
            && static_real(m, n).map(|n| n as usize) == iid_static_size(m, base);
    }
    let Node::Const(set) = m.node(support) else {
        return false;
    };
    match m.resolve(*set) {
        "reals" => !discrete,
        "nonnegreals" => matches!(family, "Gamma" | "Exponential" | "ChiSquared"),
        "integers" | "nonnegintegers" => discrete,
        "booleans" => family == "Bernoulli",
        _ => false,
    }
}

struct Affine {
    slope: Option<NodeId>,
    offset: NodeId,
}

/// Follow aliases but stop at the reification boundary. A nonlinear or opaque
/// dependency declines the rewrite rather than freezing it as a coefficient.
fn depends(m: &Module, node: NodeId, input: Ref, path: &mut Vec<NodeId>) -> bool {
    if matches!(m.node(node), Node::Ref(r) if *r == input) {
        return true;
    }
    if path.contains(&node) {
        return true;
    }
    path.push(node);
    let resolved = resolve_ref_one(m, node).0;
    let found = if resolved != node {
        depends(m, resolved, input, path)
    } else {
        m.node(node)
            .children()
            .into_iter()
            .any(|child| depends(m, child, input, path))
    };
    path.pop();
    found
}

fn affine(m: &mut Module, node: NodeId, input: Ref, vector: bool) -> Option<Affine> {
    if !depends(m, node, input, &mut Vec::new()) {
        return Some(Affine {
            slope: None,
            offset: node,
        });
    }
    if matches!(m.node(node), Node::Ref(r) if *r == input) {
        return (!vector).then(|| Affine {
            slope: Some(real(m, 1.0)),
            offset: real(m, 0.0),
        });
    }
    let resolved = resolve_ref_one(m, node).0;
    if resolved != node {
        return affine(m, resolved, input, vector);
    }
    let Node::Call(call) = m.node(node) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    let head = m.resolve(head).to_owned();
    let args = call.args.to_vec();
    if vector && let Some(slope) = vector_coefficient(m, node, input) {
        return Some(Affine {
            slope: Some(slope),
            offset: real(m, 0.0),
        });
    }
    match (head.as_str(), args.as_slice()) {
        ("neg", [x]) => {
            let a = affine(m, *x, input, vector)?;
            Some(Affine {
                slope: a.slope.map(|s| coefficient_op(m, "neg", &[s], vector)),
                offset: build_call(m, "neg", &[a.offset]),
            })
        }
        (op @ ("add" | "sub"), [x, y]) => {
            let a = affine(m, *x, input, vector)?;
            let b = affine(m, *y, input, vector)?;
            let slope = match (a.slope, b.slope) {
                (Some(a), Some(b)) => Some(coefficient_op(m, op, &[a, b], vector)),
                (Some(a), None) => Some(a),
                (None, Some(b)) if op == "sub" => Some(coefficient_op(m, "neg", &[b], vector)),
                (None, b) => b,
            };
            Some(Affine {
                slope,
                offset: build_call(m, op, &[a.offset, b.offset]),
            })
        }
        (op @ ("mul" | "divide"), [x, y]) => {
            let a = affine(m, *x, input, vector)?;
            let b = affine(m, *y, input, vector)?;
            let slope = match (a.slope, b.slope) {
                (Some(a), None) => Some(coefficient_op(m, op, &[a, b.offset], vector)),
                (None, Some(b)) if op == "mul" => {
                    Some(coefficient_op(m, op, &[b, a.offset], vector))
                }
                _ => return None,
            };
            Some(Affine {
                slope,
                offset: build_call(m, op, &[a.offset, b.offset]),
            })
        }
        _ => None,
    }
}

fn coefficient_op(m: &mut Module, op: &str, args: &[NodeId], vector: bool) -> NodeId {
    if vector {
        tilt::elementwise(m, op, args)
    } else {
        build_call(m, op, args)
    }
}

fn vector_coefficient(m: &Module, node: NodeId, input: Ref) -> Option<NodeId> {
    let is_input = |mut node| {
        for _ in 0..=m.binding_count() {
            if matches!(m.node(node), Node::Ref(r) if *r == input) {
                return true;
            }
            let (next, binding) = resolve_ref_one(m, node);
            if binding.is_none() {
                return false;
            }
            node = next;
        }
        false
    };
    if let Some(product) = expect_builtin_call(m, node, "mul")
        && let [left, right] = *product.args
        && let Some(transpose) = expect_builtin_call(m, resolve_ref_chain(m, left).0, "transpose")
        && let [value] = *transpose.args
    {
        if is_input(right) && !depends(m, value, input, &mut Vec::new()) {
            return Some(value);
        }
        if is_input(value) && !depends(m, right, input, &mut Vec::new()) {
            return Some(right);
        }
    }
    let sum = expect_builtin_call(m, node, "sum")?;
    let [product] = *sum.args else { return None };
    let product = expect_builtin_call(m, resolve_ref_chain(m, product).0, "broadcast")?;
    let [op, left, right] = *product.args else {
        return None;
    };
    if !matches!(m.node(op), Node::Const(sym) if m.resolve(*sym) == "mul") {
        return None;
    }
    if is_input(left) && !depends(m, right, input, &mut Vec::new()) {
        return Some(right);
    }
    if is_input(right) && !depends(m, left, input, &mut Vec::new()) {
        return Some(left);
    }
    None
}
