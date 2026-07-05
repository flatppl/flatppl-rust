//! The distribution registry: the ctor-name-keyed dispatch table
//! `builtin_logdensityof` and `builtin_sample` use to reach a distribution's
//! closed-form builder. Adding a distribution is a new table entry here —
//! never an [`Emitter`] or [`crate::ops`] edit.
//!
//! [`Emitter::lower_node`](crate::emitter::Emitter::lower_node)'s `Call`
//! dispatch (`emitter.rs`) recognizes the `builtin_logdensityof`/
//! `builtin_sample` heads itself and routes them to [`lower_logdensityof`]/
//! [`lower_sample`] here, rather than letting either fall through to
//! `crate::ops::lower_builtin`'s catch-all "unsupported builtin head"
//! refusal — see that module's doc comment.

use flatppl_core::{NamedKind, Node, NodeId};

use crate::emitter::Emitter;
use crate::mlir::Value;
use crate::refuse::EmitError;

/// `fn(emitter, params, variate) -> log f(variate; params)` — a
/// distribution's closed-form log-density/-mass builder (§08/§09/§12/§13).
pub type LogpdfBuilder = fn(&mut Emitter, &Params, &Value) -> Result<Value, EmitError>;

/// `fn(emitter, params) -> a drawn variate` — a distribution's sampling
/// builder (`stablehlo.rng` for straight-line dists, a hand-written
/// `stablehlo.while` for rejection-based ones).
pub type SampleBuilder = fn(&mut Emitter, &Params) -> Result<Value, EmitError>;

/// One registered distribution's builders. `sample` is `None` until that
/// distribution's `@sample` builder is added — reaching `@sample` for such a
/// distribution refuses precisely (see [`lower_sample`]), rather than
/// silently reusing `logpdf` or guessing a sampler.
pub struct DistLowering {
    pub logpdf: LogpdfBuilder,
    pub sample: Option<SampleBuilder>,
}

/// The ctor-name-keyed table: a linear scan over a short static list. The
/// full registry stays well under a hundred entries end-to-end (spec
/// §08/§09/§12/§13), so this beats the bookkeeping of a `HashMap`/`phf` for
/// no measurable runtime cost.
static REGISTRY: &[(&str, DistLowering)] = &[
    (
        "Normal",
        DistLowering {
            logpdf: normal_logpdf,
            sample: Some(normal_sample),
        },
    ),
    (
        "Cauchy",
        DistLowering {
            logpdf: cauchy_logpdf,
            sample: None,
        },
    ),
    (
        "Logistic",
        DistLowering {
            logpdf: logistic_logpdf,
            sample: None,
        },
    ),
    (
        "Laplace",
        DistLowering {
            logpdf: laplace_logpdf,
            sample: None,
        },
    ),
    (
        "Exponential",
        DistLowering {
            logpdf: exponential_logpdf,
            sample: None,
        },
    ),
    (
        "Gamma",
        DistLowering {
            logpdf: gamma_logpdf,
            sample: None,
        },
    ),
    (
        "Weibull",
        DistLowering {
            logpdf: weibull_logpdf,
            sample: None,
        },
    ),
    (
        "Pareto",
        DistLowering {
            logpdf: pareto_logpdf,
            sample: None,
        },
    ),
    (
        "InverseGamma",
        DistLowering {
            logpdf: inverse_gamma_logpdf,
            sample: None,
        },
    ),
    (
        "ChiSquared",
        DistLowering {
            logpdf: chi_squared_logpdf,
            sample: None,
        },
    ),
    (
        "LogNormal",
        DistLowering {
            logpdf: lognormal_logpdf,
            sample: None,
        },
    ),
];

/// Look up a distribution's lowering by its constructor name (`"Normal"`,
/// …). `None` for an unregistered ctor — the caller turns that into a
/// precise [`EmitError`] (refuse-don't-mislower: a not-yet-implemented
/// distribution must never silently fall through to a wrong lowering).
pub fn lookup(ctor: &str) -> Option<&'static DistLowering> {
    REGISTRY
        .iter()
        .find(|(name, _)| *name == ctor)
        .map(|(_, dist)| dist)
}

/// Resolves a `builtin_logdensityof`/`builtin_sample` kernel's kwargs — its
/// `kernel_input`, a determinizer-built `record(%field name = value, …)`
/// (spec §07) — to already-[`Emitter::lower_node`]d [`Value`]s, one named
/// field at a time.
pub struct Params {
    kernel_input: NodeId,
}

impl Params {
    /// Lower the kernel-input record field named `name` (e.g. `"mu"`,
    /// `"sigma"`) to a [`Value`]. Refuses if `name` is not a `%field` of the
    /// kernel-input record — an internal-contract violation (the
    /// determiniser always emits exactly the fields a ctor's registry entry
    /// expects), not a user-facing shape mismatch, but still reported via
    /// [`EmitError`] rather than panicking: a registry builder is reachable
    /// from an arbitrary FlatPDL `builtin_logdensityof` node, not only ones
    /// this crate's own determinizer built.
    pub fn get(&self, e: &mut Emitter, name: &str) -> Result<Value, EmitError> {
        let field = match e.node(self.kernel_input) {
            Node::Call(c) => c.named.iter().find_map(|n| {
                (n.kind == NamedKind::Field && e.resolve(n.name) == name).then_some(n.value)
            }),
            _ => None,
        };
        let field = field.ok_or_else(|| {
            EmitError::at(
                self.kernel_input,
                format!("distribution parameter '{name}' missing from kernel input"),
            )
        })?;
        e.lower_node(field)
    }
}

/// `builtin_logdensityof(kernel, kernel_input, v)` (`density.rs`'s
/// `build_density_term`): `kernel` is a bare `Const(ctor)` distribution
/// constructor symbol, `kernel_input` its kwargs record, `v` the scored
/// variate. Dispatches to `lookup(ctor).logpdf`, refusing precisely for a
/// malformed call shape or an unregistered ctor — never guessed.
pub(crate) fn lower_logdensityof(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [kernel, kernel_input, v] = <[NodeId; 3]>::try_from(args).map_err(|_| {
        EmitError::at(
            id,
            format!(
                "builtin_logdensityof: expected 3 arguments, got {}",
                args.len()
            ),
        )
    })?;

    let ctor = match e.node(kernel) {
        Node::Const(sym) => e.resolve(*sym).to_string(),
        _ => {
            return Err(EmitError::at(
                kernel,
                "builtin_logdensityof: kernel must be a bare distribution constructor",
            ));
        }
    };
    let dist = lookup(&ctor)
        .ok_or_else(|| EmitError::at(id, format!("no lowering for distribution '{ctor}'")))?;

    let params = Params { kernel_input };
    let value = e.lower_node(v)?;
    (dist.logpdf)(e, &params, &value)
}

/// `builtin_sample(rng, ctor, kernel_input)` (`flatppl_determinizer::sample`'s
/// `build_sample_term`/`lower_shared_record_sample`): `rng` is the threaded
/// RNG-state argument — deliberately UNUSED (bound as `_rng` below): this
/// vertical lowers to `stablehlo.rng`, which is XLA-seeded and takes no
/// explicit rng key, so there is nothing to lower it to (spec §07's
/// `builtin_sample` returns a `(value, new_rngstate)` pair; the advanced
/// rng-state half has no tensor form here either — see
/// `Emitter::sample_tuple_slot`'s doc comment for how a `get0(_, 1)`
/// projection of it is refused rather than mis-lowered). `ctor` is a bare
/// `Const(ctor)` distribution constructor symbol, `kernel_input` its kwargs
/// record — otherwise the same shape as [`lower_logdensityof`]'s `kernel`/
/// `kernel_input`. Dispatches to `lookup(ctor).sample`, refusing precisely
/// for a malformed call shape, an unregistered ctor, or a registered ctor
/// with no `@sample` builder yet — never guessed.
pub(crate) fn lower_sample(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [_rng, ctor, kernel_input] = <[NodeId; 3]>::try_from(args).map_err(|_| {
        EmitError::at(
            id,
            format!("builtin_sample: expected 3 arguments, got {}", args.len()),
        )
    })?;

    let ctor_name = match e.node(ctor) {
        Node::Const(sym) => e.resolve(*sym).to_string(),
        _ => {
            return Err(EmitError::at(
                ctor,
                "builtin_sample: ctor must be a bare distribution constructor",
            ));
        }
    };
    let dist = lookup(&ctor_name)
        .ok_or_else(|| EmitError::at(id, format!("no lowering for distribution '{ctor_name}'")))?;
    let sample = dist
        .sample
        .ok_or_else(|| EmitError::at(id, format!("no @sample lowering for '{ctor_name}'")))?;

    let params = Params { kernel_input };
    sample(e, &params)
}

// ---- §08 Normal -------------------------------------------------------------

/// §08 Normal, verbatim: `log f = -log(sigma) - 1/2 * log(2*pi) - (x -
/// mu)^2 / (2*sigma^2)`.
///
/// Same op sequence/count as the plan's sketch (`e.neg(&{ let l =
/// e.log(&sigma); l })`, `e.div(&e.sub(v, &mu), &sigma)`, …) — each
/// intermediate is bound to its own `let` because `Emitter`'s op helpers all
/// take `&mut self`, and Rust does not allow two nested calls to mutably
/// borrow the same `e` within one expression (the sketch's nesting is
/// illustrative of the arithmetic, not literal executable Rust).
fn normal_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mu = p.get(e, "mu")?;
    let sigma = p.get(e, "sigma")?;

    let log_sigma = e.log(&sigma);
    let neg_log_sigma = e.neg(&log_sigma);
    let c = e.scalar(-0.5 * (2.0 * std::f64::consts::PI).ln());

    let diff = e.sub(v, &mu);
    let z = e.div(&diff, &sigma);
    let half = e.scalar(-0.5);
    let z_sq = e.mul(&z, &z);
    let quad = e.mul(&half, &z_sq);

    let neg_log_sigma_plus_c = e.add(&neg_log_sigma, &c);
    Ok(e.add(&neg_log_sigma_plus_c, &quad))
}

/// §08 Normal's sampling transform, verbatim: `mu + sigma * Z`, `Z ~
/// Normal(0, 1)`. `Z` is drawn at `mu`'s own shape (`&mu.ty`) — the variate
/// shape a scalar or (later) vector-valued Normal draw needs, mirroring how
/// [`normal_logpdf`] reads its parameters via [`Params::get`]. Same
/// let-per-intermediate discipline as [`normal_logpdf`] (nested `&mut
/// Emitter` calls do not borrow-check) — the brief's `e.add(&mu,
/// &e.mul(&sigma, &z))` sketch is illustrative of the arithmetic, not
/// literal executable Rust.
fn normal_sample(e: &mut Emitter, p: &Params) -> Result<Value, EmitError> {
    let mu = p.get(e, "mu")?;
    let sigma = p.get(e, "sigma")?;

    let zero = e.scalar(0.0);
    let one = e.scalar(1.0);
    let z = e.rng("NORMAL", &zero, &one, &mu.ty);

    let sigma_z = e.mul(&sigma, &z);
    Ok(e.add(&mu, &sigma_z))
}

// ---- §08 Cauchy -------------------------------------------------------------

/// §08 Cauchy, verbatim: `log f = -log(pi) - log(gamma) - log(1 + ((x -
/// x0) / gamma)^2)`. No `@sample` builder yet (`sample: None`; Task 14).
fn cauchy_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let location = p.get(e, "location")?;
    let scale = p.get(e, "scale")?;

    let neg_log_pi = e.scalar(-std::f64::consts::PI.ln());
    let log_scale = e.log(&scale);
    let neg_log_scale = e.neg(&log_scale);

    let diff = e.sub(v, &location);
    let z = e.div(&diff, &scale);
    let z_sq = e.mul(&z, &z);
    let one = e.scalar(1.0);
    let one_plus_z_sq = e.add(&one, &z_sq);
    let log_one_plus_z_sq = e.log(&one_plus_z_sq);
    let neg_log_one_plus_z_sq = e.neg(&log_one_plus_z_sq);

    let neg_log_pi_scale = e.add(&neg_log_pi, &neg_log_scale);
    Ok(e.add(&neg_log_pi_scale, &neg_log_one_plus_z_sq))
}

// ---- §08 Logistic -----------------------------------------------------------

/// §08 Logistic, verbatim: with `u = (x - mu) / s`, `log f = -u - log(s) -
/// 2 * log(1 + exp(-u))`. No `@sample` builder yet (`sample: None`; Task
/// 14).
fn logistic_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mu = p.get(e, "mu")?;
    let s = p.get(e, "s")?;

    let diff = e.sub(v, &mu);
    let u = e.div(&diff, &s);
    let neg_u = e.neg(&u);

    let log_s = e.log(&s);
    let neg_log_s = e.neg(&log_s);

    let exp_neg_u = e.exp(&neg_u);
    let one = e.scalar(1.0);
    let one_plus_exp_neg_u = e.add(&one, &exp_neg_u);
    let log_term = e.log(&one_plus_exp_neg_u);
    let two = e.scalar(2.0);
    let two_log_term = e.mul(&two, &log_term);
    let neg_two_log_term = e.neg(&two_log_term);

    let neg_u_minus_log_s = e.add(&neg_u, &neg_log_s);
    Ok(e.add(&neg_u_minus_log_s, &neg_two_log_term))
}

// ---- §08 Laplace ------------------------------------------------------------

/// §08 Laplace, verbatim: `log f = -log(2 * b) - |x - mu| / b`. No
/// `@sample` builder yet (`sample: None`; Task 14).
fn laplace_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let location = p.get(e, "location")?;
    let scale = p.get(e, "scale")?;

    let two = e.scalar(2.0);
    let two_b = e.mul(&two, &scale);
    let log_two_b = e.log(&two_b);
    let neg_log_two_b = e.neg(&log_two_b);

    let diff = e.sub(v, &location);
    let abs_diff = e.abs(&diff);
    let term = e.div(&abs_diff, &scale);
    let neg_term = e.neg(&term);

    Ok(e.add(&neg_log_two_b, &neg_term))
}

// ---- §08 gamma-family / positive-support continuous batch -------------------
//
// Exponential/Gamma/Weibull/Pareto/InverseGamma/ChiSquared/LogNormal,
// registered alongside Normal/Cauchy/Logistic/Laplace in `REGISTRY` with
// `sample: None` (samplers land in Task 14). Gamma/InverseGamma/ChiSquared's
// log-forms need the log-gamma special function, `chlo.lgamma`
// ([`Emitter::lgamma`]); the others compose only the elementary-op helpers.

/// §08 Exponential, verbatim: `log f = log(rate) - rate * x`. No `@sample`
/// builder yet (`sample: None`; Task 14).
fn exponential_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let rate = p.get(e, "rate")?;

    let log_rate = e.log(&rate);
    let rate_x = e.mul(&rate, v);
    let neg_rate_x = e.neg(&rate_x);

    Ok(e.add(&log_rate, &neg_rate_x))
}

/// §08 Gamma, verbatim: `log f = shape * log(rate) - lgamma(shape) +
/// (shape - 1) * log(x) - rate * x`. No `@sample` builder yet (`sample:
/// None`; Task 14).
fn gamma_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let shape = p.get(e, "shape")?;
    let rate = p.get(e, "rate")?;

    let log_rate = e.log(&rate);
    let shape_log_rate = e.mul(&shape, &log_rate);

    let lgamma_shape = e.lgamma(&shape);
    let neg_lgamma_shape = e.neg(&lgamma_shape);

    let one = e.scalar(1.0);
    let shape_minus_one = e.sub(&shape, &one);
    let log_x = e.log(v);
    let shape_minus_one_log_x = e.mul(&shape_minus_one, &log_x);

    let rate_x = e.mul(&rate, v);
    let neg_rate_x = e.neg(&rate_x);

    let t1 = e.add(&shape_log_rate, &neg_lgamma_shape);
    let t2 = e.add(&t1, &shape_minus_one_log_x);
    Ok(e.add(&t2, &neg_rate_x))
}

/// §08 Weibull, verbatim: with `u = x / scale`, `log f = log(shape) -
/// log(scale) + (shape - 1) * log(u) - u^shape`. No `@sample` builder yet
/// (`sample: None`; Task 14).
fn weibull_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let shape = p.get(e, "shape")?;
    let scale = p.get(e, "scale")?;

    let log_shape = e.log(&shape);
    let log_scale = e.log(&scale);
    let neg_log_scale = e.neg(&log_scale);

    let u = e.div(v, &scale);
    let log_u = e.log(&u);
    let one = e.scalar(1.0);
    let shape_minus_one = e.sub(&shape, &one);
    let shape_minus_one_log_u = e.mul(&shape_minus_one, &log_u);

    let u_pow_shape = e.pow(&u, &shape);
    let neg_u_pow_shape = e.neg(&u_pow_shape);

    let t1 = e.add(&log_shape, &neg_log_scale);
    let t2 = e.add(&t1, &shape_minus_one_log_u);
    Ok(e.add(&t2, &neg_u_pow_shape))
}

/// §08 Pareto, verbatim: `log f = log(shape) + shape * log(scale) -
/// (shape + 1) * log(x)`. No `@sample` builder yet (`sample: None`; Task
/// 14).
fn pareto_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let shape = p.get(e, "shape")?;
    let scale = p.get(e, "scale")?;

    let log_shape = e.log(&shape);
    let log_scale = e.log(&scale);
    let shape_log_scale = e.mul(&shape, &log_scale);

    let one = e.scalar(1.0);
    let shape_plus_one = e.add(&shape, &one);
    let log_x = e.log(v);
    let shape_plus_one_log_x = e.mul(&shape_plus_one, &log_x);
    let neg_shape_plus_one_log_x = e.neg(&shape_plus_one_log_x);

    let t1 = e.add(&log_shape, &shape_log_scale);
    Ok(e.add(&t1, &neg_shape_plus_one_log_x))
}

/// §08 InverseGamma, verbatim: `log f = shape * log(scale) - lgamma(shape) -
/// (shape + 1) * log(x) - scale / x`. No `@sample` builder yet (`sample:
/// None`; Task 14).
fn inverse_gamma_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let shape = p.get(e, "shape")?;
    let scale = p.get(e, "scale")?;

    let log_scale = e.log(&scale);
    let shape_log_scale = e.mul(&shape, &log_scale);

    let lgamma_shape = e.lgamma(&shape);
    let neg_lgamma_shape = e.neg(&lgamma_shape);

    let one = e.scalar(1.0);
    let shape_plus_one = e.add(&shape, &one);
    let log_x = e.log(v);
    let shape_plus_one_log_x = e.mul(&shape_plus_one, &log_x);
    let neg_shape_plus_one_log_x = e.neg(&shape_plus_one_log_x);

    let scale_over_x = e.div(&scale, v);
    let neg_scale_over_x = e.neg(&scale_over_x);

    let t1 = e.add(&shape_log_scale, &neg_lgamma_shape);
    let t2 = e.add(&t1, &neg_shape_plus_one_log_x);
    Ok(e.add(&t2, &neg_scale_over_x))
}

/// §08 ChiSquared, verbatim: with `half_k = k / 2`, `log f = -half_k *
/// log(2) - lgamma(half_k) + (half_k - 1) * log(x) - x / 2`. `log(2)` is a
/// plain numeric constant (independent of `k`), so it is folded to a scalar
/// literal (`std::f64::consts::LN_2`) rather than emitted as its own
/// `stablehlo.log` — same reasoning as [`cauchy_logpdf`]'s `log(pi)` fold. No
/// `@sample` builder yet (`sample: None`; Task 14).
fn chi_squared_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let k = p.get(e, "k")?;

    let half = e.scalar(0.5);
    let half_k = e.mul(&half, &k);

    let ln_two = e.scalar(std::f64::consts::LN_2);
    let half_k_ln_two = e.mul(&half_k, &ln_two);
    let neg_half_k_ln_two = e.neg(&half_k_ln_two);

    let lgamma_half_k = e.lgamma(&half_k);
    let neg_lgamma_half_k = e.neg(&lgamma_half_k);

    let one = e.scalar(1.0);
    let half_k_minus_one = e.sub(&half_k, &one);
    let log_x = e.log(v);
    let half_k_minus_one_log_x = e.mul(&half_k_minus_one, &log_x);

    let two = e.scalar(2.0);
    let x_over_two = e.div(v, &two);
    let neg_x_over_two = e.neg(&x_over_two);

    let t1 = e.add(&neg_half_k_ln_two, &neg_lgamma_half_k);
    let t2 = e.add(&t1, &half_k_minus_one_log_x);
    Ok(e.add(&t2, &neg_x_over_two))
}

/// §08 LogNormal, verbatim: `log f = -log(x) - log(sigma) - 1/2 * log(2*pi) -
/// (log(x) - mu)^2 / (2*sigma^2)`. The quadratic term is composed exactly
/// like [`normal_logpdf`]'s (`z = (log(x) - mu) / sigma`, `-0.5 * z^2`), with
/// `log(x)` in place of `x` — and the same `log(x)` [`Value`] is reused for
/// both the leading `-log(x)` term and this quadratic term, rather than
/// calling [`Emitter::log`] on `v` a second time (each call emits a fresh
/// `stablehlo.log` op; unlike [`Emitter::lower_node`], these op helpers do
/// not memoize by FlatPDL `NodeId`). No `@sample` builder yet (`sample:
/// None`; Task 14).
fn lognormal_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mu = p.get(e, "mu")?;
    let sigma = p.get(e, "sigma")?;

    let log_x = e.log(v);
    let neg_log_x = e.neg(&log_x);

    let log_sigma = e.log(&sigma);
    let neg_log_sigma = e.neg(&log_sigma);

    let c = e.scalar(-0.5 * (2.0 * std::f64::consts::PI).ln());

    let diff = e.sub(&log_x, &mu);
    let z = e.div(&diff, &sigma);
    let neg_half = e.scalar(-0.5);
    let z_sq = e.mul(&z, &z);
    let quad = e.mul(&neg_half, &z_sq);

    let t1 = e.add(&neg_log_x, &neg_log_sigma);
    let t2 = e.add(&t1, &c);
    Ok(e.add(&t2, &quad))
}
