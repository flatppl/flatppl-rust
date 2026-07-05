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

use flatppl_core::{NamedKind, Node, NodeId, Scalar, ValueSet};

use crate::emitter::Emitter;
use crate::mlir::{MlirTy, Value};
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
    (
        "Uniform",
        DistLowering {
            logpdf: uniform_logpdf,
            sample: None,
        },
    ),
    (
        "Beta",
        DistLowering {
            logpdf: beta_logpdf,
            sample: None,
        },
    ),
    (
        "StudentT",
        DistLowering {
            logpdf: studentt_logpdf,
            sample: None,
        },
    ),
    (
        "GeneralizedNormal",
        DistLowering {
            logpdf: generalized_normal_logpdf,
            sample: None,
        },
    ),
    (
        "VonMises",
        DistLowering {
            logpdf: von_mises_logpdf,
            sample: None,
        },
    ),
    (
        "Bernoulli",
        DistLowering {
            logpdf: bernoulli_logpdf,
            sample: None,
        },
    ),
    (
        "Poisson",
        DistLowering {
            logpdf: poisson_logpdf,
            sample: None,
        },
    ),
    (
        "Binomial",
        DistLowering {
            logpdf: binomial_logpdf,
            sample: None,
        },
    ),
    (
        "Geometric",
        DistLowering {
            logpdf: geometric_logpdf,
            sample: None,
        },
    ),
    (
        "NegativeBinomial",
        DistLowering {
            logpdf: negative_binomial_logpdf,
            sample: None,
        },
    ),
    (
        "NegativeBinomial2",
        DistLowering {
            logpdf: negative_binomial2_logpdf,
            sample: None,
        },
    ),
    (
        "Categorical",
        DistLowering {
            logpdf: categorical_logpdf,
            sample: None,
        },
    ),
    (
        "Categorical0",
        DistLowering {
            logpdf: categorical0_logpdf,
            sample: None,
        },
    ),
    (
        "MvNormal",
        DistLowering {
            logpdf: mvnormal_logpdf,
            sample: None,
        },
    ),
    (
        "Dirichlet",
        DistLowering {
            logpdf: dirichlet_logpdf,
            sample: None,
        },
    ),
    (
        "Multinomial",
        DistLowering {
            logpdf: multinomial_logpdf,
            sample: None,
        },
    ),
    (
        "Wishart",
        DistLowering {
            logpdf: wishart_logpdf,
            sample: None,
        },
    ),
    (
        "InverseWishart",
        DistLowering {
            logpdf: inverse_wishart_logpdf,
            sample: None,
        },
    ),
    (
        "LKJ",
        DistLowering {
            logpdf: lkj_logpdf,
            sample: None,
        },
    ),
    (
        "LKJCholesky",
        DistLowering {
            logpdf: lkj_cholesky_logpdf,
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
    /// The raw (pre-[`Emitter::lower_node`]) [`NodeId`] of the scored variate
    /// `v`, when there is one. [`lower_logdensityof`] already lowers `v` to
    /// the `&Value` every `LogpdfBuilder` receives directly (so the ordinary
    /// arithmetic builders above never need this field), but
    /// [`categorical_logpdf`]/[`categorical0_logpdf`] need the pre-lowered
    /// NodeId too: their `get`/`get0` selector into `p` must be a literal
    /// integer, and a lowered `Value` (an opaque SSA name) carries no such
    /// structural information — see [`Params::variate_id`]. `None` for a
    /// [`lower_sample`]-built `Params` (`@sample` scores no variate).
    variate: Option<NodeId>,
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
        let field = self.field_id(e, name)?;
        e.lower_node(field)
    }

    /// The raw (pre-[`Emitter::lower_node`]) [`NodeId`] of the kernel-input
    /// record field named `name` — the structural half of [`Params::get`]
    /// (which lowers it to a [`Value`] immediately after this lookup), split
    /// out for a caller that needs to inspect a field BEFORE lowering it —
    /// e.g. [`uniform_logpdf`]'s `support`, a set expression like
    /// `interval(lo, hi)` with no tensor form of its own to lower (see
    /// [`Emitter::valueset_of`]'s doc comment).
    pub fn field_id(&self, e: &Emitter, name: &str) -> Result<NodeId, EmitError> {
        let field = match e.node(self.kernel_input) {
            Node::Call(c) => c.named.iter().find_map(|n| {
                (n.kind == NamedKind::Field && e.resolve(n.name) == name).then_some(n.value)
            }),
            _ => None,
        };
        field.ok_or_else(|| {
            EmitError::at(
                self.kernel_input,
                format!("distribution parameter '{name}' missing from kernel input"),
            )
        })
    }

    /// The raw (pre-[`Emitter::lower_node`]) [`NodeId`] of the scored variate
    /// `v` — the variate-side mirror of [`Params::field_id`]. Needed by
    /// [`categorical_logpdf`]/[`categorical0_logpdf`], whose `get`/`get0`
    /// selector into `p` must be inspected structurally (is it a literal
    /// integer?) before it can be used as a static slice bound; see
    /// `ops::literal_index`'s identical discipline for an ordinary `get`/
    /// `get0` call's selector. Refuses (rather than panicking) if this
    /// `Params` was built by [`lower_sample`], which has no scored variate at
    /// all — an internal-contract violation (only a `@logdensity` builder
    /// should ever call this), reported the same way as every other
    /// caller-contract mismatch in this module.
    pub fn variate_id(&self) -> Result<NodeId, EmitError> {
        self.variate.ok_or_else(|| {
            EmitError::at(
                self.kernel_input,
                "no scored variate in this context (only builtin_logdensityof provides one)",
            )
        })
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

    let params = Params {
        kernel_input,
        variate: Some(v),
    };
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

    let params = Params {
        kernel_input,
        variate: None,
    };
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

// ---- §08 remaining univariate continuous batch (Task 10) --------------------
//
// Uniform/Beta/StudentT/GeneralizedNormal/VonMises, registered alongside the
// rest of §08 in `REGISTRY` with `sample: None` (samplers land in Task 14).
// Beta/StudentT/GeneralizedNormal need only `chlo.lgamma` and the elementary
// op helpers, same as Task 9's gamma-family batch. Uniform and VonMises are
// each a special case in their own way:
//
// - Uniform's `-log(lambda(S))` needs `S`'s statically-known LENGTH, not a
//   per-observation formula in `v` at all (`v` is unused: `S`-membership
//   itself is a separate concern the measure layer handles upstream via
//   `restrict`/`truncate`, same division of labour every other §08 builder
//   here already assumes — none of them re-check their own support either,
//   e.g. `gamma_logpdf` never checks `x > 0`). See [`uniform_logpdf`].
// - VonMises needs `log I_0(kappa)`, the log of the order-0 modified Bessel
//   function of the first kind — StableHLO/CHLO has NO Bessel op at all
//   (`chlo.bessel_i0e` does not exist; no pretty or generic form parses), so
//   [`log_bessel_i0`] inlines the Abramowitz & Stegun 9.8.1/9.8.2 rational
//   approximations instead of emitting a nonexistent op.

/// The Lebesgue measure `lambda(S)` of a value-set `S`, when `S` is a
/// closed-form measurable interval: a plain `ValueSet::Interval(lo, hi)`
/// with finite, correctly-ordered bounds (length `hi - lo`), or
/// `ValueSet::UnitInterval` (length 1, `[0, 1]`). `None` for anything else —
/// `Unknown`/`Deferred` (the support's bounds are not static literals — spec
/// §03's `ValueSet::Interval` only ever holds compile-time-constant bounds,
/// never a parameter-dependent one), an unbounded set (`Reals`/`PosReals`/
/// `NonNegReals`/…, infinite Lebesgue measure — spec §08 requires `0 <
/// lambda(S) < infinity`), a discrete set, or a `CartProd`/`CartPow`/
/// `RecordSet` "box" shape: `Uniform`'s FlatPDL domain is hardcoded to
/// `scalar(real)` regardless of its support argument's shape (`crates/infer`'s
/// catalogue, `Distribution(domain: Scalar(Real), support: Structural, ...)`
/// — support is the only arg-dependent half), so a multi-dimensional support
/// set could never actually bind a usable variate downstream; refusing it
/// here rather than lowering a `-log(box-volume)` nobody could reach is the
/// refuse-don't-mislower call. [`uniform_logpdf`] turns `None` into a
/// precise refusal.
fn lebesgue_measure(vs: &ValueSet) -> Option<f64> {
    match vs {
        ValueSet::Interval(lo, hi) if lo.is_finite() && hi.is_finite() && hi > lo => Some(hi - lo),
        ValueSet::UnitInterval => Some(1.0),
        _ => None,
    }
}

/// §08 Uniform, verbatim: `log f = -log(lambda(S))`, `S` the `support`
/// parameter. `v` is unused (see the batch doc comment above). `support`'s
/// raw kernel-input [`NodeId`] — not its lowered [`Value`]: a set expression
/// like `interval(lo, hi)` has no tensor form of its own, see
/// `Emitter::valueset_of`'s doc comment — is read via [`Params::field_id`],
/// then its statically-known [`ValueSet`] via [`Emitter::valueset_of`] and
/// reduced to a length via [`lebesgue_measure`]. No `@sample` builder yet
/// (`sample: None`; Task 14).
fn uniform_logpdf(e: &mut Emitter, p: &Params, _v: &Value) -> Result<Value, EmitError> {
    let support = p.field_id(e, "support")?;
    let measure = e
        .valueset_of(support)
        .and_then(lebesgue_measure)
        .ok_or_else(|| {
            EmitError::at(
                support,
                "Uniform logpdf needs a measurable interval/box support",
            )
        })?;
    Ok(e.scalar(-measure.ln()))
}

/// §08 Beta, verbatim: `log f = (alpha - 1) * log(x) + (beta - 1) *
/// log(1 - x) - [lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)]`. No
/// `@sample` builder yet (`sample: None`; Task 14).
fn beta_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let alpha = p.get(e, "alpha")?;
    let beta = p.get(e, "beta")?;

    let one = e.scalar(1.0);
    let alpha_minus_one = e.sub(&alpha, &one);
    let log_x = e.log(v);
    let t1 = e.mul(&alpha_minus_one, &log_x);

    let beta_minus_one = e.sub(&beta, &one);
    let one_minus_x = e.sub(&one, v);
    let log_one_minus_x = e.log(&one_minus_x);
    let t2 = e.mul(&beta_minus_one, &log_one_minus_x);

    let lgamma_alpha = e.lgamma(&alpha);
    let neg_lgamma_alpha = e.neg(&lgamma_alpha);
    let lgamma_beta = e.lgamma(&beta);
    let neg_lgamma_beta = e.neg(&lgamma_beta);
    let alpha_plus_beta = e.add(&alpha, &beta);
    let lgamma_alpha_plus_beta = e.lgamma(&alpha_plus_beta);

    let t3 = e.add(&t1, &t2);
    let t4 = e.add(&neg_lgamma_alpha, &neg_lgamma_beta);
    let t5 = e.add(&t4, &lgamma_alpha_plus_beta);
    Ok(e.add(&t3, &t5))
}

/// §08 StudentT, verbatim: with `half_nu_plus_one = (nu + 1) / 2`, `log f =
/// lgamma(half_nu_plus_one) - 1/2 * log(nu * pi) - lgamma(nu / 2) -
/// half_nu_plus_one * log(1 + x^2 / nu)`. `half_nu_plus_one` is computed once
/// and its [`Value`] reused for both `lgamma`'s argument and the trailing
/// term's coefficient — the spec's `(nu + 1) / 2` appears in both positions
/// verbatim, same reuse discipline as [`lognormal_logpdf`]'s shared `log(x)`.
/// No `@sample` builder yet (`sample: None`; Task 14).
fn studentt_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let nu = p.get(e, "nu")?;

    let one = e.scalar(1.0);
    let half = e.scalar(0.5);

    let nu_plus_one = e.add(&nu, &one);
    let half_nu_plus_one = e.mul(&half, &nu_plus_one);
    let lgamma_a = e.lgamma(&half_nu_plus_one);

    let pi = e.scalar(std::f64::consts::PI);
    let nu_pi = e.mul(&nu, &pi);
    let log_nu_pi = e.log(&nu_pi);
    let half_log_nu_pi = e.mul(&half, &log_nu_pi);
    let neg_half_log_nu_pi = e.neg(&half_log_nu_pi);

    let half_nu = e.mul(&half, &nu);
    let lgamma_b = e.lgamma(&half_nu);
    let neg_lgamma_b = e.neg(&lgamma_b);

    let x_sq = e.mul(v, v);
    let x_sq_over_nu = e.div(&x_sq, &nu);
    let one_plus_x_sq_over_nu = e.add(&one, &x_sq_over_nu);
    let log_one_plus = e.log(&one_plus_x_sq_over_nu);
    let coef_log = e.mul(&half_nu_plus_one, &log_one_plus);
    let neg_coef_log = e.neg(&coef_log);

    let t1 = e.add(&lgamma_a, &neg_half_log_nu_pi);
    let t2 = e.add(&t1, &neg_lgamma_b);
    Ok(e.add(&t2, &neg_coef_log))
}

/// §08 GeneralizedNormal, verbatim: `log f = log(beta) - log(2 * alpha) -
/// lgamma(1 / beta) - (|x - mean| / alpha)^beta`. No `@sample` builder yet
/// (`sample: None`; Task 14).
fn generalized_normal_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mean = p.get(e, "mean")?;
    let alpha = p.get(e, "alpha")?;
    let beta = p.get(e, "beta")?;

    let log_beta = e.log(&beta);

    let two = e.scalar(2.0);
    let two_alpha = e.mul(&two, &alpha);
    let log_two_alpha = e.log(&two_alpha);
    let neg_log_two_alpha = e.neg(&log_two_alpha);

    let one = e.scalar(1.0);
    let inv_beta = e.div(&one, &beta);
    let lgamma_inv_beta = e.lgamma(&inv_beta);
    let neg_lgamma_inv_beta = e.neg(&lgamma_inv_beta);

    let diff = e.sub(v, &mean);
    let abs_diff = e.abs(&diff);
    let z = e.div(&abs_diff, &alpha);
    let z_pow_beta = e.pow(&z, &beta);
    let neg_z_pow_beta = e.neg(&z_pow_beta);

    let t1 = e.add(&log_beta, &neg_log_two_alpha);
    let t2 = e.add(&t1, &neg_lgamma_inv_beta);
    Ok(e.add(&t2, &neg_z_pow_beta))
}

/// §08 VonMises, verbatim: `log f = kappa * cos(x - mu) - log(2 * pi) -
/// log(I_0(kappa))`. `log(I_0(kappa))` is [`log_bessel_i0`]'s inlined A&S
/// approximation (no `chlo.bessel_i0e` op exists — see the batch doc
/// comment). No `@sample` builder yet (`sample: None`; Task 14).
fn von_mises_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mu = p.get(e, "mu")?;
    let kappa = p.get(e, "kappa")?;

    let diff = e.sub(v, &mu);
    let cos_diff = e.cos(&diff);
    let kappa_cos = e.mul(&kappa, &cos_diff);

    let neg_log_two_pi = e.scalar(-(2.0 * std::f64::consts::PI).ln());

    let log_i0 = log_bessel_i0(e, &kappa);
    let neg_log_i0 = e.neg(&log_i0);

    let t1 = e.add(&kappa_cos, &neg_log_two_pi);
    Ok(e.add(&t1, &neg_log_i0))
}

/// `log I_0(kappa)` via the Abramowitz & Stegun 9.8.1 (small-`kappa`) /
/// 9.8.2 (large-`kappa`) rational approximations, branching on `kappa <
/// 3.75` with [`Emitter::select`] — `chlo.bessel_i0e` is not a real CHLO op
/// (no pretty or generic form parses against the real StableHLO+CHLO
/// parser), so this inlines the polynomial rather than emitting a
/// nonexistent op. `select` unconditionally evaluates both operands (it is
/// not a lazy `ifelse`), so both branches are always computed here — safe:
/// `kappa` is `posreals` (spec §08), so `log(kappa)` in the large branch
/// never sees a non-positive input. Accurate to ~1e-7 (A&S's own stated
/// bound for both approximations), not machine epsilon — a deliberate,
/// documented trade-off of inlining a closed-form rational approximation for
/// a special function with no native op, verified against `scipy.stats.
/// vonmises.logpdf` (Task 10 report).
fn log_bessel_i0(e: &mut Emitter, kappa: &Value) -> Value {
    let threshold = e.scalar(3.75);
    let is_small = e.compare("LT", kappa, &threshold);

    // Small branch (A&S 9.8.1): t = (kappa / 3.75)^2, I_0 ~= a degree-6
    // polynomial in t (Horner form), then log.
    let ratio = e.div(kappa, &threshold);
    let t_small = e.mul(&ratio, &ratio);
    let i0_small = horner(
        e,
        &t_small,
        &[
            1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813,
        ],
    );
    let log_i0_small = e.log(&i0_small);

    // Large branch (A&S 9.8.2): t = 3.75 / kappa, log I_0 = kappa -
    // 1/2 * log(kappa) + log(a degree-8 polynomial in t, Horner form).
    let t_large = e.div(&threshold, kappa);
    let poly_large = horner(
        e,
        &t_large,
        &[
            0.39894228,
            0.01328592,
            0.00225319,
            -0.00157565,
            0.00916281,
            -0.02057706,
            0.02635537,
            -0.01647633,
            0.00392377,
        ],
    );
    let log_poly_large = e.log(&poly_large);
    let log_kappa = e.log(kappa);
    let half = e.scalar(0.5);
    let half_log_kappa = e.mul(&half, &log_kappa);
    let neg_half_log_kappa = e.neg(&half_log_kappa);
    let kappa_minus_half_log_kappa = e.add(kappa, &neg_half_log_kappa);
    let log_i0_large = e.add(&kappa_minus_half_log_kappa, &log_poly_large);

    e.select(&is_small, &log_i0_small, &log_i0_large)
}

/// Horner-scheme evaluation, at `t`, of the polynomial whose
/// ascending-power coefficients are `coeffs` (`coeffs[0]` the constant
/// term): composes only [`Emitter::mul`]/[`Emitter::add`]/[`Emitter::scalar`]
/// — no raw op text — shared by [`log_bessel_i0`]'s two A&S rational-
/// approximation branches. Panics on an empty `coeffs` (an internal-caller
/// invariant — both [`log_bessel_i0`] call sites pass a fixed non-empty
/// literal array — mirroring this crate's other panic-on-bad-input helpers,
/// e.g. `Emitter::vector`'s empty-elems assert).
fn horner(e: &mut Emitter, t: &Value, coeffs: &[f64]) -> Value {
    let (&last, init) = coeffs
        .split_last()
        .expect("horner: coeffs must be non-empty");
    let mut acc = e.scalar(last);
    for &c in init.iter().rev() {
        let scaled = e.mul(&acc, t);
        let c_val = e.scalar(c);
        acc = e.add(&scaled, &c_val);
    }
    acc
}

// ---- §08 univariate discrete batch (Task 11) --------------------------------
//
// Bernoulli/Poisson/Binomial/Geometric/NegativeBinomial/NegativeBinomial2/
// Categorical/Categorical0, registered alongside the rest of §08 in
// `REGISTRY` with `sample: None` (discrete `@sample` builders land in Task
// 16, alongside Multinomial and the finalized refuse-@sample set — see the
// roadmap doc). Binomial needs `logC(n, k) = lgamma(n+1) -
// lgamma(k+1) - lgamma(n-k+1)`, inlined directly in [`binomial_logpdf`] (the
// task brief's general form; NegativeBinomial/NegativeBinomial2 below use
// their own already-lgamma-reduced log-forms instead, so this closed form
// has only the one call site — no shared helper). Poisson/NegativeBinomial/
// NegativeBinomial2 also need `log(k!) = lgamma(k+1)` directly. Categorical/
// Categorical0 are a special case in their own way, same division as
// Uniform/VonMises in the continuous batches above: their density is `log
// p_k`, a `get`/`get0` selector into the probability vector `p` rather than
// a per-observation formula built from arithmetic on `v` — see
// [`categorical_logpdf`]'s doc comment.

/// §08 Bernoulli, verbatim: `log f = k * log(p) + (1 - k) * log(1 - p)`. No
/// `@sample` builder yet (`sample: None`; Task 16).
fn bernoulli_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let prob = p.get(e, "p")?;

    let log_p = e.log(&prob);
    let k_log_p = e.mul(v, &log_p);

    let one = e.scalar(1.0);
    let one_minus_k = e.sub(&one, v);
    let one_minus_p = e.sub(&one, &prob);
    let log_one_minus_p = e.log(&one_minus_p);
    let term2 = e.mul(&one_minus_k, &log_one_minus_p);

    Ok(e.add(&k_log_p, &term2))
}

/// §08 Poisson, verbatim: `log f = k * log(rate) - rate - lgamma(k + 1)`
/// (`log(k!) = lgamma(k+1)`). No `@sample` builder yet (`sample: None`; Task
/// 16).
fn poisson_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let rate = p.get(e, "rate")?;

    let log_rate = e.log(&rate);
    let k_log_rate = e.mul(v, &log_rate);
    let neg_rate = e.neg(&rate);

    let one = e.scalar(1.0);
    let k_plus_one = e.add(v, &one);
    let lgamma_k1 = e.lgamma(&k_plus_one);
    let neg_lgamma_k1 = e.neg(&lgamma_k1);

    let t1 = e.add(&k_log_rate, &neg_rate);
    Ok(e.add(&t1, &neg_lgamma_k1))
}

/// §08 Binomial, verbatim: `log f = logC(n, k) + k * log(p) + (n - k) *
/// log(1 - p)`, with `logC(n, k) = lgamma(n+1) - lgamma(k+1) -
/// lgamma(n-k+1)` (task brief, verbatim). `n - k` is computed once and its
/// `Value` reused for both `logC`'s `lgamma(n-k+1)` term and the trailing
/// `(n-k) * log(1-p)` term — the spec's `n - k` appears in both positions
/// verbatim, same reuse discipline as [`lognormal_logpdf`]'s shared `log(x)`.
/// No `@sample` builder yet (`sample: None`; Task 16).
fn binomial_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let n = p.get(e, "n")?;
    let prob = p.get(e, "p")?;

    let one = e.scalar(1.0);

    let n_plus_one = e.add(&n, &one);
    let lgamma_n1 = e.lgamma(&n_plus_one);

    let k_plus_one = e.add(v, &one);
    let lgamma_k1 = e.lgamma(&k_plus_one);
    let neg_lgamma_k1 = e.neg(&lgamma_k1);

    let n_minus_k = e.sub(&n, v);
    let n_minus_k_plus_one = e.add(&n_minus_k, &one);
    let lgamma_nmk1 = e.lgamma(&n_minus_k_plus_one);
    let neg_lgamma_nmk1 = e.neg(&lgamma_nmk1);

    let t1 = e.add(&lgamma_n1, &neg_lgamma_k1);
    let log_choose_nk = e.add(&t1, &neg_lgamma_nmk1);

    let log_p = e.log(&prob);
    let k_log_p = e.mul(v, &log_p);

    let one_minus_p = e.sub(&one, &prob);
    let log_one_minus_p = e.log(&one_minus_p);
    let n_minus_k_log_one_minus_p = e.mul(&n_minus_k, &log_one_minus_p);

    let t2 = e.add(&log_choose_nk, &k_log_p);
    Ok(e.add(&t2, &n_minus_k_log_one_minus_p))
}

/// §08 Geometric, verbatim: `log f = log(p) + k * log(1 - p)` — `k` is the
/// number of FAILURES before the first success (0-based, `k in nonnegintegers`;
/// see [`geometric_logpdf`]'s numeric verification against `scipy.stats.geom`
/// in the Task 11 report, whose own `k` convention counts TRIALS, 1-based). No
/// `@sample` builder yet (`sample: None`; Task 16).
fn geometric_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let prob = p.get(e, "p")?;

    let log_p = e.log(&prob);

    let one = e.scalar(1.0);
    let one_minus_p = e.sub(&one, &prob);
    let log_one_minus_p = e.log(&one_minus_p);
    let k_log_one_minus_p = e.mul(v, &log_one_minus_p);

    Ok(e.add(&log_p, &k_log_one_minus_p))
}

/// §08 NegativeBinomial, verbatim: `log f = logC(k + alpha - 1, alpha - 1) +
/// alpha * (log(beta) - log(beta + 1)) - k * log(beta + 1)`, with `logC(k +
/// alpha - 1, alpha - 1) = lgamma(k + alpha) - lgamma(alpha) - lgamma(k + 1)`
/// (the task brief's already-reduced closed form — computing the raw `(n, k)
/// = (k+alpha-1, alpha-1)` pair first and expanding `logC` from there, as
/// [`binomial_logpdf`] does for its own `(n, k)` pair, would reach the same
/// three lgammas via one extra `sub`/`add` pair; inlining the already-reduced
/// form here is the smaller op count). No `@sample` builder yet (`sample:
/// None`; Task 16).
fn negative_binomial_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let alpha = p.get(e, "alpha")?;
    let beta = p.get(e, "beta")?;

    let k_plus_alpha = e.add(v, &alpha);
    let lgamma_k_alpha = e.lgamma(&k_plus_alpha);
    let lgamma_alpha = e.lgamma(&alpha);
    let neg_lgamma_alpha = e.neg(&lgamma_alpha);
    let one = e.scalar(1.0);
    let k_plus_one = e.add(v, &one);
    let lgamma_k1 = e.lgamma(&k_plus_one);
    let neg_lgamma_k1 = e.neg(&lgamma_k1);

    let t1 = e.add(&lgamma_k_alpha, &neg_lgamma_alpha);
    let t2 = e.add(&t1, &neg_lgamma_k1);

    let log_beta = e.log(&beta);
    let beta_plus_one = e.add(&beta, &one);
    let log_beta_plus_one = e.log(&beta_plus_one);
    let neg_log_beta_plus_one = e.neg(&log_beta_plus_one);
    let log_ratio = e.add(&log_beta, &neg_log_beta_plus_one);
    let alpha_log_ratio = e.mul(&alpha, &log_ratio);

    let k_log_beta_plus_one = e.mul(v, &log_beta_plus_one);
    let neg_k_log_beta_plus_one = e.neg(&k_log_beta_plus_one);

    let t3 = e.add(&t2, &alpha_log_ratio);
    Ok(e.add(&t3, &neg_k_log_beta_plus_one))
}

/// §08 NegativeBinomial2, verbatim: `log f = logC(k + psi - 1, k) + k *
/// (log(mu) - log(mu + psi)) + psi * (log(psi) - log(mu + psi))`, with
/// `logC(k + psi - 1, k) = lgamma(k + psi) - lgamma(psi) - lgamma(k + 1)` —
/// same already-reduced-form reasoning as [`negative_binomial_logpdf`]'s doc
/// comment. `log(mu + psi)` is computed once and its negation reused for both
/// the `k`- and `psi`-weighted ratio terms (the spec's `mu + psi` denominator
/// appears in both positions verbatim — same reuse discipline as
/// `lognormal_logpdf`'s shared `log(x)`). No `@sample` builder yet (`sample:
/// None`; Task 16).
fn negative_binomial2_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mu = p.get(e, "mu")?;
    let psi = p.get(e, "psi")?;

    let k_plus_psi = e.add(v, &psi);
    let lgamma_k_psi = e.lgamma(&k_plus_psi);
    let lgamma_psi = e.lgamma(&psi);
    let neg_lgamma_psi = e.neg(&lgamma_psi);
    let one = e.scalar(1.0);
    let k_plus_one = e.add(v, &one);
    let lgamma_k1 = e.lgamma(&k_plus_one);
    let neg_lgamma_k1 = e.neg(&lgamma_k1);

    let t1 = e.add(&lgamma_k_psi, &neg_lgamma_psi);
    let t2 = e.add(&t1, &neg_lgamma_k1);

    let mu_plus_psi = e.add(&mu, &psi);
    let log_mu_plus_psi = e.log(&mu_plus_psi);
    let neg_log_mu_plus_psi = e.neg(&log_mu_plus_psi);

    let log_mu = e.log(&mu);
    let log_ratio_mu = e.add(&log_mu, &neg_log_mu_plus_psi);
    let k_log_ratio_mu = e.mul(v, &log_ratio_mu);

    let log_psi = e.log(&psi);
    let log_ratio_psi = e.add(&log_psi, &neg_log_mu_plus_psi);
    let psi_log_ratio_psi = e.mul(&psi, &log_ratio_psi);

    let t3 = e.add(&t2, &k_log_ratio_mu);
    Ok(e.add(&t3, &psi_log_ratio_psi))
}

/// Extract element `idx` (0-based, into the underlying `p` array) of the
/// rank-1 tensor `probs` as a `Scalar`, via `stablehlo.slice` + `stablehlo.
/// reshape` — the same slice+reshape idiom `ops::lower_get` uses for an
/// ordinary `get`/`get0` call, reimplemented here (rather than calling that
/// private-to-`ops.rs` function) because this caller already has the integer
/// index in hand, not an unlowered selector `NodeId` to re-derive it from.
/// Refuses (never panics) on a negative or out-of-(statically-known-)range
/// index, or a `probs` that isn't rank-1 — reachable from arbitrary
/// FlatPDL, not just the determiniser's own well-formed output.
fn slice_indexed_prob(
    e: &mut Emitter,
    blame: NodeId,
    probs: &Value,
    idx: i64,
) -> Result<Value, EmitError> {
    if idx < 0 {
        return Err(EmitError::at(
            blame,
            "Categorical/Categorical0 logdensity: category index out of range",
        ));
    }
    let idx = idx as u64;
    let len = match &probs.ty {
        MlirTy::Ranked(dims) if dims.len() == 1 => dims[0],
        other => {
            return Err(EmitError::at(
                blame,
                format!(
                    "Categorical/Categorical0 logdensity: 'p' must be a rank-1 tensor, got {other:?}"
                ),
            ));
        }
    };
    if let Some(len) = len {
        if idx >= len {
            return Err(EmitError::at(
                blame,
                "Categorical/Categorical0 logdensity: category index out of range",
            ));
        }
    }
    let sliced = e.slice(probs, &[idx], &[idx + 1], &[1]);
    Ok(e.reshape(&sliced, MlirTy::Scalar))
}

/// The literal-integer value of the scored variate `k`, or a precise refusal
/// naming the unsupported dynamic-gather case — [`categorical_logpdf`]/
/// [`categorical0_logpdf`]'s shared selector check, mirroring `ops::
/// literal_index`'s identical literal-only discipline for an ordinary `get`/
/// `get0` call (the determiniser's own discrete-marginal expansion,
/// `flatppl_determinizer::marginal`, always scores a `Categorical`/
/// `Categorical0` mass term at a literal atom value — see that module's doc
/// comment — so this is the shape every real caller reaching this registry
/// entry already has; a *dynamic* `k` would need a `stablehlo.gather` this
/// emitter has no helper for yet).
fn literal_variate_index(e: &Emitter, p: &Params) -> Result<i64, EmitError> {
    let id = p.variate_id()?;
    match e.node(id) {
        Node::Lit(Scalar::Int(i)) => Ok(*i),
        _ => Err(EmitError::at(
            id,
            "Categorical/Categorical0 logdensity: observed category must be a literal integer \
             (dynamic gather is not supported)",
        )),
    }
}

/// §08 Categorical, verbatim: `log f = log(p_k)`, `k` 1-based (`p_k` = `get(p,
/// k)`'s convention, spec-matching: `ops::lower_get`'s `get` head is already
/// 1-based, so `k`'s 1-based selector reduces to the same 0-based array
/// position `k - 1` that `get(p, k)` itself would slice). `v` (the eagerly-
/// [`Emitter::lower_node`]d variate `Value` every `LogpdfBuilder` receives) is
/// unused here — unlike every arithmetic-formula builder above, this density
/// is a lookup, not a function of `v`'s lowered tensor form; the un-lowered
/// selector integer read via [`Params::variate_id`] is what actually drives
/// the slice. No `@sample` builder yet (`sample: None`; Task 16 — `searchsorted(cumsum(p), U)`).
fn categorical_logpdf(e: &mut Emitter, p: &Params, _v: &Value) -> Result<Value, EmitError> {
    let probs = p.get(e, "p")?;
    let variate = p.variate_id()?;
    let k = literal_variate_index(e, p)?;
    let elem = slice_indexed_prob(e, variate, &probs, k - 1)?;
    Ok(e.log(&elem))
}

/// §08 Categorical0, verbatim: `log f = log(p_{k+1})`, `k` 0-based. Under
/// `Categorical`'s 1-based `p_j` numbering, `p_{k+1}` is exactly `get(p, k +
/// 1)`'s slice, i.e. array position `(k + 1) - 1 = k` — the same 0-based
/// array position `get0(p, k)` would slice directly. See
/// [`categorical_logpdf`]'s doc comment for the shared `v`-unused /
/// selector-read shape.
fn categorical0_logpdf(e: &mut Emitter, p: &Params, _v: &Value) -> Result<Value, EmitError> {
    let probs = p.get(e, "p")?;
    let variate = p.variate_id()?;
    let k = literal_variate_index(e, p)?;
    let elem = slice_indexed_prob(e, variate, &probs, k)?;
    Ok(e.log(&elem))
}

// ---- §08/§09 multivariate vector batch (Task 12) ----------------------------
//
// MvNormal/Dirichlet/Multinomial, registered alongside the rest of §08 in
// `REGISTRY` with `sample: None` (MvNormal's straight-line reparam sampler
// lands in Task 14, Dirichlet's rejection sampler in Task 15, Multinomial's
// in Task 16 — see the roadmap doc). Unlike every scalar builder above,
// `mu`/`cov`/`alpha`/`p`/`v` here are rank-1 (vector) or rank-2 (matrix)
// `Value`s, not `Scalar`s: [`Emitter::lgamma`]/[`Emitter::log`]/
// [`Emitter::neg`] are elementwise (same shape in, same shape out — see
// their own doc comments), so they apply to a vector operand exactly as they
// do to a scalar one; only the FINAL combination (after a
// [`Emitter::reduce_sum`] has collapsed a vector term to a `Scalar`) ever
// mixes shapes. A vector/matrix-shaped additive identity (e.g. Dirichlet's
// `alpha - 1`) needs a same-shape constant, not a bare [`Emitter::scalar`]:
// StableHLO's elementwise ops require identical operand *types* (no implicit
// scalar broadcast — see `ops::broadcast_to`'s doc comment for the same
// constraint elsewhere in this crate), so [`Emitter::constant`] is called
// directly with the operand's own `MlirTy` to get an already-shaped splat
// constant instead.

/// The statically-known length of a rank-1 vector `Value`, or a precise
/// refusal naming `blame` — [`mvnormal_logpdf`]'s `n` (task brief: "the
/// vector length, a static dim of `mu`/`x`") is baked into a scalar literal
/// constant (`-(n/2) * log(2*pi)`), which needs `n` known at EMIT time, not
/// merely well-typed. A `Dim::Dynamic` vector length is a legitimate FlatPDL
/// type elsewhere in the language (`elementof(cartpow(reals, m))` with an
/// unbound `m`), so this refuses precisely — refuse-don't-mislower — rather
/// than only surfacing as a downstream panic from some later op that assumes
/// a static shape.
fn static_vector_len(blame: NodeId, v: &Value) -> Result<u64, EmitError> {
    match &v.ty {
        MlirTy::Ranked(dims) if dims.len() == 1 => dims[0].ok_or_else(|| {
            EmitError::at(
                blame,
                "MvNormal logdensity needs a statically-known vector length for 'mu'",
            )
        }),
        other => Err(EmitError::at(
            blame,
            format!("MvNormal logdensity: 'mu' must be a rank-1 vector, got {other:?}"),
        )),
    }
}

/// `cov`'s `MlirTy` must be a square `n`x`n` matrix, matching `mu`'s own
/// statically-known length `n` — a refusal, not a downstream panic
/// (refuse-don't-mislower). Neither [`Emitter::cholesky`] nor
/// [`Emitter::tri_solve`] checks this: `cholesky` renders `a.ty` verbatim
/// with no shape validation at all, and [`Emitter::diag`] only asserts rank
/// 2 (`dims.len() == 2`), never `dims[0] == dims[1]` — so a wrong-size square
/// `cov` (e.g. `[3, 3]` against a length-2 `mu`) sails through `cholesky`/
/// `diag` and only produces operand-shape-incompatible StableHLO at the
/// final `tri_solve(L, x-mu)`, and a non-square `cov` (e.g. `[2, 3]`) reaches
/// `stablehlo.cholesky` on a non-square operand — neither is valid input to
/// any real StableHLO consumer. This guard catches both shapes up front.
fn require_square_cov(blame: NodeId, cov: &Value, n: u64) -> Result<(), EmitError> {
    match &cov.ty {
        MlirTy::Ranked(dims) if dims.len() == 2 && dims[0] == Some(n) && dims[1] == Some(n) => {
            Ok(())
        }
        other => Err(EmitError::at(
            blame,
            format!(
                "MvNormal cov must be an {n}x{n} matrix matching mu's length {n}, got {other:?}"
            ),
        )),
    }
}

/// §08 MvNormal, verbatim: `log f = -(n/2)*log(2*pi) - 1/2*log|Sigma| -
/// 1/2*(x-mu)^T Sigma^-1 (x-mu)`, with `L = cholesky(Sigma)` (lower),
/// `log|Sigma| = 2 * sum(log(diag(L)))`, and the quadratic form via `y =
/// tri_solve(L, x-mu)`, `(x-mu)^T Sigma^-1 (x-mu) = y^T y = sum(y*y)` — the
/// task brief's closed form exactly (never `Sigma^-1` explicitly: a full
/// matrix inverse has no `Emitter` helper, and solving the triangular system
/// `L y = (x-mu)` is the numerically standard way to get the same quadratic
/// form). `n`, the vector length, comes from `mu`'s own statically-known
/// shape ([`static_vector_len`]); `cov` is then checked against that same
/// `n` by [`require_square_cov`] BEFORE any matrix op runs — neither
/// `cholesky` nor `tri_solve` validates `cov`'s shape itself (see that
/// function's doc comment), so this builder must. No `@sample` builder yet
/// (`sample: None`; Task 14).
fn mvnormal_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let mu_id = p.field_id(e, "mu")?;
    let mu = e.lower_node(mu_id)?;
    let cov_id = p.field_id(e, "cov")?;
    let cov = e.lower_node(cov_id)?;
    let n = static_vector_len(mu_id, &mu)?;
    require_square_cov(cov_id, &cov, n)?;

    let l = e.cholesky(&cov);
    let diag_l = e.diag(&l);
    let log_diag_l = e.log(&diag_l);
    let sum_log_diag_l = e.reduce_sum(&log_diag_l);
    let two = e.scalar(2.0);
    let log_det = e.mul(&two, &sum_log_diag_l);
    let neg_half = e.scalar(-0.5);
    let neg_half_log_det = e.mul(&neg_half, &log_det);

    let diff = e.sub(v, &mu);
    let y = e.tri_solve(&l, &diff);
    let y_sq = e.mul(&y, &y);
    let quad = e.reduce_sum(&y_sq);
    let neg_half_quad = e.mul(&neg_half, &quad);

    let c = e.scalar(-0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln());

    let t1 = e.add(&c, &neg_half_log_det);
    Ok(e.add(&t1, &neg_half_quad))
}

/// §08 Dirichlet, verbatim: `log f = lgamma(sum(alpha)) - sum(lgamma(alpha))
/// + sum((alpha - 1) * log(x))`. `alpha - 1` needs a vector-shaped `1`
/// (`Emitter::constant(1.0, alpha.ty.clone())`, a splat — see the batch doc
/// comment on why a bare `Emitter::scalar` cannot be subtracted from a
/// vector directly). No `@sample` builder yet (`sample: None`; Task 15).
fn dirichlet_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let alpha = p.get(e, "alpha")?;

    let sum_alpha = e.reduce_sum(&alpha);
    let lgamma_sum_alpha = e.lgamma(&sum_alpha);

    let lgamma_alpha = e.lgamma(&alpha);
    let sum_lgamma_alpha = e.reduce_sum(&lgamma_alpha);
    let neg_sum_lgamma_alpha = e.neg(&sum_lgamma_alpha);

    let one_vec = e.constant(1.0, alpha.ty.clone());
    let alpha_minus_one = e.sub(&alpha, &one_vec);
    let log_x = e.log(v);
    let term = e.mul(&alpha_minus_one, &log_x);
    let sum_term = e.reduce_sum(&term);

    let t1 = e.add(&lgamma_sum_alpha, &neg_sum_lgamma_alpha);
    Ok(e.add(&t1, &sum_term))
}

/// §08 Multinomial, verbatim: `log f = lgamma(n+1) - sum(lgamma(x+1)) +
/// sum(x * log(p))`. `x + 1` needs a vector-shaped `1`, same reasoning as
/// [`dirichlet_logpdf`]'s `alpha - 1`; `n + 1` (the trial-count scalar
/// parameter, unrelated to `x`'s vector shape) needs only the ordinary
/// scalar one. No `@sample` builder yet (`sample: None`; Task 16).
fn multinomial_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let n = p.get(e, "n")?;
    let probs = p.get(e, "p")?;

    let one = e.scalar(1.0);
    let n_plus_one = e.add(&n, &one);
    let lgamma_n1 = e.lgamma(&n_plus_one);

    let one_vec = e.constant(1.0, v.ty.clone());
    let x_plus_one = e.add(v, &one_vec);
    let lgamma_x1 = e.lgamma(&x_plus_one);
    let sum_lgamma_x1 = e.reduce_sum(&lgamma_x1);
    let neg_sum_lgamma_x1 = e.neg(&sum_lgamma_x1);

    let log_p = e.log(&probs);
    let x_log_p = e.mul(v, &log_p);
    let sum_x_log_p = e.reduce_sum(&x_log_p);

    let t1 = e.add(&lgamma_n1, &neg_sum_lgamma_x1);
    Ok(e.add(&t1, &sum_x_log_p))
}

// ---- §08 matrix distribution batch (Task 13) --------------------------------
//
// Wishart/InverseWishart/LKJ/LKJCholesky, registered alongside the rest of
// §08 in `REGISTRY` with `sample: None` (no sampler is scheduled for this
// batch). The hardest batch so far: matrix trace/log-determinant, the
// multivariate gamma function, and the LKJ normalizer, composed entirely
// from Task 3's matrix helpers (`cholesky`/`diag`/`tri_solve`/`reduce_sum`)
// plus `lgamma` — never a full matrix inverse, which this emitter has no
// helper for at all.
//
// Three shared building blocks, used across all four builders below:
//
// - `log|A|` for an SPD matrix `A`, from an already-computed Cholesky factor
//   `L = cholesky(A)`: `2 * sum(log(diag(L)))` — the same identity
//   [`mvnormal_logpdf`] (Task 12) already uses for `log|Sigma|`, factored out
//   here as [`log_det_from_chol`] since every builder in this batch needs it
//   at least once (Wishart/InverseWishart twice each, LKJ once).
// - `tr(A^-1 B)`, from already-computed Cholesky factors `L_A`/`L_B`, via the
//   Frobenius identity `tr(A^-1 B) = ||L_A^-1 L_B||_F^2` (task brief,
//   verbatim): `W = tri_solve(L_A, L_B)` solves `L_A W = L_B` for the MATRIX
//   right-hand side `L_B` (`tri_solve` is shape-generic — its result type is
//   simply its r.h.s. operand's own type, vector or matrix alike; see its own
//   doc comment), then `tr = sum(W .* W)` — [`trace_via_frobenius`]. Never a
//   matrix inverse or a transposed solve.
// - `log Γ_n(a) = (n(n-1)/4) log(pi) + sum_{j=1}^n lgamma(a + (1-j)/2)`
//   (task brief, verbatim), the multivariate gamma function's log — §08's
//   Wishart/InverseWishart normalizer, [`log_mv_gamma`]. `n` is the FIXED
//   matrix dimension (already read off `scale`'s own shape by the caller —
//   see [`static_square_matrix_dim`]), so the `j` sum unrolls into `n`
//   `lgamma` calls at EMIT time (an ordinary Rust `for` loop), never a
//   StableHLO-level reduction; `a` (`nu/2`) is an ordinary runtime `Value`.
//
// LKJ/LKJCholesky additionally share `log c_n(eta)` (§08's normalizer for
// both, verbatim) — [`log_cn_lkj`] — whose own `k = 1..n-1` sum unrolls the
// same way; `eta` is again an ordinary runtime `Value`, composed via op
// helpers regardless of whether it happens to be a compile-time literal or a
// free `elementof`-declared parameter (this emitter never special-cases a
// `Value`'s origin — see e.g. [`normal_logpdf`]'s identical treatment of
// `sigma`).
//
// Unlike Wishart/InverseWishart (whose dimension `n` is always the row/
// column count of `scale`, a matrix-shaped kwarg — spec §08), LKJ/
// LKJCholesky have an explicit `n` kwarg of their own (spec: `n =
// elementof(posintegers)`) that must be spec's FIXED phase (no `elementof`/
// `draw` ancestor — spec §04) for a Rust `u64` value to exist at emit time at
// all. Verified against the real determinizer output: a fixed-phase
// binding's use site is `(%ref self n)`, one level of `(%ref self x)`
// indirection to the actual `(%bind n 3)` literal — never the literal
// inlined directly at the call site, the way e.g. a `get`/`get0` selector
// literal is (`ops::literal_index`). [`literal_fixed_positive_int`] follows
// that one level via [`Emitter::resolve_ref_one`] before matching the
// literal; a `%parameterized` (`elementof`-declared) `n` has no such literal
// to find and refuses precisely, rather than reaching a Rust `for j in
// 1..=n` with no `n` at all.

/// The statically-known dimension `n` of a square matrix `Value` (`Ranked([n,
/// n])`), or a precise refusal naming `ctor`/`param_name` — the square-matrix
/// analogue of [`static_vector_len`] (Task 12's `mu`-length check), used by
/// [`wishart_logpdf`]/[`inverse_wishart_logpdf`] to read `n` off `scale`'s own
/// shape (spec §08: "the dimension n is the row/column count of scale").
fn static_square_matrix_dim(
    blame: NodeId,
    m: &Value,
    ctor: &str,
    param_name: &str,
) -> Result<u64, EmitError> {
    match &m.ty {
        MlirTy::Ranked(dims) if dims.len() == 2 => match (dims[0], dims[1]) {
            (Some(a), Some(b)) if a == b => Ok(a),
            _ => Err(EmitError::at(
                blame,
                format!(
                    "{ctor} logdensity needs a statically-known square matrix for \
                     '{param_name}', got {:?}",
                    m.ty
                ),
            )),
        },
        other => Err(EmitError::at(
            blame,
            format!(
                "{ctor} logdensity: '{param_name}' must be a rank-2 square matrix, got {other:?}"
            ),
        )),
    }
}

/// `m`'s `MlirTy` must be exactly the square `n`x`n` matrix `param_name` is
/// expected to be — a refusal, not a downstream panic (refuse-don't-
/// mislower), mirroring [`require_square_cov`] (Task 12) for every
/// cross-check this batch needs (a scored variate against `scale`'s/`n`'s own
/// dimension): neither `cholesky`, `diag`, nor `tri_solve` validates a shape
/// mismatch itself (see `require_square_cov`'s doc comment for the same
/// reasoning), so every builder below must, before any matrix op runs.
fn require_matrix_dim(
    blame: NodeId,
    m: &Value,
    n: u64,
    ctor: &str,
    param_name: &str,
) -> Result<(), EmitError> {
    match &m.ty {
        MlirTy::Ranked(dims) if dims.len() == 2 && dims[0] == Some(n) && dims[1] == Some(n) => {
            Ok(())
        }
        other => Err(EmitError::at(
            blame,
            format!("{ctor} {param_name} must be an {n}x{n} matrix, got {other:?}"),
        )),
    }
}

/// The literal positive-integer value of the kernel-input field `field_name`
/// — needed by [`lkj_logpdf`]/[`lkj_cholesky_logpdf`] to get their explicit
/// `n` kwarg as a Rust `u64` at EMIT time, to unroll [`log_cn_lkj`]'s `k`
/// sum. Follows at most one level of `(%ref self x)` indirection via
/// [`Emitter::resolve_ref_one`] — a FIXED-phase field's use site is that
/// indirection, not the literal inlined directly (see the batch doc
/// comment) — then requires a positive `Node::Lit(Scalar::Int(_))`. Refuses
/// (never panics) for anything else, e.g. a `%parameterized`
/// (`elementof`-declared) `n`, which has no such literal to find.
fn literal_fixed_positive_int(
    e: &Emitter,
    p: &Params,
    field_name: &str,
    ctor: &str,
) -> Result<u64, EmitError> {
    let field = p.field_id(e, field_name)?;
    let resolved = e.resolve_ref_one(field);
    match e.node(resolved) {
        Node::Lit(Scalar::Int(i)) if *i > 0 => Ok(*i as u64),
        _ => Err(EmitError::at(
            field,
            format!(
                "{ctor} logdensity needs a fixed-phase positive integer literal for '{field_name}'"
            ),
        )),
    }
}

/// `log|A|` for an SPD matrix `A`, from an already-computed Cholesky factor
/// `l = cholesky(A)`: `2 * sum(log(diag(l)))` — see the batch doc comment.
fn log_det_from_chol(e: &mut Emitter, l: &Value) -> Value {
    let diag_l = e.diag(l);
    let log_diag_l = e.log(&diag_l);
    let sum_log_diag_l = e.reduce_sum(&log_diag_l);
    let two = e.scalar(2.0);
    e.mul(&two, &sum_log_diag_l)
}

/// `tr(A^-1 B)` via the Frobenius identity `tr(A^-1 B) = ||L_A^-1 L_B||_F^2`
/// (task brief, verbatim): `l_a`/`l_b` are already-computed Cholesky factors
/// of `A`/`B`. See the batch doc comment for why this needs no matrix
/// inverse or transposed solve. [`wishart_logpdf`] calls this as
/// `(l_v, l_x)` for `tr(V^-1 X)`; [`inverse_wishart_logpdf`] as `(l_x,
/// l_psi)` for `tr(Psi X^-1) = tr(X^-1 Psi)` (trace is cyclic) — see that
/// function's doc comment.
fn trace_via_frobenius(e: &mut Emitter, l_a: &Value, l_b: &Value) -> Value {
    let w = e.tri_solve(l_a, l_b);
    let w_sq = e.mul(&w, &w);
    e.reduce_sum(&w_sq)
}

/// `log Γ_n(a) = (n(n-1)/4) log(pi) + sum_{j=1}^n lgamma(a + (1-j)/2)` (task
/// brief, verbatim) — see the batch doc comment.
fn log_mv_gamma(e: &mut Emitter, n: u64, a: &Value) -> Value {
    let mut acc = e.scalar((n * (n - 1)) as f64 / 4.0 * std::f64::consts::PI.ln());
    for j in 1..=n {
        let shift = e.scalar((1.0 - j as f64) / 2.0);
        let a_j = e.add(a, &shift);
        let lgamma_j = e.lgamma(&a_j);
        acc = e.add(&acc, &lgamma_j);
    }
    acc
}

/// `log c_n(eta) = (sum_{k=1}^{n-1} (2 eta - 2 + n - k)(n - k)) log(2) +
/// sum_{k=1}^{n-1} (n - k) log B(eta + (n-k-1)/2, eta + (n-k-1)/2)`, with
/// `log B(a, a) = 2 lgamma(a) - lgamma(2a)` (task brief, verbatim) — the LKJ/
/// LKJCholesky shared normalizer (see the batch doc comment for `n`/`eta`'s
/// fixed/runtime split). The `log(2)`-exponent sum and the log-beta sum are
/// accumulated separately across the loop and combined once at the end (one
/// final `log(2)` multiply, rather than `n-1` of them). For `n = 1` (a
/// degenerate 1x1 "correlation matrix", always exactly `[1]`) the loop runs
/// zero times and this correctly returns `log(1) = 0`.
fn log_cn_lkj(e: &mut Emitter, n: u64, eta: &Value) -> Value {
    let two = e.scalar(2.0);
    let two_eta = e.mul(&two, eta);

    let mut pow2_exponent: Option<Value> = None;
    let mut logbeta_sum: Option<Value> = None;
    for k in 1..n {
        let m = n - k;
        let m_val = e.scalar(m as f64);

        let base_shift = e.scalar(m as f64 - 2.0);
        let base = e.add(&two_eta, &base_shift); // 2*eta - 2 + m
        let term = e.mul(&base, &m_val); // (2*eta - 2 + m) * m
        pow2_exponent = Some(match pow2_exponent {
            None => term,
            Some(acc) => e.add(&acc, &term),
        });

        let a_shift = e.scalar((m as f64 - 1.0) / 2.0);
        let a = e.add(eta, &a_shift); // eta + (m-1)/2
        let two_a = e.mul(&two, &a);
        let lgamma_a = e.lgamma(&a);
        let two_lgamma_a = e.mul(&two, &lgamma_a);
        let lgamma_two_a = e.lgamma(&two_a);
        let logbeta = e.sub(&two_lgamma_a, &lgamma_two_a); // 2 lgamma(a) - lgamma(2a)
        let m_logbeta = e.mul(&m_val, &logbeta);
        logbeta_sum = Some(match logbeta_sum {
            None => m_logbeta,
            Some(acc) => e.add(&acc, &m_logbeta),
        });
    }

    let ln_two = e.scalar(std::f64::consts::LN_2);
    let term1 = match pow2_exponent {
        Some(exponent) => e.mul(&exponent, &ln_two),
        None => e.scalar(0.0),
    };
    let term2 = logbeta_sum.unwrap_or_else(|| e.scalar(0.0));
    e.add(&term1, &term2)
}

/// Extract element `idx` (0-based) of a rank-1 tensor `vec` as a `Scalar`,
/// via `stablehlo.slice` + `stablehlo.reshape` — the same idiom
/// [`slice_indexed_prob`] uses for `Categorical`/`Categorical0`,
/// reimplemented narrowly here (no bounds-check/refuse plumbing) because
/// [`lkj_cholesky_logpdf`]'s `idx` always ranges over `0..n` for the ALREADY
/// statically-known `n` (its own caller's loop bound), never an arbitrary
/// selector reachable from untrusted FlatPDL.
fn vector_elem(e: &mut Emitter, vec: &Value, idx: u64) -> Value {
    let sliced = e.slice(vec, &[idx], &[idx + 1], &[1]);
    e.reshape(&sliced, MlirTy::Scalar)
}

/// §08 Wishart, verbatim: `((nu-n-1)/2) log|X| - (1/2) tr(V^-1 X) -
/// (nu*n/2) log2 - (nu/2) log|V| - logGamma_n(nu/2)`. `n` (the row/column
/// count of `scale`, i.e. `V`) comes from `scale`'s own statically-known
/// shape ([`static_square_matrix_dim`]); the variate `X` is then checked
/// against that same `n` by [`require_matrix_dim`] BEFORE any matrix op
/// runs — same discipline as [`mvnormal_logpdf`]/[`require_square_cov`]
/// (Task 12). `L_V = cholesky(V)`/`L_X = cholesky(X)` are each computed ONCE
/// and reused for both their own `log|.|` term and
/// [`trace_via_frobenius`]'s `tr(V^-1 X)`; `nu/2` is likewise computed once
/// and reused for [`log_mv_gamma`]'s argument and its own `log|V|`
/// coefficient. No `@sample` builder (`sample: None` — no sampler is
/// planned for this batch).
fn wishart_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let scale_id = p.field_id(e, "scale")?;
    let scale = e.lower_node(scale_id)?;
    let n = static_square_matrix_dim(scale_id, &scale, "Wishart", "scale")?;
    require_matrix_dim(p.variate_id()?, v, n, "Wishart", "X")?;
    let nu = p.get(e, "nu")?;

    let l_v = e.cholesky(&scale);
    let l_x = e.cholesky(v);
    let log_det_x = log_det_from_chol(e, &l_x);
    let log_det_v = log_det_from_chol(e, &l_v);
    let tr = trace_via_frobenius(e, &l_v, &l_x);

    let half = e.scalar(0.5);
    let n_plus_one = e.scalar(n as f64 + 1.0);
    let nu_minus_n1 = e.sub(&nu, &n_plus_one); // nu - n - 1
    let coef1 = e.mul(&half, &nu_minus_n1);
    let term1 = e.mul(&coef1, &log_det_x); // (nu-n-1)/2 * log|X|

    let neg_half = e.scalar(-0.5);
    let term2 = e.mul(&neg_half, &tr); // -1/2 * tr(V^-1 X)

    let n_val = e.scalar(n as f64);
    let nu_n = e.mul(&nu, &n_val);
    let ln_two = e.scalar(std::f64::consts::LN_2);
    let nu_n_ln_two = e.mul(&nu_n, &ln_two);
    let neg_half_nu_n_ln_two = e.mul(&neg_half, &nu_n_ln_two); // -(nu*n/2) * log2

    let half_nu = e.mul(&half, &nu);
    let neg_half_nu = e.neg(&half_nu);
    let term4 = e.mul(&neg_half_nu, &log_det_v); // -(nu/2) * log|V|

    let log_mvgamma = log_mv_gamma(e, n, &half_nu);
    let neg_log_mvgamma = e.neg(&log_mvgamma);

    let t1 = e.add(&term1, &term2);
    let t2 = e.add(&t1, &neg_half_nu_n_ln_two);
    let t3 = e.add(&t2, &term4);
    Ok(e.add(&t3, &neg_log_mvgamma))
}

/// §08 InverseWishart, verbatim: `(nu/2) log|Psi| - ((nu+n+1)/2) log|X| -
/// (1/2) tr(Psi X^-1) - (nu*n/2) log2 - logGamma_n(nu/2)`. Same `n`/shape-
/// guard discipline as [`wishart_logpdf`], reading `n` off `scale` (i.e.
/// `Psi`) and checking the variate `X` against it. `tr(Psi X^-1)` is
/// computed as `tr(X^-1 Psi)` instead (trace is cyclic: `tr(AB) = tr(BA)`),
/// via [`trace_via_frobenius`]`(l_x, l_psi)` — exactly the task brief's
/// "symmetric" `tr(Psi X^-1) = ||L_X^-1 L_Psi||_F^2` form, so `L_X` (needed
/// anyway for `log|X|`) doubles as the trace's left Cholesky factor instead
/// of computing a third one. `nu/2` is reused for [`log_mv_gamma`]'s
/// argument and the leading `log|Psi|` coefficient, same reuse discipline as
/// [`wishart_logpdf`]. No `@sample` builder (`sample: None`).
fn inverse_wishart_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let scale_id = p.field_id(e, "scale")?;
    let psi = e.lower_node(scale_id)?;
    let n = static_square_matrix_dim(scale_id, &psi, "InverseWishart", "scale")?;
    require_matrix_dim(p.variate_id()?, v, n, "InverseWishart", "X")?;
    let nu = p.get(e, "nu")?;

    let l_psi = e.cholesky(&psi);
    let l_x = e.cholesky(v);
    let log_det_psi = log_det_from_chol(e, &l_psi);
    let log_det_x = log_det_from_chol(e, &l_x);
    let tr = trace_via_frobenius(e, &l_x, &l_psi); // tr(X^-1 Psi) = tr(Psi X^-1)

    let half = e.scalar(0.5);
    let half_nu = e.mul(&half, &nu);
    let term1 = e.mul(&half_nu, &log_det_psi); // (nu/2) * log|Psi|

    let n_plus_one = e.scalar(n as f64 + 1.0);
    let nu_plus_n1 = e.add(&nu, &n_plus_one); // nu + n + 1
    let neg_half = e.scalar(-0.5);
    let neg_half_nu_n1 = e.mul(&neg_half, &nu_plus_n1);
    let term2 = e.mul(&neg_half_nu_n1, &log_det_x); // -(nu+n+1)/2 * log|X|

    let term3 = e.mul(&neg_half, &tr); // -1/2 * tr(...)

    let n_val = e.scalar(n as f64);
    let nu_n = e.mul(&nu, &n_val);
    let ln_two = e.scalar(std::f64::consts::LN_2);
    let nu_n_ln_two = e.mul(&nu_n, &ln_two);
    let neg_half_nu_n_ln_two = e.mul(&neg_half, &nu_n_ln_two); // -(nu*n/2) * log2

    let log_mvgamma = log_mv_gamma(e, n, &half_nu);
    let neg_log_mvgamma = e.neg(&log_mvgamma);

    let t1 = e.add(&term1, &term2);
    let t2 = e.add(&t1, &term3);
    let t3 = e.add(&t2, &neg_half_nu_n_ln_two);
    Ok(e.add(&t3, &neg_log_mvgamma))
}

/// §08 LKJ, verbatim: `log f = (eta-1) log det(C) - log c_n(eta)`. `n`
/// (fixed, spec's own explicit dimension kwarg — see the batch doc comment)
/// is read via [`literal_fixed_positive_int`], then the variate `C` is
/// checked against it by [`require_matrix_dim`] before `cholesky` runs. No
/// `@sample` builder (`sample: None`).
fn lkj_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let n = literal_fixed_positive_int(e, p, "n", "LKJ")?;
    require_matrix_dim(p.variate_id()?, v, n, "LKJ", "C")?;
    let eta = p.get(e, "eta")?;

    let l_c = e.cholesky(v);
    let log_det_c = log_det_from_chol(e, &l_c);

    let one = e.scalar(1.0);
    let eta_minus_one = e.sub(&eta, &one);
    let term1 = e.mul(&eta_minus_one, &log_det_c);

    let log_cn = log_cn_lkj(e, n, &eta);
    let neg_log_cn = e.neg(&log_cn);
    Ok(e.add(&term1, &neg_log_cn))
}

/// §08 LKJCholesky, verbatim: `log f = sum_{i=2}^{n} (n-i+2*eta-2) log L_ii -
/// log c_n(eta)`, `L_ii = diag(L)` (`i` 1-based, spec's own convention — its
/// 0-based array position is `i-1`, read via [`vector_elem`]). The variate
/// `L` (already itself the Cholesky factor — unlike [`lkj_logpdf`]'s `C`,
/// nothing here calls [`Emitter::cholesky`] at all) is checked square/sized
/// against `n` by [`require_matrix_dim`] before [`Emitter::diag`] runs. No
/// `@sample` builder (`sample: None`).
fn lkj_cholesky_logpdf(e: &mut Emitter, p: &Params, v: &Value) -> Result<Value, EmitError> {
    let n = literal_fixed_positive_int(e, p, "n", "LKJCholesky")?;
    require_matrix_dim(p.variate_id()?, v, n, "LKJCholesky", "L")?;
    let eta = p.get(e, "eta")?;

    let diag_l = e.diag(v);
    let two = e.scalar(2.0);
    let two_eta = e.mul(&two, &eta);

    let mut acc: Option<Value> = None;
    for i in 2..=n {
        let l_ii = vector_elem(e, &diag_l, i - 1);
        let log_l_ii = e.log(&l_ii);
        let coef_shift = e.scalar(n as f64 - i as f64 - 2.0);
        let coef = e.add(&two_eta, &coef_shift); // n - i + 2*eta - 2
        let term = e.mul(&coef, &log_l_ii);
        acc = Some(match acc {
            None => term,
            Some(a) => e.add(&a, &term),
        });
    }
    let sum_terms = acc.unwrap_or_else(|| e.scalar(0.0));

    let log_cn = log_cn_lkj(e, n, &eta);
    let neg_log_cn = e.neg(&log_cn);
    Ok(e.add(&sum_terms, &neg_log_cn))
}
