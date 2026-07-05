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
static REGISTRY: &[(&str, DistLowering)] = &[(
    "Normal",
    DistLowering {
        logpdf: normal_logpdf,
        sample: Some(normal_sample),
    },
)];

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
