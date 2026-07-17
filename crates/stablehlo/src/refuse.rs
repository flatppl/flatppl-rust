//! `EmitError` — refuse-don't-mislower for the StableHLO emitter: a construct
//! the emitter cannot lower is reported with a precise message, never guessed.
//!
//! ## The refuse taxonomy
//!
//! Every `EmitError::at`/`EmitError::whole` construction site in this crate,
//! grouped by module (Task 7's audit). Each is locked by a test in
//! `tests/golden.rs` unless noted otherwise — keep this list in sync when a
//! site is added, removed, or reworded.
//!
//! **`emitter.rs`** (`Emitter::lower_node`/`lower_ref`/`sample_tuple_slot`):
//! - `Lit(Scalar::Str(_))` — "string literal has no tensor form"
//! - `Node::Hole` — "bare hole has no tensor form"
//! - `Node::Axis(_)` — "axis label has no tensor form"
//! - `get`/`get0` projecting a sampled tuple's advanced rng-state slot
//!   (index 1) — "sampled rng state has no tensor form (...)": this
//!   vertical is XLA-seeded, so the threaded rng-state half of a
//!   `builtin_sample` draw has no lowering at all, ever (not a
//!   not-yet-implemented gap).
//! - `CallHead::User(_)` — "user-callable application has no lowering"
//! - `Ref{SelfMod, ..}` to an unknown top-level binding — "unresolved
//!   reference '...'"
//! - `Ref{Local, ..}` not pre-`bind`-ed by the caller — "unbound %local
//!   reference (...)"
//! - `Ref{Module(_), ..}` — "module-member reference has no lowering yet"
//!
//! **`lib.rs`** (`emit`, the mode router):
//! - input fails `flatppl_determinizer::is_flatpdl` — "input is not FlatPDL
//!   (determinize first)" (`EmitError::whole`: a module-level check, not a
//!   single-node defect)
//!
//! **`modes.rs`** (`emit_logdensity`/`emit_sample`):
//! - no public binding at all — "module has no public binding to emit as
//!   the logdensity/sample query" (`EmitError::whole`)
//! - the selected (last-public-binding) query contains no
//!   `builtin_logdensityof`/`builtin_sample` term anywhere in its subtree —
//!   "selected query output contains no density/sample term (...)" (guards
//!   the "last public binding is the query" convention)
//!
//! **`modes.rs`** (`emit_logdensity_abi`, the `inputs`/`outputs` ABI path,
//! PR-1 — see [`crate::modes::Abi`]/[`crate::modes::read_abi`]):
//! - an `elementof` parameter not listed in `inputs` — "elementof parameter
//!   '...' is not listed in `inputs`; the inputs ABI is exhaustive ..."
//!   (`inputs` is authoritative + exhaustive, design doc)
//! - `outputs` missing or empty — "`outputs` ABI binding is missing or
//!   empty; at least one output is required" (`EmitError::whole`)
//! - an `inputs` entry naming a binding absent from the determinized module
//!   — "`inputs` names '...', which is not a binding of the determinized
//!   module" (`EmitError::whole`)
//! - an `inputs` entry that is not an `elementof` parameter (i.e.
//!   `external`/`load_data`) — "`inputs` entry '...' is not an elementof
//!   parameter — external/load_data ABI inputs are not yet supported (PR-2
//!   work)"
//!
//! **`ops.rs`** (the deterministic builtin-head map):
//! - `record(...)` reached in tensor position — "record has no tensor form"
//! - an unknown builtin head — "unsupported builtin head '...'"
//! - wrong arity for any arity-checked head (`args_exact`, shared by
//!   `unary`/`binary`/`ifelse`/`get`/`get0`/`in`/`inf`) — "expected N
//!   argument(s), got M"
//! - `ifelse`'s condition is not an `in`/`compare` predicate call — "ifelse
//!   condition must be a boolean predicate (in/compare)"
//! - `broadcast_to` asked to broadcast a non-scalar, differently-shaped
//!   operand (e.g. `in`'s bounds against its variate) — "shape mismatch:
//!   cannot broadcast ... to ..."
//! - `vector()` with zero elements — "vector: expected at least one
//!   element"
//! - `vector()` whose elements are not all the same `MlirTy` (a RAGGED
//!   vector-of-vectors, e.g. inner vectors of different lengths) —
//!   "vector elements must have identical shape; ragged vector-of-vectors
//!   has no tensor form"
//! - `get`/`get0` whose computed 0-based index is negative (a selector
//!   below `get`'s 1-based floor) — "get/get0: index out of range"
//! - `get`/`get0` on a non-rank-1 container — "get/get0: only
//!   single-selector indexing into a rank-1 tensor is supported, got ..."
//! - `get`/`get0` whose computed index is `>=` a statically-known length —
//!   "get/get0: index out of range" (same message text as the negative-index
//!   case above, but a distinct guard reached only once the container has
//!   already been lowered)
//! - `get`/`get0` selector is not a literal integer — "get/get0: selector
//!   must be a literal integer"
//! - `in`'s set is not `interval(lo, hi)` — "'in': only an interval(lo, hi)
//!   set is supported" (one shared closure, invoked for either a
//!   non-`interval`-headed call or a non-`Call` set expression; the existing
//!   test exercises the non-`Call` branch, the closure itself is the one
//!   construction site)
//!
//! **`registry.rs`** (the distribution dispatch table):
//! - a kernel-input record missing a parameter a builder needs —
//!   "distribution parameter '...' missing from kernel input"
//! - `builtin_logdensityof`/`builtin_sample` wrong arity —
//!   "builtin_logdensityof/sample: expected 3 arguments, got N"
//! - kernel/ctor is not a bare `Const` distribution constructor — "...must
//!   be a bare distribution constructor"
//! - an unregistered constructor name — "no lowering for distribution '...'"
//! - a registered constructor with no `@sample` builder — "no @sample
//!   lowering for '...'" — locked by
//!   `builtin_sample_refuses_registered_ctor_without_sample_builder`
//!   (`tests/golden.rs`), reached via `VonMises` (needs a dedicated rejection
//!   sampler not in Task 15's batch; Task 15 gave `Gamma`/`InverseGamma`/
//!   `ChiSquared`/`Beta`/`StudentT`/`GeneralizedNormal`/`Dirichlet` rejection
//!   `@sample` builders, so — like `Cauchy`/`Logistic`/`Laplace` after Task
//!   14 — they no longer exercise this arm). Task 16's still-pending discrete
//!   batch (`Bernoulli`/`Poisson`/…/`Multinomial`) and the matrix batch
//!   (`Wishart`/`InverseWishart`/`LKJ`/`LKJCholesky`) keep the arm reachable.
//! - `Uniform`'s `support` parameter has no closed-form measurable
//!   interval/box `ValueSet` (`registry::lebesgue_measure` returns `None`) —
//!   "Uniform logpdf needs a measurable interval/box support" (Task 10).
//! - `MvNormal`'s `mu` has no statically-known vector length — "MvNormal
//!   logdensity needs a statically-known vector length for 'mu'" (Task 12).
//! - `MvNormal`'s `cov` is not an `n`x`n` matrix matching `mu`'s length —
//!   "MvNormal cov must be an ...x... matrix matching mu's length ..., got
//!   ..." (Task 12).
//! - a matrix-distribution (`Wishart`/`InverseWishart`) shape param with no
//!   statically-known SQUARE matrix shape — three distinct wordings for
//!   three distinct shapes, so a known-but-non-square shape is never
//!   misreported as unknown: "... logdensity needs a statically-known
//!   square matrix for '...', got ..." (a dynamic dim present) / "...
//!   logdensity: '...' must be a square matrix, got ..." (both dims
//!   statically known, just unequal, e.g. `[2, 3]`) / "... logdensity:
//!   '...' must be a rank-2 square matrix, got ..." (wrong rank entirely)
//!   (`registry::static_square_matrix_dim`, Task 13).
//! - a matrix-distribution variate that mismatches its scale/`n`'s own
//!   dimension — "... ... must be an NxN matrix, got ..."
//!   (`registry::require_matrix_dim`, Task 13).
//! - `LKJ`/`LKJCholesky`'s `n` kwarg is not a FIXED-phase positive integer
//!   literal — "... logdensity needs a fixed-phase positive integer literal
//!   for '...'" (`registry::literal_fixed_positive_int`, Task 13).
//!
//! **`types.rs`** (`mlir_type_of`):
//! - a node with no inferred type at all — "node has no inferred type"
//! - an aggregate type (`Record`/`Tuple`/`Table`) — "aggregate type has no
//!   tensor form; must be destructured"
//! - a residual measure-layer type (`Measure`/`Kernel`/`Likelihood`) —
//!   "residual measure-layer type in FlatPDL"
//! - any other non-tensor type — "type has no MLIR tensor form: {ty:?}"
//!   (names the offending type via `Debug`)
//!
//! Message wording is not perfectly uniform — e.g. `ops::args_exact`'s
//! generic arity message has no primitive-name prefix, unlike
//! `registry.rs`'s `builtin_logdensityof`/`builtin_sample` arity checks —
//! left as-is rather than threading a head name through every arity-checked
//! call site for a marginal clarity gain: `err.node` already localizes each
//! to its exact call node.

use flatppl_core::NodeId;

/// A construct `emit` cannot lower to StableHLO — reported, never mis-lowered.
///
/// `node` localizes the error to a specific IR node when one is available;
/// `whole` (`node: None`) is used for module-level refusals (e.g. the input
/// is not FlatPDL at all).
#[derive(Debug)]
pub struct EmitError {
    pub msg: String,
    pub node: Option<NodeId>,
}

impl EmitError {
    /// A refusal localized to `node`.
    pub fn at(node: NodeId, msg: impl Into<String>) -> Self {
        EmitError {
            msg: msg.into(),
            node: Some(node),
        }
    }

    /// A refusal with no single localizing node (e.g. a module-level check).
    pub fn whole(msg: impl Into<String>) -> Self {
        EmitError {
            msg: msg.into(),
            node: None,
        }
    }
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stablehlo: {}", self.msg)
    }
}
