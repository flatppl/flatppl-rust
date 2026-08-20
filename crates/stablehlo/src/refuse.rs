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
//! **`lib.rs`** ([`crate::emit`]):
//! - no `inputs`/`outputs` ABI declared (neither reserved binding present) —
//!   "no inputs/outputs ABI declared; the last-public-binding query heuristic
//!   has been removed — declare `inputs = (…)` and `outputs = (…)` ..."
//!   (`EmitError::whole`; both modes — the ABI is the sole query designation)
//!
//! **`modes.rs`** (`emit_logdensity_abi`/`emit_sample_abi`, the
//! `inputs`/`outputs` ABI path — see
//! [`crate::modes::Abi`]/[`crate::modes::read_abi`]):
//! - an `elementof` parameter not listed in `inputs` — "elementof parameter
//!   '...' is not listed in `inputs`; the inputs ABI is exhaustive ..."
//!   (`inputs` is authoritative + exhaustive, design doc)
//! - a fixed-phase input (`external`/`load_data`) reached by an output but
//!   not listed in `inputs` — "fixed-phase binding '...' is reached by an
//!   output but is not listed in `inputs`; list it in inputs to pass it as a
//!   runtime argument (...)"
//! - `outputs` missing or empty — "`outputs` ABI binding is missing or
//!   empty; at least one output is required" (`EmitError::whole`)
//! - an `inputs` entry naming a binding absent from the determinized module
//!   — "`inputs` names '...', which is not a binding of the determinized
//!   module" (`EmitError::whole`)
//! - an `inputs` entry that is neither an `elementof` parameter nor a fixed
//!   input (`external`/`load_data`) — e.g. a literal or a derived/computed
//!   binding — "`inputs` entry '...' is not an elementof parameter, external,
//!   or load_data input — only these constructs can be ABI arguments"
//! - an `inputs` entry that declares no shape — "`inputs` entry '...' declares
//!   no shape (`anything`) and cannot be promoted to a function argument (spec
//!   §13 signature); ..." (§13 `sec:determinization-signature`: `anything`
//!   "declares none and cannot be promoted"). Fires for EVERY construct over
//!   `anything`, not only `load_data`, so the message cites the signature rule
//!   rather than §07's don't-read-the-source clause — `elementof(anything)`
//!   has no source to read.
//! - a destructured aggregate entry with a column that has no tensor form of
//!   its own (a nested record/table column) — "`inputs` entry '...' column
//!   '...' has no tensor form, so this emitter cannot destructure the entry
//!   into arguments; ..." (names the column, unlike `types.rs`'s generic
//!   aggregate refusal). Worded as an EMITTER limit, not a language rule: §03
//!   Tables says "Each column is a vector or a table", so a nested table
//!   column is legal FlatPPL that this backend declines to flatten.
//! - (`emit_sample_abi`) `outputs` naming other than exactly one output —
//!   "`outputs` for a sample query must name exactly one output (the sampled
//!   value)" (`EmitError::whole`)
//! - (`emit_sample_abi`) the declared sample output reaches no rng source —
//!   "no rng source to bind to %key: the declared sample output reaches no
//!   rnginit/external(rngstates) source to thread from"
//!
//! (`external`/`load_data` inputs LISTED in `inputs` are supported — they
//! become function arguments, `load_data` shaped from its declared `valueset`,
//! values never baked. An aggregate valueset destructures into one argument per
//! column, so the `types.rs` aggregate refusal below is reached only by an
//! aggregate in tensor position, not by a table ABI input.)
//!
//! **`ops.rs`** (the deterministic builtin-head map):
//! - `record(...)` reached in tensor position — "record has no tensor form"
//! - a destructured `load_data` input used as one monolithic value — "a
//!   load_data input whose valueset is a table or record has no single tensor
//!   form; read it column-wise (`data.y`) — one argument per column"
//! - `sum` over a TABLE — the shared `norms::table_reduction_refusal`, "a table
//!   reduction has no tensor form: §07 \"Table reductions\" makes `sum` over a
//!   table a RECORD of per-column reductions, ...". `prod`/`mean`/`var`/`std`
//!   raise the identical refusal from `norms.rs` (below); the message is one
//!   function so the five cannot drift apart. A boolean ARRAY is not refused:
//!   §03 "Bool" promotes it ("`sum(mask)` to count true entries"), `infer` types
//!   the result `integers`, and `Emitter::reduce_axis` emits the matching
//!   `stablehlo.convert` to `i32`.
//! - `maximum`/`minimum` over a non-real array — "maximum/minimum: only a real
//!   array is supported, got ...". Booleans are refused here even though `sum`
//!   accepts them, and deliberately: §03's promotion covers arithmetic
//!   contexts, while $\max_i x_i$ selects an element, so `infer` keeps
//!   `maximum(bool_array)` at `booleans` and there is no `i1` `stablehlo.reduce`
//!   identity in this emitter to lower it with.
//! - an elementwise binary op whose operands do not broadcast (different rank,
//!   or an axis pair neither equal nor size-1) — "elementwise operands do not
//!   broadcast: ... — §04 broadcasting needs equal rank ... (a matrix product
//!   is the non-elementwise `*`, not `.*`)". This is `ops::require_broadcastable`
//!   turning what was an `Emitter::broadcast_pair` PANIC into a refusal; the
//!   infallible `binary`/`compare`/`select` helpers still assert, but no
//!   arity-2 elementwise head can now reach that assertion.
//! - a matrix product (`*`, spec §07 "Linear algebra") whose inner dimensions
//!   are both statically known and unequal — "matrix product inner dimensions
//!   disagree: ...". A defensive second line: `infer`'s `mul_type` already
//!   makes such a call `Type::Failed`, so the determiniser refuses it first.
//! - a matrix product whose lowered operand RANKS disagree with the inferred
//!   types the dispatch classified on — "matrix product operand ranks disagree
//!   with their inferred types: ...". Unreachable unless the two layers drift;
//!   it exists so `lower_matrix_product` cannot index past a short dim list.
//! - a BARE `mul` (surface `*`) whose operands are both non-scalar but are not one
//!   of the four products §07's `mul` row admits — two same-orientation vectors, a
//!   matrix against a transposed vector (either order), a rank-3 pair — "`*` has
//!   no meaning for these operand shapes: ... §07 \"Linear algebra\" gives `mul`
//!   the domain \"scalars, matrix-matrix, matrix-vector, scalar-matrix,
//!   scalar-vector, transposed-vector–vector, vector–transposed-vector\" ...".
//!   `infer`'s `mul_type` types every one of them `%deferred`, so lowering them
//!   elementwise would silently answer a different question.
//!   `ops::classify_bare_mul` decides on orientation-aware shapes; a scalar
//!   operand or an absent type keeps the ordinary elementwise path.
//! - an INNER product (`transpose(a) * b`) whose operand lengths are statically
//!   unequal — "inner product operand lengths disagree: ...". A defensive second
//!   line: `infer`'s `mul_type` already makes such a call `Type::Failed` ("inner
//!   product: vector lengths disagree (spec §07)"), so the determiniser refuses
//!   first and this arm is reachable only from a direct `emit` call.
//! - `transpose`/`adjoint` above rank 2 — "`transpose` has no lowering for ... §07
//!   \"Linear algebra\" gives it the domain \"vectors, matrices\"". A rank-1
//!   operand emits NOTHING (the transposition is type-level only) and a rank-2 one
//!   emits `stablehlo.transpose … dims = [1, 0]`.
//! - `rowstack`/`colstack` (§07 "Array and table operations") whose argument does
//!   not lower to a rank-2 tensor — "`rowstack` has no lowering for ... §07 ...
//!   gives it the domain \"vector of equal-length vectors\" ...". Rank 1 is a
//!   vector of scalars, rank 3 a vector of matrices. Both are test-locked
//!   (`tests/golden_stack.rs`), as is `colstack`'s own wording.
//! - `rowstack`/`colstack` whose argument is a MATRIX (any rank-2-or-higher
//!   `Array`) rather than a vector of vectors — "`rowstack`'s argument is a
//!   rank-2 array, not a vector of vectors ... §03 \"Arrays\" says vectors of
//!   vectors \"are not interpreted as matrices implicitly ...\"".
//!   `ops::require_vector_container`, on the INFERRED type: the two have the same
//!   lowered tensor (`mlir_type_of` flattens the element chain), so the rank-2
//!   check on the `MlirTy` cannot tell them apart and this would lower silently —
//!   an identity for `rowstack`, a transpose for `colstack`. `infer` agrees
//!   independently: `rowstack_type` matches only a rank-1 array of arrays, so
//!   `rowstack(matrix)` is already `%deferred`. Discriminates a nested
//!   `cartpow(cartpow(reals, 3), 2)` ABI input (rank-1 container, ACCEPTED) from
//!   `cartpow(reals, [2, 3])` (rank-2, refused) — same tensor, opposite verdicts.
//! - `rowstack`/`colstack` whose argument MIXES orientations — one element a
//!   rank-1 `Array` (column), another a `TVector` (row) — "`rowstack`'s argument
//!   mixes vector orientations ... §03 \"Arrays\" gives an array a single element
//!   type ...". `ops::require_uniform_orientation`, on the same
//!   inferred-types-only channel as `require_same_orientation`: both elements
//!   lower to the same `tensor<nxf32>`, so `lower_vector`'s identical-`MlirTy`
//!   check accepts the pair. A UNIFORMLY transposed container is accepted (§03's
//!   "the term vector will represent both …" blanket).
//! - a RAGGED `rowstack`/`colstack` argument — refused one level down by
//!   `vector()`'s identical-`MlirTy` check (below), since inference types
//!   `[[1.0, 2.0], [3.0]]` as an array of `%any` rather than reporting it.
//!   Test-locked from `rowstack` in `tests/golden_stack.rs`.
//! - `addaxes` (§07) whose `A` is a TRANSPOSED vector — "addaxes: `A` is a
//!   transposed vector ...". §07's domain column for this entry is "array", not
//!   "vector", so §03's blanket does not widen it, and the widening would change
//!   the answer: a row's tensor form is `[n]`, so a trailing axis would emit the
//!   column `[n, 1]`.
//! - `addaxes` whose `A` is not a statically-shaped array — "addaxes: `A` must be
//!   a statically-shaped array, got ...". Same static-result-shape reason as
//!   `lower_fill`'s; the test-locked case is a scalar `A`, the dynamic-extent case
//!   is defensive.
//! - `addaxes` whose `n_leading`/`n_trailing` is not a non-negative integer
//!   literal — "addaxes: `n_leading` must be a non-negative fixed integer literal
//!   (§07 ...)". ONE wording for both defects: surface `-1` arrives as `neg(1)`, a
//!   call rather than a negative `Lit`, so a separate negative-literal arm would
//!   be unreachable. `addaxes`' ARITY never reaches `args_exact` at all —
//!   inference reports it ("`addaxes` takes 3 arguments (spec §07)"), pinned in
//!   `tests/golden_stack.rs`.
//! - an operand pair of different ORIENTATION — a rank-1 `Array` (column) against
//!   a `TVector` (row) — "operands have different orientation: ... §03 makes a
//!   transposed vector a distinct type from a one-dimensional array ...".
//!   `ops::require_same_orientation`, called from `binary`, `lower_compare` and
//!   `lower_ifelse` (the branch pair). It CANNOT live in `require_broadcastable`:
//!   that sees only `MlirTy`, where both are `tensor<nxf32>`. Deliberately
//!   orientation-only, not the whole of `ArithShape::differs_from` — that also
//!   reports a scalar against an array, which is exactly what the dotted
//!   spellings broadcast and what the determiniser's synthesized
//!   `mul(literal, vector)` relies on.
//! - a BARE `add`/`sub`/`divide`/`pow` (surface `+`, `-`, `/`, `^`) whose operand
//!   shapes are outside the §07 "Operator-equivalent functions" domain for that
//!   head — `add`/`sub` take "scalars or arrays of same shape (real or complex)",
//!   `pow` "scalars (real or complex)", `divide` "scalars, array-scalar,
//!   transposed-vector–scalar (real or complex)" (flatppl-design#77, pending owner
//!   review, superseding the narrower row #75 introduced; §05 "`/` requires a
//!   scalar divisor" is unchanged, so the DIVISOR is the whole discriminator —
//!   `scalar / vector` and `array / array` refuse while an any-rank dividend over a
//!   scalar divisor lowers)
//!   — "`+` has no meaning for these
//!   operand shapes: ... Write `.+` for the elementwise form ...". Without it
//!   `Emitter::broadcast_pair` reconciles the pair, so `scalar + vector` silently
//!   emitted the number `.+` means (accepts-invalid). `ops::lower_bare_arith`
//!   decides, on the same bare-vs-`broadcast`-headed discriminator as `mul`; an
//!   `%any`/`%deferred`/absent operand type keeps the ordinary path, so the guard
//!   refuses only a pair it can prove is out of domain.
//! - the BINARY `min`/`max` (§07 "Elementary functions") over a `Bool` operand
//!   — "min/max: §07 ... gives the binary `min`/`max` the domain `reals`, and
//!   over booleans `stablehlo.minimum`/`maximum` is a conjunction / disjunction
//!   — §07's `land`/`lor`, not its `min`/`max` ...". An `Int` operand is NOT
//!   refused: §03 nests `booleans ⊂ integers ⊂ reals`, and an integer
//!   `stablehlo.minimum` is both legal and what §07's formula means.
//! - `equal`/`unequal` over a `Real` operand — "equal/unequal: §07 \"Comparison
//!   functions\" gives these the domain \"`integers`, `booleans`, strings\" and
//!   states that exact equality \"is restricted to discrete domains to avoid
//!   dependence on numerical precision\" ...". The refusal IS the lowering:
//!   emitting a float `compare EQ` would answer a question §07 declines to
//!   define, so the message names the four discretizing functions §07 offers
//!   (`integer`/`floor`/`ceil`/`round`) plus `iszero`, which §07 defines for a
//!   non-discrete input and which this map still lowers over the same operand.
//!   A STRING operand never reaches here — `types.rs` refuses it first.
//! - `isfinite`/`isinf` over a `Bool` operand — "isfinite/isinf: a boolean
//!   operand has no lowering here — these compose from `stablehlo.abs`, which is
//!   defined over signed integer, float and complex operands but not over
//!   `pred` ...". A helper that does not cover its head's whole §07 domain: §03's
//!   `booleans ⊂ integers ⊂ reals` puts a boolean inside §07's `reals`, but
//!   `Emitter::abs` is kind-polymorphic and would render
//!   `stablehlo.abs %n : tensor<i1>`, which IREE rejects outright. An `Int`
//!   operand is NOT refused — `stablehlo.abs` is defined over signed integers
//!   (executed). `isnan` is NOT refused either, and the asymmetry is deliberate:
//!   it returns before the `abs`, and `compare NE` over `tensor<i1>` is legal and
//!   answers `false`, which is right. The `abs` HEAD has the same hole
//!   (`abs(lt(x, 1.0))` fails identically at base) — pre-existing, carded
//!   separately; widening `Emitter::abs` to convert a `Bool` operand up would
//!   close both corners and retire this guard.
//! - `land`/`lor`/`lxor`/`lnot` whose operand is not a boolean-producing call —
//!   "<head> operand must be a boolean predicate (in/compare/lt/gt/le/ge/land/
//!   lor/lxor/lnot/iszero/equal/unequal/isfinite/isinf/isnan)". Shared with
//!   `ifelse`'s condition check (`ops::require_predicate_head` over
//!   `ops::PREDICATE_HEADS`). Deliberately NOT widened to any `Bool`-typed
//!   VALUE, though one renders `i1` and would emit: the boolean-value gap is
//!   left as ONE documented refusal
//!   (`flatppl-dev/stablehlo-feature-matrix.md`, prioritized gap 6) rather than
//!   half-closed inconsistently across `ifelse` and the connectives.
//! - `land`/`lor`/`lxor` whose operands have different shapes — "<head>:
//!   operands must have the same shape, got ... and ...". `Emitter::and`/`or`/
//!   `xor` render ONE type for both operands and the result.
//! - `diag` with a non-zero offset `k` — "`diag`: only the MAIN diagonal (`k =
//!   0`, §07's default) lowers, got k = N — a §07 super- or sub-diagonal needs a
//!   shifted mask and a shorter result than this lowering produces". PARTIAL
//!   against §07's entry, which defines every `k`; `Emitter::diag` masks on
//!   `row == col`. A non-literal `k` gets its own wording ("the diagonal offset
//!   `k` must be an integer literal").
//! - `diag`/`trace`/`lower_cholesky`/`row_gram`/`col_gram`/`quadform` on a
//!   non-rank-2 operand — "`<head>`: §07 \"Linear algebra\" gives this a matrix
//!   domain, so a rank-2 operand is required, got ..."; and `self_outer` on a
//!   non-rank-1 one — "`self_outer`: §07 ... gives it the domain \"vectors\"
//!   ...". Every one of these `Emitter` matrix helpers PANICS on a bad rank
//!   (`ops::matrix_dims` is what keeps them unreachable).
//! - `diag`/`trace`/`quadform`/`lower_cholesky` on a non-SQUARE matrix —
//!   "`<head>`: §07 \"Linear algebra\" gives this the domain \"square
//!   matrices\", got MxN" (`ops::square_dim`). §07's `diag` domain is the wider
//!   "matrices", but `Emitter::diag` row-sums an `n`-column mask, so on an `m`x`n`
//!   operand with `m > n` it returns `m` entries — zeros for the rows the
//!   diagonal never reaches — instead of §07's `min(m, n)`.
//! - a dynamic (`?`) axis on any of the above — "`<head>`: a dynamic matrix axis
//!   has no lowering, got ...". Not test-locked: `cartpow` over a fixed size
//!   always resolves statically.
//! - `diag`/`trace`/`lower_cholesky` on a non-`Real` matrix — "`<head>`: only a
//!   real matrix is supported, got ... — the underlying lowering emits
//!   float-typed index and identity constants ...". This is the Real-hardcode
//!   family (`ops::require_real_matrix`): `Emitter::diag` renders its
//!   iota/mask matrices and its reduction identity as floats unconditionally,
//!   and `Emitter::cholesky` renders `stablehlo.cholesky` at the operand's own
//!   kind while tagging the result `Real`, so an integer operand emits a
//!   self-contradictory module IREE rejects outright. The head refuses rather
//!   than reaching the helper. `row_gram`/`col_gram`/`quadform`/`self_outer` are
//!   NOT in this family — they run through `Emitter::dot_contract`, which widens
//!   to the operands' common §03 kind, so an integer Gram stays integer.
//!   (The same hardcode is still reachable through `registry.rs`'s
//!   `MvNormal`/`Wishart`/`LKJ` builders, which call `Emitter::cholesky`
//!   directly — carded separately, not closed here.)
//! - `quadform` whose vector length disagrees with `A`'s order — "`quadform`:
//!   `x` must have length N to match the NxN `A`, got ...".
//!   `Emitter::matvec` panics on a contracting-dim mismatch.
//! - an unknown builtin head — "unsupported builtin head '...'"
//! - wrong arity for any arity-checked head (`args_exact`, shared by
//!   `unary`/`binary`/`ifelse`/`get`/`get0`/`in`/`inf`) — "expected N
//!   argument(s), got M". `diag` is the one head with its OWN arity message
//!   ("`diag`: expected 1 or 2 argument(s), got N"), because §07 gives it an
//!   optional second argument that `args_exact`'s fixed `N` cannot express.
//! - `ifelse`'s condition is not a boolean-producing predicate call — "ifelse
//!   condition must be a boolean predicate (in/compare/lt/gt/le/ge/land/lor/
//!   lxor/lnot/iszero/equal/unequal/isfinite/isinf/isnan)". The list is
//!   `ops::PREDICATE_HEADS` verbatim — every head in this map that lowers to an
//!   `i1` — so it grows whenever a boolean head is wired. The doc here said
//!   "(in/compare)" through several waves in which the actual message had
//!   already grown; it is generated from the constant, so quote the constant.
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
//! **`aggregate.rs`** (§04 "Multi-axis aggregation", reached through
//! `ops::lower_builtin`'s `"aggregate"`/`"metricsum"` arms). Every site below is
//! locked by a test in `tests/golden.rs` unless noted:
//! - `f_reduction` is not one of §04's seven order-invariant built-ins —
//!   "aggregate: '...' is not an eligible reduction — §04 ... lists the eligible
//!   built-ins as `sum`, `prod`, `mean`, `var`, `std`, `maximum` and `minimum`"
//!   (and, for a non-`Const` reduction node, "aggregate: f_reduction must be one
//!   of the built-in reductions named bare" — that branch is NOT test-locked;
//!   `infer` types such a call `%deferred` rather than rejecting it)
//! - `output_axes` is not a literal axis list — "aggregate: output_axes must be
//!   a literal axis list `[.i, .k]` (possibly empty)" (not test-locked: the
//!   parser only ever builds the `vector`-of-`%axis` form)
//! - a repeated `output_axes` name — "aggregate: output axis '.i' is repeated —
//!   §04 requires output_axes to be distinct axis names"
//! - an `output_axes` name that indexes nothing in the body — "aggregate: output
//!   axis '.q' does not index anything in the body — §04 requires that 'every
//!   axis name in `output_axes` must occur at least once in `expr`'"
//! - a variance-marked axis (`.mu^` / `.mu_`) in a plain `aggregate` — "...
//!   carries a variance marker, which §05 ... admits only inside `metricsum`"
//!   (two sites, one for the axis list and one for a body index; the body one is
//!   test-locked)
//! - one axis name indexing TWO dimensions of the same operand (a diagonal) —
//!   "aggregate: axis '.i' indexes two dimensions of one operand, which denotes
//!   a diagonal — the aggregation frame gives each axis name one dimension".
//!   `stablehlo.broadcast_in_dim`'s dimension map must be unique, so the frame
//!   model has no form for it
//! - chained axis indexing (`A[.i][.j]`) — "aggregate: chained axis indexing
//!   ... has no frame form — write the axes in one index list"
//! - a body index that is neither an axis name, an integer literal nor `!` —
//!   "aggregate: an index in an aggregation body must be an axis name, an
//!   integer literal, or `!`" (not test-locked)
//! - §07's whole-axis `all` selector in a body — "aggregate: the `all` selector
//!   selects a whole axis ... §04 reduces 'the resulting scalars'"
//! - a partial index list (fewer selectors than the operand's rank) — "aggregate:
//!   N index selector(s) for a rank-M operand — a partial index yields an array
//!   per axis combination, and §04 reduces 'the resulting scalars'"
//! - an axis-indexed operand with no tensor form — "aggregate: an axis-indexed
//!   operand must be an array, got ..." (not test-locked)
//! - a dynamic (`?`) dimension — "aggregate: a dynamic (`?`) dimension has no
//!   aggregation frame — the frame's shape is static text ..." (not test-locked:
//!   `cartpow` over a fixed size always resolves statically at `Level::Shape`)
//! - a literal index out of range — "aggregate: 0-based index N is out of range
//!   for a dimension of length M" (two sites: a below-floor selector before the
//!   operand is lowered, and an over-length one after; neither is test-locked —
//!   `infer` reports the out-of-range index first)
//! - `!` on an axis longer than 1 — "aggregate: `!` indexes a dimension of
//!   length N — §07 ... requires that 'the indexed axis must be of length one'"
//! - axis dimensions of disagreeing length — "aggregate: axis '.i' indexes
//!   dimensions of different lengths, N and M — §04 requires that 'all array
//!   dimensions indexed with the same axis name must have the same length'"
//! - a body whose value is not the frame's shape — "aggregate: the body
//!   evaluates to ..., not to the aggregation frame's ..." (not test-locked: with
//!   every indexed operand broadcast to the FULL frame, an elementwise body is
//!   frame-shaped by construction, so this is the defensive line against a body
//!   op that reshapes)
//! - `mean`/`var`/`std` over an empty axis — "aggregate: '...' over an empty
//!   axis is undefined — §07 divides by the element count, which is zero here"
//!   (not test-locked: a zero-length `cartpow` axis)
//! - `var`/`std` over a single element — "aggregate: '...' over 1 element(s) is
//!   undefined — §07 defines it with the $n-1$ denominator"
//! - `metricsum` — "metricsum has no lowering in this backend: in general ... it
//!   needs a general indefinite matrix inverse, which StableHLO has no op for".
//!   §04's own lowering makes each lower-variance axis an `inv(metric)`
//!   contraction, and §04 requires the metric only to be "square, symmetric, and
//!   invertible"; `stablehlo.cholesky` needs positive-definiteness, so the
//!   missing piece is different machinery from the frame model, not a missing arm
//!   of it. UNCONDITIONAL by design — it declines the construct, not one variance
//!   pattern, so the message says so rather than implying every call needs an
//!   inverse (the all-upper degenerate case does not)
//!
//! **`emitter.rs`** (`Emitter::reduce_trailing_axes`, the aggregate
//! contraction):
//! - a `maximum`/`minimum` over a non-`Real` operand, or a `sum`/`prod` over a
//!   `Bool` one — "aggregate: the ... reduction has no ... identity that means
//!   what §07 defines ...". On `i1`, `stablehlo.multiply` is a conjunction and
//!   `stablehlo.add` a WRAPPING 1-bit add (parity), so neither computes what §07
//!   defines — §03 "Bool" reaches a reduction by PROMOTION ("`sum(mask)` to count
//!   true entries"), i.e. widened first. Defensive: `infer` types an axis-indexed
//!   body `%deferred`, so `aggregate`'s own inferred element kind is `Real` and
//!   `aggregate::reduce` widens the frame to it before contracting — no surface
//!   model reaches this arm today. `Emitter::reduce_axis` also widens rather
//!   than refusing — it converts a `Bool` operand to `Int`, per §03's promotion
//!   and the `integers` type `infer` now gives `sum(bool_array)` — so its own
//!   `ElemKind::Bool => "false"` additive arm is unreachable. The two routes
//!   differ only in the kind they widen to (`Int` here, `Real` on the aggregate
//!   path, whose body types `%deferred`); see `reduce_trailing_axes`'s note.
//!
//! **`emitter.rs`** (`Emitter::lower_broadcast`, the batched-density head):
//! - `broadcast(builtin_logdensityof, K, …)` whose `K` is not
//!   [`crate::registry::is_batch_safe`] — "broadcast over builtin_logdensityof
//!   of '...' is unsupported: its density builder is not rank-agnostic (batched
//!   density is sound only for univariate pure-arithmetic distributions)". A
//!   structural builder (matrix / gather / `support`) driven by batched inputs
//!   would emit shape-inconsistent StableHLO, so the allow-list is default-deny.
//!   This one gate covers every construct that reaches the batched-density path:
//!   an `iid` fan-out, a value `broadcast`, and §06's `ksuperpose` mixture, all
//!   of which the determiniser lowers through the same
//!   `broadcast(builtin_logdensityof, …)` shape. Locked by
//!   `tests/golden_ksuperpose.rs`'s
//!   `a_non_batch_safe_component_refuses_at_the_existing_gate`.
//! - `broadcast(builtin_logdensityof, …)` whose distribution argument is not a
//!   bare `Const` constructor — "broadcast(builtin_logdensityof, …):
//!   distribution must be a bare constructor"
//! - a `broadcast` callable that is neither a bare builtin name nor a reified
//!   `functionof` — "broadcast: callable must be a bare builtin name or a
//!   reified function"
//! - `broadcast` with no callable at all — "broadcast: missing callable"
//!
//! **`norms.rs`** (§07 "Reductions" and "Norms and normalization", reached
//! through `ops::lower_builtin`'s twelve bare-head arms). Every site below is
//! locked by a test in `tests/golden_norms.rs` unless noted otherwise:
//! - `prod`/`mean`/`var`/`std` over a TABLE — the shared
//!   `table_reduction_refusal`, also raised for `sum` from `ops.rs`. §07 "Table
//!   reductions" makes each a RECORD of per-column reductions, and this emitter
//!   has no record value. Checked BEFORE the argument is lowered, so the
//!   argument's own refusal cannot blame the wrong construct.
//! - `prod`/`mean`/`var`/`std` over a SCALAR or a dynamically-shaped operand —
//!   "...: §07 reduces an ARRAY, so the operand must be a statically-shaped
//!   array, got ...". Matches `ops::lower_extremum`'s guard on
//!   `maximum`/`minimum`; `ops::lower_sum` is the odd one out (it returns a
//!   scalar operand unchanged through `Emitter::reduce_full`'s zero-iteration
//!   path), which is pre-existing and untouched.
//! - `mean`/`var`/`std` over an EMPTY array — "...: over an empty array this is
//!   undefined — §07 divides by the element count, which is zero here".
//!   Reachable from surface source: `elementof(cartpow(reals, [0]))` parses,
//!   infers and determinizes cleanly. `prod` is NOT refused there — the empty
//!   product is the multiplicative identity, which the reduce already emits
//!   (executed: `1.0`).
//! - `var`/`std` over a SINGLE element — "...: over 1 element(s) this is
//!   undefined — §07 defines it with the $n-1$ denominator, and §04
//!   \"Relationship to broadcasting\" states that `var` and `std` are undefined
//!   over a single element". `aggregate::reduce` refuses the identical case; the
//!   determiniser passes a length-1 `var(v)` straight through, so the refusal
//!   has to land here. `mean` is NOT refused — it is the identity on a
//!   one-element input, which is exactly the criterion §04 states.
//! - the cumulative pair and every norm over anything but a statically-sized
//!   RANK-1 operand — "...: §07 gives this head the domain \"vectors\", so its
//!   operand must be a statically-sized rank-1 array, got ...". §07 gives
//!   `cumsum`/`cumprod` and all six norm-family heads the domain "vectors"
//!   specifically, unlike the reductions' "arrays", so a matrix has no §07
//!   meaning to lower rather than a meaning this backend declines.
//! - a `Bool` operand reaching the scan's combine table — "...: a boolean
//!   operand must be promoted before the scan". Unreachable: `lower_cumulative`
//!   converts a `Bool` operand to `Int` immediately above, per §03 "Bool"'s
//!   promotion. Kept as a refusal rather than an `unreachable!()` so a future
//!   reordering surfaces instead of panicking.
//!
//! No head in `norms.rs` refuses a non-`Real` operand. §03's
//! `booleans ⊂ integers ⊂ reals` puts an integer or boolean operand inside
//! every §07 domain here, and each head WIDENS to the kind `infer` types the
//! call as (`prod` keeps the element kind, `Int` for a boolean array; every
//! moment and every norm is `Real`) — so none reaches a helper that hardcodes
//! `Real` over an operand it cannot represent, and none reduces in `i1`.
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
