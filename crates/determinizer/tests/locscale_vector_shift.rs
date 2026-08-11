//! `derive_locscale`'s VECTOR branch checks its `shift` against the variate's shape.
//!
//! The branch emits `f_inv(y) = linsolve(scale, y − shift)` over a vector variate. It
//! guarded `scale` (must be a matrix, must be square) but never looked at `shift` at all, so
//! `sub(y, shift)` accepted a scalar, a wrong-length vector, a row vector or a matrix.
//!
//! §07 "Operator-equivalent functions" gives `sub` the domain "scalars or arrays of same
//! shape (real or complex)" — unchanged by flatppl-design#77, which widens only `mul`,
//! `divide` and the table reductions. §06's `locscale` entry states the requirement directly:
//! "`shift` and `scale` must be value-compatible with the variate of `m`; for general
//! matrix-vector affine maps use `pushfwd` directly."
//!
//! This is the MIRROR of the scalar branch's guard, and deliberately a DIFFERENT test. Over a
//! scalar variate anything non-scalar is wrong, which `confirmed_non_scalar` decides. Over a
//! vector variate a vector `shift` is exactly what is right, and only a differing shape is
//! wrong — so it takes a shape-AGREEMENT check, which is why the scalar branch's helper could
//! not simply be reused here.
//!
//! **Prove-it-is-wrong, not fail-closed.** A `Dim::Dynamic` on either side, or an unresolved
//! type, is not a confirmed disagreement and still lowers — pinned by
//! [`a_dynamic_length_shift_is_not_confirmed_and_still_lowers`]. The StableHLO bare-op domain
//! guard remains the backstop for those.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn refusal(src: &str) -> String {
    determinize(&parse_infer(src))
        .expect_err("must refuse rather than mislower")
        .reason
}

fn pir(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    flatppl_flatpir::write(&out)
}

/// A 3×3 covariance and the query, shared by every case. `lower_cholesky(cv)` is §06's own
/// spelling of the matrix-affine `scale`: "`locscale(MvNormal(zeros(n), eye(n)), mu,
/// lower_cholesky(cov))` is equivalent to `MvNormal(mu, cov)`".
fn model(shift_bindings: &str, shift: &str) -> String {
    format!(
        "cv = elementof(cartpow(reals, [3, 3]))
{shift_bindings}m0 = locscale(MvNormal(mu = [0.0, 0.0, 0.0], cov = cv), {shift}, lower_cholesky(cv))
lp = logdensityof(lawof(record(y = draw(m0))), record(y = [0.5, 0.25, 0.125]))"
    )
}

/// The headline case, and the one the branch missed entirely: a SCALAR `shift` against a
/// 3-vector variate emitted `sub(vector(0.5, 0.25, 0.125), 1.0)`, typed `%deferred`, straight
/// into `linsolve`. Decidable without any static dimension — a scalar is outside §07's row
/// whatever the lengths.
#[test]
fn a_scalar_shift_refuses_against_a_vector_variate() {
    let reason = refusal(&model("", "1.0"));
    assert!(
        reason.contains("requires a `shift` of the SAME shape as the variate")
            && reason.contains("this one is a scalar"),
        "the refusal names `shift` and its kind: {reason}"
    );
    // The message must carry the two normative anchors, since neither is obvious from the
    // emitted node alone.
    assert!(
        reason.contains("value-compatible with the variate")
            && reason.contains("arrays of same shape"),
        "the refusal cites §06's requirement and §07's `sub` domain: {reason}"
    );
}

/// A vector `shift` of the WRONG LENGTH — the case that makes this a shape-agreement check
/// rather than the scalar branch's non-scalar check. A length-4 shift is a vector, so
/// `confirmed_non_scalar` would have passed it straight through.
#[test]
fn a_wrong_length_vector_shift_refuses() {
    let reason = refusal(&model("sh = elementof(cartpow(reals, 4))\n", "sh"));
    assert!(
        reason.contains("a vector of length 4, not 3"),
        "the refusal reports both lengths, so the author can see which end is wrong: {reason}"
    );
}

/// A rank-2 `shift`. `sub` needs "arrays of same shape", and a matrix is not the variate's
/// shape whatever its dimensions.
#[test]
fn a_matrix_shift_refuses_against_a_vector_variate() {
    let reason = refusal(&model("sh = elementof(cartpow(reals, [3, 3]))\n", "sh"));
    assert!(
        reason.contains("an array of rank 2, not a vector"),
        "the refusal names the rank: {reason}"
    );
}

/// A TRANSPOSED (row) vector of the matching length. §03 makes a transposed vector "a
/// distinct type from a one-dimensional array", so equal length is not equal shape — the
/// orientation difference is a real one, and the same distinction the StableHLO emitter
/// enforces for elementwise operand pairs.
#[test]
fn a_transposed_vector_shift_refuses_on_orientation() {
    let reason = refusal(&model(
        "sh = elementof(cartpow(reals, 3))\n",
        "transpose(sh)",
    ));
    assert!(
        reason.contains("a transposed (row) vector, not a column vector"),
        "the refusal names the orientation, not a length: {reason}"
    );
}

/// The control the guard must not over-reach on: a symbolic vector `shift` of the MATCHING
/// length still lowers to the matrix-affine emission.
#[test]
fn a_matching_length_vector_shift_still_lowers() {
    let text = pir(&model("sh = elementof(cartpow(reals, 3))\n", "sh"));
    assert!(
        text.contains("linsolve") && text.contains("logabsdet"),
        "the matrix-affine bijection still emits:\n{text}"
    );
    assert!(text.contains("(sub "), "`y − shift` still emits:\n{text}");
}

/// A LITERAL vector shift lowers too — the shape comes from the literal's own inferred type,
/// not from a binding, so this exercises a different route to the same check.
#[test]
fn a_literal_vector_shift_of_matching_length_still_lowers() {
    let text = pir(&model("", "[1.0, 2.0, 3.0]"));
    assert!(
        text.contains("linsolve"),
        "a literal shift of the right length still lowers:\n{text}"
    );
}

/// **Pins the prove-it-is-wrong direction.** `lower_cholesky(cv) * sh` types as
/// `(%array 1 (%dynamic) (%scalar real))` — a vector whose LENGTH inference does not resolve.
/// That is not a confirmed disagreement, so it must keep the path it has today rather than
/// refuse. Fail-closed here would reject a shift that is very likely correct.
#[test]
fn a_dynamic_length_shift_is_not_confirmed_and_still_lowers() {
    let text = pir(&model(
        "sh = elementof(cartpow(reals, 3))\n",
        "lower_cholesky(cv) * sh",
    ));
    assert!(
        text.contains("linsolve"),
        "a dynamic-length shift is not refused:\n{text}"
    );
}

/// The SCALAR branch is untouched by this change — its own guard still refuses a vector
/// `shift`, with its own message. Guards the two branches against being collapsed into one.
#[test]
fn the_scalar_branch_guard_is_unchanged() {
    let reason = refusal(
        "\
sh = elementof(cartpow(reals, 3))
m0 = locscale(Normal(mu = 0.0, sigma = 1.0), sh, 2.0)
lp = logdensityof(lawof(record(y = draw(m0))), record(y = 0.5))",
    );
    assert!(
        reason.contains("locscale over a scalar variate requires a scalar shift"),
        "the scalar branch keeps its own wording: {reason}"
    );
}
