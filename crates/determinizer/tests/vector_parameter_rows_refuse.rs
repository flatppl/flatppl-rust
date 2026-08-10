//! Two determiniser rows synthesized §07-out-of-domain arithmetic on a VECTOR parameter.
//! Both now refuse in the row, and both keep lowering when the parameter is scalar.
//!
//! Neither model is well-formed FlatPPL. §08 "Univariate continuous distributions" gives
//! `Normal` the parameters `mu`, `sigma` with variate domain `reals`, and the
//! vector-parameter spelling is §04 "Broadcasting"'s `broadcast(Normal, means, sigmas)`
//! (`Normal.(means, sigmas)`), which "returns an **array-valued measure**" — a `broadcast`
//! node, not the bare constructor either row matches. §06's `locscale` entry states the
//! same requirement directly: "`shift` and `scale` must be value-compatible with the
//! variate of `m`".
//!
//! `infer` reports no diagnostic for either model, so before these guards the StableHLO
//! emitter's bare-op domain guard was the only gate, and before THAT each model emitted a
//! `func.func @logdensity(...) -> tensor<3xf32>` for a query scoring a SCALAR variate — a
//! rank-1 log-density, which is structurally impossible. The emitter guard stays as the
//! backstop for the unproven cases (a `%deferred` parameter type still reaches it).
use flatppl_determinizer::determinize;

/// Does NOT assert `infer` is clean. The subject is the ROW's guard, and this wave's own
/// follow-up asks inference to start rejecting a non-scalar distribution parameter — so
/// asserting silence in every case would make the whole file fail on an improvement it
/// recommends. The silence is pinned once, deliberately, in
/// [`infer_is_currently_silent_on_a_vector_distribution_parameter`].
fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn refusal(src: &str) -> flatppl_determinizer::RefuseError {
    determinize(&parse_infer(src)).expect_err("must refuse rather than mislower")
}

fn pir(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    flatppl_flatpir::write(&out)
}

/// Why these rows have to be the gate: `infer` reports NOTHING for a bare `Normal` whose
/// `sigma` is a vector, and it even types the synthesized `sqrt(add(pow(sv, 2.0), 1.0))` as a
/// scalar despite the `%deferred` `pow` under it — so the emitted marginal claims a scalar
/// `sigma` and no structural gate disagrees.
///
/// **Invert or delete this test when inference learns to reject a non-scalar distribution
/// parameter** (this wave's follow-up). Failing here should point at that work, not look
/// like an unrelated break.
#[test]
fn infer_is_currently_silent_on_a_vector_distribution_parameter() {
    let mut m = flatppl_syntax::parse(
        "\
sv = elementof(cartpow(posreals, 3))
a = draw(Normal(mu = 0.0, sigma = sv))
y = draw(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(record(y = y)), record(y = 1.0))",
    )
    .unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags.is_empty(),
        "if inference now rejects a vector distribution parameter, the row guard is no longer \
         the only gate — update this test and the TODO entry: {diags:?}"
    );
}

// ---------------------------------------------------------------------------
// The Normal–Normal conjugate row (`marginal.rs`) over a vector prior `sigma`
// ---------------------------------------------------------------------------

/// The row's parameter map is `Normal(mu = μ₀, sigma = sqrt(add(pow(σ₀, 2), pow(σ, 2))))`
/// (`marginal.md`, Row 1). §07 "Operator-equivalent functions" gives `pow` the domain
/// "scalars", so a vector `σ₀` puts `pow(σ₀, 2)` outside it — and the emitted marginal
/// would be an n-vector-parameter measure scored at the SCALAR variate the match required.
#[test]
fn a_vector_prior_sigma_refuses_in_the_conjugate_row() {
    let err = refusal(
        "\
sv = elementof(cartpow(posreals, 3))
a = draw(Normal(mu = 0.0, sigma = sv))
y = draw(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(record(y = y)), record(y = 1.0))",
    );
    assert!(
        err.construct.contains("kchain"),
        "the refusal is the marginal's: {err:?}"
    );
    assert!(
        err.reason
            .contains("conjugate pair matched but the prior's `sigma` is a vector"),
        "the refusal names the offending parameter and its kind: {err:?}"
    );
    // The message must point at the spelling that WOULD be legal, so the reason is
    // actionable rather than just a rejection.
    assert!(
        err.reason.contains("broadcast(Dist, …)"),
        "the refusal names the array-valued-measure spelling: {err:?}"
    );
}

/// A vector on the LIKELIHOOD side refuses the same way — the guard runs over both
/// parameter lists, not just the prior's.
#[test]
fn a_vector_likelihood_sigma_refuses_in_the_conjugate_row() {
    let err = refusal(
        "\
sv = elementof(cartpow(posreals, 3))
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = a, sigma = sv))
lp = logdensityof(lawof(record(y = y)), record(y = 1.0))",
    );
    assert!(
        err.reason
            .contains("conjugate pair matched but the likelihood's `sigma` is a vector"),
        "the refusal attributes the vector to the likelihood: {err:?}"
    );
}

/// Row 5 (InverseGamma prior on the variance → scaled Student t) is the third synthesized
/// site the 2026-08-06 bare-op sweep found: `build_scaled_t_logpdf` builds
/// `divide(beta, shape)` from the prior's own parameters, which with VECTOR parameters is an
/// array-over-array `divide` — outside §07's "scalars, array-scalar,
/// transposed-vector–scalar (real or complex)" domain (flatppl-design#77, pending owner
/// review) because the DIVISOR is an array. #77 widened the dividend to any rank, so this is
/// the one of the three sites where only the divisor half still refuses.
///
/// That site was benign only because `crates/stablehlo` cannot lower `loggamma`, so the whole
/// family refused before reaching it — a trap primed to become a live emitter refusal the
/// moment `loggamma` lowers. Checking the parameters in `build_conjugate_marginal` rather
/// than in each builder defuses it: the row refuses BEFORE `build_scaled_t_logpdf` runs, so
/// adding `loggamma` no longer has to fix this row too.
#[test]
fn a_vector_prior_parameter_refuses_before_the_scaled_t_builder_runs() {
    let err = refusal(
        "\
sh = elementof(cartpow(posreals, 3))
sc = elementof(cartpow(posreals, 3))
v = draw(InverseGamma(shape = sh, scale = sc))
y = draw(Normal(mu = 0.0, sigma = sqrt(v)))
lp = logdensityof(lawof(record(y = y)), record(y = 0.5))",
    );
    assert!(
        err.reason
            .contains("conjugate pair matched but the prior's `shape` is a vector"),
        "Row 5 refuses in the shared parameter check, not in its log-density builder: {err:?}"
    );
}

/// Row 2 (Gamma–Poisson) is the row that proves the check is on the row CONTRACT and not on
/// any builder's arithmetic: its parameter map is the IDENTITY — `alpha`/`beta` are the
/// prior's `shape`/`rate` reused unchanged, no arithmetic at all (`marginal.md`, Row 2). A
/// purely arithmetic reading would leave it admitting a vector `shape` and emitting a
/// `NegativeBinomial` whose log-density at the scalar variate is not a scalar.
#[test]
fn a_vector_prior_parameter_refuses_even_where_the_row_does_no_arithmetic() {
    let err = refusal(
        "\
s = elementof(cartpow(posreals, 3))
r = draw(Gamma(shape = s, rate = 2.0))
k = draw(Poisson(rate = r))
lp = logdensityof(lawof(record(k = k)), record(k = 3))",
    );
    assert!(
        err.reason
            .contains("conjugate pair matched but the prior's `shape` is a vector"),
        "the identity-map row refuses too: {err:?}"
    );
}

/// Row 5's SCALAR control still lowers, so the guard did not simply disable the row.
#[test]
fn row_five_still_lowers_with_scalar_parameters() {
    let text = pir("\
v = draw(InverseGamma(shape = 3.0, scale = 2.0))
y = draw(Normal(mu = 0.0, sigma = sqrt(v)))
lp = logdensityof(lawof(record(y = y)), record(y = 0.5))");
    assert!(
        text.contains("loggamma"),
        "the scaled-t log-density form still lowers:\n{text}"
    );
}

/// The guard must not over-reach on the row it exists to protect. Row 1's own shape with
/// SCALAR sigmas still lowers to the closed-form marginal, and `sqrt(2² + 1²) = sqrt(5)`
/// const-folds — the literal that pins the variance sum (`marginal.md`, Row 1: a variance
/// DIFFERENCE folds to `1.7320508075688772` and either term alone to `2.0`/`1.0`).
#[test]
fn scalar_sigmas_still_lower_to_the_closed_form_marginal() {
    let text = pir("\
a = draw(Normal(mu = 0.0, sigma = 2.0))
y = draw(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(record(y = y)), record(y = 1.0))");
    assert!(
        text.contains("(%field sigma 2.23606797749979)"),
        "marginal sigma is sqrt(2² + 1²) = sqrt(5):\n{text}"
    );
}

/// A parameter whose type inference leaves UNRESOLVED keeps the path it had: the guard
/// refuses only what the inferred type proves is non-scalar, and a scalar `elementof(reals)`
/// prior sigma is proven scalar, so the row still fires. (Pins the fail-open direction —
/// requiring a CONFIRMED scalar would reject the literal-parameter rows above, whose
/// literal nodes carry no inferred type at all.)
#[test]
fn a_symbolic_scalar_sigma_still_lowers() {
    let text = pir("\
s0 = elementof(posreals)
a = draw(Normal(mu = 0.0, sigma = s0))
y = draw(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(record(y = y)), record(y = 1.0))");
    assert!(
        text.contains("builtin_logdensityof Normal"),
        "the symbolic-sigma marginal still lowers:\n{text}"
    );
    assert!(
        text.contains("(pow (%ref self s0) 2.0)"),
        "the variance sum reads the symbolic sigma:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// `derive_locscale`'s scalar branch (`invert.rs`) over a vector `scale`/`shift`
// ---------------------------------------------------------------------------

/// The scalar branch emits `f_inv(y) = divide(y − shift, scale)` and
/// `logvol = log(abs(scale))`. A vector `scale` makes the DIVISOR an array — outside §07
/// `divide`'s "scalars, array-scalar, transposed-vector–scalar (real or complex)" domain
/// (flatppl-design#77, pending owner review), every form of which has a scalar divisor — and
/// makes the log-volume a vector where a log-density term must be a scalar. §06 forbids the
/// shape outright: `shift` and `scale` must be value-compatible with the variate of `m`,
/// which is `reals`.
#[test]
fn a_vector_scale_refuses_in_the_locscale_scalar_branch() {
    let err = refusal(
        "\
sc = elementof(cartpow(posreals, 3))
m0 = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, sc)
lp = logdensityof(lawof(record(y = draw(m0))), record(y = 0.5))",
    );
    assert!(
        err.reason
            .contains("locscale over a scalar variate requires a scalar scale")
            && err.reason.contains("a vector"),
        "the refusal names `scale` and its kind: {err:?}"
    );
    assert!(
        err.reason.contains("pushfwd"),
        "the refusal names §06's escape hatch for a general affine map: {err:?}"
    );
}

/// A vector `shift` is the same incompatibility on the other argument: `sub(y, shift)` with
/// a scalar `y` is outside §07 `sub`'s "scalars or arrays of same shape" domain. The old
/// guard checked neither argument for a vector, and `type_is_matrix` never looked at
/// `shift` at all.
#[test]
fn a_vector_shift_refuses_in_the_locscale_scalar_branch() {
    let err = refusal(
        "\
sh = elementof(cartpow(reals, 3))
m0 = locscale(Normal(mu = 0.0, sigma = 1.0), sh, 2.0)
lp = logdensityof(lawof(record(y = draw(m0))), record(y = 0.5))",
    );
    assert!(
        err.reason
            .contains("locscale over a scalar variate requires a scalar shift")
            && err.reason.contains("a vector"),
        "the refusal names `shift` and its kind: {err:?}"
    );
}

/// The scalar-scale case the guard must leave alone: `locscale(Normal(0, 1), 1.0, 2.0)` is
/// §06's own worked equivalence with `Normal(1.0, 2.0)`, and it still lowers to the affine
/// `divide` plus the `log|scale|` volume term.
#[test]
fn a_scalar_scale_still_lowers_in_the_locscale_scalar_branch() {
    let text = pir("\
m0 = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, 2.0)
lp = logdensityof(lawof(record(y = draw(m0))), record(y = 0.5))");
    assert!(
        text.contains("builtin_logdensityof Normal"),
        "the affine pushforward still lowers:\n{text}"
    );
}

/// A MATRIX scale over a scalar variate keeps refusing — the pre-existing `type_is_matrix`
/// case, now reached through the same guard, so the widening did not drop it.
#[test]
fn a_matrix_scale_still_refuses_in_the_locscale_scalar_branch() {
    let err = refusal(
        "\
sc = elementof(cartpow(reals, [3, 3]))
m0 = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, sc)
lp = logdensityof(lawof(record(y = draw(m0))), record(y = 0.5))",
    );
    assert!(
        err.reason
            .contains("locscale over a scalar variate requires a scalar scale")
            && err.reason.contains("a vector"),
        "a matrix is a vector-kind non-scalar to this guard (§03 makes both arrays): {err:?}"
    );
}
