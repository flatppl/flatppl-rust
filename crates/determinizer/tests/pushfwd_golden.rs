//! Determiniser synthesis of (f_inv, logvol) for known/invertible forward
//! functions in pushfwd density lowering (§06 case 1 + 3-bounded). Structural
//! only: assert the emitted change-of-variables FlatPIR, cross-checked against
//! the explicit-bijection form.
use flatppl_determinizer::determinize;
fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}
fn pir(src: &str) -> String {
    flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"))
}

#[test]
fn pushfwd_bare_exp_lowers_like_explicit_bijection() {
    // Canonical LogNormal (§06 line 382). Must equal the explicit exp_bijection form.
    let synth = pir("ln = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(ln, 0.5)");
    // Inline `bijection(...)` rather than a named `b = bijection(...)` binding:
    // the latter survives determinization as a dead `b = 0.0` binding, which
    // would break byte-equality for a reason unrelated to the change-of-variables.
    // The inline form is semantically identical and keeps binding structure equal,
    // so `assert_eq!` verifies the *whole* synthesized change-of-variables (incl.
    // the forward log-volume convention: exp ⇒ f_inv = log, logvol = identity).
    let explicit = pir(
        "ln = pushfwd(bijection(exp, log, x -> x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(ln, 0.5)",
    );
    assert!(synth.contains("builtin_logdensityof"), "got:\n{synth}");
    assert_eq!(
        synth, explicit,
        "synthesized exp must match explicit bijection(exp, log, id)"
    );
}
#[test]
fn pushfwd_eta_lambda_exp_lowers() {
    let p =
        pir("ln = pushfwd(x -> exp(x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(ln, 0.5)");
    assert!(
        p.contains("builtin_logdensityof") && p.contains("log"),
        "got:\n{p}"
    );
}

#[test]
fn pushfwd_affine_lambda_lowers() {
    // x -> 2*x + 1 : f_inv(y) = (y-1)/2, logvol = log(2) (constant).
    let p = pir(
        "d = pushfwd(x -> 2.0 * x + 1.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(p.contains("builtin_logdensityof"), "got:\n{p}");
    // f_inv = (y-1)/2, applied at the literal query point y = 0.5, is now
    // beta-reduced AND const-folded to the literal -0.25 (Buffy #263 Pass 2
    // inlines the residual `%call` that used to carry `divide(sub(_x_, 1.0), 2.0)`
    // unapplied; const-fold then reduces the folded arithmetic to one literal):
    assert!(
        p.contains("(builtin_logdensityof Normal") && p.contains(") -0.25)"),
        "f_inv(0.5) = (0.5 - 1)/2 = -0.25, inlined + folded:\n{p}"
    );
    // logvol is the constant log|2| = log(abs(2)) — unaffected by inlining:
    assert!(p.contains("(abs 2.0)"), "logvol log(2) present:\n{p}");
}

#[test]
fn pushfwd_composition_exp_affine_lowers() {
    // x -> exp(2*x) : chain. f_inv(y) = log(y)/2 ; logvol = log(2) + 2x  (chain rule).
    let p = pir(
        "d = pushfwd(x -> exp(2.0 * x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        p.contains("builtin_logdensityof") && p.contains("log"),
        "got:\n{p}"
    );
    // Composed inverse log(y)/2, applied at the literal query point y = 0.5,
    // is beta-reduced (Buffy #263 Pass 2) to divide(log(0.5), 2.0) — `log` is
    // excluded from const-fold (see `canon/fold.rs`), so the `log(0.5)` leaf
    // stays unevaluated and the `divide` around it does too (const-fold
    // requires BOTH operands literal):
    assert!(
        p.contains("(divide") && p.contains("(log 0.5)"),
        "inverse log(0.5)/2 present:\n{p}"
    );
    // Chain-rule logvol: the exp term contributes the partial-forward 2x,
    // evaluated at x = f_inv(0.5) = log(0.5)/2 (the SAME inlined inverse
    // expression, consistently substituted — `mul(2.0, divide(log(0.5), 2.0))`);
    // the affine term contributes log|2| = log(abs(2)).
    assert!(
        p.contains("(mul 2.0 (%meta ((%scalar real) %fixed reals) (divide")
            && p.contains("(abs 2.0)"),
        "chain-rule logvol (2*f_inv(0.5) + log 2) present:\n{p}"
    );
}

#[test]
fn pushfwd_noninvertible_lambda_refuses() {
    // x -> x*x is NOT injective on reals → refuse (recognized op, non-invertible here).
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> x * x, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("non-injective must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_zero_scale_affine_refuses() {
    // x -> 0.0*x + 1.0 collapses to the constant 1.0 — not injective, so the
    // literal-zero "scale" must refuse rather than synthesize a degenerate
    // f_inv = divide(acc, 0.0) / logvol = log(abs(0.0)) = -inf.
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> 0.0 * x + 1.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("literal-zero mul scale must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_zero_divisor_affine_refuses() {
    // x -> x / 0.0 is undefined everywhere — the literal-zero divisor must
    // refuse rather than synthesize f_inv = mul(acc, 0.0) / logvol = -inf.
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> x / 0.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("literal-zero divide denominator must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_log_over_unrestricted_domain_refuses() {
    // `ln x` is undefined for x <= 0; Normal(mu=0, sigma=1)'s support is all of
    // ℝ, not positive. Lowering `pushfwd(log, Normal)` would synthesize a
    // well-formed-looking `(f_inv=exp, logvol=neg(log(x)))` change-of-variables
    // that is only valid on the positive half of the base support — silently a
    // SUB-probability measure (integrates to ~0.5, not 1). Refuse rather than
    // mislower (mirrors derive_pow's is_positive_domain guard; §06 log defined
    // on positive reals).
    let e = determinize(&parse_infer(
        "d = pushfwd(log, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("pushfwd(log, Normal) over a non-positive-support base must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_log_chain_over_unrestricted_domain_refuses() {
    // Same silent-sub-probability danger as the bare-log case, but with `log`
    // appearing inside a scalar-chain forward body (`2.0*log(x)`) rather than
    // as the bare builtin. The chain-walk guard is conservative: it refuses
    // ANY chain containing `log` unless the base domain is provably positive,
    // regardless of where in the chain `log` sits.
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> 2.0 * log(x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("a chain containing log over a non-positive-support base must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_pow_in_composition_refuses() {
    // pow nested inside a composition (not the single top-level op) has an
    // unverifiable input domain — the chain walk unconditionally refuses it.
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> exp(pow(x, 2.0)), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("pow inside a composition must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_divide_chain_lowers() {
    // x -> x/2 : f_inv(y) = mul(y, 2.0) ; logvol = neg(log(abs(2.0))) = -log 2.
    let p =
        pir("d = pushfwd(x -> x / 2.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)");
    assert!(p.contains("builtin_logdensityof"), "got:\n{p}");
    // Inverse mul(y, 2.0), applied at the literal query point y = 0.5, is now
    // beta-reduced AND const-folded to the literal 1.0 (Buffy #263 Pass 2
    // inlines the residual `%call` that used to carry `mul(_x_, 2.0)` unapplied;
    // both operands are then literal reals, so const-fold reduces it further):
    assert!(
        p.contains("(builtin_logdensityof Normal") && p.contains(") 1.0)"),
        "f_inv(0.5) = 0.5 * 2 = 1.0, inlined + folded:\n{p}"
    );
    // logvol is neg(log(abs(2.0))) — the DivByLit sign (negative contribution),
    // a constant unaffected by inlining; leaf substrings (the printer wedges
    // `%meta` type annotations between ops):
    assert!(
        p.contains("(neg") && p.contains("(abs 2.0)"),
        "logvol -log(2) present:\n{p}"
    );
}

#[test]
fn pushfwd_matrix_affine_lowers() {
    // MvNormal construction (§06 case 1, §08 MvNormal `mu + lower_cholesky(cov) * _`):
    // the forward map `mu + L * x` over a 2-vector standard normal
    // `iid(Normal(0,1), 2)` is a matrix-vector affine bijection.
    //   f_inv(y) = linsolve(L, y - mu)   (solve L x = y - mu)
    //   logvol   = logabsdet(L)          (CONSTANT: a linear map's Jacobian is L)
    // Cross-check (Σ = L Lᵀ): logdensityof(iid N(0,1), f_inv(v)) - logabsdet(L)
    // = -n/2·log 2π - ½‖L⁻¹(v-mu)‖² - log|det L|
    // = -n/2·log 2π - ½(v-mu)ᵀΣ⁻¹(v-mu) - ½·log|det Σ|  ≡  log N(v; mu, Σ).
    let p = pir("mu = [0.0, 0.0]\n\
         L = [[1.0, 0.0], [0.0, 1.0]]\n\
         d = pushfwd(x -> mu + L * x, iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])");
    assert!(p.contains("builtin_logdensityof"), "got:\n{p}");
    // f_inv(y) = linsolve(L, y - mu): the preimage solve (with its y - mu RHS).
    assert!(
        p.contains("(linsolve") && p.contains("(sub"),
        "f_inv = linsolve(L, y - mu) present:\n{p}"
    );
    // logvol = logabsdet(L): the CONSTANT forward log-volume.
    assert!(
        p.contains("(logabsdet"),
        "logvol = logabsdet(L) present:\n{p}"
    );
}

#[test]
fn pushfwd_matrix_affine_nonidentity_logdet() {
    // pushfwd_matrix_affine_lowers uses an IDENTITY L, so log|det L| = 0 — that
    // can't distinguish a correct log-det from a wrong one (added instead of
    // subtracted, or L vs L^-1 would ALSO give 0). A non-identity L with a
    // nonzero mu pins a numerically-meaningful log-det. Byte-equal against the
    // explicit bijection(f, f_inv, logvol) form (cleaner than substring-matching
    // through the printer's wedged `%meta` annotations, and pins the WHOLE
    // synthesized change-of-variables, not just that `logabsdet` appears
    // somewhere).
    let synth = pir("mu = [1.0, 2.0]\n\
         L = [[2.0, 0.0], [0.0, 3.0]]\n\
         d = pushfwd(x -> mu + L * x, iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])");
    let explicit = pir("mu = [1.0, 2.0]\n\
         L = [[2.0, 0.0], [0.0, 3.0]]\n\
         d = pushfwd(bijection(x -> mu + L * x, x -> linsolve(L, x - mu), x -> logabsdet(L)), iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])");
    assert!(synth.contains("builtin_logdensityof"), "got:\n{synth}");
    assert_eq!(
        synth, explicit,
        "synthesized non-identity matrix-affine bijection must match the explicit \
         bijection(f, x -> linsolve(L, x - mu), x -> logabsdet(L)) form"
    );
}

#[test]
fn pushfwd_matrix_affine_broadcast_add_lowers() {
    // pushfwd_matrix_affine_lowers exercises the plain `add` outer form
    // (`mu + L * x`); this pins the OTHER pinned outer form, the dotted/
    // broadcast `mu .+ L * x` (`broadcast(Const("add"), mu, mul(L, x))`),
    // recognised by affine_add_operands's `broadcast` arm. Byte-equal against
    // the plain-`add` version with identical mu/L: both outer forms must
    // synthesize the exact same change-of-variables.
    let broadcast = pir("mu = [0.0, 0.0]\n\
         L = [[1.0, 0.0], [0.0, 1.0]]\n\
         d = pushfwd(x -> mu .+ L * x, iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])");
    let plain = pir("mu = [0.0, 0.0]\n\
         L = [[1.0, 0.0], [0.0, 1.0]]\n\
         d = pushfwd(x -> mu + L * x, iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])");
    assert!(
        broadcast.contains("builtin_logdensityof"),
        "got:\n{broadcast}"
    );
    assert_eq!(
        broadcast, plain,
        "broadcast (.+) and plain (+) matrix-affine outer forms must lower identically"
    );
}

#[test]
fn pushfwd_scalar_scale_over_vector_refuses() {
    // A SCALAR affine map (2.0 * x, no matrix) over a VECTOR variate: its true
    // log-volume is n*log|2| (summed over all n axes), not the scalar chain's
    // single log|2| — the module doc calls this exact danger out (§ vector-
    // variate guard). Must refuse rather than fall through to the scalar-chain
    // path and silently emit the wrong (too-small) log-volume.
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> 2.0 * x, iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(d, [0.1, 0.2, 0.3])",
    ))
    .expect_err("scalar affine over a vector variate must refuse, not scalar-chain-lower");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_coupled_nonlinear_multivariate_refuses() {
    // x -> exp(x) + L * x is a COUPLED NONLINEAR multivariate map: the additive
    // term exp(x) depends on the input, so the forward Jacobian is not the
    // constant L and logabsdet(L) would be the wrong log-volume. Refuse rather
    // than mislower (the shift `mu` must not reference the input placeholder).
    let e = determinize(&parse_infer(
        "L = [[1.0, 0.0], [0.0, 1.0]]\n\
         d = pushfwd(x -> exp(x) + L * x, iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])",
    ))
    .expect_err("coupled nonlinear multivariate map must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_three_op_interior_exp_lowers() {
    // x -> 2*exp(x) + 1 : f = 2e^x+1, f' = 2e^x, log|f'| = log 2 + x. The exp op
    // is INTERIOR (not outermost) — locks that its local logvol is evaluated at
    // its own partial-forward point (x), not a shallow/wrong depth.
    let p = pir(
        "d = pushfwd(x -> 2.0 * exp(x) + 1.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        p.contains("builtin_logdensityof") && p.contains("log"),
        "got:\n{p}"
    );
    // Chain-rule logvol: exp's partial-forward point is the bare placeholder x
    // (no other ops sit between exp and the input here), contributing the term
    // `x`; the mul-by-2 contributes log|2| = log(abs(2)):
    assert!(p.contains("(abs 2.0)"), "logvol log(2) term present:\n{p}");
}

#[test]
fn pushfwd_elementwise_exp_lowers() {
    // broadcast(exp, _) over a 3-vector → elementwise LogNormal, DIAGONAL Jacobian.
    // (The `_` hole is only legal inside `fn(…)`, so the forward is spelled
    // `fn(broadcast(exp, _))` — a one-input lambda `arg1 -> broadcast(exp, arg1)`.)
    // f_inv(y) = broadcast(log, y); logvol(x) = sum(broadcast(<id>, x)) = Σ xᵢ, so
    // the pushfwd emits  logdensityof(iid N(0,1), broadcast(log, v))
    //   − sum(broadcast(<id>, broadcast(log, v)))  = Σᵢ [logN(0,1)(log vᵢ) − log vᵢ]
    // — n independent LogNormals (per-cell change-of-variables, log-det summed).
    let src = "d = pushfwd(fn(broadcast(exp, _)), iid(Normal(mu = 0.0, sigma = 1.0), 3))\nlp = logdensityof(d, [0.5, 0.6, 0.7])";
    let out = determinize(&parse_infer(src)).expect("must lower");
    let p = flatppl_flatpir::write(&out);
    // This is the capture-fix golden: the nested identity map's own `_x_`
    // (asserted below) only survives un-substituted if `is_flatpdl` — one of
    // the pass's four hard invariants — actually holds on the result.
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{p}"
    );
    // Before Buffy #263 Pass 2, the f_inv- and logvol-applying `%call`s were
    // left un-reduced, so `sum`'s argument type inferred `%deferred` (a
    // residual user-call, not yet a concrete array shape). Pass 2 inlines
    // both calls, so `sum`'s argument (the per-cell `broadcast`) now infers
    // concretely as the 3-array `cartpow reals 3` it actually is.
    assert!(
        p.contains("builtin_logdensityof")
            && p.contains(
                "(sum (%meta ((%array 1 (3) (%scalar real)) %fixed (cartpow reals 3)) (broadcast"
            ),
        "got:\n{p}"
    );
    // The per-cell inverse is broadcast(log, y): the inner iid density is scored at
    // it, so `broadcast log` must appear.
    assert!(
        p.contains("(broadcast log"),
        "f_inv = broadcast(log, y) present:\n{p}"
    );
    // The per-cell logvol identity map `x -> x` is a NESTED reification
    // (`functionof(_x_ -> _x_)`, boundary name `_x_`) sitting INSIDE the body
    // of the OUTER logvol reification (also boundary name `_x_`, per the
    // synthesizer's uniform placeholder naming — see `kernel::shadows_name`).
    // Pass 2 must inline the outer call WITHOUT capturing the inner one's own
    // `_x_`: the inner identity's body must survive as the bare placeholder
    // ref, not get rewritten to the outer's substituted value.
    assert!(
        p.contains("(functionof (%ref %local _x_) %specinputs ((x (%ref %local _x_))))"),
        "nested per-cell identity map must keep its OWN _x_ unsubstituted (no capture):\n{p}"
    );
}

#[test]
fn pushfwd_elementwise_byte_equals_explicit() {
    // pushfwd_elementwise_exp_lowers only substring-matches (`(sum ...`,
    // `(broadcast log`); the matrix arm added a non-degenerate byte-equal test
    // (pushfwd_matrix_affine_nonidentity_logdet) specifically because substring
    // matches can miss a sign/shape bug that a full structural pin catches.
    // Byte-equal the synthesized elementwise bijection against the explicit
    // `bijection(f, f_inv, logvol)` form it should be identical to:
    //   f_inv(y)   = broadcast(log, y)          (per-cell scalar inverse)
    //   logvol(x)  = sum(broadcast(x -> x, x))  (diagonal log-det: g_logvol for
    //                                             exp is the identity, per-cell,
    //                                             summed)
    // `f_inv`/`logvol` are spelled `x -> ...` (not `fn(_)`) because the
    // synthesizer always names its emitted lambdas' boundary "x"/"_x_"
    // ([`lambda`] in invert.rs) — spelling the explicit comparison form with
    // `fn(_)` sugar (which names its placeholder differently) would break byte
    // equality for a reason unrelated to the change-of-variables, exactly as
    // pushfwd_bare_exp_lowers_like_explicit_bijection's doc comment notes for
    // the dead-binding pitfall. The explicit bijection is inlined (not bound to
    // a name) for the same reason.
    let synth = pir(
        "d = pushfwd(fn(broadcast(exp, _)), iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(d, [0.5, 0.6, 0.7])",
    );
    let explicit = pir(
        "d = pushfwd(bijection(fn(broadcast(exp, _)), x -> broadcast(log, x), \
         x -> sum(broadcast(x -> x, x))), iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(d, [0.5, 0.6, 0.7])",
    );
    assert!(synth.contains("builtin_logdensityof"), "got:\n{synth}");
    assert_eq!(
        synth, explicit,
        "synthesized elementwise bijection must match the explicit \
         bijection(f, x -> broadcast(log, x), x -> sum(broadcast(x -> x, x))) form"
    );
}

#[test]
fn pushfwd_elementwise_coupled_refuses() {
    // A COUPLED broadcast mixing TWO variate slots — `broadcast(add, x, x)` (= x .+ x)
    // — is a single-input lambda whose Jacobian is NOT diagonal in the single-variate
    // sense (the input feeds two operand slots). Refuse rather than mislower with a
    // per-cell diagonal log-det.
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> broadcast(add, x, x), iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(d, [0.5, 0.6, 0.7])",
    ))
    .expect_err("coupled 2-slot broadcast must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("refuse"), "got: {e:?}");
    assert!(
        msg.contains("coupled"),
        "expected the specific coupled-broadcast branch reason, got: {e:?}"
    );
}

#[test]
fn pushfwd_matrix_affine_non_square_refuses() {
    // A 2x3 bracket-literal L infers to the NESTED vec-of-vec matrix shape
    // (Array{shape:[2], elem: Array{shape:[3], elem: Real}}) — not the flat
    // rank-2 shape rowstack/colstack/lower_cholesky produce. linsolve/logabsdet
    // need a square matrix; the non-square guard must recognise this nested
    // form too, or a genuine non-square L silently lowers instead of refusing.
    let e = determinize(&parse_infer(
        "mu = [0.0, 0.0]\n\
         L = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n\
         d = pushfwd(x -> mu .+ L * x, iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(d, [0.5, 0.5])",
    ))
    .expect_err("non-square bracket-literal L must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

// ---------------------------------------------------------------------------
// Structural projection (§06 case 2): pushfwd(fn(get(_, [fields])), M) is a
// MARGINALIZATION. For an explicit field-keyed product (keyword `joint` /
// record-of-draws), the marginal density is closed-form: the sum of just the
// SELECTED components' densities at the projected point (the unselected
// components integrate to 1 and drop). Non-product M, or iid/jointchain/relabel
// (index-remapping, out of scope), refuse.
// ---------------------------------------------------------------------------

#[test]
fn projection_over_keyword_joint_marginalizes() {
    // pushfwd(fn(get(_, ["a"])), joint(a = Normal, b = Exponential)) projects to
    // the {a} marginal: the Normal component only. b (Exponential) is
    // marginalized out — it must NOT contribute a density term.
    // Closed form: logdensityof(marginal, record(a = 0.5)) = logdensityof(Normal(0,1), 0.5).
    let p = pir(
        "j = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Exponential(rate = 1.0))\n\
         pr = pushfwd(fn(get(_, [\"a\"])), j)\n\
         lp = logdensityof(pr, record(a = 0.5))",
    );
    // Exactly ONE scored component (the Normal), not two.
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        1,
        "marginal keeps only the selected Normal component:\n{p}"
    );
    // The kept Normal component is present; the marginalized-out Exponential is
    // gone entirely (its dead measure binding is swept to 0.0).
    assert!(
        p.contains("Normal"),
        "Normal (kept) component present:\n{p}"
    );
    assert!(
        !p.contains("Exponential"),
        "Exponential (marginalized-out) component absent:\n{p}"
    );
    // The projected point's field value (0.5) is scored — the marginal is the
    // Normal density at a = 0.5.
    assert!(p.contains("0.5"), "projected point 0.5 scored:\n{p}");
}

#[test]
fn projection_over_keyword_joint_two_fields_marginalizes_middle() {
    // Select {a, c} from a 3-field joint — the middle field b is marginalized
    // out. Two kept components (Normal + Gamma), the dropped one (Exponential)
    // absent.
    let p = pir("j = joint(a = Normal(mu = 0.0, sigma = 1.0), \
                           b = Exponential(rate = 1.0), \
                           c = Gamma(shape = 2.0, rate = 1.0))\n\
         pr = pushfwd(fn(get(_, [\"a\", \"c\"])), j)\n\
         lp = logdensityof(pr, record(a = 0.5, c = 0.7))");
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        2,
        "marginal keeps the two selected components (a, c):\n{p}"
    );
    assert!(
        !p.contains("Exponential"),
        "middle field b (Exponential) marginalized out:\n{p}"
    );
}

#[test]
fn projection_over_nonproduct_refuses() {
    // A projection over a NON-product measure (a bare Normal) has no explicit
    // product structure, so the marginal is not closed-form here — refuse
    // (§06 case 2 permits "numerically or static error"; we refuse).
    let e = determinize(&parse_infer(
        "pr = pushfwd(fn(get(_, [\"a\"])), Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(pr, record(a = 0.5))",
    ))
    .expect_err("projection over a non-product measure must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn projection_over_iid_refuses() {
    // A projection over `iid` (positional / index-keyed product) needs index
    // remapping — out of scope for the field-keyed marginal here. Refuse with a
    // clear reason rather than mislower (noted follow-up).
    let e = determinize(&parse_infer(
        "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         pr = pushfwd(fn(get(_, [\"a\"])), m)\n\
         lp = logdensityof(pr, record(a = 0.5))",
    ))
    .expect_err("projection over iid must refuse (scoped to field-keyed products)");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn projection_drops_unnormalized_component_refuses() {
    // §06 case 2's closed-form marginal ("unselected components integrate to 1
    // and drop") holds ONLY when each DROPPED component is a NORMALIZED
    // probability measure. Here the dropped field `b` is
    // `weighted(2.0, Exponential(rate=1.0))`, whose total mass is 2 (Mass::Finite,
    // not Mass::Normalized) — the true {a} marginal is
    // `logdensityof(Normal, a) + log(2.0)`, but dropping `b` unchecked would
    // silently omit the `+log(2.0)` factor. Refuse rather than mislower.
    let e = determinize(&parse_infer(
        "j = joint(a = Normal(mu = 0.0, sigma = 1.0), \
                   b = weighted(2.0, Exponential(rate = 1.0)))\n\
         pr = pushfwd(fn(get(_, [\"a\"])), j)\n\
         lp = logdensityof(pr, record(a = 0.5))",
    ))
    .expect_err(
        "projection dropping a non-normalized component must refuse, not silently \
         drop the log-mass",
    );
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn projection_duplicate_field_refuses() {
    // `get(_, ["a", "a"])` selects the SAME field twice: building a sub-joint
    // with two `%field a` entries would double-count `a`'s density term
    // (`lower_keyword_joint` sums one term per named entry). Refuse rather than
    // silently double-count.
    let e = determinize(&parse_infer(
        "j = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Exponential(rate = 1.0))\n\
         pr = pushfwd(fn(get(_, [\"a\", \"a\"])), j)\n\
         lp = logdensityof(pr, record(a = 0.5))",
    ))
    .expect_err("duplicate selected field must refuse, not double-count");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}
