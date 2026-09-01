//! StableHLO emission for the §06 `ksuperpose` mixture density.
//!
//! The emitter needed NO new code for this construct: the determiniser lowers an
//! applied `ksuperpose` to `logsumexp(broadcast(add, broadcast(log, w),
//! broadcast(builtin_logdensityof, K, broadcast(record, …), x)))`, every head of
//! which the emitter already lowers — the batched-density path
//! (`Emitter::lower_broadcast`), elementwise `log`/`add`, and the shift-by-max
//! `logsumexp`. These goldens pin the emitted text; the numbers below were
//! EXECUTED and matched against an independent scipy oracle.
//!
//! # Numeric verification (EXECUTED, not asserted here)
//!
//! Route: `iree-base-compiler` + `iree-base-runtime`,
//! `compile_str(src, target_backends=["llvm-cpu"], input_type="stablehlo")` run
//! under `local-task` in a python-mcp scratch env — the route
//! `.superpowers/sdd/2026-08-05-joint-constructs-the-joint/wave-hloagg-report.md`
//! documents. Oracle:
//! `logsumexp(log(w) + norm.logpdf(x, mu, sigma)) − log(w.sum())`, computed
//! before any engine output was read.
//!
//! The `gold` column says where the executed module came from. `Y` means the
//! frozen file in `tests/goldens/` was read off disk and compiled, so for those
//! three rows the pinned text and the verified number cannot diverge. The other
//! seven were emitted from the source shown below into a scratch directory and
//! compiled from there — they exercise the same lowering but pin no file, so a
//! future emitter change could move them without reddening a golden.
//!
//! `w = [0.3, 1.2]` (unnormalized, so `Z = 1.5` and `log Z = 0.405465108`),
//! `mus = [-1.0, 2.0]`, `sigmas = [1.0, 0.5]`.
//!
//! | case | x | gold | IREE f32 | scipy oracle | abs diff |
//! |---|---|---|---|---|---|
//! | normal mixture, normalized | 0.5 | Y | −3.411415339 | −3.411415108 | 2.3e−07 |
//! | normal mixture, normalized | −1.0 | — | −2.528376579 | −2.528376324 | 2.6e−07 |
//! | normal mixture, normalized | 2.0 | — | −0.447547168 | −0.447547243 | 7.5e−08 |
//! | normal mixture, normalized | 5.0 | — | −18.331153870 | −18.331151868 | 2.0e−06 |
//! | normal mixture, UNnormalized | 0.5 | — | −3.005950212 | −3.005949999 | 2.1e−07 |
//! | shared scalar sigma | 0.5 | Y | −2.043938637 | −2.043938533 | 1.0e−07 |
//! | singular (size-one) mu | 0.5 | — | −1.380489111 | −1.380489098 | 1.3e−08 |
//! | Dirac superposition | 1.5 | Y | −0.223143533 | −0.223143551 | 1.8e−08 |
//! | Dirac superposition | 0.0 | — | −1.609437943 | −1.609437912 | 3.0e−08 |
//! | one zero weight (drops out) | 0.5 | — | −4.543469906 | −4.543469796 | 1.1e−07 |
//!
//! Every difference is f32 roundoff (relative error ≤ 1.1e−07; the 2.0e−06 entry
//! is on a value of magnitude 18).
//!
//! The normalized and UNnormalized rows at x = 0.5 differ by exactly
//! `log Z = log 1.5 = 0.405465108`, which is the arithmetic check that catches a
//! transcribed pair swapped between them — an earlier revision of this table
//! carried the unnormalized numbers on both rows.
//!
//! # Known deviation from §06: ALL-zero weights give NaN, not −∞
//!
//! §06: "When $\sum_i w_i = 0$, every weight being zero, the result is the zero
//! measure: its density is $0$ and its log-density $-\infty$ everywhere". The
//! emitted module returns **NaN** instead. The cause is the shared
//! shift-by-max `logsumexp` (`ops::lower_logsumexp`): with every term `−∞`, the
//! max is `−∞` and `term − max` is `−∞ − (−∞) = NaN`.
//!
//! This is PRE-EXISTING and not specific to the lift — the variadic spelling
//! `superpose(weighted(0.0, N₁), weighted(0.0, N₂))` was executed through the
//! same route and also returns NaN. Fixing it belongs to `lower_logsumexp`, whose
//! output every `superpose` golden pins, so it is reported rather than changed
//! here. No test below asserts the NaN: pinning non-conformant output would
//! entrench it.

use flatppl_stablehlo::{Mode, emit};

fn emit_logdensity(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "inference errors: {errors:?}");
    let d = flatppl_determinizer::determinize(&m).expect("must lower, not refuse");
    emit(&d, Mode::LogDensity, &Default::default()).expect("must emit")
}

const NORMAL_MIXTURE_SRC: &str = "\
flatppl_compat = \"0.1\"
w = [0.3, 1.2]
mus = [-1.0, 2.0]
sigmas = [1.0, 0.5]
mix = normalize(ksuperpose(Normal, w)(mu = mus, sigma = sigmas))
lp = logdensityof(mix, 0.5)
outputs = (lp)
";

const SCALAR_SIGMA_SRC: &str = "\
flatppl_compat = \"0.1\"
w = [0.3, 1.2]
mus = [-1.0, 2.0]
mix = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))
lp = logdensityof(mix, 0.5)
outputs = (lp)
";

const DIRAC_SRC: &str = "\
flatppl_compat = \"0.1\"
p = [0.2, 0.8]
labels = [0.0, 1.5]
c = normalize(ksuperpose(Dirac, p)(value = labels))
lp = logdensityof(c, 1.5)
outputs = (lp)
";

/// The two-Normal mixture with unnormalized weights `w = [0.3, 1.2]`, normalized.
/// Executed == scipy oracle at four points (see the module header).
#[test]
fn normal_mixture_matches_frozen_golden() {
    let out = emit_logdensity(NORMAL_MIXTURE_SRC);
    let golden = include_str!("goldens/ksuperpose_normal_mixture_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/ksuperpose_normal_mixture_logdensity.mlir)"
    );
}

/// One shared SCALAR sigma held constant across the components (§06: "Non-collection
/// arguments are held constant"). Executed == scipy oracle at x = 0.5.
#[test]
fn scalar_sigma_mixture_matches_frozen_golden() {
    let out = emit_logdensity(SCALAR_SIGMA_SRC);
    let golden = include_str!("goldens/ksuperpose_scalar_sigma_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/ksuperpose_scalar_sigma_logdensity.mlir)"
    );
}

/// §08's categorical over arbitrary values,
/// `normalize(ksuperpose(Dirac, p)(value = labels))`. Executed == scipy oracle at
/// both support points.
#[test]
fn dirac_superposition_matches_frozen_golden() {
    let out = emit_logdensity(DIRAC_SRC);
    let golden = include_str!("goldens/ksuperpose_dirac_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/ksuperpose_dirac_logdensity.mlir)"
    );
}

/// The mixture is emitted as ONE batched component-density evaluation over the
/// family axis, reduced by the shift-by-max `logsumexp` — never `N` separate
/// logpdf blocks. The batch dimension is the component count.
#[test]
fn the_mixture_emits_one_batched_density_reduced_by_logsumexp() {
    let out = emit_logdensity(NORMAL_MIXTURE_SRC);
    assert!(
        out.contains("tensor<2xf32>"),
        "the component axis is the batch dimension:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum") && out.contains("applies stablehlo.add"),
        "logsumexp is the shift-by-max form: a maximum reduce then an add reduce:\n{out}"
    );
    // Two `log` on rank-1: the log-weights, and `Normal`'s own `log(sigma)`.
    assert_eq!(
        out.matches("stablehlo.log %").count(),
        4,
        "log-weights + log(sigma) on the batch, then logsumexp's log and logZ's:\n{out}"
    );
}

/// §06's mass sentence gives `Z = Σᵢ wᵢ` for a Markov component, so the
/// normalizer is a `sum` reduce over the weight vector followed by a `log` and a
/// `subtract` — no `totalmass`, and the whole thing stays a closed-form scalar.
#[test]
fn the_normalizer_is_a_sum_reduce_over_the_weights() {
    let normalized = emit_logdensity(NORMAL_MIXTURE_SRC);
    let unnormalized = emit_logdensity(
        "flatppl_compat = \"0.1\"\n\
         w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         sigmas = [1.0, 0.5]\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = sigmas)\n\
         lp = logdensityof(mix, 0.5)\n\
         outputs = (lp)\n",
    );
    assert!(
        normalized.len() > unnormalized.len(),
        "the normalized form adds the logZ tail"
    );
    assert!(
        normalized.contains("stablehlo.subtract") && !unnormalized.ends_with("subtract"),
        "logZ is subtracted from the mixture density:\n{normalized}"
    );
    assert_eq!(
        normalized.matches("applies stablehlo.add").count(),
        2,
        "two add-reduces: logsumexp's inner sum, and Z = sum(w):\n{normalized}"
    );
}

/// A SINGULAR (size-one) family argument, which §06 expands by repetition. The
/// emitter reconciles the size-one axis against the component axis, so this needs
/// no determiniser-side tiling. Executed == scipy oracle at x = 0.5.
#[test]
fn a_singular_family_argument_emits_against_the_component_axis() {
    let out = emit_logdensity(
        "flatppl_compat = \"0.1\"\n\
         w = [0.3, 1.2]\n\
         mix = normalize(ksuperpose(Normal, w)(mu = [1.0], sigma = [0.5, 2.0]))\n\
         lp = logdensityof(mix, 0.5)\n\
         outputs = (lp)\n",
    );
    assert!(
        out.contains("tensor<2xf32>"),
        "the size-one mu expands to the 2-component axis:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum"),
        "still a logsumexp:\n{out}"
    );
}

/// A component whose density builder is not rank-agnostic cannot be evaluated
/// batched over the family axis. The existing `is_batch_safe` gate refuses it —
/// refuse-don't-mislower — so the mixture inherits exactly the same restriction
/// `iid` and value-`broadcast` already have.
///
/// `NegativeBinomial2` is the component: its builder uses `get0`/`reshape`, and
/// it carries no dedicated batched builder either. This test used to use
/// `Uniform`, which became batch-safe once its density gained §08's support mask
/// (the mask carries the variate's shape).
#[test]
fn a_non_batch_safe_component_refuses_at_the_existing_gate() {
    let mut m = flatppl_syntax::parse(
        "flatppl_compat = \"0.1\"\n\
         w = [0.3, 1.2]\n\
         mix = ksuperpose(NegativeBinomial2, w)(mu = [1.0, 2.0], psi = 2.0)\n\
         lp = logdensityof(mix, 3)\n\
         outputs = (lp)\n",
    )
    .unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let d = flatppl_determinizer::determinize(&m).expect("determinizes");
    let err = emit(&d, Mode::LogDensity, &Default::default()).expect_err("must refuse");
    assert!(err.msg.contains("not rank-agnostic"), "got: {}", err.msg);
}
