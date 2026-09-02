//! StableHLO lowering of the §09 `particle-physics` densities the corpora need:
//! `CrystalBall`, `Argus` and `ContinuedPoisson`.
//!
//! Reached exactly like a §08 base constructor — the determiniser emits the BARE
//! member name as the `builtin_logdensityof` kernel tag, so the module
//! qualification is gone before `registry::lookup` runs. §09 leaves each
//! normalizing constant as "a normalizing constant"; the closed forms the
//! builders emit are derived in their doc comments.
//!
//! **Executed, not merely pinned.** The full 30-observation mixture model
//! `corpora/coverage/b_mass_peak` (flatppl-testsuite) was emitted with this
//! branch and run through the suite's Enzyme-JAX executor
//! (`unified/stablehlo_exec.py`), giving at `f = 0.4, 0.2, 0.6, 0.05`:
//!
//! ```text
//! 36.7424201965332  34.45578384399414  34.26357650756836  25.241474151611328
//! ```
//!
//! against that model's frozen f64 scipy oracle
//!
//! ```text
//! 36.74238243088467 34.455733396696225 34.263534136158064 25.24140288970264
//! ```
//!
//! — agreeing to 1.0e-6 … 2.8e-6 relative, which is f32 precision through a
//! 30-term `logsumexp` chain (`Dtype::F32`, as every emitted module here is).
//! The two closed forms were separately checked against
//! `scipy.stats.crystalball.logpdf` (0 … 9e-16 absolute) and a direct quadrature
//! of §09's own Argus formula (≤ 9e-10 relative, the quadrature's own residual).

use flatppl_core::Module;

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    // Notes are expected here: §09's Argus support is `interval(0, resonance)`,
    // which the type system cannot express, so the catalogue row carries an
    // honest-degrade note.
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_abi(src: &str) -> Module {
    let mut m = parse_infer(src);
    let syms: Vec<flatppl_core::Symbol> =
        ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse")
}

fn emit(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_abi(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

fn emit_err(src: &str) -> String {
    let err = flatppl_stablehlo::emit(
        &determinize_abi(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse");
    format!("{err:?}")
}

// CrystalBall's density is piecewise, so the emission needs a `compare` + `select`
// pair for the tail/core branch, and its normalizer needs `chlo.erf` for the
// Gaussian core's integral.
#[test]
fn crystal_ball_emits_a_selected_branch_and_an_erf_normalizer() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
x = elementof(reals)
lp = logdensityof(lawof(hep.CrystalBall(m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0)), x)
inputs = (x)
outputs = lp";
    let mlir = emit(src);
    assert!(
        mlir.contains("stablehlo.compare") && mlir.contains("stablehlo.select"),
        "the piecewise tail/core branch is a compare + select:\n{mlir}"
    );
    assert!(
        mlir.contains("chlo.erf"),
        "the Gaussian core's normalizer needs erf:\n{mlir}"
    );
}

// Argus at §09's typical `power = 0.5`: the incomplete-gamma normalizer collapses
// to `erf`, so it emits without an igamma op (which this emitter does not carry).
#[test]
fn argus_at_power_half_emits_an_erf_normalizer() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
x = elementof(reals)
lp = logdensityof(lawof(hep.Argus(resonance = 5.29, slope = -20.0, power = 0.5)), x)
inputs = (x)
outputs = lp";
    let mlir = emit(src);
    assert!(
        mlir.contains("chlo.erf"),
        "the power = 0.5 normalizer is erf-expressible:\n{mlir}"
    );
}

// An `iid` over a §09 member is the BATCHED (dotted) density, not an unroll: the
// determiniser emits one `builtin_logdensityof.(D, broadcast(record, …), vec)`
// under a `sum`. Both builders are rank-agnostic, so they are listed in
// `is_batch_safe`; without that this refused as "not rank-agnostic".
//
// Executed, not merely pinned. Under the suite's Enzyme-JAX executor:
//
//   iid(CrystalBall(5.279, 0.003, 1.5, 3.0), 5) at [5.270, 5.276, 5.279, 5.281, 5.285]
//     -> 18.620861053466797   vs the §09 closed form 18.62083959717085   (1.15e-6 rel)
//   iid(Argus(5.29, -20.0, 0.5), 5)             at [4.80, 5.00, 5.10, 5.20, 5.25]
//     -> 2.836592674255371    vs the §09 closed form 2.8365942512560958  (5.56e-7 rel)
//
// both the f32 band. The CrystalBall oracle equals
// `scipy.stats.crystalball.logpdf(...).sum()` exactly.
#[test]
fn iid_over_a_hep_member_emits_a_batched_density() {
    for (ctor, args) in [
        (
            "CrystalBall",
            "m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0",
        ),
        ("Argus", "resonance = 5.29, slope = -20.0, power = 0.5"),
    ] {
        let src = format!(
            "hep = standard_module(\"particle-physics\", \"0.1\")\n\
             obs = [5.1, 5.15, 5.2, 5.22, 5.25]\n\
             x = elementof(reals)\n\
             lp = logdensityof(lawof(iid(hep.{ctor}({args}), 5)), obs)\n\
             inputs = (x)\n\
             outputs = lp"
        );
        let mlir = emit(&src);
        assert!(
            mlir.contains("stablehlo.reduce"),
            "the batched density reduces over the axis rather than unrolling:\n{mlir}"
        );
        // One density expression, not one per observation: the batched path is
        // what makes an n-observation iid independent of n in emitted size.
        assert!(
            mlir.matches("chlo.erf").count() <= 2,
            "{ctor} must emit its normalizer once, not once per element:\n{mlir}"
        );
    }
}

// The batched kernel input wraps each per-batch-constant parameter in a size-1
// `vector`. `argus_logpdf` reads `power` STRUCTURALLY (its normalizer has a
// closed form only at 0.5), so it is the one §09 builder that notices, and it
// strips the wrapper via `unbatch_field_id`. Without the strip, a batched Argus
// refused as a non-literal `power` even when the literal was 0.5.
#[test]
fn batched_argus_still_reads_its_literal_power() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
obs = [5.1, 5.15, 5.2]
x = elementof(reals)
lp = logdensityof(lawof(iid(hep.Argus(resonance = 5.29, slope = -20.0, power = 1.5), 3)), obs)
inputs = (x)
outputs = lp";
    let msg = emit_err(src);
    assert!(
        msg.contains("lower incomplete gamma"),
        "a batched non-0.5 power must reach the SAME refusal as the scalar case, \
         not a spurious non-literal one: {msg}"
    );
}

// Refuse-don't-mislower: at any other `power` the normalizer is a genuine lower
// incomplete gamma, for which there is no lowering here. The refusal names the
// missing function rather than guessing a normalizer.
#[test]
fn argus_at_another_power_refuses_naming_the_incomplete_gamma() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
x = elementof(reals)
lp = logdensityof(lawof(hep.Argus(resonance = 5.29, slope = -20.0, power = 1.5)), x)
inputs = (x)
outputs = lp";
    let msg = emit_err(src);
    assert!(
        msg.contains("lower incomplete gamma"),
        "the refusal must name the missing function: {msg}"
    );
}

// ---- §09 ContinuedPoisson ---------------------------------------------------
//
// `ContinuedPoisson` is a `%finite` measure, not a probability measure, so
// `lawof` refuses it (§04: "an unnormalized measure is not its own law"). Every
// shape below therefore reaches the density the way HistFactory's own converter
// output does: `functionof(...)` + `likelihoodof(...)`.
//
// Executed, not merely pinned. Under the suite's IREE executor
// (`unified/stablehlo_exec.py`), against the §09 closed form
// `x*log(rate) - rate - gammaln(x+1)` evaluated with scipy in f64:
//
//   scalar, rate = 4.5
//     x =  0.0  -> -4.500000476837158   vs -4.5                 (4.8e-07 abs)
//     x =  1.0  -> -2.995922565460205   vs -2.995922603223726    (3.8e-08 abs)
//     x =  3.0  -> -1.779528021812439   vs -1.779527278899233    (7.4e-07 abs)
//     x =  3.7  -> -1.6713192462921143  vs -1.6713187782433527   (4.7e-07 abs)
//     x = 12.5  -> -6.959110260009766   vs -6.959108696541275    (1.6e-06 abs)
//     x = -0.5  -> -inf                 vs -inf                  (exact)
//     x = -1.5  -> -inf                 vs -inf                  (exact)
//
//   dotted staterror, tau = [400, 100], scored at tau
//     g = [1.0, 1.0] -> -7.137420654296875  vs -7.137236096803235  (2.6e-05 rel)
//     g = [1.1, 0.9] -> -9.54931640625      vs -9.549215740855573  (1.1e-05 rel)
//     g = [1.2, 0.8] -> -16.523193359375    vs -16.52296851064216  (1.4e-05 rel)
//
// `x = 3.7` is the point that matters: a non-integer variate is exactly what
// §09 adds this measure for, and `Poisson`'s counting-measure mass cannot score
// it. The dotted band is wider because `lgamma(401) ~ 1990` cancels down to a
// result of order ten, which is f32 arithmetic, not a lowering error.
//
// This crate evaluates no density (`flatppl-rust` has no evaluator), so it can
// pin the emitted MODULE but never a number. Those seven scalar points are
// pinned as an executed gate by `corpora/stablehlo/continued_poisson` in
// flatppl-testsuite, against the same scipy `gammaln` closed form; the dotted
// shape is gated by `corpora/hs3/conversions/histfactory` against an
// independent closed-form Poisson-product oracle. What the tests below add is
// the structure that gate cannot see.

const CONTINUED_POISSON_SRC: &str = "\
hep = standard_module(\"particle-physics\", \"0.1\")
x = elementof(nonnegreals)
aux = functionof(hep.ContinuedPoisson(rate = 4.5))
aux_lik = likelihoodof(aux, x)
lp = logdensityof(aux_lik, record(x = x))
inputs = (x)
outputs = (lp)";

// The whole density: one `log(rate)`, one `lgamma(x + 1)`, and the support mask.
// §09's own formula reads the Poisson factorial as `Γ(x+1)`, which is the ONLY
// difference from `poisson_logpdf` — so an emission that dropped the `lgamma`
// for a `Poisson`-style integer path would score a non-integer variate wrongly
// with no structural complaint.
#[test]
fn continued_poisson_emits_a_lgamma_continuation_under_a_support_mask() {
    let mlir = emit(CONTINUED_POISSON_SRC);
    assert_eq!(
        mlir.matches("chlo.lgamma").count(),
        1,
        "the gamma continuation is the whole point of §09's entry:\n{mlir}"
    );
    assert_eq!(
        mlir.matches("stablehlo.log ").count(),
        1,
        "`log(rate)` once:\n{mlir}"
    );
    assert!(
        mlir.contains("compare GE") && mlir.contains("dense<0x7F800000>"),
        "§09's support is `nonnegreals`, so the density is masked to `-inf` \
         below zero:\n{mlir}"
    );
}

/// Freeze the exact emitted text: any drift (op count, ordering, the mask, the
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn continued_poisson_matches_frozen_golden() {
    let out = emit(CONTINUED_POISSON_SRC);
    let golden = include_str!("goldens/continued_poisson_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/continued_poisson_logdensity.mlir)"
    );
}

// HistFactory's `staterror` constraint, the shape that motivated the builder:
// `functionof(hep.ContinuedPoisson.(g .* tau))` observed at `tau`. The
// determiniser keeps this axis-native, so it reaches
// `registry::lower_logdensityof_batched`, and before `ContinuedPoisson` was
// batch-safe the WHOLE model refused with "not rank-agnostic" — no HistFactory
// conversion carrying a `staterror` modifier could emit at all.
#[test]
fn a_dotted_staterror_constraint_emits_one_batched_density() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
tau = [400.0, 100.0]
g = elementof(cartpow(posreals, 2))
aux = functionof(hep.ContinuedPoisson.(g .* tau))
aux_lik = likelihoodof(aux, tau)
lp = logdensityof(aux_lik, record(g = g))
inputs = (g)
outputs = (lp)";
    let mlir = emit(src);
    assert!(
        mlir.contains("tensor<2xi1>"),
        "the support mask must carry the variate's rank-1 shape, so the reduce \
         that follows sums TWO terms rather than one:\n{mlir}"
    );
    assert!(
        mlir.contains("stablehlo.reduce"),
        "the per-bin densities are reduced over the axis:\n{mlir}"
    );
    // One density expression, not one per bin: the batched path is what makes an
    // n-bin staterror independent of n in emitted size.
    assert_eq!(
        mlir.matches("chlo.lgamma").count(),
        1,
        "the continuation must be emitted once, not once per bin:\n{mlir}"
    );
}

// A LITERAL variate must reach the same guarded formula as an ABI one: the
// emitted module carries the variate constant, the `GE 0` guard, the guarded
// `select` feeding the formula, and the `-inf` branch — at every one of the
// three cases §09 distinguishes.
//
// This is the regression a value gate cannot see. An emitter that special-cased
// a literal would be free to fold `x = 3.0` onto a `Poisson` integer path (same
// number, wrong measure, and then wrong at `x = 3.7`) or fold `x = -0.5`
// straight to a bare `-inf` constant, dropping the guarded evaluation that keeps
// the reverse-mode gradient `nan`-free. Both would still match every frozen
// value. The executed numbers for these points are in the header table and
// gated by `corpora/stablehlo/continued_poisson`.
#[test]
fn a_literal_variate_reaches_the_same_guarded_formula_at_each_support_case() {
    // (variate literal, the constant it must emit) — in support at an integer,
    // in support at a NON-integer, and below the support. A negative literal is
    // emitted as its magnitude plus a `negate`, which is why the third case
    // matches `dense<0.5>` rather than a signed constant.
    for (variate, constant) in [
        ("3.0", "dense<3.0>"),
        ("3.7", "dense<3.7>"),
        ("-0.5", "dense<0.5>"),
    ] {
        let src = format!(
            "hep = standard_module(\"particle-physics\", \"0.1\")\n\
             d = elementof(reals)\n\
             aux = functionof(hep.ContinuedPoisson(rate = 4.5))\n\
             aux_lik = likelihoodof(aux, {variate})\n\
             lp = logdensityof(aux_lik, record(d = d))\n\
             inputs = (d)\n\
             outputs = (lp)"
        );
        let mlir = emit(&src);
        assert!(
            mlir.contains(constant),
            "the variate literal {variate} must reach the module as {constant}:\n{mlir}"
        );
        assert!(
            mlir.contains("compare GE"),
            "{variate}: the support guard must survive a literal variate — a \
             folded-away guard is how a below-zero variate starts scoring a \
             finite number:\n{mlir}"
        );
        assert_eq!(
            mlir.matches("chlo.lgamma").count(),
            1,
            "{variate}: the gamma continuation must survive a literal variate — \
             an integer literal must NOT reach a `Poisson` factorial path:\n{mlir}"
        );
        assert!(
            mlir.contains("dense<0x7F800000>") && mlir.contains("stablehlo.negate"),
            "{variate}: the `-inf` off-support branch must be emitted at every \
             literal, including one below zero — folding it to a bare constant \
             would drop the guarded evaluation `mask_support` exists for:\n{mlir}"
        );
    }
}
