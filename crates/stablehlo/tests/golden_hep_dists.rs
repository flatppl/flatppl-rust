//! StableHLO lowering of the two §09 `particle-physics` densities the coverage
//! corpus needs: `CrystalBall` and `Argus`.
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
