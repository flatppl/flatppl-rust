//! §09 `special-functions.erf` / `erfc` lower to `chlo.erf` / `chlo.erfc`.
//!
//! Executed through the testsuite's Enzyme-JAX `hlo_call` (enzyme_ad
//! 0.0.15+flatppl.13) against mpmath at x = -1.3, 0, 0.7, 10 and 26: in f64
//! the value, the gradient 2/√π·exp(-x²) and a `Normal(mu = erf(x))` density
//! agree to 4e-16 relative. f32 agrees to 3e-7. `erfc(10)` =
//! 2.088487583762545e-45 needs its own op: `1 - erf(10)` is 0.

use flatppl_core::Module;
use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

const SF: &str = "sf = standard_module(\"special-functions\", \"0.1\")";

fn determinize_abi(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots: Vec<_> = ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_options(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("must determinize, not refuse")
}

fn emit_with(src: &str, mode: Mode, dtype: Dtype) -> Result<String, String> {
    let opts = EmitOptions {
        dtype,
        ..Default::default()
    };
    flatppl_stablehlo::emit(&determinize_abi(src), mode, &opts).map_err(|e| format!("{e:?}"))
}

fn emit(src: &str) -> String {
    emit_with(src, Mode::LogDensity, Dtype::F32).expect("must emit")
}

#[test]
fn erf_and_erfc_lower_to_their_chlo_ops() {
    for (dtype, ty) in [(Dtype::F32, "f32"), (Dtype::F64, "f64")] {
        for (member, op) in [("erf", "chlo.erf"), ("erfc", "chlo.erfc")] {
            let src = format!(
                "{SF}\na = elementof(reals)\nv = sf.{member}(a)\ninputs = (a)\noutputs = (v)"
            );
            let out = emit_with(&src, Mode::LogDensity, dtype).expect("must emit");
            assert!(
                out.contains(&format!("{op} %arg0 : tensor<{ty}> -> tensor<{ty}>")),
                "{member} {ty}:\n{out}"
            );
        }
    }
}

// `1 - erf(x)` loses every digit of erfc in the upper tail.
#[test]
fn erfc_is_not_spelled_through_erf() {
    let out = emit(&format!(
        "{SF}\na = elementof(reals)\nv = sf.erfc(a)\ninputs = (a)\noutputs = (v)"
    ));
    assert!(!out.contains("chlo.erf "), "{out}");
    assert!(!out.contains("stablehlo.subtract"), "{out}");
}

#[test]
fn a_named_argument_binds_positionally() {
    let out = emit(&format!(
        "{SF}\na = elementof(reals)\nv = sf.erf(x = a)\ninputs = (a)\noutputs = (v)"
    ));
    assert!(out.contains("chlo.erf %arg0"), "{out}");
}

#[test]
fn a_broadcast_call_stays_elementwise() {
    let out = emit(&format!(
        "{SF}\na = elementof(cartpow(reals, 3))\nv = sf.erfc.(a)\ninputs = (a)\noutputs = (v)"
    ));
    assert!(
        out.contains(": tensor<3xf32> -> tensor<3xf32>") && out.contains("chlo.erfc %"),
        "{out}"
    );
}

// A probit-style mean: the member call feeds a density parameter.
#[test]
fn erf_feeds_a_density_parameter() {
    let out = emit(&format!(
        "{SF}\nmu = elementof(reals)\nx = draw(Normal(mu = sf.erf(mu), sigma = 1.0))\n\
         lp = logdensityof(lawof(record(x = x)), record(x = 0.5))\n\
         inputs = (mu)\noutputs = (lp)"
    ));
    assert!(out.contains("chlo.erf %arg0"), "{out}");
}

#[test]
fn erf_feeds_a_sampled_parameter() {
    let src = format!(
        "{SF}\nmu = external(reals)\n\
         x, state = rand(Normal(mu = sf.erf(mu), sigma = 1.0), rnginit(23))\n\
         inputs = (mu)\noutputs = (x, state)"
    );
    let opts = EmitOptions {
        restrict_enzyme_compatible: false,
        ..Default::default()
    };
    let out =
        flatppl_stablehlo::emit(&determinize_abi(&src), Mode::Sample, &opts).expect("must emit");
    assert!(out.contains("chlo.erf %arg0"), "{out}");
}

// Constant arguments fold through libm. Oracle: mpmath erfc(10), correctly rounded.
#[test]
fn a_constant_argument_folds_with_the_tail_intact() {
    let out = emit_with(
        &format!("{SF}\na = elementof(reals)\nv = a + sf.erfc(10.0)\ninputs = (a)\noutputs = (v)"),
        Mode::LogDensity,
        Dtype::F64,
    )
    .expect("must emit");
    assert!(!out.contains("chlo.erfc"), "{out}");
    // MLIR spells an f64 splat as its little-endian bytes in hex.
    let hex = out
        .split("dense<\"0x")
        .nth(1)
        .and_then(|rest| rest.get(..16))
        .expect("folded constant");
    let bytes = u64::from_str_radix(hex, 16).expect("hex").to_be_bytes();
    let value = f64::from_le_bytes(bytes);
    let expected = 2.088487583762545e-45;
    assert!(
        ((value - expected) / expected).abs() < 1e-15,
        "{value} vs {expected}"
    );
}

#[test]
fn a_member_without_a_chlo_op_refuses_naming_itself() {
    let err = emit_with(
        &format!(
            "{SF}\na = elementof(reals)\nv = sf.bessel_j(1.0, a)\ninputs = (a)\noutputs = (v)"
        ),
        Mode::LogDensity,
        Dtype::F32,
    )
    .expect_err("bessel_j has no StableHLO lowering");
    assert!(
        err.contains("special-functions.bessel_j has no StableHLO lowering"),
        "{err}"
    );
}
