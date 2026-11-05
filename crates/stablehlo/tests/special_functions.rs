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

/// The folded value of the one constant a module returns beside its input.
fn folded(member: &str, x: f64, dtype: Dtype) -> f64 {
    let out = emit_with(
        &format!(
            "{SF}\na = elementof(reals)\nv = sf.{member}({x:?})\ninputs = (a)\noutputs = (a, v)"
        ),
        Mode::LogDensity,
        dtype,
    )
    .expect("must emit");
    assert!(!out.contains("chlo."), "{member}({x}) must fold:\n{out}");
    let literal = out
        .split("stablehlo.constant dense<")
        .nth(1)
        .and_then(|rest| rest.split('>').next())
        .expect("folded constant");
    // MLIR spells a splat as a decimal or as its little-endian bytes in hex.
    match literal
        .strip_prefix("\"0x")
        .and_then(|h| h.strip_suffix('"'))
    {
        Some(hex) if hex.len() == 16 => {
            f64::from_le_bytes(u64::from_str_radix(hex, 16).unwrap().to_be_bytes())
        }
        Some(hex) => f32::from_le_bytes(u32::from_str_radix(hex, 16).unwrap().to_be_bytes()) as f64,
        None => literal.parse().expect("decimal literal"),
    }
}

// Constant arguments fold through libm, and the fold agrees with scipy.special 1.18
// (`erf`/`erfc` on float64 and on float32 inputs) as the executed module does. Below
// the smallest normal the paths differ only in underflow: libm rounds erfc(27) to the
// subnormal 5.2e-319 where scipy and the CPU runtime give 0, and f32 erfc(10) to
// 2.8e-45 where scipy gives 1.4e-45 and the runtime 0. So the floor is that normal.
#[test]
fn a_constant_argument_folds_to_the_scipy_value() {
    let cases: [(&str, f64, Dtype, f64); 22] = [
        ("erfc", 10.0, Dtype::F64, 2.0884875837625446e-45),
        ("erfc", 26.0, Dtype::F64, 5.663192408856145e-296),
        ("erfc", 27.0, Dtype::F64, 0.0),
        ("erfc", -1.3, Dtype::F64, 1.9340079449406524),
        ("erfc", 0.7, Dtype::F64, 0.32219880616258156),
        ("erf", -1.3, Dtype::F64, -0.9340079449406524),
        ("erf", 0.7, Dtype::F64, 0.6778011938374184),
        ("erfc", 10.0, Dtype::F32, 1.401298464324817e-45),
        ("erfc", 26.0, Dtype::F32, 0.0),
        ("erfc", 27.0, Dtype::F32, 0.0),
        ("erfc", -1.3, Dtype::F32, 1.9340078830718994),
        ("erfc", 0.7, Dtype::F32, 0.3221988081932068),
        ("erf", -1.3, Dtype::F32, -0.9340079426765442),
        ("erf", 0.7, Dtype::F32, 0.6778011918067932),
        ("erfcx", 10.0, Dtype::F64, 0.05614099274382259),
        ("erfcx", -1.3, Dtype::F64, 10.481318741176324),
        ("erfcx", 0.7, Dtype::F64, 0.525930337349441),
        ("erfcx", 1e4, Dtype::F64, 5.641895807268084e-05),
        ("erfcx", 10.0, Dtype::F32, 0.05614099279046059),
        ("erfcx", -1.3, Dtype::F32, 10.481317520141602),
        ("erfcx", 0.7, Dtype::F32, 0.5259303450584412),
        ("erfcx", 1e4, Dtype::F32, 5.641895768349059e-05),
    ];
    for (member, x, dtype, scipy) in cases {
        let (rtol, floor) = match dtype {
            Dtype::F64 => (1e-15, f64::MIN_POSITIVE),
            Dtype::F32 => (2.4e-7, f32::MIN_POSITIVE as f64),
        };
        let got = folded(member, x, dtype);
        assert!(
            got.is_finite() && (got - scipy).abs() <= rtol * scipy.abs() + floor,
            "{member}({x}) at rtol {rtol:e}: folded {got:e}, scipy {scipy:e}"
        );
    }
    // In the normal range the f64 tail is exact to the last bits, not hidden by the floor.
    let tail = folded("erfc", 10.0, Dtype::F64);
    assert!(((tail - 2.0884875837625446e-45) / 2.0884875837625446e-45).abs() < 1e-15);
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

// erfcx has no CHLO op. It is `exp(x²)·erfc(x)` below the split and a continued
// fraction above, joined by select. Executed against mpmath on M4 and GB10 out to
// x = 1e4: f64 value 4.2e-16 and gradient 1.3e-15, f32 2.1e-7 and 1.5e-6.
#[test]
fn erfcx_lowers_to_differentiable_stablehlo() {
    for (dtype, ty, depth) in [(Dtype::F32, "f32", 12), (Dtype::F64, "f64", 40)] {
        let src =
            format!("{SF}\na = elementof(reals)\nv = sf.erfcx(x = a)\ninputs = (a)\noutputs = (v)");
        let out = emit_with(&src, Mode::LogDensity, dtype).expect("must emit");
        assert!(!out.contains("chlo.erfcx"), "{out}");
        assert!(
            out.contains("chlo.erfc %") && out.contains(&format!("-> tensor<{ty}>")),
            "{out}"
        );
        assert_eq!(
            out.matches("stablehlo.select").count(),
            3,
            "two clamps and the join:\n{out}"
        );
        assert_eq!(out.matches("stablehlo.divide").count(), depth + 1, "{out}");
    }
    let out = emit(&format!(
        "{SF}\na = elementof(cartpow(reals, 3))\nv = sf.erfcx.(a)\ninputs = (a)\noutputs = (v)"
    ));
    assert!(
        out.contains("-> tensor<3xf32>") && out.contains("chlo.erfc %"),
        "{out}"
    );
}
