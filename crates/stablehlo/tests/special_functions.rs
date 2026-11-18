//! §09 `special-functions.erf` / `erfc` lower to `chlo.erf` / `chlo.erfc`.
//!
//! Executed through the testsuite's Enzyme-JAX `hlo_call` (enzyme_ad
//! 0.0.15+flatppl.13) against mpmath at x = -1.3, 0, 0.7, 10 and 26: in f64
//! the value, the gradient 2/√π·exp(-x²) and a `Normal(mu = erf(x))` density
//! agree to 4e-16 relative. f32 agrees to 3e-7. `erfc(10)` =
//! 2.088487583762545e-45 needs its own op: `1 - erf(10)` is 0.

use flatppl_core::Module;
use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

const SF: &str = "sf = standard_module(\"special-functions\", \"0.1\")";

fn determinize_abi(src: &str) -> Module {
    determinize_with(src, &flatppl_stablehlo::LOWERING_OPTIONS)
}

fn determinize_with(src: &str, options: &flatppl_determinizer::LoweringOptions<'_>) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots: Vec<_> = ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_options(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        options,
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

// These members have no CHLO primitive. Portable loops supply their values.
// The testsuite executes the resulting values and derivatives against SciPy.
#[test]
fn remaining_special_functions_lower_with_live_inputs() {
    for expression in [
        "real(sf.bessel_j(v, x))",
        "real(sf.bessel_i(z = x, v = v))",
        "sf.bessel_y(v, x)",
        "sf.bessel_k(v, x)",
        "sf.airy(x)",
        "sf.gammainc(v, x)",
        "sf.betainc(v, 3.0, x)",
    ] {
        let source = format!(
            "{SF}\nv = external(reals)\nx = external(reals)\ninputs = (v, x)\noutputs = {expression}"
        );
        emit(&source);
    }
}

#[test]
fn complex_bessel_projections_preserve_broadcast_shape_and_aliases() {
    let source = format!(
        "{SF}\nx = external(cartpow(reals, 3))\norder = external(reals)\nj = sf.bessel_j\nv = j.(order, x)\ninputs = (order, x)\noutputs = (real.(v), imag.(v), abs.(v), abs2.(v))"
    );
    let out = emit(&source);
    assert!(out.contains("tensor<3xf32>"));
}

// Keep the reciprocal and rational correction as the only divisions. The old
// continued fraction made derivatives of batched Gaussian scores slow to compile.
#[test]
fn erfcx_lowers_to_differentiable_stablehlo() {
    for (dtype, ty) in [(Dtype::F32, "f32"), (Dtype::F64, "f64")] {
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
        assert_eq!(out.matches("stablehlo.divide").count(), 2, "{out}");
    }
    let out = emit(&format!(
        "{SF}\na = elementof(cartpow(reals, 3))\nv = sf.erfcx.(a)\ninputs = (a)\noutputs = (v)"
    ));
    assert!(
        out.contains("-> tensor<3xf32>") && out.contains("chlo.erfc %"),
        "{out}"
    );
}

// digamma and polygamma lower to their CHLO ops. Executed against mpmath on M4 and
// GB10: f64 value and gradient 1.3e-15 (2.2e-14 at digamma(-0.5) = 0.036, where the
// reflection cancels), f32 1.3e-6. Enzyme differentiates polygamma in `x` only, and
// the integer order reaches the op through a convert, which Enzyme treats as inactive.
#[test]
fn digamma_and_polygamma_lower_to_their_chlo_ops() {
    for (dtype, ty, int) in [(Dtype::F32, "f32", "i32"), (Dtype::F64, "f64", "i64")] {
        let src =
            format!("{SF}\na = elementof(reals)\nv = sf.digamma(a)\ninputs = (a)\noutputs = (v)");
        let out = emit_with(&src, Mode::LogDensity, dtype).expect("must emit");
        assert!(
            out.contains(&format!(
                "chlo.digamma %arg0 : tensor<{ty}> -> tensor<{ty}>"
            )),
            "{out}"
        );
        let src = format!(
            "{SF}\nn = elementof(integers)\na = elementof(reals)\nv = sf.polygamma(n = n, x = a)\n\
             inputs = (n, a)\noutputs = (v)"
        );
        let out = emit_with(&src, Mode::LogDensity, dtype).expect("must emit");
        assert!(
            out.contains(&format!(
                "stablehlo.convert %arg0 : (tensor<{int}>) -> tensor<{ty}>"
            )) && out.contains(&format!(
                "chlo.polygamma %0, %arg1 : tensor<{ty}>, tensor<{ty}> -> tensor<{ty}>"
            )),
            "{out}"
        );
    }
    let out = emit(&format!(
        "{SF}\na = elementof(cartpow(reals, 3))\nv = sf.polygamma.(2, a)\ninputs = (a)\noutputs = (v)"
    ));
    assert!(
        out.contains("chlo.polygamma")
            && out.contains("tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>"),
        "{out}"
    );
}

#[test]
fn normal_log_cdf_keeps_a_status_free_abi_for_checked_inputs() {
    let prior = "\
x = elementof(posreals)
inputs = x
P = normalize(truncate(Normal(0.0, 1.0), interval(0.0, inf)))
outputs = logdensityof(P, x)";
    let masked = format!(
        "{SF}\nx = external(reals)\ninputs = x\noutputs = sf.log_ndtr(ifelse(isnan(x), 0.0, x))"
    );
    for source in [prior, masked.as_str()] {
        let out = emit_with(source, Mode::LogDensity, Dtype::F64).unwrap();
        assert!(!out.contains("flatppl.runtime_status"), "{out}");
    }
}

#[test]
fn normal_log_cdf_splits_double_precision_gpu_fusion() {
    let source = format!("{SF}\nx = external(reals)\ninputs = x\noutputs = sf.log_ndtr(x)");
    let program = determinize_abi(&source);
    for (target, dtype, barriers) in [
        (Target::Cpu, Dtype::F64, 0),
        (Target::Gpu, Dtype::F32, 0),
        (Target::Gpu, Dtype::F64, 1),
    ] {
        let options = EmitOptions {
            target,
            dtype,
            ..Default::default()
        };
        let out = flatppl_stablehlo::emit(&program, Mode::LogDensity, &options).unwrap();
        assert_eq!(
            out.matches("stablehlo.optimization_barrier").count(),
            barriers
        );
    }
}

#[test]
fn normal_log_cdf_compacts_only_fixed_infinite_lanes() {
    let mask = (0..64)
        .map(|i| if i % 4 == 0 { "true" } else { "false" })
        .collect::<Vec<_>>()
        .join(", ");
    let pointwise = "f(x, keep) = sf.log_ndtr(ifelse(keep, x, -inf))\noutputs = f.(x, keep)";
    let source =
        format!("{SF}\nx = external(cartpow(reals, 64))\nkeep = [{mask}]\ninputs = x\n{pointwise}");
    let out = emit_with(&source, Mode::LogDensity, Dtype::F64).unwrap();
    let tail = out
        .lines()
        .find(|line| line.contains("stablehlo.composite \"enzyme.math.log_ndtr\""))
        .unwrap();
    assert!(tail.contains("tensor<16xf64>"), "{tail}");

    let source = format!(
        "{SF}\nx = external(cartpow(reals, 64))\nkeep = external(cartpow(booleans, 64))\n\
         inputs = (x, keep)\n{pointwise}"
    );
    let out = emit_with(&source, Mode::LogDensity, Dtype::F64).unwrap();
    let tail = out
        .lines()
        .find(|line| line.contains("stablehlo.composite \"enzyme.math.log_ndtr\""))
        .unwrap();
    assert!(tail.contains("tensor<64xf64>"), "{tail}");
}

#[test]
fn normal_interval_mass_uses_only_endpoint_cdfs() {
    let source = "\
mu = elementof(reals)
sigma = elementof(posreals)
lo = elementof(reals)
hi = elementof(reals)
inputs = (mu, sigma, lo, hi)
outputs = log(totalmass(truncate(Normal(mu, sigma), interval(lo, hi))))";
    let out = emit_with(source, Mode::LogDensity, Dtype::F64).unwrap();
    assert_eq!(out.matches("chlo.erfc ").count(), 0, "{out}");
    assert_eq!(
        out.matches("stablehlo.composite \"enzyme.math.log_ndtr\"")
            .count(),
        2,
        "{out}"
    );
}

#[test]
fn normal_interval_subtraction_skips_fixed_one_sided_lanes() {
    let lower = (0..64)
        .map(|i| if i % 4 == 0 { "-0.2" } else { "-inf" })
        .collect::<Vec<_>>()
        .join(", ");
    let prefix = "mu = external(cartpow(reals, 64))\n\
        f(x, a) = log(totalmass(truncate(Normal(x, 1.0), interval(a, 1.0))))\n";
    let (fixed, dynamic) = (
        format!("{prefix}lo = [{lower}]\ninputs = mu\noutputs = f.(mu, lo)"),
        format!(
            "{prefix}lo = external(cartpow(reals, 64))\ninputs = (mu, lo)\noutputs = f.(mu, lo)"
        ),
    );
    let options = EmitOptions {
        dtype: Dtype::F64,
        ..Default::default()
    };
    for (source, width) in [(fixed, 16), (dynamic, 64)] {
        let module = determinize_with(&source, &options.lowering_options());
        let ir = flatppl_stablehlo::emit(&module, Mode::LogDensity, &options).unwrap();
        let difference = ir
            .lines()
            .find(|line| line.contains("stablehlo.exponential_minus_one"));
        assert!(
            difference.is_some_and(|line| line.contains(&format!("tensor<{width}xf64>"))),
            "{difference:?}"
        );
    }
}

// Direct tails, logarithms and scaled Bessels must remain live arithmetic with
// every parameter exposed, including named argument binding and cell broadcasts.
#[test]
fn tail_and_scaled_extensions_preserve_parameters_and_broadcasts() {
    for expression in [
        "sf.gammaincc(x = x, a = a)",
        "sf.log_gammainc(a, x)",
        "sf.log_gammaincc(a, x)",
        "sf.betaincc(a, b, x)",
        "sf.log_betainc(a, b, x)",
        "sf.log_betaincc(a, b, x)",
        "sf.beta(a, b)",
        "sf.logbeta(a, b)",
        "real(sf.bessel_i_scaled(v = a, z = x))",
        "sf.bessel_k_scaled(a, x)",
        "sf.log_bessel_i(a, x)",
        "sf.log_bessel_k(a, x)",
        "sf.log_ndtr(x)",
    ] {
        for dtype in [Dtype::F32, Dtype::F64] {
            let source = format!(
                "{SF}\na = external(reals)\nb = external(reals)\nx = external(reals)\ninputs = (a, b, x)\noutputs = {expression}"
            );
            let out = emit_with(&source, Mode::LogDensity, dtype).expect(expression);
            assert!(out.contains("flatppl.runtime_status"), "{expression}");
        }
    }
    let source = format!(
        "{SF}\na = external(reals)\nx = external(cartpow(reals, 3))\nf = sf.bessel_i_scaled\nv = f.(a, x)\ninputs = (a, x)\noutputs = (real.(v), imag.(v))"
    );
    let out = emit(&source);
    assert!(out.contains("tensor<3xf32>"));
}
