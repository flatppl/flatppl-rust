//! §07 "Scalar restrictions and constructors" (`integer`, `boolean`, `complex`,
//! `cis`, `conj`) and "Approximation functions" (`polynomial`, `bernstein`,
//! `stepwise`) as emitter heads. Structural only: the numbers are checked by
//! executing the modules under Enzyme-JAX against numpy/scipy in the PR.

use flatppl_stablehlo::{Dtype, EmitOptions, Mode, emit};

fn emitted(outputs: &str) -> Result<String, String> {
    emitted_as(outputs, Dtype::F32)
}

fn emitted_as(outputs: &str, dtype: Dtype) -> Result<String, String> {
    let src = format!(
        "flatppl_compat = \"0.1\"\nz0 = elementof(reals)\ninputs = (z0)\noutputs = {outputs}"
    );
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "inference errors: {diags:?}"
    );
    let d = flatppl_determinizer::determinize(&m).expect("must lower, not refuse");
    emit(
        &d,
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

fn emits(outputs: &str) -> String {
    emitted(outputs).unwrap_or_else(|e| panic!("{outputs} must emit: {e}"))
}

// Inference types `integer(x)` as an integer, so it lowers to an integer tensor
// only where `x` is integral by construction. Any other real is refused: §07
// makes a non-integral value an error, and an integer tensor has no NaN.
#[test]
fn integer_lowers_an_integral_operand_and_refuses_a_computed_real() {
    for head in ["round", "floor", "ceil"] {
        let out = emits(&format!("(integer({head}(z0)))"));
        assert!(
            out.contains("stablehlo.convert") && out.contains("-> tensor<i32>"),
            "{head}:\n{out}"
        );
        assert!(!out.contains("stablehlo.select"), "{head}:\n{out}");
    }
    // The consumers of an integer: an index, an exact comparison, `div`.
    emits("(get([1.0, 2.0, 3.0], integer(round(z0))))");
    emits("(ifelse(integer(floor(z0)) == 3, z0, 2.0))");
    emits("(div(integer(ceil(z0)), 2) + 1)");
    emits("(integer(z0 > 0.0) + integer(3.0))");
    for expr in ["integer(z0)", "integer(z0 * 2.0)", "integer(abs(z0))"] {
        let e = emitted(&format!("({expr})")).expect_err("a computed real must refuse");
        assert!(
            e.contains("`integer(x)` of a computed real has no lowering here")
                && e.contains("`integer(round(x))`"),
            "{expr}: {e}"
        );
    }
}

#[test]
fn integer_refuses_literal_values_outside_the_emitted_integer_width() {
    for (dtype, valid, invalid) in [
        (
            Dtype::F32,
            ["2147483520.0", "-2147483648.0"],
            ["2147483648.0", "-2147483904.0"],
        ),
        (
            Dtype::F64,
            ["9223372036854774784.0", "-9223372036854775808.0"],
            ["9223372036854775808.0", "-9223372036854777856.0"],
        ),
    ] {
        for value in valid {
            let expr = format!("integer(round({value}))");
            emitted_as(&expr, dtype).unwrap_or_else(|e| panic!("{expr}: {e}"));
            if !value.starts_with('-') {
                emitted_as(&format!("integer({value})"), dtype).unwrap();
            }
        }
        for value in invalid {
            for expr in [
                format!("integer(round({value}))"),
                format!("integer(floor({value}))"),
                format!("integer(ceil({value}))"),
            ] {
                let err = emitted_as(&expr, dtype).expect_err(&expr);
                assert!(err.contains("integer width"), "{expr}: {err}");
            }
        }
        assert!(emitted_as(&format!("integer({})", invalid[0]), dtype).is_err());
        assert!(emitted_as("integer(1e300)", dtype).is_err());
        emitted_as("integer(round(z0))", dtype).unwrap();
    }
    assert!(emitted_as("integer(2147483600.0)", Dtype::F32).is_err());
    assert!(emitted_as("integer(round(2147483647))", Dtype::F32).is_err());
    assert!(emitted_as("integer(round(9223372036854775807))", Dtype::F64).is_err());
}

#[test]
fn boolean_lowers_a_written_bit_or_a_boolean() {
    emits("(ifelse(boolean(1), z0, 2.0))");
    emits("(boolean(z0 > 0.0))");
    let e = emitted("(boolean(integer(round(z0))))").expect_err("a computed number must refuse");
    assert!(
        e.contains("`boolean(x)` of a computed number has no lowering here"),
        "{e}"
    );
}

#[test]
fn a_complex_value_refuses_and_its_projections_lower() {
    for head in ["complex(z0, 1.0)", "cis(z0)", "conj(complex(z0, 1.0))"] {
        let e = emitted(&format!("({head})")).expect_err("a complex output must refuse");
        assert!(
            e.contains("makes a complex value, which has no tensor form here"),
            "{head}: {e}"
        );
    }
    emits("(abs(complex(z0, 2.0) * cis(0.5)) + real(conj(complex(1.0, z0))))");
    // `conj` of a real is the identity.
    let out = emits("(conj(z0))");
    assert!(!out.contains("stablehlo.negate"), "{out}");
}

#[test]
fn polynomial_is_horner_over_static_coefficients() {
    let out = emits("(polynomial([1.0, -2.0, 0.5], z0))");
    assert_eq!(out.matches("stablehlo.multiply").count(), 2, "{out}");
    emits("(polynomial([z0, 2.0, z0 * z0], 1.5))");
}

#[test]
fn bernstein_is_de_casteljau() {
    // Degree 3: 3 + 2 + 1 combinations of two multiplies each.
    let out = emits("(bernstein([1.0, 3.0, 2.0, 0.5], z0))");
    assert_eq!(out.matches("stablehlo.multiply").count(), 12, "{out}");
}

#[test]
fn stepwise_selects_a_bin_and_is_nan_outside() {
    let out = emits("(stepwise([0.0, 1.0, 2.5, 4.0], [1.5, -2.0, 3.0], z0))");
    assert_eq!(out.matches("stablehlo.select").count(), 3, "{out}");
    assert!(
        out.contains("compare  LE") || out.contains("compare LE"),
        "last bin closed:\n{out}"
    );
    // MLIR needs the NaN outside the edges as exact bits, not `NaN.0`.
    assert!(out.contains("dense<0x7FC00000>"), "a bit-exact NaN:\n{out}");
    assert!(!out.contains("NaN.0"), "{out}");
}

#[test]
fn integer_checks_constant_bindings_without_freezing_runtime_inputs() {
    for (bindings, succeeds) in [
        ("x = 1e300\ninputs = z0\n", false),
        ("x = 1e300\ninputs = (z0, x)\n", true),
    ] {
        let source = format!("z0 = elementof(reals)\n{bindings}outputs = integer(round(x))\n");
        let mut module = flatppl_syntax::parse(&source).unwrap();
        flatppl_infer::infer(&mut module);
        let module = flatppl_determinizer::determinize(&module).unwrap();
        let result = emit(&module, Mode::LogDensity, &Default::default());
        assert_eq!(result.is_ok(), succeeds, "{result:?}");
    }
}
