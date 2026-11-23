use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit(source: &str, dtype: Dtype) -> Result<String, flatppl_stablehlo::EmitError> {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    flatppl_stablehlo::emit(
        &module,
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            ..EmitOptions::default()
        },
    )
}

#[test]
fn dropped_vector_cell_must_fit_the_backend_integer_width() {
    let query = "dummy = external(reals)\ninputs = dummy\noutputs = get([1, 2147483648], 1)";
    let error = emit(query, Dtype::F32).unwrap_err();
    assert!(error.to_string().contains("2147483648 is outside i32"));
    let f64 = emit(query, Dtype::F64).unwrap();
    assert!(f64.contains("tensor<i64>"));
    assert!(!f64.contains("2147483648"));
}

#[test]
fn demand_drops_unused_abi_reads_and_real_permutations() {
    let scalars =
        "x = external(reals)\ny = external(reals)\ninputs = (x, y)\noutputs = get([x, y], 1)";
    for (dtype, suffix) in [(Dtype::F32, "f32"), (Dtype::F64, "f64")] {
        let direct = emit(scalars, dtype).unwrap();
        assert!(direct.contains(&format!("%arg1: tensor<{suffix}>")));
        assert!(direct.contains("return %arg0"));
        for op in ["transpose", "adjoint"] {
            let source = format!(
                "x = external(cartpow(reals, [2, 3]))\ny = external(cartpow(reals, [3, 2]))\n\
                 inputs = (x, y)\n_payload = [x, {op}(y)]\noutputs = sum(get(_payload, 1))"
            );
            let hlo = emit(&source, dtype).unwrap();
            assert!(hlo.contains(&format!("%arg1: tensor<3x2x{suffix}>")));
            for dead in ["transpose", "reshape", "concatenate", "slice"] {
                assert!(!hlo.contains(&format!("stablehlo.{dead}")), "{hlo}");
            }
        }
    }
}

#[test]
fn demand_keeps_arithmetic_on_unused_runtime_inputs() {
    let query =
        "x = external(reals)\ny = external(reals)\ninputs = (x, y)\noutputs = get([x, exp(y)], 1)";
    let hlo = emit(query, Dtype::F64).unwrap();
    assert!(hlo.contains("stablehlo.exponential"), "{hlo}");
}

#[test]
fn a_syntactic_shape_requirement_survives_direct_flatpdl_emission() {
    let query = |size| {
        format!("x = external(cartpow(reals, 4))\nn = {size}\ninputs = x\noutputs = tile(x, n)")
    };
    for size in ["[1]", "1 + 0", "0"] {
        let error = emit(&query(size), Dtype::F64).unwrap_err().to_string();
        assert!(
            error.contains("size written as positive integer literals"),
            "{size}: {error}"
        );
    }
    assert!(emit(&query("1"), Dtype::F64).is_ok());
    let literal = "x = external(cartpow(reals, 4))\ninputs = x\noutputs = tile(x, 1)";
    assert!(emit(literal, Dtype::F64).is_ok());

    let split = "x = external(cartpow(reals, 4))\nn = [2]\ninputs = x\noutputs = joinblocks(splitblocks(x, n))";
    let error = emit(split, Dtype::F64).unwrap_err().to_string();
    assert!(error.contains("size written as positive integer literals"));
    let direct = split.replace("splitblocks(x, n)", "splitblocks(x, 2)");
    assert!(emit(&direct, Dtype::F64).unwrap().contains("return %arg0"));
}
