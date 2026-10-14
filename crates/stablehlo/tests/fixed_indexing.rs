use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit(source: &str, dtype: Dtype) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let module = flatppl_determinizer::determinize(&module).unwrap();
    flatppl_stablehlo::emit(
        &module,
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            ..Default::default()
        },
    )
    .unwrap()
}

fn assert_fixed_values(ir: &str, values: &[f64], dtype: Dtype) {
    use std::fmt::Write;
    let mut bits = String::new();
    for &value in values {
        let bytes = match dtype {
            Dtype::F32 => (value as f32).to_le_bytes().to_vec(),
            Dtype::F64 => value.to_le_bytes().to_vec(),
        };
        for byte in bytes {
            write!(bits, "{byte:02X}").unwrap();
        }
    }
    assert!(ir.contains(&format!("dense<\"0x{bits}\">")), "{ir}");
    assert!(!ir.contains("stablehlo.gather"), "{ir}");
}

#[test]
fn lag_matrix_is_fixed_before_backend_compilation() {
    let source = "y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]\n\
        times = indicesof(zeros(4)) .+ 3\n\
        lagged(t) = y[t .- indicesof(zeros(3))]\n\
        X = rowstack(lagged.(times))\n\
        p = elementof(cartpow(reals, 3))\ninputs = p\noutputs = X * p";
    for dtype in [Dtype::F32, Dtype::F64] {
        assert_fixed_values(
            &emit(source, dtype),
            &[30., 20., 10., 40., 30., 20., 50., 40., 30., 60., 50., 40.],
            dtype,
        );
    }
}

#[test]
fn fixed_selection_preserves_nested_cells_and_repeated_indices() {
    for (expression, expected) in [
        ("pick.([3, 1, 3])", vec![50., 60., 10., 20., 50., 60.]),
        (
            "pick.([[3, 1], [2, 2]])",
            vec![50., 60., 10., 20., 30., 40., 30., 40.],
        ),
        (
            "pick0.([[2, 0], [1, 1]])",
            vec![50., 60., 10., 20., 30., 40., 30., 40.],
        ),
    ] {
        let source = format!(
            "xs = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]\n\
            pick(i) = xs[i]\npick0(i) = get0(xs, i)\n\
            p = elementof(reals)\ninputs = p\noutputs = {expression}"
        );
        assert_fixed_values(&emit(&source, Dtype::F32), &expected, Dtype::F32);
    }
}

#[test]
fn fixed_selection_uses_each_callable_batchs_source_cells() {
    let ir = emit(
        "xs = [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],\n\
               [[70.0, 80.0], [90.0, 100.0], [110.0, 120.0]]]\n\
        pick(x, i) = x[i]\np = elementof(reals)\ninputs = p\n\
        outputs = pick.(xs, [[3, 1], [2, 2]])",
        Dtype::F32,
    );
    assert_fixed_values(&ir, &[50., 60., 10., 20., 90., 100., 90., 100.], Dtype::F32);
    assert!(ir.contains("tensor<2x2x2xf32>"), "{ir}");
}

#[test]
fn query_inputs_remain_live_when_the_source_binding_is_fixed() {
    for inputs in ["xs", "indices", "(xs, indices)"] {
        let source = format!(
            "xs = [10.0, 20.0, 30.0]\nindices = [3, 1, 3]\n\
            pick(i) = xs[i]\ninputs = {inputs}\noutputs = pick.(indices)"
        );
        let ir = emit(&source, Dtype::F32);
        let gather = ir
            .lines()
            .find(|line| line.contains("stablehlo.gather"))
            .unwrap();
        assert!(gather.contains("tensor<3xf32>"), "{ir}");
        assert!(!ir.contains("0x0000F041000020410000F041"), "{ir}");
    }
}

#[test]
fn fixed_gather_materialization_is_bounded_by_result_size() {
    for n in [4096, 4097] {
        let source = format!(
            "xs = [1.0, 2.0]\nindices = fill(2, {n})\n\
            pick(i) = xs[i]\np = elementof(reals)\n\
            inputs = p\noutputs = pick.(indices)"
        );
        let ir = emit(&source, Dtype::F32);
        if n == 4096 {
            assert_fixed_values(&ir, &vec![2.; n], Dtype::F32);
        } else {
            assert!(ir.contains("stablehlo.gather"), "{ir}");
            assert!(ir.contains("tensor<4097xf32>"), "{ir}");
        }
    }
}

#[test]
fn invalid_fixed_indices_and_overflow_keep_runtime_lowering() {
    let ir = emit(
        "xs = [10.0, 20.0, 30.0]\npick(i) = xs[i]\n\
        p = elementof(reals)\ninputs = p\noutputs = pick.([0, 4])",
        Dtype::F32,
    );
    assert!(ir.contains("stablehlo.gather"), "{ir}");
    assert!(ir.contains("dense<[-1, 3]>"), "{ir}");
    let ir = emit(
        "shift(t) = t .+ [2147483647, 1]\n\
        p = elementof(reals)\ninputs = p\noutputs = shift.([1, 2])",
        Dtype::F32,
    );
    assert!(ir.contains("stablehlo.add"), "{ir}");
    assert!(ir.contains("tensor<2x2xi32>"), "{ir}");
}
