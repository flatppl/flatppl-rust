use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

fn emit(body: &str, dtype: Dtype, target: Target) -> String {
    let source = format!("la=standard_module(\"ext-linear-algebra\",\"0.1\")\n{body}");
    let mut module = flatppl_syntax::parse(&source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let module = flatppl_determinizer::determinize_with_options(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &module,
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            target,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn rectangular_decompositions_and_least_squares_preserve_shapes() {
    for dtype in [Dtype::F32, Dtype::F64] {
        for target in [Target::Cpu, Target::Gpu] {
            for (m, n) in [(3, 2), (2, 3)] {
                let output = emit(
                    &format!(
                        "A=external(cartpow(reals,[{m},{n}]))\nb=external(cartpow(reals,{m}))\ninputs=(A,b)\nf=la.svd(A)\noutputs=(f.U,f.S,f.V,la.lstsq(A,b),la.rank(A))"
                    ),
                    dtype,
                    target,
                );
                let real = if matches!(dtype, Dtype::F32) {
                    "f32"
                } else {
                    "f64"
                };
                assert!(output.contains(&format!("tensor<{m}x2x{real}>")));
                assert!(output.contains(&format!("tensor<{n}x2x{real}>")));
            }
        }
    }
}

#[test]
fn integer_inputs_produce_real_factors_and_solutions() {
    let output = emit(
        "A=external(cartpow(integers,[3,2]))\nb=external(cartpow(integers,3))\ninputs=(A,b)\nf=la.svd(A)\noutputs=(f.U,f.V,la.lstsq(A,b))",
        Dtype::F64,
        Target::Cpu,
    );
    assert!(output.contains("-> (tensor<3x2xf64>, tensor<2x2xf64>, tensor<2xf64>,"));
}

#[test]
fn complex_projections_and_matrix_cell_broadcast_lower() {
    emit(
        "Ar=external(cartpow(reals,[3,2]))\nAi=external(cartpow(reals,[3,2]))\nA=complex.(Ar,Ai)\ninputs=(Ar,Ai)\nf=la.svd(A)\noutputs=(real.(f.U),imag.(f.V),f.S,la.rank(A))",
        Dtype::F64,
        Target::Cpu,
    );
    emit(
        "A=external(cartpow(cartpow(reals,[3,2]),4))\nb=external(cartpow(cartpow(reals,3),4))\ninputs=(A,b)\nf=la.svd.(A)\noutputs=(f.U,f.S,f.V,la.lstsq.(A,b),la.rank.(A))",
        Dtype::F32,
        Target::Gpu,
    );
}

#[test]
fn empty_matrices_keep_thin_factor_and_solution_shapes() {
    for (m, n) in [(0, 0), (0, 3), (3, 0)] {
        let m_expr = if m == 0 { "n" } else { "3" };
        let n_expr = if n == 0 { "n" } else { "3" };
        let output = emit(
            &format!(
                "t=external(reals)\nn=lengthof(filter(fn(_>2),[1]))\nA=fill(t,[{m_expr},{n_expr}])\nb=fill(t,[{m_expr}])\ninputs=t\nf=la.svd(A)\noutputs=(f.U,f.S,f.V,la.lstsq(A,b),la.rank(A))"
            ),
            Dtype::F64,
            Target::Cpu,
        );
        assert!(output.contains(&format!("tensor<{m}x0xf64>")));
        assert!(output.contains(&format!("tensor<{n}x0xf64>")));
    }
}
