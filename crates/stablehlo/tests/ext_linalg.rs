use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(&model, Mode::LogDensity, &EmitOptions::default()).unwrap()
}

#[test]
fn matrix_rhs_solutions_keep_columns_and_replace_only_solution_rows() {
    for (shape, expression, result) in [
        ("[3,2]", "e.lstsq(A,b,0.001,0.0)", "2x4"),
        ("[3,3]", "e.solve_triangular(A,b,false,true)", "3x4"),
        ("[3,3]", "e.cholesky_solve(A,b)", "3x4"),
        ("[3,3]", "e.lu_solve(e.lu(A),b)", "3x4"),
    ] {
        let ir = emit(&format!(
            "e=standard_module(\"ext-linear-algebra\",\"0.1\")\n\
             A=external(cartpow(reals,{shape}))\nb=external(cartpow(reals,[3,4]))\n\
             inputs=(A,b)\noutputs={expression}"
        ));
        let signature = ir.lines().find(|l| l.contains("func.func @")).unwrap();
        assert!(
            signature.contains(&format!("-> (tensor<{result}xf32>")),
            "{signature}"
        );
    }
}

#[test]
fn pseudoinverse_exchanges_rectangular_axes_and_preserves_complex_elements() {
    let ir = emit(
        "e=standard_module(\"ext-linear-algebra\",\"0.1\")\n\
         Ar=external(cartpow(reals,[2,3]))\nAi=external(cartpow(reals,[2,3]))\n\
         p=e.pinv(complex.(Ar,Ai))\ninputs=(Ar,Ai)\noutputs=(real.(p),imag.(p))",
    );
    let signature = ir.lines().find(|l| l.contains("func.func @")).unwrap();
    assert!(
        signature.contains("-> (tensor<3x2xf32>, tensor<3x2xf32>"),
        "{signature}"
    );
}

#[test]
fn hermitian_values_and_vector_columns_retain_batch_axes() {
    let ir = emit(
        "e=standard_module(\"ext-linear-algebra\",\"0.1\")\n\
         A=external(cartpow(cartpow(reals,[2,2]),3))\n\
         values(a)=e.eigvalsh(a)\nvectors(a)=e.eigh(a).vectors\n\
         inputs=A\noutputs=(values.(A),vectors.(A))",
    );
    let signature = ir.lines().find(|l| l.contains("func.func @")).unwrap();
    assert!(
        signature.contains("-> (tensor<3x2xf32>, tensor<3x2x2xf32>"),
        "{signature}"
    );
}

#[test]
fn empty_rank_is_an_authored_output_even_when_status_has_the_same_ssa() {
    let ir = emit(
        "e=standard_module(\"ext-linear-algebra\",\"0.1\")\n\
         z=lengthof(filter(fn(_ > 2),[1]))\nt=external(reals)\n\
         inputs=t\noutputs=e.rank(fill(t,[z,z]))",
    );
    assert!(ir.contains("flatppl.runtime_status = 1 : i64"), "{ir}");
    let signature = ir.lines().find(|l| l.contains("func.func @")).unwrap();
    assert!(
        signature.contains("-> (tensor<i32>, tensor<i32>)"),
        "{signature}"
    );
}

#[test]
fn authored_zero_does_not_change_the_slot_of_a_runtime_failure() {
    let ir = emit("x=external(reals)\ninputs=x\noutputs=(0,integer(round(x)))");
    assert!(ir.contains("flatppl.runtime_status = 2 : i64"), "{ir}");
    let signature = ir.lines().find(|l| l.contains("func.func @")).unwrap();
    assert!(
        signature.contains("-> (tensor<i32>, tensor<i32>, tensor<i32>)"),
        "{signature}"
    );
}
