use flatppl_stablehlo::{BatchSpec, Dtype, ElemKind, Emitter, MlirTy, Value};

#[test]
fn a_diagonal_covariance_update_retains_matrix_vector_multiplication() {
    let mut module = flatppl_syntax::parse(
        "x = elementof(cartpow(reals, 2))\ninputs = x\n\
         A = rowstack([[x[1], 0.3], [0.4, x[2]], [0.1, 0.2]])\n\
         outputs = (eye(2) + diagmat(x) - col_gram(A)) * [0.6, -0.1]",
    )
    .unwrap();
    assert!(flatppl_infer::infer(&mut module).is_empty());
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap();
    let ir = flatppl_stablehlo::emit(
        &lowered,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap();
    // At x=[0.25,0.5], AᵀA=[[0.2325,0.295],[0.295,0.38]], so the result
    // is [0.64,-0.289]. Contracting columns preserves that vector law;
    // an elementwise fallback instead refuses the rank-2/rank-1 operands.
    assert!(ir.contains("contracting_dims = [1] x [0]"), "{ir}");
    assert!(ir.contains("-> tensor<2xf32>"), "{ir}");
}

#[test]
fn diagonal_matrices_preserve_cells_and_tensorize_host_batches() {
    let mut module =
        flatppl_syntax::parse("x = external(cartpow(reals, 3))\ny = diagmat(x)").unwrap();
    assert!(flatppl_infer::infer(&mut module).is_empty());
    let x = module.intern("x");
    let y = module.intern("y");
    let x = module.binding(module.binding_by_name(x).unwrap()).rhs;
    let y = module.binding(module.binding_by_name(y).unwrap()).rhs;
    let mut emitter = Emitter::new(&module, Dtype::F32);
    let input = Value {
        ssa: "%arg0".into(),
        ty: MlirTy::Ranked(vec![Some(3)]),
        elem: ElemKind::Real,
    };
    emitter.bind(x, input.clone());
    let output = emitter.lower_node(y).unwrap();
    emitter
        .tensorize_batched(
            "main",
            &[(input.ssa, input.ty, input.elem)],
            &[&output],
            &BatchSpec {
                shape: vec![2, 4],
                input_axes: vec![vec![Some(1), Some(0)]],
            },
        )
        .expect("diagmat must tensorize without the host loop fallback");
}
