use flatppl_stablehlo::{BatchSpec, Dtype, ElemKind, Emitter, MlirTy, Value};

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
