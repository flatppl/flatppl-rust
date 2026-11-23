use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{Module, Node, Scalar};
use flatppl_optimise::{Profile, optimise};

const INPUT: &str = "a = external(cartpow(reals, [2, 3]))\ninputs = a\n";

fn optimise_model(source: &str) -> Option<Module> {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    optimise(&model, None, Profile::Portable)
}

#[test]
fn a_single_matrix_cell_swaps_its_written_indices() {
    for (head, transposition, i, j) in [("get", "transpose", 3, 2), ("get0", "adjoint", 2, 1)] {
        let source = format!("{INPUT}outputs = {head}({transposition}(a), {i}, {j})\n");
        let model = optimise_model(&source).expect("the single matrix cell should simplify");
        let output = model
            .bindings()
            .find(|(_, binding)| model.resolve(binding.name) == "outputs")
            .unwrap()
            .1
            .rhs;
        let cell = expect_builtin_call(&model, output, head).unwrap();
        assert!(
            matches!(model.node(cell.args[0]), Node::Ref(reference) if model.resolve(reference.name) == "a")
        );
        assert_eq!(model.node(cell.args[1]), &Node::Lit(Scalar::Int(j)));
        assert_eq!(model.node(cell.args[2]), &Node::Lit(Scalar::Int(i)));
    }
}

#[test]
fn matrix_cell_rewrite_retains_unproved_bounds_and_shared_storage() {
    for outputs in [
        "get(transpose(a), 4, 2)",
        "get0(adjoint(a), 2, 2)",
        "(get(transpose(a), 3, 2), get(a, 2, 3))",
        "(get(transpose(a), 3, 2), sin.(a))",
    ] {
        assert!(
            optimise_model(&format!("{INPUT}outputs = {outputs}\n")).is_none(),
            "{outputs}"
        );
    }
}

#[test]
fn a_named_callback_keeps_its_original_matrix_selection() {
    let source = "
        a = elementof(cartpow(reals, [2, 3]))
        body = get(transpose(a), 3, 2)
        f = functionof(body, a = a)
        x = external(cartpow(cartpow(reals, [2, 3]), 2))
        inputs = x
        outputs = f.(x)
    ";
    assert!(optimise_model(source).is_none());
}
