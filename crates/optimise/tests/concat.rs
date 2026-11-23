use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{Module, Node, NodeId, Scalar};
use flatppl_optimise::{Profile, optimise};

const INPUT: &str = "a = external(cartpow(reals, 3))\ninputs = a\n";

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

fn outputs(m: &Module) -> NodeId {
    m.bindings()
        .find(|(_, binding)| m.resolve(binding.name) == "outputs")
        .unwrap()
        .1
        .rhs
}

#[test]
fn a_written_concat_selection_resolves_its_partition_and_index_base() {
    for (head, source, index) in [
        ("get", "get(cat(a, [0.0]), 3)", 3),
        ("get0", "get0(cat([7.0, 8.0], a, [0.0]), 4)", 2),
    ] {
        let model = optimise_model(&format!("{INPUT}outputs = {source}\n")).unwrap();
        let read = expect_builtin_call(&model, outputs(&model), head).unwrap();
        assert!(matches!(model.node(read.args[0]), Node::Ref(r) if model.resolve(r.name) == "a"));
        assert_eq!(model.node(read.args[1]), &Node::Lit(Scalar::Int(index)));
    }
}

#[test]
fn fixed_real_concat_selection_returns_the_selected_cell() {
    let source = "x = cat([1.0, 2.0], [3.0])\noutputs = get0(x, 2)\n";
    let model = optimise_model(source).unwrap();
    assert_eq!(model.node(outputs(&model)), &Node::Lit(Scalar::Real(3.0)));
}

#[test]
fn a_concat_selection_keeps_status_and_shared_producer_paths() {
    for source in [
        "get(cat(a, [0.0]), 4)",
        "get(cat(a, [0.0]), 5)",
        "(get(cat(a, [0.0]), 1), get(a, 1))",
        "(get(cat(a, [0.0]), 1), sin.(a))",
        "(get(cat(a .+ a, [0.0]), 1), get(a .+ a, 1))",
    ] {
        assert!(
            optimise_model(&format!("{INPUT}outputs = {source}\n")).is_none(),
            "{source}"
        );
    }
}

#[test]
fn a_promoted_input_hides_its_authored_concat_partitions() {
    let source = "packed = cat([1.0], [2.0])\ninputs = packed\noutputs = get(packed, 2)\n";
    assert!(optimise_model(source).is_none());
}

#[test]
fn singleton_stacks_preserve_fixed_data_and_the_matrix_axes() {
    for (head, lead, trail) in [("rowstack", 1, 0), ("colstack", 0, 1)] {
        let source =
            format!("data = record(income = [1.0, 2.0, 3.0])\noutputs = {head}([data.income])\n");
        let model = optimise_model(&source).unwrap();
        let axes = expect_builtin_call(&model, outputs(&model), "addaxes").unwrap();
        assert!(expect_builtin_call(&model, axes.args[0], "get").is_some());
        assert_eq!(model.node(axes.args[1]), &Node::Lit(Scalar::Int(lead)));
        assert_eq!(model.node(axes.args[2]), &Node::Lit(Scalar::Int(trail)));
    }
}

#[test]
fn singleton_stack_keeps_a_promoted_data_field() {
    let source = "data = record(income = [1.0, 2.0, 3.0])\nactual_data = data\ninputs = actual_data\noutputs = colstack([data.income])\n";
    assert!(optimise_model(source).is_none());
}
