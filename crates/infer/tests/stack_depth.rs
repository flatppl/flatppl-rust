//! Graph depth does not consume the host thread's call stack.

use std::sync::Arc;

use flatppl_core::{Module, Phase, ScalarType, Type};
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

fn infer_on_small_stack(mut module: Module, bundle: ModuleBundle) -> Module {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let diags = infer_module(&mut module, &bundle, Level::Shape);
            assert!(
                !diags.iter().any(|d| d.severity == Severity::Error),
                "{diags:?}"
            );
            module
        })
        .unwrap()
        .join()
        .unwrap()
}

fn assert_result(module: &Module, name: &str, scalar: ScalarType) {
    let (_, binding) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .unwrap();
    assert_eq!(module.type_of(binding.rhs), Some(&Type::Scalar(scalar)));
    assert_eq!(module.phase_of(binding.rhs), Some(Phase::Fixed));
}

#[test]
fn deep_module_chain_infers_on_a_small_stack() {
    let count = 512;
    let mut bundle = ModuleBundle::new();
    for i in 0..count {
        let source = if i + 1 == count {
            "value = add(1, 2)\n".to_string()
        } else {
            format!(
                "dep = load_module(\"m{}.flatppl\")\nvalue = dep.value\n",
                i + 1
            )
        };
        bundle.insert(
            format!("m{i}.flatppl"),
            Arc::new(flatppl_syntax::parse(&source).unwrap()),
        );
    }
    let root =
        flatppl_syntax::parse("dep = load_module(\"m0.flatppl\")\nresult = dep.value\n").unwrap();
    let module = infer_on_small_stack(root, bundle);
    assert_result(&module, "result", ScalarType::Integer);
}

#[test]
fn nested_callable_substitutions_keep_their_contexts() {
    let mut source = "f0(x) = x + x\n".to_string();
    for i in 1..=8 {
        source.push_str(&format!("f{i}(x) = f{}(x)\n", i - 1));
    }
    source.push_str("integer = f8(1)\nreal = f8(1.5)\n");
    let module = flatppl_syntax::parse(&source).unwrap();
    let module = infer_on_small_stack(module, ModuleBundle::new());
    assert_result(&module, "integer", ScalarType::Integer);
    assert_result(&module, "real", ScalarType::Real);
}

#[test]
fn deep_inline_graph_infers_on_a_small_stack() {
    let terms = std::iter::repeat_n("1", 3000)
        .collect::<Vec<_>>()
        .join(" + ");
    let module = flatppl_syntax::parse(&format!("result = {terms}\n")).unwrap();
    let module = infer_on_small_stack(module, ModuleBundle::new());
    assert_result(&module, "result", ScalarType::Integer);
}

#[test]
fn deep_metricsum_body_infers_on_a_small_stack() {
    let terms = std::iter::repeat_n("v[.i^]", 2000)
        .collect::<Vec<_>>()
        .join(" + ");
    let module = flatppl_syntax::parse(&format!(
        "v = [1.0, 2.0, 3.0]\nresult = metricsum(eye(3), [.i^], {terms})\n"
    ))
    .unwrap();
    let module = infer_on_small_stack(module, ModuleBundle::new());
    let (_, result) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "result")
        .unwrap();
    assert_eq!(
        module.type_of(result.rhs),
        Some(&Type::Array {
            shape: Box::new([flatppl_core::Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real))
        })
    );
}

#[test]
fn shape_substitutions_stop_at_each_boundary() {
    let mut source = "a0 = elementof(cartpow(reals, 3))\n".to_string();
    for i in 1..=16 {
        source.push_str(&format!(
            "n{i} = lengthof(a{})\na{i} = zeros([n{i}])\n",
            i - 1
        ));
    }
    let boundaries = (1..=16)
        .map(|i| format!("p{i} = n{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arguments = std::iter::repeat_n("7", 16).collect::<Vec<_>>().join(", ");
    source.push_str(&format!(
        "f = functionof(zeros([n16]), {boundaries})\nresult = f({arguments})\n"
    ));
    let module = flatppl_syntax::parse(&source).unwrap();
    let module = infer_on_small_stack(module, ModuleBundle::new());
    for i in 1..=16 {
        let (_, boundary) = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == format!("n{i}"))
            .unwrap();
        assert_eq!(module.phase_of(boundary.rhs), Some(Phase::Parameterized));
    }
    let (_, result) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "result")
        .unwrap();
    let Some(Type::Array { shape, elem }) = module.type_of(result.rhs) else {
        panic!("expected an array")
    };
    assert_eq!(elem.as_ref(), &Type::Scalar(ScalarType::Real));
    // The substituted count is seven. Unknown is sound; the old extent three is not.
    assert!(
        shape.as_ref() == [flatppl_core::Dim::Dynamic]
            || shape.as_ref() == [flatppl_core::Dim::Static(7)]
    );
}
