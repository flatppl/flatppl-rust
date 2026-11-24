use flatppl_cli::{Format, read_module, write_module};
use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{
    Binding, BindingId, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar,
};
use flatppl_syntax::Syntax;

fn formats() -> impl Iterator<Item = (&'static str, Format, Syntax)> {
    [
        ("FlatPPL Full", Format::FlatPpl, Syntax::Full),
        ("FlatPPL Minimal", Format::FlatPpl, Syntax::Minimal),
        #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
        ("FlatPIR text", Format::FlatPir, Syntax::Full),
        #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
        ("FlatPIR JSON", Format::FlatPirJson, Syntax::Full),
    ]
    .into_iter()
}

fn bind(module: &mut Module, name: &str, rhs: NodeId) {
    let public = !name.starts_with('_');
    let name = module.intern(name);
    module.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public,
        synthetic: false,
    });
}

fn binding_id(module: &Module, name: &str) -> Option<BindingId> {
    module
        .bindings()
        .find_map(|(id, binding)| (module.resolve(binding.name) == name).then_some(id))
}

fn rhs(module: &Module, name: &str) -> NodeId {
    module.binding(binding_id(module, name).unwrap()).rhs
}

fn self_ref(module: &mut Module, name: &str) -> NodeId {
    let name = module.intern(name);
    module.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }))
}

fn double_expression(module: &mut Module, mut shared: NodeId, depth: usize) -> NodeId {
    for _ in 0..depth {
        shared = build_call(module, "add", &[shared, shared]);
    }
    shared
}

fn diamond(depth: usize) -> Module {
    let mut module = Module::new();
    let first = module.alloc(Node::Lit(Scalar::Real(1.5)));
    let shared = double_expression(&mut module, first, depth);
    let offset = module.alloc(Node::Lit(Scalar::Real(3.0)));
    let total = build_call(&mut module, "add", &[shared, offset]);
    let adjusted = build_call(&mut module, "sub", &[shared, offset]);
    bind(&mut module, "total", total);
    bind(&mut module, "adjusted", adjusted);
    module
}

// Evaluate the three kinds of expressions in these fixtures. All expected
// results are exactly representable, including after the repeated doubling.
fn value(module: &Module, id: NodeId, local: Option<f64>) -> f64 {
    match module.node(id) {
        Node::Lit(Scalar::Real(value)) => *value,
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) => {
            let binding = module.binding_by_name(*name).unwrap();
            value(module, module.binding(binding).rhs, local)
        }
        Node::Ref(Ref {
            ns: RefNs::Local, ..
        }) => local.unwrap(),
        Node::Call(call) => {
            let CallHead::Builtin(head) = call.head else {
                panic!("expected an arithmetic call");
            };
            let [lhs, rhs] = &*call.args else {
                panic!("expected two operands");
            };
            let lhs = value(module, *lhs, local);
            let rhs = value(module, *rhs, local);
            match module.resolve(head) {
                "add" => lhs + rhs,
                "sub" => lhs - rhs,
                op => panic!("unexpected operation {op}"),
            }
        }
        node => panic!("unexpected expression {node:?}"),
    }
}

fn check_compact_roundtrip(formats: impl Iterator<Item = (&'static str, Format, Syntax)>) {
    let module = diamond(16);
    for (label, format, syntax) in formats {
        let text = write_module(format, &module, syntax).unwrap();
        assert!(
            text.len() < 32 * 1024,
            "{label} expanded to {} bytes",
            text.len()
        );
        let reparsed = read_module(format, &text).unwrap();
        let public: Vec<_> = reparsed
            .public_bindings()
            .map(|(_, binding)| reparsed.resolve(binding.name))
            .collect();
        assert_eq!(public, ["total", "adjusted"], "{label}");
        assert_eq!(value(&reparsed, rhs(&reparsed, "total"), None), 98_307.0);
        assert_eq!(value(&reparsed, rhs(&reparsed, "adjusted"), None), 98_301.0);
        assert_eq!(write_module(format, &reparsed, syntax).unwrap(), text);
    }
}

#[test]
fn flatppl_round_trips_a_shared_numeric_dag_compactly() {
    check_compact_roundtrip(formats().filter(|(_, format, _)| *format == Format::FlatPpl));
}

#[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
#[test]
fn flatpir_round_trips_a_shared_numeric_dag_compactly() {
    check_compact_roundtrip(formats().filter(|(_, format, _)| *format != Format::FlatPpl));
}

#[test]
fn a_support_check_can_be_shared_with_its_numeric_expression() {
    let mut module = flatppl_syntax::parse(
        "x = elementof(reals)\nscore = ifelse(in(x / 2.0, interval(0.0, 1.0)), 1.0, 0.0)",
    )
    .unwrap();
    let score = binding_id(&module, "score").unwrap();
    let check = module.binding(score).rhs;
    let shared = double_expression(&mut module, check, 12);
    module.set_binding_rhs(score, shared);
    for (label, format, syntax) in formats() {
        let text = write_module(format, &module, syntax).unwrap();
        assert!(text.len() < 32 * 1024, "{label} expanded the support check");
        let parsed = read_module(format, &text).unwrap();
        assert_eq!(
            write_module(format, &parsed, syntax).unwrap(),
            text,
            "{label}"
        );
    }
}

#[test]
fn generated_names_do_not_capture_authored_names_or_unresolved_references() {
    for (label, format, syntax) in formats() {
        let module = diamond(8);
        let text = write_module(format, &module, syntax).unwrap();
        let reparsed = read_module(format, &text).unwrap();
        let (_, generated) = reparsed
            .bindings()
            .find(|(_, binding)| !binding.public)
            .expect("the shared DAG should need a private binding");
        let generated = reparsed.resolve(generated.name);

        for declared in [true, false] {
            let mut module = module.clone();
            if declared {
                let sentinel = module.alloc(Node::Lit(Scalar::Real(7.0)));
                bind(&mut module, generated, sentinel);
            }
            let name = module.intern(generated);
            let probe = module.alloc(Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }));
            bind(&mut module, "probe", probe);

            let text = write_module(format, &module, syntax).unwrap();
            let reparsed = read_module(format, &text).unwrap();
            assert_eq!(value(&reparsed, rhs(&reparsed, "total"), None), 387.0);
            let probe = rhs(&reparsed, "probe");
            if declared {
                assert_eq!(value(&reparsed, probe, None), 7.0, "{label}");
            } else {
                assert!(binding_id(&reparsed, generated).is_none(), "{label}");
                let name = match reparsed.node(probe) {
                    Node::Const(name) if format == Format::FlatPpl => *name,
                    Node::Ref(Ref {
                        ns: RefNs::SelfMod,
                        name,
                    }) => *name,
                    node => panic!("{label} changed the unresolved expression to {node:?}"),
                };
                assert_eq!(reparsed.resolve(name), generated, "{label}");
            }
        }
    }
}

#[test]
fn shared_callable_body_stays_inside_its_input_scope() {
    let mut module = flatppl_syntax::parse("f = functionof(_x_, x = _x_)").unwrap();
    let function = binding_id(&module, "f").unwrap();
    let Node::Call(mut call) = module.node(module.binding(function).rhs).clone() else {
        panic!("expected functionof");
    };
    call.args[0] = double_expression(&mut module, call.args[0], 6);
    let body = module.alloc(Node::Call(call));
    module.set_binding_rhs(function, body);

    for (label, format, syntax) in formats() {
        let text = write_module(format, &module, syntax).unwrap();
        let reparsed = read_module(format, &text).unwrap();
        assert_eq!(reparsed.binding_count(), 1, "{label}");
        let Node::Call(call) = reparsed.node(rhs(&reparsed, "f")) else {
            panic!("expected functionof");
        };
        let Some(Inputs::Spec(inputs)) = &call.inputs else {
            panic!("expected an explicit input boundary");
        };
        let [(name, input)] = &**inputs else {
            panic!("expected a single input");
        };
        assert_eq!(reparsed.resolve(*name), "x");
        assert_eq!(input.ns, RefNs::Local);
        assert_eq!(reparsed.resolve(input.name), "_x_");
        assert_eq!(value(&reparsed, call.args[0], Some(2.0)), 128.0, "{label}");
    }
}

#[test]
fn shared_expressions_preserve_abi_tuple_entries() {
    let mut module = diamond(8);
    let leaf = module.alloc(Node::Lit(Scalar::Real(1.5)));
    let raw = double_expression(&mut module, leaf, 5);
    let other_use = build_call(&mut module, "sub", &[raw, leaf]);
    bind(&mut module, "other_use", other_use);
    let input = module.alloc(Node::Lit(Scalar::Real(4.0)));
    bind(&mut module, "named_input", input);
    let output = module.alloc(Node::Lit(Scalar::Real(5.0)));
    bind(&mut module, "named_output", output);

    for (binding, name) in [("inputs", "named_input"), ("outputs", "named_output")] {
        let named = self_ref(&mut module, name);
        // An editor can serialize an uninferred AST with a raw input entry.
        // Naming that expression would turn it into a valid ABI input.
        let signature = build_call(&mut module, "tuple", &[named, raw]);
        bind(&mut module, binding, signature);
    }

    for (label, format, syntax) in formats() {
        let text = write_module(format, &module, syntax).unwrap();
        let reparsed = read_module(format, &text).unwrap();
        assert!(reparsed.bindings().any(|(_, b)| !b.public), "{label}");
        for (binding, name) in [("inputs", "named_input"), ("outputs", "named_output")] {
            let tuple = expect_builtin_call(&reparsed, rhs(&reparsed, binding), "tuple")
                .unwrap_or_else(|| panic!("{label} changed the {binding} tuple"));
            let [named, raw] = *tuple.args else {
                panic!("{label} changed the {binding} arity");
            };
            assert!(
                matches!(reparsed.node(named), Node::Ref(Ref { ns: RefNs::SelfMod, name: symbol })
                if reparsed.resolve(*symbol) == name),
                "{label} changed the {binding} reference"
            );
            assert!(
                expect_builtin_call(&reparsed, raw, "add").is_some(),
                "{label} replaced a raw {binding} entry with a reference"
            );
            assert_eq!(value(&reparsed, raw, None), 48.0, "{label}");
        }
    }
}

#[cfg(feature = "determinize")]
#[test]
fn named_reification_cut_preserves_both_applications() {
    let mut module = flatppl_syntax::parse(
        "ambient = elementof(reals)\n\
         body = ambient + 3.0\n\
         f = functionof(body, x = ambient)\n\
         first = f(x = 2.0)\n\
         second = f(x = 5.0)",
    )
    .unwrap();
    let ambient = self_ref(&mut module, "ambient");
    let shared = double_expression(&mut module, ambient, 5);
    let offset = module.alloc(Node::Lit(Scalar::Real(3.0)));
    let body = build_call(&mut module, "add", &[shared, offset]);
    let body_binding = binding_id(&module, "body").unwrap();
    module.set_binding_rhs(body_binding, body);
    let outside = build_call(&mut module, "sub", &[shared, offset]);
    bind(&mut module, "outside", outside);

    for (label, format, syntax) in formats() {
        let text = write_module(format, &module, syntax).unwrap();
        let mut reparsed = read_module(format, &text).unwrap();
        let body = expect_builtin_call(&reparsed, rhs(&reparsed, "body"), "add").unwrap();
        assert!(
            expect_builtin_call(&reparsed, body.args[0], "add").is_some(),
            "{label} extracted the subtree read through the reification boundary"
        );
        let diagnostics = flatppl_infer::infer(&mut reparsed);
        assert!(diagnostics.is_empty(), "{label}: {diagnostics:?}");
        let roots = [reparsed.intern("first"), reparsed.intern("second")];
        let lowered = flatppl_determinizer::determinize_with_roots(
            &reparsed,
            &flatppl_infer::ModuleBundle::new(),
            Some(&roots),
        )
        .unwrap_or_else(|error| panic!("{label}: {error:?}"));
        assert_eq!(
            value(&lowered, rhs(&lowered, "first"), None),
            67.0,
            "{label}"
        );
        assert_eq!(
            value(&lowered, rhs(&lowered, "second"), None),
            163.0,
            "{label}"
        );
    }
}

#[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
#[test]
fn named_flatpir_expression_retains_type_phase_and_support() {
    use flatppl_core::ty::{Phase, ScalarType, Type, ValueSet};

    let mut module = Module::new();
    let one = module.alloc(Node::Lit(Scalar::Real(1.0)));
    let mut shared = one;
    for _ in 0..24 {
        shared = build_call(&mut module, "add", &[shared, one]);
    }
    let total = build_call(&mut module, "add", &[shared, shared]);
    bind(&mut module, "total", total);
    let ty = Type::Scalar(ScalarType::Real);
    let phase = Phase::Fixed;
    let set = ValueSet::PosReals;
    module.set_type(shared, ty.clone());
    module.set_phase(shared, phase);
    module.set_valueset(shared, set.clone());

    for (label, format, syntax) in formats().filter(|(_, format, _)| *format != Format::FlatPpl) {
        let text = write_module(format, &module, syntax).unwrap();
        let reparsed = read_module(format, &text).unwrap();
        let (_, generated) = reparsed.bindings().find(|(_, b)| !b.public).unwrap();
        assert_eq!(reparsed.type_of(generated.rhs), Some(&ty), "{label}");
        assert_eq!(reparsed.phase_of(generated.rhs), Some(phase), "{label}");
        assert_eq!(reparsed.valueset_of(generated.rhs), Some(&set), "{label}");
    }
}

#[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
#[test]
fn flatpir_can_read_a_shared_graph_with_a_deep_linear_arm() {
    let mut module = diamond(8);
    let one = module.alloc(Node::Lit(Scalar::Real(1.0)));
    let mut chain = rhs(&module, "total");
    for _ in 0..flatppl_core::DEFAULT_MAX_DEPTH {
        chain = build_call(&mut module, "add", &[chain, one]);
    }
    bind(&mut module, "chain", chain);
    for (label, format, syntax) in formats().filter(|(_, format, _)| *format != Format::FlatPpl) {
        let text = write_module(format, &module, syntax).unwrap();
        let parsed = read_module(format, &text).unwrap();
        assert_eq!(
            value(&parsed, rhs(&parsed, "chain"), None),
            387.0 + flatppl_core::DEFAULT_MAX_DEPTH as f64,
            "{label}"
        );
    }
}
