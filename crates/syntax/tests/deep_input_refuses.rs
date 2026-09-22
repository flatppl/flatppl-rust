//! Expression breadth, nesting, and the remaining constructed-tree guard.

fn nested_parens(depth: usize) -> String {
    format!("x = {}1.0{}\n", "(".repeat(depth), ")".repeat(depth))
}

#[test]
fn nesting_beyond_the_former_budget_parses() {
    // Redundant grouping adds source depth without constructing a deep tree.
    let depth = 6_000;
    let module = flatppl_syntax::parse(&nested_parens(depth)).unwrap();
    let root = module.bindings().next().unwrap().1.rhs;
    assert!(
        matches!(module.node(root), flatppl_core::Node::Lit(flatppl_core::Scalar::Real(v)) if *v == 1.0)
    );
    assert_eq!(
        module.span_of(root),
        Some(flatppl_core::Span {
            start: (depth + 4) as u32,
            end: (depth + 7) as u32
        })
    );
}

#[test]
fn nested_expression_productions_use_a_bounded_native_stack() {
    // Each wrapper introduces a different recursive edge of the grammar.
    // Its expected lowered child is checked without a recursive printer.
    let wrappers = [
        ("f(", ")", "f", 0),
        ("record(k = ", ")", "record", 0),
        ("[", "]", "vector", 0),
        ("(", ",)", "tuple", 0),
        ("x[", "]", "get", 1),
        ("-(", ")", "neg", 0),
        ("2 ^ (", ")", "pow", 1),
        ("(arg -> ", ")", "functionof", 0),
        ("fn(add(_, ", "))", "functionof", 0),
        ("base.functionof(", ", z = _z_)", "functionof", 0),
        ("f.(", ")", "broadcast", 1),
        ("(f)(", ")", "%call", 0),
    ];
    let depth = 1_800;
    let mut expr = "7".to_owned();
    for i in 0..depth {
        let (prefix, suffix, _, _) = wrappers[i % wrappers.len()];
        expr = format!("{prefix}{expr}{suffix}");
    }
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            use flatppl_core::{CallHead, Node, Scalar};
            let module = flatppl_syntax::parse(&format!("result = {expr}\n")).unwrap();
            let mut id = module.bindings().next().unwrap().1.rhs;
            for i in (0..depth).rev() {
                let (_, _, expected, child) = wrappers[i % wrappers.len()];
                let Node::Call(call) = module.node(id) else {
                    panic!("missing wrapper {i}")
                };
                match call.head {
                    CallHead::Builtin(sym) => assert_eq!(module.resolve(sym), expected),
                    CallHead::User(_) => assert_eq!(expected, "%call"),
                }
                id = if expected == "record" {
                    call.named[0].value
                } else {
                    call.args[child]
                };
                if i % wrappers.len() == 8 {
                    let Node::Call(add) = module.node(id) else {
                        panic!("missing fn body")
                    };
                    id = add.args[1];
                }
            }
            assert!(matches!(module.node(id), Node::Lit(Scalar::Int(7))));
        })
        .unwrap()
        .join()
        .unwrap();
}

fn deep_print_roundtrip(syntax: flatppl_syntax::Syntax) {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            use flatppl_core::{CallHead, Node, Scalar};
            let terms = 4_000;
            let source = format!("result = f({})", vec!["1"; terms].join(" + "));
            let module = flatppl_syntax::parse(&source).unwrap();
            let text = flatppl_syntax::print_with(&module, syntax);
            let reparsed = flatppl_syntax::parse(&text).unwrap();
            let root = reparsed.bindings().next().unwrap().1.rhs;
            let Node::Call(call) = reparsed.node(root) else {
                panic!("outer call")
            };
            let mut id = call.args[0];
            for _ in 1..terms {
                let Node::Call(add) = reparsed.node(id) else {
                    panic!("addition chain")
                };
                assert!(matches!(add.head, CallHead::Builtin(op) if reparsed.resolve(op) == "add"));
                assert!(matches!(
                    reparsed.node(add.args[1]),
                    Node::Lit(Scalar::Int(1))
                ));
                id = add.args[0];
            }
            assert!(matches!(reparsed.node(id), Node::Lit(Scalar::Int(1))));
            assert_eq!(flatppl_syntax::print_with(&reparsed, syntax), text);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn full_printing_is_stack_independent() {
    deep_print_roundtrip(flatppl_syntax::Syntax::Full);
}

#[test]
fn minimal_printing_is_stack_independent() {
    deep_print_roundtrip(flatppl_syntax::Syntax::Minimal);
}

#[test]
fn breadth_is_not_charged_as_depth() {
    // A wide array exceeds the former 262144-token cap without deep nesting.
    let elems = (0..132_000)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    assert!(
        flatppl_syntax::parse(&format!("x = [{elems}]\n")).is_ok(),
        "a wide flat array is not deep and must parse"
    );
}

fn flat_sum(terms: usize) -> String {
    format!(
        "x = {}\n",
        std::iter::repeat_n("1", terms)
            .collect::<Vec<_>>()
            .join(" + ")
    )
}

#[test]
fn a_flat_operator_chain_is_guarded_before_recursive_consumers() {
    assert!(flatppl_syntax::parse(&flat_sum(1_000)).is_ok());
    let err = flatppl_syntax::parse(&flat_sum(50_000))
        .expect_err("a left-deep constructed tree must be bounded");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("constructed expression depth") && msg.contains("4096"),
        "the refusal must name the constructed-tree limit: {msg}"
    );
}
