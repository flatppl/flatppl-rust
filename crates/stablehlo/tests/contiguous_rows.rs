//! Rows that select adjacent equal-width blocks of one vector stack as one slice.

fn emit(source: &str) -> String {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots: Vec<_> = ["inputs", "outputs"].map(|name| module.intern(name)).into();
    let lowered = flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .expect("determinize");
    flatppl_stablehlo::emit(
        &lowered,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("emit")
}

fn rows(selections: &str, batched: bool) -> String {
    let (declaration, outputs) = if batched {
        ("cartpow(cartpow(reals, 6), 4)", "rows.(x)")
    } else {
        ("cartpow(reals, 6)", "rows(x)")
    };
    emit(
        &format!(
            "flatppl_compat = \"0.1\"\n\
         x = elementof({declaration})\n\
         rows(v) = broadcast(prod, [{selections}])\n\
         inputs = (x)\noutputs = {outputs}\n"
        )
        .replace("{v}", "exp.(v)"),
    )
}

/// Counts slices of the selected vector and concatenates that build the row stack.
fn row_ops(ir: &str, batched: bool) -> (usize, usize) {
    let (source, stacked) = if batched {
        ("(tensor<4x6xf32>)", "-> tensor<4x2x3xf32>")
    } else {
        ("(tensor<6xf32>)", "-> tensor<2x3xf32>")
    };
    let count = |op: &str, ty: &str| {
        ir.lines()
            .filter(|line| line.contains(op) && line.contains(ty))
            .count()
    };
    (
        count("stablehlo.slice", source),
        count("stablehlo.concatenate", stacked),
    )
}

#[test]
fn adjacent_rows_are_one_slice() {
    for batched in [false, true] {
        let ir = rows("get({v}, [1, 2, 3]), get({v}, [4, 5, 6])", batched);
        assert_eq!(row_ops(&ir, batched), (1, 0), "{ir}");
    }
}

#[test]
fn rows_out_of_order_keep_their_slices() {
    for batched in [false, true] {
        let ir = rows("get({v}, [4, 5, 6]), get({v}, [1, 2, 3])", batched);
        assert_eq!(row_ops(&ir, batched), (2, 1), "{ir}");
    }
}
