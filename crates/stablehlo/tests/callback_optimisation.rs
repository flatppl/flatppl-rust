use flatppl_stablehlo::{EmitError, EmitOptions, Mode, emit};

fn compile(source: &str, restrict_enzyme_compatible: bool) -> Result<String, EmitError> {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let options = EmitOptions {
        restrict_enzyme_compatible,
        ..Default::default()
    };
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let flatpdl = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .unwrap();
    emit(&flatpdl, Mode::LogDensity, &options)
}

#[test]
fn named_callback_body_reverses_only_its_cell_axis() {
    let source = "\
        a = [1.0, 2.0]\n\
        body = reverse(reverse(a))\n\
        f = functionof(body, a = a)\n\
        x = external(cartpow(cartpow(reals, 2), 3))\n\
        inputs = x\n\
        outputs = f.(x)\n";
    for restricted in [false, true] {
        let hlo = compile(source, restricted).unwrap();
        let reversals: Vec<_> = hlo
            .lines()
            .filter(|line| line.contains("stablehlo.reverse"))
            .collect();
        assert_eq!(reversals.len(), 2, "{hlo}");
        assert!(
            reversals.iter().all(|line| line.contains("dims = [1]")),
            "{hlo}"
        );
    }
}

#[test]
fn ordinary_double_reverse_still_eliminates_both_reversals() {
    let input = "x = external(cartpow(reals, 2))\ninputs = x\n";
    let original = format!("{input}outputs = reverse(reverse(x))\n");
    let expected = format!("{input}outputs = x\n");
    for restricted in [false, true] {
        assert_eq!(
            compile(&original, restricted).unwrap(),
            compile(&expected, restricted).unwrap()
        );
    }
}
