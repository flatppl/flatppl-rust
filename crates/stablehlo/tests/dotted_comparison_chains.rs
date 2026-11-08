//! `a .< b .<= c` over arrays emits the same program as the hand-written
//! `land.(a .< b, b .<= c)` (spec §05: dotted operators are broadcast sugar, and a
//! comparison chain lowers to `land`).

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

const INPUTS: &str = "a = elementof(cartpow(reals, 3))\nb = elementof(cartpow(reals, 3))\n\
                      c = elementof(cartpow(reals, 3))\nd = elementof(cartpow(reals, 3))\n\
                      inputs = (a, b, c, d)\n";

#[test]
fn dotted_chains_emit_the_hand_written_land_chain() {
    for (chain, explicit) in [
        ("a .< b .<= c", "land.(a .< b, b .<= c)"),
        ("a .< b .<= c .> d", "land.(land.(a .< b, b .<= c), c .> d)"),
    ] {
        assert_eq!(
            emit(&format!("{INPUTS}outputs = {chain}\n")),
            emit(&format!("{INPUTS}outputs = {explicit}\n")),
            "{chain}"
        );
    }
}
