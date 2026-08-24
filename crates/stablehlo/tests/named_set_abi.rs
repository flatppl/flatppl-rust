//! An ABI input declared over a NAMED set lowers to the same tensor as the inline
//! spelling — the four-vector idiom `fvs = cartpow(reals, 4)` / `v = elementof(fvs)`.
//!
//! §04 "Design": "Expressions are single or nested calls that bind expressions
//! (literal or by name reference) to inputs of callables" — a name reference is
//! admitted wherever the expression is. §03 "Cartesian power": "`cartpow(S, size)`
//! produces the Cartesian power of `S` with shape `size` ... `cartpow(reals, 3)`
//! represents $\mathbb{R}^3$". So `elementof(fvs)` is an $\mathbb{R}^4$ point and its
//! ABI slot is `tensor<4xf32>`.
//!
//! At `dc9f6a1` this refused with "type has no MLIR tensor form: Deferred": inference
//! never followed the `%ref` to the set binding, so the ABI argument had no type. No
//! frozen `.mlir` is added here — the assertion is that the named spelling emits text
//! IDENTICAL to the inline spelling, which the executed goldens elsewhere already pin.

use flatppl_core::Module;

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(diags.is_empty(), "infer errors: {diags:?}");
    m
}

fn emit(src: &str) -> String {
    let mut m = parse_infer(src);
    let syms: Vec<flatppl_core::Symbol> =
        ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    let d = flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse");
    flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

/// `elementof(<named cartpow>)` in the ABI: a size-4 real vector argument.
#[test]
fn named_cartpow_abi_input_is_a_rank_1_tensor() {
    let out = emit(
        "fvs = cartpow(reals, 4)\n\
         v = elementof(fvs)\n\
         mu = elementof(reals)\n\
         s = sum(v)\n\
         lp = logdensityof(Normal(mu = mu, sigma = 1.0), s)\n\
         inputs = (v, mu)\n\
         outputs = (lp)\n",
    );
    assert!(
        out.contains("func.func @logdensity(%arg0: tensor<4xf32>, %arg1: tensor<f32>)"),
        "the named four-vector must take an ABI slot of tensor<4xf32>:\n{out}"
    );
}

/// The name reference is the only difference between the two sources, so the emitted
/// MLIR must be byte-identical.
#[test]
fn named_and_inline_abi_emit_identical_mlir() {
    let body = "mu = elementof(reals)\n\
                s = sum(v)\n\
                lp = logdensityof(Normal(mu = mu, sigma = 1.0), s)\n\
                inputs = (v, mu)\n\
                outputs = (lp)\n";
    for set in ["cartpow(reals, 4)", "cartpow(reals, [2, 3])"] {
        let inline = emit(&format!("v = elementof({set})\n{body}"));
        let named = emit(&format!("s_named = {set}\nv = elementof(s_named)\n{body}"));
        assert_eq!(
            inline, named,
            "`elementof({set})` and its named form emit different MLIR"
        );
    }
}
