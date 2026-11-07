//! §06 `Counting(support = S)` is the counting measure on the integers restricted to
//! `S`, so its density is the support gate on `S ∩ ℤ`, lowered in the determiniser
//! like `Lebesgue`'s. No `builtin_logdensityof(Counting, …)` reaches a backend.

use flatppl_determinizer::determinize;

fn lowered(point: &str, support: &str) -> String {
    let src = format!(
        "x = elementof({point})\nlp = logdensityof(Counting({support}), x)\n\
         inputs = (x)\noutputs = (lp)\n"
    );
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("Counting density lowers");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert!(!pir.contains("builtin_logdensityof"), "{pir}");
    pir
}

#[test]
fn an_integer_point_needs_only_the_support() {
    let pir = lowered("integers", "interval(0, 10)");
    assert!(pir.contains("(in ") && !pir.contains("floor"), "{pir}");
}

#[test]
fn a_real_point_must_also_be_whole() {
    // §07 `equal` is not defined on reals, so the fractional part is tested with `le`.
    let pir = lowered("reals", "interval(0, 10)");
    assert!(pir.contains("floor") && pir.contains("(le "), "{pir}");
    let pir = lowered("cartpow(reals, 2)", "cartpow(interval(0, 10), 2)");
    assert!(pir.contains("floor") && pir.contains("lall"), "{pir}");
}
