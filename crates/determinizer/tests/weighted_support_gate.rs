//! Spec §06: `weighted(f, M)` has density `f(x)` times `M`'s density, which is zero
//! outside `M`'s support, so the product is zero there whatever `f` is. A weight
//! that is NaN outside the support (a square root of a negative argument) must not
//! turn the −∞ log-density into NaN: the weight is applied inside the support gate,
//! at a point inside the support.
use flatppl_determinizer::determinize_with_roots;
use flatppl_infer::ModuleBundle;

fn lp(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let root = m.intern("lp");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[root])).expect("must lower");
    let text = flatppl_flatpir::write(&out);
    let start = text.find("(%bind lp").expect("lp binding");
    text[start..].lines().next().unwrap().to_string()
}

#[test]
fn a_weight_outside_the_support_is_not_evaluated_there() {
    for op in ["weighted", "logweighted"] {
        let outside = lp(&format!(
            "M = {op}(x -> sqrt(x - 1.0), Lebesgue(support = interval(1.0, 2.0)))\n\
             lp = logdensityof(M, 0.5)"
        ));
        assert!(
            !outside.contains("(sqrt (sub 0.5"),
            "{op}: weight read the raw point: {outside}"
        );
        assert!(
            outside.contains("(neg inf)"),
            "{op}: no -inf gate: {outside}"
        );
    }
}

#[test]
fn a_record_weight_is_gated_by_the_box_and_reads_a_point_inside_it() {
    let outside = lp(
        "S = cartprod(x = interval(1.0, 2.0), y = interval(0.0, 1.0))\n\
         M = weighted((x, y) -> sqrt(x - 1.0) + y, Lebesgue(support = S))\n\
         lp = logdensityof(M, record(x = 0.5, y = 0.25))",
    );
    assert!(
        !outside.contains("(sqrt (sub 0.5"),
        "weight read the raw point: {outside}"
    );
    assert!(outside.contains("(neg inf)"), "no -inf gate: {outside}");
}

// In a fixed-atom mixture a component is absent where its whole base density is
// −∞, so a zero mixture weight there is replaced before `log`. For an atom the
// −∞ comes from the base's support gate. Peeling that gate for the weight must
// not hide it from the guard, or `log(0)` is differentiated in the untaken arm
// and the gradient is NaN.
#[test]
fn an_atom_weight_guard_still_sees_the_support_gate() {
    let mut m = flatppl_syntax::parse(
        "a = elementof(posreals)\nb = elementof(posreals)\np = elementof(unitinterval)\n\
         x = elementof(reals)\n\
         M = superpose(weighted(0.2, Dirac(1.0)), weighted(0.8 * (1 - p), Beta(a, b)), \
         weighted(0.8 * p, Dirac(0.0)))\nlp = logdensityof(M, x)",
    )
    .unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let root = m.intern("lp");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[root])).expect("must lower");
    let text = flatppl_syntax::print(&out);
    // The Beta component's weight `0.8 * (1 - p)` sits in `log(ifelse(D <= -inf, 1.0, w))`,
    // where D must be the gated density `ifelse(cond, …, -inf)`, not the bare arm.
    let weight = text.find("0.8 * (1 - p)").expect("Beta weight");
    let guard = &text[text[..weight].rfind("log(").expect("log of the weight")..weight];
    assert_eq!(
        guard.matches("-inf").count(),
        2,
        "the Beta weight must be guarded by its gated density:\n{text}"
    );
}
