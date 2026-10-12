use flatppl_determinizer::{determinize, is_flatpdl};

#[test]
fn exact_gamma_tail_and_full_support_masses_need_no_cdf() {
    for (base, lower) in [
        ("Gamma(1, rate)", "1"),
        ("Gamma(2, rate)", "lower"),
        ("Gamma(5, rate)", "2"),
        ("Gamma(0.01, rate)", "0"),
        ("InverseGamma(0.4, rate)", "0"),
    ] {
        let source = format!(
            "rate = external(posreals)\nlower = external(reals)\n\
             inputs = (rate, lower)\n\
             outputs = logdensityof(normalize(truncate({base}, interval({lower}, inf))), 4)"
        );
        let mut model = flatppl_syntax::parse(&source).unwrap();
        let _ = flatppl_infer::infer(&mut model);
        let lowered = determinize(&model).unwrap();
        is_flatpdl(&lowered).unwrap();
        let source = flatppl_syntax::print(&lowered);
        assert!(!source.contains("builtin_touniform"), "{base}: {source}");
    }
}
