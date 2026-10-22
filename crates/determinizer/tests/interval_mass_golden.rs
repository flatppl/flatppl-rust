use flatppl_determinizer::{determinize, is_flatpdl};

#[test]
fn undefined_interval_mass_guards_survive_serialization() {
    let source = "lower = external(reals)\ninputs = lower\n\
        outputs = logdensityof(normalize(truncate(StudentT(4), interval(lower, inf))), 1)";
    let mut model = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    let lowered = determinize(&model).unwrap();
    let pir = flatppl_flatpir::write(&lowered);
    let mut reread = flatppl_flatpir::read(&pir).unwrap();
    let diagnostics = flatppl_infer::infer(&mut reread);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let source = flatppl_syntax::print(&lowered);
    let mut reparsed = flatppl_syntax::parse(&source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut reparsed);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

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

#[test]
fn positive_count_normalization_survives_callback_and_text_round_trip() {
    let source = "p = external(cartpow(posreals, 2))\ninputs = p\n\
        K(a, b) = normalize(truncate(NegativeBinomial(a, b), interval(1, inf)))\n\
        outputs = (logdensityof(K(p[1], p[2]), 2),\n\
          logdensityof(normalize(truncate(Poisson(p[1]), interval(1, inf))), 0))";
    let mut model = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    let lowered = determinize(&model).unwrap();
    is_flatpdl(&lowered).unwrap();
    let source = flatppl_syntax::print(&lowered);
    let mut reparsed = flatppl_syntax::parse(&source).unwrap();
    assert!(flatppl_infer::infer(&mut reparsed).is_empty());
}

fn lower(source: &str) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    let lowered = determinize(&model).unwrap();
    is_flatpdl(&lowered).unwrap();
    let printed = flatppl_syntax::print(&lowered);
    let mut reparsed = flatppl_syntax::parse(&printed).unwrap();
    assert!(flatppl_infer::infer(&mut reparsed).is_empty(), "{printed}");
    printed
}

fn poisson_atom(k: i64) -> String {
    format!("builtin_logdensityof(Poisson, record(rate = rate), {k})")
}

#[test]
fn discrete_interval_normalizer_sums_the_atoms_on_both_closed_ends() {
    // §03 interval(lo, hi) is closed, so Z keeps the atoms at 2 and at 4.
    let source = lower(
        "rate = external(posreals)\ninputs = rate\n\
         outputs = logdensityof(normalize(truncate(Poisson(rate), interval(2, 4))), 3)",
    );
    assert!(!source.contains("builtin_touniform"), "{source}");
    for k in 2..=4 {
        assert!(
            source.contains(&poisson_atom(k)),
            "atom {k} missing: {source}"
        );
    }
    for k in [1, 5] {
        assert!(
            !source.contains(&poisson_atom(k)),
            "atom {k} is outside: {source}"
        );
    }
}

#[test]
fn discrete_upper_tail_normalizer_removes_the_atoms_below_the_lower_end() {
    // interval(2.5, inf) keeps 3, 4, ...: Z = 1 - P(0) - P(1) - P(2).
    let source = lower(
        "rate = external(posreals)\ninputs = rate\n\
         outputs = logdensityof(normalize(truncate(Poisson(rate), interval(2.5, inf))), 3)",
    );
    assert!(source.contains("expm1"), "{source}");
    for k in 0..=2 {
        assert!(
            source.contains(&poisson_atom(k)),
            "atom {k} missing: {source}"
        );
    }
    assert!(!source.contains(&poisson_atom(3)), "{source}");
}

#[test]
fn discrete_truncate_masses_reach_a_mixture_normalizer() {
    let source = lower(
        "rate = external(posreals)\ninputs = rate\n\
         outputs = logdensityof(normalize(superpose(\
           weighted(0.3, truncate(Poisson(rate), interval(0, 5))),\
           weighted(0.7, truncate(Binomial(n = 10, p = 0.3), interval(3, 10))))), 4)",
    );
    assert!(
        source.contains(&poisson_atom(0)) && source.contains(&poisson_atom(5)),
        "{source}"
    );
}

#[test]
fn a_truncated_integer_law_is_gated_at_an_integer_witness() {
    // A Categorical kernel accepts only integer observations.
    let source = lower(
        "outputs = logdensityof(normalize(truncate(Categorical(p = [0.2, 0.3, 0.5]), \
         interval(1, 2))), 1)",
    );
    assert!(
        source.contains("ifelse(1 in interval(1, 2), 1, 1)"),
        "{source}"
    );
}
