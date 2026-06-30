use flatppl_determinizer::determinize;

// A two-independent-Gaussian product scored at data: logdensityof(lawof(record(...)), v)
// must lower to a SUM of two builtin_logdensityof terms, no `lawof`/`draw`/`joint` left.
#[test]
fn product_of_gaussians_lowers_to_sum_of_builtin_logdensityof() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(lawof(record(a = a, b = b)), record(a = 0.5, b = 0.5))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let out = determinize(&m).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two density terms:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// weighted(w, M): logdensityof → log(w) + logdensityof(M, v)
#[test]
fn weighted_lowers_to_log_w_plus_density() {
    let src = "\
w = 2.0
m = weighted(w, Normal(mu = 0.0, sigma = 1.0))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_logdensityof"),
        "inner density present:\n{pir}"
    );
    // log(w) is the weight term — assert the `(log ` call head, not a bare "log"
    // substring (which `builtin_logdensityof` would satisfy tautologically).
    assert!(pir.contains("(log "), "log(w) call present:\n{pir}");
    assert!(pir.contains("add"), "add(log(w), density) present:\n{pir}");
    assert!(
        !pir.contains("weighted") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// logweighted(lw, M): logdensityof → lw + logdensityof(M, v)
#[test]
fn logweighted_lowers_to_lw_plus_density() {
    let src = "\
lw = -0.5
m = logweighted(lw, Normal(mu = 0.0, sigma = 1.0))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_logdensityof"),
        "inner density present:\n{pir}"
    );
    assert!(pir.contains("add"), "add(lw, density) present:\n{pir}");
    assert!(
        !pir.contains("logweighted") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// superpose(M1, M2): logdensityof → logsumexp(density(M1,v), density(M2,v))
#[test]
fn superpose_lowers_to_logsumexp_of_densities() {
    let src = "\
m = superpose(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(pir.contains("logsumexp"), "logsumexp present:\n{pir}");
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two inner density terms:\n{pir}"
    );
    assert!(
        !pir.contains("superpose") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// normalize(M) where M is ALREADY a probability measure: Z = 1, logZ = 0, so
// logdensityof lowers to the identity — just logdensityof(M, v). Crucially NO
// `totalmass` is emitted (it is OUT of FlatPDL), and the result is genuinely
// conformant.
#[test]
fn normalize_of_probability_measure_lowers_to_identity_density() {
    let src = "\
m = normalize(Normal(mu = 0.0, sigma = 1.0))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_logdensityof"),
        "inner density present:\n{pir}"
    );
    // `totalmass` must NOT survive — it is a measure-query op, OUT of FlatPDL.
    assert!(
        !pir.contains("totalmass"),
        "totalmass must not be emitted:\n{pir}"
    );
    // Check the normalize combinator op itself is gone — use "(normalize " to avoid
    // matching the "%normalized" mass annotation that appears in FlatPIR %meta types.
    assert!(
        !pir.contains("(normalize ") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// normalize(M) of an UNNORMALIZED measure has no closed-form mass rule in this
// MVP. The determiniser must REFUSE rather than emit `totalmass`.
#[test]
fn normalize_of_unnormalized_measure_refuses() {
    let src = "\
w = 2.0
inner = weighted(w, Normal(mu = 0.0, sigma = 1.0))
m = normalize(inner)
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let err = determinize(&m).expect_err("unnormalized normalize must refuse, not lower");
    assert_eq!(
        err.construct, "normalize",
        "refusal names normalize: {err:?}"
    );
    assert!(
        err.reason.contains("closed-form mass rule") && err.reason.contains("totalmass"),
        "refusal explains the missing mass rule: {err:?}"
    );
}

// truncate(M, S): logdensityof → ifelse(in(v, S), density(M, v), neg(inf)).
// The gate is the `_ in R` membership builtin (FlatPIR head `in`), which infers
// to a boolean — NOT `elementof` (a set-valued param-decl that would type to
// %deferred as a 2-arg call).
#[test]
fn truncate_lowers_to_ifelse_with_in_gate() {
    let src = "\
S = interval(0.0, 1.0)
m = truncate(Normal(mu = 0.0, sigma = 1.0), S)
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(pir.contains("ifelse"), "ifelse present:\n{pir}");
    // The membership gate is `(in v S)`, a boolean — and NOT `elementof`.
    assert!(
        pir.contains("(in "),
        "boolean `in` membership gate present:\n{pir}"
    );
    assert!(
        !pir.contains("elementof"),
        "no ill-typed elementof gate:\n{pir}"
    );
    assert!(
        pir.contains("builtin_logdensityof"),
        "inner density present:\n{pir}"
    );
    assert!(pir.contains("neg"), "neg(inf) present:\n{pir}");
    assert!(
        !pir.contains("truncate") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// pushfwd(bijection(exp, log, identity), M): logdensityof → density(M, log(v)) - identity(log(v))
#[test]
fn pushfwd_bijection_lowers_to_sub_density_logvol() {
    let src = "\
bij = bijection(exp, log, identity)
m = pushfwd(bij, Normal(mu = 0.0, sigma = 1.0))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_logdensityof"),
        "inner density present:\n{pir}"
    );
    assert!(pir.contains("sub"), "sub(density, logvol) present:\n{pir}");
    assert!(
        !pir.contains("pushfwd") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// kchain(M, K) with a DISCRETE-FINITE latent (Bernoulli, 2 atoms) marginalizes
// to the mass-weighted logsumexp:
//   logsumexpᵢ[ logdensityof(M, aᵢ) + logdensityof(K(aᵢ), v) ]
// For a 2-atom Bernoulli latent and a 1-component Normal kernel that means:
//   - one outer `logsumexp` with 2 arguments,
//   - 2 mass terms (the latent's log-pmf at 0 and at 1) + 2 kernel terms = 4
//     `builtin_logdensityof` calls total,
//   - the `−logN` uniform/biased-MC form is NOT used (each branch carries the
//     latent's own mass term), and
//   - no `kchain` / `lawof` / `draw` / `kernelof` survives.
#[test]
fn kchain_discrete_bernoulli_latent_lowers_to_mass_weighted_logsumexp() {
    let src = "\
z = draw(Bernoulli(p = 0.3))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(pir.contains("logsumexp"), "outer logsumexp present:\n{pir}");
    // 2 mass terms + 2 kernel terms over the 2 Bernoulli atoms.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        4,
        "mass-weighted: 2 atoms × (latent pmf + kernel density):\n{pir}"
    );
    // Each branch adds a mass term to a kernel term.
    assert!(pir.contains("add"), "mass-weighted add per branch:\n{pir}");
    assert!(
        !pir.contains("kchain")
            && !pir.contains("lawof")
            && !pir.contains("(draw ")
            && !pir.contains("kernelof"),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    determinize(&m).expect("must lower, not refuse")
}
