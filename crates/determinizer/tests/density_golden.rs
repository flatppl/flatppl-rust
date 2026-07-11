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

// A positional-arg constructor `Normal(0.0, 1.0)` is equivalent to the keyword
// form `Normal(mu = 0.0, sigma = 1.0)` (spec §04 calling conventions: positional
// args bind to the ordered parameter names). The density side must lower it —
// producing the identical FlatPDL as the keyword form — not refuse. Regression
// for buffy #143 (@logdensity path).
#[test]
fn positional_constructor_lowers_same_as_keyword() {
    let positional = "\
a = draw(Normal(0.0, 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let keyword = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let pir_pos = flatppl_flatpir::write(&determinize_src(positional));
    let pir_kw = flatppl_flatpir::write(&determinize_src(keyword));
    assert!(
        pir_pos.contains("builtin_logdensityof")
            && pir_pos.contains("(record (%field mu 0.0) (%field sigma 1.0))"),
        "positional lowers to builtin_logdensityof with the named kernel-input record:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword forms lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
}

// The positional→keyword equivalence is NOT Normal-specific: it binds positional
// args to the distribution's ordered §08 parameter names from the catalogue.
// `Gamma` has params ["shape", "rate"] (two, differently named than Normal's
// mu/sigma), so `Gamma(2.0, 3.0)` must bind `shape=2.0, rate=3.0`. Regression for
// buffy #143 (generality across distributions, @logdensity path).
#[test]
fn positional_gamma_constructor_lowers_same_as_keyword() {
    let positional = "\
a = draw(Gamma(2.0, 3.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let keyword = "\
a = draw(Gamma(shape = 2.0, rate = 3.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let pir_pos = flatppl_flatpir::write(&determinize_src(positional));
    let pir_kw = flatppl_flatpir::write(&determinize_src(keyword));
    assert!(
        pir_pos.contains("builtin_logdensityof")
            && pir_pos.contains("(record (%field shape 2.0) (%field rate 3.0))"),
        "positional Gamma binds to its ordered params shape/rate:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword Gamma lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
}

// Single-parameter arity: `Exponential` has params ["rate"], so a one-positional
// call `Exponential(1.5)` binds `rate=1.5`. Confirms the positional mapping is not
// tied to the two-parameter shape. Regression for buffy #143 (single-arg
// positional constructor, @logdensity path).
#[test]
fn positional_exponential_single_arg_lowers_same_as_keyword() {
    let positional = "\
a = draw(Exponential(1.5))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let keyword = "\
a = draw(Exponential(rate = 1.5))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let pir_pos = flatppl_flatpir::write(&determinize_src(positional));
    let pir_kw = flatppl_flatpir::write(&determinize_src(keyword));
    assert!(
        pir_pos.contains("builtin_logdensityof") && pir_pos.contains("(record (%field rate 1.5))"),
        "positional Exponential binds its single param rate:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword Exponential lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
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
    // §07 `logsumexp(v)` takes a single real VECTOR — the emitted call must wrap its
    // per-component densities in a `vector`, not pass them as variadic scalars. The
    // annotated FlatPIR of the vector form reads `(logsumexp (%meta ((%array …) …
    // (vector …)))`; a (wrong) variadic form would show a scalar-typed first arg
    // `(logsumexp (%meta ((%scalar …`.
    assert!(
        pir.contains("(logsumexp (%meta ((%array"),
        "logsumexp must take a single vector (array-typed) argument, not variadic scalars (§07):\n{pir}"
    );
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

// normalize(truncate(Ctor, S)) takes the CDF-Z transport path (`kernel_and_input`
// builds a `builtin_touniform(kernel, kernel_input, ·)` pair for the closed-form
// Z). A POSITIONAL-arg base constructor (`Normal(0.0, 1.0)`) is equivalent to the
// keyword form (spec §04 calling conventions) and must lower to the identical
// FlatPDL, not refuse. Regression for buffy gap A: `kernel_and_input` was the one
// remaining keyword-only site post-#143 (`split_kernel_constructor` positional
// support), refusing with "primitive constructor with positional args not
// supported" on a positional truncation base.
#[test]
fn normalize_truncate_positional_ctor_lowers_same_as_keyword() {
    let positional = "\
hn = normalize(truncate(Normal(0.0, 1.0), interval(0.0, inf)))
lp = logdensityof(hn, 0.5)";
    let keyword = "\
hn = normalize(truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf)))
lp = logdensityof(hn, 0.5)";
    let pir_pos = flatppl_flatpir::write(&determinize_src(positional));
    let pir_kw = flatppl_flatpir::write(&determinize_src(keyword));
    assert!(
        pir_pos.contains("builtin_touniform"),
        "CDF-Z transport present:\n{pir_pos}"
    );
    assert!(
        pir_pos.contains("builtin_logdensityof"),
        "inner density present:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword truncation bases lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
}

// The same positional≡keyword equivalence for a non-Normal constructor, in the
// eight-schools shape (`tau ~ normalize(truncate(Cauchy(0, 5), interval(0, inf)))`).
// `Cauchy` has params ["location", "scale"] (§08), differently named/ordered from
// Normal's mu/sigma — confirms the fix is not Normal-specific.
#[test]
fn normalize_truncate_positional_cauchy_lowers_same_as_keyword() {
    let positional = "\
hn = normalize(truncate(Cauchy(0.0, 5.0), interval(0.0, inf)))
lp = logdensityof(hn, 1.0)";
    let keyword = "\
hn = normalize(truncate(Cauchy(location = 0.0, scale = 5.0), interval(0.0, inf)))
lp = logdensityof(hn, 1.0)";
    let pir_pos = flatppl_flatpir::write(&determinize_src(positional));
    let pir_kw = flatppl_flatpir::write(&determinize_src(keyword));
    assert!(
        pir_pos.contains("builtin_touniform"),
        "CDF-Z transport present:\n{pir_pos}"
    );
    assert!(
        pir_pos.contains("builtin_logdensityof"),
        "inner density present:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword Cauchy truncation bases lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
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
    // §07 `logsumexp(v)` takes a single real VECTOR: the per-atom branches must be
    // wrapped in a `vector` (array-typed arg), not passed as variadic scalars.
    assert!(
        pir.contains("(logsumexp (%meta ((%array"),
        "logsumexp must take a single vector argument (§07), not variadic scalars:\n{pir}"
    );
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

// kchain(M, K) with a DISCRETE-FINITE `Categorical` latent. `Categorical(p)` is
// 1-BASED: its atoms are {1, …, n} where n is the static length of the `p`
// vector. A length-3 `p` therefore enumerates atoms {1, 2, 3} — NOT {0, 1, 2}.
// The marginal lowers to the same mass-weighted logsumexp as the Bernoulli case:
//   - one outer `logsumexp` with 3 arguments (one per atom),
//   - 3 mass terms (Categorical log-pmf at atoms 1, 2, 3) + 3 kernel terms = 6
//     `builtin_logdensityof` calls,
//   - each branch is an `add` of a mass term and a kernel term (mass-weighted,
//     not the biased `−logN` uniform form), and
//   - no `kchain` / `lawof` / `draw` / `kernelof` survives.
// The 1-based atom values must appear as the scored value of the Categorical
// mass terms (`(builtin_logdensityof Categorical … 1)`, `… 2`, `… 3`).
#[test]
fn kchain_discrete_categorical_latent_lowers_to_mass_weighted_logsumexp() {
    let src = "\
z = draw(Categorical(p = [0.2, 0.3, 0.5]))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("logsumexp").count(),
        1,
        "one outer logsumexp over the 3 atoms:\n{pir}"
    );
    assert!(
        pir.contains("(logsumexp (%meta ((%array"),
        "logsumexp must take a single vector argument (§07), not variadic scalars:\n{pir}"
    );
    // 3 mass terms + 3 kernel terms over the 3 Categorical atoms {1, 2, 3}.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        6,
        "mass-weighted: 3 atoms × (latent pmf + kernel density):\n{pir}"
    );
    // 1-based atoms: each atom is pinned into the kernel's `mu`, so the kernel
    // bodies carry `(%field mu 1)`, `(%field mu 2)`, `(%field mu 3)` — never 0.
    assert!(pir.contains("Categorical"), "Categorical mass term:\n{pir}");
    assert!(
        pir.contains("(%field mu 1)")
            && pir.contains("(%field mu 2)")
            && pir.contains("(%field mu 3)"),
        "Categorical atoms are 1-based {{1, 2, 3}}:\n{pir}"
    );
    assert!(
        !pir.contains("(%field mu 0)"),
        "1-based Categorical must not enumerate atom 0:\n{pir}"
    );
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

// kchain(M, K) with a DISCRETE-FINITE `Categorical0` latent. `Categorical0(p)` is
// the 0-BASED variant: its atoms are {0, …, n-1}. A length-3 `p` enumerates atoms
// {0, 1, 2}. This is the only structural difference from the `Categorical` case
// above (same n, same logsumexp / term-count shape), so it pins that the
// determiniser reads the 0-based offset off the constructor name, not the vector.
#[test]
fn kchain_discrete_categorical0_latent_lowers_to_zero_based_atoms() {
    let src = "\
z = draw(Categorical0(p = [0.2, 0.3, 0.5]))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("logsumexp").count(),
        1,
        "one outer logsumexp over the 3 atoms:\n{pir}"
    );
    assert!(
        pir.contains("(logsumexp (%meta ((%array"),
        "logsumexp must take a single vector argument (§07), not variadic scalars:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        6,
        "mass-weighted: 3 atoms × (latent pmf + kernel density):\n{pir}"
    );
    // 0-based atoms: each atom is pinned into the kernel's `mu`, so the kernel
    // bodies carry `(%field mu 0)`, `(%field mu 1)`, `(%field mu 2)` — never 3.
    assert!(
        pir.contains("Categorical0"),
        "Categorical0 mass term:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0)")
            && pir.contains("(%field mu 1)")
            && pir.contains("(%field mu 2)"),
        "Categorical0 atoms are 0-based {{0, 1, 2}}:\n{pir}"
    );
    assert!(
        !pir.contains("(%field mu 3)"),
        "0-based Categorical0 must not enumerate the out-of-range atom 3:\n{pir}"
    );
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

// kchain(M, K) with a DISCRETE-FINITE `Binomial` latent. `Binomial(n, p)` has
// n+1 atoms {0, 1, …, n} (inclusive of both 0 and n), read from the STATIC INT
// `n` kwarg (not a vector length). `n = 2` therefore enumerates atoms {0, 1, 2}
// — three atoms, so the same 3-branch logsumexp shape as the Categorical cases:
//   - one outer `logsumexp` with 3 arguments,
//   - 3 Binomial mass terms + 3 kernel terms = 6 `builtin_logdensityof` calls, and
//   - no residual measure layer.
// This exercises the `static_int` (rather than `static_vector_len`) atom-count
// path in the classifier.
#[test]
fn kchain_discrete_binomial_latent_lowers_to_mass_weighted_logsumexp() {
    let src = "\
z = draw(Binomial(n = 2, p = 0.5))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("logsumexp").count(),
        1,
        "one outer logsumexp over the n+1 = 3 atoms:\n{pir}"
    );
    assert!(
        pir.contains("(logsumexp (%meta ((%array"),
        "logsumexp must take a single vector argument (§07), not variadic scalars:\n{pir}"
    );
    // 3 mass terms + 3 kernel terms over the 3 Binomial atoms {0, 1, 2}.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        6,
        "mass-weighted: (n+1) atoms × (latent pmf + kernel density):\n{pir}"
    );
    // Atoms {0, …, n} inclusive: each atom is pinned into the kernel's `mu`, so
    // the kernel bodies carry `(%field mu 0)`, `(%field mu 1)`, `(%field mu 2)`.
    assert!(pir.contains("Binomial"), "Binomial mass term:\n{pir}");
    assert!(
        pir.contains("(%field mu 0)")
            && pir.contains("(%field mu 1)")
            && pir.contains("(%field mu 2)"),
        "Binomial atoms run {{0, …, n}} inclusive:\n{pir}"
    );
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

// A likelihood query `logdensityof(likelihoodof(K, obs), θ)` is handled at the
// `logdensityof` ENTRY (not via the measure-density recursion): arg2 `θ` is the
// PARAMETER point (a record), and the variate is the `obs` baked into the
// likelihood (§06 "Likelihood construction": densityof(likelihoodof(K,obs),θ) =
// pdf(κ(θ), obs)). Each θ field value is inlined into THIS query's density
// subtree by substituting `(%ref self <name>)` — so with θ = record(mu = 2.0) and
// a `mu = elementof(reals)` param, the density scores `Normal(mu = 2.0)` at the
// baked obs `0.5`, i.e. the θ value 2.0 lands in the `mu` field of the emitted
// `builtin_logdensityof`. The `elementof` param declaration is left in place
// (valid FlatPDL — an unused free param), and no `likelihoodof` / `lawof` / draw
// survives.
#[test]
fn likelihoodof_query_inlines_theta_into_density() {
    let src = "\
mu = elementof(reals)
k = Normal(mu = mu, sigma = 1.0)
L = likelihoodof(k, 0.5)
lp = logdensityof(L, record(mu = 2.0))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_logdensityof"),
        "kernel density present:\n{pir}"
    );
    // The θ value 2.0 is inlined into the mu field; the density scores at θ = 2.0,
    // NOT at the free `mu` param (which would be a `(%ref self mu)` left dangling).
    assert!(
        pir.contains("(%field mu 2.0)"),
        "θ value 2.0 inlined into the mu field:\n{pir}"
    );
    // The `elementof` param declaration remains as an unused free param — valid
    // FlatPDL, and NOT mutated per-query (each query keeps its own θ point).
    assert!(
        pir.contains("elementof"),
        "the mu param declaration is left in place:\n{pir}"
    );
    assert!(
        !pir.contains("likelihoodof") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// Keyword/record `joint(x = M1, y = M2)` (§04 example, §06 "joint and iid
// (independent products)"): the variate is a RECORD keyed by the same field
// names, and the density is the sum of each component scored at its matching
// record field — `logdensityof(joint(x = M1, y = M2), record(x = vx, y = vy))`
// = `logdensityof(M1, vx) + logdensityof(M2, vy)`. Unlike positional `joint`
// (which slices a flat `cat` vector via `get0` and so needs a scalar-component
// guard), a record field can be ANY shape — no such guard applies here.
#[test]
fn keyword_joint_lowers_to_sum_of_field_densities() {
    let src = "\
j = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Exponential(rate = 1.0))
lp = logdensityof(j, record(x = 0.5, y = 1.0))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two component densities (one per field):\n{pir}"
    );
    assert!(!pir.contains("(joint "), "no joint left:\n{pir}");
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)"),
        "x component scores Normal(mu=0,sigma=1):\n{pir}"
    );
    assert!(
        pir.contains("(%field rate 1.0)"),
        "y component scores Exponential(rate=1):\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// A keyword-joint value record MISSING one of the joint's component fields is
// malformed — refuse rather than silently drop the missing component's
// density term. Pins whichever stage actually rejects it (inference may
// reject the mismatched record shape before determinize ever sees it, or the
// determinizer's own field lookup may refuse first).
#[test]
fn keyword_joint_missing_value_field_refuses() {
    let src = "\
j = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Exponential(rate = 1.0))
lp = logdensityof(j, record(x = 0.5))";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    if diagnostics
        .iter()
        .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        // Inference itself rejects the shape-mismatched value record — pin
        // that as the actual refusal point rather than proceeding to
        // determinize (which would then be exercising an already-invalid
        // module).
        return;
    }
    let err =
        determinize(&m).expect_err("a joint value record missing a component field must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("missing field") || msg.contains("record"),
        "refusal should name the missing field / record shape: {msg}"
    );
}

// A `joint` mixing positional and keyword components is neither the
// positional `cat`-variate form nor the keyword record-variate form — refuse
// rather than guess which one was meant. Pins whichever stage actually
// rejects the mixed call (the parser/inference may already reject a call
// mixing positional args after keyword args, or the determinizer's own
// `lower_joint` dispatch may refuse first).
#[test]
fn mixed_positional_keyword_joint_refuses() {
    let src = "\
j = joint(Normal(mu = 0.0, sigma = 1.0), y = Exponential(rate = 1.0))
lp = logdensityof(j, record(x = 0.5, y = 1.0))";
    let parsed = flatppl_syntax::parse(src);
    let mut m = match parsed {
        Err(_) => return, // the parser itself rejects mixed positional/keyword args
        Ok(m) => m,
    };
    let diagnostics = flatppl_infer::infer(&mut m);
    if diagnostics
        .iter()
        .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        // Inference rejects the mixed-form joint before determinize sees it.
        return;
    }
    let err = determinize(&m).expect_err("a mixed positional/keyword joint must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("joint"),
        "refusal should name the joint construct: {msg}"
    );
}

// A keyword-joint VALUE record carrying a stray positional element mixed with
// its named fields (`record(0.9, x = 0.5, y = 1.0)`) must refuse — not
// silently drop the positional slot and score only the named fields. Mirrors
// the equivalent guard already on `match_independent_record` ("value record
// with positional args").
#[test]
fn keyword_joint_value_record_with_positional_args_refuses() {
    let src = "\
j = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Exponential(rate = 1.0))
lp = logdensityof(j, record(0.9, x = 0.5, y = 1.0))";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    if diagnostics
        .iter()
        .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        // Inference itself rejects the value record shape before determinize
        // ever sees it — pin that as the actual refusal point.
        return;
    }
    let err = determinize(&m).expect_err(
        "a joint value record with a stray positional arg must refuse, not silently drop it",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("positional"),
        "refusal should name the positional arg in the value record: {msg}"
    );
}

// Field-name matching must be truly name-based, not an accident of the value
// record's field ORDER matching the joint declaration's order. Score the same
// joint at a REORDERED value record (`y` before `x`) and assert the emitted
// FlatPIR is byte-identical to scoring the in-order record — since
// name-matching is order-independent, the two must produce exactly the same
// pairing (Normal density at 0.5, Exponential density at 1.0), not a
// positional-index regression that would swap the values.
#[test]
fn keyword_joint_matches_fields_by_name_not_order() {
    let in_order = "\
j = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Exponential(rate = 1.0))
lp = logdensityof(j, record(x = 0.5, y = 1.0))";
    let reordered = "\
j = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Exponential(rate = 1.0))
lp = logdensityof(j, record(y = 1.0, x = 0.5))";
    let pir_in_order = flatppl_flatpir::write(&determinize_src(in_order));
    let pir_reordered = flatppl_flatpir::write(&determinize_src(reordered));
    assert_eq!(
        pir_in_order, pir_reordered,
        "name-based field matching must be order-independent:\nin-order:\n{pir_in_order}\nreordered:\n{pir_reordered}"
    );
}

// The design rationale's core claim for keyword `joint` is "no scalar
// restriction — build_density_term domain-checks the component". Exercise a
// joint mixing a scalar component (`Normal`) with a NON-SCALAR component
// (`MvNormal`, vector domain) and confirm both lower to their own
// builtin_logdensityof term rather than being refused or mis-sliced.
#[test]
fn keyword_joint_lowers_non_scalar_component() {
    let src = "\
j = joint(x = Normal(mu = 0.0, sigma = 1.0), y = MvNormal(mu = [0.0, 0.0], cov = eye(2)))
lp = logdensityof(j, record(x = 0.5, y = [0.2, 0.3]))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two component densities, one scalar and one non-scalar:\n{pir}"
    );
    assert!(!pir.contains("(joint "), "no joint left:\n{pir}");
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)"),
        "x component scores Normal(mu=0,sigma=1):\n{pir}"
    );
    assert!(
        pir.contains("MvNormal"),
        "y component scores the non-scalar MvNormal:\n{pir}"
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

// A scalar draw scored at a STRUCTURED variate (record / vector) is a type
// mismatch (spec §06: the variate shape must match the data shape). Inference
// does not reject it, so the determinizer must REFUSE rather than emit an
// ill-typed builtin_logdensityof scoring a scalar Normal at a record/vector
// (refuse a definite measure-domain-vs-variate kind mismatch).
#[test]
fn scalar_draw_scored_at_record_variate_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = record(x = 0.5)))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let err = determinize(&m).expect_err("a scalar measure scored at a record variate must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("variate") || msg.contains("domain"),
        "refusal should name the variate/domain mismatch: {msg}"
    );
}

#[test]
fn scalar_draw_scored_at_vector_variate_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = [0.1, 0.2, 0.3]))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let err = determinize(&m).expect_err("a scalar measure scored at a vector variate must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("variate") || msg.contains("domain"),
        "refusal should name the variate/domain mismatch: {msg}"
    );
}

// ---------------------------------------------------------------------------
// locscale (§06 line 369/402): `locscale(m, shift, scale)` is shorthand for
// `pushfwd(x -> scale * x + shift, m)`. Density lowering reuses the affine
// change-of-variables: scalar scale → f_inv(y) = (y - shift)/scale,
// logvol = log|scale|; matrix scale (the MvNormal Cholesky case) →
// f_inv(y) = linsolve(scale, y - shift), logvol = logabsdet(scale).
// ---------------------------------------------------------------------------

fn ls_pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    flatppl_flatpir::write(&determinize(&m).expect("locscale must lower"))
}

#[test]
fn locscale_scalar_lowers() {
    // locscale(Normal(0,1), 2.0, 3.0) ≡ Normal(2, 3): the affine change of
    // variables f(x) = 3x + 2, f_inv(y) = (y - 2)/3, logvol = log|3|. Density:
    //   logdensityof(Normal(0,1), (y - 2)/3) - log|3|
    // = -½log2π - ½((y-2)/3)² - log 3  ≡  log N(y; 2, 3).
    let p =
        ls_pir("d = locscale(Normal(mu = 0.0, sigma = 1.0), 2.0, 3.0)\nlp = logdensityof(d, 0.5)");
    assert!(p.contains("builtin_logdensityof"), "got:\n{p}");
    // f_inv preimage (y - 2)/3 = divide(sub(y, 2.0), 3.0):
    assert!(
        p.contains("(divide") && p.contains("(sub (%ref %local _x_) 2.0)"),
        "f_inv = (y - 2)/3 present:\n{p}"
    );
    // logvol = log|3| = log(abs(3.0)):
    assert!(p.contains("(abs 3.0)"), "logvol log|3| present:\n{p}");
}

#[test]
fn locscale_scalar_equals_affine_pushfwd() {
    // The defining identity (§06): locscale(m, shift, scale) ==
    // pushfwd(x -> scale * x + shift, m). Byte-equal FlatPDL proves it — the two
    // surfaces must lower to the exact same change-of-variables.
    let locscale =
        ls_pir("d = locscale(Normal(mu = 0.0, sigma = 1.0), 2.0, 3.0)\nlp = logdensityof(d, 0.5)");
    let affine = ls_pir(
        "d = pushfwd(x -> 3.0 * x + 2.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert_eq!(
        locscale, affine,
        "locscale(m, shift, scale) must lower identically to pushfwd(x -> scale*x + shift, m)"
    );
}

#[test]
fn locscale_matrix_lowers() {
    // locscale(MvNormal(0, I), mu, L) with L a square (Cholesky) matrix ≡
    // MvNormal(mu, L Lᵀ). Matrix-affine change of variables:
    //   f_inv(y) = linsolve(L, y - mu),  logvol = logabsdet(L).
    let p = ls_pir(
        "cov = [[4.0, 1.0], [1.0, 3.0]]\n\
         d = locscale(MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]]), \
                      [1.0, 2.0], lower_cholesky(cov))\n\
         lp = logdensityof(d, [0.5, 0.5])",
    );
    assert!(p.contains("builtin_logdensityof"), "got:\n{p}");
    // f_inv = linsolve(L, y - mu): the preimage solve with its y - mu RHS.
    assert!(
        p.contains("(linsolve") && p.contains("(sub"),
        "f_inv = linsolve(L, y - mu) present:\n{p}"
    );
    // logvol = logabsdet(L): the constant forward log-volume.
    assert!(
        p.contains("(logabsdet"),
        "logvol = logabsdet(L) present:\n{p}"
    );
}

#[test]
fn locscale_matrix_equals_affine_pushfwd() {
    // Matrix defining identity: locscale(MvNormal(0,I), mu, L) ==
    // pushfwd(x -> L * x + mu, MvNormal(0,I)). Byte-equal FlatPDL.
    let locscale = ls_pir(
        "cov = [[4.0, 1.0], [1.0, 3.0]]\n\
         L = lower_cholesky(cov)\n\
         d = locscale(MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]]), [1.0, 2.0], L)\n\
         lp = logdensityof(d, [0.5, 0.5])",
    );
    let affine = ls_pir(
        "cov = [[4.0, 1.0], [1.0, 3.0]]\n\
         L = lower_cholesky(cov)\n\
         d = pushfwd(x -> L * x + [1.0, 2.0], MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]]))\n\
         lp = logdensityof(d, [0.5, 0.5])",
    );
    assert_eq!(
        locscale, affine,
        "matrix locscale must lower identically to pushfwd(x -> L*x + mu, m)"
    );
}

// ---------------------------------------------------------------------------
// `derive_locscale` refuse boundaries (invert.rs ~371-434). These are
// CHARACTERIZATION tests: they lock in behavior the code ALREADY refuses
// (not RED→GREEN) — the happy-path tests above never exercised the 5
// documented refuse conditions or the new `type_is_matrix` helper.
// ---------------------------------------------------------------------------

fn ls_refuse(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let err = determinize(&m).expect_err("must refuse, not lower");
    format!("{err:?}")
}

#[test]
fn locscale_scalar_variate_matrix_scale_refuses() {
    // A SCALAR-variate base (`Normal`) paired with a MATRIX `scale`: variate-
    // incompatible (a scalar variate has no matrix affine map). Must refuse
    // rather than mislower.
    let msg = ls_refuse(
        "d = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, [[1.0, 0.0], [0.0, 1.0]])\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(msg.contains("refuse"), "got: {msg}");
}

#[test]
fn locscale_vector_variate_scalar_scale_refuses() {
    // A VECTOR-variate base (`MvNormal`) paired with a SCALAR `scale`: the true
    // forward log-volume would be n*log|scale| (summed over all n axes), not
    // the scalar form's single log|scale| — the same danger the vector-variate
    // guard closes for plain `pushfwd`. Must refuse rather than mislower.
    let msg = ls_refuse(
        "d = locscale(MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]]), [1.0, 2.0], 3.0)\n\
         lp = logdensityof(d, [0.5, 0.5])",
    );
    assert!(msg.contains("refuse"), "got: {msg}");
}

#[test]
fn locscale_zero_scalar_scale_refuses() {
    // A literal-zero scalar scale collapses the forward map to the constant
    // `shift` (not injective) and makes `log|scale| = -inf`; must refuse rather
    // than synthesize a degenerate change-of-variables (mirrors the affine-`mul`
    // literal-zero guard in `classify`/`derive_chain`).
    let msg = ls_refuse(
        "d = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, 0.0)\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(msg.contains("refuse"), "got: {msg}");
}

// Unnormalized posterior: `bayesupdate(L, prior)` = `dν(θ) = L(θ)·dπ(θ)`, so
//   logdensityof(bayesupdate(L, prior), θ)
//     = logdensityof(L, θ) + logdensityof(prior, θ)   (§06 "Likelihoods and
//   posteriors": lowers to `logweighted(fn(logdensityof(L, _)), prior)`).
// This is the HMC inference target — the log-posterior up to the dropped
// (constant) evidence. Must lower to TWO builtin_logdensityof terms (L's kernel
// scored at obs, the prior scored at θ) combined with `add`, not refuse.
#[test]
fn bayesupdate_lowers_to_loglik_plus_logprior() {
    let src = "\
mu = elementof(reals)
prior = joint(mu = Normal(mu = 0.0, sigma = 1.0))
model = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)
L = likelihoodof(model, 0.5)
post = bayesupdate(L, prior)
lp = logdensityof(post, record(mu = 0.3))";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    // Two builtin_logdensityof terms (one for L's kernel, one for the prior),
    // combined with `add`.
    assert!(pir.contains("builtin_logdensityof"), "got:\n{pir}");
    assert!(
        pir.matches("builtin_logdensityof").count() >= 2,
        "loglik + logprior, got:\n{pir}"
    );
    assert!(pir.contains("(add "), "log-posterior is a sum, got:\n{pir}");
}

// Refuse-don't-mislower: a `bayesupdate` whose PRIOR cannot lower (here a prior
// that marginalizes an internal CONTINUOUS non-conjugate latent — a
// non-enumerable `kchain` marginal) must propagate that sub-lowering Err and
// refuse the whole posterior, never emit a partial density.
#[test]
fn bayesupdate_with_non_lowerable_prior_refuses() {
    let src = "\
mu = elementof(reals)
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(mu = draw(Normal(mu = 0.0, sigma = z))), z = z)
badprior = kchain(lawof(record(z = z)), k)
model = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)
L = likelihoodof(model, 0.5)
post = bayesupdate(L, badprior)
lp = logdensityof(post, record(mu = 0.3))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let err =
        determinize(&m).expect_err("a bayesupdate whose prior cannot lower must refuse, not lower");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("kchain") || msg.contains("non-enumerable"),
        "refusal should propagate the prior's sub-lowering failure: {msg}"
    );
}

// A `bayesupdate` prior that is a `lawof`-wrapped record of `~`-bound draws
// (the bi2/eight-schools shape: `prior = lawof(record(mu = mu, tau = tau, ...))`)
// must lower, not refuse. `lower_bayesupdate` hands its prior straight to
// `lower_measure_density`, whose dispatch has no `lawof` arm — unlike the
// `logdensityof(lawof(M), v)` query ENTRY point, which strips a top-level
// `lawof` before ever reaching the dispatcher. A `lawof` reaching the
// dispatcher as a SUB-measure (here, bayesupdate's prior argument) needs its
// own unwrap-and-recurse arm.
#[test]
fn bayesupdate_with_lawof_record_prior_lowers() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Exponential(rate = 1.0))
prior = lawof(record(a = a, b = b))
model = functionof(Normal(mu = a, sigma = b), a = a, b = b)
L = likelihoodof(model, 0.5)
post = bayesupdate(L, prior)
lp = logdensityof(post, record(a = 0.3, b = 1.0))";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    // prior (lawof-record) scored: a Normal + b Exponential term, plus the likelihood.
    assert!(pir.contains("builtin_logdensityof"), "got:\n{pir}");
    assert!(
        pir.matches("builtin_logdensityof").count() >= 3,
        "loglik + 2 prior fields, got:\n{pir}"
    );
}
