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
