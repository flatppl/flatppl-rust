use flatppl_determinizer::determinize;

mod common;
use common::pir_binding;

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

// The positional→keyword mapping must also cover §06 *fundamental measures*
// (`Dirac`/`Lebesgue`/`Counting`), which are NOT in the §08 distribution
// catalogue — so `constructor_param_names` (not `distribution_param_names`)
// must resolve `Dirac`'s ordered param `["value"]`. `Dirac(0)` (positional)
// therefore binds `value=0` and lowers to the identical FlatPDL as the keyword
// form `Dirac(value = 0)`. Regression for buffy #246 (positional Dirac was
// refused because the distribution catalogue has no Dirac row).
#[test]
fn positional_dirac_constructor_lowers_same_as_keyword() {
    let positional = "\
a = draw(Dirac(0.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.0))";
    let keyword = "\
a = draw(Dirac(value = 0.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.0))";
    let pir_pos = flatppl_flatpir::write(&determinize_src(positional));
    let pir_kw = flatppl_flatpir::write(&determinize_src(keyword));
    assert!(
        pir_pos.contains("builtin_logdensityof") && pir_pos.contains("(record (%field value 0.0))"),
        "positional Dirac binds its single §06 param value:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword Dirac lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
}

// §04 auto-splatting: a multi-output function whose body is a record
// (`gamma_shape_rate(μ,σ) = record(shape = …, rate = …)`) called as the sole
// positional argument to a constructor whose params match those fields
// (`Gamma(gamma_shape_rate(…))`) must distribute the record's fields across the
// constructor's params — NOT bind the whole record to `shape` and drop `rate`.
// The record arrives as an opaque CALL (not a literal record) at build time,
// so each field is pulled with `get(arg, "field")`; canon Pass 2
// (`inline_user_calls`) then beta-reduces the call into a literal record and
// Pass 3 (`flatten_structural`) resolves the resulting static `get` to the
// literal field value directly — no residual `get` accessor survives.
// Regression for buffy #247 (the splat wasn't firing → emitted `record(shape =
// gamma_shape_rate(…))` with `rate` missing) AND re-baselined for buffy #263
// Pass 3 (this test used to pin the unresolved `get(call, "field")` shape;
// that shape is now flattened away, not weakened — both literal values still
// land on the correct params).
#[test]
fn multi_output_record_call_auto_splats_into_constructor() {
    let src = "\
gamma_shape_rate(mu, sigma) = record(shape = mu, rate = sigma)
a = draw(Gamma(gamma_shape_rate(2.0, 1.0)))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    // Both constructor params are bound to their resolved literal values —
    // `rate` is not dropped.
    assert!(
        pir.contains("(%field shape 2.0)") && pir.contains("(%field rate 1.0)"),
        "both Gamma params bound to their resolved literal values after auto-splat + flatten (rate not dropped):\n{pir}"
    );
    // Pass 3 has flattened the splat's `get(call, "field")` accessors to the
    // literal fields directly — no residual `get` remains.
    assert!(
        !pir.contains("(get "),
        "canon Pass 3 (flatten_structural) resolves the splat's static get accessor:\n{pir}"
    );
    assert!(
        !pir.contains("(%field shape (%call") && !pir.contains("(%field shape (record"),
        "shape must NOT hold the whole record (the pre-fix bug):\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// §04 auto-splat applies at ANY arity: a positional record whose field names
// match the callable's parameter names splats — including a single-param
// callable. `Dirac`'s sole `value` param: `Dirac(record(value = 5.0))` splats to
// `Dirac(value = 5.0)` (a point mass at the SCALAR, extracted via `get` at
// build time), NOT at the record. Both engines agree — inference types the
// variate scalar and the determiniser extracts the field. The record-VALUE
// form is the keyword `Dirac(value = record(...))`, which is not a positional
// splat. Regression for the §04-literal auto-splat arity semantics AND
// re-baselined for buffy #263 Pass 3: the splat's container here is ALREADY a
// literal `record(...)`, so `flatten_structural` resolves the `get` this test
// used to pin immediately — the determinized FlatPDL shows the literal `5.0`
// directly, not a `get` accessor.
#[test]
fn positional_record_auto_splats_at_any_arity_keyword_record_is_the_value() {
    // Positional record → splat → `value` bound to the record's `value`
    // field, which canon Pass 3 flattens directly to the literal `5.0` (no
    // residual `get`); the scored variate is the scalar.
    let splat = "\
a = draw(Dirac(record(value = 5.0)))
lp = logdensityof(lawof(record(a = a)), record(a = 5.0))";
    let out_splat = determinize_src(splat);
    let pir_splat = flatppl_flatpir::write(&out_splat);
    assert!(
        pir_splat.contains("builtin_logdensityof Dirac")
            && pir_splat.contains("(%field value 5.0)"),
        "positional Dirac(record(value=v)) auto-splats to the literal value field:\n{pir_splat}"
    );
    assert!(
        !pir_splat.contains("(get "),
        "canon Pass 3 (flatten_structural) resolves the splat's static get accessor:\n{pir_splat}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out_splat).is_ok());

    // Keyword record → NOT a splat → `value` bound to the whole record.
    let value = "\
a = draw(Dirac(value = record(value = 5.0)))
lp = logdensityof(lawof(record(a = a)), record(a = record(value = 5.0)))";
    let pir_value = flatppl_flatpir::write(&determinize_src(value));
    assert!(
        pir_value.contains("builtin_logdensityof Dirac") && !pir_value.contains("(get "),
        "keyword Dirac(value = record(...)) is the record value, not a splat:\n{pir_value}"
    );
}

// A record-of-draws prior with a BIJECTION-TRANSFORMED field
// (`sigma = sqrt(sigma2)`, `sigma2` a draw) lowers the field's marginal as the
// pushforward of the inner draw's law under the transform (§06 pushfwd
// change-of-variables), NOT a refuse. `sqrt` is `pow(_, 0.5)`, a spec §06
// "Known-bijection registry" member. Regression for buffy #245: the
// record-of-draws path used to reject any field that was not a bare draw.
#[test]
fn transformed_draw_prior_field_lowers_as_pushfwd() {
    let src = "\
sigma2 ~ InverseGamma(2, 2)
sigma = sqrt(sigma2)
prior = lawof(record(sigma = sigma))
lp = logdensityof(prior, record(sigma = 1.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    // Scored as the inner InverseGamma law at the sqrt-preimage, minus the
    // change-of-variables log-volume: `sub(builtin_logdensityof(InverseGamma,
    // …, pow(1.5, 2)), logvol(pow(1.5, 2)))`.
    assert!(
        pir.contains("builtin_logdensityof") && pir.contains("InverseGamma"),
        "inner InverseGamma density present:\n{pir}"
    );
    assert!(
        pir.contains("(sub "),
        "pushfwd change-of-variables subtracts the log-volume:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("lawof"),
        "measure layer eliminated:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// The transformed-draw field composes with a DEPENDENT prior: `sigma =
// sqrt(sigma2)` and siblings whose measure references `sigma`
// (`alpha ~ Normal(0, sigma * 3)`) — the linear-regression shape. Both the
// transformed field and the dependent siblings must lower (three density
// terms), with the sibling measures referencing the pinned `sigma`. Guards the
// combined transform+dependency case flagged during the #245 investigation.
#[test]
fn dependent_and_transformed_prior_fields_lower() {
    let src = "\
sigma2 ~ InverseGamma(5, 5)
sigma = sqrt(sigma2)
alpha ~ Normal(0, sigma * 3)
beta ~ Normal(0, sigma * 3)
prior = lawof(record(alpha = alpha, beta = beta, sigma = sigma))
lp = logdensityof(prior, record(alpha = 0.55, beta = 2.34, sigma = 0.11))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    // Three density terms: two Normal (alpha, beta) + one InverseGamma (the
    // sqrt-pushfwd of sigma2).
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "alpha + beta + sigma density terms:\n{pir}"
    );
    assert!(
        pir.contains("InverseGamma") && pir.contains("Normal"),
        "both the transformed InverseGamma marginal and the dependent Normals present:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("lawof"),
        "measure layer eliminated:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// The §06 known-bijection registry, broadened to the monotone §07 elementary
// functions whose inverse is itself a built-in: each, as a transformed-draw
// prior field over a base whose support lies in the function's domain, lowers
// as a pushforward change-of-variables (`sub(logdensityof(M, f_inv(y)),
// logvol(f_inv(y)))`) rather than refusing. (Numeric correctness of each
// log-volume is verified against a scipy oracle in the testsuite; here we pin
// the structural lowering.)
#[test]
fn extended_bijection_registry_lowers_transformed_draw_priors() {
    // (forward, base with support ⊆ forward's domain).
    let cases = [
        ("log1p", "Exponential(1.0)"),     // domain (−1, ∞) ⊇ nonnegreals
        ("expm1", "Normal(0.0, 1.0)"),     // domain ℝ
        ("log10", "Exponential(1.0)"),     // domain posreals ⊇ nonnegreals a.e.
        ("logit", "Beta(2.0, 2.0)"),       // domain (0, 1)
        ("invlogit", "Normal(0.0, 1.0)"),  // domain ℝ
        ("probit", "Beta(2.0, 2.0)"),      // domain (0, 1)
        ("invprobit", "Normal(0.0, 1.0)"), // domain ℝ
        ("atan", "Normal(0.0, 1.0)"),      // domain ℝ
        ("sinh", "Normal(0.0, 1.0)"),      // domain ℝ
        ("asinh", "Normal(0.0, 1.0)"),     // domain ℝ
        ("tanh", "Normal(0.0, 1.0)"),      // domain ℝ
    ];
    for (f, base) in cases {
        let src = format!(
            "raw ~ {base}\nt = {f}(raw)\nprior = lawof(record(t = t))\n\
             lp = logdensityof(prior, record(t = 0.3))"
        );
        let out = determinize_src(&src);
        let pir = flatppl_flatpir::write(&out);
        assert!(
            pir.contains("builtin_logdensityof") && pir.contains("(sub "),
            "pushfwd({f}, {base}) must lower as a change-of-variables:\n{pir}"
        );
        assert!(
            !pir.contains("(draw ") && !pir.contains("lawof"),
            "pushfwd({f}, {base}) measure layer must be eliminated:\n{pir}"
        );
        assert!(
            flatppl_determinizer::is_flatpdl(&out).is_ok(),
            "pushfwd({f}, {base}) must be valid FlatPDL"
        );
    }
}

// Domain guard (refuse-don't-mislower): a bijection whose domain constrains the
// base support refuses when the base can fall outside it. `logit`'s domain is
// (0, 1); a `Normal` base (support ℝ) puts mass outside — lowering would
// synthesise a sub-probability measure. Must refuse.
#[test]
fn logit_prior_over_real_support_base_refuses() {
    let src = "raw ~ Normal(0.0, 1.0)\nt = logit(raw)\nprior = lawof(record(t = t))\n\
               lp = logdensityof(prior, record(t = 0.3))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let e = determinize(&m).expect_err("logit over a real-support base must refuse");
    assert_eq!(e.construct, "logit", "got: {e:?}");
    assert!(e.reason.contains("(0, 1)"), "got: {e:?}");
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

// normalize(superpose(weighted(w₁, A₁), …)) of NORMALIZED mixands with
// variate-independent scalar weights is a convex superposition: by §06 the total
// mass is additive/multiplicative, Z = Σ wᵢ · totalmass(Aᵢ) = Σ wᵢ (a closed-form
// scalar), so it lowers to the superpose density minus `log(Σ wᵢ)`, NOT a refuse.
// Weights need NOT sum to one — here 0.3 + 0.5 = 0.8 — the normalizer is the
// general `log(add w₁ w₂)`. Regression for buffy #262 (dissimilar-mixture).
#[test]
fn normalize_superpose_convex_mixture_lowers_with_sum_weights_normalizer() {
    let src = "\
m = normalize(superpose(\
weighted(0.3, Normal(mu = 0.0, sigma = 1.0)), \
weighted(0.5, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("logsumexp"),
        "mixture logsumexp present:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "one density term per mixand (Normal + Gamma):\n{pir}"
    );
    // The Z = Σ wᵢ normalizer: `sub(<superpose density>, log(add(0.3, 0.5)))`.
    // Canon Pass 1 const-folds the literal `add(0.3, 0.5)` to `0.8` (the four
    // basic arithmetic ops are IEEE-754-exact, so this is safe/bit-identical);
    // `log` itself is deliberately left unevaluated (Buffy #263 Pass 1: no
    // transcendental folding), so the normalizer surfaces as a bare `(log 0.8)`
    // — distinct from the per-mixand `(add (log 0.3) …)` terms.
    assert!(
        pir.contains("(log 0.8)"),
        "log(Σ wᵢ) additive-mass normalizer sums the weights to the folded 0.8:\n{pir}"
    );
    assert!(
        !pir.contains("normalize")
            && !pir.contains("superpose")
            && !pir.contains("weighted")
            && !pir.contains("lawof")
            && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        !pir.contains("totalmass"),
        "no totalmass query op emitted:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// The spec's canonical mixture idiom (§06 "Additive superposition"):
// `normalize(superpose(weighted(p, M1), weighted(1 - p, M2)))` with weights p and
// `1 - p`. It lowers via the SAME convex-superposition rule — the weights-sum-to-
// one case is Z = add(p, sub(1, p)) with no symbolic sum-to-one proof (the
// backend evaluates log Z → log 1 = 0). Proves the dissimilar-mixture shape
// unblocks. Regression for buffy #262.
#[test]
fn normalize_superpose_one_minus_p_mixture_idiom_lowers() {
    let src = "\
p = 0.4
m = normalize(superpose(\
weighted(p, Normal(mu = 0.0, sigma = 1.0)), \
weighted(1.0 - p, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("logsumexp"),
        "mixture logsumexp present:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "one density term per mixand:\n{pir}"
    );
    // Z = add(p, 1 - p) with p = 0.4 a literal: canon Pass 1's
    // `resolve_alias_refs` inlines the trivial alias `(%ref self p)` to `0.4`,
    // then const-fold reduces `add(0.4, sub(1.0, 0.4))` to the literal `1.0`
    // (the basic arithmetic ops are IEEE-754-exact, so this is bit-identical
    // to evaluating it at run time) — the normalizer surfaces as a bare
    // `(log 1.0)`, distinct from the per-mixand `(add (log 0.4) …)` terms.
    assert!(
        pir.contains("(log 1.0)"),
        "log(p + (1 - p)) additive-mass normalizer folds the weights to 1.0:\n{pir}"
    );
    assert!(
        !pir.contains("normalize") && !pir.contains("superpose") && !pir.contains("weighted"),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// Recognizer boundary (refuse-don't-mislower): a superposition of BARE
// (unweighted) mixands is NOT the convex-combination shape the rule handles —
// each component must be an explicit `weighted(wᵢ, Aᵢ)` so the weights `wᵢ` are
// available to form Z = Σ wᵢ. A bare `superpose(A, B)` keeps the unnormalized
// refuse. (Its Z = 2 is closed-form too, but out of the chosen scope.)
#[test]
fn normalize_superpose_bare_mixands_refuses() {
    let src = "\
m = normalize(superpose(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 1.0)))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let err = determinize(&m).expect_err("bare-mixand superpose must refuse, not lower");
    assert_eq!(
        err.construct, "normalize",
        "refusal names normalize: {err:?}"
    );
    assert!(
        err.reason.contains("closed-form mass rule"),
        "refusal explains the missing mass rule: {err:?}"
    );
}

// Recognizer boundary: a weighted mixand whose base is NOT a probability measure
// (here `Lebesgue`, locally-finite) has no unit total mass, so Z ≠ Σ wᵢ and the
// convex-superposition rule does not apply — refuse rather than mislower.
#[test]
fn normalize_superpose_non_normalized_mixand_refuses() {
    let src = "\
m = normalize(superpose(\
weighted(0.5, Lebesgue(support = reals)), \
weighted(0.5, Normal(mu = 0.0, sigma = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let err = determinize(&m).expect_err("non-normalized mixand must refuse, not lower");
    assert_eq!(
        err.construct, "normalize",
        "refusal names normalize: {err:?}"
    );
}

// The scored VALUE of a record-variate density may be a NAMED binding referring
// to a record literal (`theta = record(...)`, a `Ref(SelfMod, theta)`), not an
// inline `record(...)`. `match_independent_record` resolves one ref level (as the
// measure side does) so the ref form lowers the same as the inline form — no
// "value must be a record" refuse. Regression for buffy #264 (surfaced verifying
// #262: dissimilar-mixture's `theta = record(...)` posterior score).
#[test]
fn record_variate_score_value_by_ref_lowers() {
    let src = "\
a = draw(Normal(0.0, 1.0))
b = draw(Normal(1.0, 2.0))
theta = record(a = 0.5, b = 0.5)
lp = logdensityof(lawof(record(a = a, b = b)), theta)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "both record components lower through the ref-valued theta:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// Same ref-valued-score fix on the keyword-`joint` value path
// (`lower_keyword_joint`): a `joint(x = …, y = …)` scored at a `theta = record(x
// = …, y = …)` ref lowers, not "joint value must be a record". Regression for
// buffy #264.
#[test]
fn keyword_joint_score_value_by_ref_lowers() {
    let src = "\
theta = record(x = 0.5, y = 1.0)
lp = logdensityof(joint(x = Normal(0.0, 1.0), y = Normal(1.0, 2.0)), theta)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "both joint components lower through the ref-valued theta:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
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
    // f_inv preimage (y - 2)/3, applied at the literal query point y = 0.5, is
    // now beta-reduced AND const-folded to the literal -0.5 (Buffy #263 Pass 2
    // inlines the residual `%call` that used to carry `divide(sub(_x_, 2.0), 3.0)`
    // unapplied; const-fold then reduces the folded arithmetic to one literal):
    assert!(
        p.contains("(builtin_logdensityof Normal") && p.contains(") -0.5)"),
        "f_inv(0.5) = (0.5 - 2)/3 = -0.5, inlined + folded:\n{p}"
    );
    // logvol = log|3| = log(abs(3.0)) — a constant, unaffected by inlining:
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

// The control for `refuse.rs::duplicate_draw_across_fields_refuses`: two
// INDEPENDENT draws are k = n = 2 with a full-rank (identity) Jacobian, so the
// density exists and is the product of marginals. This must keep lowering — the
// distinctness guard has to distinguish this from the duplicate case, which before
// the fix produced byte-identical FlatPDL.
#[test]
fn distinct_draws_still_lower_to_sum() {
    let src = "\
y1 = draw(Normal(mu = 0.0, sigma = 1.0))
y2 = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = y1, b = y2)), record(a = 0.5, b = 0.25))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let out = determinize(&m).expect("distinct draws must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two independent density terms:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// §06 "Engine contract for pushfwd density evaluation" case 1 mandates a registry
// including "affine maps composed from add/sub/neg/mul/divide (with positive
// scaling)". `invert.rs::derive_chain` already implements it, but the record-field
// guard's unary-only shape test could never route a BINARY call to it, so this
// spelling refused while the explicit `pushfwd(x -> 2.0*x + 1.0, Normal(0,1))`
// spelling lowered correctly — two spellings §06 calls equivalent.
#[test]
fn affine_transformed_field_lowers() {
    let src = "\
x = draw(Normal(mu = 0.0, sigma = 1.0))
y = 2.0 * x + 1.0
lp = logdensityof(lawof(record(y = y)), record(y = 2.0))";
    let m = {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    };
    let out = determinize(&m).expect("registry-mandated affine map must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term:\n{pir}"
    );
    // The change-of-variables Jacobian, pinned BY VALUE. Do not weaken this to
    // `contains("log")`: the string `builtin_logdensityof` itself contains `log`,
    // so such an assertion is satisfied by the very term it is meant to be
    // distinguished from, and a regression dropping the Jacobian would still leave
    // one `builtin_logdensityof`, no `lawof`, and valid FlatPDL. `(abs 2.0)` is
    // reachable only from the log-volume of the scale 2.0 (`logvol = log|c|`).
    let lp = pir_binding(&pir, "lp");
    assert!(
        lp.contains("(sub ") && lp.contains("(log ") && lp.contains("(abs 2.0)"),
        "density minus the log-volume log|2| — the §06 change of variables:\n{lp}"
    );
    // The point the inner Normal is scored at: f_inv(2.0) = (2.0 − 1.0)/2 = 0.5,
    // const-folded. Together with log|2| this is exactly the expression the
    // independent Julia oracle validated at -1.737085713764618.
    assert!(
        lp.contains(" 0.5)"),
        "inner density evaluated at the preimage 0.5:\n{lp}"
    );
    assert!(!pir.contains("lawof"), "measure layer gone:\n{pir}");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());

    // §06 declares this the same measure as the explicit spelling, so the two must
    // emit the identical scored expression.
    let pushfwd_spelling = "\
b = pushfwd(x -> 2.0 * x + 1.0, Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(b, 2.0)";
    let pir_pf = {
        let mut m = flatppl_syntax::parse(pushfwd_spelling).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let out = determinize(&m).expect("explicit pushfwd spelling must lower");
        flatppl_flatpir::write(&out)
    };
    assert_eq!(
        lp,
        pir_binding(&pir_pf, "lp"),
        "the two §06-equivalent spellings must emit the same scored expression"
    );
}

// The §06 case-1 registry entry the record-field guard's unary-only shape test
// most conspicuously could not reach: the matrix-vector affine map
// `mu + lower_cholesky(cov) * _` (the MvNormal construction, §08). `add`/`mul`
// over a VECTOR draw is a binary call, so this spelling refused; now it routes to
// `invert::derive_matrix_affine` and lowers to the matrix change of variables
// `logdensityof(M, linsolve(L, y - mu)) - logabsdet(L)`. Verified numerically
// against `logpdf(MvNormal(mu, L Lᵀ), y)` (Distributions.jl): both this spelling
// and the explicit `pushfwd(x -> L * x + mu, M)` score -2.9244728372492634.
#[test]
fn matrix_affine_transformed_field_lowers() {
    let record_spelling = "\
cov = rowstack([[4.0, 2.0], [2.0, 3.0]])
L = lower_cholesky(cov)
z = draw(MvNormal(mu = [0.0, 0.0], cov = eye(2)))
y = L * z + [1.0, 2.0]
lp = logdensityof(lawof(record(y = y)), record(y = [1.5, 2.5]))";
    let pushfwd_spelling = "\
cov = rowstack([[4.0, 2.0], [2.0, 3.0]])
L = lower_cholesky(cov)
d = pushfwd(x -> L * x + [1.0, 2.0], MvNormal(mu = [0.0, 0.0], cov = eye(2)))
lp = logdensityof(d, [1.5, 2.5])";
    let pir = {
        let mut m = flatppl_syntax::parse(record_spelling).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let out = determinize(&m).expect("registry-mandated matrix-affine map must lower");
        assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
        flatppl_flatpir::write(&out)
    };
    assert!(
        pir.contains("(linsolve") && pir.contains("(logabsdet"),
        "matrix change of variables f_inv = linsolve(L, y - mu), logvol = logabsdet(L):\n{pir}"
    );
    assert!(!pir.contains("lawof"), "measure layer gone:\n{pir}");

    // §06 declares the two spellings the same measure, so the scored binding must
    // be the identical expression, not merely a numerically equal one.
    let pir_pf = {
        let mut m = flatppl_syntax::parse(pushfwd_spelling).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let out = determinize(&m).expect("explicit pushfwd spelling must lower");
        flatppl_flatpir::write(&out)
    };
    assert_eq!(
        pir_binding(&pir, "lp"),
        pir_binding(&pir_pf, "lp"),
        "the two §06-equivalent spellings must emit the same scored expression"
    );
}

// §06 "Transformation and projection" prints these two programs adjacently and
// calls them the same measure, the second under "The equivalent in stochastic-node
// form is:":
//     mu = Normal(mu = 0, sigma = 1); nu = pushfwd(exp, mu)
//     mu = Normal(mu = 0, sigma = 1); x ~ mu; y = exp(x); nu = lawof(y)
// The first lowered; the second refused at the primitive-measure path, because a
// bare scalar `lawof(<derived>)` never reaches the record-field guard. Both must
// now lower, and to the SAME density.
#[test]
fn bare_lawof_of_derived_scalar_lowers_like_pushfwd() {
    let node_form = "\
mu = Normal(mu = 0.0, sigma = 1.0)
x = draw(mu)
y = exp(x)
lp = logdensityof(lawof(y), 1.6487212707001282)";
    let pushfwd_form = "\
mu = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(exp, mu)
lp = logdensityof(nu, 1.6487212707001282)";
    let lower = |src: &str| {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let out = determinize(&m).expect("both §06 spellings must lower");
        assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
        flatppl_flatpir::write(&out)
    };
    let a = lower(node_form);
    let b = lower(pushfwd_form);
    assert_eq!(
        a.matches("builtin_logdensityof").count(),
        1,
        "stochastic-node form has one density term:\n{a}"
    );
    assert!(!a.contains("lawof"), "measure layer gone:\n{a}");
    assert_eq!(
        a.matches("builtin_logdensityof").count(),
        b.matches("builtin_logdensityof").count(),
        "both §06 spellings lower to the same shape:\n{a}\n---\n{b}"
    );
    // §06 declares the two spellings the same measure, so the scored binding must
    // be the identical expression, not merely a numerically equal one — the same
    // bar `matrix_affine_transformed_field_lowers` holds the record spelling to.
    assert_eq!(
        pir_binding(&a, "lp"),
        pir_binding(&b, "lp"),
        "the two §06-equivalent spellings must emit the same scored expression"
    );
}

// §04 "Reification to measures", **Identity law**: "`lawof(draw(m))` is
// equivalent to `m`." The UNtransformed sibling of the case above — a bare
// scalar `lawof(x)` over a `~`-bound draw — reached the same primitive-measure
// refusal, because the law of a scalar draw is not itself a constructor call.
// It must score as `m` directly, with no change-of-variables term.
#[test]
fn bare_lawof_of_draw_scores_the_drawn_measure() {
    let src = "\
mu = Normal(mu = 0.0, sigma = 1.0)
x = draw(mu)
lp = logdensityof(lawof(x), 0.5)";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("§04 identity law: lawof(draw(m)) is m");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term, the drawn measure's own:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    // The identity law adds no volume element: `m` is scored at the variate as-is.
    assert!(
        !pir.contains("(sub ") && !pir.contains("(log "),
        "no change of variables for an untransformed draw:\n{pir}"
    );
}

// §06 "Transformation and projection" prints the stochastic-node form with `~` and
// a NAMED `lawof` binding, not with `draw(...)` inline and `lawof` inside the
// query — so pin the spec's own program text as written. This is not extra
// code coverage: `~` is parser sugar for `draw(…)` and lowers to identical IR, and
// the `nu = lawof(…)` hop is covered elsewhere. What it holds is that the program
// a reader copies out of §06 lowers to the same `lp` as the `pushfwd` spelling
// printed beside it, and that the `nu` scaffold binding is swept rather than
// surviving as a measure-layer node.
#[test]
fn spec_literal_stochastic_node_form_lowers() {
    let spec_text = "\
mu = Normal(mu = 0.0, sigma = 1.0)
x ~ mu
y = exp(x)
nu = lawof(y)
lp = logdensityof(nu, 1.6487212707001282)";
    let pushfwd_form = "\
mu = Normal(mu = 0.0, sigma = 1.0)
nu = pushfwd(exp, mu)
lp = logdensityof(nu, 1.6487212707001282)";
    let pir = flatppl_flatpir::write(&determinize_src(spec_text));
    let pir_pf = flatppl_flatpir::write(&determinize_src(pushfwd_form));
    assert_eq!(
        pir_binding(&pir, "lp"),
        pir_binding(&pir_pf, "lp"),
        "§06's two spellings, as the spec writes them, must emit one expression"
    );
    assert!(!pir.contains("lawof"), "measure layer gone:\n{pir}");
}

// A value reaching TWO distinct draws is refused, and that refusal is what makes
// the single-draw shape test safe: `resolve_component_draw` admits a transformed
// value only when it is a function of exactly one draw, so a coupled joint can
// never be read as a map of one of its draws. `x1 - x2` is §06 case 3 (a
// dimension-reducing map that is not a coordinate projection), where a static
// error is the default and symbolic/numeric fallbacks are explicitly optional —
// so refusing is conformant, not a gap.
#[test]
fn bare_lawof_of_a_two_draw_value_refuses() {
    let src = "\
x1 = draw(Normal(mu = 0.0, sigma = 1.0))
x2 = draw(Normal(mu = 0.0, sigma = 1.0))
d = x1 - x2
lp = logdensityof(lawof(d), 0.25)";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let err = determinize(&m).expect_err("a value reaching two draws must refuse, not mislower");
    // Asserted on the refused CONSTRUCT, not the message: this refusal's wording is
    // itself slated to be rewritten (it names a "primitive measure", which tells
    // the author of `x1 - x2` nothing), and that rewrite must not have to touch
    // this test. The construct is the two-draw expression itself.
    assert_eq!(
        err.construct, "sub",
        "the refusal must land on the two-draw expression — anywhere else means the \
         single-draw shape test was widened and something new now lowers: {} / {}",
        err.construct, err.reason
    );
}

// Holds the dispatcher's order: a constructor is read BEFORE the node is offered to
// the value law. A dependent prior's `Normal(mu = z, sigma = 1.0)` is a product the
// chain rule scores with `z` pinned, yet it also passes the value law's shape test (it
// reaches exactly one draw), so a reversed order would hand it to `pushfwd` as the map
// `x -> Normal(mu = x, sigma = 1.0)` and every dependent product would stop lowering.
#[test]
fn dependent_constructor_is_read_as_a_product_not_a_map() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(record(z = z, y = y)), record(z = 0.1, y = 0.2))";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "chain rule: p(z) * p(y | z), two terms:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.1)"),
        "y's measure keeps its dependent mu, pinned to z's scored value:\n{pir}"
    );
    assert!(
        !pir.contains("(pushfwd") && !pir.contains("(sub "),
        "a dependent product carries no change of variables:\n{pir}"
    );
}

// §04 "Reification to measures": `lawof(x)` is the **TOTAL** law of `x`, and stochastic
// ancestors "are internal stochastic nodes in the traced sub-DAG, not boundary inputs,
// so `lawof` integrates them out". So for `y ~ Normal(mu = z, sigma = 1)` with `z`
// latent, `lawof(y)` is y's MARGINAL; scoring `Normal(mu = z, …)` emits the conditional,
// a wrong number rather than a refusal. The FIXED-parameter half is the control that
// stops the marginalization becoming a blanket rule.
//
// This pair is a `CONJUGATE_TABLE` row, so it lowers to `Normal(0, √2)`. The pair with
// NO row still refuses (`implicit_marginal_golden.rs`); the row's maths is in
// `src/marginal.md`.
#[test]
fn bare_lawof_of_a_draw_with_a_latent_parameter_marginalizes() {
    let latent = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.3)";
    let lp_y = pir_binding(&flatppl_flatpir::write(&determinize_src(latent)), "lp_y");
    assert_eq!(
        lp_y.matches("builtin_logdensityof").count(),
        1,
        "the marginal is one density term, not a product over the ancestor:\n{lp_y}"
    );
    assert!(
        lp_y.contains("(%field mu 0.0)") && lp_y.contains("(%field sigma 1.4142135623730951)"),
        "the ancestor is integrated out — sqrt(σ0² + σ²) = sqrt(2):\n{lp_y}"
    );
    assert!(
        !lp_y.contains("(%field sigma 1.0)") && !lp_y.contains("(%ref self z)"),
        "not the CONDITIONAL: neither the likelihood's own sigma nor a residual ref to z:\n{lp_y}"
    );

    // Control: a fixed/parametric parameter needs no marginalization, and keeps its own
    // sigma rather than the marginal's.
    let fixed = "\
zz = elementof(reals)
y = draw(Normal(mu = zz, sigma = 1.0))
lp = logdensityof(lawof(y), 0.3)";
    let pir = flatppl_flatpir::write(&determinize_src(fixed));
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "a fixed parameter is no stochastic ancestor — must still lower:\n{pir}"
    );
    assert!(
        pir_binding(&pir, "lp").contains("(%field sigma 1.0)"),
        "a fixed parameter's law is not widened by a marginal that is not there:\n{pir}"
    );
}

// The ordering that JUSTIFIES the marginalization guard: score `lawof(y)` FIRST, then
// let a second query pin `z`. Nothing downstream refuses — by the time the
// residual-`draw` scan runs, `z` is a literal — so this emitted the conditional
// `p(y | z = 0.1)` as a finished, conformance-passing number where §04 asks for y's
// marginal.
#[test]
fn bare_lawof_scored_before_a_later_query_pins_the_latent_marginalizes() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.3)
lp_z = logdensityof(lawof(z), 0.1)";
    let lp_y = pir_binding(&flatppl_flatpir::write(&determinize_src(src)), "lp_y");
    assert!(
        lp_y.contains("(%field mu 0.0)") && lp_y.contains("(%field sigma 1.4142135623730951)"),
        "y's marginal, not the conditional at the later-pinned latent:\n{lp_y}"
    );
    assert!(
        !lp_y.contains("(%field mu 0.1)"),
        "the later query's point must not reach y's law:\n{lp_y}"
    );
}

// A measure built from ANOTHER value's law is not parameterized by that value.
// §04 "Reification to measures", *Phase of the reified law*: "the resulting measure
// is itself deterministic (of parameterized or fixed phase): `lawof` absorbs
// stochasticity into the reified law rather than propagating it outward." So in
// `y = draw(pushfwd(exp, lawof(z)))` the base consumes `z`'s LAW, not its value:
// `z` is no ancestor of `y`, and `lawof(y)` is an honest LogNormal. The
// marginalization guard must therefore stop at a `lawof` argument — walking into it
// would refuse a model that is fine, and re-split the two §06 spellings this path
// unifies.
//
// Verified numerically: every spelling below scores its LogNormal / affine /
// truncated / weighted value against Distributions.jl exactly (e.g.
// `logpdf(LogNormal(0,1), exp(0.5)) = -1.5439385332046727`).
#[test]
fn a_measure_over_another_values_law_is_not_a_stochastic_ancestor() {
    let inline = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(pushfwd(exp, lawof(z)))
lp = logdensityof(lawof(y), 1.6487212707001282)";
    let named = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
zl = lawof(z)
y = draw(pushfwd(exp, zl))
lp = logdensityof(lawof(y), 1.6487212707001282)";
    let direct = "\
lp = logdensityof(pushfwd(exp, Normal(mu = 0.0, sigma = 1.0)), 1.6487212707001282)";
    let lp_direct = pir_binding(&flatppl_flatpir::write(&determinize_src(direct)), "lp");
    for src in [inline, named] {
        let pir = flatppl_flatpir::write(&determinize_src(src));
        assert_eq!(
            pir_binding(&pir, "lp"),
            lp_direct,
            "a law-of-a-law base must score as the plain pushforward:\n{pir}"
        );
    }

    // The other combinators over `lawof(z)`, all of which the guard also refused.
    // Each marker is the computation that combinator alone contributes, so none of
    // these passes just because SOMETHING lowered: the support gate for `truncate`,
    // the affine preimage `(0.3 - 1) / 2` for `locscale`, the weight for `weighted`.
    for (src, marker) in [
        (
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(truncate(lawof(z), interval(0.0, inf)))
lp = logdensityof(lawof(y), 0.3)",
            "(in 0.3",
        ),
        (
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(locscale(lawof(z), 1.0, 2.0))
lp = logdensityof(lawof(y), 0.3)",
            "-0.35",
        ),
        (
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(weighted(0.5, lawof(z)))
lp = logdensityof(lawof(y), 0.3)",
            "(log 0.5)",
        ),
    ] {
        let pir = flatppl_flatpir::write(&determinize_src(src));
        assert!(
            pir.contains(marker),
            "combinator over lawof(z) must lower, keeping its own `{marker}` term:\n{pir}"
        );
        assert!(!pir.contains("lawof"), "measure layer gone:\n{pir}");
    }

    // But a genuinely RANDOM operand outside the `lawof` argument is a stochastic
    // ancestor and must still refuse — the walk skips only what `lawof` encloses.
    for src in [
        "\
q = draw(Uniform(a = 0.0, b = 1.0))
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(weighted(q, lawof(z)))
lp = logdensityof(lawof(y), 0.3)",
        "\
q = draw(Normal(mu = 0.0, sigma = 1.0))
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(pushfwd(x -> x + q, lawof(z)))
lp = logdensityof(lawof(y), 0.3)",
    ] {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let err = determinize(&m)
            .expect_err("a random weight / random forward map randomizes the law — refuse");
        assert!(
            err.reason
                .contains("marginalizes over a stochastic ancestor"),
            "must refuse as a marginal: {}",
            err.reason
        );
    }
}

// A value law reached as a NESTED measure must not pin its binding: the variate it
// is scored at belongs to the ENCLOSING value, a different draw. Here `lawof(z)` is
// the base of `y`'s truncation, so it is scored at `y`'s variate 0.3 — pinning `z`
// to that asserted a value for a draw the query never mentioned, and `w = z + 1.0`
// came out as the number 1.3 in conformance-passing output. `lp` itself was right,
// which is what made it silent. With the pin confined to the query's own measure,
// `z = draw(...)` stays referenced by `w` and the driver refuses the program.
#[test]
fn a_nested_value_law_does_not_pin_the_enclosing_variate() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(truncate(lawof(z), interval(0.0, inf)))
w = z + 1.0
lp = logdensityof(lawof(record(y = y)), record(y = 0.3))";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let err = determinize(&m)
        .expect_err("a live consumer of the scaffold draw must refuse, not receive a value");
    assert_eq!(
        err.construct, "draw",
        "the surviving draw is what refuses: {} / {}",
        err.construct, err.reason
    );

    // Without the live consumer, the same law scores fine and the scaffold draw is
    // swept — so the refusal above is about `w`, not about nesting a `lawof`.
    let no_consumer = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(truncate(lawof(z), interval(0.0, inf)))
lp = logdensityof(lawof(record(y = y)), record(y = 0.3))";
    let pir = flatppl_flatpir::write(&determinize_src(no_consumer));
    assert!(
        pir.contains("(ifelse") && !pir.contains("lawof"),
        "the truncated law still lowers:\n{pir}"
    );
}

// C1: the marginalization guard's measure-expression half is NOT reachable from
// `lower_value_law`. When `lawof`'s argument is a MEASURE the `lawof` is stripped and
// `build_density_term` scores the constructor, so `lawof(Normal(mu = a, sigma = 1))`
// with `a` latent emitted the conditional at whatever value a later query pinned `a`
// to. Against Distributions.jl: the emitted -1.4989385332046727
// (`logpdf(Normal(0.1, 1), exp(0.5))`) vs the marginal -1.8280121234846454
// (`logpdf(Normal(0, √2), exp(0.5))`) — wrong by 0.329 nats, conformance-passing.
//
// Both strip points must apply the total-law reading: the dispatcher's arm and
// `measure_of_arg` at the query entry (hence the direct query below, which bypasses the
// dispatcher entirely).
//
// The Normal-prior-on-a-Normal-mean pair IS a `CONJUGATE_TABLE` row, so the direct query
// lowers to `Normal(0, √2)`. The nested cases still refuse, on the §06 reference-measure
// gate over a `lawof` base — a different guard, so their reason is not pinned here.
#[test]
fn lawof_of_a_draw_parameterized_measure_marginalizes() {
    // Direct query on the stochastic law — strips at `measure_of_arg`, never reaching the
    // dispatcher's `lawof` arm. The conditional `Normal(0.1, 1)` that used to escape here
    // carried the later query's point; the marginal carries the prior's, widened.
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp_m = logdensityof(lawof(Normal(mu = a, sigma = 1.0)), 0.5)
lp_a = logdensityof(lawof(a), 0.1)";
    let lp_m = pir_binding(&flatppl_flatpir::write(&determinize_src(src)), "lp_m");
    assert!(
        lp_m.contains("(%field mu 0.0)") && lp_m.contains("(%field sigma 1.4142135623730951)"),
        "the measure expression's total law is the marginal Normal(0, sqrt 2):\n{lp_m}"
    );
    assert!(
        !lp_m.contains("(%field mu 0.1)"),
        "not the conditional at the later-pinned latent:\n{lp_m}"
    );

    // Both rows below USED to refuse, and — measured against base 01c4e48 — for DIFFERENT
    // reasons, neither of which was a rule about this shape:
    //
    //   `draw(pushfwd(exp, lawof(…)))`  refused at §06's REFERENCE-MEASURE gate on the
    //                                   `pushfwd`: "the variate does not prove a reference
    //                                   measure … so the volume element is undecided".
    //   `draw(truncate(lawof(…), S))`   refused at the VALUE-VERSUS-MEASURE discriminator
    //                                   (that row now lives in its own characterization test).
    //
    // Both traced to the pre-identity typing: `lawof(<measure>)` wrapped its argument, so
    // `pushfwd`/`truncate` of it kept a MEASURE domain and the `draw` node typed as a measure —
    // which denied the pushfwd a reference measure it could name, and tripped the discriminator
    // on the truncate. With `lawof` typed as the identity (#73), `y` types as the real it is,
    // both blockers dissolve, and the shapes reach the marginal machinery — producing the SAME
    // `Normal(0, √2)` this test's first half pins. The property the test is named for is what is
    // asserted:
    // the MARGINAL is used, never the conditional at the later-pinned latent.
    for (label, src) in [
        (
            "pushforward of the stochastic law",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(pushfwd(exp, lawof(Normal(mu = a, sigma = 1.0))))
lp_y = logdensityof(lawof(y), 1.6487212707001282)
lp_a = logdensityof(lawof(a), 0.1)",
        ),
        (
            "named, so it cannot depend on the argument being inline",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
ml = lawof(Normal(mu = a, sigma = 1.0))
y = draw(pushfwd(exp, ml))
lp_y = logdensityof(lawof(y), 1.6487212707001282)
lp_a = logdensityof(lawof(a), 0.1)",
        ),
    ] {
        let lp_y = pir_binding(&flatppl_flatpir::write(&determinize_src(src)), "lp_y");
        assert!(
            lp_y.contains("(%field sigma 1.4142135623730951)"),
            "{label}: must use the marginal Normal(0, √2):\n{lp_y}"
        );
        assert!(
            !lp_y.contains("(%field mu 0.1)"),
            "{label}: not the conditional at the later-pinned latent:\n{lp_y}"
        );
    }

    // Control: a measure argument whose parameters are FIXED is not stochastic, and
    // `lawof(lawof(z))` / `lawof(truncate(lawof(z), …))` are deterministic measures
    // built from another law — none of these may be caught.
    for src in [
        "\
p = elementof(reals)
lp = logdensityof(lawof(Normal(mu = p, sigma = 1.0)), 0.5)",
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(lawof(z)), 0.3)",
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(truncate(lawof(z), interval(0.0, inf))), 0.3)",
    ] {
        let pir = flatppl_flatpir::write(&determinize_src(src));
        assert!(
            pir.contains("builtin_logdensityof"),
            "a deterministic measure's law must still lower:\n{pir}"
        );
    }
}

// The measure-expression guard tested `Type::Measure`, and a REIFICATION types as
// `Kernel` — so `F = functionof(Normal(mu = a, sigma = 1.0))` with `a` latent slipped
// past it and `logdensityof(lawof(F), 0.5)` emitted the conditional `Normal(0.1, 1)`
// where §04 "Reification to measures" asks for the TOTAL law, the marginal
// `Normal(0, √2)` — a conformance-passing number wrong by 0.329 nats.
//
// The marginal now LOWERS in the reified spelling too. A `CONJUGATE_TABLE` row reads a
// bare distribution constructor and the wrapper is not one, so the routing unwraps a
// CLOSED reification (no boundary inputs) to its body first — §04 forbids a nullary
// callable "as this would make them equivalent to known values", so the body IS the
// reification's value. Pinned as BYTE-IDENTICAL emission against the plain spelling:
// anything less would let the two spellings drift back apart.
#[test]
fn lawof_of_a_closed_reification_lowers_identically_to_its_body() {
    for (reified, plain) in [
        // Direct query on the reification's law — the shape that emitted the number.
        (
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = functionof(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(F), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = Normal(mu = a, sigma = 1.0)
lp = logdensityof(lawof(F), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
        ),
        // Inline, so the routing cannot depend on the reification being named.
        (
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(functionof(Normal(mu = a, sigma = 1.0))), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(Normal(mu = a, sigma = 1.0)), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
        ),
        // NESTED wrappers unwrap to a fixpoint. §04's rationale clause applies at every
        // level, so two closed layers mean the body just as one does, and the routing
        // must not stop after the first. Before the fixpoint this refused.
        (
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = functionof(functionof(Normal(mu = a, sigma = 1.0)))
lp = logdensityof(lawof(F), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = Normal(mu = a, sigma = 1.0)
lp = logdensityof(lawof(F), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
        ),
        // The `kernelof(Normal(…))` row that used to sit here is GONE: §04 says
        // `kernelof`'s "`x` must not be a measure", and `infer` now rejects it, so the
        // spelling can no longer reach the determiniser to be compared against
        // anything. Its replacement is the rejection test
        // `kernelof_of_a_measure_is_a_static_error` below. A spec-legal `kernelof`
        // spelling stays covered by
        // `kernelof_of_a_value_lowers_identically_to_lawof_of_that_value`.
    ] {
        let pir_reified = flatppl_flatpir::write(&determinize_src(reified));
        let pir_plain = flatppl_flatpir::write(&determinize_src(plain));
        // The marginal is `Normal(0, √2)`, NOT the conditional `Normal(0.1, 1)` at the
        // later-pinned `a` — the 0.329-nat error this shape once emitted.
        assert!(
            pir_reified.contains("1.4142135623730951"),
            "must lower to the marginal Normal(0, √2):\n{pir_reified}"
        );
        assert_eq!(
            pir_reified, pir_plain,
            "reified and plain spellings lower to identical FlatPDL:\nreified:\n{pir_reified}\nplain:\n{pir_plain}"
        );
    }

    // A PARAMETERISED reification (one WITH boundary inputs) keeps refusing: reaching
    // its body needs a value bound to each input, which is §04 kernel-boundary
    // semantics, not an unwrapping. The message must say so rather than blame the rows.
    for src in [
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
s = elementof(posreals)
F = functionof(Normal(mu = a, sigma = s))
lp = logdensityof(lawof(F), 0.5)",
        // `kernelof` takes a VALUE (§04), so the parameterised kernel spelling wraps the
        // measure in a `draw` — §04's identity law makes that equivalent to the kernel.
        // The pre-gate spelling `kernelof(Normal(…))` is now a static error and would
        // make this a determiniser assertion about an ill-formed module.
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
s = elementof(posreals)
K = kernelof(draw(Normal(mu = a, sigma = s)))
lp = logdensityof(lawof(K), 0.5)",
    ] {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let err = determinize(&m)
            .expect_err("a parameterised reification's law is not yet lowerable — refuse");
        assert!(
            err.reason.contains("PARAMETERISED reification")
                && err.reason.contains("boundary input"),
            "must refuse naming the boundary inputs, not the conjugate rows: {}",
            err.reason
        );
    }

    // A `draw` OF the reification reaches the same conclusion by the value-side half
    // of the guard, whose reason names the ancestor rather than the law. Asserted
    // separately so neither half is credited with the other's coverage.
    for src in [
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = functionof(Normal(mu = a, sigma = 1.0))
w = draw(F)
lp = logdensityof(lawof(w), 0.5)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = functionof(Normal(mu = a, sigma = 1.0))
y = draw(pushfwd(exp, F))
lp = logdensityof(lawof(y), 1.6487212707001282)
lp_a = logdensityof(lawof(record(a = a)), record(a = 0.1))",
    ] {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let err = determinize(&m)
            .expect_err("a draw of a draw-parameterized reification must refuse too");
        assert!(
            err.reason
                .contains("marginalizes over a stochastic ancestor"),
            "must refuse as a marginal: {}",
            err.reason
        );
    }

    // Control: what makes this a guard and not a ban on `lawof(<reification>)`. A
    // reification over FIXED or literal parameters has no stochastic ancestor to
    // integrate out and must still lower to its one density term.
    for src in [
        "\
p = elementof(reals)
F = functionof(Normal(mu = p, sigma = 1.0))
lp = logdensityof(lawof(F), 0.5)",
        "\
F = functionof(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(F), 0.5)",
    ] {
        let out = determinize_src(src);
        let pir = flatppl_flatpir::write(&out);
        assert_eq!(
            pir_binding(&pir, "lp")
                .matches("builtin_logdensityof")
                .count(),
            1,
            "a reification over fixed parameters must still lower:\n{pir}"
        );
        assert!(
            flatppl_determinizer::is_flatpdl(&out).is_ok(),
            "is_flatpdl failed:\n{pir}"
        );
    }
}

// The spec-LEGAL `kernelof` spelling: §04 "Kernels and `kernelof`" reifies a VALUE
// ("`x` must not be a measure") and makes `kernelof(x)` equivalent to
// `functionof(lawof(x))`. So `kernelof(a)` scored as a measure must match `lawof(a)`,
// which §04's identity law makes `a`'s own `Normal(0, 1)`.
#[test]
fn kernelof_of_a_value_lowers_identically_to_lawof_of_that_value() {
    let reified = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(kernelof(a), 0.5)";
    let plain = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(a), 0.5)";
    let pir_reified = flatppl_flatpir::write(&determinize_src(reified));
    let pir_plain = flatppl_flatpir::write(&determinize_src(plain));
    assert!(
        pir_reified.contains("builtin_logdensityof"),
        "kernelof of a value must lower:\n{pir_reified}"
    );
    assert_eq!(
        pir_reified, pir_plain,
        "kernelof(a) and lawof(a) lower to identical FlatPDL:\nkernelof:\n{pir_reified}\nlawof:\n{pir_plain}"
    );
}

// The unwrap is CLOSED-only, and "closed" is not the same as "boundary lists nothing".
// A `%autoinputs` reification whose body holds a free `%local` placeholder auto-traces
// to ZERO inputs — a `%local` is never an `elementof` leaf — yet the placeholder is
// bound by nothing once the wrapper is gone. §04 "Placeholders and holes" forbids the
// module ("All placeholders must appear both in the expression to be reified and the
// boundary input keyword arguments"), and `infer` now rejects it at the front door
// (`reification_type`'s undeclared-placeholder error). This guard stays the backstop for
// a caller that reaches the determiniser without acting on inference's diagnostics, and
// must refuse rather than score a dangling ref.
//
// The placeholder sits in SCALE position deliberately. In `mu` position the conjugate
// row and the pre-existing reified-measure path emit the same dangling ref either way,
// so the guard would not be observable; in `sigma` position the row fires on the
// unwrapped body and would emit
// `(%field sigma (sqrt (add 1.0 (pow (%ref %local _v_) 2.0))))` — which is what this
// test catches. Verified discriminating by mutation, not assumed.
#[test]
fn a_closed_reification_whose_body_holds_a_free_placeholder_is_not_unwrapped() {
    let pir = "(%module\n\
      (%bind a (draw (Normal (%kwarg mu 0.0) (%kwarg sigma 1.0))))\n\
      (%bind F (functionof (Normal (%kwarg mu (%ref self a)) (%kwarg sigma (%ref %local _v_))) %autoinputs %deferred))\n\
      (%bind lp (logdensityof (lawof (%ref self F)) 0.5)))";
    let mut m = flatppl_flatpir::read(pir).expect("probe FlatPIR must parse");
    let _ = flatppl_infer::infer(&mut m);
    match determinize(&m) {
        Ok(out) => panic!(
            "must refuse rather than emit a dangling placeholder:\n{}",
            flatppl_flatpir::write(&out)
        ),
        Err(e) => assert!(
            e.reason.contains("placeholder"),
            "refusal must name the placeholder, not the boundary inputs it does not have: {}",
            e.reason
        ),
    }
}

// §04 "Kernels and `kernelof`": "`kernelof(x, kwargs...)` is equivalent to
// `functionof(lawof(x), kwargs...)` […]" — the elided tail scopes what the inner `lawof`
// marginalizes over to the subgraph the `kwargs` delimit, which is vacuous for the
// boundary-less spellings here. So `M = kernelof(a)` and `M = functionof(lawof(a))` are
// one expression in two spellings, and `logdensityof(lawof(M), 0.5)` is one request
// either way. The `functionof(lawof(a))` spelling lowered while `kernelof(a)` refused:
// both wrappers are `Kernel`-typed, but only the `kernelof` one made the
// measure-expression guard admit the WRAPPER and then run the draw walk on `a`'s own
// `draw` — a VALUE. The guard now reads the unwrapped body, so the spellings agree.
//
// Pinned BYTE-IDENTICAL (same binding name in each) rather than merely both-lowering:
// anything less lets the §04 equivalence drift apart again. The scored value is
// `logpdf(Normal(0, 1), 0.5) = -1.0439385332046727` (Distributions.jl) — `a`'s own law,
// since §04 "Phase of the reified law" has the reification absorb `a`'s stochasticity
// ("`lawof` absorbs stochasticity into the reified law rather than propagating it
// outward"), leaving nothing to marginalize.
//
// The outer `lawof` takes a MEASURE, which §04 states no rule for ("`lawof` reifies a
// value node"). That admissibility question is settled separately, and either way it
// applies to both spellings alike, so it cannot pull them apart.
#[test]
fn lawof_of_a_closed_kernelof_lowers_identically_to_its_section_04_equivalent() {
    let mut emitted: Vec<String> = Vec::new();
    for reifier in ["kernelof(a)", "functionof(lawof(a))"] {
        let src = format!(
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
M = {reifier}
lp = logdensityof(lawof(M), 0.5)"
        );
        let pir = flatppl_flatpir::write(&determinize_src(&src));
        assert!(
            !pir.contains("1.4142135623730951"),
            "nothing is marginalized here, so no widened sigma:\n{pir}"
        );
        emitted.push(pir);
    }
    assert_eq!(
        emitted[0], emitted[1],
        "§04 makes `kernelof(a)` and `functionof(lawof(a))` one expression; \
         their laws must lower identically:\nkernelof:\n{}\nfunctionof(lawof):\n{}",
        emitted[0], emitted[1]
    );

    // And scoring the closed kernel WITHOUT the outer `lawof` — legal on its own terms,
    // since §06 "Uniform kernel extension" identifies a closed kernel with its measure
    // and `logdensityof` "require[s] closed measures (i.e. nullary kernels) as inputs" —
    // reaches the same term. This is the pair the split was measured on.
    let direct = flatppl_flatpir::write(&determinize_src(
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
M = kernelof(a)
lp = logdensityof(M, 0.5)",
    ));
    assert_eq!(
        direct, emitted[0],
        "`logdensityof(kernelof(a), v)` and `logdensityof(lawof(kernelof(a)), v)` are one \
         request:\ndirect:\n{direct}\nvia lawof:\n{}",
        emitted[0]
    );
    assert!(
        direct.contains("builtin_logdensityof Normal")
            && direct.contains("(%field mu 0.0)")
            && direct.contains("(%field sigma 1.0)"),
        "the scored term must be `a`'s own Normal(0, 1) density at 0.5 \
         (-1.0439385332046727, Distributions.jl):\n{direct}"
    );
}

// §04's `kernelof` equivalence holds for ANY `x`, not just a bare `draw`, so the fix
// generalizes past the roster pair above and these classes must be pinned too. Each row
// is `lawof(kernelof(b))` against the plain `lawof(b)`, BYTE-IDENTICAL — the `kernelof` is
// written inline so both modules carry the same bindings and the whole emission compares.
//
// All three refused at `b79517a`, the first two through the MARGINAL-law fall-through and
// the third through the value-versus-measure arm. Each value is Distributions.jl:
//
// * `b = 2.0 * a`  — affine pushforward, `logpdf(Normal(0, 2), 0.5) = -1.643335713764618`,
//   emitted as `logdensityof(Normal(0, 1), 0.25) - log(abs(2.0))`.
// * `b = exp(a)`   — `logpdf(LogNormal(0, 1), 0.5) = -0.46601785960382813`, emitted with
//   §06's support gate on `posreals`.
// * `b = draw(Normal(mu = a, sigma = 1.0))` — a MARGINAL body, so `lawof` does integrate
//   `a` out here: `logpdf(Normal(0, √2), 0.5) = -1.3280121234846454`.
//
// The third row is the one that shows the guard still marginalizes when the body calls for
// it: reading the unwrapped body suppressed nothing, it only stopped the walk from
// treating a VALUE body as a draw-parameterized measure.
#[test]
fn lawof_of_a_closed_kernelof_over_a_derived_value_matches_the_plain_spelling() {
    for (body, expected) in [
        ("2.0 * a", "(abs 2.0)"),
        ("exp(a)", "posreals"),
        (
            "draw(Normal(mu = a, sigma = 1.0))",
            "(%field sigma 1.4142135623730951)",
        ),
    ] {
        let reified = flatppl_flatpir::write(&determinize_src(&format!(
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = {body}
lp = logdensityof(lawof(kernelof(b)), 0.5)"
        )));
        let plain = flatppl_flatpir::write(&determinize_src(&format!(
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = {body}
lp = logdensityof(lawof(b), 0.5)"
        )));
        assert_eq!(
            reified, plain,
            "`lawof(kernelof(b))` must lower as `lawof(b)` for b = {body}:\nreified:\n{reified}\nplain:\n{plain}"
        );
        assert!(
            reified.contains(expected),
            "b = {body} must keep its own lowering ({expected}):\n{reified}"
        );
    }
}

// §04's `kernelof` clause on a MEASURE argument — the sibling of
// `nested_kernelof_is_a_static_error` and the same clause. This replaces the
// `kernelof(Normal(…))` row removed from
// `lawof_of_a_closed_reification_lowers_identically_to_its_body`, which used to pin that
// the ill-formed spelling at least agreed with the well-formed one it resembles. The
// diagnostic points at `functionof`, which §04 (per flatppl-design#73) names as the
// construct that DOES reify a measure node to a kernel directly.
#[test]
fn kernelof_of_a_measure_is_a_static_error() {
    let mut m = flatppl_syntax::parse(
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
K = kernelof(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(K), 0.5)",
    )
    .unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags.iter().any(|d| {
            let m = format!("{d:?}");
            m.contains("kernelof` reifies value nodes")
                && m.contains("is a measure")
                && m.contains("functionof")
        }),
        "kernelof of a measure must be a static error naming §04 and functionof: {diags:?}"
    );
    // `functionof` of the same measure stays legal — it is the construct §04 points to.
    let mut ok = flatppl_syntax::parse(
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
F = functionof(Normal(mu = a, sigma = 1.0))
lp = logdensityof(lawof(F), 0.5)",
    )
    .unwrap();
    let ok_diags = flatppl_infer::infer(&mut ok);
    assert!(
        ok_diags.is_empty(),
        "functionof of a measure must stay legal: {ok_diags:?}"
    );
}

// §04's `kernelof` clause, now ENFORCED in `infer` — this test was the placeholder for the
// flip and its predecessor's TODO named exactly this change. §04 "Kernels and `kernelof`"
// says `kernelof` "reifies (typically stochastic) value nodes" and "`x` must not be a
// measure"; a closed `kernelof(a)` IS a measure by §06 "Uniform kernel extension"
// ("identify measures with nullary kernels"), so the OUTER `kernelof` here is ill-formed.
//
// It used to infer with NO diagnostic and lower to the same term as the single-layer
// spelling — the previous wave pinned that equality defensively, to record a defensible
// value for an ill-formed module rather than to bless the spelling. `infer` now rejects it,
// so the pin becomes a rejection test and the equality it asserted is moot.
#[test]
fn nested_kernelof_is_a_static_error() {
    let mut m = flatppl_syntax::parse(
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
M = kernelof(kernelof(a))
lp = logdensityof(lawof(M), 0.5)",
    )
    .unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags.iter().any(|d| {
            let m = format!("{d:?}");
            m.contains("kernelof` reifies value nodes") && m.contains("is a kernel")
        }),
        "the outer kernelof must be a static error naming §04: {diags:?}"
    );
    // The well-formed single-layer spelling stays clean — the gate rejects the extra
    // layer, not `kernelof` of a value.
    let mut ok = flatppl_syntax::parse(
        "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
M = kernelof(a)
lp = logdensityof(lawof(M), 0.5)",
    )
    .unwrap();
    let ok_diags = flatppl_infer::infer(&mut ok);
    assert!(
        ok_diags.is_empty(),
        "kernelof of a value must stay legal: {ok_diags:?}"
    );
}

// RESOLVED. This was a characterization pin recording an unnormalized density that `lawof`
// presented as a law, written explicitly to flip visibly once the gap was ruled on. The owner
// ruled: no implicit normalization — "we want to be honest here, we do not want hidden magic" —
// so `draw` of a measure whose mass is not a probability is a STATIC ERROR and the escape is the
// user writing `normalize(...)`.
//
// The rule is #73's equation read right-to-left: #73 gives `lawof(m)` = `lawof(draw(m))` and
// requires `lawof`'s argument to be `%normalized`, so a draw from an unnormalized measure has no
// law. A §04/§06 sentence follows as a design PR; the gate lives in `infer`
// (`ops::draw_mass_gate`).
//
// NOTE why this asserts the DIAGNOSTIC and not a determiniser refusal: `determinize_src` calls
// `infer` and discards its diagnostics, so the determiniser still walks an ill-formed module and
// would still emit the old density here. Through the CLI the gate stops the pipeline (exit 3).
// The gate is an infer-level static error, so `infer` is where it is pinned.
#[test]
fn draw_of_an_unnormalized_measure_is_a_static_error() {
    for (label, src) in [
        (
            "bounded truncation of a stochastic law",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
S = interval(0.0, 3.0)
y = draw(truncate(lawof(Normal(mu = a, sigma = 1.0)), S))
lp = logdensityof(lawof(y), 0.5)",
        ),
        (
            "inline set, so it cannot depend on the set being named",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(truncate(lawof(Normal(mu = a, sigma = 1.0)), interval(0.0, 3.0)))
lp = logdensityof(lawof(y), 0.5)",
        ),
        (
            "half-line, an unbounded truncation set",
            "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(truncate(lawof(Normal(mu = a, sigma = 1.0)), interval(0.0, inf)))
lp = logdensityof(lawof(y), 0.3)",
        ),
    ] {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let diags = flatppl_infer::infer(&mut m);
        assert!(
            diags.iter().any(|d| {
                let t = format!("{d:?}");
                t.contains("`draw` requires a probability measure") && t.contains("normalize(...)")
            }),
            "{label}: must be a static error naming the escape: {diags:?}"
        );
    }

    // POSITIVE CONTROL: writing the normalization explicitly is the escape, and it must still
    // reach the CDF transport path — the same `builtin_touniform` route
    // `normalize(truncate(Ctor, S))` takes elsewhere in this file. Without this, the refusals
    // above would be satisfied by a gate that rejected the whole family.
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(normalize(truncate(lawof(Normal(mu = a, sigma = 1.0)), interval(0.0, 3.0))))
lp = logdensityof(lawof(y), 0.5)";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags.is_empty(),
        "the explicit `normalize` escape must infer clean: {diags:?}"
    );
}

// §13 "Determinization": "`draw` nodes take their values from the explicit `point`,
// unless marginalized out." So wherever the point determines a draw's value —
// directly, or through an invertible transform — the draw must be PINNED, and a
// downstream consumer evaluates at the query point. Every position below preserves the
// variate, so the point reaches the draw.
//
// The boundary is pinned in BOTH directions: the pin must NOT widen to record fields
// (`a_nested_value_law_does_not_pin_the_enclosing_variate`), since a record field's
// measure describes the FIELD's draw, so a `lawof` inside it names a third value the
// point says nothing about.
#[test]
fn variate_preserving_positions_pin_the_draw_from_the_point() {
    for (src, expect) in [
        // `joint` field: `v.a` is the field measure's own variate.
        (
            "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
lp = logdensityof(joint(a = lawof(y)), record(a = 0.3))",
            "(%bind w 1.3)",
        ),
        // `truncate`'s base is scored at the same point.
        (
            "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
lp = logdensityof(truncate(lawof(y), interval(0.0, inf)), 0.3)",
            "(%bind w 1.3)",
        ),
        // `locscale`'s preimage is invertible, so the point determines the draw:
        // `y = (0.3 - 1.0) / 2.0 = -0.35`, hence `w = 0.65`. Behaviour-visible: without
        // this threading the program refuses.
        (
            "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
lp = logdensityof(locscale(lawof(y), 1.0, 2.0), 0.3)",
            "(%bind w 0.65)",
        ),
        // `lawof(lawof(y))` — the unwrap must carry the point through.
        (
            "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
lp = logdensityof(lawof(lawof(y)), 0.3)",
            "(%bind w 1.3)",
        ),
        // A reified body.
        (
            "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
lp = logdensityof(functionof(lawof(y)), 0.3)",
            "(%bind w 1.3)",
        ),
        // A `likelihoodof` kernel, scored at the baked observation.
        (
            "\
mu = elementof(reals)
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
L = likelihoodof(lawof(y), 0.3)
lp = logdensityof(L, record(mu = 2.0))",
            "(%bind w 1.3)",
        ),
    ] {
        let pir = flatppl_flatpir::write(&determinize_src(src));
        assert!(
            pir.contains(expect),
            "§13: the point determines this draw, so `w` evaluates at it — expected \
             `{expect}`:\n{pir}"
        );
        assert!(!pir.contains("(draw "), "no draw survives:\n{pir}");
    }

    // A pushforward's preimage is "through an invertible transform", so the draw is
    // determined too — here `y = log(exp(0.5)) = 0.5`, not swept to a placeholder.
    let pir = flatppl_flatpir::write(&determinize_src(
        "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
w = y + 1.0
lp = logdensityof(pushfwd(exp, lawof(y)), 1.6487212707001282)",
    ));
    assert!(
        pir.contains("(%bind y (%meta") && pir.contains("(log 1.6487212707001282)"),
        "y takes the invertible preimage of the point:\n{pir}"
    );
}

// An unimplemented MEASURE combinator reached as a pseudo-transform must REFUSE. The
// assertion is on the outcome, not the wording: `crate::invert` says "no analytic
// inverse", which is imprecise — `mixture` has no type rule, so nothing knows it is a
// map at all. Gating on the transform's inferred type instead ALSO refused `A * x + b`,
// which inverts correctly (`bare_matrix_affine_value_law_lowers_like_the_record_spelling`),
// because `%deferred` propagates outward from any operand. Discriminate on the head op,
// not the result type.
#[test]
fn an_unimplemented_measure_combinator_as_a_transform_refuses() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(mixture(record(w = 0.5, m = lawof(z))))
lp = logdensityof(lawof(y), 0.3)";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let err = determinize(&m).expect_err("an unimplemented measure combinator must refuse");
    assert!(
        !err.reason.is_empty(),
        "refusal carries a reason: {}",
        err.reason
    );
}

// The bare and record spellings of one matrix-affine value law must lower identically.
// `A * x + b` types as `%deferred` at the `add` (`mul(matrix, vector)` has no type rule
// and deferredness propagates outward), yet `crate::invert`'s matrix-affine grammar
// handles it and emits `linsolve`/`logabsdet`. A guard reading that deferred RESULT type
// as "unreadable map" refused the bare spelling while the record one kept lowering. Same
// bar `bare_lawof_of_derived_scalar_lowers_like_pushfwd` holds for the scalar case.
#[test]
fn bare_matrix_affine_value_law_lowers_like_the_record_spelling() {
    let prelude = "\
A = [[2.0, 0.0], [0.0, 3.0]]
b = [1.0, 1.0]
x = draw(MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]]))
y = A * x + b
";
    let bare = flatppl_flatpir::write(&determinize_src(&format!(
        "{prelude}lp = logdensityof(lawof(y), [1.0, 1.0])"
    )));
    let record = flatppl_flatpir::write(&determinize_src(&format!(
        "{prelude}lp = logdensityof(lawof(record(y = y)), record(y = [1.0, 1.0]))"
    )));
    let lp = pir_binding(&bare, "lp");
    assert!(
        lp.contains("linsolve") && lp.contains("logabsdet"),
        "the matrix-affine inverse and its log-volume are emitted:\n{lp}"
    );
    assert_eq!(
        lp,
        pir_binding(&record, "lp"),
        "bare and record spellings of one measure must lower identically:\n{bare}\n---\n{record}"
    );
}

// `lower_value_law` sits on the measure dispatcher's FALLTHROUGH, so a bare stochastic
// value serves in ANY measure position. The two tests below pin that composition for
// `weighted` and `superpose`; the combinator tests above all use a primitive constructor
// as the inner measure, so nothing else would notice if the composition went away.
//
// `weighted(w, lawof(y))`: §06 `weighted` gives `log(w) + logdensityof(lawof(y), v)`,
// where the inner term is `y`'s own draw scored at the query's variate.
#[test]
fn weighted_over_a_value_law_lowers_to_log_w_plus_the_value_density() {
    let src = "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(weighted(0.5, lawof(y)), 0.3)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    let lp = pir_binding(&pir, "lp");
    // `(log 0.5)` as a call head, not a bare "log" substring, which
    // `builtin_logdensityof` would satisfy tautologically.
    assert!(
        lp.contains("(add ") && lp.contains("(log 0.5)"),
        "log(w) added to the inner density:\n{lp}"
    );
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        1,
        "exactly one inner density — the value law's own draw:\n{lp}"
    );
    // The inner density is scored at the QUERY's variate. A value law that pinned its
    // draw and then scored somewhere else is the failure this rules out.
    assert!(
        lp.contains(" 0.3)"),
        "the inner density is scored at the query's variate:\n{lp}"
    );
    assert!(
        !pir.contains("weighted") && !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// `superpose(lawof(y), Normal(...))`: a value law and an ordinary primitive measure
// mixed in one §06 measure sum. Both mixands are scored at the query's variate and
// combined with `logsumexp` over a vector, the same shape
// `superpose_lowers_to_logsumexp_of_densities` requires of two constructors.
#[test]
fn superpose_of_a_value_law_and_a_constructor_lowers_to_logsumexp() {
    let src = "\
y = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(superpose(lawof(y), Normal(mu = 1.0, sigma = 1.0)), 0.3)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    let lp = pir_binding(&pir, "lp");
    assert!(
        lp.contains("(logsumexp (%meta ((%array"),
        "logsumexp must take a single vector (array-typed) argument, not variadic \
         scalars (§07):\n{lp}"
    );
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        2,
        "one density per mixand — the value law and the constructor:\n{lp}"
    );
    // The mixands keep their OWN parameters. Collapsing both onto one kernel input
    // would still emit two density terms and still pass `is_flatpdl`, so assert the
    // two `mu`s separately rather than trusting the term count.
    assert!(
        lp.contains("(%field mu 0.0)") && lp.contains("(%field mu 1.0)"),
        "each mixand keeps its own parameters:\n{lp}"
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

// The second net for the same §04 rule, at the pre-existing reified-measure path.
// `density::lower_reified_measure` screened a `%specinputs` boundary for placeholder
// ENTRIES, which cannot see this shape: an `%autoinputs` boundary lists `elementof`
// leaves only, so a placeholder in the body is declared NOWHERE and the screen read the
// entry list as clean. At `482d26f` this exact module inferred with zero diagnostics and
// scored `builtin_logdensityof(Normal, record(mu = _v_, sigma = 1.0), 0.5)` — a dangling
// `%local` ref in a density.
//
// `infer` is the front door now and rejects the module for any caller that acts on
// diagnostics. This test is the caller that does not, so it still reaches the screen.
// Verified discriminating by mutation, three ways: with the screen disabled the module
// refuses anyway, but only on the `%failed` type inference leaves behind, with the bare
// reason "undeclared placeholder" and no §04 citation; with the infer error disabled
// instead, this screen alone produces the refusal below; with BOTH disabled the dangling
// `(%ref %local _v_)` is scored again. The placeholder sits in `mu` position — where the
// free-`%local` guard on the UNWRAP path is invisible, which is why that guard did not
// already cover it.
#[test]
fn an_autoinputs_reification_reaching_an_undeclared_placeholder_refuses_to_score() {
    let pir = "(%module\n\
      (%bind F (functionof (Normal (%kwarg mu (%ref %local _v_)) (%kwarg sigma 1.0)) \
      %autoinputs %deferred))\n\
      (%bind lp (logdensityof (lawof (%ref self F)) 0.5)))";
    let mut m = flatppl_flatpir::read(pir).expect("probe FlatPIR must parse");
    let _ = flatppl_infer::infer(&mut m);
    match determinize(&m) {
        Ok(out) => panic!(
            "must refuse rather than score a dangling placeholder:\n{}",
            flatppl_flatpir::write(&out)
        ),
        Err(e) => {
            assert!(
                e.reason
                    .contains("placeholder that no boundary input declares"),
                "the refusal must name the undeclared placeholder as the obstacle: {}",
                e.reason
            );
            assert!(
                e.reason.contains("§04"),
                "and cite the rule it enforces: {}",
                e.reason
            );
        }
    }
}
