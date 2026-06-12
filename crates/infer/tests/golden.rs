//! Inference golden tests against the spec-§11 annotated FlatPIR fixtures.
//!
//! The bare and annotated fixture pairs are both spec-sourced (no engine
//! oracle): inference over the bare module must reproduce the annotated one.
//! Comparison is in canonical written form, so hand-formatting in the fixture
//! files cannot confound. `model.flatpir` needs cross-module inference for
//! its `L` binding (`load_module` is deferred until multi-file fixtures
//! exist), so it is checked binding-by-binding with the gap explicit.

use std::fs;
use std::path::PathBuf;

use flatppl_infer::{Severity, infer};

fn fixture(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../fixtures/flatpir", name]
        .iter()
        .collect();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// helpers.flatpir is single-module: inference must reproduce the annotated
/// golden exactly (canonical form).
#[test]
fn helpers_inference_matches_spec_golden() {
    let mut module = flatppl_flatpir::read(&fixture("helpers.flatpir")).unwrap();
    let diags = infer(&mut module);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");

    let annotated = flatppl_flatpir::read(&fixture("helpers-annotated.flatpir")).unwrap();
    assert_eq!(
        flatppl_flatpir::write(&module),
        flatppl_flatpir::write(&annotated),
        "inferred annotations diverge from the spec golden"
    );
}

/// model.flatpir: every single-module binding matches the golden; `L`
/// (likelihood over a cross-module kernel) is honestly `%deferred` until
/// `load_module` support lands, with a note diagnostic saying so.
#[test]
fn model_inference_single_module_part() {
    let mut module = flatppl_flatpir::read(&fixture("model.flatpir")).unwrap();
    let diags = infer(&mut module);
    let out = flatppl_flatpir::write(&module);

    for expected in [
        "(load_module (%meta %module %fixed %unknown)",
        "(elementof (%meta (%scalar real) %parameterized reals) reals)",
        "(draw (%meta (%scalar real) %stochastic reals)",
        "(Normal (%meta (%measure (%domain (%scalar real)) (%mass %normalized)) %fixed reals)",
        "(add (%meta (%scalar real) %stochastic reals)",
        "(likelihoodof (%meta %deferred %fixed %unknown)",
    ] {
        assert!(out.contains(expected), "missing `{expected}` in:\n{out}");
    }
    assert!(
        diags.iter().any(|d| d.severity == Severity::Note
            && d.message
                .contains("cross-module references are not inferred yet")),
        "expected a cross-module gap note, got: {diags:?}"
    );
}

// ---- unit tests over surface snippets ----

fn infer_src(src: &str) -> (flatppl_core::Module, Vec<flatppl_infer::Diagnostic>) {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut module);
    (module, diags)
}

fn meta_of(src: &str, needle: &str) -> String {
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    let line = out
        .lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no line containing `{needle}` in:\n{out}"));
    line.trim().to_string()
}

#[test]
fn arithmetic_promotion() {
    // int ⊔ int = int; real dominates; divide is real division.
    assert!(meta_of("x = 1 + 2", "add").contains("(%scalar integer)"));
    assert!(meta_of("x = 1 + 2.0", "add").contains("(%scalar real)"));
    assert!(meta_of("x = 1 / 2", "divide").contains("(%scalar real)"));
    assert!(meta_of("x = 1 < 2", "lt").contains("(%scalar boolean)"));
}

#[test]
fn phases_follow_the_ancestor_rule() {
    let src = "a = elementof(reals)\nb ~ Normal(0.0, 1.0)\nc = a + b\nd = 1 + 2";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(out.contains("(elementof (%meta (%scalar real) %parameterized reals) reals)"));
    assert!(out.contains("(draw (%meta (%scalar real) %stochastic reals)"));
    // c joins parameterized ⊔ stochastic = stochastic.
    assert!(
        out.contains("(add (%meta (%scalar real) %stochastic reals) (%ref self a) (%ref self b))"),
        "got:\n{out}"
    );
    assert!(out.contains("(add (%meta (%scalar integer) %fixed integers) 1 2)"));
}

#[test]
fn containers_and_access() {
    assert!(meta_of("v = [1.0, 2.0, 3.0]", "vector").contains("(%array 1 (3) (%scalar real))"));
    let src = "r = record(mu = 0.0, n = 3)\nx = r.mu";
    assert!(meta_of(src, "(get ").contains("(%scalar real)"));
    let src = "t = (1.0, true)\nx = t[2]";
    assert!(meta_of(src, "(get ").contains("(%scalar boolean)"));
}

#[test]
fn iid_static_count_shapes_the_domain() {
    let needle = "(iid";
    let meta = meta_of("x ~ iid(Normal(0.0, 1.0), 3)", needle);
    assert!(
        meta.contains("(%measure (%domain (%array 1 (3) (%scalar real))) (%mass %normalized))"),
        "got: {meta}"
    );
}

#[test]
fn reference_cycle_is_an_error() {
    let (module, diags) = infer_src("x = y + 1\ny = x + 1");
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("reference cycle")),
        "got: {diags:?}"
    );
    let out = flatppl_flatpir::write(&module);
    assert!(out.contains("%failed"), "got:\n{out}");
}

#[test]
fn unknown_op_is_an_honest_gap() {
    let (module, diags) = infer_src("x = frobnicate(1, 2)");
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Note
                && d.message.contains("no type rule for `frobnicate`")),
        "got: {diags:?}"
    );
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(frobnicate (%meta %deferred %fixed %unknown) 1 2)"),
        "got:\n{out}"
    );
}

// ---- inference levels ----

#[test]
fn level_phase_annotates_no_types() {
    let mut module = flatppl_syntax::parse("a = elementof(reals)\nb ~ Normal(a, 1.0)").unwrap();
    flatppl_infer::infer_with(&mut module, flatppl_infer::Level::Phase);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(draw (%meta %deferred %stochastic %deferred)"),
        "got:\n{out}"
    );
    assert!(
        !out.contains("%scalar"),
        "types must not be annotated:\n{out}"
    );
}

#[test]
fn level_shape_resolves_fixed_dims() {
    let src = "J = 8\nx ~ iid(Normal(0.0, 1.0), J)";
    // Type level: the computed count stays dynamic.
    let mut m = flatppl_syntax::parse(src).unwrap();
    flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Type);
    assert!(
        flatppl_flatpir::write(&m).contains("(%array 1 (%dynamic) (%scalar real))"),
        "got:\n{}",
        flatppl_flatpir::write(&m)
    );
    // Shape level: J resolves through the fixed-phase ref.
    let mut m = flatppl_syntax::parse(src).unwrap();
    flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Shape);
    assert!(
        flatppl_flatpir::write(&m).contains("(%array 1 (8) (%scalar real))"),
        "got:\n{}",
        flatppl_flatpir::write(&m)
    );
}

#[test]
fn shape_resolver_arithmetic_and_length_observer() {
    // Arithmetic over fixed ints, and `lengthof` short-circuiting off the
    // inferred dim (never evaluating the array).
    let src = "J = 4\nx ~ iid(Normal(0.0, 1.0), J + J)";
    let (module, _) = infer_src(src);
    assert!(
        flatppl_flatpir::write(&module).contains("(%array 1 (8)"),
        "got:\n{}",
        flatppl_flatpir::write(&module)
    );

    let src = "v = [1.0, 2.0]\nn = lengthof(v)\nx ~ iid(Normal(0.0, 1.0), n)";
    let (module, _) = infer_src(src);
    assert!(
        flatppl_flatpir::write(&module).contains("(%array 1 (2)"),
        "got:\n{}",
        flatppl_flatpir::write(&module)
    );

    // A stochastic count is NOT resolvable — stays dynamic.
    let src = "n ~ Poisson(5.0)\nx ~ iid(Normal(0.0, 1.0), n)";
    let (module, _) = infer_src(src);
    assert!(
        flatppl_flatpir::write(&module).contains("(%array 1 (%dynamic)"),
        "got:\n{}",
        flatppl_flatpir::write(&module)
    );
}

#[test]
fn mvnormal_dim_from_mu_type() {
    let src = "m = MvNormal(mu = [0.0, 0.0], cov = eye(2))";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(%measure (%domain (%array 1 (2) (%scalar real))) (%mass %normalized))"),
        "got:\n{out}"
    );
}

#[test]
fn broadcast_distribution_head_is_a_measure_over_the_array() {
    // §04 broadcasting: a distribution-constructor head yields a measure
    // over the array of per-cell variates; draw produces the array.
    let src = "mus = [0.0, 1.0, 2.0]\ny ~ broadcast(Normal, mus, 1.0)";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains(
            "(broadcast (%meta (%measure (%domain (%array 1 (3) (%scalar real))) (%mass %normalized)) %fixed (cartpow reals 3))"
        ),
        "got:\n{out}"
    );
    assert!(
        out.contains("(draw (%meta (%array 1 (3) (%scalar real)) %stochastic (cartpow reals 3))"),
        "got:\n{out}"
    );
}

#[test]
fn broadcast_user_kernel_head_with_keyword_data() {
    let src = "mu_g = [0.0, 1.0]\n\
               k = kernelof(_m_ + 0.0, m = _m_)\n\
               obs ~ broadcast(k, m = mu_g)";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains(
            "(broadcast (%meta (%measure (%domain (%array 1 (2) (%scalar real))) (%mass %normalized)) %fixed (cartpow reals 2))"
        ),
        "got:\n{out}"
    );
}

#[test]
fn weighted_types_from_its_base_measure() {
    // §06: weighted(weight, base) — the measure is the SECOND argument.
    let src = "f = fn(_ * 2.0)\nm = normalize(weighted(f, Lebesgue(reals)))";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(normalize (%meta (%measure (%domain (%scalar real)) (%mass %normalized)) %fixed reals)"),
        "got:\n{out}"
    );
}

// ---- value sets and total-mass classes ----

#[test]
fn mass_classes_compose() {
    let src = "lam = Lebesgue(reals)\n\
               lu = Lebesgue(unitinterval)\n\
               t = truncate(Normal(0.0, 1.0), interval(0, inf))\n\
               n = normalize(t)\n\
               post = bayesupdate(n, n)";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    // Unbounded reference measure: infinite but boundedly finite.
    assert!(
        out.contains("(Lebesgue (%meta (%measure (%domain (%scalar real)) (%mass %locallyfinite)) %fixed reals) reals)"),
        "got:\n{out}"
    );
    // Bounded support: finite.
    assert!(
        out.contains("(Lebesgue (%meta (%measure (%domain (%scalar real)) (%mass %finite)) %fixed unitinterval) unitinterval)"),
        "got:\n{out}"
    );
    // Truncation demotes %normalized to %finite (renormalization is not
    // optional); normalize restores %normalized; bayesupdate is %unknown.
    assert!(
        out.contains("(truncate (%meta (%measure (%domain (%scalar real)) (%mass %finite))"),
        "got:\n{out}"
    );
    assert!(
        out.contains("(normalize (%meta (%measure (%domain (%scalar real)) (%mass %normalized))"),
        "got:\n{out}"
    );
    assert!(
        out.contains("(bayesupdate (%meta (%measure (%domain (%scalar real)) (%mass %unknown))"),
        "got:\n{out}"
    );
}

#[test]
fn normalize_of_known_infinite_mass_is_a_static_error() {
    let (module, diags) = infer_src("m = normalize(Lebesgue(reals))");
    assert!(
        diags
            .iter()
            .any(|d| d.severity == Severity::Error && d.message.contains("infinite total mass")),
        "got: {diags:?}"
    );
    let out = flatppl_flatpir::write(&module);
    assert!(out.contains("%failed"), "got:\n{out}");
}

#[test]
fn valueset_producers_and_simplex_chain() {
    // The §08 support column: a Dirichlet draw lands on the simplex; softmax
    // lands on the simplex; the broadcast/categorical mass story rides it.
    let src = "x ~ Dirichlet([1.0, 1.0, 1.0])\nz = softmax([0.0, 1.0])\nc = Categorical(x)";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(draw (%meta (%array 1 (3) (%scalar real)) %stochastic (stdsimplex 3))"),
        "got:\n{out}"
    );
    assert!(
        out.contains("(softmax (%meta (%array 1 (2) (%scalar real)) %fixed (stdsimplex 2))"),
        "got:\n{out}"
    );
    assert!(
        out.contains("(Categorical (%meta (%measure (%domain (%scalar integer)) (%mass %normalized)) %stochastic posintegers)"),
        "got:\n{out}"
    );
}

#[test]
fn level_valueset_vs_normalization() {
    let src = "m = Normal(0.0, 1.0)";
    // Valueset level: support filled, mass still %deferred.
    let mut module = flatppl_syntax::parse(src).unwrap();
    flatppl_infer::infer_with(&mut module, flatppl_infer::Level::Valueset);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %deferred)) %fixed reals)"),
        "got:\n{out}"
    );
    // Normalization level: mass filled.
    let mut module = flatppl_syntax::parse(src).unwrap();
    flatppl_infer::infer_with(&mut module, flatppl_infer::Level::Normalization);
    let out = flatppl_flatpir::write(&module);
    assert!(out.contains("(%mass %normalized)"), "got:\n{out}");
}

// ---- coverage hardening ----

/// Inference must be total over the whole surface fixture corpus: no panics,
/// no error diagnostics, and the ANNOTATED output must survive a strict
/// FlatPIR read → write round-trip (exercising the three-slot %meta and
/// %mass forms on arbitrary real models, not just the goldens).
#[test]
fn corpus_inference_smoke_and_annotated_roundtrip() {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../fixtures/flatppl"]
        .iter()
        .collect();
    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("flatppl") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = fs::read_to_string(&path).unwrap();
        let mut module = flatppl_syntax::parse(&src).unwrap();
        let diags = infer(&mut module);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");

        let annotated = flatppl_flatpir::write(&module);
        let reread = flatppl_flatpir::read(&annotated)
            .unwrap_or_else(|e| panic!("{name}: annotated output unreadable: {e}\n{annotated}"));
        assert_eq!(
            flatppl_flatpir::write(&reread),
            annotated,
            "{name}: annotated FlatPIR is not a write fixpoint"
        );
    }
}

#[test]
fn l1unit_simplex_guard() {
    // Literal nonnegative weights widen to a common named set, so the
    // simplex guard fires; a negative entry defeats it (natural fallback).
    let (module, _) = infer_src("w = l1unit([0.3, 0.7])");
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(l1unit (%meta (%array 1 (2) (%scalar real)) %fixed (stdsimplex 2))"),
        "got:\n{out}"
    );

    let (module, _) = infer_src("w = l1unit([0.3, -0.7])");
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(l1unit (%meta (%array 1 (2) (%scalar real)) %fixed (cartpow reals 2))"),
        "got:\n{out}"
    );
}

#[test]
fn weighted_fixed_scalar_mass_rules() {
    // A fixed scalar weight rescales: classes survive, %normalized demotes
    // to %finite (the constant is unknown); a function weight is %unknown.
    let src = "a = weighted(2.5, Lebesgue(reals))\n\
               b = weighted(2.5, Normal(0.0, 1.0))";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains(
            "(%bind a (weighted (%meta (%measure (%domain (%scalar real)) (%mass %locallyfinite))"
        ),
        "got:\n{out}"
    );
    assert!(
        out.contains(
            "(%bind b (weighted (%meta (%measure (%domain (%scalar real)) (%mass %finite))"
        ),
        "got:\n{out}"
    );
}

#[test]
fn joint_mass_products() {
    let src = "j1 = joint(a = Normal(0.0, 1.0), b = Beta(1.0, 1.0))\n\
               j2 = joint(a = Normal(0.0, 1.0), b = Lebesgue(reals))";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    // normalized × normalized = normalized; normalized × locallyfinite =
    // locallyfinite (Normal ⊗ Lebesgue is infinite but boundedly finite).
    assert!(
        out.contains("(%bind j1 (joint (%meta (%measure (%domain (%record (a (%scalar real)) (b (%scalar real)))) (%mass %normalized))"),
        "got:\n{out}"
    );
    assert!(
        out.contains("(%bind j2 (joint (%meta (%measure (%domain (%record (a (%scalar real)) (b (%scalar real)))) (%mass %locallyfinite))"),
        "got:\n{out}"
    );
}

/// Every §08 distribution row in the catalogue: constructs as a measure,
/// mass %normalized, and a support that is never weaker than the domain's
/// natural extent. Catches transcription slips in the table itself.
#[test]
fn distribution_catalogue_sweep() {
    let scalar_dists = [
        "Uniform",
        "Normal",
        "GeneralizedNormal",
        "Cauchy",
        "StudentT",
        "Logistic",
        "LogNormal",
        "Exponential",
        "Gamma",
        "Weibull",
        "InverseGamma",
        "Beta",
        "ChiSquared",
        "VonMises",
        "Laplace",
        "Bernoulli",
        "Categorical",
        "Categorical0",
        "Binomial",
        "Geometric",
        "NegativeBinomial",
        "NegativeBinomial2",
        "Poisson",
    ];
    for name in scalar_dists {
        let src = format!("m = {name}(0.5, 0.5)");
        let (module, _) = infer_src(&src);
        let out = flatppl_flatpir::write(&module);
        assert!(
            out.contains("(%mass %normalized)"),
            "{name}: not a normalized measure:\n{out}"
        );
        assert!(
            !out.contains("%fixed %unknown)"),
            "{name}: support missing:\n{out}"
        );
    }
    // Multivariate rows: dims ride the length parameter's type.
    let (module, _) = infer_src("m = Dirichlet([1.0, 1.0])");
    let out = flatppl_flatpir::write(&module);
    assert!(out.contains("(stdsimplex 2)"), "got:\n{out}");
    let (module, _) = infer_src("m = Multinomial(5, [0.5, 0.5])");
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(%mass %normalized)") && out.contains("(cartpow nonnegintegers"),
        "got:\n{out}"
    );
}

#[test]
fn get_failure_paths() {
    // A missing record field and an out-of-range tuple index are %failed.
    let (module, _) = infer_src("r = record(a = 1.0)\nx = get(r, \"b\")");
    assert!(flatppl_flatpir::write(&module).contains("%failed"));
    let (module, _) = infer_src("t = (1.0, true)\nx = get(t, 3)");
    assert!(flatppl_flatpir::write(&module).contains("%failed"));
    // get0 is 0-based: index 1 of a pair is its second component.
    let (module, _) = infer_src("t = (1.0, true)\nx = get0(t, 1)");
    assert!(
        flatppl_flatpir::write(&module).contains("(get0 (%meta (%scalar boolean)"),
        "got:\n{}",
        flatppl_flatpir::write(&module)
    );
}

#[test]
fn set_expression_readers() {
    // Negative literal bounds arrive as `neg` calls; cartpow nests.
    let src = "a = elementof(interval(-1.0, 1.0))\nb = elementof(cartpow(integers, 3))";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(elementof (%meta (%scalar real) %parameterized (interval -1.0 1.0))"),
        "got:\n{out}"
    );
    assert!(
        out.contains("(elementof (%meta (%array 1 (3) (%scalar integer)) %parameterized (cartpow integers 3))"),
        "got:\n{out}"
    );
}

#[test]
fn reference_measure_mass_arms() {
    // iid of a locally finite measure stays locally finite; truncating one
    // to a bounded window is finite; Counting on integers is locally finite.
    let src = "a = iid(Lebesgue(reals), 3)\n\
               b = truncate(Lebesgue(reals), interval(0.0, 1.0))\n\
               c = Counting(integers)";
    let (module, _) = infer_src(src);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("(%bind a (iid (%meta (%measure (%domain (%array 1 (3) (%scalar real))) (%mass %locallyfinite))"),
        "got:\n{out}"
    );
    assert!(
        out.contains(
            "(%bind b (truncate (%meta (%measure (%domain (%scalar real)) (%mass %finite))"
        ),
        "got:\n{out}"
    );
    assert!(
        out.contains("(%bind c (Counting (%meta (%measure (%domain (%scalar integer)) (%mass %locallyfinite))"),
        "got:\n{out}"
    );
}
