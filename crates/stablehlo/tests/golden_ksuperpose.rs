//! StableHLO emission for direct weighted kernel broadcasts (§06).
//!
//! Scalar goldens preserve the existing logsumexp density calculation.
//! Structural checks cover nested cells, multiple axes and compact emission.
//! Numeric value and derivative oracles live in flatppl-testsuite's
//! `tests/test_mixture_families.py`.

use flatppl_stablehlo::{Mode, emit};

fn emit_logdensity(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "inference errors: {errors:?}");
    let d = flatppl_determinizer::determinize(&m).expect("must lower, not refuse");
    emit(&d, Mode::LogDensity, &Default::default()).expect("must emit")
}

#[test]
fn multivariate_mixture_has_one_batched_density_at_any_component_count() {
    let mut operations = Vec::new();
    for n in [2, 128] {
        let src = format!(
            "flatppl_compat = \"0.1\"\n\
             w = ones({n})\n\
             mus = external(cartpow(cartpow(reals, 2), {n}))\n\
             covs = [eye(2)]\n\
             mix = ksuperpose(MvNormal, w, mu = mus, cov = covs)\n\
             y = [0.2, 0.3]\n\
             lp = logdensityof(mix, y)\n\
             inputs = (w, mus, covs, y)\n\
             outputs = (lp)\n"
        );
        let hlo = emit_logdensity(&src);
        assert!(hlo.contains("stablehlo.triangular_solve"), "{hlo}");
        assert!(
            n == 2 || !hlo.contains(&format!("tensor<{n}x{n}x2xf32>")),
            "quadratic gather operand at N={n}"
        );
        operations.push(
            hlo.lines()
                .filter(|line| line.contains("stablehlo."))
                .count(),
        );
    }
    assert_eq!(
        operations[0], operations[1],
        "operation count must not grow with N"
    );
}

const NORMAL_MIXTURE_SRC: &str = "\
flatppl_compat = \"0.1\"
w = [0.3, 1.2]
mus = [-1.0, 2.0]
sigmas = [1.0, 0.5]
mix = normalize(ksuperpose(Normal, w, mu = mus, sigma = sigmas))
lp = logdensityof(mix, 0.5)
outputs = (lp)
";

const SCALAR_SIGMA_SRC: &str = "\
flatppl_compat = \"0.1\"
w = [0.3, 1.2]
mus = [-1.0, 2.0]
mix = normalize(ksuperpose(Normal, w, mu = mus, sigma = 1.0))
lp = logdensityof(mix, 0.5)
outputs = (lp)
";

const DIRAC_SRC: &str = "\
flatppl_compat = \"0.1\"
p = [0.2, 0.8]
labels = [0.0, 1.5]
c = normalize(ksuperpose(Dirac, p, value = labels))
lp = logdensityof(c, 1.5)
outputs = (lp)
";

/// The two-Normal mixture with unnormalized weights `w = [0.3, 1.2]`, normalized.
/// Executed == scipy oracle at four points (see the module header).
#[test]
fn normal_mixture_matches_frozen_golden() {
    let out = emit_logdensity(NORMAL_MIXTURE_SRC);
    let golden = include_str!("goldens/ksuperpose_normal_mixture_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/ksuperpose_normal_mixture_logdensity.mlir)"
    );
}

/// One shared SCALAR sigma held constant across the components (§06: "Non-collection
/// arguments are held constant"). Executed == scipy oracle at x = 0.5.
#[test]
fn scalar_sigma_mixture_matches_frozen_golden() {
    let out = emit_logdensity(SCALAR_SIGMA_SRC);
    let golden = include_str!("goldens/ksuperpose_scalar_sigma_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/ksuperpose_scalar_sigma_logdensity.mlir)"
    );
}

/// §08's categorical over arbitrary values,
/// `normalize(ksuperpose(Dirac, p, value = labels))`. Executed == scipy oracle at
/// both support points.
#[test]
fn dirac_superposition_matches_frozen_golden() {
    let out = emit_logdensity(DIRAC_SRC);
    let golden = include_str!("goldens/ksuperpose_dirac_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/ksuperpose_dirac_logdensity.mlir)"
    );
}

/// The mixture is emitted as ONE batched component-density evaluation over the
/// family axis, reduced by the shift-by-max `logsumexp` — never `N` separate
/// logpdf blocks. The batch dimension is the component count.
#[test]
fn the_mixture_emits_one_batched_density_reduced_by_logsumexp() {
    let out = emit_logdensity(&format!(
        "{NORMAL_MIXTURE_SRC}\ninputs = (w, mus, sigmas)\n"
    ));
    assert!(
        out.contains("tensor<2xf32>"),
        "the component axis is the batch dimension:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum") && out.contains("applies stablehlo.add"),
        "logsumexp is the shift-by-max form: a maximum reduce then an add reduce:\n{out}"
    );
    // Two `log` on rank-1: the log-weights, and `Normal`'s own `log(sigma)`.
    assert_eq!(
        out.matches("stablehlo.log %").count(),
        4,
        "log-weights + log(sigma) on the batch, then logsumexp's log and logZ's:\n{out}"
    );
}

/// §06's mass sentence gives `Z = Σᵢ wᵢ` for a Markov component, so the
/// normalizer is a `sum` reduce over the weight vector followed by a `log` and a
/// `subtract` — no `totalmass`, and the whole thing stays a closed-form scalar.
#[test]
fn the_normalizer_is_a_sum_reduce_over_the_weights() {
    let normalized = emit_logdensity(NORMAL_MIXTURE_SRC);
    let unnormalized = emit_logdensity(
        "flatppl_compat = \"0.1\"\n\
         w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         sigmas = [1.0, 0.5]\n\
         mix = ksuperpose(Normal, w, mu = mus, sigma = sigmas)\n\
         lp = logdensityof(mix, 0.5)\n\
         outputs = (lp)\n",
    );
    assert!(
        normalized.len() > unnormalized.len(),
        "the normalized form adds the logZ tail"
    );
    assert!(
        normalized.contains("stablehlo.subtract") && !unnormalized.ends_with("subtract"),
        "logZ is subtracted from the mixture density:\n{normalized}"
    );
    assert_eq!(
        normalized.matches("applies stablehlo.add").count(),
        2,
        "two add-reduces: logsumexp's inner sum, and Z = sum(w):\n{normalized}"
    );
}

/// A SINGULAR (size-one) family argument, which §06 expands by repetition. The
/// emitter reconciles the size-one axis against the component axis, so this needs
/// no determiniser-side tiling. Executed == scipy oracle at x = 0.5.
#[test]
fn a_singular_family_argument_emits_against_the_component_axis() {
    let out = emit_logdensity(
        "flatppl_compat = \"0.1\"\n\
         w = [0.3, 1.2]\n\
         mix = normalize(ksuperpose(Normal, w, mu = [1.0], sigma = [0.5, 2.0]))\n\
         lp = logdensityof(mix, 0.5)\n\
         outputs = (lp)\n",
    );
    assert!(
        out.contains("tensor<2xf32>"),
        "the size-one mu expands to the 2-component axis:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum"),
        "still a logsumexp:\n{out}"
    );
}

/// A component whose density builder is not rank-agnostic cannot be evaluated
/// batched over the family axis. The existing `is_batch_safe` gate refuses it —
/// refuse-don't-mislower — so the mixture inherits exactly the same restriction
/// `iid` and value-`broadcast` already have.
///
/// `NegativeBinomial2` is the component: its builder uses `get0`/`reshape`, and
/// it carries no dedicated batched builder either. This test used to use
/// `Uniform`, which became batch-safe once its density gained §08's support mask
/// (the mask carries the variate's shape).
#[test]
fn a_non_batch_safe_component_refuses_at_the_existing_gate() {
    let mut m = flatppl_syntax::parse(
        "flatppl_compat = \"0.1\"\n\
         w = [0.3, 1.2]\n\
         mix = ksuperpose(NegativeBinomial2, w, mu = [1.0, 2.0], psi = 2.0)\n\
         lp = logdensityof(mix, 3)\n\
         outputs = (lp)\n",
    )
    .unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let d = flatppl_determinizer::determinize(&m).expect("determinizes");
    let err = emit(&d, Mode::LogDensity, &Default::default()).expect_err("must refuse");
    assert!(err.msg.contains("not rank-agnostic"), "got: {}", err.msg);
}

#[test]
fn ordinary_broadcast_mixtures_emit_expanded_normalizers() {
    for (weights, means) in [
        ("[2.]", "[-1.,2.]"),
        ("2.", "[-1.,2.]"),
        ("rowstack([[2.],[3.]])", "rowstack([[-1.,2.,3.]])"),
    ] {
        let hlo = emit_logdensity(&format!(
            "w = {weights}\nmeans = {means}\nm = ksuperpose(Normal, w, means, sigma = 1.)\nz = totalmass(m)\nlp = logdensityof(normalize(m), 0.5)\noutputs = (lp,z)"
        ));
        assert!(hlo.contains("stablehlo.reduce"), "{hlo}");
        assert!(
            !hlo.contains("tensor<6x3xf32>"),
            "flattening must not duplicate tensor rows: {hlo}"
        );
    }
}

#[test]
fn empty_broadcasts_keep_zero_mass_and_empty_density_reductions() {
    // Derived zero extents are dynamic in source inference. As in golden_norms,
    // instantiate a zero-length backend input after determinizing valid source.
    fn empty_type(ty: flatppl_core::Type) -> flatppl_core::Type {
        use flatppl_core::{Dim, Type};
        match ty {
            Type::Array { shape, elem } => Type::Array {
                shape: shape
                    .iter()
                    .map(|d| {
                        if *d == Dim::Static(7) {
                            Dim::Static(0)
                        } else {
                            *d
                        }
                    })
                    .collect(),
                elem: Box::new(empty_type(*elem)),
            },
            other => other,
        }
    }
    for (weights, means, input) in [
        ("[1.]", "external(cartpow(reals, 7))", "means"),
        ("2.", "external(cartpow(reals, 7))", "means"),
        ("external(cartpow(nonnegreals, 7))", "[0.]", "w"),
    ] {
        let source = format!(
            "w = {weights}\nmeans = {means}\nm = ksuperpose(Normal, w, means, sigma = 1.)\nz = totalmass(m)\nlp = logdensityof(m, 0.5)\ninputs = {input}\noutputs = (z,lp)"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        flatppl_infer::infer(&mut module);
        let mut lowered = flatppl_determinizer::determinize(&module).unwrap();
        use flatppl_core::Idx;
        for i in 0..lowered.node_count() {
            let id = flatppl_core::NodeId::from_usize(i);
            if let Some(ty) = lowered.type_of(id).cloned() {
                lowered.set_type(id, empty_type(ty));
            }
        }
        let hlo = emit(&lowered, Mode::LogDensity, &Default::default()).unwrap();
        // Empty sums return 0 and the empty logsumexp returns -infinity.
        assert!(hlo.contains("tensor<0xf32>"), "{hlo}");
        assert!(hlo.contains("dense<0xFF800000>"), "{hlo}");
        assert!(
            hlo.contains("stablehlo.add across dimensions = [0]"),
            "{hlo}"
        );
    }
}
