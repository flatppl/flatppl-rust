// `BinnedPoissonProcess(bins, intensity)` — spec §08's fixed-bins equivalence
// `pushfwd(fn(bincounts(bins, _)), PoissonProcess(intensity))`, lowered as one
// Poisson count per bin with mean the intensity's mass there — and the limits it
// refuses instead of mis-lowering.

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn lowered(src: &str) -> (String, std::collections::HashMap<String, usize>) {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    let text = flatppl_syntax::print(&out);
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{text}"
    );
    // Count shared IR nodes once. Printed guards repeat their operands.
    let mut pending: Vec<_> = out.bindings().map(|(_, b)| b.rhs).collect();
    let mut seen = std::collections::HashSet::new();
    let mut calls = std::collections::HashMap::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = out.node(id);
        if let flatppl_core::Node::Call(call) = node
            && let flatppl_core::CallHead::Builtin(name) = call.head
        {
            *calls.entry(out.resolve(name).to_owned()).or_insert(0) += 1;
        }
        pending.extend(node.children());
    }
    (text, calls)
}

// The HS3 `bincounts_extended_dist` shape. Each bin's mean is the rate times the
// shape's CDF difference, and one Poisson broadcast scores all four counts.
#[test]
fn weighted_shape_lowers_to_poisson_counts_over_cdf_differences() {
    let (text, calls) = lowered(
        "\
n_bkg = elementof(posreals)
bshape = Normal(mu = 1.0, sigma = 0.5)
proc = BinnedPoissonProcess([0.0, 0.5, 1.0, 1.5, 2.0], weighted(n_bkg, bshape))
lp = logdensityof(proc, [3, 12, 20, 9])",
    );
    assert!(
        text.contains("builtin_logdensityof.(") && text.contains("Poisson"),
        "one Poisson broadcast over the counts:\n{text}"
    );
    assert_eq!(
        text.matches("exp(").count(),
        4,
        "one bin mass per bin:\n{text}"
    );
    assert_eq!(
        calls["builtin_touniform"], 8,
        "two CDF evaluations per bin:\n{text}"
    );
    assert!(
        !text.contains("BinnedPoissonProcess") && !text.contains("weighted"),
        "measure layer gone:\n{text}"
    );
}

// Bin masses of a flat `Uniform`, a `Lebesgue` window and their `superpose` are
// overlap lengths, so no `Uniform` CDF reaches the target. The keyword spelling
// binds like the positional one.
#[test]
fn flat_intensities_lower_to_overlap_lengths() {
    let (text, _) = lowered(
        "\
s = elementof(posreals)
flat = Uniform(interval(0.0, 6.0))
window = Lebesgue(support = interval(0.5, 5.0))
proc = BinnedPoissonProcess(intensity = superpose(weighted(s, flat), weighted(0.3, window)), bins = [0.0, 1.0, 3.0, 6.0])
lp = logdensityof(proc, [2, 7, 4])",
    );
    assert!(
        !text.contains("builtin_touniform"),
        "no CDF for flat intensities:\n{text}"
    );
    assert_eq!(
        text.matches("/ 6.0").count(),
        3,
        "the uniform's share of each bin:\n{text}"
    );
}

#[test]
fn polynomial_intensity_uses_unnormalized_bin_masses() {
    // Integral of 1 + 2x on [0, 1/2] and [1/2, 1]: 3/4 and 5/4.
    for intensity in [
        "weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 1)))",
        "truncate(weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(reals)), interval(0, 1))",
    ] {
        let (text, _) = lowered(&format!(
            "I = {intensity}\nJ = I\nlp = logdensityof(BinnedPoissonProcess([-1, 0, 0.5, 1, 2], J), [0, 0, 0, 0])"
        ));
        assert!(text.contains("[0.0, 0.75, 1.25, 0.0]"), "{text}");
        assert!(!text.contains("%log_integral"), "{text}");
    }
}

#[test]
fn polynomial_zero_mass_bin_keeps_positive_count_for_poisson_scoring() {
    // The first bin does not intersect [0, 1]; its rate is zero and observing
    // one event there has log density -infinity, not the zero-count score.
    let (text, _) = lowered(
        "I = weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 1)))\nlp = logdensityof(BinnedPoissonProcess([-1, 0, 1], I), [1, 0])",
    );
    assert!(text.contains("[0.0, 2.0]"), "zero and full mass: {text}");
    assert!(
        text.contains("[1, 0]"),
        "positive count is retained: {text}"
    );
    assert!(text.contains("Poisson"), "Poisson scores zero rate: {text}");
}

#[test]
fn polynomial_intensity_preserves_live_extra_and_composition() {
    let (text, _) = lowered(
        "a = elementof(nonnegreals)\nI = weighted((x, p) -> polynomial([1.0, p], x), Lebesgue(interval(0, 1)), a)\nJ = superpose(weighted(2.0, I), Lebesgue(interval(0, 1)))\nlp = logdensityof(BinnedPoissonProcess([0, 0.5, 1], J), [1, 2])\ninputs = a\noutputs = lp",
    );
    assert!(text.contains('a'), "live coefficient: {text}");
    assert!(!text.contains("BinnedPoissonProcess"), "{text}");
}

#[test]
fn polynomial_bins_require_static_edges_and_finite_full_support() {
    for src in [
        "e = elementof(reals)\nI = weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 1)))\nlp = logdensityof(BinnedPoissonProcess([0, e, 1], I), [0, 0])\ninputs = e\noutputs = lp",
        "I = weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(reals))\nlp = logdensityof(BinnedPoissonProcess([0, 1], I), [0])",
    ] {
        let err = determinize(&parse_infer(src)).expect_err("unsupported bin mass must refuse");
        assert!(
            err.reason
                .contains("cannot derive this intensity mass per bin"),
            "{err:?}"
        );
        let err = flatppl_determinizer::determinize_with_options(
            &parse_infer(src),
            &flatppl_infer::ModuleBundle::new(),
            None,
            &flatppl_determinizer::LoweringOptions {
                numerical_integrals: true,
                ..Default::default()
            },
        )
        .expect_err("numerical opt-in still requires finite static intervals and edges");
        assert!(
            err.reason
                .contains("cannot derive this intensity mass per bin")
        );
    }
}

// §08 gives `bins` §07 `bincounts`'s record form for multi-dimensional bins. That
// form is out of the lowered fragment and must refuse.
#[test]
fn a_record_of_edge_vectors_refuses() {
    let src = "\
intensity = weighted(5.0, joint(a = Normal(0.0, 1.0), b = Normal(0.0, 1.0)))
proc = BinnedPoissonProcess(record(a = [0.0, 1.0, 2.0], b = [0.0, 1.0]), intensity)
lp = logdensityof(proc, rowstack([[1], [2]]))";
    let err = determinize(&parse_infer(src)).expect_err("a record of edges must refuse");
    assert!(
        err.reason.contains("multi-dimensional bins"),
        "refusal names the limit: {err:?}"
    );
}

// A discrete intensity has atoms, so a bin's half-open edge changes its mass. No
// closed-form rule covers it, and the lowering refuses rather than guess.
#[test]
fn a_discrete_intensity_refuses() {
    let src = "\
proc = BinnedPoissonProcess([0.0, 2.0, 4.0], weighted(3.0, Poisson(rate = 2.0)))
lp = logdensityof(proc, [1, 2])";
    let err = determinize(&parse_infer(src)).expect_err("a discrete intensity must refuse");
    assert!(
        err.reason.contains("intensity has an atom"),
        "refusal names the atom: {err:?}"
    );
}

// A `Dirac` intensity is a normalized measure over the reals with an atom, so a
// CDF difference misplaces its mass on a half-open bin edge: the atom at 1.0
// belongs to `[1, 2]`, not `[0, 1)`. It must refuse.
#[test]
fn a_dirac_intensity_refuses() {
    let src = "\
proc = BinnedPoissonProcess([0.0, 1.0, 2.0], weighted(3.0, Dirac(1.0)))
lp = logdensityof(proc, [0, 4])";
    let err = determinize(&parse_infer(src)).expect_err("a Dirac intensity must refuse");
    assert!(
        err.reason.contains("intensity has an atom"),
        "refusal names the atom: {err:?}"
    );
}

// The same atom breaks the CDF normalizer of a truncation: `interval(1.0, 2.0)`
// holds the atom, yet `F(2) − F(1)` is zero. Main lowered this with exit 0.
#[test]
fn a_truncated_dirac_normalizer_refuses() {
    let src = "\
t = normalize(truncate(Dirac(1.0), interval(1.0, 2.0)))
lp = logdensityof(t, 1.0)";
    let err = determinize(&parse_infer(src)).expect_err("a truncated Dirac must refuse");
    assert!(
        err.reason.contains("univariate continuous base"),
        "refusal names the limit: {err:?}"
    );
}

// §07's own edge forms are ordinary vectors: `extlinspace(0.0, 2.0, 3)` has five
// points, so its four bins read their edges through `get0` like any static vector.
#[test]
fn linspace_edges_lower_through_get0() {
    let (text, calls) = lowered(
        "\
n_bkg = elementof(posreals)
proc = BinnedPoissonProcess(extlinspace(0.0, 2.0, 3), weighted(n_bkg, Normal(1.0, 0.5)))
lp = logdensityof(proc, [1, 12, 20, 2])",
    );
    assert_eq!(
        calls["get0"], 5,
        "the four bins share five edge nodes:\n{text}"
    );
}

// Any other edge vector of static length is read element by element.
#[test]
fn a_computed_static_edge_vector_lowers_through_get0() {
    let (text, calls) = lowered(
        "\
edges = cumsum([0.0, 1.0, 2.0])
proc = BinnedPoissonProcess(edges, weighted(5.0, Normal(1.0, 0.5)))
lp = logdensityof(proc, [1, 3])",
    );
    assert_eq!(
        calls["get0"], 3,
        "the two bins share three edge nodes:\n{text}"
    );
}

#[test]
fn edges_of_no_static_length_refuse() {
    let src = "\
n = elementof(posintegers)
proc = BinnedPoissonProcess(linspace(0.0, 2.0, n), weighted(5.0, Normal(1.0, 0.5)))
lp = logdensityof(proc, [1, 3])";
    let err = determinize(&parse_infer(src)).expect_err("dynamic edges must refuse");
    assert!(
        err.reason.contains("no static length"),
        "refusal names the cause: {err:?}"
    );
}

// The atomless-leaf list is closed, and it covers the §09 particle-physics shapes:
// a module-qualified `CrystalBall` intensity still gets its CDF difference.
#[test]
fn a_module_leaf_intensity_lowers() {
    let (text, calls) = lowered(
        "\
hep = standard_module(\"particle-physics\", \"0.1\")
shape = hep.CrystalBall(m0 = 1.0, sigma = 0.3, alpha = 1.5, n = 3.0)
proc = BinnedPoissonProcess([0.0, 1.0, 2.0], weighted(10.0, shape))
lp = logdensityof(proc, [4, 5])",
    );
    assert_eq!(
        calls["builtin_touniform"], 4,
        "two CDF evaluations for each of two bins:\n{text}"
    );
}
