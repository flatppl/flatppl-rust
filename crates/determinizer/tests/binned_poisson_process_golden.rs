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

fn lowered(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    let text = flatppl_syntax::print(&out);
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{text}"
    );
    text
}

// The HS3 `bincounts_extended_dist` shape. Each bin's mean is the rate times the
// shape's CDF difference, and one Poisson broadcast scores all four counts.
#[test]
fn weighted_shape_lowers_to_poisson_counts_over_cdf_differences() {
    let text = lowered(
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
        text.matches("builtin_touniform").count(),
        8,
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
    let text = lowered(
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
    let text = lowered(
        "\
n_bkg = elementof(posreals)
proc = BinnedPoissonProcess(extlinspace(0.0, 2.0, 3), weighted(n_bkg, Normal(1.0, 0.5)))
lp = logdensityof(proc, [1, 12, 20, 2])",
    );
    assert_eq!(
        text.matches("get0(extlinspace(0.0, 2.0, 3)").count(),
        8,
        "each of four bins reads its two edges:\n{text}"
    );
}

// Any other edge vector of static length is read element by element.
#[test]
fn a_computed_static_edge_vector_lowers_through_get0() {
    let text = lowered(
        "\
edges = cumsum([0.0, 1.0, 2.0])
proc = BinnedPoissonProcess(edges, weighted(5.0, Normal(1.0, 0.5)))
lp = logdensityof(proc, [1, 3])",
    );
    assert_eq!(
        text.matches("get0(").count(),
        4,
        "each bin reads its two edges:\n{text}"
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
    let text = lowered(
        "\
hep = standard_module(\"particle-physics\", \"0.1\")
shape = hep.CrystalBall(m0 = 1.0, sigma = 0.3, alpha = 1.5, n = 3.0)
proc = BinnedPoissonProcess([0.0, 1.0, 2.0], weighted(10.0, shape))
lp = logdensityof(proc, [4, 5])",
    );
    assert_eq!(
        text.matches("builtin_touniform").count(),
        4,
        "two CDF evaluations for each of two bins:\n{text}"
    );
}
