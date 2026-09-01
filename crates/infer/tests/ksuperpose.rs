//! Type, shape and phase rules for `ksuperpose(kernel, weights)` — spec §06
//! "Additive superposition" and §04's arity row.
//!
//! The lift is a KERNEL; applying it to a parameter family contracts the family
//! axis into a mixture over the components' shared variate. Both halves are
//! pinned here, plus the two static errors §06 names (a family argument whose
//! family-axis count is not one, and one whose size is neither $N$ nor one).

use flatppl_infer::{Severity, infer};

fn infer_src(src: &str) -> (flatppl_core::Module, Vec<flatppl_infer::Diagnostic>) {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut module);
    (module, diags)
}

fn pir(src: &str) -> String {
    let (module, diags) = infer_src(src);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    flatppl_flatpir::write(&module)
}

fn rejects(src: &str, expected: &str) {
    let (_, diags) = infer_src(src);
    let messages: Vec<&str> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains(expected)),
        "no error containing `{expected}`; got {messages:?}"
    );
}

const TWO_NORMALS: &str = "\
w = [0.3, 1.2]
mus = [-1.0, 2.0]
sigmas = [1.0, 0.5]
mix = ksuperpose(Normal, w)(mu = mus, sigma = sigmas)
";

/// §06: "`ksuperpose(kernel, weights)` is itself a kernel". Its declared inputs
/// are the component constructor's own §08 parameter names, since §06 passes the
/// family "as to `broadcast`".
#[test]
fn the_lift_is_a_kernel_declaring_the_components_parameters() {
    let out = pir(TWO_NORMALS);
    assert!(
        out.contains("(%kernel (%inputs mu sigma) (%mass %finite))"),
        "the lift must be a kernel over Normal's own parameters:\n{out}"
    );
}

/// §06: applied to a parameter family the lift "yields the mixture
/// $\nu = \sum_i w_i\,\kappa(\theta_i)$" — ONE measure over the components'
/// shared variate. So the family axis is CONTRACTED: unlike `broadcast`, whose
/// applied form is the independent product over an ARRAY variate, the mixture's
/// domain is the component's per-cell variate.
#[test]
fn applying_the_lift_contracts_the_family_axis_into_one_measure() {
    let out = pir(TWO_NORMALS);
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %finite))"),
        "the mixture is a scalar-variate measure, not a measure over an array:\n{out}"
    );
    assert!(
        !out.contains("(%measure (%domain (%array 1 (2)"),
        "the family axis must not survive into the variate (that is `broadcast`, \
         the independent product):\n{out}"
    );
}

/// §06: the weights "need not be normalized, so the result is generally
/// unnormalized, of total mass $\sum_i w_i\,\mathrm{totalmass}(\kappa(\theta_i))$,
/// which is $\sum_i w_i$ for a Markov `kernel`". So a Markov component gives
/// `%finite`, never `%normalized` — and `draw` must therefore reject the bare
/// mixture, exactly as it rejects any unnormalized measure.
#[test]
fn a_markov_component_makes_the_mixture_finite_not_normalized() {
    let out = pir(TWO_NORMALS);
    assert!(
        out.contains("(%mass %finite)"),
        "an unnormalized mixture is `%finite`:\n{out}"
    );
    rejects(
        &format!("{TWO_NORMALS}y = draw(mix)\n"),
        "there is no draw from an unnormalized measure",
    );
}

/// The §06 example spelling `normalize(ksuperpose(Normal, weights)(mu = means,
/// sigma = sigmas))` is a probability measure and IS drawable.
#[test]
fn normalize_recovers_a_probability_measure_that_draws() {
    let out = pir("w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         sigmas = [1.0, 0.5]\n\
         mix = normalize(ksuperpose(Normal, w)(mu = mus, sigma = sigmas))\n\
         y ~ mix\n");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized))"),
        "normalize gives a probability measure:\n{out}"
    );
}

/// §06: "Non-collection arguments are held constant across the components." A
/// scalar `sigma` is therefore legal against a length-2 weight vector and needs
/// no size agreement at all.
#[test]
fn a_scalar_family_argument_is_held_constant() {
    let out = pir("w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %finite))"),
        "a held-constant scalar parameter still types the mixture:\n{out}"
    );
}

/// §06: "every collection argument must have size $N$ or be singular (size one),
/// and size-one arguments are expanded by repetition to size $N$".
#[test]
fn a_singular_family_argument_is_admitted() {
    let out = pir("w = [0.3, 1.2]\n\
         mix = ksuperpose(Normal, w)(mu = [1.0], sigma = [0.5, 2.0])\n");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %finite))"),
        "a size-one family argument expands by repetition:\n{out}"
    );
}

/// The other half of the same sentence: a size that is neither $N$ nor one is a
/// static error, since $N$ is fixed by `weights` and nothing reconciles a third
/// length.
#[test]
fn a_family_argument_of_the_wrong_size_is_a_static_error() {
    rejects(
        "w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0, 3.0]\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n",
        "must have size 2 along the family axis — the length of `weights` — or be singular",
    );
}

/// §06: "an argument's family axes are its leading axes in excess of the rank
/// (number of axes) of the parameter it feeds, and any count other than one is a
/// static error". `Normal`'s `mu` has rank 0, so a two-axis argument carries two
/// family axes. A nested array counts its axes, so `[[…], […]]` is caught as
/// readily as a declared matrix.
#[test]
fn two_family_axes_over_a_scalar_parameter_is_a_static_error() {
    rejects(
        "w = [0.3, 1.2]\n\
         mus = [[0.0, 1.0], [2.0, 3.0]]\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n",
        "`mu` has rank 0, so a collection with 2 axes gives 2 family axes",
    );
}

/// The same sentence in the other direction: a family argument at the
/// parameter's own rank carries ZERO family axes, which "any count other than
/// one" also refuses. A shared covariance must be spelled with a singular family
/// axis; only a NON-collection is held constant.
#[test]
fn zero_family_axes_over_a_matrix_parameter_is_a_static_error() {
    rejects(
        "w = [0.2, 0.8]\n\
         mus = rowstack([[0.0, 0.0], [3.0, 3.0]])\n\
         cov = rowstack([[1.0, 0.2], [0.2, 1.0]])\n\
         mix = ksuperpose(MvNormal, w)(mu = mus, cov = cov)\n",
        "`cov` has rank 2, so a collection with 2 axes gives 0 family axes",
    );
}

/// §06: "Within the family the same-number-of-axes requirement of *Collection
/// arguments* does not apply, so the components may be multivariate — a vector
/// parameter takes an $N \times d$ matrix while a matrix parameter takes an
/// $N \times d \times d$ array." The mixture's variate is the COMPONENT
/// variate: a vector of $d$, not an array over the family.
#[test]
fn a_multivariate_family_mixes_over_the_component_variate() {
    let out = pir("w = [0.2, 0.5, 0.3]\n\
         mus = rowstack([[0.0, 0.0], [3.0, 3.0], [-2.0, 1.0]])\n\
         c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])\n\
         c2 = rowstack([[2.0, 0.0], [0.0, 0.5]])\n\
         c3 = rowstack([[1.5, -0.3], [-0.3, 1.5]])\n\
         covs = [c1, c2, c3]\n\
         mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n\
         y ~ mix\n");
    assert!(
        out.contains("(%measure (%domain (%array 1 (2) (%scalar real))) (%mass %normalized))"),
        "the mixture is a measure over MvNormal's own vector variate:\n{out}"
    );
}

/// A singular family axis expands by repetition at any rank: one $d \times d$
/// covariance serves all $N$ components when it is spelled $1 \times d \times d$.
#[test]
fn a_singular_family_axis_expands_at_matrix_rank() {
    let out = pir("w = [0.2, 0.5, 0.3]\n\
         mus = rowstack([[0.0, 0.0], [3.0, 3.0], [-2.0, 1.0]])\n\
         c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])\n\
         covs = [c1]\n\
         mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n\
         y ~ mix\n");
    assert!(
        out.contains("(%measure (%domain (%array 1 (2) (%scalar real))) (%mass %normalized))"),
        "a size-one cov axis expands over the three components:\n{out}"
    );
}

/// A table family works by ROW axis with per-column element rank: `MvNormal`'s
/// `mu` column holds vectors and its `cov` column holds matrices.
#[test]
fn a_table_family_takes_its_column_element_ranks() {
    let out = pir("w = [0.2, 0.8]\n\
         c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])\n\
         pars = table(mu = [[0.0, 0.0], [3.0, 3.0]], cov = [c1, c1])\n\
         mix = normalize(ksuperpose(MvNormal, w)(pars))\n\
         y ~ mix\n");
    assert!(
        out.contains("(%domain (%array 1 "),
        "the mixture is a measure over a vector variate:\n{out}"
    );
    // A vector-element column over a SCALAR parameter is the two-family-axes error.
    rejects(
        "w = [0.2, 0.8]\n\
         pars = table(mu = [[0.0, 0.0], [3.0, 3.0]], sigma = [1.0, 0.5])\n\
         mix = ksuperpose(Normal, w)(pars)\n",
        "the table's rows are one axis and `mu` has rank 0",
    );
}

/// §06 fixes $N$ as "the length of `weights`", so a scalar weight argument leaves
/// $N$ undefined rather than merely unknown.
#[test]
fn scalar_weights_are_a_static_error() {
    rejects(
        "mix = ksuperpose(Normal, 0.5)(mu = [1.0], sigma = 1.0)\n",
        "weights must be a vector",
    );
}

/// §04's arity row gives `ksuperpose` exactly two distinguished inputs.
#[test]
fn the_lift_takes_exactly_two_arguments() {
    rejects(
        "mix = ksuperpose(Normal)\n",
        "`ksuperpose` takes 2 positional arguments",
    );
    rejects(
        "w = [0.3, 1.2]\nmix = ksuperpose(Normal, w, w)\n",
        "`ksuperpose` takes 2 positional arguments",
    );
}

/// §06: "A table counts as having one axis, its rows". So a table family argument
/// is measured against $N$ by its row count and never trips the multi-axis error.
#[test]
fn a_table_family_argument_counts_its_rows_as_the_one_axis() {
    let out = pir("w = [0.3, 1.2]\n\
         params = table(mu = [-1.0, 2.0], sigma = [1.0, 0.5])\n\
         mix = ksuperpose(Normal, w)(params)\n");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %finite))"),
        "a 2-row table is a legal 2-component family:\n{out}"
    );
    rejects(
        "w = [0.3, 1.2]\n\
         params = table(mu = [-1.0, 2.0, 3.0], sigma = [1.0, 0.5, 0.25])\n\
         mix = ksuperpose(Normal, w)(params)\n",
        "must have size 2 along the family axis — the length of `weights` — or be singular",
    );
}

/// §08: "For a categorical over arbitrary values rather than integer indices,
/// superpose Diracs at those values: `normalize(ksuperpose(Dirac, p)(value =
/// labels))`". `Dirac` is a §06 fundamental measure outside the §08 catalogue, so
/// its variate has to come from its own `value` argument.
#[test]
fn a_dirac_superposition_takes_its_variate_from_the_values() {
    let out = pir("p = [0.2, 0.8]\n\
         labels = [0.0, 1.5]\n\
         c = normalize(ksuperpose(Dirac, p)(value = labels))\n\
         z ~ c\n");
    assert!(
        out.contains("(%kernel (%inputs value) (%mass %finite))"),
        "the Dirac lift declares Dirac's own `value` parameter:\n{out}"
    );
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized))"),
        "the normalized categorical is a probability measure over the labels' type:\n{out}"
    );
}

/// A REIFIED component contributes its own declared inputs rather than a
/// constructor's parameter names, and its `%normalized` body still demotes to
/// `%finite` under the weights.
#[test]
fn a_reified_kernel_component_contributes_its_own_inputs() {
    let out = pir("w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         k = kernelof(draw(Normal(mu = _m_, sigma = 1.0)), m = _m_)\n\
         mix = ksuperpose(k, w)(m = mus)\n");
    assert!(
        out.contains("(%kernel (%inputs m) (%mass %finite))"),
        "the lift declares the reified kernel's own input `m`:\n{out}"
    );
}

/// §06: $N$ "need not be statically known". Weights supplied at runtime rather
/// than written as a literal still type the mixture — its variate comes from the
/// component, not from the family.
#[test]
fn runtime_weights_still_type_the_mixture() {
    let out = pir("w = external(cartpow(nonnegreals, 2))\n\
         mus = external(cartpow(reals, 2))\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %finite))"),
        "externally-supplied weights still give the unnormalized mixture:\n{out}"
    );
}

/// PARAMETERIZED weights — mixture weights being inferred, the common real
/// spelling — leave the total mass `%unknown`: nothing static bounds a parameter,
/// and §06's all-zero case makes even zero admissible, which is not `%finite`'s
/// guarantee. `normalize` still recovers a probability measure, so the model is
/// drawable; and §04's ancestor rule carries the parameterized phase through with
/// no `ksuperpose` override.
#[test]
fn parameterized_weights_leave_the_mass_unknown_but_still_normalize() {
    let out = pir("w = elementof(cartpow(nonnegreals, 2))\n\
         mix = normalize(ksuperpose(Normal, w)(mu = [-1.0, 2.0], sigma = 1.0))\n\
         y ~ mix\n");
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %unknown)) %parameterized"),
        "a parameterized weight vector gives an unknown mass and a parameterized \
         measure:\n{out}"
    );
    assert!(
        out.contains("(%measure (%domain (%scalar real)) (%mass %normalized)) %parameterized"),
        "normalize still recovers a drawable probability measure:\n{out}"
    );
}

/// §04's ancestor rule needs no `ksuperpose` override: a parameterized family
/// argument makes the mixture parameterized through the ordinary phase join.
#[test]
fn a_parameterized_family_argument_makes_a_parameterized_mixture() {
    let out = pir("w = [0.3, 1.2]\n\
         t = elementof(reals)\n\
         mix = ksuperpose(Normal, w)(mu = [1.0], sigma = t)\n");
    let line = out
        .lines()
        .find(|l| l.contains("(ksuperpose"))
        .unwrap_or_else(|| panic!("no ksuperpose line in:\n{out}"));
    assert!(
        line.contains("%parameterized"),
        "a parameterized sigma makes the applied mixture parameterized:\n{line}"
    );
}
