//! Argument SHAPE classes: a scalar, vector or matrix where a §07 or §08 row declares one.
//!
//! §08 writes a scalar parameter as `elementof(<scalar set>)` (`mu = elementof(reals)`)
//! and gives `MvNormal`'s `mu` as a vector and `cov` as a matrix. §07 "Elementary
//! functions" states "All accept scalar arguments and return scalar results", and its
//! Domains column gives vectors and matrices elsewhere. Before this check
//! `Normal(mu = v, sigma = v)` typed a measure over `reals` and `linsolve(v, v)` a vector,
//! both at exit 0. The classes come from the catalogue row, so one rule covers every row.
//!
//! Assertions read the diagnostic set, and every refusal must carry a location.

use flatppl_infer::{Diagnostic, Severity, infer};

fn errors(src: &str) -> Vec<Diagnostic> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn assert_refuses(src: &str, needle: &str) {
    let errs = errors(src);
    assert!(
        errs.iter()
            .any(|d| d.message.contains(needle) && d.node.is_some()),
        "`{src}` must refuse with a located message containing {needle:?}; got: {errs:?}"
    );
}

fn assert_clean(src: &str) {
    let errs = errors(src);
    assert!(errs.is_empty(), "`{src}` must type; got: {errs:?}");
}

/// The printed FlatPIR binding line of `name`.
fn binding(src: &str, name: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    assert!(
        diags.iter().all(|d| d.severity != Severity::Error),
        "`{src}` must type; got: {diags:?}"
    );
    let text = flatppl_flatpir::write(&m);
    text.lines()
        .find(|l| l.contains(&format!("(%bind {name} ")))
        .unwrap_or_else(|| panic!("no binding `{name}` in:\n{text}"))
        .to_string()
}

const V: &str = "v = [1.0, 2.0]\n";
const M: &str = "M = rowstack([[2.0, 0.0], [0.0, 2.0]])\n";
/// A spec-valid `%deferred` real scalar: one field of a record-state `markovchain` draw.
/// §06 makes that trajectory a table, which the type rule does not build yet.
const DEFERRED: &str = "step = s -> joint(a = Normal(mu = s.a, sigma = 1.0))\n\
                        traj ~ markovchain(step, record(a = 0.0), 3)\n\
                        q = traj[1].a\n";

// ---- §08 scalar parameters ------------------------------------------------------

#[test]
fn a_vector_where_section_08_declares_a_scalar_is_refused() {
    assert_refuses(
        &format!("{V}m = Normal(mu = v, sigma = 1.0)"),
        "`Normal`'s `mu` is a vector, but spec §08 declares it a scalar",
    );
    assert_refuses(
        &format!("{V}m = Normal(0.0, v)"),
        "`Normal`'s `sigma` is a vector",
    );
    // The remedy names the broadcast spelling.
    assert_refuses(
        &format!("{V}m = Gamma(shape = v, rate = 1.0)"),
        "`Gamma.(…)`",
    );
}

#[test]
fn scalar_parameters_and_their_broadcast_still_type() {
    assert_clean("m = Normal(0.0, 1.0)");
    assert_clean(&format!("{V}m = Normal.(mu = v, sigma = 1.0)"));
    // `Uniform`'s `support` is a set, not a scalar, and stays unchecked.
    assert_clean("m = Uniform(interval(0.0, 1.0))");
}

/// §04 "Calling conventions" makes `f(table(a = x, b = y))` equivalent to
/// `f(a = x, b = y)`, so a table's columns reach a scalar parameter as vectors.
#[test]
fn a_splatted_table_column_is_checked_like_a_keyword_argument() {
    assert_refuses(
        "t = table(mu = [0.0, 1.0], sigma = [1.0, 2.0])\nm = Normal(t)",
        "`Normal`'s `mu` is a vector",
    );
    assert_clean("r = record(mu = 0.0, sigma = 1.0)\nm = Normal(r)");
}

// ---- §08 vector and matrix parameters -------------------------------------------

#[test]
fn a_scalar_where_section_08_declares_a_vector_is_refused() {
    assert_refuses(
        "m = Dirichlet(alpha = 2.0)",
        "`Dirichlet`'s `alpha` is a scalar, but spec §08 declares it a vector",
    );
    assert_refuses(
        &format!("{M}m = MvNormal(mu = 1.0, cov = M)"),
        "`MvNormal`'s `mu` is a scalar",
    );
}

#[test]
fn a_non_matrix_where_section_08_declares_a_matrix_is_refused() {
    assert_refuses(
        &format!("{V}m = MvNormal(mu = v, cov = v)"),
        "`MvNormal`'s `cov` is a vector, but spec §08 declares it a matrix",
    );
}

#[test]
fn vector_and_matrix_parameters_still_type() {
    assert_clean("m = Dirichlet(alpha = [1.0, 2.0])");
    assert_clean(&format!("{V}{M}m = MvNormal(mu = v, cov = M)"));
    assert_clean("m = Multinomial(n = 3, p = [0.25, 0.75])");
}

// ---- §07 scalar parameters ------------------------------------------------------

#[test]
fn a_vector_where_section_07_declares_a_scalar_is_refused() {
    assert_refuses(
        &format!("{V}x = sqrt(v)"),
        "`sqrt`'s `x` is a vector, but spec §07 declares it a scalar. Apply `sqrt` \
         elementwise with `sqrt.(…)`",
    );
    assert_refuses(
        &format!("{V}x = atan2(v, 1.0)"),
        "`atan2`'s `y` is a vector",
    );
    assert_refuses(
        &format!("{V}x = linspace(0.0, v, 3)"),
        "`linspace`'s `to` is a vector",
    );
    assert_refuses(
        "x = ifelse([true, false], 1.0, 2.0)",
        "`ifelse`'s `cond` is a vector",
    );
    assert_refuses(&format!("{V}x = eye(v)"), "`eye`'s `n` is a vector");
}

#[test]
fn scalar_arguments_and_their_broadcast_still_type() {
    assert_clean("x = sqrt(2.0)\ny = atan2(1.0, 2.0)\nz = ifelse(true, 1.0, 2.0)");
    assert_clean(&format!("{V}x = sqrt.(v)\ny = ifelse.(v .< 1.5, 1.0, 2.0)"));
    assert_clean("x = exp(complex(1.0, 2.0))");
}

// ---- §07 vector parameters ------------------------------------------------------

#[test]
fn a_non_vector_where_section_07_declares_a_vector_is_refused() {
    assert_refuses(
        "x = l2norm(1.0)",
        "`l2norm`'s `v` is a scalar, but spec §07 declares it a vector",
    );
    assert_refuses(&format!("{M}x = cumsum(M)"), "`cumsum`'s `xs` is a matrix");
    assert_refuses(&format!("{V}x = conv(1.0, v)"), "`conv`'s `v` is a scalar");
}

/// §03 "Arrays": "The term vector will represent both non-transposed vectors
/// (one-dimensional arrays) and transposed vectors".
#[test]
fn vector_arguments_still_type_in_either_orientation() {
    assert_clean(&format!("{V}x = l2norm(v)\ny = softmax(v)\nz = cumsum(v)"));
    assert_clean(&format!("{V}x = l2norm(transpose(v))"));
}

// ---- §07 matrix parameters ------------------------------------------------------

#[test]
fn a_non_matrix_where_section_07_declares_a_matrix_is_refused() {
    assert_refuses(
        &format!("{V}x = linsolve(v, v)"),
        "`linsolve`'s `A` is a vector, but spec §07 declares it a matrix",
    );
    assert_refuses(&format!("{V}x = det(v)"), "`det`'s `A` is a vector");
    // §03 "Arrays" does not read a vector of vectors as a matrix.
    assert_refuses(
        "x = det([[1.0, 0.0], [0.0, 1.0]])",
        "`det`'s `A` is a vector of vectors, but spec §07 declares it a matrix. Spec §03",
    );
}

#[test]
fn matrix_arguments_still_type() {
    assert_clean(&format!(
        "{V}{M}x = linsolve(M, v)\ny = det(M)\nz = quadform(M, v)"
    ));
}

// ---- unresolved arguments --------------------------------------------------------

/// Only a decided mismatch is an error: `%any` (a placeholder, an unshaped load) and
/// `%deferred` are never refused.
#[test]
fn an_unresolved_argument_is_not_refused() {
    assert_clean("f = fn(sqrt(_))");
    assert_clean(
        "d = load_data(\"x.csv\", anything)\nx = sqrt(d)\nm = Normal(mu = d, sigma = 1.0)",
    );
    assert_clean(&format!(
        "{DEFERRED}x = sqrt(q)\nm = Normal(mu = q, sigma = 1.0)"
    ));
}

/// A row's nominal result presumes its argument's declared shape, so a `%deferred`
/// argument leaves the result `%deferred` instead of a confirmed scalar.
#[test]
fn a_deferred_argument_leaves_the_result_deferred() {
    let src = format!(
        "{DEFERRED}s = sqrt(q)\nf = floor(q)\nx = exp(sqrt(q))\nm = Normal(mu = q, sigma = 1.0)"
    );
    for name in ["s", "f", "x"] {
        let line = binding(&src, name);
        assert!(
            line.contains("(%meta (%deferred %stochastic %unknown)"),
            "`{name}` must stay %deferred; got: {line}"
        );
    }
    // §08 fixes a distribution's variate domain by the row, not by its parameters.
    let line = binding(&src, "m");
    assert!(
        line.contains("(%measure (%domain (%scalar real)) (%mass %normalized)) %stochastic reals)"),
        "the measure keeps its §08 domain; got: {line}"
    );
}

/// `%any` is a placeholder's honest "no constraint", so a lambda body keeps the
/// nominal type.
#[test]
fn an_any_argument_keeps_the_nominal_result() {
    let line = binding("f = fn(sqrt(_))", "f");
    assert!(
        line.contains("(%scalar real) %parametric nonnegreals) (sqrt"),
        "got: {line}"
    );
}

// ---- §07 divide: the array-scalar forms -----------------------------------------

/// §07 gives `divide` the forms "scalars, array-scalar, transposed-vector–scalar". The
/// array forms typed `%deferred` before.
#[test]
fn an_array_over_a_scalar_divides_to_its_shape() {
    let src = format!("{V}{M}a = v / 2.0\nb = transpose(v) / 2\nc = M / 2.0");
    assert!(binding(&src, "a").contains("(%meta ((%array 1 (2) (%scalar real))"));
    assert!(binding(&src, "b").contains("(%meta ((%tvector 2 (%scalar real))"));
    assert!(binding(&src, "c").contains("(%meta ((%array 2 (2 2) (%scalar real))"));
}

// ---- §09 module rows -------------------------------------------------------------

/// §09 declares a member's parameters as §07 and §08 do: `erf`'s Domains cell is
/// `reals`, and `Landau`'s `loc = elementof(reals)`.
#[test]
fn a_vector_where_section_09_declares_a_scalar_is_refused() {
    let modules = "sf = standard_module(\"special-functions\", \"0.1\")\n\
                   hep = standard_module(\"particle-physics\", \"0.1\")\n";
    assert_refuses(
        &format!("{modules}{V}y = sf.erf(v)"),
        "`sf.erf`'s `x` is a vector, but spec §09 declares it a scalar",
    );
    assert_refuses(
        &format!("{modules}{V}m = hep.Landau(loc = v, scale = 1.0)"),
        "`hep.Landau`'s `loc` is a vector, but spec §09 declares it a scalar",
    );
    assert_clean(&format!(
        "{modules}{V}y = sf.erf(0.5)\nz = sf.erf.(v)\nm = hep.Landau(loc = 0.0, scale = 1.0)\n\
         n = hep.Landau.(loc = v, scale = 1.0)"
    ));
}

// ---- per-cell checks ---------------------------------------------------------------

/// A `broadcast` cell is an ordinary call of its head: over a vector of vectors each
/// cell is a vector.
#[test]
fn a_broadcast_cell_outside_the_declared_shape_is_refused() {
    let nested = "N = [[1.0, 2.0], [3.0, 4.0]]\n";
    assert_refuses(
        &format!("{nested}y = sqrt.(N)"),
        "a `broadcast` cell of `sqrt` passes `x` a vector, but spec §07 declares it a scalar",
    );
    assert_refuses(
        &format!("{nested}m = Normal.(mu = N, sigma = 1.0)"),
        "a `broadcast` cell of `Normal` passes `mu` a vector",
    );
    // `l2norm`'s cells are the inner vectors, which its vector domain admits.
    assert_clean(&format!("{nested}z = l2norm.(N)"));
    assert_clean(&format!(
        "{M}y = sqrt.(M)\nm = Normal.(mu = M, sigma = 1.0)"
    ));
}

/// A §09 head broadcasts like a built-in one (§04): a table iterates its rows and splats
/// each row record, and keyword cells obey the declared shapes as positional ones do.
#[test]
fn a_section_09_broadcast_reads_rows_keywords_and_splats() {
    let hep = "hep = standard_module(\"particle-physics\", \"0.1\")\n";
    assert_clean(&format!(
        "{hep}tb = table(loc = [0.0, 1.0], scale = [1.0, 2.0])\nm = hep.Landau.(tb)\n\
         n = Normal.(table(mu = [0.0, 1.0], sigma = [1.0, 2.0]))"
    ));
    let nested = "N = [[1.0, 2.0], [3.0, 4.0]]\n";
    for call in ["hep.Landau.(loc = N, scale = 1.0)", "hep.Landau.(N, 1.0)"] {
        assert_refuses(
            &format!("{hep}{nested}m = {call}"),
            "a `broadcast` cell of `hep.Landau` passes `loc` a vector, but spec §09 declares it a scalar",
        );
    }
}

/// A refusal inside a substituted callable body is the call's error, not a silent
/// `%failed` at exit 0.
#[test]
fn a_refusal_inside_a_substituted_body_is_reported() {
    assert_refuses(
        "k = functionof(sqrt(_x_), x = _x_)\ny = broadcast(k, x = [[1.0, 2.0], [3.0, 4.0]])",
        "`sqrt`'s `x` is a vector",
    );
    assert_refuses(
        "mk = kernelof(draw(Normal(mu = _m_, sigma = 1.0)), mu = _m_)\n\
         y = draw(mk(record(mu = 1.5)))",
        "`Normal`'s `mu` is a record",
    );
}

/// A `%failed` result dominates every row that receives it, so a refusal under a
/// substitution reaches the call's type and is reported, whichever row consumes it.
#[test]
fn a_refusal_under_any_consuming_row_is_reported() {
    assert_refuses(
        "k = functionof(Normal(mu = sqrt(_x_), sigma = 1.0), x = _x_)\ny ~ k(x = [1.0, 2.0])",
        "`sqrt`'s `x` is a vector",
    );
    for body in [
        "Normal(mu = sqrt(_x_), sigma = 1.0)",
        "ifelse(true, 1.0, sqrt(_x_))",
        "sum(sqrt(_x_))",
        "lengthof([sqrt(_x_), 1.0])",
        "checked(1.0, isfinite(sqrt(_x_)))",
    ] {
        assert_refuses(
            &format!(
                "N = [[1.0, 2.0], [3.0, 4.0]]\nk = functionof({body}, x = _x_)\n\
                 y = broadcast(k, x = N)"
            ),
            "`sqrt`'s `x` is a vector",
        );
    }
}

/// The two census models that typed at exit 0: a scalar intercept added to a vector
/// predictor inside a broadcast kernel (flatppl-js fixtures/random-intercepts.flatppl),
/// and a matrix broadcast handing `MvNormal` a scalar `mu` cell
/// (fixtures/joint-mvnormal-component.flatppl).
#[test]
fn the_census_kernel_broadcasts_are_refused() {
    assert_refuses(
        "intercepts = [0.0, 1.0]\nx_per_group = [[0.1, 0.2], [0.3, 0.4]]\n\
         k = functionof(iid(Normal(mu = _i_ + 2.0 .* _x_, sigma = 1.0), 2), i = _i_, x = _x_)\n\
         y = broadcast(k, i = intercepts, x = x_per_group)",
        "`add` operands are outside its domain",
    );
    assert_refuses(
        "mu_g = rowstack([[0.0, 0.0], [1.0, -1.0]])\ncov = rowstack([[1.0, 0.3], [0.3, 0.5]])\n\
         k = functionof(MvNormal(mu = _m_, cov = cov), m = _m_)\ny = broadcast(k, m = mu_g)",
        "`MvNormal`'s `mu` is a scalar",
    );
}

/// A nested `fn`'s `_` is its own (§04): substituting the outer argument leaves the
/// inner hole to the inner `broadcast`, so this types without error.
#[test]
fn a_nested_fn_placeholder_binds_to_the_nearest_fn() {
    assert_clean("f = fn(broadcast(fn(_ + 1), _))\ny = f([2, 3])");
}

/// §07 "Approximation functions" gives `x` the domains "real or complex",
/// `unitinterval` and "real": a scalar. An array of points is a broadcast.
#[test]
fn an_array_x_of_an_approximation_function_is_refused() {
    let c = "c = [1.0, 2.0, 3.0]\nxs = [0.1, 0.2]\n";
    assert_refuses(
        &format!("{c}y = polynomial(c, xs)"),
        "`polynomial`'s `x` is a vector",
    );
    assert_refuses(
        &format!("{c}y = bernstein(c, xs)"),
        "`bernstein`'s `x` is a vector",
    );
    assert_refuses(
        &format!("{c}y = stepwise([0.0, 1.0, 2.0], [1.0, 2.0], xs)"),
        "`stepwise`'s `x` is a vector",
    );
    assert_clean(&format!(
        "{c}y = polynomial(c, 0.5)\nz = broadcast(fn(polynomial(c, _)), xs)"
    ));
}
