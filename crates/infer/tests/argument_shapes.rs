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
        "`det`'s `A` is a vector of arrays, but spec §07 declares it a matrix. Spec §03",
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
    assert_clean("n = [[1.0, 2.0], [3.0, 4.0]]\nx = sqrt(mul(n, n))");
}

/// A row's nominal result presumes its argument's declared shape, so a `%deferred`
/// argument leaves the result `%deferred` instead of a confirmed scalar.
#[test]
fn a_deferred_argument_leaves_the_result_deferred() {
    let src = "n = [[1.0, 2.0], [3.0, 4.0]]\nq = mul(n, n)\ns = sqrt(q)\nf = floor(q)\n\
               x = exp(sqrt(q))\nm = Normal(mu = q, sigma = 1.0)";
    for name in ["s", "f", "x"] {
        let line = binding(src, name);
        assert!(
            line.contains("(%meta (%deferred %fixed %unknown)"),
            "`{name}` must stay %deferred; got: {line}"
        );
    }
    // §08 fixes a distribution's variate domain by the row, not by its parameters.
    let line = binding(src, "m");
    assert!(
        line.contains("(%measure (%domain (%scalar real)) (%mass %normalized)) %fixed reals)"),
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
