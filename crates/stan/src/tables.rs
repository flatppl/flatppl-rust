//! The FlatPPL → Stan mapping tables, `tables/*.ron`, parsed once.
//!
//! `distributions.ron` maps laws and `functions.ron` maps functions; both are
//! checked against Stan 2.40 by the tables' own verifier. `sets.ron` documents
//! the declaration of each set; [`crate::stan_type`] implements it.

use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct LawRow {
    pub id: String,
    pub flatppl: String,
    pub module: Option<String>,
    pub params: Vec<String>,
    pub pattern: String,
    pub vars: Vec<String>,
    pub kind: String,
    pub stan: String,
    pub suffix: String,
    pub lhs: String,
    pub args: String,
    pub variate: String,
    pub sizes: Vec<(String, String)>,
    pub vectorized: bool,
    pub lcdf: bool,
    pub normalized: bool,
    pub guard: String,
    /// The pair the interval-mass helper takes far in the upper tail:
    /// `lccdf`, or `mirror_lcdf` for a symmetric law. Set only where a
    /// BridgeStan probe matched scipy; absent rows keep `T[lo, hi]`.
    #[serde(default)]
    pub tail: String,
    /// The argument (`{scale}`) that is the support's lower end, where the
    /// variate type has no literal one: the helper clamps the interval to it.
    #[serde(default)]
    pub floor: String,
    pub functions: String,
    pub status: String,
    pub note: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct FnRow {
    pub id: String,
    pub flatppl: String,
    pub module: Option<String>,
    pub args: Vec<String>,
    pub kind: String,
    pub stan: String,
    pub functions: String,
    pub data_only: bool,
    pub vectorized: bool,
    pub broadcast: String,
    pub caveat: String,
    pub status: String,
    /// The argument kinds of the row's vector overload (`vector, vector,
    /// vector, real`): a broadcast `f.(…)` whose argument ranks match is one
    /// call of it. Empty: no vector overload.
    #[serde(default)]
    pub overload: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct SetRow {
    pub set: String,
    pub block: String,
    pub decl: String,
    pub chart: String,
    pub status: String,
    pub note: String,
}

fn parse<T: for<'de> Deserialize<'de>>(name: &str, text: &str) -> Vec<T> {
    ron::from_str(text).unwrap_or_else(|e| panic!("tables/{name}: {e}"))
}

pub(crate) fn laws() -> &'static [LawRow] {
    static T: OnceLock<Vec<LawRow>> = OnceLock::new();
    T.get_or_init(|| {
        parse(
            "distributions.ron",
            include_str!("../tables/distributions.ron"),
        )
    })
}

pub(crate) fn functions() -> &'static [FnRow] {
    static T: OnceLock<Vec<FnRow>> = OnceLock::new();
    T.get_or_init(|| parse("functions.ron", include_str!("../tables/functions.ron")))
}

#[cfg(test)]
pub(crate) fn sets() -> &'static [SetRow] {
    static T: OnceLock<Vec<SetRow>> = OnceLock::new();
    T.get_or_init(|| parse("sets.ron", include_str!("../tables/sets.ron")))
}

/// The law rows whose pattern's outer call is `head`, most specific
/// first. `locscale(StudentT(nu), mu, sigma)` is found under `locscale`.
pub(crate) fn law_rows_by_head(head: &str, module: Option<&str>) -> Vec<&'static LawRow> {
    let mut rows: Vec<&LawRow> = laws()
        .iter()
        .filter(|r| {
            module_matches(r.module.as_deref(), module) && r.pattern.split('(').next() == Some(head)
        })
        .collect();
    rows.sort_by_key(|r| r.kind != "composition");
    rows
}

/// The row for a function. One `functions.ron` row may list several
/// standard-module members (`"legendre, hermite"`).
pub(crate) fn function(name: &str, module: Option<&str>) -> Option<&'static FnRow> {
    functions().iter().find(|r| {
        module_matches(r.module.as_deref(), module) && r.flatppl.split(", ").any(|n| n == name)
    })
}

/// A table module is `"<name> <version>"`; a call names the module only.
fn module_matches(row: Option<&str>, call: Option<&str>) -> bool {
    match (row, call) {
        (None, None) => true,
        (Some(r), Some(c)) => r.split(' ').next() == Some(c),
        _ => false,
    }
}

/// The Stan Math library function names (`tables/library.txt`).
fn library() -> &'static std::collections::HashSet<&'static str> {
    static L: std::sync::OnceLock<std::collections::HashSet<&'static str>> =
        std::sync::OnceLock::new();
    L.get_or_init(|| {
        include_str!("../tables/library.txt")
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect()
    })
}

/// Suffixes Stan gives densities, distribution functions and generators.
const SUFFIXES: &[&str] = &["_lpdf", "_lpmf", "_lcdf", "_lccdf", "_rng", "_lp"];

/// `true` when a user function must not take `name`: a library function, the
/// distribution behind one (`normal` of `normal_lpdf`), a reserved word, or a
/// name with a Stan suffix.
pub(crate) fn clashes_with_stan(name: &str) -> bool {
    let lib = library();
    lib.contains(name)
        || RESERVED.contains(&name)
        || SUFFIXES.iter().any(|s| name.ends_with(s))
        || SUFFIXES
            .iter()
            .any(|s| lib.contains(format!("{name}{s}").as_str()))
}

/// Stan keywords and reserved names; a FlatPPL name in this list gets a
/// trailing underscore.
pub(crate) const RESERVED: &[&str] = &[
    "for",
    "in",
    "while",
    "repeat",
    "until",
    "if",
    "then",
    "else",
    "true",
    "false",
    "target",
    "functions",
    "model",
    "data",
    "parameters",
    "quantities",
    "transformed",
    "generated",
    "int",
    "real",
    "complex",
    "vector",
    "row_vector",
    "matrix",
    "array",
    "tuple",
    "simplex",
    "ordered",
    "positive_ordered",
    "unit_vector",
    "sum_to_zero_vector",
    "sum_to_zero_matrix",
    "cholesky_factor_corr",
    "cholesky_factor_cov",
    "corr_matrix",
    "cov_matrix",
    "column_stochastic_matrix",
    "row_stochastic_matrix",
    "void",
    "return",
    "break",
    "continue",
    "print",
    "reject",
    "fatal_error",
    "lower",
    "upper",
    "offset",
    "multiplier",
    "profile",
    "struct",
    "typedef",
    "export",
    "auto",
    "extern",
    "var",
    "static",
    "lp__",
    "jacobian",
];

#[cfg(test)]
mod tests {
    #[test]
    fn tables_parse() {
        assert!(super::laws().len() > 40);
        assert!(super::functions().len() > 100);
        assert!(super::sets().len() > 10);
        assert_eq!(
            super::law_rows_by_head("Bernoulli", None)[0].kind,
            "composition"
        );
    }
}
