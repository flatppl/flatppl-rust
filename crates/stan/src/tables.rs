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
    /// The Stan function does not enforce the `variate` support (von Mises
    /// returns a finite value outside (-pi, pi), `lkj_corr_cholesky` for rows
    /// that are not unit length): a variate whose declaration does not state
    /// it is tested against it. Set where a BridgeStan probe outside the
    /// support returned a finite value.
    #[serde(default)]
    pub unchecked_support: bool,
    /// The Stan function returns -inf outside the `variate` support, at
    /// every end (a user function that tests its own support): a mixture
    /// component needs no support test. Set where a BridgeStan probe outside
    /// each end returned -inf.
    #[serde(default)]
    pub total_support: bool,
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
    /// The argument kinds of the row's array overloads, `;`-separated
    /// (`vector, vector, vector, real; matrix, matrix, matrix, column`): a
    /// broadcast `f.(…)` whose arguments match one is one call of it. A
    /// `column` is `addaxes(u, 0, 1)`, passed as `u`. Empty: no array overload.
    #[serde(default)]
    pub overload: String,
    /// The `core` rows whose definitions this row's `functions` text calls:
    /// a program prints them first, each once.
    #[serde(default)]
    pub uses: Vec<String>,
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

/// The `core` row `id`: Stan definitions that other rows share.
pub(crate) fn core(id: &str) -> Option<&'static FnRow> {
    functions().iter().find(|r| r.kind == "core" && r.id == id)
}

/// A `functions` text split into its top-level definitions.
pub(crate) fn definitions(text: &str) -> Vec<String> {
    let mut defs: Vec<String> = Vec::new();
    let (mut cur, mut depth) = (String::new(), 0i32);
    for line in text.lines() {
        cur.push_str(line);
        cur.push('\n');
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        if depth == 0 && line.contains('}') {
            defs.push(cur.trim_end().to_string());
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        defs.push(cur.trim_end().to_string());
    }
    defs
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
    "complex_vector",
    "complex_row_vector",
    "complex_matrix",
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

    /// A Stan definition lives in one row: rows that share one name a `core`
    /// row in `uses`, so two texts of one signature cannot drift apart.
    #[test]
    fn each_definition_lives_in_one_row() {
        let mut seen = std::collections::HashMap::new();
        let texts = super::functions()
            .iter()
            .map(|r| (r.id.as_str(), r.functions.as_str()))
            .chain(
                super::laws()
                    .iter()
                    .map(|r| (r.id.as_str(), r.functions.as_str())),
            );
        for (id, text) in texts {
            // The split counts braces, so a comment must hold none.
            for line in text.lines() {
                if let Some((_, comment)) = line.split_once("//") {
                    assert!(
                        !comment.contains(['{', '}']),
                        "row `{id}`: a brace in the comment `{comment}` breaks the definition split"
                    );
                }
            }
            for d in super::definitions(text) {
                let signature = d.lines().next().unwrap_or_default().to_string();
                if let Some(other) = seen.insert(signature.clone(), id) {
                    panic!("`{signature}` is defined in rows `{other}` and `{id}`");
                }
            }
        }
        for r in super::functions() {
            for u in &r.uses {
                assert!(
                    super::core(u).is_some(),
                    "row `{}` uses `{u}`, not a core row",
                    r.id
                );
            }
        }
    }

    /// No golden program holds a template placeholder of a law row
    /// (`{resonance}`) that no `fill` replaced: every row bound or guard
    /// prints through the law's arguments.
    #[test]
    fn goldens_hold_no_unfilled_placeholder() {
        let mut names: Vec<String> = Vec::new();
        for r in super::laws() {
            names.extend(r.vars.iter().cloned());
            names.extend(r.params.iter().cloned());
            names.extend(r.sizes.iter().map(|(n, _)| n.clone()));
        }
        names.sort();
        names.dedup();
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        let mut seen = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            if path.extension().is_none_or(|x| x != "stan") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            seen += 1;
            for n in &names {
                // A one-element array of a declared variable (`{x}`) is Stan.
                let declared =
                    text.contains(&format!(" {n};")) || text.contains(&format!(" {n} ="));
                assert!(
                    declared || !text.contains(&format!("{{{n}}}")),
                    "{}: unfilled placeholder `{{{n}}}`",
                    path.display()
                );
            }
        }
        assert!(seen > 100, "goldens read: {seen}");
    }
}
