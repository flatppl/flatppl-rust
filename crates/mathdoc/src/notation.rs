//! One notation key for document exports and the viewer JSON response.
//! Render example signatures through the same lowering as model expressions.
//! A note is prose with inline mathematics, so every consumer prints its
//! parameter letters with the printer it uses for the rows.

use std::collections::BTreeSet;

use flatppl_core::{CallHead, Module, Node};

use crate::{
    ast::{Math, Sym},
    lower::apply_builtin,
};

/// One piece of a legend note.
#[derive(Clone, Debug, PartialEq)]
pub enum NoteSegment {
    Text(String),
    Math(Math),
}

#[derive(Clone, Debug, PartialEq)]
pub struct NotationEntry {
    pub form: Math,
    /// The builtin the entry explains (`Normal`, `lawof`, `reals`), a key for
    /// hosts; the math view never shows a code spelling beside the notation.
    pub name: String,
    /// Prose with inline mathematics; rendered per target by
    /// [`NotationEntry::note_html`] and the export formats.
    pub note: Vec<NoteSegment>,
}

impl NotationEntry {
    /// The note as an HTML fragment: text escaped, mathematics as inline
    /// `<math>` written by the MathML printer.
    pub fn note_html(&self) -> String {
        self.note
            .iter()
            .map(|seg| match seg {
                NoteSegment::Text(t) => crate::mathml::escape(t),
                NoteSegment::Math(m) => format!("<math>{}</math>", crate::mathml::expr(m)),
            })
            .collect()
    }

    /// The note with each mathematical piece rendered by `math` and each
    /// text piece by `text` (an export format's own escaping).
    pub fn note_with(
        &self,
        text: impl Fn(&str) -> String,
        math: impl Fn(&Math) -> String,
    ) -> String {
        self.note
            .iter()
            .map(|seg| match seg {
                NoteSegment::Text(t) => text(t),
                NoteSegment::Math(m) => math(m),
            })
            .collect()
    }
}

fn text(s: &str) -> NoteSegment {
    NoteSegment::Text(s.to_string())
}

fn math(m: Math) -> NoteSegment {
    NoteSegment::Math(m)
}

fn param(name: &str) -> Math {
    Math::ident(name, None)
}

/// The legend entries for `module`: one per distribution, law, `reals` and
/// — when a row is a set function `ν(𝘈) = ∫_𝘈 …` — the set letter 𝘈.
pub fn entries(module: &Module, rows: &[crate::render::BindingRender]) -> Vec<NotationEntry> {
    let catalogue = flatppl_infer::builtin_catalogue();
    let mut names = BTreeSet::new();
    if rows
        .iter()
        .any(|b| contains_sym(&b.statement.lhs, Sym::MeasurableSet))
    {
        names.insert("measurable-set");
    }
    let mut stack: Vec<_> = module.bindings().map(|(_, b)| b.rhs).collect();
    let mut visited = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let name = match module.node(id) {
            Node::Call(call) => match call.head {
                CallHead::Builtin(head) => Some(module.resolve(head)),
                CallHead::User(_) => None,
            },
            Node::Const(sym) => Some(module.resolve(*sym)),
            _ => None,
        };
        if let Some(name) = name
            && (catalogue.base_is_distribution(name)
                || matches!(name, "lawof" | "kernelof" | "reals"))
        {
            names.insert(if name == "kernelof" { "lawof" } else { name });
        }
        module.for_each_child(id, |c| stack.push(c));
    }
    names.into_iter().map(entry).collect()
}

/// Whether `m` holds the symbol `sym` anywhere.
fn contains_sym(m: &Math, sym: Sym) -> bool {
    let mut stack = vec![m];
    while let Some(n) = stack.pop() {
        if matches!(n, Math::Sym(s) if *s == sym) {
            return true;
        }
        stack.extend(n.children());
    }
    false
}

fn entry(name: &str) -> NotationEntry {
    let (form, note) = match name {
        "measurable-set" => (
            Math::Sym(Sym::MeasurableSet),
            vec![text("A generic measurable set.")],
        ),
        "reals" => (
            Math::Sym(Sym::ExtendedReals),
            vec![text("Extended real numbers, including both infinities.")],
        ),
        "lawof" => (
            apply_builtin(name, vec![Math::letter('X')]),
            vec![
                text("Probability law (distribution) of "),
                math(Math::letter('X')),
                text("."),
            ],
        ),
        _ => {
            let params = flatppl_infer::distribution_param_names(name).unwrap_or_default();
            // Conventional letters where the parameter names are words
            // (`shape`, `rate`, `location`); the note names each by its
            // letter.
            let letters: Vec<&str> = match name {
                "Gamma" | "InverseGamma" => vec!["alpha", "beta"],
                "Exponential" | "Poisson" => vec!["lambda"],
                "Weibull" => vec!["k", "lambda"],
                "Cauchy" => vec!["x_0", "gamma"],
                "Laplace" => vec!["x_0", "b"],
                _ => params.iter().map(String::as_str).collect(),
            };
            let args = letters.iter().map(|p| Math::ident(p, None)).collect();
            let note = match name {
                "Normal" => vec![
                    text("Normal distribution with mean "),
                    math(param("mu")),
                    text(" and variance "),
                    math(Math::pow(param("sigma"), Math::int(2))),
                    text("."),
                ],
                "MvNormal" => vec![
                    text("Multivariate normal distribution with mean vector "),
                    math(param("mu")),
                    text(" and covariance matrix "),
                    math(param("Sigma")),
                    text("."),
                ],
                "StudentT" => vec![
                    text("Student t distribution with "),
                    math(param("nu")),
                    text(" degrees of freedom."),
                ],
                "ChiSquared" => vec![
                    text("Chi-squared distribution with "),
                    math(param("k")),
                    text(" degrees of freedom."),
                ],
                "Gamma" => vec![
                    text("Gamma distribution with shape "),
                    math(param("alpha")),
                    text(" and rate "),
                    math(param("beta")),
                    text("."),
                ],
                "Exponential" => vec![
                    text("Exponential distribution with rate "),
                    math(param("lambda")),
                    text("."),
                ],
                "Poisson" => vec![
                    text("Poisson distribution with rate "),
                    math(param("lambda")),
                    text("."),
                ],
                "Cauchy" => vec![
                    text("Cauchy distribution with location "),
                    math(param("x_0")),
                    text(" and scale "),
                    math(param("gamma")),
                    text("."),
                ],
                "Laplace" => vec![
                    text("Laplace distribution with location "),
                    math(param("x_0")),
                    text(" and scale "),
                    math(param("b")),
                    text("."),
                ],
                "Logistic" => vec![
                    text("Logistic distribution with location "),
                    math(param("mu")),
                    text(" and scale "),
                    math(param("s")),
                    text("."),
                ],
                "Beta" => vec![
                    text("Beta distribution with shape parameters "),
                    math(param("alpha")),
                    text(" and "),
                    math(param("beta")),
                    text("."),
                ],
                "Dirichlet" => vec![
                    text("Dirichlet distribution with concentration vector "),
                    math(param("alpha")),
                    text("."),
                ],
                "Binomial" => vec![
                    text("Binomial distribution with "),
                    math(param("n")),
                    text(" trials and success probability "),
                    math(param("p")),
                    text("."),
                ],
                "Bernoulli" => vec![
                    text("Bernoulli distribution with success probability "),
                    math(param("p")),
                    text("."),
                ],
                "LogNormal" => vec![
                    text("Log-normal distribution with log-mean "),
                    math(param("mu")),
                    text(" and log-standard deviation "),
                    math(param("sigma")),
                    text("."),
                ],
                "InverseGamma" => vec![
                    text("Inverse-gamma distribution with shape "),
                    math(param("alpha")),
                    text(" and scale "),
                    math(param("beta")),
                    text("."),
                ],
                "Weibull" => vec![
                    text("Weibull distribution with shape "),
                    math(param("k")),
                    text(" and scale "),
                    math(param("lambda")),
                    text("."),
                ],
                "Uniform" => vec![
                    text("Uniform distribution on the set "),
                    math(param("S")),
                    text("."),
                ],
                // Anything else: the distribution and its parameters by the
                // letters of the form, in the source order.
                _ => {
                    let mut note = vec![text(&format!(
                        "{name} distribution with parameter{} ",
                        if letters.len() == 1 { "" } else { "s" }
                    ))];
                    for (i, l) in letters.iter().enumerate() {
                        if i > 0 {
                            note.push(text(if i + 1 == letters.len() {
                                " and "
                            } else {
                                ", "
                            }));
                        }
                        note.push(math(param(l)));
                    }
                    note.push(text("."));
                    note
                }
            };
            (apply_builtin(name, args), note)
        }
    };
    NotationEntry {
        form,
        name: name.to_string(),
        note,
    }
}
