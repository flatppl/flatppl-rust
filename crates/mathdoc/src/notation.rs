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
    pub source: String,
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

pub fn entries(module: &Module) -> Vec<NotationEntry> {
    let catalogue = flatppl_infer::builtin_catalogue();
    let mut names = BTreeSet::new();
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

fn entry(name: &str) -> NotationEntry {
    let (form, source, note) = match name {
        "reals" => (
            Math::Sym(Sym::ExtendedReals),
            "reals".to_string(),
            vec![text("Extended real numbers, including both infinities.")],
        ),
        "lawof" => (
            apply_builtin(name, vec![Math::letter('X')]),
            "lawof(X)".to_string(),
            vec![
                text("Probability law (distribution) of "),
                math(Math::letter('X')),
                text("."),
            ],
        ),
        _ => {
            let params = flatppl_infer::distribution_param_names(name).unwrap_or_default();
            let args = params.iter().map(|p| Math::ident(p, None)).collect();
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
                "Gamma" | "Exponential" => vec![text("Rate parameterisation.")],
                "InverseGamma" | "Weibull" => vec![text("Scale parameterisation.")],
                "Uniform" => vec![
                    text("Uniform distribution on the set "),
                    math(param("S")),
                    text("."),
                ],
                _ => vec![text("Arguments follow the source parameter order.")],
            };
            (
                apply_builtin(name, args),
                format!("{name}({})", params.join(", ")),
                note,
            )
        }
    };
    NotationEntry { form, source, note }
}
