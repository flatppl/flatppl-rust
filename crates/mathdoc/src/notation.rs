//! One notation key for document exports and the viewer JSON response.
//! Render example signatures through the same lowering as model expressions.

use std::collections::BTreeSet;

use flatppl_core::{CallHead, Module, Node};

use crate::{
    ast::{Math, Sym},
    lower::apply_builtin,
};

#[derive(Clone, Debug, PartialEq)]
pub struct NotationEntry {
    pub form: Math,
    pub source: String,
    pub note: String,
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
            "Extended real numbers, including both infinities.",
        ),
        "lawof" => (
            apply_builtin(name, vec![Math::letter('X')]),
            "lawof(X)".to_string(),
            "Probability law (distribution) of X.",
        ),
        _ => {
            let params = flatppl_infer::distribution_param_names(name).unwrap_or_default();
            let args = params.iter().map(|p| Math::ident(p, None)).collect();
            let note = match name {
                "Normal" => {
                    "Mean and variance. The source standard deviation stays explicitly squared, even when numeric."
                }
                "MvNormal" => "Mean vector and covariance matrix.",
                "StudentT" => "Degrees of freedom; zero location and unit scale.",
                "ChiSquared" => "Degrees of freedom.",
                "Gamma" | "Exponential" => "Rate parameterisation.",
                "InverseGamma" | "Weibull" => "Scale parameterisation.",
                "Uniform" => "Uniform probability measure on the displayed set.",
                _ => "Arguments follow the source parameter order.",
            };
            (
                apply_builtin(name, args),
                format!("{name}({})", params.join(", ")),
                note,
            )
        }
    };
    NotationEntry {
        form,
        source,
        note: note.to_string(),
    }
}
