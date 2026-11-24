use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{Call, Module, Node, NodeId};

use crate::{Key, Query};

pub(super) fn args(
    module: &Module,
    query: &Query<'_>,
    call: &Call,
    op: &str,
) -> Option<(&'static str, Vec<NodeId>)> {
    let &[kernel, parameters, value] = call.args.as_ref() else {
        return None;
    };
    if !call.named.is_empty() {
        return None;
    }
    let Node::Const(kernel) = module.node(kernel) else {
        return None;
    };
    let normal = &["mu", "sigma"][..];
    let cauchy = &["location", "scale"][..];
    let (rule, fields): (&str, &[&str]) = match (op, module.resolve(*kernel)) {
        ("builtin_touniform", "Normal") => ("Normal CDF", normal),
        ("builtin_touniform", "Cauchy") => ("Cauchy CDF", cauchy),
        ("builtin_touniform", "Exponential") => ("Exponential CDF", &["rate"]),
        ("builtin_touniform", "Logistic") => ("Logistic CDF", &["mu", "s"]),
        ("builtin_touniform", "LogNormal") => ("LogNormal CDF", normal),
        ("builtin_tonormal", "LogNormal") => ("LogNormal to normal", normal),
        ("builtin_fromnormal", "LogNormal") => ("LogNormal from normal", normal),
        ("builtin_tonormal", "Normal") => ("Normal to normal", normal),
        ("builtin_fromnormal", "Normal") => ("Normal from normal", normal),
        ("builtin_fromuniform", "Cauchy") => ("Cauchy quantile", cauchy),
        ("builtin_fromuniform", "Exponential") => ("Exponential quantile", &["rate"]),
        ("builtin_fromuniform", "Logistic") => ("Logistic quantile", &["mu", "s"]),
        _ => return None,
    };
    let Key::Node(parameters) = query.key(module, parameters).ok()? else {
        return None;
    };
    let record = expect_builtin_call(module, parameters, "record")?;
    let mut args = fields
        .iter()
        .map(|name| {
            record
                .named
                .iter()
                .find(|arg| module.resolve(arg.name) == *name)
                .map(|arg| arg.value)
        })
        .collect::<Option<Vec<_>>>()?;
    args.push(value);
    Some((rule, args))
}
