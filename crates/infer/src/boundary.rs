//! Spec §04 "Calling conventions": whether a sole record or table argument is
//! bound whole to a callable's one positional input or splatted over its inputs.
//!
//! The choice is made from the callable's input interface, before the argument's
//! domain is looked at (see [`crate::callable::Interface::takes_one`]). A callable
//! that takes one positional input receives the record whole, even when that
//! input's domain then rejects it. A callable that does not splats, and a field
//! or column that names no input is a static error.
//!
//! The checks here cover the places that apply the convention without a call
//! expression: the steps of `kchain` and `jointchain`. A user call is checked in
//! `ops`, which uses [`whole_binding_mismatch`]. The parameter point of a likelihood
//! and the `bayesupdate` prior's variate do not follow the convention: spec §06
//! `likelihoodof` binds the point, a record, to every input by name, one input
//! included.

use flatppl_core::{CallHead, Dim, Node, NodeId, RefNs, Symbol, Type};

use crate::callable::{Interface, interface};
use crate::ops::{
    ArgInfo, NamedInfo, cat_compose, component_variate, input_entries, local_reification,
    resolve_binding_refs,
};
use crate::trace::Inferencer;

/// The declared domain of each input of the reified callable `callee`, in input
/// order. `None` for an input that declares none: a lambda placeholder, or a
/// boundary whose binding has no concrete value type yet.
pub(crate) fn declared_input_types(inf: &Inferencer<'_, '_>, callee: NodeId) -> Vec<Option<Type>> {
    let Some((reif, _)) = local_reification(inf, callee) else {
        return Vec::new();
    };
    let Some(entries) = input_entries(inf, reif) else {
        return Vec::new();
    };
    entries
        .iter()
        .map(|(_, decl)| {
            if decl.ns != RefNs::SelfMod {
                return None;
            }
            let binding = inf.module.binding_by_name(decl.name)?;
            let ty = inf.lookup_type(inf.module.binding(binding).rhs)?;
            is_concrete_value(ty).then(|| ty.clone())
        })
        .collect()
}

fn is_concrete_value(ty: &Type) -> bool {
    match ty {
        Type::Scalar(_) => true,
        Type::Array { elem, .. } | Type::TVector { elem, .. } => is_concrete_value(elem),
        Type::Record(fields) => fields.iter().all(|(_, t)| is_concrete_value(t)),
        Type::Tuple(parts) => parts.iter().all(is_concrete_value),
        Type::Table { columns, .. } => columns.iter().all(|(_, t)| is_concrete_value(t)),
        _ => false,
    }
}

/// True only when no value of type `actual` can lie in a domain of type
/// `expected`. Scalar kinds are not compared: an integer is a real and the
/// kinds are decided elsewhere. Anything undecided is compatible.
fn incompatible(expected: &Type, actual: &Type) -> bool {
    use Type::*;
    match (expected, actual) {
        (Scalar(_), Scalar(_)) => false,
        (Scalar(_), Array { .. } | TVector { .. } | Record(_) | Tuple(_) | Table { .. }) => true,
        (Array { .. } | TVector { .. }, Scalar(_) | Record(_) | Tuple(_) | Table { .. }) => true,
        (Record(_), Scalar(_) | Array { .. } | TVector { .. } | Tuple(_) | Table { .. }) => true,
        (Tuple(_), Scalar(_) | Array { .. } | TVector { .. } | Record(_) | Table { .. }) => true,
        (Table { .. }, Scalar(_) | Array { .. } | TVector { .. } | Record(_) | Tuple(_)) => true,
        (Array { shape: a, elem: x }, Array { shape: b, elem: y }) => {
            a.len() != b.len()
                || a.iter().zip(b.iter()).any(|pair| match pair {
                    (Dim::Static(p), Dim::Static(q)) => p != q,
                    _ => false,
                })
                || incompatible(x, y)
        }
        (TVector { len: a, elem: x }, TVector { len: b, elem: y }) => {
            matches!((a, b), (Dim::Static(p), Dim::Static(q)) if p != q) || incompatible(x, y)
        }
        (Record(a), Record(b)) => {
            a.len() != b.len()
                || a.iter()
                    .any(|(name, t)| match b.iter().find(|(n, _)| n == name) {
                        Some((_, u)) => incompatible(t, u),
                        None => true,
                    })
        }
        (Tuple(a), Tuple(b)) => {
            a.len() != b.len() || a.iter().zip(b.iter()).any(|(t, u)| incompatible(t, u))
        }
        _ => false,
    }
}

/// A short FlatPPL-flavoured spelling of a type for a diagnostic.
fn show(inf: &Inferencer<'_, '_>, ty: &Type) -> String {
    match ty {
        Type::Scalar(kind) => kind.to_string(),
        Type::Array { shape, elem } => {
            let dims: Vec<String> = shape.iter().map(Dim::to_string).collect();
            format!("array of {} [{}]", show(inf, elem), dims.join(", "))
        }
        Type::TVector { len, elem } => format!("transposed vector of {} [{len}]", show(inf, elem)),
        Type::Record(fields) => format!(
            "record({})",
            fields
                .iter()
                .map(|(n, t)| format!("{}: {}", inf.module.resolve(*n), show(inf, t)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::Tuple(parts) => format!(
            "tuple({})",
            parts
                .iter()
                .map(|t| show(inf, t))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::Table { columns, .. } => format!(
            "table({})",
            columns
                .iter()
                .map(|(n, t)| format!("{}: {}", inf.module.resolve(*n), show(inf, t)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => "an undecided type".to_string(),
    }
}

fn is_aggregate(ty: &Type) -> bool {
    matches!(ty, Type::Record(_) | Type::Table { .. })
}

const WHOLE_HINT: &str = "; the callable takes one positional input, so a record or table \
    is bound whole, not splatted (spec §04 \"Calling conventions\")";

/// The message for a sole argument bound whole to the callable's first input
/// whose declared domain rejects it, or `None` when it is admitted or undecided.
///
/// `declared` is [`declared_input_types`] of the callee, `names` its input names.
pub(crate) fn whole_binding_mismatch(
    inf: &Inferencer<'_, '_>,
    names: &[String],
    declared: &[Option<Type>],
    actual: &Type,
) -> Option<String> {
    let expected = declared.first()?.as_ref()?;
    if !incompatible(expected, actual) {
        return None;
    }
    let hint = if is_aggregate(actual) { WHOLE_HINT } else { "" };
    Some(format!(
        "bound whole to input `{}`, which expects {}, got {}{hint}",
        names.first().map_or("", String::as_str),
        show(inf, expected),
        show(inf, actual),
    ))
}

fn field_names(inf: &Inferencer<'_, '_>, ty: &Type) -> Option<Vec<(Symbol, String)>> {
    let fields = match ty {
        Type::Record(fields) => fields,
        Type::Table { columns, .. } => columns,
        _ => return None,
    };
    Some(
        fields
            .iter()
            .map(|(n, _)| (*n, inf.module.resolve(*n).to_string()))
            .collect(),
    )
}

/// `densityof(L, point)`, `logdensityof(L, point)` and `bayesupdate(L, prior)`
/// evaluate the likelihood's kernel at the point. For `bayesupdate` the point is the
/// prior's variate. The point is a record bound to the kernel's inputs by name, for
/// any number of inputs (spec §06 `likelihoodof`); see [`check_point`].
pub(crate) fn refuse_bad_likelihood_point(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
) -> Option<Type> {
    let (point_ty, what) = match name {
        "densityof" | "logdensityof" => (args.get(1)?.1.clone(), "point"),
        "bayesupdate" => match &args.get(1)?.1 {
            Type::Measure { domain, .. } => ((**domain).clone(), "prior's variate"),
            _ => return None,
        },
        _ => return None,
    };
    let (likelihood, ..) = args.first()?;
    if !matches!(args[0].1, Type::Likelihood { .. }) {
        return None;
    }
    let at = args[1].0;
    let interface = match interface(inf, *likelihood) {
        Some(interface) => interface,
        None => joint_likelihood_interface(inf, &args[0])?,
    };
    check_point(inf, name, what, at, &interface, &point_ty)
}

/// `joint_likelihood(L1, L2, ...)` is `L(θ) = L1(θ) + L2(θ)` and is equivalent to
/// `likelihoodof(joint(K1, K2, ...), obs)` (spec §06 "Combining likelihoods"). A
/// `joint` of kernels takes "the union of the component kernels' inputs by name,
/// so the result only supports keyword arguments" (§06 "Joint measures"), so a
/// record point splats over that union when every term splats.
///
/// `None` when a term's interface is unknown: the union is not an interface then.
fn joint_likelihood_interface(inf: &Inferencer<'_, '_>, likelihood: &ArgInfo) -> Option<Interface> {
    let node = resolve_binding_refs(inf, likelihood.0);
    let Node::Call(call) = inf.module.node(node) else {
        return None;
    };
    let CallHead::Builtin(op) = call.head else {
        return None;
    };
    if inf.module.resolve(op) != "joint_likelihood" || call.args.is_empty() {
        return None;
    }
    let mut names: Vec<String> = Vec::new();
    for &term in call.args.iter() {
        let term_interface = interface(inf, term)?;
        for name in term_interface.names {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    Some(Interface::required(names, false))
}

/// The parameter point of a likelihood is a record with one field per input of the
/// kernel, "bound to those inputs by name for any number of inputs, so a kernel with
/// the single input `mu` is evaluated at `record(mu = 1.0)`, not at `1.0`" (spec §06
/// `likelihoodof`). A point of any other kind, a field that names no input, and an
/// input without a field are each a located static error. An undecided type is
/// never refused.
fn check_point(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    what: &str,
    at: NodeId,
    interface: &Interface,
    point: &Type,
) -> Option<Type> {
    let list = |names: &[&str]| {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let inputs: Vec<&str> = interface.names.iter().map(String::as_str).collect();
    let Type::Record(_) = point else {
        if matches!(
            point,
            Type::Deferred | Type::Any | Type::Var(_) | Type::Failed(_)
        ) {
            return None;
        }
        let example = inputs.first().map_or_else(
            || "record(x = 1.0)".to_string(),
            |n| format!("record({n} = ..)"),
        );
        inf.diags.push(crate::Diagnostic::error_at(
            at,
            format!(
                "`{name}`: the {what} is {}, but the parameter point of a likelihood is a \
                 record with one field per input ({}), bound by name for any number of inputs \
                 (spec §06 `likelihoodof`), so even a kernel with one input is evaluated at \
                 `{example}`, not at the bare value",
                show(inf, point),
                list(&inputs),
            ),
        ));
        return Some(Type::Failed(
            format!("{name}: the {what} is not a record").into(),
        ));
    };
    let fields = field_names(inf, point)?;
    let unknown: Vec<&str> = fields
        .iter()
        .map(|(_, n)| n.as_str())
        .filter(|n| !inputs.contains(n))
        .collect();
    let missing: Vec<&str> = inputs
        .iter()
        .copied()
        .filter(|n| !fields.iter().any(|(_, f)| f == n))
        .collect();
    if unknown.is_empty() && missing.is_empty() {
        return None;
    }
    let mut problems = Vec::new();
    if !unknown.is_empty() {
        problems.push(format!("has no input named {}", list(&unknown)));
    }
    if !missing.is_empty() {
        problems.push(format!("has no field for the input {}", list(&missing)));
    }
    inf.diags.push(crate::Diagnostic::error_at(
        at,
        format!(
            "`{name}`: the {what} is bound to the likelihood inputs ({}) by name, but {} (spec \
             §06 `likelihoodof`: one field per input)",
            list(&inputs),
            problems.join(" and "),
        ),
    ));
    Some(Type::Failed(
        format!("{name}: the {what} does not match the likelihood inputs").into(),
    ))
}

/// Check each step of a `kchain` or `jointchain` against the value the chain
/// feeds it: the `cat` of every variate to its left (spec §06 "Dependent
/// composition").
///
/// The keyword form `jointchain(name1 = M, name2 = K1, ...)` is `jointchain` of
/// `relabel`led components (§06 "Keyword form"): "Each later kernel receives the
/// record of the preceding named variates, bound to its inputs by name for any
/// number of inputs, so `jointchain(a = M, b = K)` calls `K(a = a)`." Every input of
/// a later kernel must name a preceding label ([`check_labelled_step`]).
///
/// The positional form keeps the ordinary calling convention: a step taking one
/// positional input receives the fed value whole and the input's declared domain
/// must admit it. Any other step splats a record, whose fields must be exactly the
/// step's inputs.
pub(crate) fn refuse_bad_chain_boundary(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    if !matches!(name, "kchain" | "jointchain") {
        return None;
    }
    // Keyword form `jointchain(name1 = M, name2 = K1, ...)`: the components are
    // the keyword values. Its fed value is the previous component's variate, not
    // the `cat` of all of them (see the doc comment).
    let labelled = args.is_empty() && !named.is_empty() && name == "jointchain";
    let steps: Vec<(NodeId, &Type)> = if labelled {
        named.iter().map(|(_, node, ty, _)| (*node, ty)).collect()
    } else if named.is_empty() {
        args.iter().map(|(node, ty, _)| (*node, ty)).collect()
    } else {
        return None;
    };
    if steps.len() < 2 {
        return None;
    }
    // A chain whose first component is a kernel is applied with additional
    // arguments that every later step receives after its fed value (§06
    // "Dependent composition"), so the step's whole interface is not the fed
    // value's: the application binds each step instead.
    if matches!(steps[0].1, Type::Kernel { .. }) {
        return None;
    }
    let labels: Vec<String> = if labelled {
        named
            .iter()
            .map(|(label, ..)| inf.module.resolve(*label).to_string())
            .collect()
    } else {
        Vec::new()
    };
    let mut variates: Vec<Type> = Vec::with_capacity(steps.len());
    for (index, (node, ty)) in steps.iter().enumerate() {
        if labelled
            && index > 0
            && matches!(ty, Type::Kernel { .. })
            && let Some(interface) = interface(inf, *node)
            && let Some(failed) =
                check_labelled_step(inf, index, *node, &interface, &labels[..index], &variates)
        {
            return Some(failed);
        }
        if !labelled
            && index > 0
            && matches!(ty, Type::Kernel { .. })
            && let Some(interface) = interface(inf, *node)
        {
            // One variate to the left is fed as it is; two or more are `cat`ed.
            let fed = match variates.as_slice() {
                [only] => only.clone(),
                _ => cat_compose(&variates),
            };
            if !matches!(fed, Type::Deferred)
                && let Some(failed) = check_step(inf, name, index, *node, &interface, &fed)
            {
                return Some(failed);
            }
        }
        variates.push(component_variate(inf, *node, ty)?);
    }
    None
}

/// A later kernel of a keyword-form `jointchain` receives the record of the preceding
/// named variates, "bound to its inputs by name for any number of inputs" (§06
/// "Keyword form"). Each input must name a preceding label, and its declared domain
/// must admit that label's variate.
fn check_labelled_step(
    inf: &mut Inferencer<'_, '_>,
    index: usize,
    step: NodeId,
    interface: &Interface,
    labels: &[String],
    variates: &[Type],
) -> Option<Type> {
    let declared = declared_input_types(inf, step);
    let list = |names: Vec<&str>| {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let unmatched: Vec<&str> = interface
        .names
        .iter()
        .map(String::as_str)
        .filter(|n| !labels.iter().any(|l| l == n))
        .collect();
    let message = if !unmatched.is_empty() {
        format!(
            "`jointchain` step {index}: the kernel's input {} names no preceding component \
             ({}); in the keyword form a later kernel receives the record of the preceding \
             named variates, bound to its inputs by name (spec §06 \"Keyword form\")",
            list(unmatched),
            list(labels.iter().map(String::as_str).collect()),
        )
    } else {
        let mismatch = interface.names.iter().enumerate().find_map(|(i, input)| {
            let at = labels.iter().position(|l| l == input)?;
            let expected = declared.get(i)?.as_ref()?;
            incompatible(expected, variates.get(at)?)
                .then(|| (input.clone(), expected.clone(), variates[at].clone()))
        });
        let (input, expected, fed) = mismatch?;
        format!(
            "`jointchain` step {index}: the variate of `{input}` is {}, but the kernel's input \
             `{input}` expects {} (spec §06 \"Keyword form\")",
            show(inf, &fed),
            show(inf, &expected),
        )
    };
    inf.diags.push(crate::Diagnostic::error_at(step, message));
    Some(Type::Failed(
        format!("jointchain: step {index} does not bind by name").into(),
    ))
}

fn check_step(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    index: usize,
    step: NodeId,
    interface: &Interface,
    fed: &Type,
) -> Option<Type> {
    let declared = declared_input_types(inf, step);
    let message = if interface.takes_one() {
        let expected = declared.first().and_then(Option::as_ref)?;
        if !incompatible(expected, fed) {
            return None;
        }
        format!(
            "`{name}` step {index}: cannot unify the value fed from the previous steps, {}, \
             with the step's input `{}`, which expects {} (spec §06 \"Dependent composition\": \
             a step receives the `cat` of every variate to its left; it takes one positional \
             input, so a record is bound whole, spec §04 \"Calling conventions\")",
            show(inf, fed),
            interface.names.first().map_or("", String::as_str),
            show(inf, expected),
        )
    } else {
        let Some(fields) = field_names(inf, fed) else {
            let step_inputs: Vec<String> =
                interface.names.iter().map(|n| format!("`{n}`")).collect();
            let message = format!(
                "`{name}` step {index}: the step does not take one positional input \
                 (inputs: {}), so it needs a record to splat, but the previous steps feed \
                 {} (spec §04 \"Calling conventions\")",
                step_inputs.join(", "),
                show(inf, fed),
            );
            inf.diags.push(crate::Diagnostic::error_at(step, message));
            return Some(Type::Failed(
                format!("{name}: step {index} needs a record").into(),
            ));
        };
        let extra: Vec<&str> = fields
            .iter()
            .map(|(_, n)| n.as_str())
            .filter(|n| !interface.names.iter().any(|i| i == n))
            .collect();
        let missing: Vec<&str> = interface
            .names
            .iter()
            .map(String::as_str)
            .filter(|n| !fields.iter().any(|(_, f)| f == n))
            .collect();
        if extra.is_empty() && missing.is_empty() {
            return None;
        }
        let part = |label: &str, names: &[&str]| {
            if names.is_empty() {
                String::new()
            } else {
                format!(
                    " {label} {}",
                    names
                        .iter()
                        .map(|n| format!("`{n}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        };
        format!(
            "`{name}` step {index}: the fed record {} is splatted over the step's inputs, but \
             it{}{} (spec §04 \"Calling conventions\": field names that do not match the \
             callable's argument names are a static error)",
            show(inf, fed),
            part("has no input for field", &extra),
            part(
                if extra.is_empty() {
                    "lacks field"
                } else {
                    "and lacks field"
                },
                &missing
            ),
        )
    };
    inf.diags.push(crate::Diagnostic::error_at(step, message));
    Some(Type::Failed(
        format!("{name}: step {index} boundary").into(),
    ))
}
