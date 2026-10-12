//! Prior text uses the language parser; only brms constructor names are lowered.
use crate::{Error, PriorBound, ResolvedPrior, Result};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Module, NamedArg, Node, NodeId, Ref, RefNs, Scalar};

pub(crate) enum PriorNode {
    Measure(NodeId),
    Constant(NodeId),
}

pub(crate) fn emit_correlation(
    b: &mut Builder<'_>,
    prior: &ResolvedPrior,
    dimension: usize,
    bindings: &[&str],
) -> Result<NodeId> {
    if prior.lb.is_some() || prior.ub.is_some() {
        return Err(Error(
            "correlation priors do not support scalar bounds".into(),
        ));
    }
    let PriorNode::Measure(node) = emit(b, prior, None, Some(dimension), bindings)? else {
        return Err(Error(
            "correlation priors require an LKJ law; matrix constants are unsupported".into(),
        ));
    };
    if let Node::Call(call) = b.m.node(node)
        && matches!(call.head, CallHead::Builtin(name) if b.m.resolve(name) == "LKJCholesky")
        && call.args.len() == 2
        && call.named.is_empty()
        && scalar(b.m, call.args[0]) == Some(dimension as f64)
    {
        return Ok(node);
    }
    Err(Error(format!(
        "correlation prior requires an LKJCholesky law of dimension {dimension}"
    )))
}

pub(crate) fn emit(
    b: &mut Builder<'_>,
    prior: &ResolvedPrior,
    support: Option<NodeId>,
    dimension: Option<usize>,
    bindings: &[&str],
) -> Result<PriorNode> {
    let node = expression(b, &prior.prior, dimension, bindings)?;
    let constant = match b.m.node(node) {
        Node::Call(call)
            if matches!(call.head, CallHead::Builtin(s) if b.m.resolve(s) == "constant")
                && call.args.len() == 1
                && call.named.is_empty() =>
        {
            Some(call.args[0])
        }
        _ => None,
    };
    let lower = prior
        .lb
        .as_ref()
        .map(|v| bound(b, v, bindings))
        .transpose()?;
    let upper = prior
        .ub
        .as_ref()
        .map(|v| bound(b, v, bindings))
        .transpose()?;
    let restriction = if lower.is_some() || upper.is_some() {
        let lo = lower.unwrap_or_else(|| {
            let inf = b.call_head("inf");
            b.call("neg", &[inf])
        });
        let hi = upper.unwrap_or_else(|| b.call_head("inf"));
        let interval = b.call("interval", &[lo, hi]);
        Some(match support {
            Some(support) => intersect(b, support, interval)?,
            None => interval,
        })
    } else {
        support
    };
    if let Some(value) = constant {
        let number = scalar(b.m, value)
            .filter(|v| v.is_finite())
            .ok_or_else(|| Error("constant(c) requires a finite numeric literal".into()))?;
        if let Some(support) = restriction {
            let (lo, hi, open) = limits(b, support)?;
            let lo =
                scalar(b.m, lo).ok_or_else(|| Error("constant(c) requires fixed bounds".into()))?;
            let hi =
                scalar(b.m, hi).ok_or_else(|| Error("constant(c) requires fixed bounds".into()))?;
            if number < lo || number > hi || (open && number == lo) {
                return Err(Error(
                    "constant(c) lies outside its parameter bounds".into(),
                ));
            }
        }
        return Ok(PriorNode::Constant(value));
    }
    let flat = match b.m.node(node) {
        Node::Call(call) => match call.head {
            CallHead::User(head) if matches!(b.m.node(head), Node::Ref(Ref { ns: RefNs::Module(alias), name }) if b.m.resolve(*alias) == "reg" && b.m.resolve(*name) == "Flat") => {
                Some(call.clone())
            }
            _ => None,
        },
        _ => None,
    };
    let measure = if let Some(call) = flat {
        let existing = call.args.first().copied().or_else(|| {
            call.named
                .iter()
                .find(|arg| b.m.resolve(arg.name) == "support")
                .map(|arg| arg.value)
        });
        if call.args.len() > 1
            || call
                .named
                .iter()
                .any(|arg| b.m.resolve(arg.name) != "support")
            || (!call.args.is_empty() && !call.named.is_empty())
        {
            return Err(Error("Flat accepts one support argument".into()));
        }
        let support = match (existing, restriction) {
            (Some(a), Some(c)) => Some(intersect(b, a, c)?),
            (a, c) => a.or(c),
        };
        b.module_user_call("reg", "Flat", &support.into_iter().collect::<Vec<_>>())
    } else if let Some(support) = restriction {
        restrict(b, node, support)?
    } else {
        node
    };
    Ok(PriorNode::Measure(measure))
}

fn restrict(b: &mut Builder<'_>, measure: NodeId, support: NodeId) -> Result<NodeId> {
    if let Node::Call(call) = b.m.node(measure).clone() {
        if matches!(call.head, CallHead::Builtin(s) if b.m.resolve(s) == "locscale")
            && call.args.len() == 3
            && call.named.is_empty()
        {
            // Pull scalar interval restrictions through the affine chart so
            // the engine can normalize the primitive law with its CDF.
            let (lo, hi, open) = limits(b, support)?;
            if !open {
                let (location, scale) = (call.args[1], call.args[2]);
                let transform = |b: &mut Builder<'_>, value| {
                    // Infinite endpoints only change sign. Dividing infinity
                    // by a live scale poisons its reverse derivative with 0*inf.
                    if let Some(endpoint) = scalar(b.m, value)
                        && endpoint.is_infinite()
                    {
                        if let Some(scale) = scalar(b.m, scale) {
                            return b.lit_real(endpoint / scale);
                        }
                        let zero = b.lit_real(0.0);
                        let positive = b.call("gt", &[scale, zero]);
                        let reflected = b.call("neg", &[value]);
                        return b.call("ifelse", &[positive, value, reflected]);
                    }
                    if let (Some(value), Some(location), Some(scale)) = (
                        scalar(b.m, value),
                        scalar(b.m, location),
                        scalar(b.m, scale),
                    ) {
                        return b.lit_real((value - location) / scale);
                    }
                    let shifted = b.call("sub", &[value, location]);
                    b.call("divide", &[shifted, scale])
                };
                let lo = transform(b, lo);
                let hi = transform(b, hi);
                let (lo, hi) = match scalar(b.m, scale) {
                    Some(scale) if scale > 0.0 => (lo, hi),
                    Some(scale) if scale < 0.0 => (hi, lo),
                    _ => {
                        let pair = b.array(&[lo, hi]);
                        (b.call("minimum", &[pair]), b.call("maximum", &[pair]))
                    }
                };
                let interval = b.call("interval", &[lo, hi]);
                let law = restrict(b, call.args[0], interval)?;
                return Ok(b.call("locscale", &[law, location, scale]));
            }
        }
        if matches!(call.head, CallHead::Builtin(s) if b.m.resolve(s) == "normalize")
            && call.args.len() == 1
            && let Node::Call(inner) = b.m.node(call.args[0]).clone()
            && matches!(inner.head, CallHead::Builtin(s) if b.m.resolve(s) == "truncate")
            && inner.args.len() == 2
        {
            let support = intersect(b, inner.args[1], support)?;
            return restrict(b, inner.args[0], support);
        }
    }
    let restricted = b.call("truncate", &[measure, support]);
    Ok(b.call("normalize", &[restricted]))
}

fn bound(b: &mut Builder<'_>, value: &PriorBound, bindings: &[&str]) -> Result<NodeId> {
    match value {
        PriorBound::Number(value) if value.is_finite() => Ok(b.lit_real(*value)),
        PriorBound::Number(_) => Err(Error(
            "prior bounds must be finite numbers or expressions".into(),
        )),
        PriorBound::Expression(text) => expression(b, text, None, bindings),
    }
}

fn expression(
    b: &mut Builder<'_>,
    text: &str,
    dimension: Option<usize>,
    bindings: &[&str],
) -> Result<NodeId> {
    let parsed = flatppl_syntax::parse(&format!(
        "reg = standard_module(\"regression\", \"0.1\")\n__prior_expr = ({text})"
    ))
    .map_err(|e| Error(format!("invalid prior expression `{text}`: {e}")))?;
    let rows: Vec<_> = parsed.bindings().collect();
    if rows.len() != 2
        || parsed.resolve(rows[0].1.name) != "reg"
        || parsed.resolve(rows[1].1.name) != "__prior_expr"
    {
        return Err(Error("a prior must contain exactly one expression".into()));
    }
    copy(b, &parsed, rows[1].1.rhs, dimension, bindings)
}

fn reference(b: &mut Builder<'_>, source: &Module, r: Ref, bindings: &[&str]) -> Result<Ref> {
    let name = source.resolve(r.name);
    let ns = match r.ns {
        RefNs::SelfMod if !bindings.contains(&name) => {
            return Err(Error(format!(
                "prior references unavailable parameter `{name}`"
            )));
        }
        RefNs::Module(alias) => RefNs::Module(b.sym(source.resolve(alias))),
        ns => ns,
    };
    Ok(Ref {
        ns,
        name: b.sym(name),
    })
}

fn copy(
    b: &mut Builder<'_>,
    source: &Module,
    id: NodeId,
    dimension: Option<usize>,
    bindings: &[&str],
) -> Result<NodeId> {
    let node = match source.node(id) {
        Node::Lit(value) => Node::Lit(value.clone()),
        Node::Const(name) if bindings.contains(&source.resolve(*name)) => {
            return Ok(b.self_ref(source.resolve(*name)));
        }
        Node::Const(name) => Node::Const(b.sym(source.resolve(*name))),
        Node::Ref(r) => Node::Ref(reference(b, source, *r, bindings)?),
        Node::Hole => Node::Hole,
        Node::Axis(axis) => Node::Axis(flatppl_core::Axis {
            name: b.sym(source.resolve(axis.name)),
            variance: axis.variance,
        }),
        Node::Call(call) => {
            let args = call
                .args
                .iter()
                .map(|&v| copy(b, source, v, dimension, bindings))
                .collect::<Result<Vec<_>>>()?;
            let named = call
                .named
                .iter()
                .map(|arg| {
                    Ok(NamedArg {
                        kind: arg.kind,
                        name: b.sym(source.resolve(arg.name)),
                        value: copy(b, source, arg.value, dimension, bindings)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if let CallHead::Builtin(name) = call.head
                && let Some(node) = alias(b, source.resolve(name), &args, &named, dimension)?
            {
                return Ok(node);
            }
            let head = match call.head {
                CallHead::Builtin(name) => CallHead::Builtin(b.sym(source.resolve(name))),
                CallHead::User(head) => CallHead::User(copy(b, source, head, dimension, bindings)?),
            };
            let inputs = match &call.inputs {
                Some(Inputs::Spec(values)) => Some(Inputs::Spec(
                    values
                        .iter()
                        .map(|(name, r)| {
                            Ok((
                                b.sym(source.resolve(*name)),
                                reference(b, source, *r, bindings)?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into(),
                )),
                value => value.clone(),
            };
            Node::Call(Call {
                head,
                args: args.into(),
                named: named.into(),
                inputs,
            })
        }
    };
    Ok(b.m.alloc(node))
}

fn alias(
    b: &mut Builder<'_>,
    name: &str,
    args: &[NodeId],
    named: &[NamedArg],
    dimension: Option<usize>,
) -> Result<Option<NodeId>> {
    let (target, arity) = match name {
        "normal" => ("Normal", 2),
        "exponential" => ("Exponential", 1),
        "gamma" => ("Gamma", 2),
        "inv_gamma" => ("InverseGamma", 2),
        "beta" => ("Beta", 2),
        "lognormal" => ("LogNormal", 2),
        "logistic" => ("Logistic", 2),
        "weibull" => ("Weibull", 2),
        "double_exponential" => ("Laplace", 2),
        "chi_square" => ("ChiSquared", 1),
        "student_t" => ("StudentT", 3),
        "cauchy" => ("Cauchy", 2),
        "uniform" => ("Uniform", 2),
        "lkj" | "lkj_corr" | "lkj_corr_cholesky" => ("LKJCholesky", 1),
        "Flat" => {
            let head = b.module_call("reg", "Flat");
            return Ok(Some(b.m.alloc(Node::Call(Call {
                head: CallHead::User(head),
                args: args.to_vec().into(),
                named: named.to_vec().into(),
                inputs: None,
            }))));
        }
        _ => return Ok(None),
    };
    if args.len() != arity || !named.is_empty() {
        return Err(Error(format!(
            "brms prior alias `{name}` requires {arity} positional arguments"
        )));
    }
    if matches!(name, "student_t" | "cauchy")
        && scalar(b.m, args[arity - 1]).is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        return Err(Error(format!(
            "brms prior `{name}` requires a positive finite scale"
        )));
    }
    Ok(Some(match name {
        "student_t" => {
            let standard = b.call(target, &args[..1]);
            b.call("locscale", &[standard, args[1], args[2]])
        }
        "cauchy" => {
            let zero = b.lit_real(0.0);
            let one = b.lit_real(1.0);
            let standard = b.call(target, &[zero, one]);
            b.call("locscale", &[standard, args[0], args[1]])
        }
        "uniform" => {
            let interval = b.call("interval", args);
            b.call(target, &[interval])
        }
        "lkj" | "lkj_corr" | "lkj_corr_cholesky" => {
            let n = dimension
                .ok_or_else(|| Error("an LKJ prior requires a group correlation target".into()))?;
            let n = b.lit_int(n as i64);
            b.call(target, &[n, args[0]])
        }
        _ => b.call(target, args),
    }))
}

fn limits(b: &mut Builder<'_>, support: NodeId) -> Result<(NodeId, NodeId, bool)> {
    match b.m.node(support) {
        Node::Call(call)
            if matches!(call.head, CallHead::Builtin(s) if b.m.resolve(s) == "interval")
                && call.args.len() == 2 =>
        {
            Ok((call.args[0], call.args[1], false))
        }
        Node::Const(name) => {
            let name = b.m.resolve(*name).to_owned();
            let inf = b.call_head("inf");
            let lo = match name.as_str() {
                "reals" => b.call("neg", &[inf]),
                "posreals" | "nonnegreals" | "unitinterval" => b.lit_real(0.0),
                _ => {
                    return Err(Error(
                        "scalar prior bounds require a real interval support".into(),
                    ));
                }
            };
            let hi = if name == "unitinterval" {
                b.lit_real(1.0)
            } else {
                inf
            };
            Ok((lo, hi, name == "posreals"))
        }
        _ => Err(Error(
            "scalar prior bounds require a real interval support".into(),
        )),
    }
}

fn intersect(b: &mut Builder<'_>, a: NodeId, c: NodeId) -> Result<NodeId> {
    let (al, au, ao) = limits(b, a)?;
    let (cl, cu, co) = limits(b, c)?;
    let (Some(alv), Some(clv)) = (scalar(b.m, al), scalar(b.m, cl)) else {
        if ao || co {
            return Err(Error("open Flat supports with parameter-dependent bounds are not representable as an interval".into()));
        }
        let lo = b.array(&[al, cl]);
        let lo = b.call("maximum", &[lo]);
        let hi = b.array(&[au, cu]);
        let hi = b.call("minimum", &[hi]);
        return Ok(b.call("interval", &[lo, hi]));
    };
    let lo = if alv >= clv { al } else { cl };
    let hi = match (scalar(b.m, au), scalar(b.m, cu)) {
        (Some(a), Some(c)) => {
            if a <= c {
                au
            } else {
                cu
            }
        }
        _ => {
            let pair = b.array(&[au, cu]);
            b.call("minimum", &[pair])
        }
    };
    let open = (ao && alv >= clv) || (co && clv >= alv);
    if open {
        if scalar(b.m, lo) == Some(0.0) && scalar(b.m, hi) == Some(f64::INFINITY) {
            return Ok(b.call_head("posreals"));
        }
        return Err(Error("a finite upper bound on an open Flat support is not representable as an interval; use Flat(interval(...))".into()));
    }
    Ok(b.call("interval", &[lo, hi]))
}

fn scalar(module: &Module, id: NodeId) -> Option<f64> {
    match module.node(id) {
        Node::Lit(Scalar::Real(v)) => Some(*v),
        Node::Lit(Scalar::Int(v)) => Some(*v as f64),
        Node::Const(s) if module.resolve(*s) == "inf" => Some(f64::INFINITY),
        Node::Call(call)
            if matches!(call.head, CallHead::Builtin(s) if module.resolve(s) == "neg")
                && call.args.len() == 1 =>
        {
            scalar(module, call.args[0]).map(|v| -v)
        }
        _ => None,
    }
}
