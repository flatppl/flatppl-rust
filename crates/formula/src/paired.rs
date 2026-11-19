use crate::{Error, Family, Formula, Link, Result, Spec, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs};

pub(crate) const SECOND_MEAN: &str = "mu2";
pub(crate) const SECOND_DATA: &str = "y2_data";
pub(crate) const OBSERVED: &str = "y_observed";
pub(crate) const AUXILIARIES: [&str; 2] = ["sigma2", "rescor"];

pub(crate) fn parse(spec: &Spec, family: Family, first: &Formula) -> Result<Option<Formula>> {
    let Some(second) = &spec.paired_formula else {
        if spec.paired_family.is_some() || spec.paired_link.is_some() {
            return Err(Error(
                "a second family or link requires a second formula".into(),
            ));
        }
        return Ok(None);
    };
    let supported_second = matches!(
        spec.paired_family.as_deref(),
        None | Some("gaussian" | "bernoulli")
    );
    if family != Family::Gaussian
        || !supported_second
        || !spec.dpars.is_empty()
        || !spec.nlpars.is_empty()
    {
        return Err(Error(
            "paired formulas require a Gaussian first response and a Gaussian or Bernoulli second response with constant-in-row scales".into(),
        ));
    }
    if spec.paired_link.is_some() && spec.paired_family.is_none() {
        return Err(Error(
            "a second link requires an explicit second family".into(),
        ));
    }
    let second = crate::parse_formula(second)?;
    if first.response == second.response {
        return Err(Error("paired formulas require distinct responses".into()));
    }
    if first.has_response_additions() || second.has_response_additions() {
        return Err(Error(
            "paired formulas do not support response additions".into(),
        ));
    }
    if [first, &second].iter().any(|formula| {
        formula.groups.iter().any(|group| {
            group
                .columns
                .iter()
                .any(|column| column == &first.response || column == &second.response)
        })
    }) {
        return Err(Error(
            "paired responses cannot also be grouping columns".into(),
        ));
    }
    Ok(Some(second))
}

pub(crate) fn expected(b: &mut Builder<'_>) -> NodeId {
    let means = [b.self_ref("mu"), b.self_ref(SECOND_MEAN)];
    let means = b.array(&means);
    b.call("colstack", &[means])
}

pub(crate) fn draw(b: &mut Builder<'_>, family: Family) -> Result<NodeId> {
    let means = [b.self_ref("mu"), b.self_ref(SECOND_MEAN)];
    let sigma = scale(b, "sigma")?;
    let law = broadcast(b, "Normal", &[means[0], sigma]);
    let first = b.call("draw", &[law]);
    b.bind("y", first);
    let first = b.self_ref("y");
    let law = if family == Family::Bernoulli {
        broadcast(b, "Bernoulli", &[means[1]])
    } else {
        let sigma2 = scale(b, "sigma2")?;
        let correlation = b.self_ref("rescor");
        let (mean, scale) = conditional(b, means, [sigma, sigma2], correlation, first);
        broadcast(b, "Normal", &[mean, scale])
    };
    let second = b.call("draw", &[law]);
    b.bind("y2", second);
    let second = b.self_ref("y2");
    Ok(b.call_fields("record", &[("y", first), ("y2", second)]))
}

fn scale(b: &mut Builder<'_>, name: &str) -> Result<NodeId> {
    let symbol = b.sym(name);
    let binding =
        b.m.binding_by_name(symbol)
            .expect("paired scale was emitted");
    let rhs = b.m.binding(binding).rhs;
    if crate::prior_expr::scalar(b.m, rhs).is_some_and(|value| value <= 0.0) {
        return Err(Error(format!("paired Gaussian `{name}` must be positive")));
    }
    Ok(b.self_ref(name))
}

pub(crate) fn observation(b: &mut Builder<'_>) -> NodeId {
    let first = b.self_ref("y_data");
    let second = b.self_ref(SECOND_DATA);
    b.call_fields("record", &[("y", first), ("y2", second)])
}

pub(crate) fn use_partial_outcomes(
    module: &mut Module,
    family: Family,
    links: [Link; 2],
) -> Result<()> {
    let mut b = Builder::new(module);
    let fail = || Error("partial outcomes require a complete paired formula template".into());
    let log_lik = b.sym("log_lik");
    let log_lik = b.m.binding_by_name(log_lik).ok_or_else(fail)?;
    let likelihood = b.sym("L");
    let likelihood = b.m.binding_by_name(likelihood).ok_or_else(fail)?;
    let Node::Call(Call {
        inputs: Some(Inputs::Spec(inputs)),
        ..
    }) = b.m.node(b.m.binding(log_lik).rhs)
    else {
        return Err(fail());
    };
    let parameters: Vec<_> = inputs
        .iter()
        .map(|(name, _)| b.m.resolve(*name).to_owned())
        .collect();
    crate::emit::external_vector(&mut b, OBSERVED, "posintegers");
    let rows = pointwise(&mut b, family, Some(links));
    let score = crate::emit::parameter_function(&mut b, "functionof", rows, &parameters);
    b.m.set_binding_rhs(log_lik, score);
    let observed = crate::response::observed_likelihood(&mut b, rows, &parameters);
    b.m.set_binding_rhs(likelihood, observed);
    Ok(())
}

pub(crate) fn pointwise(b: &mut Builder<'_>, family: Family, links: Option<[Link; 2]>) -> NodeId {
    let [first_mean, second_mean] = if links.is_some() {
        ["eta_mu", "eta_mu2"]
    } else {
        ["mu", SECOND_MEAN]
    };
    let mut names = vec![first_mean, second_mean, "sigma", "y_data", SECOND_DATA];
    if links.is_some() {
        names.push(OBSERVED);
    }
    if family == Family::Gaussian {
        names.extend(AUXILIARIES);
    }
    let refs: Vec<_> = names
        .iter()
        .map(|name| Ref {
            ns: RefNs::Local,
            name: b.sym(&format!("_{name}_")),
        })
        .collect();
    let values: Vec<_> = refs
        .iter()
        .map(|&reference| b.m.alloc(Node::Ref(reference)))
        .collect();
    let [mu, mu2, sigma, first, second, rest @ ..] = values.as_slice() else {
        unreachable!("paired model has five row inputs")
    };
    let (mut mu, mut mu2, mut sigma, first, second) = (*mu, *mu2, *sigma, *first, *second);
    let zero = b.lit_real(0.0);
    let one = b.lit_real(1.0);
    let (observed, rest) = if let Some([first_link, second_link]) = links {
        let (&pattern, rest) = rest.split_first().expect("paired observation pattern");
        let first_only = b.lit_int(1);
        let second_only = b.lit_int(2);
        let both = b.lit_int(3);
        let first_missing = b.call("equal", &[pattern, second_only]);
        let second_missing = b.call("equal", &[pattern, first_only]);
        let joint = b.call("equal", &[pattern, both]);
        // Pick finite link inputs and scales before evaluating an inactive score.
        let half = b.lit_real(0.5);
        let second_default = if family == Family::Bernoulli && second_link == Link::Log {
            b.lit_real(-1.0)
        } else {
            half
        };
        let eta = b.call("ifelse", &[first_missing, half, mu]);
        let eta2 = b.call("ifelse", &[second_missing, second_default, mu2]);
        mu = first_link.inverse(b, eta);
        mu2 = second_link.inverse(b, eta2);
        sigma = b.call("ifelse", &[first_missing, one, sigma]);
        (Some([first_missing, second_missing, joint]), rest)
    } else {
        (None, rest)
    };
    let law = b.call("Normal", &[mu, sigma]);
    let first_score = b.call("logdensityof", &[law, first]);
    let law = if let [sigma2, correlation] = rest {
        let (mu, sigma, sigma2, correlation, first) =
            if let Some([_, second_missing, joint]) = observed {
                // A singleton uses its marginal Normal and has no residual correlation term.
                let mu = b.call("ifelse", &[joint, mu, zero]);
                let sigma = b.call("ifelse", &[joint, sigma, one]);
                let sigma2 = b.call("ifelse", &[second_missing, one, *sigma2]);
                let correlation = b.call("ifelse", &[joint, *correlation, zero]);
                let first = b.call("ifelse", &[joint, first, zero]);
                (mu, sigma, sigma2, correlation, first)
            } else {
                (mu, sigma, *sigma2, *correlation, first)
            };
        let (mean, scale) = conditional(b, [mu, mu2], [sigma, sigma2], correlation, first);
        b.call("Normal", &[mean, scale])
    } else {
        b.call("Bernoulli", &[mu2])
    };
    let second_score = b.call("logdensityof", &[law, second]);
    let (first_score, second_score) = if let Some([first_missing, second_missing, _]) = observed {
        (
            b.call("ifelse", &[first_missing, zero, first_score]),
            b.call("ifelse", &[second_missing, zero, second_score]),
        )
    } else {
        (first_score, second_score)
    };
    let score = b.call("add", &[first_score, second_score]);
    let head = b.sym("functionof");
    let inputs = names.iter().map(|name| b.sym(name)).zip(refs).collect();
    let row_score = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([score]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs)),
    }));
    let mut args = vec![row_score];
    args.extend(names.into_iter().map(|name| b.self_ref(name)));
    b.call("broadcast", &args)
}

fn conditional(
    b: &mut Builder<'_>,
    [mu, mu2]: [NodeId; 2],
    [sigma, sigma2]: [NodeId; 2],
    correlation: NodeId,
    first: NodeId,
) -> (NodeId, NodeId) {
    let delta = broadcast(b, "sub", &[first, mu]);
    let residual = broadcast(b, "divide", &[delta, sigma]);
    let loading = b.call("mul", &[correlation, sigma2]);
    let adjustment = broadcast(b, "mul", &[loading, residual]);
    let mean = broadcast(b, "add", &[mu2, adjustment]);
    let one = b.lit_real(1.0);
    let left = b.call("sub", &[one, correlation]);
    let right = b.call("add", &[one, correlation]);
    let variance = b.call("mul", &[left, right]);
    let deviation = b.call("sqrt", &[variance]);
    let scale = b.call("mul", &[sigma2, deviation]);
    (mean, scale)
}
