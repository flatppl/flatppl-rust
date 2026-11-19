use crate::{Error, Family, Formula, Result, Spec, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

pub(crate) const SECOND_MEAN: &str = "mu2";
pub(crate) const SECOND_DATA: &str = "y2_data";
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

pub(crate) fn pointwise(b: &mut Builder<'_>, family: Family) -> NodeId {
    let mut names = vec!["mu", SECOND_MEAN, "sigma", "y_data", SECOND_DATA];
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
    let (mu, mu2, sigma, first, second) = (*mu, *mu2, *sigma, *first, *second);
    let law = b.call("Normal", &[mu, sigma]);
    let first_score = b.call("logdensityof", &[law, first]);
    let law = if let [sigma2, correlation] = rest {
        let (mean, scale) = conditional(b, [mu, mu2], [sigma, *sigma2], *correlation, first);
        b.call("Normal", &[mean, scale])
    } else {
        b.call("Bernoulli", &[mu2])
    };
    let second_score = b.call("logdensityof", &[law, second]);
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
