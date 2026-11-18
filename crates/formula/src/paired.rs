use crate::{Error, Family, Formula, Result, Spec, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

pub(crate) const SECOND_MEAN: &str = "mu2";
pub(crate) const SECOND_DATA: &str = "y2_data";
pub(crate) const AUXILIARIES: [&str; 2] = ["sigma2", "rescor"];

pub(crate) fn parse(spec: &Spec, family: Family, first: &Formula) -> Result<Option<Formula>> {
    let Some(second) = &spec.paired_formula else {
        return Ok(None);
    };
    if family != Family::Gaussian || !spec.dpars.is_empty() || !spec.nlpars.is_empty() {
        return Err(Error(
            "paired formulas require the Gaussian family with two mean predictors and constant-in-row scales".into(),
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

pub(crate) fn draw(b: &mut Builder<'_>) -> Result<NodeId> {
    let means = [b.self_ref("mu"), b.self_ref(SECOND_MEAN)];
    let scales = [scale(b, "sigma")?, scale(b, "sigma2")?];
    let correlation = b.self_ref("rescor");
    let law = broadcast(b, "Normal", &[means[0], scales[0]]);
    let first = b.call("draw", &[law]);
    b.bind("y", first);
    let first = b.self_ref("y");
    let (mean, scale) = conditional(b, means, scales, correlation, first);
    let law = broadcast(b, "Normal", &[mean, scale]);
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

pub(crate) fn pointwise(b: &mut Builder<'_>) -> NodeId {
    let names = [
        "mu",
        SECOND_MEAN,
        "sigma",
        "sigma2",
        "rescor",
        "y_data",
        SECOND_DATA,
    ];
    let refs = names.map(|name| Ref {
        ns: RefNs::Local,
        name: b.sym(&format!("_{name}_")),
    });
    let [mu, mu2, sigma, sigma2, correlation, first, second] =
        refs.map(|reference| b.m.alloc(Node::Ref(reference)));
    let law = b.call("Normal", &[mu, sigma]);
    let first_score = b.call("logdensityof", &[law, first]);
    let (mean, scale) = conditional(b, [mu, mu2], [sigma, sigma2], correlation, first);
    let law = b.call("Normal", &[mean, scale]);
    let second_score = b.call("logdensityof", &[law, second]);
    let score = b.call("add", &[first_score, second_score]);
    let head = b.sym("functionof");
    let inputs = names
        .map(|name| b.sym(name))
        .into_iter()
        .zip(refs)
        .collect();
    let row_score = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([score]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs)),
    }));
    let mut args = vec![row_score];
    args.extend(names.map(|name| b.self_ref(name)));
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
