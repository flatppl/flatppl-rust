use crate::{Recipe, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

pub(crate) fn data_name(index: usize) -> String {
    format!("cs_data_{}", index + 1)
}

pub(crate) fn coefficient_name(index: usize) -> String {
    format!("bcs_{}", index + 1)
}

pub(crate) fn adjust_cuts(
    b: &mut Builder<'_>,
    recipe: &Recipe,
    mut cuts: Vec<NodeId>,
) -> Vec<NodeId> {
    for (index, _) in recipe.formula.category_specific.iter().enumerate() {
        let values = b.self_ref(&data_name(index));
        let coefficients = b.self_ref(&coefficient_name(index));
        for (step, cut) in cuts.iter_mut().enumerate() {
            let index = b.lit_int((step + 1) as i64);
            let coefficient = b.call("get", &[coefficients, index]);
            let effect = broadcast(b, "mul", &[coefficient, values]);
            *cut = broadcast(b, "sub", &[*cut, effect]);
        }
    }
    cuts
}

fn logits(b: &mut Builder<'_>, eta: NodeId, cuts: &[NodeId]) -> NodeId {
    let mut logits = Vec::with_capacity(cuts.len() + 1);
    let mut current = b.lit_real(0.0);
    logits.push(current);
    // The log odds of category j + 1 versus j are eta - cut[j].
    for &cut in cuts {
        let step = b.call("sub", &[eta, cut]);
        current = b.call("add", &[current, step]);
        logits.push(current);
    }
    b.array(&logits)
}

pub(crate) fn log_prob(b: &mut Builder<'_>, eta: NodeId, cuts: &[NodeId], value: NodeId) -> NodeId {
    let logits = logits(b, eta, cuts);
    let log_probs = b.call("logsoftmax", &[logits]);
    let first = b.lit_int(1);
    let last = b.lit_int((cuts.len() + 1) as i64);
    let index = b.call("max", &[value, first]);
    let index = b.call("min", &[index, last]);
    let valid = b.call("equal", &[value, index]);
    let score = b.call("get", &[log_probs, index]);
    let outside = b.lit_real(f64::NEG_INFINITY);
    b.call("ifelse", &[valid, score, outside])
}

pub(crate) fn observation(
    b: &mut Builder<'_>,
    eta: NodeId,
    cuts: &[NodeId],
    scalar: bool,
) -> NodeId {
    if scalar {
        let logits = logits(b, eta, cuts);
        let probabilities = b.call("softmax", &[logits]);
        return b.call("Categorical", &[probabilities]);
    }
    emit_rows(b, eta, cuts, |b, probabilities| {
        b.call("Categorical", &[probabilities])
    })
}

pub(crate) fn probabilities(b: &mut Builder<'_>, eta: NodeId, cuts: &[NodeId]) -> NodeId {
    emit_rows(b, eta, cuts, |_, probabilities| probabilities)
}

fn emit_rows(
    b: &mut Builder<'_>,
    eta: NodeId,
    cuts: &[NodeId],
    finish: impl FnOnce(&mut Builder<'_>, NodeId) -> NodeId,
) -> NodeId {
    let mut inputs = Vec::new();
    let mut row_arg = |b: &mut Builder<'_>, name: &str| {
        let input = Ref {
            ns: RefNs::Local,
            name: b.sym(&format!("_{name}_")),
        };
        inputs.push((b.sym(name), input));
        b.m.alloc(Node::Ref(input))
    };
    let row_eta = row_arg(b, "eta");
    let row_cuts: Vec<_> = (1..=cuts.len())
        .map(|i| row_arg(b, &format!("cut{i}")))
        .collect();
    let row_logits = logits(b, row_eta, &row_cuts);
    let probabilities = b.call("softmax", &[row_logits]);
    let result = finish(b, probabilities);
    let head = CallHead::Builtin(b.sym("functionof"));
    let kernel = b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([result]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    let mut args = vec![kernel, eta];
    args.extend_from_slice(cuts);
    b.call("broadcast", &args)
}
