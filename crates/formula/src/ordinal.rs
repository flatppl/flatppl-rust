use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

/// Keep the category vector inside a scalar row kernel when broadcasting rows.
pub(crate) fn observation(
    b: &mut Builder<'_>,
    eta: NodeId,
    cuts: &[NodeId],
    scalar: bool,
) -> NodeId {
    let name = b.sym("eta");
    let input = Ref {
        ns: RefNs::Local,
        name: b.sym("_eta_"),
    };
    let value = if scalar {
        eta
    } else {
        b.m.alloc(Node::Ref(input))
    };
    let mut inputs = vec![(name, input)];
    let mut args = vec![eta];
    let mut differences = Vec::new();
    for (i, &cut) in cuts.iter().enumerate() {
        let cut = if scalar {
            cut
        } else {
            let name = b.sym(&format!("cut{}", i + 1));
            let input = Ref {
                ns: RefNs::Local,
                name: b.sym(&format!("_cut{}_", i + 1)),
            };
            inputs.push((name, input));
            args.push(cut);
            b.m.alloc(Node::Ref(input))
        };
        let difference = b.call("sub", &[cut, value]);
        differences.push(difference);
    }
    let first = b.call("invlogit", &[differences[0]]);
    let mut probabilities = vec![first];
    for pair in differences.windows(2) {
        // sigmoid(hi) - sigmoid(lo), without subtracting nearly equal CDFs.
        let gap = b.call("sub", &[pair[0], pair[1]]);
        let gap = b.call("expm1", &[gap]);
        let gap = b.call("neg", &[gap]);
        let lo = b.call("neg", &[pair[0]]);
        let lo = b.call("invlogit", &[lo]);
        let hi = b.call("invlogit", &[pair[1]]);
        let mass = b.call("mul", &[lo, hi]);
        probabilities.push(b.call("mul", &[mass, gap]));
    }
    let last = *differences.last().expect("an ordinal model has thresholds");
    let tail = b.call("neg", &[last]);
    // The complementary logistic form retains the upper tail near one.
    probabilities.push(b.call("invlogit", &[tail]));
    let probabilities = b.array(&probabilities);
    let law = b.call("Categorical", &[probabilities]);
    if scalar {
        return law;
    }
    let head = CallHead::Builtin(b.sym("functionof"));
    let kernel = b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([law]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    args.insert(0, kernel);
    b.call("broadcast", &args)
}
