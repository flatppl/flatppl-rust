use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

/// Keep the category vector inside a scalar row kernel when broadcasting rows.
pub(crate) fn observation(b: &mut Builder<'_>, eta: NodeId, thresholds: usize) -> NodeId {
    let name = b.sym("eta");
    let input = Ref {
        ns: RefNs::Local,
        name: b.sym("_eta_"),
    };
    let value = b.m.alloc(Node::Ref(input));
    let cuts = b.self_ref("cuts");
    let mut differences = Vec::new();
    for i in 1..=thresholds {
        let index = b.lit_int(i as i64);
        let cut = b.call("get", &[cuts, index]);
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
    let last = b.lit_int(thresholds as i64);
    let last = b.call("get", &[cuts, last]);
    let tail = b.call("sub", &[value, last]);
    // The complementary logistic form retains the upper tail near one.
    probabilities.push(b.call("invlogit", &[tail]));
    let probabilities = b.array(&probabilities);
    let law = b.call("Categorical", &[probabilities]);
    let head = CallHead::Builtin(b.sym("functionof"));
    let kernel = b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([law]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(Box::new([(name, input)]))),
    }));
    b.call("broadcast", &[kernel, eta])
}
