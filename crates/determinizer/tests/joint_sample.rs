//! §06 `joint(M1, M2, ...)` sampling: the components are drawn in turn through the
//! record-of-draws fold, threading the rng. The keyword form's variate is that record,
//! the positional form's is the `cat` of the component variates.
use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Scalar};
use flatppl_determinizer::determinize;

fn determinized(src: &str) -> (Module, String) {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let out = determinize(&m).expect("a joint samples");
    let pir = flatppl_flatpir::write(&out);
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok(), "{pir}");
    (out, pir)
}

fn rhs(m: &Module, name: &str) -> NodeId {
    m.bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap_or_else(|| panic!("no binding `{name}`"))
        .1
        .rhs
}

/// Follow self references to the defining node.
fn resolve(m: &Module, mut id: NodeId) -> NodeId {
    for _ in 0..64 {
        match m.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => match m.binding_by_name(r.name) {
                Some(b) => id = m.binding(b).rhs,
                None => return id,
            },
            _ => return id,
        }
    }
    id
}

fn builtin<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a flatppl_core::Call> {
    match m.node(id) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if m.resolve(s) == name) => Some(c),
        _ => None,
    }
}

/// A structural key, so a sample printed at several uses counts once.
fn key(m: &Module, id: NodeId) -> String {
    let id = resolve(m, id);
    match m.node(id) {
        Node::Call(c) => {
            let head = match c.head {
                CallHead::Builtin(s) => m.resolve(s).to_string(),
                _ => "user".into(),
            };
            let args: Vec<_> = c.args.iter().map(|&a| key(m, a)).collect();
            let named: Vec<_> = c
                .named
                .iter()
                .map(|n| format!("{}={}", m.resolve(n.name), key(m, n.value)))
                .collect();
            format!("({head} {} {})", args.join(" "), named.join(" "))
        }
        Node::Const(s) => m.resolve(*s).to_string(),
        Node::Lit(s) => format!("{s:?}"),
        other => format!("{other:?}"),
    }
}

fn samples(m: &Module, id: NodeId, out: &mut Vec<NodeId>, seen: &mut Vec<String>) {
    let id = resolve(m, id);
    if builtin(m, id, "builtin_sample").is_some() {
        let k = key(m, id);
        if !seen.contains(&k) {
            seen.push(k);
            out.push(id);
        }
    }
    for child in m.node(id).children() {
        samples(m, child, out, seen);
    }
}

/// The exact key chain: every distinct `builtin_sample` consumes the successor key of
/// the one before it, the first consumes `seed`, and no key is read twice. Returns the
/// constructor and first parameter of each draw, in draw order.
fn key_chain(m: &Module, output: &str) -> Vec<(String, String)> {
    let mut all = Vec::new();
    samples(m, rhs(m, output), &mut all, &mut Vec::new());
    let predecessor = |id: NodeId| -> Option<NodeId> {
        let rng = resolve(m, builtin(m, id, "builtin_sample").unwrap().args[0]);
        let get = builtin(m, rng, "get0")?;
        assert!(
            matches!(m.node(get.args[1]), Node::Lit(Scalar::Int(1))),
            "a key must be a sample's successor slot"
        );
        Some(resolve(m, get.args[0]))
    };
    let mut chain = Vec::new();
    let mut cur = *all
        .iter()
        .find(|&&s| {
            !all.iter()
                .any(|&o| predecessor(o).is_some_and(|p| key(m, p) == key(m, s)))
        })
        .expect("a last draw");
    loop {
        let c = builtin(m, cur, "builtin_sample").unwrap();
        let first = c.args[2];
        let params = builtin(m, resolve(m, first), "record").unwrap();
        chain.push((key(m, c.args[1]), key(m, params.named[0].value)));
        match predecessor(cur) {
            Some(p) => cur = p,
            None => {
                assert!(
                    builtin(m, resolve(m, c.args[0]), "rnginit").is_some(),
                    "the first draw must consume the seed"
                );
                break;
            }
        }
    }
    chain.reverse();
    assert_eq!(chain.len(), all.len(), "every draw lies on one key chain");
    chain
}

fn gamma(shape: f64) -> (String, String) {
    ("Gamma".into(), format!("{:?}", Scalar::Real(shape)))
}

#[test]
fn a_positional_joint_samples_to_the_cat_of_its_components() {
    let (m, pir) = determinized(
        "seed = rnginit(0)\n\
         draws = rand(joint(Gamma(2.0, 1.0), Gamma(3.0, 1.0)), seed)\n",
    );
    assert_eq!(key_chain(&m, "draws"), [gamma(2.0), gamma(3.0)]);
    let draws = rhs(&m, "draws");
    assert!(builtin(&m, draws, "cat").is_some(), "{pir}");
    assert!(!pir.contains("(joint "), "{pir}");
}

#[test]
fn a_keyword_joint_samples_to_the_record_of_its_components() {
    let (m, pir) = determinized(
        "seed = rnginit(0)\n\
         draws = rand(joint(a = Gamma(2.0, 1.0), b = Gamma(3.0, 1.0)), seed)\n",
    );
    assert_eq!(key_chain(&m, "draws"), [gamma(2.0), gamma(3.0)]);
    let draws = rhs(&m, "draws");
    let record = builtin(&m, draws, "record").unwrap_or_else(|| panic!("{pir}"));
    let fields: Vec<_> = record.named.iter().map(|n| m.resolve(n.name)).collect();
    assert_eq!(fields, ["a", "b"]);
}

/// A drawn positional joint is a shared latent here: `y` reads `v[1]`, and the record
/// keeps `v` whole. Each component is drawn once, then `y`, on one key chain.
#[test]
fn a_drawn_positional_joint_feeds_a_later_draw() {
    let (m, pir) = determinized(
        "v ~ joint(Gamma(2.0, 1.0), Gamma(3.0, 1.0))\n\
         y ~ Normal(0.0, sqrt(v[1]))\n\
         prior = lawof(record(v = v, y = y))\n\
         seed = rnginit(0)\n\
         draws = rand(prior, seed)\n",
    );
    let chain = key_chain(&m, "draws");
    assert_eq!(chain[..2], [gamma(2.0), gamma(3.0)]);
    assert_eq!(chain[2].0, "Normal");
    assert_eq!(chain.len(), 3);
    assert!(
        pir.contains("(sqrt (%meta ((%scalar real) %fixed reals) (get (%ref self v) 1)))"),
        "{pir}"
    );
}
