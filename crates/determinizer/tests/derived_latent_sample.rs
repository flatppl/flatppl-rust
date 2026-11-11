//! Spec §04/§07: sampling threads one rng through every draw in dependency order,
//! including a draw whose kernel reads a binding derived from earlier draws.
use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Scalar};
use flatppl_determinizer::determinize;

fn determinized(src: &str) -> (Module, String) {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let out = determinize(&m).expect("the model samples");
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

/// A draw whose kernel reads a binding DERIVED from earlier draws depends on those
/// draws: they are sampled first, once each, on one key chain, and the derived
/// binding reads their values.
#[test]
fn a_draw_reading_a_derived_binding_samples_its_latents_first() {
    for (latents, derived, read, chain) in [
        (
            "a ~ Gamma(2.0, 1.0)\nb ~ Gamma(3.0, 1.0)\n",
            "v = [a, b]\n",
            "sqrt(v[1])",
            vec!["Gamma", "Gamma", "Normal"],
        ),
        (
            "a ~ Gamma(2.0, 1.0)\n",
            "v = [a, 1.0]\n",
            "v[1]",
            vec!["Gamma", "Normal"],
        ),
        (
            "a ~ Gamma(2.0, 1.0)\n",
            "v = a * 2.0\n",
            "v",
            vec!["Gamma", "Normal"],
        ),
    ] {
        for fields in ["v = v, y = y", "y = y"] {
            let src = format!(
                "{latents}{derived}y ~ Normal(0.0, {read})\n\
                 prior = lawof(record({fields}))\n\
                 seed = rnginit(0)\n\
                 draws = rand(prior, seed)\n"
            );
            let (m, pir) = determinized(&src);
            let got: Vec<_> = key_chain(&m, "draws").into_iter().map(|(c, _)| c).collect();
            assert_eq!(got, chain, "{src}\n{pir}");
            // y's scale reads the derived binding, which reads the sampled latents.
            assert!(pir.contains("(%ref self v)"), "{src}\n{pir}");
            assert!(!pir.contains("(draw "), "{src}\n{pir}");
        }
    }
}
