//! §06 `jointchain(M, K1, ...)` sampling: the base is drawn, then each kernel applied
//! to the variates to its left, on one rng key chain.
use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Scalar};
use flatppl_determinizer::determinize;

fn determinized(src: &str) -> (Module, String) {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let out = determinize(&m).expect("a jointchain samples");
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

fn record_fields(m: &Module, name: &str) -> Vec<String> {
    let record = builtin(m, resolve(m, rhs(m, name)), "record").expect("a record variate");
    record
        .named
        .iter()
        .map(|n| m.resolve(n.name).to_string())
        .collect()
}

fn ctors(chain: &[(String, String)]) -> Vec<&str> {
    chain.iter().map(|(c, _)| c.as_str()).collect()
}

const SEED: &str = "seed = rnginit(0)\n";

/// The issue's shape: a keyword `joint` base, then an inline kernel whose body is a
/// keyword `joint` reading the base's field. The variate is the merged record.
#[test]
fn an_inline_step_kernel_draws_after_its_base() {
    let (m, _) = determinized(&format!(
        "{SEED}prior = jointchain(joint(v = Dirichlet([2.0, 3.0, 5.0])), r -> joint(y = Normal(0.0, r.v[1])))\n\
         draws = rand(prior, seed)\n"
    ));
    assert_eq!(ctors(&key_chain(&m, "draws")), ["Dirichlet", "Normal"]);
    assert_eq!(record_fields(&m, "draws"), ["v", "y"]);
}

/// Three components: each kernel receives the record of every field to its left.
#[test]
fn a_three_step_chain_threads_one_key_through_every_draw() {
    let (m, pir) = determinized(&format!(
        "{SEED}prior = jointchain(joint(a = Gamma(2.0, 1.0)), r -> joint(b = Normal(0.0, sqrt(r.a))), \
         r -> joint(c = Normal(r.b, r.a)))\n\
         draws = rand(prior, seed)\n"
    ));
    let chain = key_chain(&m, "draws");
    assert_eq!(ctors(&chain), ["Gamma", "Normal", "Normal"]);
    assert_eq!(record_fields(&m, "draws"), ["a", "b", "c"]);
    assert!(!pir.contains("(jointchain "), "{pir}");
}

/// The keyword form calls each kernel with the record of the preceding labels.
#[test]
fn the_keyword_form_samples_to_the_record_of_its_labels() {
    let (m, _) = determinized(&format!(
        "{SEED}prior = jointchain(a = Gamma(2.0, 1.0), b = r -> Normal(0.0, sqrt(r.a)))\n\
         draws = rand(prior, seed)\n"
    ));
    assert_eq!(ctors(&key_chain(&m, "draws")), ["Gamma", "Normal"]);
    assert_eq!(record_fields(&m, "draws"), ["a", "b"]);
}

/// `kernelof`, `functionof` and named lambda steps sample like the inline lambda.
#[test]
fn kernelof_and_named_steps_sample() {
    for step in [
        "a = elementof(posreals)\ny ~ Normal(0.0, sqrt(a))\nK = kernelof(record(y = y))\n",
        "a = elementof(posreals)\nK = functionof(joint(y = Normal(0.0, sqrt(a))))\n",
        "K = r -> joint(y = Normal(0.0, sqrt(r.a)))\n",
    ] {
        let (m, _) = determinized(&format!(
            "{SEED}{step}prior = jointchain(joint(a = Gamma(2.0, 1.0)), K)\n\
             draws = rand(prior, seed)\n"
        ));
        assert_eq!(
            ctors(&key_chain(&m, "draws")),
            ["Gamma", "Normal"],
            "{step}"
        );
        assert_eq!(record_fields(&m, "draws"), ["a", "y"], "{step}");
    }
}

/// Positional scalar components: the kernel's one input is the prior scalar, and the
/// variate is the `cat` of the draws.
#[test]
fn a_scalar_chain_samples_to_the_cat_of_its_draws() {
    let (m, _) = determinized(&format!(
        "{SEED}prior = jointchain(Gamma(2.0, 1.0), x -> Normal(0.0, sqrt(x)))\n\
         draws = rand(prior, seed)\n"
    ));
    assert_eq!(ctors(&key_chain(&m, "draws")), ["Gamma", "Normal"]);
    assert!(builtin(&m, resolve(&m, rhs(&m, "draws")), "cat").is_some());
}

/// A drawn chain feeds a derived value, read once per draw.
#[test]
fn a_drawn_chain_feeds_a_derived_value() {
    let (m, _) = determinized(&format!(
        "{SEED}d ~ jointchain(joint(v = Gamma(2.0, 1.0)), r -> joint(y = Normal(0.0, sqrt(r.v))))\n\
         z = d.y / sqrt(d.v)\n\
         draws = rand(lawof(z), seed)\n"
    ));
    assert_eq!(ctors(&key_chain(&m, "draws")), ["Gamma", "Normal"]);
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect_err("must refuse").reason
}

#[test]
fn iid_of_a_chain_refuses_accurately() {
    let reason = refusal(&format!(
        "{SEED}prior = jointchain(joint(a = Gamma(2.0, 1.0)), r -> joint(b = Normal(0.0, sqrt(r.a))))\n\
         draws = rand(iid(prior, 3), seed)\n"
    ));
    assert!(reason.contains("iid of a jointchain"), "{reason}");
    assert!(!reason.contains("statically-resolved shape"), "{reason}");
}

#[test]
fn kchain_sampling_stays_deferred() {
    let reason = refusal(&format!(
        "{SEED}prior = kchain(Gamma(2.0, 1.0), x -> Normal(0.0, sqrt(x)))\n\
         draws = rand(prior, seed)\n"
    ));
    assert!(reason.contains("deferred"), "{reason}");
}
