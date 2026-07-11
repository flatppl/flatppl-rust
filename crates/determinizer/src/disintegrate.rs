//! Structural disintegration split for the explicit-DAG case (spec §06
//! "Structural disintegration").
//!
//! `disintegrate(selector, lawof(record(field…)))` returns `(kernel,
//! marginal)`, the structural inverse of `jointchain(marginal, kernel) ≡
//! joint`. The SELECTED fields (named by the selector) become the kernel's
//! output law, conditioned on the NON-selected fields, which are the kernel's
//! `%specinputs` boundary; the NON-selected fields also form the marginal law.
//!
//! Only the explicit `lawof(record(…))` DAG case is handled here — every other
//! shape (a non-`lawof(record)` measure, a selector that is not a clean
//! field-name partition of the record, an empty selected or non-selected set)
//! yields `None`, and the caller refuses (§06 permits refusing intractable /
//! non-explicit disintegrations; refuse-don't-mislower).

use crate::density::{expect_builtin_call, resolve_ref_one};
use flatppl_core::{
    Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, Scalar, Symbol,
};

/// Split `disint_node = disintegrate(selector, lawof(record(fields)))` into a
/// `(kernel, marginal)` pair, structurally (spec §06 "Structural
/// disintegration").
///
/// - `kernel` = `kernelof(record(<selected %fields, verbatim value nodes>),
///   %specinputs([(non_sel_name, non_sel_value_ref), …]))` — the conditional of
///   the SELECTED fields, with the NON-selected fields as boundary inputs (so
///   `likelihoodof(kernel, data)` scores the selected fields given the rest).
/// - `marginal` = `lawof(record(<non-selected %fields, verbatim value nodes>))`
///   — the joint law over the NON-selected fields.
///
/// Returns `None` (caller refuses) for any shape outside the explicit DAG case:
/// - `disint_node` is not a 2-arg `disintegrate` call;
/// - the measure (after one ref hop) is not `lawof(record(field…))`, or the
///   record carries positional args / a non-`%field` named entry;
/// - the selector is not a string literal or `vector` of string literals
///   naming fields of the record (a name not present in the record, or an
///   empty selector, refuses);
/// - the selected or the non-selected set is empty (a vacuous split);
/// - a NON-selected field's value is not a bare `%ref` (its boundary-input
///   `Ref` cannot be formed — outside the explicit-DAG shape).
pub(crate) fn split_disintegrate(m: &mut Module, disint_node: NodeId) -> Option<(NodeId, NodeId)> {
    // 1. Recognize `disintegrate(selector, M)` with exactly two arguments.
    let c = expect_builtin_call(m, disint_node, "disintegrate")?;
    if c.args.len() != 2 {
        return None;
    }
    let selector = c.args[0];
    let measure = c.args[1];

    // Resolve `M` one ref hop to `lawof(record(field…))`.
    let (measure_resolved, _) = resolve_ref_one(m, measure);
    let lawof = expect_builtin_call(m, measure_resolved, "lawof")?;
    if lawof.args.len() != 1 {
        return None;
    }
    let record_arg = lawof.args[0];
    let (record_resolved, _) = resolve_ref_one(m, record_arg);
    let record = expect_builtin_call(m, record_resolved, "record")?;
    if !record.args.is_empty() {
        // A record with positional args is not a field-keyed product.
        return None;
    }
    // Gather the record's `%field name value` entries (verbatim value nodes),
    // in order; refuse any non-`%field` named entry.
    let mut fields: Vec<(Symbol, NodeId)> = Vec::with_capacity(record.named.len());
    for na in record.named.iter() {
        if na.kind != NamedKind::Field {
            return None;
        }
        fields.push((na.name, na.value));
    }
    if fields.is_empty() {
        return None;
    }

    // 2. Parse the selector into the SELECTED field-name set.
    let selected_names = selector_names(m, selector)?;

    // Every selected name must be a field of the record.
    let is_selected = |name: Symbol| selected_names.iter().any(|s| m.resolve(name) == &**s);
    let all_present = selected_names
        .iter()
        .all(|s| fields.iter().any(|(n, _)| m.resolve(*n) == &**s));
    if !all_present {
        return None;
    }

    // 3. Partition the record fields into SELECTED and NON-selected, preserving
    //    record order within each group.
    let mut selected_fields: Vec<(Symbol, NodeId)> = Vec::new();
    let mut nonselected_fields: Vec<(Symbol, NodeId)> = Vec::new();
    for &(name, value) in &fields {
        if is_selected(name) {
            selected_fields.push((name, value));
        } else {
            nonselected_fields.push((name, value));
        }
    }
    // A vacuous split (nothing selected, or nothing left to condition on) is not
    // a structural disintegration.
    if selected_fields.is_empty() || nonselected_fields.is_empty() {
        return None;
    }

    // The kernel's boundary inputs are the NON-selected fields. Each entry is
    // `(name, ref)` where `ref` is how the value is referenced — a bare `%ref`
    // in the explicit-DAG shape (`(%field theta1 (%ref self theta1))`). A
    // non-ref value falls outside the case → refuse.
    let mut spec_inputs: Vec<(Symbol, Ref)> = Vec::with_capacity(nonselected_fields.len());
    for &(name, value) in &nonselected_fields {
        let Node::Ref(r) = m.node(value) else {
            return None;
        };
        spec_inputs.push((name, *r));
    }

    // 4. Build the kernel: `kernelof(record(<selected fields>), %specinputs(…))`.
    let record_sym = m.intern("record");
    let kernel_body = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(record_sym),
        args: Vec::<NodeId>::new().into(),
        named: selected_fields
            .iter()
            .map(|&(name, value)| NamedArg {
                kind: NamedKind::Field,
                name,
                value,
            })
            .collect::<Vec<_>>()
            .into(),
        inputs: None,
    }));
    let kernelof_sym = m.intern("kernelof");
    let kernel = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(kernelof_sym),
        args: vec![kernel_body].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: Some(Inputs::Spec(spec_inputs.into())),
    }));

    // 5. Build the marginal: `lawof(record(<non-selected fields>))`.
    let marginal_body = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(record_sym),
        args: Vec::<NodeId>::new().into(),
        named: nonselected_fields
            .iter()
            .map(|&(name, value)| NamedArg {
                kind: NamedKind::Field,
                name,
                value,
            })
            .collect::<Vec<_>>()
            .into(),
        inputs: None,
    }));
    let lawof_sym = m.intern("lawof");
    let marginal = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(lawof_sym),
        args: vec![marginal_body].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }));

    Some((kernel, marginal))
}

/// The field names a `disintegrate` selector picks (spec §06: works like `get`
/// — `"obs"` selects field `obs`; `["obs", …]` selects each named field).
/// `Some` only when every entry is a literal string (a bare `Scalar::Str`, or a
/// `vector(...)` of `Scalar::Str`); an index selector, a non-literal, or an
/// empty `vector` ⇒ `None` (mirrors `flatppl_infer`'s `selector_field_names`).
fn selector_names(m: &Module, node: NodeId) -> Option<Vec<Box<str>>> {
    match m.node(node) {
        Node::Lit(Scalar::Str(s)) => Some(vec![s.clone()]),
        Node::Call(c) => {
            let CallHead::Builtin(op) = c.head else {
                return None;
            };
            if m.resolve(op) != "vector" {
                return None;
            }
            let names: Option<Vec<Box<str>>> = c
                .args
                .iter()
                .map(|&a| match m.node(a) {
                    Node::Lit(Scalar::Str(s)) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            // An empty selector (`[]` → `(vector)`) has no defined meaning.
            names.filter(|v| !v.is_empty())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::split_disintegrate;
    use flatppl_core::{Call, CallHead, Inputs, Module, NamedKind, Node, NodeId, Ref, RefNs};

    fn parse_infer(src: &str) -> Module {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    }

    /// Locate the `disintegrate(...)` call node in an inferred module (the rhs of
    /// the desugared `__0xN` binding).
    fn find_disintegrate(m: &Module) -> NodeId {
        for (_, b) in m.bindings() {
            if let Node::Call(c) = m.node(b.rhs) {
                if let CallHead::Builtin(op) = c.head {
                    if m.resolve(op) == "disintegrate" {
                        return b.rhs;
                    }
                }
            }
        }
        panic!("no disintegrate node in module");
    }

    /// The builtin-call node at `id`, asserting its head name.
    fn expect_call<'a>(m: &'a Module, id: NodeId, name: &str) -> &'a Call {
        let Node::Call(c) = m.node(id) else {
            panic!("node is not a call: {:?}", m.node(id));
        };
        let CallHead::Builtin(op) = c.head else {
            panic!("call is not a builtin");
        };
        assert_eq!(m.resolve(op), name, "unexpected head");
        c
    }

    /// The `%field` entries of a `record(...)` node as `(name, value)` pairs.
    fn record_fields(m: &Module, id: NodeId) -> Vec<(String, NodeId)> {
        let rec = expect_call(m, id, "record");
        assert!(rec.args.is_empty(), "record has positional args");
        rec.named
            .iter()
            .map(|na| {
                assert_eq!(na.kind, NamedKind::Field, "non-field entry in record");
                (m.resolve(na.name).to_string(), na.value)
            })
            .collect()
    }

    /// A `(%ref self <name>)` value node → its bound name.
    fn self_ref_name(m: &Module, id: NodeId) -> String {
        match m.node(id) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => m.resolve(*name).to_string(),
            other => panic!("expected `(%ref self …)`, got {other:?}"),
        }
    }

    /// A minimal bi3-shape joint: two prior variates and an iid observation,
    /// bundled into a joint law and split with `disintegrate(["obs"], …)`.
    const BI3: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Gamma(alpha = 2, beta = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
forward_kernel, prior = disintegrate([\"obs\"], joint_model)";

    #[test]
    fn splits_bi3_into_kernel_and_marginal() {
        let mut m = parse_infer(BI3);
        let disint = find_disintegrate(&m);
        let (kernel, marginal) =
            split_disintegrate(&mut m, disint).expect("bi3 explicit-DAG disintegration must split");

        // Kernel = kernelof(record(obs = <obs rhs>), %specinputs(theta1, theta2)):
        // the SELECTED field `obs` is the body, the NON-selected fields are the
        // boundary inputs (verbatim value refs).
        let k = expect_call(&m, kernel, "kernelof");
        assert_eq!(k.args.len(), 1, "kernelof takes one positional body");
        let body_fields = record_fields(&m, k.args[0]);
        assert_eq!(
            body_fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["obs"],
            "kernel body must be exactly the selected `obs` field"
        );
        assert_eq!(
            self_ref_name(&m, body_fields[0].1),
            "obs",
            "kernel body `obs` field must carry the verbatim `(%ref self obs)`"
        );
        // Boundary inputs are exactly the NON-selected fields, as `(name, ref)`.
        let Some(Inputs::Spec(entries)) = &k.inputs else {
            panic!("kernel must carry %specinputs");
        };
        let got: Vec<(String, RefNs, String)> = entries
            .iter()
            .map(|(nm, r)| {
                (
                    m.resolve(*nm).to_string(),
                    r.ns,
                    m.resolve(r.name).to_string(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("theta1".into(), RefNs::SelfMod, "theta1".into()),
                ("theta2".into(), RefNs::SelfMod, "theta2".into()),
            ],
            "boundary inputs must be exactly the non-selected fields bound to their value refs"
        );

        // Marginal = lawof(record(theta1 = …, theta2 = …)) over non-selected only.
        let law = expect_call(&m, marginal, "lawof");
        assert_eq!(law.args.len(), 1, "lawof takes one positional body");
        let marg_fields = record_fields(&m, law.args[0]);
        assert_eq!(
            marg_fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["theta1", "theta2"],
            "marginal must carry exactly the non-selected fields, in order"
        );
        assert_eq!(self_ref_name(&m, marg_fields[0].1), "theta1");
        assert_eq!(self_ref_name(&m, marg_fields[1].1), "theta2");
    }

    #[test]
    fn refuses_selector_naming_a_non_field() {
        // `["bogus"]` names no field of the record → refuse (None).
        let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
obs ~ Normal(mu = theta1, sigma = 1)
joint_model = lawof(record(theta1 = theta1, obs = obs))
fk, pr = disintegrate([\"bogus\"], joint_model)";
        let mut m = parse_infer(src);
        let disint = find_disintegrate(&m);
        assert!(
            split_disintegrate(&mut m, disint).is_none(),
            "a selector naming no field must refuse"
        );
    }

    #[test]
    fn refuses_selecting_all_fields() {
        // Selecting every field leaves an empty non-selected (conditioning) set
        // → a vacuous split → refuse.
        let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
obs ~ Normal(mu = theta1, sigma = 1)
joint_model = lawof(record(theta1 = theta1, obs = obs))
fk, pr = disintegrate([\"theta1\", \"obs\"], joint_model)";
        let mut m = parse_infer(src);
        let disint = find_disintegrate(&m);
        assert!(
            split_disintegrate(&mut m, disint).is_none(),
            "selecting all fields (empty conditioning set) must refuse"
        );
    }
}
