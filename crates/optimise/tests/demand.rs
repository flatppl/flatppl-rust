use std::collections::HashSet;

use flatppl_core::{Module, Node, NodeId, Scalar};
use flatppl_optimise::{Dtype, Profile, RulePolicy, Target, optimise, optimise_with_rules};

fn infer(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    module
}

fn rhs(module: &Module, name: &str) -> NodeId {
    module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .unwrap()
        .1
        .rhs
}

fn references(module: &Module, root: NodeId) -> HashSet<&str> {
    let mut names = HashSet::new();
    module.collect_referenced_names(root, &mut names);
    names.into_iter().map(|name| module.resolve(name)).collect()
}

const QUERY: &str = "
    _seed = 7.0
    promoted = _seed
    other_input = 11.0
    inputs = (promoted, other_input)
    _expensive = exp(tanh(exp(abs(1.2 * 3.4 + 5.6))))
    _group = (2.0, 3.0, _expensive)
    _bundle = record(group = _group, side = 5.0, unused = _expensive)
    left = get(get(_bundle, \"group\"), 1)
    right = get(_bundle, \"side\")
    _payload = [promoted, _expensive, 5.0, 6.0]
    chosen = get(_payload, 1)
    result = left + chosen
    outputs = (result, right)
";

#[test]
fn nested_component_demands_keep_both_outputs_and_query_inputs() {
    let mut m = infer(QUERY);
    let outputs = m.intern("outputs");
    let profile = Profile::StableHlo {
        target: Target::Cpu,
        dtype: Dtype::F64,
    };
    let out = optimise(&m, Some(&[outputs]), profile).expect("demand pruning");

    assert_eq!(out.node(rhs(&out, "left")), &Node::Lit(Scalar::Real(2.0)));
    assert_eq!(out.node(rhs(&out, "right")), &Node::Lit(Scalar::Real(5.0)));
    assert_eq!(
        references(&out, rhs(&out, "chosen")),
        HashSet::from(["promoted"])
    );
    assert_eq!(
        references(&out, rhs(&out, "outputs")),
        HashSet::from(["result", "right"])
    );
    for (id, binding) in m.bindings() {
        assert_eq!(out.binding_by_name(binding.name), Some(id));
    }
    assert_eq!(rhs(&out, "inputs"), rhs(&m, "inputs"));
    assert_eq!(rhs(&out, "_seed"), rhs(&m, "_seed"));
}

#[test]
fn default_portable_demand_tracks_public_uses_of_private_helpers() {
    let m = infer(QUERY);
    let out = optimise(&m, None, Profile::Portable).expect("demand pruning");
    assert_eq!(
        references(&out, rhs(&out, "chosen")),
        HashSet::from(["promoted"])
    );
    assert_eq!(out.node(rhs(&out, "left")), &Node::Lit(Scalar::Real(2.0)));
    assert_eq!(rhs(&out, "_payload"), rhs(&m, "_payload"));
    assert_eq!(rhs(&out, "_expensive"), rhs(&m, "_expensive"));
}

#[test]
fn whole_value_use_keeps_constructor_and_its_dependencies() {
    let m = infer(&format!("{QUERY}\nwhole = _bundle\n"));
    let out = optimise(&m, None, Profile::Portable).unwrap_or_else(|| m.clone());
    assert_eq!(
        references(&out, rhs(&out, "whole")),
        HashSet::from(["_bundle"])
    );
    assert_eq!(
        references(&out, rhs(&out, "_bundle")),
        HashSet::from(["_group", "_expensive"])
    );
    assert_eq!(
        references(&out, rhs(&out, "right")),
        HashSet::from(["_bundle"])
    );
}

#[test]
fn component_proofs_keep_runtime_checks_and_reification_inputs() {
    let mut m = infer(
        "
        check = external(booleans)
        _guarded = checked(9.0, check)
        guarded = get(record(wanted = 4.0, other = _guarded), \"wanted\")
        theta = elementof(reals)
        _reified = get((4.0, theta), 1)
        callable = functionof(_reified)
        idx = 1
        inputs = idx
        selection = get([2.0, 7.0, 11.0], idx)
    ",
    );
    let roots = [
        m.intern("guarded"),
        m.intern("callable"),
        m.intern("selection"),
    ];
    let out = optimise(&m, Some(&roots), Profile::Portable).unwrap_or_else(|| m.clone());
    assert!(references(&out, rhs(&out, "guarded")).contains("_guarded"));
    assert!(references(&out, rhs(&out, "_reified")).contains("theta"));
    assert!(references(&out, rhs(&out, "selection")).contains("idx"));
}

#[test]
fn portable_demand_keeps_nonselected_runtime_ancestry() {
    let mut m = infer(
        "x = external(reals)\ny = external(reals)\ninputs = (x, y)\noutputs = get([x, y], 1)",
    );
    let outputs = m.intern("outputs");
    let out = optimise(&m, Some(&[outputs]), Profile::Portable).unwrap_or_else(|| m.clone());
    assert_eq!(
        references(&out, rhs(&out, "outputs")),
        HashSet::from(["x", "y"])
    );
}

#[test]
fn a_disabled_demand_pass_keeps_cell_reads() {
    let m = infer(QUERY);
    let pass = RulePolicy::from_selectors(["pass:demand-prune".parse().unwrap()]);
    for rules in [pass, RulePolicy::all_disabled()] {
        let out =
            optimise_with_rules(&m, None, Profile::Portable, &rules).unwrap_or_else(|| m.clone());
        assert!(references(&out, rhs(&out, "left")).contains("_bundle"));
        assert!(references(&out, rhs(&out, "chosen")).contains("_payload"));
    }
}

#[test]
fn stan_reads_of_a_fixed_record_keep_their_binding() {
    let mut m = infer(
        "
        a = elementof(cartpow(reals, 3))
        s = elementof(posreals)
        d = (g = [3, 1, 2, 1, 3], y = [0.4, 1.0, 1.3, 2.0, 0.3])
        inputs = (a, s)
        outputs = logdensityof(Normal.(a[d.g], s), d.y)
    ",
    );
    let outputs = m.intern("outputs");
    let out = optimise(&m, Some(&[outputs]), Profile::Stan).unwrap_or_else(|| m.clone());
    assert!(references(&out, rhs(&out, "outputs")).contains("d"));
    let portable = optimise(&m, Some(&[outputs]), Profile::Portable).expect("demand pruning");
    assert!(!references(&portable, rhs(&portable, "outputs")).contains("d"));
}
