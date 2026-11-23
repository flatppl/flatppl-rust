use flatppl_core::Module;
use flatppl_optimise::{Dtype, Profile, RulePolicy, Target, optimise, optimise_with_rules};

const HOIST_ALIAS: &str = "\
x = external(cartpow(reals, 8))
y = x
inputs = y
full = -x
picked = full[[8, 3, 5]]
";

fn optimise_output(source: &str) -> Option<Module> {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let output = module.intern("outputs");
    optimise(
        &module,
        Some(&[output]),
        Profile::StableHlo {
            target: Target::Cpu,
            dtype: Dtype::F32,
        },
    )
}

#[test]
fn opaque_consumer_of_a_promoted_alias_keeps_full_negation() {
    let model = |expr| format!("{HOIST_ALIAS}outputs = {expr}\n");
    assert!(optimise_output(&model("sum(picked)")).is_some());
    assert!(optimise_output(&model("sum(picked) + sum(sin.(y))")).is_none());
}

#[test]
fn other_projections_of_a_promoted_field_keep_nested_selection() {
    let model = "\
r = external(cartprod(x = cartpow(reals, 8)))
wa = external(cartpow(reals, 2))
wb = external(cartpow(reals, 2))
wc = external(cartpow(reals, 2))
inputs = (r, wa, wb, wc)
inner = r.x[[8, 3, 6, 1]]
a = get0(inner, [2, 1])
b = get0(r.x, [5, 2])
c = get0(r.x, [5, 2])
outputs = (sum(b .* wb) + sum(a .* wa)) + sum(c .* wc)
";
    assert!(optimise_output(model).is_none());
}

#[test]
fn alternating_rule_policies_only_affect_the_current_optimisation() {
    let mut model =
        flatppl_syntax::parse("x = external(reals)\ninputs = x\noutputs = identity(x)\n").unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    let outputs = [model.intern("outputs")];
    let default = RulePolicy::default();
    let keep_identity = RulePolicy::new(&["portable.identity"], &[]).unwrap();
    let keep_negation = RulePolicy::new(&["portable.double-negation"], &[]).unwrap();
    for (rules, removes_identity) in [
        (&default, true),
        (&keep_identity, false),
        (&keep_negation, true),
        (&keep_identity, false),
        (&default, true),
    ] {
        let result = optimise_with_rules(&model, Some(&outputs), Profile::Portable, rules);
        let graph = result.as_ref().unwrap_or(&model);
        assert_eq!(
            flatppl_flatpir::write(graph).contains("(identity "),
            !removes_identity
        );
    }
}
