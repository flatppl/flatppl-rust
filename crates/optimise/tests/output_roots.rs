use flatppl_core::{CallHead, Node, RefNs};
use flatppl_optimise::{Dtype, Profile, Target, optimise};

#[test]
fn optimise_the_fields_of_a_named_record_without_replacing_its_abi_reference() {
    let mut module = flatppl_syntax::parse(
        "
x = external(cartpow(reals, 8))
y = external(reals)
inputs = (x, y)
report = record(sample = get(-x, [8, 2]), offset = y)
outputs = report
",
    )
    .unwrap();
    assert!(
        flatppl_infer::infer(&mut module)
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error)
    );
    let output = module.intern("outputs");
    let report = module.intern("report");
    let bound = module.binding_by_name(output).unwrap();
    let report_id = module.binding_by_name(report).unwrap();
    let optimised = optimise(
        &module,
        Some(&[output]),
        Profile::StableHlo {
            target: Target::Cpu,
            dtype: Dtype::F64,
        },
    )
    .expect("the CPU can negate the two selected elements");
    assert_eq!(module.binding(bound).rhs, optimised.binding(bound).rhs);
    assert!(
        matches!(optimised.node(optimised.binding(bound).rhs),Node::Ref(r) if r.ns==RefNs::SelfMod&&r.name==report)
    );
    let Node::Call(fields) = optimised.node(optimised.binding(report_id).rhs) else {
        panic!("record")
    };
    assert_eq!(optimised.resolve(fields.named[0].name), "sample");
    let Node::Call(neg) = optimised.node(fields.named[0].value) else {
        panic!("negated subset")
    };
    assert!(matches!(neg.head,CallHead::Builtin(s) if optimised.resolve(s)=="neg"));
    assert_eq!(optimised.resolve(fields.named[1].name), "offset");
}
