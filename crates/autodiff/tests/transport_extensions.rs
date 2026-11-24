use flatppl_infer::{ModuleBundle, Severity};

#[test]
fn new_real_transport_rules_roundtrip_through_printed_source() {
    for (head, kernel, parameters, select) in [
        ("touniform", "Exponential", "rate = a", &["a", "x"][..]),
        ("touniform", "Logistic", "mu = a, s = b", &["a", "b", "x"]),
        (
            "tonormal",
            "LogNormal",
            "mu = a, sigma = b",
            &["a", "b", "x"],
        ),
        (
            "fromnormal",
            "LogNormal",
            "mu = a, sigma = b",
            &["a", "b", "x"],
        ),
    ] {
        let inputs = if select.len() == 2 { "a, x" } else { "a, b, x" };
        let source = format!(
            "a = elementof(reals)\nb = elementof(posreals)\nx = elementof(reals)\n\
             inputs = ({inputs})\noutputs = builtin_{head}({kernel}, record({parameters}), x)\n"
        );
        let mut parsed = flatppl_syntax::parse(&source).unwrap();
        assert!(
            flatppl_infer::infer(&mut parsed)
                .iter()
                .all(|d| d.severity != Severity::Error)
        );
        let lowered = flatppl_autodiff::determinize_query(&parsed, &ModuleBundle::new()).unwrap();
        for selection in std::iter::once(select).chain(select.iter().map(std::slice::from_ref)) {
            for method in [
                flatppl_autodiff::gradient,
                flatppl_autodiff::jvp,
                flatppl_autodiff::vjp,
            ] {
                let mut ad = method(&lowered, selection).unwrap();
                flatppl_autodiff::prepare_source_with_options(&mut ad, &Default::default())
                    .unwrap();
                let mut replay = flatppl_syntax::parse(&flatppl_syntax::print(&ad)).unwrap();
                let diagnostics = flatppl_infer::infer(&mut replay);
                assert!(
                    diagnostics.iter().all(|d| d.severity != Severity::Error),
                    "{diagnostics:?}"
                );
            }
        }
    }
}
