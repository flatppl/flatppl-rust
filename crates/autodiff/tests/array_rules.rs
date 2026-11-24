use flatppl_autodiff::{determinize_query, gradient, jvp, prepare_source, vjp};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn lower(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}

#[test]
fn array_rules_preserve_source_types_with_both_operands_active() {
    for (prefix, expression, selected) in [
        (
            "a = elementof(cartpow(reals, 5))\nb = elementof(cartpow(reals, 3))",
            "sum(conv(a,b))",
            &["a", "b"][..],
        ),
        (
            "a = elementof(cartpow(reals, 5))\nb = elementof(cartpow(reals, 1))",
            "sum(crosscorr(a,b))",
            &["a", "b"],
        ),
        (
            "a = elementof(cartpow(reals, [3, 2]))",
            "sum(diag(a,-1))",
            &["a"],
        ),
        (
            "a = elementof(cartpow(reals, 4))\nb = elementof(unitinterval)",
            "quantile(a,b)",
            &["a", "b"],
        ),
        (
            "a = elementof(cartpow(reals, 4))\nb = elementof(unitinterval)",
            "bernstein(a,b)",
            &["a", "b"],
        ),
        (
            "a = elementof(cartpow(reals, 4))",
            "sum(logsoftmax(l2unit(a)))",
            &["a"],
        ),
        (
            "a = elementof(cartpow(reals, [2, 2]))\nb = elementof(cartpow(reals, 2))",
            "quadform(a,b)",
            &["a", "b"],
        ),
    ] {
        let inputs = selected.join(", ");
        for differentiate in [gradient, jvp, vjp] {
            let model = lower(&format!(
                "{prefix}\ninputs = ({inputs})\noutputs = {expression}"
            ));
            let mut derivative =
                differentiate(&model, selected).unwrap_or_else(|err| panic!("{expression}: {err}"));
            prepare_source(&mut derivative);
            lower(&flatppl_syntax::print(&derivative));
        }
    }
}
