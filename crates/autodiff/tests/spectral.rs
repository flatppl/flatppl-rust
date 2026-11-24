use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_infer::{ModuleBundle, Severity};

#[test]
fn real_spectral_products_round_trip_as_ordinary_source() {
    for (call, score) in [
        ("l.svd(A)", "sum(y.S)"),
        ("l.cond(A)", "y"),
        ("l.pinv(A)", "sum(y)"),
        ("l.lstsq(A, b, 0.7, 0.02)", "sum(y)"),
    ] {
        for (index, product) in [gradient, jvp, vjp].into_iter().enumerate() {
            let scalar = index == 0;
            let output = if scalar { score } else { "y" };
            let source = format!(
                "l = standard_module(\"ext-linear-algebra\", \"0.1\")\n\
                 A = elementof(cartpow(reals, [3, 2]))\n\
                 b = elementof(cartpow(reals, 3))\ninputs = (A, b)\ny = {call}\noutputs = {output}\n"
            );
            let module = lower(&source);
            let derivative = product(&module, &["A"]).unwrap();
            lower(&flatppl_syntax::print(&derivative));
        }
    }
}

fn lower(source: &str) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|diag| diag.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}
