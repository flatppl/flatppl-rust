use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
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
fn runtime_orbital_label_accepts_continuous_products_only() {
    let source = r#"
hep = standard_module("particle-physics", "0.1")
l = elementof(nonnegintegers)
p = elementof(posreals)
d = elementof(posreals)
inputs = (l, p, d)
"#;
    for expression in [
        "hep.blatt_weisskopf(l,p,d)",
        "real(hep.resonance_breitwigner(p,2.0,0.17,0.3,0.5,l,d))",
        "imag(hep.resonance_breitwigner(p,2.0,0.17,0.3,0.5,l,d))",
    ] {
        let module = lower(&format!("{source}\noutputs = {expression}\n"));
        for transform in [gradient, jvp, vjp] {
            let ad = transform(&module, &["p", "d"]).unwrap();
            lower(&flatppl_syntax::print(&ad));
            assert!(transform(&module, &["l"]).is_err());
        }
    }
}
