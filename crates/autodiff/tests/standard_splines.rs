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
fn retained_splines_emit_source_products_and_keep_primal_calls() {
    for (member, args, output) in [
        ("interp_linear", "x, t, y", "curve"),
        ("interp_cubic", "x, t, y", "curve"),
        ("interp_pchip", "x, t, y", "curve"),
        ("bspline_basis", "x, t, 1", "sum(curve)"),
        ("bspline_derivative", "x, t, 1", "sum(curve)"),
        ("bspline_integral", "x, 1.5, t, 1", "sum(curve)"),
        ("bspline_penalty", "t, 1, 0", "curve[1, 2]"),
    ] {
        let source = format!(
            "s = standard_module(\"splines\", \"0.1\")\n\
             x = elementof(reals)\nt = elementof(cartpow(reals, 4))\n\
             y = elementof(cartpow(reals, 4))\ninputs = (x, t, y)\n\
             curve = s.{member}({args})\noutputs = {output}\n"
        );
        let module = lower(&source);
        for transform in [gradient, jvp, vjp] {
            let derivative = transform(&module, &["x", "t", "y"]).unwrap();
            let printed = flatppl_syntax::print(&derivative);
            assert!(printed.contains(member), "{printed}");
            lower(&printed);
        }
    }
}

#[test]
fn runtime_orthogonal_polynomial_degree_is_passive_in_all_products() {
    let module = lower(
        "p = standard_module(\"polynomials\", \"0.1\")\n\
         x = elementof(reals)\nn = elementof(integers)\n\
         inputs = (n, x)\noutputs = p.legendre(n, x)\n",
    );
    for transform in [gradient, jvp, vjp] {
        let derivative = transform(&module, &["x"]).unwrap();
        lower(&flatppl_syntax::print(&derivative));
        let refusal = transform(&module, &["n"]).unwrap_err();
        assert!(refusal.to_string().contains("differentiable real"));
    }
}
