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
fn dotted_convolution_accepts_named_arguments_and_singleton_frames() {
    let source = r#"
v = elementof(cartpow(cartpow(reals, 5), [2, 1]))
k = elementof(cartpow(cartpow(reals, 3), [1, 4]))
inputs = (v, k)
result = conv.(kernel = k, v = v)
"#;
    for (outputs, derive) in [
        ("sum(sum.(result))", gradient as fn(&Module, &[&str]) -> _),
        ("result", jvp),
        ("result", vjp),
    ] {
        let module = lower(&format!("{source}\noutputs = {outputs}\n"));
        let differentiated = derive(&module, &["v", "k"]).unwrap();
        lower(&flatppl_syntax::print(&differentiated));
    }
}

#[test]
fn batched_result_cells_and_repeated_matrix_cotangents_survive_reparse() {
    let source = r#"
x = elementof(cartpow(cartpow(reals, 3), 2))
A = elementof(cartpow(cartpow(reals, [2, 3]), 2))
inputs = (x, A)
p = softmax.(x)
gram = row_gram.(A)
outputs = (p, gram, gram)
"#;
    let module = lower(source);
    for derive in [jvp, vjp] {
        let differentiated = derive(&module, &["x", "A"]).unwrap();
        lower(&flatppl_syntax::print(&differentiated));
    }
}

#[test]
fn batched_normalization_reverse_masks_inactive_cells() {
    let source = r#"
x = elementof(cartpow(cartpow(reals, 3), 2))
predicate = elementof(booleans)
inputs = (x, predicate)
outputs = ifelse(predicate, sum(sum.(l2unit.(x))), 2.0 * sum(sum.(x)))
"#;
    let module = lower(source);
    let differentiated = gradient(&module, &["x"]).unwrap();
    lower(&flatppl_syntax::print(&differentiated));
}
