use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn infer(module: &mut Module) {
    let diagnostics = flatppl_infer::infer(module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
}

fn check_transforms(source: &str, inputs: &[&str]) {
    let mut model = flatppl_syntax::parse(source).unwrap();
    infer(&mut model);
    let module = determinize_query(&model, &ModuleBundle::new()).unwrap();
    for transform in [gradient, jvp, vjp] {
        let transformed = transform(&module, inputs).unwrap();
        let mut reparsed = flatppl_syntax::parse(&flatppl_syntax::print(&transformed)).unwrap();
        infer(&mut reparsed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn differentiates_explicit_and_dotted_cumulative_maps() {
    check_transforms(
        "
x = elementof(cartpow(reals, [2, 3]))
rows = indicesof(zeros(2))
prefix(i, x) = cumsum(x[i, :])
sums = rowstack(prefix.(rows, [x]))
vectors = [x[1, :], x[2, :]]
peaks = rowstack(cummax.(vectors))
inputs = x
outputs = sum(sums .* sums) + sum(peaks .* peaks)
",
        &["x"],
    );
}

#[test]
fn differentiates_nested_fixed_event_callbacks() {
    check_transforms(
        "
means = elementof(cartpow(reals, 2))
points = elementof(cartpow(reals, 4))
quadratic(x, m) = -((x - m)^2)
mixture(x, m) = logsumexp(quadratic.(x, m))
inputs = (points, means)
outputs = sum(mixture.(points, [means]))
",
        &["points", "means"],
    );
}
