use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_infer::{ModuleBundle, Severity};

#[test]
fn projected_record_scan_keeps_carry_and_parameter_dependencies() {
    let source = "
rate = elementof(reals)
initial = record(mean = 0, precision = 1)
step(carry, observation, params) = record(
    mean = carry.mean + carry.precision * observation * params.rate,
    precision = carry.precision + params.rate
)
project(carry) = carry.mean
trajectory = scan(step, initial, [1, 2, 3], record(rate = rate))
inputs = rate
outputs = sum(project.(trajectory))
";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
    for differentiate in [gradient, jvp, vjp] {
        let derivative = differentiate(&deterministic, &["rate"]).unwrap();
        let text = flatppl_syntax::print(&derivative);
        let mut reparsed = flatppl_syntax::parse(&text).unwrap();
        let diagnostics = flatppl_infer::infer(&mut reparsed);
        assert!(
            diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{diagnostics:?}\n{text}"
        );
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn vector_record_scan_products_serialize_with_a_native_scan() {
    let source = "
state = elementof(cartpow(reals, 2))
scale = elementof(reals)
step(s, time, p) = record(y = s.y .* p, t = time)
times = [0.0, 1.0, 2.0]
trajectory = scan(step, record(y = state, t = 0.0), times, scale).y
inputs = (state, scale)
outputs = sum(rowstack(trajectory))
";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
    let deterministic = determinize_query(&module, &ModuleBundle::new()).unwrap();
    for differentiate in [gradient, jvp, vjp] {
        let derivative = differentiate(&deterministic, &["state", "scale"]).unwrap();
        let source = flatppl_syntax::print(&derivative);
        assert!(source.matches("scan(").count() >= 2);
        let mut reparsed = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut reparsed);
        assert!(
            diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{diagnostics:?}"
        );
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}
