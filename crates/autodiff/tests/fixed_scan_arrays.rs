use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_infer::{ModuleBundle, Severity};

fn lower(source: &str) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}

fn check_derivatives(source: &str) {
    let deterministic = lower(source);
    for differentiate in [gradient, jvp, vjp] {
        let derivative = differentiate(&deterministic, &["rate"]).unwrap();
        flatppl_determinizer::is_flatpdl(&derivative).unwrap();
        let text = flatppl_syntax::print(&derivative);
        let mut reparsed = flatppl_syntax::parse(&text).unwrap();
        let diagnostics = flatppl_infer::infer(&mut reparsed);
        assert!(
            diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{diagnostics:?}\n{text}"
        );
    }
}

#[test]
fn terminal_record_scan_projects_a_matrix_carry() {
    check_derivatives(
        "
rate = elementof(reals)
initial = record(grid = fill(1.0, [2, 2]), total = 1.0)
step(state, value, rate) = record(
    grid = (state.grid .* rate) .+ (state.total * real(value)),
    total = state.total + real(value) * rate
)
trajectory = scan(step, initial, indicesof(zeros(100)), rate)
inputs = rate
outputs = sum(trajectory[100].grid)
",
    );
}

#[test]
fn vector_scan_keeps_all_array_carry_outputs() {
    check_derivatives(
        "
rate = elementof(reals)
step(state, value, rate) = state .+ [rate, real(value) * rate]
trajectory = scan(step, [0.0, 0.0], [1, 2, 3], rate)
inputs = rate
outputs = sum(rowstack(trajectory))
",
    );
}

#[test]
fn record_scan_gathers_with_the_promoted_index() {
    for (selector, expected, minimum) in [
        ("trajectory[which + 1]", "[which + 1]", 1),
        ("get0(trajectory, which)", ", which)", 2),
    ] {
        let source = format!(
            "
rate = elementof(reals)
which = 2
initial = record(grid = fill(1.0, [2, 2]))
step(state, value, rate) = record(grid = state.grid .* rate .+ real(value))
trajectory = scan(step, initial, [1, 2, 3], rate)
inputs = (rate, which)
outputs = sum({selector}.grid)
"
        );
        let deterministic = lower(&source);
        for differentiate in [gradient, jvp, vjp] {
            let derivative = differentiate(&deterministic, &["rate"]).unwrap();
            let text = flatppl_syntax::print(&derivative);
            assert!(text.matches(expected).count() >= minimum, "{text}");
        }
    }
}
