use flatppl_autodiff::{determinize_query, vjp};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn model(steps: usize, offset: usize, terminal: &str) -> Module {
    let rows = steps + 1;
    let source = format!(
        "
initial = elementof(cartpow(reals, 3))
matrix = elementof(cartpow(reals, [3, 3]))
emissions = elementof(cartpow(reals, [{rows}, 3]))
steps = {rows} - 1
update(alpha, t, p) = maximum(alpha) .+ log.(p.matrix * exp.(alpha .- maximum(alpha))) .+ p.emissions[t, :]
inputs = (initial, matrix, emissions)
outputs = scan(update, initial, indicesof(zeros(steps)) .+ {offset},
    record(matrix = matrix, emissions = emissions))[{terminal}]
"
    );
    let mut source = flatppl_syntax::parse(&source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut source);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&source, &ModuleBundle::new()).unwrap()
}

const INPUTS: &[&str] = &["initial", "matrix", "emissions"];

#[test]
fn long_terminal_markov_scan_vjp_graph_has_constant_size() {
    let short = vjp(&model(499, 1, "steps"), INPUTS).unwrap();
    let long = vjp(&model(2_000, 1, "steps"), INPUTS).unwrap();
    assert_eq!(short.node_count(), long.node_count());
}

#[test]
fn terminal_markov_scan_requires_its_observation_itinerary() {
    // Both programs define valid output vectors, but have different
    // observation cotangents than the matched terminal scan.
    let shifted = model(499, 0, "steps");
    let interior = model(499, 1, "steps - 1");
    assert!(vjp(&shifted, INPUTS).is_err());
    assert!(vjp(&interior, INPUTS).is_err());
}
