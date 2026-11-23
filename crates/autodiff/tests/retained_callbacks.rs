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

fn lower(source: &str) -> Module {
    let mut model = flatppl_syntax::parse(source).unwrap();
    infer(&mut model);
    determinize_query(&model, &ModuleBundle::new()).unwrap()
}

fn check_transforms(source: &str, inputs: &[&str]) {
    let module = lower(source);
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

const PROBABILITY_SCAN: &str = "
q = elementof(unitinterval)
high = elementof(unitinterval)
low = elementof(unitinterval)
items = elementof(cartpow(reals, 65))
step(state, r, p) = record(previous = state.next, next = p.low + (p.high - p.low) * invlogit(logit(state.next) + r))
project(state) = state.previous
weights = project.(scan(step, record(previous = q, next = q), items, record(high = high, low = low)))
inputs = (q, high, low, items)
outputs = sum(weights)
";
const SCAN_INPUTS: &[&str] = &["q", "high", "low", "items"];

#[test]
fn differentiates_a_long_projected_probability_scan() {
    check_transforms(PROBABILITY_SCAN, SCAN_INPUTS);
}

#[test]
fn a_two_lag_scan_does_not_match_the_one_carry_rule() {
    let source = PROBABILITY_SCAN.replace("logit(state.next)", "logit(state.previous)");
    let module = lower(&source);
    assert!(gradient(&module, SCAN_INPUTS).is_err());
}

#[test]
fn differentiates_rectangular_row_maximum() {
    check_transforms(
        "
x = elementof(cartpow(reals, [2, 3]))
rowmax = aggregate(maximum, [.r], x[.r, .c])
inputs = x
outputs = sum(rowmax)
",
        &["x"],
    );
}

#[test]
fn differentiates_independent_normal_rows() {
    check_transforms(
        "
x = elementof(cartpow(cartpow(reals, 4), 3))
scale = elementof(posreals)
parameters = broadcast(record, mu = [[0.0]], sigma = [[scale]])
inputs = (x, scale)
outputs = sum(builtin_logdensityof.(Normal, parameters, x))
",
        &["x", "scale"],
    );
}
