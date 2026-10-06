//! Host query inputs must survive lowering at the declared module boundary.
use flatppl_determinizer::{LoweringOptions, determinize_with_options};
use flatppl_infer::ModuleBundle;

mod common;

fn output(src: &str, options: &LoweringOptions<'_>) -> String {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut module);
    let lowered = determinize_with_options(&module, &ModuleBundle::new(), None, options)
        .expect("the host query must lower");
    common::pir_binding(&flatppl_flatpir::write(&lowered), "outputs")
}

#[test]
fn record_queries_match_explicit_field_projections() {
    let options = LoweringOptions::default();
    for (model, point, projected, measure) in [
        (
            "x ~ Normal(mu = 0.0, sigma = 1.0)",
            "elementof(cartprod(x = reals))",
            "record(x = point.x)",
            "lawof(record(x = x))",
        ),
        (
            "x ~ Normal(mu = 0.0, sigma = 1.0)\ny ~ Normal(mu = x, sigma = 2.0)",
            "elementof(cartprod(x = reals))",
            "record(x = point.x)",
            "likelihoodof(kernelof(record(y = y), x = x), record(y = 1.0))",
        ),
        (
            "",
            "elementof(cartprod(a = cartprod(x = reals)))",
            "record(a = record(x = point.a.x))",
            "joint(a = joint(x = Normal(mu = 2.0, sigma = 3.0)))",
        ),
        (
            "x ~ Normal(mu = 0.0, sigma = 1.0)",
            "record(x = 99.0)",
            "record(x = point.x)",
            "lawof(record(x = x))",
        ),
    ] {
        let prefix = format!("{model}\npoint = {point}\ninputs = point\n");
        assert_eq!(
            output(
                &format!("{prefix}outputs = logdensityof({measure}, point)"),
                &options
            ),
            output(
                &format!("{prefix}outputs = logdensityof({measure}, {projected})"),
                &options
            ),
            "record query must read the runtime fields for {measure}"
        );
    }
}

#[test]
fn intact_rand_result_preserves_state_without_changing_legacy_default() {
    let query = "state = rnginit(0)\noutputs = rand(state, Normal(mu = 0.0, sigma = 1.0))";
    let strict = LoweringOptions {
        preserve_rand_tuple: true,
        ..Default::default()
    };
    let pair = output(query, &strict);
    let value = common::call_arg(&pair, "tuple", 0);
    let state = common::call_arg(&pair, "tuple", 1);
    let legacy = output(query, &LoweringOptions::default());
    assert_eq!(value, common::call_arg(&legacy, "%bind", 1));
    assert_eq!(common::call_arg(&value, "get0", 1), "0");
    assert_eq!(common::call_arg(&state, "get0", 1), "1");
    assert_eq!(
        common::call_arg(&value, "get0", 0),
        common::call_arg(&state, "get0", 0),
        "value and advanced state must come from the same sample"
    );
}
