//! Black-box members (flatppl-bridge) compile to `stablehlo.custom_call
//! @flatppl.box` operations that carry the protocol's primitive queries.

use std::sync::Arc;

use flatppl_host::{BoxSpec, Context, Diagnostic, EmitOptions, Export, parse_describe};

const INTERFACE: &str = "posterior = external(anything)
K = external(anything)
f = external(anything)
std = external(anything)
";

const DESCRIBE: &str = r#"{
  "posterior": {"kind": "measure", "queries": ["builtin_logdensityof"], "gradient": true, "vjp": false, "variate": "cartpow(reals, 2)", "normalized": false},
  "K": {"kind": "kernel", "queries": ["builtin_logdensityof", "builtin_sample"], "gradient": true, "vjp": false, "input": "cartprod(mu = reals, sigma = posreals)", "variate": "reals", "normalized": true},
  "f": {"kind": "function", "queries": ["apply"], "gradient": true, "vjp": true, "input": "cartpow(reals, 3)", "output": "reals"},
  "std": {"kind": "measure", "queries": ["builtin_logdensityof"], "gradient": true, "vjp": false, "variate": "reals", "normalized": true}
}"#;

fn compile_with(
    model: &str,
    interface: &str,
    describe: &str,
    options: &EmitOptions,
) -> Result<Export, Diagnostic> {
    let mut context = Context::default();
    let iface = context.parse(interface, None, None)?;
    context.register("iface.flatppl", &iface)?;
    let source = format!("bb = load_module(\"iface.flatppl\")\n{model}");
    let module = context.parse(&source, None, None)?;
    let spec: Arc<BoxSpec> = Arc::new(parse_describe("demo", describe).unwrap());
    module.compile_with_boxes(options, &[("bb".to_string(), spec)])
}

fn compile(model: &str) -> Result<Export, Diagnostic> {
    compile_with(model, INTERFACE, DESCRIBE, &EmitOptions::default())
}

fn box_lines(stablehlo: &str) -> Vec<&str> {
    stablehlo
        .lines()
        .filter(|line| line.contains("stablehlo.custom_call @flatppl.box("))
        .collect()
}

#[test]
fn a_measure_member_scores_as_one_density_query() {
    let export = compile(
        "theta = elementof(cartpow(reals, 2))\n\
         inputs = theta\n\
         outputs = logdensityof(bb.posterior, theta)",
    )
    .unwrap();
    let calls = box_lines(&export.stablehlo);
    assert_eq!(calls.len(), 1, "{}", export.stablehlo);
    let call = calls[0];
    for part in [
        "box = \"demo\"",
        "member = \"posterior\"",
        "operation = \"logdensity\"",
        "query = \"builtin_logdensityof(B.posterior, x)\"",
        "{name = \"x\", valueset = \"cartpow(reals, 2)\", rowwise = false}",
        "rows = []",
        "api_version = 4 : i32",
        ": (tensor<2xf32>) -> tensor<f32>",
    ] {
        assert!(call.contains(part), "missing {part}: {call}");
    }
}

#[test]
fn iid_of_a_kernel_member_is_one_broadcast_query() {
    let export = compile(
        "mu = elementof(reals)\n\
         inputs = mu\n\
         outputs = logdensityof(iid(bb.K(mu = mu, sigma = 1.0), 3), [0.1, 0.2, 0.3])",
    )
    .unwrap();
    let calls = box_lines(&export.stablehlo);
    assert_eq!(calls.len(), 1, "{}", export.stablehlo);
    let call = calls[0];
    for part in [
        "query = \"broadcast(builtin_logdensityof, B.K, input, x)\"",
        "{name = \"input\", valueset = \"cartprod(mu = reals, sigma = reals)\", rowwise = false, fields = [\"mu\", \"sigma\"]}",
        "{name = \"x\", valueset = \"reals\", rowwise = true}",
        "rows = [3]",
        "-> tensor<3xf32>",
    ] {
        assert!(call.contains(part), "missing {part}: {call}");
    }
    assert!(export.stablehlo.contains("stablehlo.reduce"));
}

#[test]
fn joint_and_weighted_reduce_to_member_queries() {
    let export = compile(
        "mu = elementof(reals)\n\
         x = elementof(reals)\n\
         M = joint(a = bb.K(mu = mu, sigma = 1.0), b = weighted(2.0, bb.std))\n\
         inputs = (mu, x)\n\
         outputs = logdensityof(M, record(a = x, b = 0.2))",
    )
    .unwrap();
    let calls = box_lines(&export.stablehlo);
    assert_eq!(calls.len(), 2, "{}", export.stablehlo);
    assert!(calls.iter().any(|c| c.contains("member = \"K\"")));
    assert!(calls.iter().any(|c| c.contains("member = \"std\"")));
}

#[test]
fn a_draw_from_a_kernel_member_scores_in_the_joint_density() {
    let export = compile(
        "mu = elementof(reals)\n\
         y ~ bb.K(mu = mu, sigma = 1.0)\n\
         inputs = mu\n\
         outputs = logdensityof(lawof(record(y = y)), record(y = 0.5))",
    )
    .unwrap();
    assert_eq!(box_lines(&export.stablehlo).len(), 1);
}

#[test]
fn a_function_member_is_an_application_query() {
    let export = compile(
        "x = elementof(cartpow(reals, 3))\n\
         inputs = x\n\
         outputs = 2.0 * bb.f(x) + 1.0",
    )
    .unwrap();
    let calls = box_lines(&export.stablehlo);
    assert_eq!(calls.len(), 1);
    assert!(calls[0].contains("operation = \"apply\""), "{}", calls[0]);
    assert!(calls[0].contains("query = \"B.f(input)\""), "{}", calls[0]);
    assert!(
        calls[0].contains(": (tensor<3xf32>) -> tensor<f32>"),
        "{}",
        calls[0]
    );
}

#[test]
fn a_broadcast_function_member_is_one_query() {
    let export = compile(
        "xs = elementof(cartpow(cartpow(reals, 3), 4))\n\
         inputs = xs\n\
         outputs = sum(bb.f.(xs))",
    )
    .unwrap();
    let calls = box_lines(&export.stablehlo);
    assert_eq!(calls.len(), 1, "{}", export.stablehlo);
    assert!(
        calls[0].contains("query = \"broadcast(B.f, input)\""),
        "{}",
        calls[0]
    );
    assert!(calls[0].contains("rows = [4]"), "{}", calls[0]);
    assert!(calls[0].contains("valueset = \"cartpow(reals, 3)\", rowwise = true"));
}

#[test]
fn a_sample_seeds_the_box_and_advances_the_model_state() {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        ..EmitOptions::default()
    };
    let export = compile_with(
        "state = rnginit(0)\n\
         mu = elementof(reals)\n\
         inputs = (state, mu)\n\
         outputs = rand(iid(bb.K(mu = mu, sigma = 2.0), 4), state)",
        INTERFACE,
        DESCRIBE,
        &options,
    )
    .unwrap();
    let calls = box_lines(&export.stablehlo);
    assert_eq!(calls.len(), 1, "{}", export.stablehlo);
    assert!(calls[0].contains("operation = \"sample\""));
    assert!(calls[0].contains("query = \"builtin_sample(rng, B.K, input, 4)\""));
    assert!(calls[0].contains("sizes = [4]"));
    assert!(calls[0].contains("(tensor<2xui64>, tensor<f32>, tensor<f32>) -> tensor<4xf32>"));
    assert!(export.stablehlo.contains("stablehlo.rng_bit_generator"));
}

fn message(result: Result<Export, Diagnostic>) -> String {
    match result {
        Ok(_) => panic!("the model compiled"),
        Err(error) => error.message,
    }
}

fn refusal(model: &str) -> String {
    message(compile(model))
}

#[test]
fn sampling_an_unnormalized_member_refuses() {
    let message =
        refusal("state = rnginit(0)\ninputs = state\noutputs = rand(bb.posterior, state)");
    assert!(
        message.contains("requires a probability measure"),
        "{message}"
    );
    let message = refusal("x ~ bb.posterior\ninputs = x\noutputs = x");
    assert!(
        message.contains("requires a probability measure"),
        "{message}"
    );
}

#[test]
fn a_member_without_the_query_refuses() {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        ..EmitOptions::default()
    };
    let message = message(compile_with(
        "state = rnginit(0)\ninputs = state\noutputs = rand(bb.std, state)",
        INTERFACE,
        DESCRIBE,
        &options,
    ));
    assert!(
        message.contains("does not answer `builtin_sample`"),
        "{message}"
    );
}

#[test]
fn a_kernel_without_its_input_is_no_measure() {
    let message = refusal("x = elementof(reals)\ninputs = x\noutputs = logdensityof(bb.K, x)");
    assert!(message.contains("apply it to an input"), "{message}");
}

#[test]
fn measure_algebra_without_a_primitive_reduction_refuses() {
    let message = refusal(
        "x = elementof(reals)\ninputs = x\noutputs = logdensityof(normalize(bb.posterior), [x, x])",
    );
    assert!(message.contains("normalize"), "{message}");
}

#[test]
fn the_interface_must_match_the_box() {
    let missing = message(compile_with(
        "x = elementof(reals)\ninputs = x\noutputs = x",
        "posterior = external(anything)\nextra = external(anything)\n",
        DESCRIBE,
        &EmitOptions::default(),
    ));
    assert!(
        missing.contains("box `demo` has no member `extra`"),
        "{missing}"
    );
    let typed = message(compile_with(
        "x = elementof(reals)\ninputs = x\noutputs = x",
        "posterior = external(reals)\n",
        DESCRIBE,
        &EmitOptions::default(),
    ));
    assert!(
        typed.contains("must be declared `external(anything)`"),
        "{typed}"
    );
}

#[test]
fn a_box_query_inside_a_broadcast_body_refuses() {
    let message = refusal(
        "xs = elementof(cartpow(cartpow(reals, 3), 2))\n\
         v = elementof(cartpow(reals, 3))\n\
         g = functionof(bb.f(v) + 1.0, v = v)\n\
         inputs = xs\n\
         outputs = sum(g.(xs))",
    );
    assert!(message.contains("top-level operation"), "{message}");
}

#[test]
fn the_ordinary_rules_reduce_around_member_leaves() {
    for output in [
        "logdensityof(likelihoodof(bb.K, 0.3), record(mu = x, sigma = 1.0))",
        "densityof(bb.std, x)",
        "logdensityof(logweighted(-1.0, bb.std), x)",
        "logdensityof(truncate(bb.std, interval(0.0, 2.0)), x)",
        "logdensityof(pushfwd(bijection(exp, log, y -> y), bb.std), x)",
    ] {
        let export = compile(&format!(
            "x = elementof(reals)\ninputs = x\noutputs = {output}"
        ))
        .unwrap_or_else(|e| panic!("{output}: {}", e.message));
        assert_eq!(box_lines(&export.stablehlo).len(), 1, "{output}");
    }
}
