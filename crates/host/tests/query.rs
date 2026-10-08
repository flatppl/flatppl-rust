use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn registered_module_exports_a_typed_query_abi() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse("distribution = Normal(3.0, 1.0)", None, None)?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        r#"
        m = load_module("model.flatppl")
        distribution = m.distribution
        p = elementof(cartprod(z = reals, a = cartpow(reals, 2)))
        inputs = p
        result = record(total = p.z + logdensityof(iid(distribution, 2), p.a), original = p.a)
        outputs = result
        "#,
        None,
        None,
    )?;
    // The loaded module owns its source and resolved imports.
    drop(context);
    drop(model);
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(&exported)?;
    assert_eq!(
        schema["inputs"],
        json!([{
            "name": "p",
            "value": {
                "kind": "record",
                "fields": [
                    {"name": "z", "value": {
                        "kind": "tensor", "index": 0, "dtype": "float32",
                        "shape": [], "value_type": "real"
                    }},
                    {"name": "a", "value": {
                        "kind": "tensor", "index": 1, "dtype": "float32",
                        "shape": [2], "value_type": "real[2]"
                    }}
                ]
            }
        }])
    );
    assert_eq!(
        schema["output"],
        json!({
            "kind": "record",
            "fields": [
                {"name": "total", "value": {
                    "kind": "tensor", "index": 0, "dtype": "float32",
                    "shape": [], "value_type": "real"
                }},
                {"name": "original", "value": {
                    "kind": "tensor", "index": 1, "dtype": "float32",
                    "shape": [2], "value_type": "real[2]"
                }}
            ]
        })
    );
    assert_eq!(exported.output_names, [Some("result".into())]);
    assert!(!exported.multiple_outputs);
    let signature = exported.stablehlo.lines().nth(1).unwrap().trim();
    assert_eq!(
        signature,
        "func.func @main(%arg0: tensor<f32>, %arg1: tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>) {"
    );
    Ok(())
}

#[test]
fn batch_axes_preserve_cell_shapes_in_the_host_abi() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "p = elementof(cartprod(x = cartpow(reals, 3), scale = reals))\ninputs = p\noutputs = (p.scale * sum(p.x), p.x)",
        None,
        None,
    )?;
    let options = EmitOptions::default();
    let batch = BatchSpec {
        shape: vec![2, 4],
        input_axes: vec![vec![Some(1), Some(0)], vec![None, None]],
    };
    let exported = serde_json::to_value(query.compile_batched(&options, &batch)?)?;
    assert_eq!(exported["batch_shape"], json!([2, 4]));
    assert_eq!(
        exported["inputs"][0]["value"]["fields"][0]["value"]["shape"],
        json!([4, 2, 3])
    );
    assert_eq!(
        exported["inputs"][0]["value"]["fields"][1]["value"]["shape"],
        json!([])
    );
    assert_eq!(exported["output"]["items"][0]["shape"], json!([2, 4]));
    assert_eq!(exported["output"]["items"][1]["shape"], json!([2, 4, 3]));
    let identity = query.compile_batched(
        &options,
        &BatchSpec {
            shape: vec![],
            input_axes: vec![vec![], vec![]],
        },
    )?;
    assert_eq!(identity.stablehlo, query.compile(&options)?.stablehlo);
    Ok(())
}

#[test]
fn batched_gathers_do_not_loop_over_rows() -> Result<(), Box<dyn std::error::Error>> {
    for selection in ["[1, 3, 5]", "indices"] {
        let query = Context::default().parse(
            &format!(
                "x = elementof(cartpow(reals, 5))\nindices = external(cartpow(integers, 3))\ninputs = (x, indices)\noutputs = sum(x[{selection}])"
            ),
            None,
            None,
        )?;
        for index_axis in [None, Some(0)] {
            let exported = query.compile_batched(
                &EmitOptions::default(),
                &BatchSpec {
                    shape: vec![7],
                    input_axes: vec![vec![Some(0)], vec![index_axis]],
                },
            )?;
            assert!(
                !exported.stablehlo.contains("stablehlo.while"),
                "{}",
                exported.stablehlo
            );
        }
    }
    Ok(())
}

#[test]
fn primitive_batching_composes_without_function_name_checks()
-> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "rows = external(cartpow(cartpow(reals, 4), 3))\nscore(x) = atan(x) + loggamma(1.0 + x*x)\ninputs = rows\noutputs = score(sum(sum.(rows)))",
        None,
        None,
    )?;
    let exported = query.compile_batched(
        &EmitOptions::default(),
        &BatchSpec {
            shape: vec![2, 5],
            input_axes: vec![vec![Some(0), Some(2)]],
        },
    )?;
    assert!(
        !exported.stablehlo.contains("stablehlo.while"),
        "{}",
        exported.stablehlo
    );
    Ok(())
}

#[test]
fn batched_scan_keeps_one_time_loop() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "alpha = 0.5\nxs = external(cartpow(reals, 8))\nupdate(s, x) = tanh(alpha*s+x)\nstates = scan(update, 0.2, xs)\ninputs = (alpha, xs)\noutputs = sum(states)",
        None,
        None,
    )?;
    for axes in [
        vec![Some(0), None],
        vec![None, Some(1)],
        vec![Some(0), Some(0)],
    ] {
        let exported = query.compile_batched(
            &EmitOptions::default(),
            &BatchSpec {
                shape: vec![3],
                input_axes: axes.into_iter().map(|axis| vec![axis]).collect(),
            },
        )?;
        assert_eq!(
            exported.stablehlo.matches("stablehlo.while").count(),
            1,
            "{}",
            exported.stablehlo
        );
        assert!(!exported.stablehlo.contains("main_cell"));
        assert!(exported.stablehlo.contains("tensor<3x8xf32>"));
    }
    Ok(())
}
