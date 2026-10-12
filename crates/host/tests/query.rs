use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn bound_columns_export_their_matrix_shape() -> Result<(), Box<dyn std::error::Error>> {
    use flatppl_host::Constant;
    let mut context = Context::default();
    let model = context.parse(
        "N = external(posintegers)\nx = external(cartpow(reals, N))\nX = colstack([x .- 2.0, x .* 2.0])",
        None, None,
    )?;
    let model = context.set(
        &model,
        &[
            ("N".into(), Constant::Integer(3)),
            (
                "x".into(),
                Constant::Array {
                    shape: vec![3],
                    data: vec![
                        Constant::Real(1.0),
                        Constant::Real(2.0),
                        Constant::Real(3.0),
                    ],
                },
            ),
        ],
    )?;
    context.register("columns.flatppl", &model)?;
    let query = context.parse(
        "m = load_module(\"columns.flatppl\")\noutputs = m.X",
        None,
        None,
    )?;
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(&exported)?;
    assert_eq!(schema["output"]["shape"], json!([3, 2]));
    assert_eq!(schema["output"]["value_type"], "real[3, 2]");
    Ok(())
}

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
fn batched_matrix_products_do_not_loop_over_points() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "a = external(cartpow(reals, [2, 3]))\nb = external(cartpow(reals, [3, 4]))\nx = external(cartpow(reals, 4))\nr = (a*b)*x\ninputs = (a, b, x)\noutputs = (r, transpose(r)*r)",
        None,
        None,
    )?;
    for shape in [vec![5, 7], vec![5, 0]] {
        let exported = query.compile_batched(
            &EmitOptions::default(),
            &BatchSpec {
                shape,
                input_axes: vec![vec![Some(1), None], vec![None, Some(2)], vec![None, None]],
            },
        )?;
        assert!(
            !exported.stablehlo.contains("stablehlo.while"),
            "{}",
            exported.stablehlo
        );
    }
    Ok(())
}

#[test]
fn constant_products_keep_native_tensor_multiplication() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "x = elementof(cartpow(reals, 3))\ny = [1.0, 2.0, 3.0]\nc = [2.0, 0.0, -3.0]\ninputs = (x, y)\noutputs = (c .* x, y .* c, x .* y)",
        None,
        None,
    )?;
    let ir = query.compile(&EmitOptions::default())?.stablehlo;
    assert_eq!(ir.matches("stablehlo.multiply").count(), 2, "{ir}");
    assert_eq!(ir.matches("stablehlo.dot_general").count(), 1, "{ir}");
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
fn batched_boolean_guards_do_not_loop_over_rows() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "x = elementof(reals)\ninputs = x\nguard = lor(land(x >= 0.0, x <= 1.0), lxor(x < -1.0, lnot(x < 2.0)))\noutputs = ifelse(guard, x*x, -x)",
        None,
        None,
    )?;
    let exported = query.compile_batched(
        &EmitOptions::default(),
        &BatchSpec {
            shape: vec![2, 4],
            input_axes: vec![vec![Some(1), Some(0)]],
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

#[test]
fn batched_scan_density_keeps_one_time_loop() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        alpha ~ Uniform(interval(-2.0, 2.0))
        update(s, x) = record(mean = s.mean + alpha*x)
        states = scan(update, record(mean = 0.0), [1.0, 2.0])
        read_mean(s) = s.mean
        probabilities = invlogit.(read_mean.(states))
        choice ~ Bernoulli.(probabilities)
        joint = lawof(record(alpha = alpha, choice = choice))
        theta = 0.5
        inputs = theta
        outputs = logdensityof(joint, record(alpha = theta, choice = [1, 0]))
        "#,
        None,
        None,
    )?;
    let exported = query.compile_batched(
        &EmitOptions::default(),
        &BatchSpec {
            shape: vec![4],
            input_axes: vec![vec![Some(0)]],
        },
    )?;
    assert_eq!(
        exported.stablehlo.matches("stablehlo.while").count(),
        1,
        "{}",
        exported.stablehlo
    );
    Ok(())
}

#[test]
fn gpu_segment_slices_stay_out_of_batched_exports() -> Result<(), Box<dyn std::error::Error>> {
    // Contiguous runs within the padding bound share one reduction.
    let rows =
        |range: std::ops::Range<usize>| range.map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
    let query = Context::default().parse(
        &format!(
            "x = elementof(cartpow(reals, 40))\ninputs = x\noutputs = (sum(x[[{}]]), sum(x[[{}]]), sum(x[[{}]]), sum(x[[{}]]))",
            rows(1..3),
            rows(3..6),
            rows(6..7),
            rows(7..9)
        ),
        None,
        None,
    )?;
    let options = EmitOptions {
        target: flatppl_host::Target::Gpu,
        ..EmitOptions::default()
    };
    let gather = "\"stablehlo.gather\"";
    // The gradient differentiates the unbatched export, whose slices have a
    // sliced adjoint. The batched forward graph keeps the gather.
    assert!(!query.compile(&options)?.stablehlo.contains(gather));
    let batch = BatchSpec {
        shape: vec![4],
        input_axes: vec![vec![Some(0)]],
    };
    assert!(
        query
            .compile_batched(&options, &batch)?
            .stablehlo
            .contains(gather)
    );
    Ok(())
}

#[test]
fn gpu_scalar_column_stays_out_of_batched_exports() -> Result<(), Box<dyn std::error::Error>> {
    // Two samples with two scalar factors each take the scalar column on GPU.
    let query = Context::default().parse(
        "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\n\
         theta = elementof(cartpow(reals, 4))\ninputs = theta\n\
         a = theta[1]\ng = theta[[2, 3]]\ns = theta[4]\n\
         factors = array(cat(fill(a, 2), fill(s, 2), g, fill(a, 2), fill(s, 2), fill(1.0, 2)), [2, 3, 2], [1, 2, 3])\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([[1.0, 2.0], [3.0, 4.0]]), \
         fill(0.0, [2, 1, 2]), factors))",
        None,
        None,
    )?;
    let options = EmitOptions {
        target: flatppl_host::Target::Gpu,
        ..EmitOptions::default()
    };
    let column = "2x2x1xf32>";
    // The gradient differentiates the unbatched export, where the column
    // contracts the bins once per channel. In a batched forward graph XLA
    // builds the folded rows' expected counts in one kernel.
    assert!(query.compile(&options)?.stablehlo.contains(column));
    let batch = BatchSpec {
        shape: vec![4],
        input_axes: vec![vec![Some(0)]],
    };
    let batched = query.compile_batched(&options, &batch)?.stablehlo;
    assert!(!batched.contains(column), "{batched}");
    Ok(())
}
