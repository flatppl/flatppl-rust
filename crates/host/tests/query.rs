use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn imported_observations_score_a_composed_broadcast_kernel()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse(
        r#"
        y_data = [1.0, 2.0, 4.0]
        mu ~ Normal(0.0, 2.0)
        y ~ locscale.(StudentT(5.0), fill(mu, 3), [0.3, 0.6, 0.9])
        K_model = kernelof(record(y = y), mu = mu)
        "#,
        None,
        None,
    )?;
    context.register("model.flatppl", &model)?;
    for observations in [
        "m.y_data",
        "record(data = record(y = m.y_data)).data.y",
        "[1.0, 2.0, 4.0]",
    ] {
        let query = context.parse(
            &format!(
                "m = load_module(\"model.flatppl\")\np = elementof(reals)\ninputs = p\n\
                 outputs = logdensityof(m.K_model(mu = p), record(y = {observations}))"
            ),
            None,
            None,
        )?;
        query.compile(&EmitOptions::default())?;
    }
    Ok(())
}

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

#[test]
fn batched_fixed_pyhf_yields_keep_tensor_batching() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\na = elementof(reals)\ninputs = a\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([[1.0, 2.0], [3.0, 4.0]]), \
         fill(0.0, [2, 1, 2]), array(cat(fill(a, 2), fill(a, 2)), [2, 1, 2], [1, 2, 3])))",
        None,
        None,
    )?;
    let exported = query.compile_batched(
        &EmitOptions {
            target: flatppl_host::Target::Gpu,
            ..EmitOptions::default()
        },
        &BatchSpec {
            shape: vec![4],
            input_axes: vec![vec![Some(0)]],
        },
    )?;
    // An untyped barrier recipe previously serialized this batch.
    assert!(!exported.stablehlo.contains("stablehlo.while"));
    Ok(())
}

#[test]
fn batched_pyhf_nominals_keep_runtime_overrides() -> Result<(), Box<dyn std::error::Error>> {
    for (binding, rows) in [
        ("x = 1.0", "[[x, 2.0], [3.0, 4.0]]"),
        ("x = [1.0, 2.0]", "[x, [3.0, 4.0]]"),
    ] {
        let query = Context::default().parse(
            &format!(
                "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\n{binding}\ninputs = x\n\
                 outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack({rows}), \
                 fill(0.0, [2, 1, 2]), fill(1.0, [2, 1, 2])))"
            ),
            None,
            None,
        )?;
        query.compile_batched(
            &EmitOptions {
                target: flatppl_host::Target::Gpu,
                ..EmitOptions::default()
            },
            &BatchSpec {
                shape: vec![4],
                input_axes: vec![vec![Some(0)]],
            },
        )?;
    }
    Ok(())
}

#[test]
fn batched_pyhf_classes_keep_mixed_nominal_kinds() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\na = elementof(reals)\ninputs = a\n\
         shifts = fill(0.0, [1, 1, 2])\n\
         factors = array(cat(fill(a, 2)), [1, 1, 2], [1, 2, 3])\n\
         y1 = pyhf.sample_yields(rowstack([[1, 2]]), shifts, factors)\n\
         y2 = pyhf.sample_yields(rowstack([[3.0, 4.0]]), shifts, factors)\n\
         outputs = pyhf.expected_counts(rowstack([get(y1, 1, all), get(y2, 1, all)]))",
        None,
        None,
    )?;
    query.compile_batched(
        &EmitOptions {
            target: flatppl_host::Target::Gpu,
            ..EmitOptions::default()
        },
        &BatchSpec {
            shape: vec![4],
            input_axes: vec![vec![Some(0)]],
        },
    )?;
    Ok(())
}

#[test]
fn gpu_pyhf_channels_share_lanes_without_serializing_batches()
-> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        pyhf = standard_module("pyhf-helpers", "0.1")
        x = elementof(reals)
        inputs = x
        a = pyhf.expected_counts(pyhf.sample_yields(
            rowstack([[1.0, 2.0], [3.0, 4.0]]), fill(0.0, [2, 1, 2]),
            array(cat(fill(x, 2), fill(x, 2)), [2, 1, 2], [1, 2, 3])))
        b = pyhf.expected_counts(pyhf.sample_yields(
            rowstack([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), fill(0.0, [2, 1, 3]),
            array(cat(fill(x, 3), fill(x, 3)), [2, 1, 3], [1, 2, 3])))
        c = pyhf.expected_counts(pyhf.sample_yields(
            rowstack([[2.0, 1.0], [4.0, 3.0]]), fill(0.0, [2, 1, 2]),
            array(cat(fill(2.0 * x, 2), fill(x, 2)), [2, 1, 2], [1, 2, 3])))
        outputs = cat(a, b, c, a)
        "#,
        None,
        None,
    )?;
    use flatppl_host::{Dtype, Target};
    for (target, dtype) in [
        (Target::Gpu, Dtype::F64),
        (Target::Gpu, Dtype::F32),
        (Target::Cpu, Dtype::F64),
    ] {
        let options = EmitOptions {
            target,
            dtype,
            ..EmitOptions::default()
        };
        for batch in [None, Some(4)] {
            let exported = match batch {
                None => query.compile(&options)?,
                Some(size) => query.compile_batched(
                    &options,
                    &BatchSpec {
                        shape: vec![size],
                        input_axes: vec![vec![Some(0)]],
                    },
                )?,
            };
            // Unequal channel widths share fourteen sample lanes. Reusing a channel
            // still returns its bins twice, and the host batch stays tensorized.
            let elem = if matches!(dtype, Dtype::F64) {
                "f64"
            } else {
                "f32"
            };
            let lanes = if batch.is_some() {
                format!("tensor<4x14x{elem}>")
            } else {
                format!("tensor<14x{elem}>")
            };
            assert_eq!(
                exported
                    .stablehlo
                    .lines()
                    .any(|line| line.contains("stablehlo.optimization_barrier")
                        && line.contains(&lanes)),
                target == Target::Gpu && matches!(dtype, Dtype::F64)
            );
            assert!(!exported.stablehlo.contains("stablehlo.while"));
            let schema = serde_json::to_value(&exported)?;
            assert_eq!(
                schema["output"]["shape"],
                if batch.is_some() {
                    json!([4, 9])
                } else {
                    json!([9])
                }
            );
        }
    }
    Ok(())
}

#[test]
fn gpu_pyhf_pool_stays_out_of_callable_scopes() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        pyhf = standard_module("pyhf-helpers", "0.1")
        theta = elementof(reals)
        f(x) = cat(
            pyhf.expected_counts(pyhf.sample_yields(
                rowstack([[1.0, 2.0], [3.0, 4.0]]), fill(0.0, [2, 1, 2]),
                array(cat(fill(x, 2), fill(x, 2)), [2, 1, 2], [1, 2, 3]))),
            pyhf.expected_counts(pyhf.sample_yields(
                rowstack([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), fill(0.0, [2, 1, 3]),
                array(cat(fill(x, 3), fill(x, 3)), [2, 1, 3], [1, 2, 3]))),
            pyhf.expected_counts(pyhf.sample_yields(
                rowstack([[2.0, 1.0], [4.0, 3.0]]), fill(0.0, [2, 1, 2]),
                array(cat(fill(2.0 * x, 2), fill(x, 2)), [2, 1, 2], [1, 2, 3]))))
        inputs = theta
        outputs = f.([theta, 2.0 * theta])
        "#,
        None,
        None,
    )?;
    let exported = query.compile(&EmitOptions {
        target: flatppl_host::Target::Gpu,
        dtype: flatppl_host::Dtype::F64,
        ..EmitOptions::default()
    })?;
    // A broadcast binds x within its own frame. A module-level pool cannot
    // capture that temporary binding.
    assert!(
        !exported
            .stablehlo
            .contains("stablehlo.optimization_barrier")
    );
    assert_eq!(
        serde_json::to_value(&exported)?["output"]["shape"],
        json!([2, 7])
    );
    Ok(())
}

#[test]
fn gpu_pyhf_pool_respects_aggregate_axes() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        pyhf = standard_module("pyhf-helpers", "0.1")
        theta = elementof(cartpow(reals, 2))
        a = pyhf.expected_counts(pyhf.sample_yields(
            rowstack([[1.0, 2.0], [3.0, 4.0]]), fill(0.0, [2, 1, 2]),
            array(cat(fill(theta[1], 2), fill(theta[1], 2)), [2, 1, 2], [1, 2, 3])))
        b = aggregate(sum, [], sum(pyhf.expected_counts(pyhf.sample_yields(
            rowstack([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), fill(0.0, [2, 1, 3]),
            array(cat(fill(theta[1], 3), fill(theta[1], 3)), [2, 1, 3], [1, 2, 3])))) * theta[.i])
        c = pyhf.expected_counts(pyhf.sample_yields(
            rowstack([[2.0, 1.0], [4.0, 3.0]]), fill(0.0, [2, 1, 2]),
            array(cat(fill(2.0 * theta[1], 2), fill(theta[1], 2)), [2, 1, 2], [1, 2, 3])))
        inputs = theta
        outputs = cat(a, c, [b])
        "#,
        None,
        None,
    )?;
    let exported = query.compile(&EmitOptions {
        target: flatppl_host::Target::Gpu,
        dtype: flatppl_host::Dtype::F64,
        ..EmitOptions::default()
    })?;
    // Aggregate bodies stay in their axis scope, even when one retained
    // helper only captures a module input.
    assert!(
        !exported
            .stablehlo
            .contains("stablehlo.optimization_barrier")
    );
    assert_eq!(
        serde_json::to_value(&exported)?["output"]["shape"],
        json!([5])
    );
    Ok(())
}
