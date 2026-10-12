use flatppl_host::{Constant, Context, EmitOptions};

#[test]
fn simultaneous_sizes_and_nested_table_constants_keep_column_values()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let module = context.parse(
        "n = external(posintegers)\ndata = external(cartpow(cartprod(a = reals, nested = cartprod(b = reals)), n))\ninputs = (n, data)\noutputs = sum(data.a) + sum(data.nested.b)", None, None,
    )?;
    let column = |x| Constant::Array {
        shape: vec![1],
        data: vec![Constant::Integer(x)],
    };
    let bound = context.set(
        &module,
        &[
            (
                "data".into(),
                Constant::Record(vec![
                    (
                        "nested".into(),
                        Constant::Table(vec![("b".into(), column(4))]),
                    ),
                    ("a".into(), column(3)),
                ]),
            ),
            ("n".into(), Constant::Bool(true)),
        ],
    )?;
    let expected = context.parse("outputs = sum([3.0]) + sum([4.0])", None, None)?;
    assert_eq!(
        bound.compile(&EmitOptions::default())?.stablehlo,
        expected.compile(&EmitOptions::default())?.stablehlo
    );
    Ok(())
}

#[test]
fn bound_sizes_match_literal_sources_without_changing_the_original()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let source = "n = external(posintegers)\ndata = external(cartpow(reals, n))\ninputs = data\noutputs = sum(data) / lengthof(data)";
    let query = context.parse(source, None, None)?;
    for n in [3, 5] {
        let bound = context.set(&query, &[("n".into(), Constant::Integer(n))])?;
        let expected = context.parse(
            &source.replace("external(posintegers)", &n.to_string()),
            None,
            None,
        )?;
        assert_eq!(
            bound.compile(&EmitOptions::default())?.stablehlo,
            expected.compile(&EmitOptions::default())?.stablehlo
        );
    }
    Ok(())
}

#[test]
fn binding_inputs_matches_explicit_partial_application() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let source = "n = external(posintegers)\nx = external(reals)\ny = external(reals)\ninputs = (n, x, y)\noutputs = n * (x + y)";
    let original = context.parse(source, None, None)?;
    let before = original.compile(&EmitOptions::default())?.stablehlo;
    let mut bound = original.clone();
    let mut expected = source.to_owned();
    for (name, value, declaration, literal, old_inputs, new_inputs) in [
        (
            "n",
            Constant::Integer(3),
            "n = external(posintegers)",
            "n = 3",
            "inputs = (n, x, y)",
            "inputs = (x, y)",
        ),
        (
            "y",
            Constant::Integer(4),
            "y = external(reals)",
            "y = 4.0",
            "inputs = (x, y)",
            "inputs = x",
        ),
        (
            "x",
            Constant::Real(2.0),
            "x = external(reals)",
            "x = 2.0",
            "inputs = x\n",
            "",
        ),
    ] {
        bound = context.set(&bound, &[(name.into(), value)])?;
        expected = expected
            .replace(declaration, literal)
            .replace(old_inputs, new_inputs);
        let literal = context.parse(&expected, None, None)?;
        assert_eq!(
            bound.compile(&EmitOptions::default())?.stablehlo,
            literal.compile(&EmitOptions::default())?.stablehlo
        );
    }
    assert_eq!(original.compile(&EmitOptions::default())?.stablehlo, before);
    Ok(())
}

#[test]
fn fixed_values_obey_declared_domains_and_preserve_real_embedding()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    for (domain, valid, invalid, literal) in [
        (
            "posintegers",
            Constant::Integer(3),
            Constant::Real(0.5),
            "3",
        ),
        (
            "unitinterval",
            Constant::Real(1.0),
            Constant::Real(10.0),
            "1.0",
        ),
        (
            "booleans",
            Constant::Bool(true),
            Constant::Real(0.5),
            "true",
        ),
        (
            "cartpow(reals, 2)",
            Constant::Array {
                shape: vec![2],
                data: vec![Constant::Integer(1), Constant::Integer(2)],
            },
            Constant::Array {
                shape: vec![3],
                data: vec![Constant::Real(1.0); 3],
            },
            "[1.0, 2.0]",
        ),
        (
            "cartprod(reals, posintegers)",
            Constant::Array {
                shape: vec![2],
                data: vec![Constant::Real(0.5), Constant::Integer(2)],
            },
            Constant::Array {
                shape: vec![2],
                data: vec![Constant::Real(0.5), Constant::Real(2.5)],
            },
            "[0.5, 2.0]",
        ),
        (
            "ordered(3)",
            Constant::Array {
                shape: vec![3],
                data: vec![
                    Constant::Real(0.5),
                    Constant::Integer(1),
                    Constant::Real(2.5),
                ],
            },
            // Strictly increasing: a tie is outside the set.
            Constant::Array {
                shape: vec![3],
                data: vec![
                    Constant::Real(0.0),
                    Constant::Real(0.0),
                    Constant::Real(1.0),
                ],
            },
            "[0.5, 1.0, 2.5]",
        ),
    ] {
        let source = format!("x = external({domain})\noutputs = x");
        let module = context.parse(&source, None, None)?;
        let bound = context.set(&module, &[("x".into(), valid)])?;
        let expected = context.parse(&format!("x = {literal}\noutputs = x"), None, None)?;
        assert_eq!(
            bound.compile(&EmitOptions::default())?.stablehlo,
            expected.compile(&EmitOptions::default())?.stablehlo
        );
        assert!(
            context.set(&module, &[("x".into(), invalid)]).is_err(),
            "{domain}"
        );
    }
    Ok(())
}

#[test]
fn fixed_vector_equalities_allow_roundoff_without_projection()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let vector = |data: [f64; 3]| Constant::Array {
        shape: vec![3],
        data: data.into_iter().map(Constant::Real).collect(),
    };
    for (set, values, outside) in [
        (
            "zerosum(3)",
            [0.1, 0.2, -0.3],
            [0.1, 0.2, -0.30000000000001],
        ),
        (
            "stdsimplex(3)",
            [0.1, 0.2, 0.7000000000000001],
            [0.1, 0.2, 0.70000000000001],
        ),
    ] {
        let module = context.parse(&format!("x = external({set})"), None, None)?;
        let valid = vector(values);
        let bound = context.set(&module, &[("x".into(), valid.clone())])?;
        assert_eq!(bound.constant("x")?, valid);
        for invalid in [outside, [f64::INFINITY, 0.0, f64::NEG_INFINITY]] {
            assert!(
                context
                    .set(&module, &[("x".into(), vector(invalid))])
                    .is_err()
            );
        }
    }
    let simplex = context.parse("x = external(stdsimplex(3))", None, None)?;
    for invalid in [[-f64::EPSILON, 0.5, 0.5], [0.0; 3]] {
        assert!(
            context
                .set(&simplex, &[("x".into(), vector(invalid))])
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn packed_f32_constants_use_their_source_precision() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let module = context.parse("x = external(zerosum(3))", None, None)?;
    let values = [0.1_f32, 0.2, -0.3].map(f64::from);
    let residual = values.iter().sum::<f64>().abs();
    let scale = 8.0 * 3.0 * values.iter().map(|v| v.abs()).sum::<f64>();
    assert!(residual <= scale * f64::from(f32::EPSILON) / 2.0);
    assert!(residual > scale * f64::EPSILON / 2.0);

    let packed = serde_json::json!({"kind": "array", "value": {
        "shape": [3], "dtype": "real32", "data": values
    }});
    let single: Constant = serde_json::from_str(&packed.to_string())?;
    assert_eq!(serde_json::to_value(&single)?, packed);
    let bound = context.set(&module, &[("x".into(), single)])?;
    let double = Constant::Array {
        shape: vec![3],
        data: values.map(Constant::Real).to_vec(),
    };
    assert_eq!(bound.constant("x")?, double);
    assert!(context.set(&module, &[("x".into(), double)]).is_err());
    let outside: Constant = serde_json::from_value(serde_json::json!({"kind": "array", "value": {
        "shape": [3], "dtype": "real32", "data": [0.1, 0.2, -0.3001]
    }}))?;
    assert!(context.set(&module, &[("x".into(), outside)]).is_err());
    Ok(())
}

#[test]
fn fixed_vector_equalities_handle_finite_overflow() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let module = context.parse("x = external(zerosum(4))", None, None)?;
    let vector = |values: [f64; 4]| Constant::Array {
        shape: vec![4],
        data: values.map(Constant::Real).to_vec(),
    };
    let valid = vector([f64::MAX, f64::MAX, -f64::MAX, -f64::MAX]);
    let bound = context.set(&module, &[("x".into(), valid.clone())])?;
    assert_eq!(bound.constant("x")?, valid);
    let zero = vector([0.0; 4]);
    assert_eq!(
        context
            .set(&module, &[("x".into(), zero.clone())])?
            .constant("x")?,
        zero
    );
    assert!(
        context
            .set(
                &module,
                &[("x".into(), vector([f64::MAX, f64::MAX, -f64::MAX, 0.0]))]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn an_empty_array_is_refused_with_the_source_rule() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let module = context.parse(
        "data = external(cartpow(reals, external(nonnegintegers)))\noutputs = sum(data)",
        None,
        None,
    )?;
    let empty = Constant::Array {
        shape: vec![0],
        data: vec![],
    };
    let Err(error) = context.set(&module, &[("data".into(), empty)]) else {
        panic!("an empty array was bound");
    };
    assert!(
        error.to_string().contains(
            "a constant array with a zero extent has no FlatPPL literal: \
             `array` sizes must be positive (spec §07)"
        ),
        "{error}"
    );
    Ok(())
}
