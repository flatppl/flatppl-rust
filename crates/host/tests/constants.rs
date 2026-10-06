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
