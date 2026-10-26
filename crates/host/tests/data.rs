use std::path::PathBuf;

use flatppl_host::{Context, EmitOptions};

fn directory(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("flatppl-host-data-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn imported_json_uses_its_source_path_and_declared_fields_and_prefixes()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("relative");
    std::fs::create_dir_all(dir.join("model"))?;
    std::fs::write(
        dir.join("model/data.json"),
        r#"{"index":[2,1,9],"x":[1.5,3.5,99],"unused":"ignored"}"#,
    )?;
    let source = "m = load_module(\"model/main.flatppl\")\nx = elementof(reals)\ninputs = x\noutputs = (m.data.x .* x, m.data.index)";
    for domain in [
        "cartprod(x = cartpow(reals, 2), index = cartpow(posintegers, 2))",
        "cartpow(cartprod(x = reals, index = posintegers), 2)",
    ] {
        std::fs::write(
            dir.join("model/main.flatppl"),
            format!("data = load_data(\"data.json\", {domain})"),
        )?;
        let mut context = Context::default();
        let model = context.parse(source, None, dir.join("query.flatppl").to_str())?;
        let literal = context.parse(
            "x = elementof(reals)\ninputs = x\noutputs = ([1.5, 3.5] .* x, [2, 1])",
            None,
            None,
        )?;
        assert_eq!(
            model.compile(&EmitOptions::default())?.stablehlo,
            literal.compile(&EmitOptions::default())?.stablehlo
        );
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn fixed_data_is_a_context_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("snapshot");
    let file = dir.join("data.json");
    std::fs::write(&file, "[2.5]")?;
    let source = "data = load_data(\"data.json\", cartpow(reals, 1))\noutputs = data";
    let path = dir.join("query.flatppl");
    let mut context = Context::default();
    let first = context.parse(source, None, path.to_str())?;
    let before = first.compile(&EmitOptions::default())?.stablehlo;
    std::fs::write(&file, "[4.5]")?;
    let same_context = context.parse(source, None, path.to_str())?;
    assert_eq!(
        same_context.compile(&EmitOptions::default())?.stablehlo,
        before
    );
    let fresh = Context::default().parse(source, None, path.to_str())?;
    let expected = context.parse("outputs = [4.5]", None, None)?;
    assert_eq!(
        fresh.compile(&EmitOptions::default())?.stablehlo,
        expected.compile(&EmitOptions::default())?.stablehlo
    );
    std::fs::remove_file(file)?;
    assert_eq!(first.compile(&EmitOptions::default())?.stablehlo, before);
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn json_selection_checks_required_content_and_domains() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("validation");
    let path = dir.join("query.flatppl");
    let source =
        "data = load_data(\"data.json\", cartprod(x = cartpow(posintegers, 2)))\noutputs = data.x";
    for json in [
        r#"{"x":[1,2,0]}"#,
        r#"{"other":[1,2]}"#,
        r#"{"x":[1]}"#,
        r#"{"x":[1,0]}"#,
    ] {
        std::fs::write(dir.join("data.json"), json)?;
        let query = Context::default().parse(source, None, path.to_str())?;
        let result = query.compile(&EmitOptions::default());
        assert_eq!(result.is_ok(), json == r#"{"x":[1,2,0]}"#);
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn missing_runtime_and_unused_sources_are_not_read() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        "data = load_data(\"missing.csv\", cartpow(reals, 2))\ndead = load_data(\"missing.json\", anything)\ninputs = data\noutputs = sum(data)",
        None, None,
    )?;
    let expected = context.parse(
        "data = external(cartpow(reals, 2))\ninputs = data\noutputs = sum(data)",
        None,
        None,
    )?;
    assert_eq!(
        query.compile(&EmitOptions::default())?.stablehlo,
        expected.compile(&EmitOptions::default())?.stablehlo
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn cached_data_cannot_escape_a_bundle_through_a_symlink() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = directory("bundle");
    let bundle = dir.join("bundle");
    std::fs::create_dir_all(&bundle)?;
    std::fs::write(dir.join("outside.json"), "[1.0]")?;
    std::os::unix::fs::symlink(dir.join("outside.json"), bundle.join("data.json"))?;
    let model = bundle.join("main.flatppl");
    std::fs::write(
        &model,
        "outputs = load_data(\"data.json\", cartpow(reals, 1))",
    )?;
    let mut context = Context::default();
    let standalone = context
        .load(model.to_str().unwrap())?
        .compile(&EmitOptions::default())?;
    let restricted = context.load(bundle.to_str().unwrap())?;
    assert!(restricted.compile(&EmitOptions::default()).is_err());
    std::fs::remove_file(bundle.join("data.json"))?;
    std::fs::write(bundle.join("data.json"), "[1.0]")?;
    let contained = Context::default().load(bundle.to_str().unwrap())?;
    assert_eq!(
        contained.compile(&EmitOptions::default())?.stablehlo,
        standalone.stablehlo
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

// ---- §07 formats ----------------------------------------------------------
//
// `fixtures/data/` holds one table in every format §07 requires (regenerate with
// `generate.py`): Arrow IPC from polars, Safetensors packed from the format
// definition. Each test compiles a selection and compares it with the same values
// written as literals.

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/data")
}

fn compile_at(source: &str) -> Result<String, flatppl_host::Diagnostic> {
    let path = fixtures().join("query.flatppl");
    let mut context = Context::default();
    let module = context.parse(source, None, path.to_str())?;
    module.compile(&EmitOptions::default()).map(|e| e.stablehlo)
}

/// The returned constants of an emitted module, in return order, as numbers.
/// Fixed data emits each value as one `stablehlo.constant`, hex-packed or decimal.
fn returned(stablehlo: &str) -> Vec<Vec<f64>> {
    let values: std::collections::HashMap<&str, Vec<f64>> = stablehlo
        .lines()
        .filter_map(|line| {
            let (ssa, rhs) = line.trim().split_once(" = stablehlo.constant dense<")?;
            let (body, ty) = rhs.rsplit_once("> : ")?;
            let numbers = if let Some(hex) = body.strip_prefix("\"0x") {
                let bytes: Vec<u8> = (0..hex.len() - 1)
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect();
                assert!(ty.ends_with("f32>"), "{ty}");
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b) as f64)
                    .collect()
            } else {
                body.trim_matches(['[', ']'])
                    .split(',')
                    .map(|x| match x.trim() {
                        "true" => 1.0,
                        "false" => 0.0,
                        x => x.parse().unwrap(),
                    })
                    .collect()
            };
            Some((ssa, numbers))
        })
        .collect();
    let ret = stablehlo
        .lines()
        .find_map(|line| line.trim().strip_prefix("return "))
        .expect("a return");
    // A matrix may arrive through a reshape and an identity transpose of its
    // row-major constant; follow those back to it.
    let source = |ssa: &str| {
        stablehlo.lines().find_map(|line| {
            let (lhs, rhs) = line.trim().split_once(" = ")?;
            if lhs != ssa {
                return None;
            }
            let operand = rhs.strip_prefix("stablehlo.reshape ").or_else(|| {
                rhs.strip_prefix("stablehlo.transpose ")
                    .filter(|r| r.contains("dims = [0, 1]"))
            })?;
            operand.split([' ', ',']).next()
        })
    };
    ret.split(" : ")
        .next()
        .unwrap()
        .split(", ")
        .map(|ssa| {
            let mut ssa = ssa;
            while !values.contains_key(ssa) {
                ssa = source(ssa).unwrap_or_else(|| panic!("`{ssa}` is not a constant"));
            }
            values[ssa].clone()
        })
        .collect()
}

fn refusal(source: &str) -> String {
    compile_at(source).expect_err("must refuse").to_string()
}

#[test]
fn every_section_07_format_reads_the_same_table() {
    let want = vec![vec![1.5, -2.0], vec![2.0, 0.0], vec![1.0, 0.0]];
    for file in [
        "events.json",
        "events_columns.json",
        "events.csv",
        "events.wsv",
        "events.arrow",
        "events.arrows",
    ] {
        let source = format!(
            "data = load_data(\"{file}\", cartpow(cartprod(a = reals, b = integers, flag = booleans), 2))\n\
             outputs = (data.a, data.b, data.flag)"
        );
        let out = compile_at(&source).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(
            out.contains("-> (tensor<2xf32>, tensor<2xi32>, tensor<2xi1>)"),
            "{file}: {out}"
        );
        assert_eq!(returned(&out), want, "{file}");
    }
}

#[test]
fn a_one_column_delimited_source_reads_as_a_vector() {
    let out =
        compile_at("data = load_data(\"column.csv\", cartpow(reals, 3))\noutputs = data").unwrap();
    assert_eq!(returned(&out), vec![vec![0.5, -1.0, 2.75]]);
    let err = refusal("data = load_data(\"events.csv\", cartpow(reals, 3))\noutputs = data");
    assert!(
        err.contains("exactly one column, and this one has 4"),
        "{err}"
    );
}

#[test]
fn delimited_cells_are_checked_against_their_declared_set() {
    for (domain, reason) in [
        (
            "cartprod(a = integers)",
            "cell `1.5` is not a value of its declared set",
        ),
        ("cartprod(a = posreals)", "outside"),
        ("cartprod(flag = reals)", "cell `true`"),
        ("cartprod(missing = reals)", "missing column `missing`"),
    ] {
        let err = refusal(&format!(
            "data = load_data(\"events.csv\", cartpow({domain}, 3))\noutputs = data"
        ));
        assert!(err.contains(reason), "{domain}: {err}");
    }
    let err = refusal(
        "data = load_data(\"events.csv\", cartpow(cartprod(a = reals), 4))\noutputs = data.a",
    );
    assert!(err.contains("requested 4 entries but only 3"), "{err}");
}

#[test]
fn safetensors_keys_form_nested_records() {
    let source = "net = load_data(\"net.safetensors\", cartprod(\
        W = cartpow(reals, [2, 3]), b = cartpow(reals, 2), count = integers, \
        enc = cartprod(layer = cartprod(weight = cartpow(integers, 2), \
        mask = cartpow(booleans, 2), half = cartpow(reals, 2)))))\n\
        outputs = (net.W, net.b, net.count, net.enc.layer.weight, net.enc.layer.mask, net.enc.layer.half)";
    let out = compile_at(source).unwrap();
    assert!(
        out.contains(
            "-> (tensor<2x3xf32>, tensor<2xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi1>, \
             tensor<2xf32>)"
        ),
        "{out}"
    );
    assert_eq!(
        returned(&out),
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.5],
            vec![0.25, -0.5],
            vec![5.0],
            vec![3.0, -4.0],
            vec![1.0, 0.0],
            vec![1.5, -0.125],
        ]
    );
}

#[test]
fn safetensors_dtypes_and_shapes_are_checked_against_the_value_set() {
    for (domain, reason) in [
        (
            "cartprod(W = cartpow(integers, [2, 3]))",
            "a real value where the declared set holds integers",
        ),
        (
            "cartprod(enc = cartprod(layer = cartprod(weight = cartpow(booleans, 2))))",
            "an integer value where the declared set holds booleans",
        ),
        (
            "cartprod(W = cartpow(reals, [3, 2]))",
            "tensor shape [2, 3] does not match",
        ),
        (
            "cartprod(W = cartpow(reals, [2, 2]))",
            "tensor shape [2, 3] does not match",
        ),
        (
            "cartprod(b = cartpow(reals, 4))",
            "tensor shape [3] does not match",
        ),
        (
            "cartprod(enc = cartprod(other = reals))",
            "missing field `other`",
        ),
    ] {
        let err = refusal(&format!(
            "net = load_data(\"net.safetensors\", {domain})\noutputs = net"
        ));
        assert!(err.contains(reason), "{domain}: {err}");
    }
}

#[test]
fn safetensors_key_rules_refuse_the_file() {
    for (file, reason) in [
        (
            "empty_segment",
            "key `enc..weight` has an empty path segment",
        ),
        ("empty_key", "key `...` is empty after trimming its dots"),
        (
            "duplicate_path",
            "keys trimming to `enc.weight` name the same record path",
        ),
        (
            "prefix_and_leaf",
            "`enc` is both a tensor and a record path prefix",
        ),
        ("fp8", "dtype F8_E4M3 has no FlatPPL value set"),
    ] {
        let err = refusal(&format!(
            "net = load_data(\"{file}.safetensors\", cartprod(x = reals))\noutputs = net"
        ));
        assert!(err.contains(reason), "{file}: {err}");
    }
}

#[test]
fn an_unknown_extension_refuses_at_the_call() {
    let err = compile_at(
        "x = 1.0\ndata = load_data(\"events.parquet\", cartpow(reals, 2))\noutputs = data",
    )
    .expect_err("must refuse");
    let text = err.to_string();
    assert!(
        text.contains("unsupported file extension `.parquet`"),
        "{text}"
    );
    assert_eq!(err.position.as_ref().map(|p| p.line), Some(2), "{err:?}");
}

#[test]
fn malformed_tables_refuse_with_the_reason() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("malformed");
    let path = dir.join("query.flatppl");
    for (file, text, reason) in [
        ("ragged.csv", "a,b\n1,2\n3\n", "CSV"),
        (
            "ragged.wsv",
            "a b\n1 2\n3\n",
            "row 2 has 1 fields, but the first row names 2 columns",
        ),
        ("empty.csv", "a,b\n1,\n3,4\n", "a value is missing"),
        (
            "names.csv",
            "a,a\n1,2\n",
            "column names in the first row must be nonempty and distinct",
        ),
        (
            "rows.json",
            r#"[{"a": 1, "b": 2}, {"a": 3}]"#,
            "row 2 is missing field `b`",
        ),
    ] {
        std::fs::write(dir.join(file), text)?;
        let source = format!(
            "data = load_data(\"{file}\", cartpow(cartprod(a = reals, b = reals), 2))\noutputs = data.b"
        );
        let err = Context::default()
            .parse(&source, None, path.to_str())?
            .compile(&EmitOptions::default())
            .err()
            .expect("must refuse")
            .to_string();
        assert!(err.contains(reason), "{file}: {err}");
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}
