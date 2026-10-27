use std::path::PathBuf;

use flatppl_host::{Context, Dtype, EmitOptions};

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
        ("cartprod(a = integers)", "cell `1.5` is not an integer"),
        ("cartprod(a = posreals)", "outside"),
        (
            "cartprod(a = booleans)",
            "cell `1.5` is not `true` or `false`",
        ),
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
        (
            "ragged.csv",
            "a,b\n1,2\n3\n",
            "row 2 has 1 fields, but the first row names 2 columns",
        ),
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
        (
            "strings.json",
            r#"{"a": [1, 2], "b": ["1.5", "2"]}"#,
            "JSON string \"1.5\" is not a number or a boolean",
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

/// Untyped formats carry no integer/real distinction, so the value decides: an
/// exactly integral value is an integer, and a CSV `true`/`false` reads as 1/0.
#[test]
fn untyped_numbers_are_integers_when_exactly_integral() {
    let out = compile_at(
        "data = load_data(\"untyped.csv\", cartpow(cartprod(i = integers, flag = reals), 3))\n\
         outputs = (data.i, data.flag)",
    )
    .unwrap();
    assert_eq!(
        returned(&out),
        vec![vec![3.0, 3.0, 1000.0], vec![1.0, 0.0, 1.0]]
    );
    let out = compile_at(
        "data = load_data(\"untyped.json\", cartprod(i = cartpow(integers, 3)))\noutputs = data.i",
    )
    .unwrap();
    assert_eq!(returned(&out), vec![vec![3.0, 3.0, 1000.0]]);
    for file in ["untyped.csv", "untyped.json"] {
        let err = refusal(&format!(
            "data = load_data(\"{file}\", cartprod(x = cartpow(integers, 3)))\noutputs = data.x"
        ));
        assert!(
            err.contains("3.5` is not an integer") || err.contains("3.5 is not an integer"),
            "{file}: {err}"
        );
    }
}

/// §07 lets a value set omit columns, so a column no reader decodes refuses only
/// when it is selected. Arrow is typed: a missing value refuses, also in a
/// `Null`-typed column, and a string column is not a number.
#[test]
fn arrow_columns_refuse_only_when_selected() {
    let out = compile_at(
        "data = load_data(\"mixed.arrow\", cartpow(cartprod(a = reals), 2))\noutputs = data.a",
    )
    .unwrap();
    assert_eq!(returned(&out), vec![vec![1.5, 2.5]]);
    for (column, reason) in [
        ("n", "a value is missing"),
        ("z", "has the Arrow type Null, so every value is missing"),
        ("s", "column `s` holds Arrow strings"),
        ("t", "column `t` has unsupported Arrow type Timestamp"),
        ("l", "column `l` has unsupported Arrow type FixedSizeList"),
    ] {
        let err = refusal(&format!(
            "data = load_data(\"mixed.arrow\", cartpow(cartprod({column} = reals), 2))\n\
             outputs = data.{column}"
        ));
        assert!(err.contains(reason), "{column}: {err}");
    }
}

/// §07's first-`n` read covers vector and table entries only, so a
/// multi-dimensional array must match its declared shape on every axis.
#[test]
fn a_matrix_must_match_its_declared_shape() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("matrix");
    std::fs::write(dir.join("m.json"), "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]")?;
    let path = dir.join("query.flatppl");
    let compile = |shape: &str| {
        Context::default()
            .parse(
                &format!("m = load_data(\"m.json\", cartpow(reals, {shape}))\noutputs = m"),
                None,
                path.to_str(),
            )
            .unwrap()
            .compile(&EmitOptions::default())
    };
    let err = compile("[2, 2]").err().expect("must refuse").to_string();
    assert!(
        err.contains("an axis holds 3 entries, but the declared shape needs exactly 2"),
        "{err}"
    );
    assert_eq!(
        returned(&compile("[3, 3]").map_err(|e| e.to_string())?.stablehlo),
        vec![(1..=9).map(f64::from).collect::<Vec<_>>()]
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

/// Every dtype §07 maps, through §03 inclusion: float into reals, integer into
/// integers, BOOL into booleans.
#[test]
fn every_safetensors_dtype_reads_its_values() {
    let fields = [
        ("f64", "reals"),
        ("f32", "reals"),
        ("f16", "reals"),
        ("bf16", "reals"),
        ("i64", "integers"),
        ("i32", "integers"),
        ("i16", "integers"),
        ("i8", "integers"),
        ("u64", "integers"),
        ("u32", "integers"),
        ("u16", "integers"),
        ("u8", "integers"),
        ("bool", "booleans"),
    ];
    let domain = fields
        .iter()
        .map(|(name, set)| format!("{name} = cartpow({set}, 2)"))
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = fields
        .iter()
        .map(|(name, _)| format!("t.{name}"))
        .collect::<Vec<_>>()
        .join(", ");
    let out = compile_at(&format!(
        "t = load_data(\"dtypes.safetensors\", cartprod({domain}))\noutputs = ({outputs})"
    ))
    .unwrap();
    let want: Vec<Vec<f64>> = [
        [1.5, -2.0],
        [1.5, -2.0],
        [1.5, -2.0],
        [1.5, -3.0],
        [1.0, -2.0],
        [1.0, -2.0],
        [1.0, -2.0],
        [1.0, -2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 0.0],
    ]
    .iter()
    .map(|v| v.to_vec())
    .collect();
    assert_eq!(returned(&out), want);
    let err = refusal(
        "t = load_data(\"u64.safetensors\", cartprod(x = cartpow(integers, 1)))\noutputs = t.x",
    );
    assert!(
        err.contains("U64 value 9223372036854775808 exceeds int64"),
        "{err}"
    );
}

/// The typed columns after every other Arrow layout read their values across two
/// record batches, from the file, the stream and LZ4- and ZSTD-compressed files.
#[test]
fn arrow_typed_columns_read_past_every_layout() {
    for file in [
        "types.arrow",
        "types.arrows",
        "compressed.arrow",
        "zstd.arrow",
    ] {
        let columns = [
            ("i8", "integers"),
            ("i16", "integers"),
            ("i32", "integers"),
            ("i64", "integers"),
            ("u8", "integers"),
            ("u16", "integers"),
            ("u32", "integers"),
            ("u64", "integers"),
            ("f16", "reals"),
            ("f32", "reals"),
            ("bool", "booleans"),
            ("dict_f64", "reals"),
            ("dict_i16", "integers"),
        ];
        let domain = columns
            .iter()
            .map(|(name, set)| format!("{name} = {set}"))
            .collect::<Vec<_>>()
            .join(", ");
        let outputs = columns
            .iter()
            .map(|(name, _)| format!("t.{name}"))
            .collect::<Vec<_>>()
            .join(", ");
        let out = compile_at(&format!(
            "t = load_data(\"{file}\", cartpow(cartprod({domain}), 4))\noutputs = ({outputs})"
        ))
        .unwrap_or_else(|e| panic!("{file}: {e}"));
        let signed = vec![1.0, -2.0, 1.0, -2.0];
        let unsigned = vec![1.0, 2.0, 1.0, 2.0];
        let real = vec![1.5, -2.0, 1.5, -2.0];
        let mut want = vec![signed; 4];
        want.extend(vec![unsigned; 4]);
        want.extend(vec![real.clone(), real.clone(), vec![1.0, 0.0, 1.0, 0.0]]);
        want.extend(vec![real, vec![1.0, -2.0, 1.0, -2.0]]);
        assert_eq!(returned(&out), want, "{file}");
        for (column, reason) in [
            ("f64", "a value is missing"),
            ("utf8", "holds Arrow strings"),
            ("large_utf8", "holds Arrow strings"),
            ("view", "holds Arrow strings"),
            ("dict", "holds Arrow strings"),
            ("binary", "unsupported Arrow type"),
            ("list", "unsupported Arrow type"),
            ("large_list", "unsupported Arrow type"),
            ("list_view", "unsupported Arrow type"),
            ("fixed_list", "unsupported Arrow type"),
            ("struct", "unsupported Arrow type"),
            ("sparse", "unsupported Arrow type"),
            ("dense", "unsupported Arrow type"),
            ("ree", "unsupported Arrow type"),
            ("time", "unsupported Arrow type"),
            ("null", "has the Arrow type Null, so every value is missing"),
            ("dict_null", "a value is missing"),
        ] {
            let err = refusal(&format!(
                "t = load_data(\"{file}\", cartpow(cartprod({column} = reals), 4))\noutputs = t.{column}"
            ));
            assert!(err.contains(reason), "{file} {column}: {err}");
        }
    }
    let err =
        refusal("t = load_data(\"u64.arrow\", cartpow(cartprod(x = integers), 1))\noutputs = t.x");
    assert!(err.contains("outside int64"), "{err}");
    let out =
        compile_at("t = load_data(\"u64.arrow\", cartpow(cartprod(a = reals), 1))\noutputs = t.a")
            .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(returned(&out), vec![vec![1.5]]);
    let err = refusal(
        "t = load_data(\"bad_key.arrows\", cartpow(cartprod(x = reals), 2))\noutputs = t.x",
    );
    assert!(err.contains("Arrow IPC"), "{err}");
    let err = refusal(
        "t = load_data(\"negative_body.arrows\", cartpow(cartprod(a = reals), 3))\noutputs = t.a",
    );
    assert!(err.contains("Invalid IPC message body length"), "{err}");
    for (file, column, reason) in [
        (
            "negative_offset.arrows",
            "a",
            "malformed file (a buffer leaves its message body)",
        ),
        (
            "huge_rows.arrows",
            "n",
            "column `n` has the Arrow type Null, so every value is missing",
        ),
        ("bad_block.arrow", "a", "Arrow IPC"),
        (
            "short_block.arrow",
            "a",
            "malformed file (a footer block disagrees with its message)",
        ),
        (
            "huge_lz4_2e50.arrow",
            "a",
            "states 1125899906842624 decompressed bytes",
        ),
        (
            "huge_lz4_2e62.arrow",
            "a",
            "states 4611686018427387904 decompressed bytes",
        ),
        (
            "zstd_claim.arrow",
            "a",
            "states 500000000 decompressed bytes, and its frames hold at most 131072",
        ),
        (
            "many_null_batches.arrows",
            "n",
            "column `n` has the Arrow type Null, so every value is missing",
        ),
    ] {
        let err = refusal(&format!(
            "t = load_data(\"{file}\", cartpow(cartprod({column} = reals), 3))\noutputs = t.{column}"
        ));
        assert!(err.contains(reason), "{file}: {err}");
    }
    let out = compile_at(
        "t = load_data(\"lz4_runs.arrow\", cartpow(cartprod(x = reals), 1100))\noutputs = t.x",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    // A million zeros in a small ZSTD body: compressed blocks may expand to 128 KiB each.
    let zeros = compile_at(
        "t = load_data(\"zstd_zeros.arrow\", cartpow(cartprod(a = reals), 3))\noutputs = t.a",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(returned(&zeros), vec![vec![0.0; 3]]);
    let mut want = vec![1.5; 1000];
    want.extend((0..100).map(|i| f64::from(i) * 0.25));
    assert_eq!(returned(&out), vec![want]);
}

/// A JSON integer beyond int64 refuses only when selected.
#[test]
fn an_unselected_json_integer_beyond_int64_reads() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("bigint");
    std::fs::write(
        dir.join("data.json"),
        r#"{"a": [1.5], "big": [9223372036854775808]}"#,
    )?;
    let path = dir.join("query.flatppl");
    let read = |field: &str| {
        Context::default()
            .parse(
                &format!(
                    "data = load_data(\"data.json\", cartprod({field} = cartpow(reals, 1)))\noutputs = data.{field}"
                ),
                None,
                path.to_str(),
            )
            .and_then(|model| model.compile(&EmitOptions::default()))
            .map(|out| out.stablehlo)
            .map_err(|e| e.to_string())
    };
    let literal = Context::default()
        .parse("outputs = [1.5]", None, None)?
        .compile(&EmitOptions::default())?
        .stablehlo;
    assert_eq!(read("a")?, literal);
    let err = read("big").expect_err("must refuse");
    assert!(
        err.contains("integer 9223372036854775808 exceeds int64"),
        "{err}"
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

/// Every integer constant in an emitted module, with its element width, so a
/// test can check each value fits the type it is written at.
fn integer_constants(stablehlo: &str) -> Vec<(i128, u32)> {
    stablehlo
        .lines()
        .filter_map(|line| {
            let (_, rhs) = line.trim().split_once(" = stablehlo.constant dense<")?;
            let (body, ty) = rhs.rsplit_once("> : ")?;
            let bits = ty.trim_end_matches('>').rsplit(['<', 'x']).next()?;
            let bits: u32 = bits.strip_prefix('i')?.parse().ok()?;
            Some((body, bits))
        })
        .filter(|&(_, bits)| bits > 1)
        .flat_map(|(body, bits)| {
            body.split(['[', ']', ','])
                .filter(|s| !s.trim().is_empty())
                .map(move |s| (s.trim().parse::<i128>().expect("integer literal"), bits))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// An integer beyond the dtype's integer width refuses instead of wrapping,
/// and an integer beyond 2^53 reads into reals with ordinary rounding.
#[test]
fn baked_integers_fit_their_emitted_width() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("width");
    std::fs::write(
        dir.join("data.json"),
        r#"{"i": [3000000000, 1], "r": [9007199254740993, 1]}"#,
    )?;
    let path = dir.join("query.flatppl");
    let compile = |field: &str, set: &str, dtype: Dtype| {
        Context::default()
            .parse(
                &format!(
                    "data = load_data(\"data.json\", cartprod({field} = cartpow({set}, 2)))\noutputs = data.{field}"
                ),
                None,
                path.to_str(),
            )
            .and_then(|model| {
                model.compile(&EmitOptions {
                    dtype,
                    ..EmitOptions::default()
                })
            })
            .map(|out| out.stablehlo)
            .map_err(|e| e.to_string())
    };
    let err = compile("i", "integers", Dtype::F32).expect_err("must refuse");
    assert!(
        err.contains("integer 3000000000 is outside i32, the integer width of the f32 dtype"),
        "{err}"
    );
    let wide = compile("i", "integers", Dtype::F64)?;
    let constants = integer_constants(&wide);
    assert!(constants.contains(&(3_000_000_000, 64)), "{wide}");
    for (value, bits) in constants {
        let limit = 1i128 << (bits - 1);
        assert!(
            (-limit..limit).contains(&value),
            "{value} overflows i{bits}"
        );
    }
    assert_eq!(
        returned(&compile("r", "reals", Dtype::F64)?),
        vec![vec![9007199254740992.0, 1.0]]
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

/// The parser writes `-2147483648` as `neg(2147483648)`. Only the value as
/// written has to fit the width, and a refusal names that value.
#[test]
fn negative_integer_literals_fit_as_written() {
    for source in [
        "outputs = -2147483648",
        "outputs = [2147483647, -2147483648]",
    ] {
        let out = compile_at(source).unwrap_or_else(|e| panic!("{source}: {e}"));
        let constants = integer_constants(&out);
        assert!(constants.contains(&(-2147483648, 32)), "{out}");
        assert!(
            constants.iter().all(|&(v, _)| i32::try_from(v).is_ok()),
            "{out}"
        );
    }
    let err = refusal("outputs = -2147483649");
    assert!(
        err.contains("integer -2147483649 is outside i32, the integer width of the f32 dtype"),
        "{err}"
    );
}

#[test]
fn arrow_rejects_blocks_over_the_codec_limit_before_allocating() {
    let fixture = |name: &str| {
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/data")
                .join(name),
        )
        .unwrap()
    };
    let mut zstd = fixture("zstd_claim.arrow");
    let at = zstd
        .windows(4)
        .position(|w| w == [0x28, 0xb5, 0x2f, 0xfd])
        .unwrap();
    let mut skippable = zstd.clone();
    skippable[at..at + 4].copy_from_slice(&0x184d_2a50_u32.to_le_bytes());
    skippable[at + 4..at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    let err = flatppl_data::Table::from_arrow_ipc_values(&skippable)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not hold its codec's frames"), "{err}");

    zstd[at - 8..at].copy_from_slice(&131_073_i64.to_le_bytes());
    let rle_header = (131_073_u32 << 3) | 3;
    zstd[at + 7..at + 10].copy_from_slice(&rle_header.to_le_bytes()[..3]);
    let err = flatppl_data::Table::from_arrow_ipc_values(&zstd)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not hold its codec's frames"), "{err}");

    let mut lz4 = fixture("lz4_runs.arrow");
    let at = lz4
        .windows(4)
        .position(|w| w == [0x04, 0x22, 0x4d, 0x18])
        .unwrap();
    let flags = lz4[at + 4];
    let max = 1_u32 << (16 + 2 * (u32::from((lz4[at + 5] >> 4) & 7) - 4));
    let header = at + 7 + if flags & 8 != 0 { 8 } else { 0 } + if flags & 1 != 0 { 4 } else { 0 };
    lz4[header..header + 4].copy_from_slice(&(0x8000_0000 | (max + 1)).to_le_bytes());
    let err = flatppl_data::Table::from_arrow_ipc_values(&lz4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not hold its codec's frames"), "{err}");
}

#[test]
fn arrow_rejects_out_of_range_message_and_buffer_offsets() {
    let fixture = |name: &str| {
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/data")
                .join(name),
        )
        .unwrap()
    };
    let mut stream = fixture("negative_offset.arrows");
    let pattern = [24_i64, 0, -24, 24].map(i64::to_le_bytes).concat();
    let at = stream
        .windows(pattern.len())
        .position(|w| w == pattern)
        .unwrap();
    stream[at + 16..at + 24].copy_from_slice(&(u32::MAX as i64 + 25).to_le_bytes());
    let err = flatppl_data::Table::from_arrow_ipc_values(&stream)
        .unwrap_err()
        .to_string();
    assert!(err.contains("a buffer leaves its message body"), "{err}");

    let mut file = fixture("bad_block.arrow");
    let old = 1_000_000_000_i64.to_le_bytes();
    let at = file.windows(8).position(|w| w == old).unwrap();
    let mut oversized_body = file.clone();
    oversized_body[at..at + 8].copy_from_slice(&8_i64.to_le_bytes());
    oversized_body[at + 16..at + 24].copy_from_slice(&i64::MAX.to_le_bytes());
    let err = flatppl_data::Table::from_arrow_ipc_values(&oversized_body)
        .unwrap_err()
        .to_string();
    assert!(err.contains("a footer block leaves the file"), "{err}");
    file[at..at + 8].copy_from_slice(&(u32::MAX as i64).to_le_bytes());
    assert!(flatppl_data::Table::from_arrow_ipc_values(&file).is_err());
}
