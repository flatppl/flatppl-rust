use flatppl_formula::{DataMode, DataValue, GroupLevel, Table, build, prepare_data, read};

#[test]
fn text_tables_share_model_frames_and_prediction_levels() -> Result<(), Box<dyn std::error::Error>>
{
    let spec = read(r#"{"formula":"y ~ x + f + flag + (1 | g)"}"#)?;
    let expected = build(
        &spec,
        &Table::from_json(include_str!("fixtures/text/typed.json"))?,
    )?;
    assert_eq!(expected.data.rows, vec![0, 2, 3]);
    assert_eq!(
        expected.data.constants["x_1"],
        DataValue::Real64(vec![1., 3., 4.])
    );
    assert_eq!(
        expected.recipe.predictors["mu"].groups[0].levels,
        vec![GroupLevel::Integer(i64::MIN), GroupLevel::Integer(i64::MAX)]
    );
    let prediction =
        Table::from_json(r#"[{"x":5.0,"f":"a","flag":false,"g":9223372036854775807}]"#)?;
    let expected_prediction = prepare_data(&expected.recipe, &prediction, DataMode::Prediction)?;
    for table in [
        Table::from_csv(include_bytes!("fixtures/text/table.csv").as_slice())?,
        Table::from_json(include_str!("fixtures/text/records.json"))?,
        Table::from_json(include_str!("fixtures/text/columns.json"))?,
    ] {
        let model = build(&spec, &table)?;
        assert_eq!(model.data, expected.data);
        assert_eq!(
            serde_json::to_value(&model.recipe)?,
            serde_json::to_value(&expected.recipe)?
        );
        assert_eq!(
            flatppl_syntax::print(&model.module),
            flatppl_syntax::print(&expected.module)
        );
        assert_eq!(
            prepare_data(&model.recipe, &prediction, DataMode::Prediction)?,
            expected_prediction
        );
    }
    Ok(())
}

#[test]
fn typed_precision_and_json_numeric_boundaries_are_preserved()
-> Result<(), Box<dyn std::error::Error>> {
    let source = r#"{"nrows":2,"columns":[
          {"name":"y","values":{"kind":"int","values":[1,2]}},
          {"name":"x","values":{"values":[0.1,0.2],"kind":"real32"}}
        ]}"#;
    let table = Table::from_json(source)?;
    let spec = read(&format!(r#"{{"formula":"y ~ x","data":{source}}}"#))?;
    let model = build(&spec, &table)?;
    assert_eq!(model.data, build(&spec, spec.data.as_ref().unwrap())?.data);
    assert_eq!(
        model.data.constants["x_1"],
        DataValue::Real32(vec![0.1, 0.2])
    );
    let (loaded, sidecar) = model.with_data_source("model.data.json")?;
    assert_eq!(
        sidecar["x_1"],
        serde_json::json!([f64::from(0.1_f32), f64::from(0.2_f32)])
    );
    assert_eq!(sidecar["y_data"], serde_json::json!([1, 2]));
    assert!(!sidecar.as_object().unwrap().contains_key("N"));
    let mut parsed = flatppl_syntax::parse(&flatppl_syntax::print(&loaded))?;
    assert!(flatppl_infer::infer(&mut parsed).is_empty());
    let exact = Table::from_json(r#"{"y":[1,2],"x":[9007199254740994,1e0]}"#)?;
    assert_eq!(
        build(&spec, &exact)?.data.constants["x_1"],
        DataValue::Real64(vec![9007199254740994., 1.])
    );
    for json in [
        r#"{"x":[9223372036854775808]}"#,
        r#"{"x":[18446744073709551616]}"#,
        r#"{"x":[-9223372036854775809]}"#,
        r#"{"x":[9007199254740993,1.0]}"#,
        r#"{"x":[1,"2"]}"#,
        r#"{"x":[true,1]}"#,
        r#"{"x":[1e999]}"#,
    ] {
        assert!(Table::from_json(json).is_err(), "{json}");
    }
    for csv in ["x\n9223372036854775808\n", "x\n9007199254740993\n1.0\n"] {
        assert!(Table::from_csv(csv.as_bytes()).is_err());
    }
    Ok(())
}
