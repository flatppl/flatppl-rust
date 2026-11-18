use flatppl_formula::{ColumnValues, DataMode, DataValue, build, prepare_data, read};

#[test]
fn partial_credit_recipe_keeps_category_order_and_numeric_cs_data() {
    let mut spec = read(include_str!("fixtures/cumulative.formula.json")).unwrap();
    spec.family = "acat".into();
    spec.formula = "y ~ cs(x)".into();
    spec.priors.retain(|prior| prior.class == "Intercept");
    let mut table = spec.data.take().unwrap();
    table.columns[1].values = ColumnValues::Real32(vec![1.0, 2.0, 3.0, 4.0]);
    table.columns[1].valid = Some(vec![true, true, true, false]);
    let converted = build(&spec, &table).unwrap();

    assert_eq!(converted.data.rows, vec![0, 1, 2]);
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![2, 1, 3])
    );
    assert_eq!(
        converted.data.constants["cs_data_1"],
        DataValue::Real32(vec![1.0, 2.0, 3.0])
    );
    assert_eq!(converted.info.score_offset, 0.0);
    let slope = converted
        .info
        .parameters
        .iter()
        .find(|p| p.name == "bcs_x[2]")
        .unwrap();
    assert_eq!((&*slope.binding, slope.index), ("bcs_1", Some(2)));

    let serialized = serde_json::to_string(&converted.recipe).unwrap();
    let recipe = serde_json::from_str(&serialized).unwrap();
    table.columns[0].values = ColumnValues::Factor {
        levels: vec!["high".into(), "low".into(), "medium".into()],
        codes: vec![2, 1, 0, 2],
        ordered: true,
    };
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Training).unwrap(),
        converted.data
    );
}
