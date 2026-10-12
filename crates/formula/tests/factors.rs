use flatppl_formula::{
    ColumnValues, Contrast, DataDocument, DataMode, DataValue, DesignComponent, Recipe, Table,
    build, prepare_data, read,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Oracle {
    tables: BTreeMap<String, Table>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    formula: String,
    table: String,
    contrasts: BTreeMap<String, Contrast>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    rows: Vec<usize>,
    factor_levels: BTreeMap<String, Levels>,
    x: Matrix,
    brms_column_names: Vec<String>,
    assign: Vec<usize>,
    means: BTreeMap<String, f64>,
    prediction_x: Option<Matrix>,
}

#[derive(Deserialize)]
struct Levels {
    levels: Vec<String>,
    ordered: bool,
}

#[derive(Deserialize)]
struct Matrix {
    column_names: Vec<String>,
    values: Vec<Vec<f64>>,
}

fn oracle() -> Oracle {
    serde_json::from_str(include_str!("fixtures/factor-matrix.json")).unwrap()
}

// Evaluate the saved public recipe independently of conversion, as a host may
// do for diagnostics. Expected rows, columns and values come from live base R.
fn matrix(recipe: &Recipe, data: &DataDocument) -> Vec<Vec<f64>> {
    let input =
        |predictor: usize, row: usize| match &data.constants[&format!("x_{}", predictor + 1)] {
            DataValue::Real32(values) => f64::from(values[row]),
            DataValue::Real64(values) => values[row],
            DataValue::Integers(values) => values[row] as f64,
            DataValue::Integer(_) => panic!("predictor input must be a vector"),
        };
    (0..data.rows.len())
        .map(|row| {
            let mut values = Vec::new();
            if recipe.predictors["mu"].formula.intercept {
                values.push(1.0);
            }
            for column in &recipe.predictors["mu"].design.columns {
                let mut value = 1.0;
                for component in &column.components {
                    value *= match *component {
                        DesignComponent::Numeric { predictor } => input(predictor, row),
                        DesignComponent::Contrast { predictor, column } => {
                            let factor = recipe.predictors["mu"].design.predictors[predictor]
                                .factor
                                .as_ref()
                                .unwrap();
                            factor.matrix[input(predictor, row) as usize - 1][column]
                        }
                        DesignComponent::Indicator { predictor, level } => {
                            if input(predictor, row) as usize == level + 1 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                }
                values.push(value);
            }
            values
        })
        .collect()
}

fn assert_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} row count");
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.len(), expected.len(), "{context} column count");
        for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-12,
                "{context} [{row},{column}]: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn factor_design_and_prediction_match_r_model_matrices() {
    let oracle = oracle();
    for case in oracle.cases {
        let mut spec = read(&serde_json::json!({ "formula": case.formula }).to_string()).unwrap();
        spec.options.contrasts = case.contrasts;
        let converted = build(&spec, &oracle.tables[&case.table]).unwrap();
        let recipe = &converted.recipe;
        assert_eq!(
            converted.data.rows, case.expected.rows,
            "{} rows",
            case.name
        );
        let offset = usize::from(recipe.predictors["mu"].formula.intercept);
        assert_eq!(
            recipe.predictors["mu"]
                .design
                .columns
                .iter()
                .map(|c| &c.name)
                .collect::<Vec<_>>(),
            case.expected.x.column_names[offset..]
                .iter()
                .collect::<Vec<_>>(),
            "{} column names",
            case.name
        );
        assert_eq!(
            converted
                .info
                .parameters
                .iter()
                .filter(
                    |parameter| parameter.name.starts_with("b_") && parameter.name != "b_Intercept"
                )
                .map(|parameter| parameter.name.as_str())
                .collect::<Vec<_>>(),
            case.expected.brms_column_names[offset..]
                .iter()
                .map(|name| format!("b_{name}"))
                .collect::<Vec<_>>(),
            "{} brms coefficient names",
            case.name
        );
        assert_eq!(
            recipe.predictors["mu"]
                .design
                .columns
                .iter()
                .map(|c| c.term)
                .collect::<Vec<_>>(),
            case.expected.assign[offset..],
            "{} term assignments",
            case.name
        );
        for predictor in &recipe.predictors["mu"].design.predictors {
            if let Some(expected) = case.expected.factor_levels.get(&predictor.column) {
                let actual = predictor.factor.as_ref().unwrap();
                assert_eq!(actual.levels, expected.levels, "{} levels", case.name);
                assert_eq!(actual.ordered, expected.ordered, "{} ordered", case.name);
            }
        }
        assert_matrix(
            &matrix(recipe, &converted.data),
            &case.expected.x.values,
            &case.name,
        );
        for (column, &mean) in recipe.predictors["mu"]
            .design
            .columns
            .iter()
            .zip(&recipe.predictors["mu"].means)
        {
            assert!(
                (mean - case.expected.means[&column.name]).abs() < 2e-12,
                "{} mean of {}",
                case.name,
                column.name
            );
        }
        if let Some(prediction) = &case.expected.prediction_x {
            let restored: Recipe =
                serde_json::from_str(&serde_json::to_string(recipe).unwrap()).unwrap();
            let prepared = prepare_data(
                &restored,
                &oracle.tables["prediction"],
                DataMode::Prediction,
            )
            .unwrap();
            assert_matrix(
                &matrix(&restored, &prepared),
                &prediction.values,
                &format!("{} prediction", case.name),
            );
            for (actual, expected) in restored.predictors["mu"]
                .means
                .iter()
                .zip(&recipe.predictors["mu"].means)
            {
                assert!((actual - expected).abs() < 2e-12);
            }
        }
    }
}

#[test]
fn prediction_reuses_factor_levels_and_preserves_numeric_precision() {
    let oracle = oracle();
    let case = oracle
        .cases
        .iter()
        .find(|case| case.name == "prediction_interaction")
        .unwrap();
    let mut spec = read(&serde_json::json!({ "formula": case.formula }).to_string()).unwrap();
    spec.options.contrasts = case.contrasts.clone();
    let converted = build(&spec, &oracle.tables[&case.table]).unwrap();
    let mut table = oracle.tables["prediction"].clone();
    table
        .columns
        .iter_mut()
        .find(|c| c.name == "x")
        .unwrap()
        .values = ColumnValues::Real32(vec![8.0, -2.0, 5.0]);
    let data = prepare_data(&converted.recipe, &table, DataMode::Prediction).unwrap();
    assert_eq!(
        data.constants["x_1"],
        DataValue::Real32(vec![8.0, -2.0, 5.0])
    );
    assert_eq!(data.constants["x_2"], DataValue::Integers(vec![3, 3, 3]));
    assert_matrix(
        &matrix(&converted.recipe, &data),
        &case.expected.prediction_x.as_ref().unwrap().values,
        "f32 prediction",
    );

    // The training factor dropped its unused label. A prediction cannot turn
    // that label into a new coefficient or reuse an existing code silently.
    let ColumnValues::Factor { codes, .. } = &mut table
        .columns
        .iter_mut()
        .find(|c| c.name == "f")
        .unwrap()
        .values
    else {
        panic!("fixture factor")
    };
    codes[0] = 3;
    assert!(
        prepare_data(&converted.recipe, &table, DataMode::Prediction)
            .unwrap_err()
            .0
            .contains("unseen level `unused`")
    );
}

#[test]
fn coefficient_names_cannot_alias_the_derived_intercept() {
    let mut spec = read(
        r#"{
        "formula": "y ~ I",
        "data": {"nrows": 2, "columns": [
            {"name": "y", "values": {"kind": "real64", "values": [0, 1]}},
            {"name": "I", "values": {"kind": "factor", "values": {
                "levels": ["base", "ntercept"], "codes": [0, 1], "ordered": false
            }}}
        ]}
    }"#,
    )
    .unwrap();
    let table = spec.data.take().unwrap();
    assert!(
        build(&spec, &table)
            .err()
            .unwrap()
            .0
            .contains("duplicate brms name `Intercept`")
    );
    spec.formula = "y ~ 0 + I".into();
    let converted = build(&spec, &table).unwrap();
    let parameters = &converted.info.parameters;
    assert_eq!(
        parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>(),
        ["b_Ibase", "b_Intercept", "sigma"]
    );
    let parameter = &parameters[1];
    assert_eq!(
        (
            parameter.binding.as_str(),
            parameter.index,
            parameter.derived
        ),
        ("b", Some(2), false)
    );
}

#[test]
fn larger_ordered_factors_use_indicators_or_other_contrasts() {
    let table = Table {
        nrows: 17,
        columns: vec![
            flatppl_formula::Column {
                name: "y".into(),
                values: ColumnValues::Real64(vec![0.0; 17]),
                valid: None,
            },
            flatppl_formula::Column {
                name: "o".into(),
                values: ColumnValues::Factor {
                    levels: (1..=17).map(|i| i.to_string()).collect(),
                    codes: (0..17).collect(),
                    ordered: true,
                },
                valid: None,
            },
        ],
    };
    let mut spec = read(r#"{"formula":"y ~ o"}"#).unwrap();
    let error = build(&spec, &table).err().unwrap();
    assert!(error.0.contains("at most 16 observed levels"));
    spec.options
        .contrasts
        .insert("o".into(), Contrast::Treatment);
    let converted = build(&spec, &table).unwrap();
    assert_eq!(converted.recipe.predictors["mu"].design.columns.len(), 16);
    assert_eq!(
        converted.recipe.predictors["mu"].design.columns[0].name,
        "o2"
    );
    assert_eq!(
        converted.recipe.predictors["mu"].design.columns[15].name,
        "o17"
    );

    spec.options.contrasts.clear();
    spec.formula = "y ~ 0 + o".into();
    let converted = build(&spec, &table).unwrap();
    let identity: Vec<_> = (0..17)
        .map(|row| {
            (0..17)
                .map(|column| if row == column { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    assert_matrix(
        &matrix(&converted.recipe, &converted.data),
        &identity,
        "ordered full indicators",
    );
}
