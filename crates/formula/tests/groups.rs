use flatppl_formula::{
    ColumnValues, DataDocument, DataMode, DataValue, DesignComponent, Recipe, Spec, Table, build,
    prepare_data, read,
};
use serde::Deserialize;
use serde_json::Value;

fn table() -> Table {
    read(
        r#"{
      "formula":"y ~ 1", "data":{"nrows":4,"columns":[
        {"name":"y","values":{"kind":"real64","values":[2,4,8,7]}},
        {"name":"x","values":{"kind":"real32","values":[1,2,3,4]}},
        {"name":"g","values":{"kind":"factor","values":{"levels":["B","A"],"codes":[1,0,1,0]}}},
        {"name":"h","values":{"kind":"int","values":[1,2,2,1]}},
        {"name":"f","values":{"kind":"factor","values":{"levels":["b","a"],"codes":[0,1,0,1]}}}
      ]}}
    "#,
    )
    .unwrap()
    .data
    .unwrap()
}

fn spec(formula: &str) -> Spec {
    read(&serde_json::json!({"formula":formula}).to_string()).unwrap()
}

// Interpret the public prediction recipe to check its group columns against
// independently generated brms make_standata fixtures.
fn columns(recipe: &Recipe, data: &DataDocument, index: usize) -> Vec<Vec<f64>> {
    let group = &recipe.predictors["mu"].groups[index];
    let input = |predictor: usize, row: usize| match &data.constants[&group.inputs[predictor]] {
        DataValue::Real32(v) => f64::from(v[row]),
        DataValue::Real64(v) => v[row],
        DataValue::Integers(v) => v[row] as f64,
        DataValue::Integer(_) => panic!("group design inputs are vectors"),
    };
    let mut out = Vec::new();
    if recipe.predictors["mu"].formula.groups[index].intercept {
        out.push(vec![1.0; data.rows.len()]);
    }
    for column in &group.design.columns {
        out.push(
            (0..data.rows.len())
                .map(|row| {
                    column
                        .components
                        .iter()
                        .map(|component| match *component {
                            DesignComponent::Numeric { predictor } => input(predictor, row),
                            DesignComponent::Contrast { predictor, column } => {
                                group.design.predictors[predictor]
                                    .factor
                                    .as_ref()
                                    .unwrap()
                                    .matrix[input(predictor, row) as usize - 1][column]
                            }
                            DesignComponent::Indicator { predictor, level } => {
                                if input(predictor, row) as usize == level + 1 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                        })
                        .product()
                })
                .collect(),
        );
    }
    out
}

#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    formula: String,
    standata: Value,
}

#[test]
fn group_designs_and_indices_match_brms() {
    let oracle: Oracle = serde_json::from_str(include_str!("fixtures/group-matrix.json")).unwrap();
    for case in oracle.cases {
        let converted = build(&spec(&case.formula), &table()).unwrap();
        let expected_groups = case
            .standata
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| key.starts_with("J_"))
            .count();
        assert_eq!(
            converted.recipe.predictors["mu"].groups.len(),
            expected_groups,
            "{} group blocks",
            case.name
        );
        for (i, group) in converted.recipe.predictors["mu"].groups.iter().enumerate() {
            let id = i + 1;
            assert_eq!(
                group.levels.len() as u64,
                case.standata[format!("N_{id}")],
                "{} levels",
                case.name
            );
            assert_eq!(
                group.coefficients.len() as u64,
                case.standata[format!("M_{id}")],
                "{} dimensions",
                case.name
            );
            let indices: Vec<i64> =
                serde_json::from_value(case.standata[format!("J_{id}")].clone()).unwrap();
            assert_eq!(
                converted.data.constants[&group.index],
                DataValue::Integers(indices),
                "{} indices",
                case.name
            );
            for (j, column) in columns(&converted.recipe, &converted.data, i)
                .iter()
                .enumerate()
            {
                let expected: Vec<f64> =
                    serde_json::from_value(case.standata[format!("Z_{id}_{}", j + 1)].clone())
                        .unwrap();
                assert_eq!(column, &expected, "{} design column {}", case.name, j + 1);
            }
            assert_eq!(
                group.correlation.is_some(),
                group.coefficients.len() > 1 && !case.formula.contains("||")
            );
        }
    }
}

#[test]
fn group_recipe_preserves_names_coordinates_and_prediction_coding() {
    let converted = build(&spec("y ~ h + (1+x|g/h)"), &table()).unwrap();
    assert_eq!(
        converted.recipe.predictors["mu"]
            .groups
            .iter()
            .map(|group| group.column.as_str())
            .collect::<Vec<_>>(),
        ["g", "g:h"]
    );
    assert_eq!(
        converted.recipe.predictors["mu"].groups[1]
            .levels
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["A_1", "A_2", "B_1", "B_2"]
    );
    let p = converted
        .info
        .parameters
        .iter()
        .find(|p| p.name == "r_g[A,x]")
        .unwrap();
    assert_eq!(p.indices.as_deref(), Some(&[2, 2][..]));
    assert!(p.derived);
    let p = converted
        .info
        .parameters
        .iter()
        .find(|p| p.name == "z_1[2,1]")
        .unwrap();
    assert_eq!(p.indices.as_deref(), Some(&[2, 1][..]));
    assert!(!p.derived);
    assert!(
        converted
            .info
            .parameters
            .iter()
            .any(|p| p.name == "cor_g__Intercept__x" && p.derived)
    );
    let recipe: Recipe =
        serde_json::from_str(&serde_json::to_string(&converted.recipe).unwrap()).unwrap();
    let mut newdata = table();
    newdata.columns.retain(|column| column.name != "y");
    newdata.columns[0].values = ColumnValues::Real32(vec![4.0, 3.0, 2.0, 1.0]);
    let prepared = prepare_data(&recipe, &newdata, DataMode::Prediction).unwrap();
    assert_eq!(
        columns(&recipe, &prepared, 0),
        [vec![1.0; 4], vec![4.0, 3.0, 2.0, 1.0]]
    );
    assert_eq!(
        columns(&recipe, &prepared, 1),
        columns(&recipe, &prepared, 0)
    );
    newdata.columns[2].values = ColumnValues::Int(vec![3, 2, 2, 1]);
    assert!(
        prepare_data(&recipe, &newdata, DataMode::Prediction)
            .unwrap_err()
            .0
            .contains("unseen level")
    );
}

#[test]
fn group_expressions_and_overlapping_blocks_follow_brms() {
    let converted = build(&spec("y ~ (1|g) + (0+I(x^2)|g)"), &table()).unwrap();
    assert_eq!(converted.recipe.predictors["mu"].groups.len(), 2);
    assert_eq!(
        converted.recipe.predictors["mu"].groups[1].coefficients,
        ["IxE2"]
    );
    assert_eq!(
        columns(&converted.recipe, &converted.data, 1),
        [vec![1.0, 4.0, 9.0, 16.0]]
    );
    for formula in [
        "y ~ (1+x|g)+(1|g)",
        "y ~ (1+x|g)+(1+x||g)",
        "y ~ (0+x|g)+(0+x+f|g)",
    ] {
        assert!(
            build(&spec(formula), &table())
                .err()
                .unwrap()
                .0
                .contains("duplicated group effect")
        );
    }
}
