use flatppl_formula::{DataDocument, DataValue, ModelInfo, Recipe};
use flatppl_wasm_api::{convert_str, import_formula_str, prepare_formula_str};
use serde::Deserialize;

#[derive(Deserialize)]
struct Imported {
    source: String,
    data: DataDocument,
    recipe: Recipe,
    info: ModelInfo,
}

#[test]
fn formula_template_and_recipe_survive_the_json_boundary() -> Result<(), Box<dyn std::error::Error>>
{
    let json = import_formula_str(
        r#"{"formula":"y ~ x + f"}"#,
        Some(include_bytes!(
            "../../formula/tests/fixtures/arrow/training.arrows"
        )),
    )?;
    let imported: Imported = serde_json::from_str(&json)?;
    flatppl_syntax::parse(&imported.source)?;
    assert_eq!(imported.info.training_rows, vec![0, 1, 3]);
    assert_eq!(
        imported.data.constants["x_1"],
        DataValue::Real32(vec![0., 1., 3.])
    );
    let recipe = serde_json::to_string(&imported.recipe)?;
    let prepared = prepare_formula_str(
        &recipe,
        include_bytes!("../../formula/tests/fixtures/arrow/prediction.arrows"),
        "prediction",
    )?;
    let data: DataDocument = serde_json::from_str(&prepared)?;
    assert_eq!(data.rows, vec![0, 2]);
    assert_eq!(data.constants["N"], DataValue::Integer(2));
    assert_eq!(data.constants["x_2"], DataValue::Integers(vec![2, 1]));

    let source = convert_str(
        include_str!("../../formula/tests/fixtures/gaussian.formula.json"),
        "formula",
        "flatppl",
    )?;
    let module = flatppl_syntax::parse(&format!("{source}\nexpected_N = 3\n"))?;
    let rhs = |name| {
        module
            .bindings()
            .find(|(_, binding)| module.resolve(binding.name) == name)
            .unwrap()
            .1
            .rhs
    };
    assert!(module.structural_eq(rhs("N"), rhs("expected_N")));
    Ok(())
}
