#![cfg(feature = "formula")]

use flatppl_formula::{DataDocument, DataMode, Recipe};
use std::{fs, process::Command};

#[test]
fn formula_conversion_writes_explicit_reusable_data_sidecars() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-formula-cli-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let input = directory.join("model.formula.json");
    let output = directory.join("model.flatppl");
    let source = include_str!("../../formula/tests/fixtures/gaussian.formula.json");
    fs::write(&input, source).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("convert")
        .arg(&input)
        .arg(&output)
        .arg("--no-header")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let data: DataDocument =
        serde_json::from_str(&fs::read_to_string(output.with_extension("data.json")).unwrap())
            .unwrap();
    let recipe: Recipe =
        serde_json::from_str(&fs::read_to_string(output.with_extension("recipe.json")).unwrap())
            .unwrap();
    let spec = flatppl_formula::read(source).unwrap();
    assert_eq!(
        data,
        flatppl_formula::prepare_data(&recipe, spec.data.as_ref().unwrap(), DataMode::Training)
            .unwrap()
    );
    let imported = flatppl_syntax::parse(&fs::read_to_string(&output).unwrap()).unwrap();
    let direct = flatppl_formula::build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        flatppl_syntax::print(&imported),
        flatppl_syntax::print(&direct.module)
    );
    fs::remove_dir_all(directory).unwrap();
}
