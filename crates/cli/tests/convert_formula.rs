#![cfg(feature = "formula")]

use flatppl_formula::{DataDocument, DataMode, DataPlacement, Recipe, Spec, Table};
use std::{fs, path::Path, process::Command};

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

fn assert_conversion(output: &Path, spec: &Spec, table: &Table) {
    let direct = flatppl_formula::build(spec, table).unwrap();
    let module = flatppl_syntax::parse(&fs::read_to_string(output).unwrap()).unwrap();
    assert_eq!(
        flatppl_syntax::print(&module),
        flatppl_syntax::print(&direct.module)
    );
    for (extension, expected) in [
        ("data.json", serde_json::to_value(&direct.data).unwrap()),
        ("recipe.json", serde_json::to_value(&direct.recipe).unwrap()),
        ("model.json", serde_json::to_value(&direct.info).unwrap()),
    ] {
        let actual: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output.with_extension(extension)).unwrap())
                .unwrap();
        assert_eq!(actual, expected, "{extension}");
    }
}

#[test]
fn external_text_data_and_placement_match_direct_conversion() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-formula-cli-text-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let input = directory.join("model.formula.json");
    let output = directory.join("model.flatppl");
    let mut spec = flatppl_formula::read(r#"{"formula":"y ~ x + f + flag + (1 | g)"}"#).unwrap();
    let table =
        Table::from_json(include_str!("../../formula/tests/fixtures/text/typed.json")).unwrap();
    for (name, source, placement) in [
        (
            "table.csv",
            include_str!("../../formula/tests/fixtures/text/table.csv"),
            "inline",
        ),
        (
            "records.json",
            include_str!("../../formula/tests/fixtures/text/records.json"),
            "external",
        ),
        (
            "columns.json",
            include_str!("../../formula/tests/fixtures/text/columns.json"),
            "inline",
        ),
        (
            "typed.json",
            include_str!("../../formula/tests/fixtures/text/typed.json"),
            "external",
        ),
    ] {
        // The command-line value must override the opposite specification value.
        spec.options.data_placement = if placement == "inline" {
            DataPlacement::External
        } else {
            DataPlacement::Inline
        };
        fs::write(&input, serde_json::to_string(&spec).unwrap()).unwrap();
        fs::write(directory.join(name), source).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .current_dir(&directory)
            .args([
                "convert",
                "model.formula.json",
                "model.flatppl",
                "--data",
                name,
            ])
            .args(["--data-placement", placement, "--no-header"])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        spec.options.data_placement = if placement == "inline" {
            DataPlacement::Inline
        } else {
            DataPlacement::External
        };
        assert_conversion(&output, &spec, &table);
    }

    spec.data = Some(table);
    fs::write(&input, serde_json::to_string(&spec).unwrap()).unwrap();
    let conflict = directory.join("conflict.flatppl");
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("convert")
        .arg(&input)
        .arg(&conflict)
        .arg("--data")
        .arg(directory.join("table.csv"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    for extension in ["flatppl", "data.json", "recipe.json", "model.json"] {
        assert!(!conflict.with_extension(extension).exists());
    }
    fs::remove_dir_all(directory).unwrap();
}

#[cfg(feature = "formula-arrow")]
#[test]
fn external_arrow_retains_typed_columns_and_dictionary_recipe() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-formula-cli-arrow-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let input = directory.join("model.formula.json");
    let output = directory.join("model.flatppl");
    let spec = flatppl_formula::read(r#"{"formula":"y ~ x + f"}"#).unwrap();
    fs::write(&input, serde_json::to_string(&spec).unwrap()).unwrap();
    for (name, bytes) in [
        (
            "table.arrow",
            include_bytes!("../../formula/tests/fixtures/arrow/training.arrow").as_slice(),
        ),
        (
            "table.arrows",
            include_bytes!("../../formula/tests/fixtures/arrow/training.arrows").as_slice(),
        ),
    ] {
        let path = directory.join(name);
        fs::write(&path, bytes).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("convert")
            .arg(&input)
            .arg(&output)
            .arg("--data")
            .arg(path)
            .arg("--no-header")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_conversion(&output, &spec, &Table::from_arrow_ipc(bytes).unwrap());
    }
    fs::remove_dir_all(directory).unwrap();
}
