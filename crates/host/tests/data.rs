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
