use std::path::PathBuf;

use flatppl_bundle::{Compression, pack};
use flatppl_host::{Context, EmitOptions};

const PORTABLE: &[u8] = include_bytes!("../../bundle/tests/fixtures/portable.zip");

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "flatppl-host-bundles-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn archive(files: &[(&str, &[u8])]) -> Vec<u8> {
    pack(
        files
            .iter()
            .map(|(name, bytes)| ((*name).to_owned(), bytes.to_vec())),
        Compression::Deflated,
    )
    .unwrap()
}

#[test]
fn archive_load_and_memory_load_keep_imports_and_data_outside_the_hot_path()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("snapshot");
    let file = dir.join("model.flatppl.zip");
    std::fs::write(&file, PORTABLE)?;
    let mut context = Context::default();
    let from_file = context.load(file.to_str().unwrap())?;
    let from_bytes = context.load_bundle("memory.zip", PORTABLE)?;
    let hlo = from_file.compile(&EmitOptions::default())?.stablehlo;
    assert_eq!(from_bytes.compile(&EmitOptions::default())?.stablehlo, hlo);
    context.register("model.flatppl.zip", &from_file)?;
    let reimported = context.parse("point = elementof(reals)\nm = load_module(\"model.flatppl.zip\", theta = point)\ninputs = point\noutputs = m.outputs", None, dir.join("query.flatppl").to_str())?;
    reimported.compile(&EmitOptions::default())?;
    let sliced = context.slice(&from_file, &["inputs", "outputs"])?;
    assert_eq!(sliced.compile(&EmitOptions::default())?.stablehlo, hlo);
    std::fs::remove_file(file)?;
    assert_eq!(from_file.compile(&EmitOptions::default())?.stablehlo, hlo);
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn separate_archives_and_import_instances_keep_their_own_definitions()
-> Result<(), Box<dyn std::error::Error>> {
    let child = archive(&[(
        "main.flatppl",
        b"offset = external(reals)\nvalue = offset + 2",
    )]);
    let other = archive(&[("main.flatppl", b"value = 9")]);
    let outer = archive(&[
        ("main.flatppl", b"path = \"child.zip\"\na = load_module(path, offset = 1.0)\nb = load_module(path, offset = 3.0)\nc = load_module(\"other.zip\")\noutputs = (a.value, b.value, c.value)"),
        ("child.zip", &child), ("other.zip", &other),
    ]);
    let mut context = Context::default();
    let model = context.load_bundle("model.zip", &outer)?;
    let expected = context.parse("outputs = (3.0, 5.0, 9)", None, None)?;
    assert_eq!(
        model.compile(&EmitOptions::default())?.stablehlo,
        expected.compile(&EmitOptions::default())?.stablehlo
    );
    Ok(())
}

#[test]
fn archive_data_stays_lazy_and_missing_data_never_reads_the_host()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = archive(&[(
        "main.flatppl",
        b"data = load_data(\"absent.json\", cartpow(reals, 2))\ninputs = data\noutputs = sum(data)",
    )]);
    Context::default()
        .load_bundle("lazy.zip", &bytes)?
        .compile(&EmitOptions::default())?;
    let missing = archive(&[(
        "main.flatppl",
        b"outputs = load_data(\"absent.json\", cartpow(reals, 2))",
    )]);
    let dir = directory("missing-data");
    std::fs::write(dir.join("absent.json"), "[99, 100]")?;
    let zip = dir.join("missing.zip");
    std::fs::write(&zip, missing)?;
    let model = Context::default().load(zip.to_str().unwrap())?;
    assert!(model.compile(&EmitOptions::default()).is_err());
    let ordinary = Context::default().parse(
        "outputs = load_data(\"absent.json\", cartpow(reals, 2))",
        None,
        dir.join("loose.flatppl").to_str(),
    )?;
    ordinary.compile(&EmitOptions::default())?;
    let unmapped = Context::default().parse(
        "outputs = load_data(\"file:///__flatppl_bundle__/unmapped.json\", cartpow(reals, 2))",
        None,
        None,
    )?;
    assert!(
        unmapped
            .compile(&EmitOptions::default())
            .err()
            .unwrap()
            .message
            .contains("no mapped member")
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn directory_and_archive_imports_enforce_their_own_roots() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = directory("containment");
    let root = dir.join("bundle");
    std::fs::create_dir_all(&root)?;
    let outside = dir.join("outside.flatppl");
    std::fs::write(&outside, "x = 1")?;
    std::fs::write(
        root.join("main.flatppl"),
        format!(
            "m = load_module({:?})\noutputs = m.x",
            outside.to_str().unwrap()
        ),
    )?;
    assert!(Context::default().load(root.to_str().unwrap()).is_err());
    let contained = archive(&[
        ("main.flatppl", b"m = load_module(\"sub\")\noutputs = m.x"),
        (
            "sub/main.flatppl",
            b"m = load_module(\"../outside.flatppl\")\nx = m.x",
        ),
        ("outside.flatppl", b"x = 1"),
    ]);
    assert!(
        Context::default()
            .load_bundle("contained.zip", &contained)
            .is_err()
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn directory_links_resolve_before_parent_traversal() -> Result<(), Box<dyn std::error::Error>> {
    let dir = directory("physical");
    let root = dir.join("bundle");
    std::fs::create_dir_all(root.join("actual/deep"))?;
    std::os::unix::fs::symlink("actual/deep", root.join("link"))?;
    std::fs::write(root.join("actual/helper.flatppl"), "x = 3")?;
    std::fs::write(
        root.join("main.flatppl"),
        "m = load_module(\"link/../helper.flatppl\")\noutputs = m.x",
    )?;
    let mut context = Context::default();
    let model = context.load(root.to_str().unwrap())?;
    let expected = context.parse("outputs = 3", None, None)?;
    assert_eq!(
        model.compile(&EmitOptions::default())?.stablehlo,
        expected.compile(&EmitOptions::default())?.stablehlo
    );
    std::fs::remove_dir_all(dir)?;
    Ok(())
}
