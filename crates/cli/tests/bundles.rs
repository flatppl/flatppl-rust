//! Archive inputs use the same model graph through every CLI read path.

#![cfg(all(
    feature = "infer",
    feature = "stablehlo",
    feature = "mathdoc",
    feature = "prepare"
))]

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

mod common;
use common::Scratch;

const PORTABLE: &[u8] = include_bytes!("../../bundle/tests/fixtures/portable.zip");

fn run(args: &[&str], paths: &[&Path]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(args)
        .args(paths)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn archive_models_keep_imports_and_data_through_cli_verbs() {
    let dir = Scratch::new("archive-verbs");
    let archive = dir.path("model.flatppl.zip");
    fs::write(&archive, PORTABLE).unwrap();
    let literal = dir.path("literal.flatppl");
    fs::write(&literal, "theta = elementof(reals)\ninputs = theta\noutputs = logdensityof(Normal(0, 1), theta) + sum(logdensityof.(Normal.(theta, 1), [1.0, 2.0]))").unwrap();
    let actual = run(&["stablehlo"], &[&archive]);
    let expected = run(&["stablehlo"], &[&literal]);
    assert_eq!(actual.stdout, expected.stdout);
    run(&["infer"], &[&archive, &dir.path("model.flatpir")]);
    run(&["prepare"], &[&archive]);
    run(&["determinize"], &[&archive]);
    run(
        &["convert", "--no-header"],
        &[&archive, &dir.path("model.html")],
    );
    assert!(
        fs::read_to_string(dir.path("model.html"))
            .unwrap()
            .contains("data-flatppl-binding=\"outputs\"")
    );

    let unpacked = dir.path("working-copy");
    run(&["unpack"], &[&archive, &unpacked]);
    assert_eq!(
        fs::read(unpacked.join("data/raw.bin")).unwrap(),
        [0, 1, 255, 128]
    );
    assert_eq!(
        fs::read(unpacked.join("data/observed-link.json")).unwrap(),
        b"[1, 2]\n"
    );
    assert_eq!(run(&["stablehlo"], &[&unpacked]).stdout, expected.stdout);
    let repeated = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("unpack")
        .arg(&archive)
        .arg(&unpacked)
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert_eq!(
        fs::read(unpacked.join("data/raw.bin")).unwrap(),
        [0, 1, 255, 128]
    );
}

#[test]
fn pack_keeps_fixed_aliases_comments_and_binary_data() {
    let dir = Scratch::new("pack-sources");
    let source = "% Keep this comment and spacing.\nprior_path = \"prior.flatppl\"\nprior = load_module(prior_path, mu = 2.0)\nraw = load_data(\"raw.bin\", cartpow(reals, 4))\nx = elementof(reals)\ninputs = x\noutputs = logdensityof(prior.Pi, x)\n";
    fs::write(dir.path("model.flatppl"), source).unwrap();
    fs::write(
        dir.path("prior.flatppl"),
        "mu = external(reals)\nPi = Normal(mu, 1)\n",
    )
    .unwrap();
    fs::write(dir.path("raw.bin"), [0, 255, 128, 1]).unwrap();
    let archive = dir.path("export.zip");
    run(&["pack"], &[&dir.path("model.flatppl"), &archive]);
    let bytes = fs::read(&archive).unwrap();
    let bundle =
        flatppl_bundle::Bundle::open("export.zip", bytes.into(), Default::default()).unwrap();
    let bundled_source = String::from_utf8(bundle.read("main.flatppl").unwrap().to_vec()).unwrap();
    let expected = source
        .replace(
            "load_module(prior_path,",
            "load_module(\"modules/prior.flatppl\",",
        )
        .replace("load_data(\"raw.bin\"", "load_data(\"data/raw.bin\"");
    assert_eq!(bundled_source, expected);
    assert_eq!(&*bundle.read("data/raw.bin").unwrap(), &[0, 255, 128, 1]);
    assert_eq!(
        run(&["stablehlo"], &[&archive]).stdout,
        run(&["stablehlo"], &[&dir.path("model.flatppl")]).stdout
    );
}

#[test]
fn archive_and_directory_imports_enforce_their_own_roots() {
    let dir = Scratch::new("bundle-boundary");
    fs::create_dir(dir.path("model")).unwrap();
    fs::write(dir.path("outside.flatppl"), "x = 99").unwrap();
    fs::write(
        dir.path("model/main.flatppl"),
        "m = load_module(\"../outside.flatppl\")\nx = m.x",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(dir.path("model"))
        .arg(dir.path("out.flatpir"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!dir.path("out.flatpir").exists());

    let make_child = |value: u8| {
        flatppl_bundle::pack(
            [(
                "main.flatppl".to_string(),
                format!("value = {value}").into_bytes(),
            )],
            flatppl_bundle::Compression::Stored,
        )
        .unwrap()
    };
    let archive = flatppl_bundle::pack([
        ("main.flatppl".to_string(), b"a = load_module(\"a.zip\")\nb = load_module(\"b.zip\")\nx = elementof(reals)\ninputs = x\noutputs = a.value * x + b.value".to_vec()),
        ("a.zip".to_string(), make_child(2)),
        ("b.zip".to_string(), make_child(3)),
    ], flatppl_bundle::Compression::Stored).unwrap();
    fs::write(dir.path("nested.zip"), archive).unwrap();
    fs::write(
        dir.path("literal.flatppl"),
        "x = elementof(reals)\ninputs = x\noutputs = 2 * x + 3",
    )
    .unwrap();
    assert_eq!(
        run(&["stablehlo"], &[&dir.path("nested.zip")]).stdout,
        run(&["stablehlo"], &[&dir.path("literal.flatppl")]).stdout
    );
    run(
        &["pack"],
        &[&dir.path("nested.zip"), &dir.path("repacked.zip")],
    );
    assert_eq!(
        run(&["stablehlo"], &[&dir.path("repacked.zip")]).stdout,
        run(&["stablehlo"], &[&dir.path("literal.flatppl")]).stdout
    );
}

#[test]
fn host_export_keeps_registered_sources_readable_by_the_cli() {
    let dir = Scratch::new("host-export");
    let mut context = flatppl_host::Context::default();
    let model = context.parse("value = 7.0", None, None).unwrap();
    context.register("anonymous", &model).unwrap();
    let query = context.parse("m = load_module(\"anonymous\")\nx = elementof(reals)\ninputs = x\noutputs = x + m.value", None, None).unwrap();
    let archive = dir.path("export.zip");
    fs::write(&archive, query.pack().unwrap()).unwrap();
    let literal = dir.path("literal.flatppl");
    fs::write(
        &literal,
        "x = elementof(reals)\ninputs = x\noutputs = x + 7.0",
    )
    .unwrap();
    assert_eq!(
        run(&["stablehlo"], &[&archive]).stdout,
        run(&["stablehlo"], &[&literal]).stdout
    );
}

#[test]
fn directory_scope_is_part_of_module_identity() {
    let dir = Scratch::new("directory-scope");
    fs::create_dir(dir.path("sub")).unwrap();
    fs::write(dir.path("outside.flatppl"), "value = 3").unwrap();
    fs::write(
        dir.path("sub/main.flatppl"),
        "outside = load_module(\"../outside.flatppl\")\nvalue = outside.value",
    )
    .unwrap();
    let main = dir.path("main.flatppl");
    let direct = "a = load_module(\"sub/main.flatppl\")\nvalue = a.value\n";
    fs::write(&main, direct).unwrap();
    run(&["infer"], &[&dir.path(""), &dir.path("direct.flatpir")]);

    fs::write(&main, format!("{direct}b = load_module(\"sub\")\n")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(dir.path(""))
        .arg(dir.path("narrow.flatpir"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!dir.path("narrow.flatpir").exists());
}

#[test]
#[cfg(feature = "fmtlint")]
fn archive_format_checks_and_lint_do_not_edit_the_container() {
    let dir = Scratch::new("archive-format");
    let bytes = flatppl_bundle::pack(
        [("main.flatppl".into(), b"x = 1\n".to_vec())],
        flatppl_bundle::Compression::Stored,
    )
    .unwrap();
    let archive = dir.path("model.zip");
    fs::write(&archive, &bytes).unwrap();
    run(&["fmt", "--check"], &[&archive]);
    run(&["lint"], &[&archive]);
    let edit = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("fmt")
        .arg(&archive)
        .output()
        .unwrap();
    assert!(!edit.status.success());
    assert_eq!(fs::read(&archive).unwrap(), bytes);
}
