#![cfg(all(feature = "determinize", feature = "stablehlo"))]

use std::process::Command;

fn run(arguments: &[&std::ffi::OsStr]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn cli_emits_portable_preparation_and_supports_the_reduction_switch() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-statistics-cli-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("query.flatppl");
    std::fs::write(&source, "mu=elementof(reals)\ny=external(cartpow(reals,3))\ninputs=(mu,y)\noutputs=logdensityof(iid(Normal(mu,1.0),3),y)").unwrap();
    for (form, extension) in [("flatppl", "flatppl"), ("flatpir", "flatpir")] {
        let prep = directory.join(format!("prepare.{extension}"));
        let score = run(&[
            "determinize".as_ref(),
            source.as_os_str(),
            "--emit".as_ref(),
            form.as_ref(),
            "--prepare-data".as_ref(),
            prep.as_os_str(),
            "--data-input".as_ref(),
            "y".as_ref(),
        ]);
        for text in [score, std::fs::read_to_string(prep).unwrap()] {
            let mut module = if form == "flatpir" {
                flatppl_flatpir::read(&text).unwrap()
            } else {
                flatppl_syntax::parse(&text).unwrap()
            };
            flatppl_infer::infer(&mut module);
            assert!(flatppl_determinizer::is_flatpdl(&module).is_ok());
        }
    }
    let prep = directory.join("prepare.mlir");
    let score = run(&[
        "stablehlo".as_ref(),
        source.as_os_str(),
        "--prepare-data".as_ref(),
        prep.as_os_str(),
        "--data-input".as_ref(),
        "y".as_ref(),
        "--dtype".as_ref(),
        "f64".as_ref(),
    ]);
    assert!(!score.contains("tensor<3xf64>"));
    assert!(
        std::fs::read_to_string(prep)
            .unwrap()
            .contains("tensor<3xf64>")
    );

    std::fs::write(
        &source,
        "mu=elementof(reals)\ninputs=mu\noutputs=logdensityof(iid(Normal(mu,1.0),3),[0.0,1.0,2.0])",
    )
    .unwrap();
    let enabled = run(&["determinize".as_ref(), source.as_os_str()]);
    let disabled = run(&[
        "determinize".as_ref(),
        source.as_os_str(),
        "--sufficient-statistics=false".as_ref(),
    ]);
    assert!(enabled.contains("__likelihood_summary_"));
    assert!(!disabled.contains("__likelihood_summary_"));
    let disabled = run(&[
        "stablehlo".as_ref(),
        source.as_os_str(),
        "--sufficient-statistics=false".as_ref(),
    ]);
    assert!(disabled.contains("stablehlo.reduce"));
    std::fs::remove_dir_all(directory).unwrap();
}
