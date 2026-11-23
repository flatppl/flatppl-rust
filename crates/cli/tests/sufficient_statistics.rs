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

#[test]
fn cli_packing_switch_selects_the_preparation_abi() {
    let directory = std::env::temp_dir().join(format!("flatppl-pack-cli-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("query.flatppl");
    let prep = directory.join("prepare.mlir");
    std::fs::write(&source,"a=elementof(cartpow(posreals,2))\nb=elementof(posreals)\ny=external(cartpow(interval(0.1,10),4))\ng=[1,2,1,2]\ninputs=(a,b,y)\noutputs=logdensityof(Gamma.(a[g],b),y)").unwrap();
    for (flag, arguments) in [("--pack-summaries=true", 3), ("--pack-summaries=false", 8)] {
        let score = run(&[
            "stablehlo".as_ref(),
            source.as_os_str(),
            "--prepare-data".as_ref(),
            prep.as_os_str(),
            "--data-input".as_ref(),
            "y".as_ref(),
            flag.as_ref(),
        ]);
        let signature = score
            .lines()
            .find(|line| line.contains("func.func @logdensity("))
            .unwrap();
        assert_eq!(signature.matches("%arg").count(), arguments, "{signature}");
        let prepared = std::fs::read_to_string(&prep).unwrap();
        let signature = prepared
            .lines()
            .find(|line| line.contains("func.func @logdensity("))
            .unwrap();
        assert_eq!(
            signature.contains("tensor<6xf32>"),
            arguments == 3,
            "{signature}"
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn explanations_cover_both_passes_and_leave_every_output_byte_unchanged() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-reduction-reports-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("query.flatppl");
    let prep = directory.join("prepare.out");
    std::fs::write(directory.join("counts.json"), "[0,1,3]").unwrap();
    for (model, preparation, expected) in [
        (
            "a=elementof(reals)\ninputs=a\noutputs=logdensityof(iid(Normal(a,1.0),3),[0.0,1.0,2.0])",
            false,
            "reduced by scalar moments",
        ),
        (
            "a=elementof(interval(0.01,0.99))\ny=load_data(\"counts.json\",cartpow(nonnegintegers,3))\ninputs=a\noutputs=logdensityof(iid(NegativeBinomial(1e6,a),3),y)",
            false,
            "materialized data",
        ),
        (
            "a=elementof(cartpow(reals,3))\ny=external(cartpow(reals,5))\ng=[1,2,1,2,3]\ninputs=(a,y)\noutputs=logdensityof(Normal.(a[g],1.0),y)",
            true,
            "5 observations, 3 occupied groups, 1 singletons",
        ),
        (
            "a=elementof(posreals)\ny=external(cartpow(nonnegintegers,3))\ninputs=(a,y)\noutputs=logdensityof(iid(NegativeBinomial(1e6,a),3),y)",
            true,
            "retained:",
        ),
    ] {
        std::fs::write(&source, model).unwrap();
        for verb in [
            vec!["determinize"],
            vec!["determinize", "--emit", "flatpir"],
            vec!["stablehlo", "--dtype", "f64"],
        ] {
            let mut args = verb
                .iter()
                .map(std::ffi::OsString::from)
                .collect::<Vec<_>>();
            args.push(source.clone().into_os_string());
            if preparation {
                args.extend([
                    "--prepare-data".into(),
                    prep.clone().into_os_string(),
                    "--data-input".into(),
                    "y".into(),
                ]);
            }
            let plain = Command::new(env!("CARGO_BIN_EXE_flatppl"))
                .args(&args)
                .output()
                .unwrap();
            let prepared = plain
                .status
                .success()
                .then(|| std::fs::read(&prep).ok())
                .flatten();
            args.push("--explain-reductions".into());
            let traced = Command::new(env!("CARGO_BIN_EXE_flatppl"))
                .args(&args)
                .output()
                .unwrap();
            assert_eq!(plain.status.code(), traced.status.code());
            assert_eq!(plain.stdout, traced.stdout);
            if let Some(prepared) = prepared {
                assert_eq!(prepared, std::fs::read(&prep).unwrap());
            }
            let report = String::from_utf8(traced.stderr).unwrap();
            assert!(report.contains(expected), "{model}\n{report}");
            if !preparation {
                assert!(plain.status.success(), "{report}");
            }
        }
    }
    std::fs::write(
        &source,
        "a=elementof(reals)\ninputs=a\noutputs=logdensityof(iid(Normal(a,1.0),3),[0.0,1.0,2.0])",
    )
    .unwrap();
    let traced = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args([
            "stablehlo",
            "--sufficient-statistics=false",
            "--explain-reductions",
        ])
        .arg(&source)
        .output()
        .unwrap();
    assert!(traced.status.success());
    assert!(
        String::from_utf8(traced.stderr)
            .unwrap()
            .contains("disabled")
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn fixed_category_files_fold_after_loading_and_promoted_labels_stay_live() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-file-labels-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("query.flatppl");
    std::fs::write(directory.join("labels.json"), "[0,1,2,0,1,2,0]").unwrap();
    for (inputs, live) in [("p", false), ("(p,y)", true)] {
        let query = format!(
            "p=elementof(stdsimplex(3))\nt=elementof(stdsimplex(3))\ny=load_data(\"labels.json\",cartpow(integers,7))\nL=likelihoodof(functionof(iid(Categorical0(t),7),t=t),y)\nP=bayesupdate(L,joint(t=Dirichlet([1.,1.,1.])))\ninputs={inputs}\noutputs=logdensityof(normalize(P),record(t=p))"
        );
        std::fs::write(&source, query).unwrap();
        let flatpdl = run(&[
            "determinize".as_ref(),
            "--keep".as_ref(),
            "inputs".as_ref(),
            "--keep".as_ref(),
            "outputs".as_ref(),
            source.as_os_str(),
        ]);
        assert_eq!(flatpdl.contains("bincounts("), live, "{flatpdl}");
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn fixed_record_groups_reduce_after_loading_and_promoted_records_stay_live() {
    let directory =
        std::env::temp_dir().join(format!("flatppl-record-groups-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("query.flatppl");
    std::fs::write(
        directory.join("data.json"),
        r#"{"rows":{"g":[2,1,1,3,2],"y":[4.0,0.0,2.0,1.0,3.0]}}"#,
    )
    .unwrap();
    for (inputs, fixed) in [("mu", true), ("(mu,d)", false)] {
        let query = format!(
            "mu=elementof(cartpow(reals,4))\nd=load_data(\"data.json\",cartprod(rows=cartprod(g=cartpow(posintegers,5),y=cartpow(reals,5))))\nrows=d.rows\ninputs={inputs}\noutputs=logdensityof(Normal.(mu[rows.g],1.0),rows.y)"
        );
        std::fs::write(&source, query).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .args(["determinize", "--explain-reductions"])
            .arg(&source)
            .output()
            .unwrap();
        let report = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{report}");
        assert_eq!(
            report.contains("5 observations, 3 occupied groups, 1 singletons"),
            fixed,
            "{report}"
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}
