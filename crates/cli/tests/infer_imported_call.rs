use std::{fs, process::Command};

mod common;
use common::Scratch;

#[test]
fn imported_calls_are_typed_per_argument_and_module_substitution() {
    let dir = Scratch::new("infer-imported-call");
    fs::write(
        dir.path("helper.flatppl"),
        "scale = external(anything)\nf(x) = x + x\ng(x) = record(x = x, scale = scale)\n",
    )
    .unwrap();
    let input = dir.path("main.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(&input, "L = load_module(\"helper.flatppl\", scale = 2.0)\nM = load_module(\"helper.flatppl\", scale = 2)\na = L.f([1.0, 2.0])\nb = L.f(x = [1.0, 2.0, 3.0])\nc = M.f([1, 2, 3, 4])\nd = L.f.([1.0, 2.0, 3.0])\nr = L.g(1)\ns = M.g(1)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["infer", "--level=type"])
        .arg(input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let out = fs::read_to_string(output).unwrap();
    for (name, size, scalar) in [
        ("a", 2, "real"),
        ("b", 3, "real"),
        ("c", 4, "integer"),
        ("d", 3, "real"),
    ] {
        let line = out
            .lines()
            .find(|line| line.contains(&format!("(%bind {name} ")))
            .unwrap();
        assert!(
            line.contains(&format!(
                "(%bind {name} (%meta ((%array 1 ({size}) (%scalar {scalar}))"
            )),
            "{line}"
        );
    }
    for (name, scalar) in [("r", "real"), ("s", "integer")] {
        let line = out
            .lines()
            .find(|line| line.contains(&format!("(%bind {name} ")))
            .unwrap();
        assert!(
            line.contains(&format!("(scale (%scalar {scalar}))")),
            "{line}"
        );
    }
}

#[test]
fn callable_reexports_keep_argument_domains_and_load_substitutions() {
    let dir = Scratch::new("infer-callable-reexport");
    fs::write(
        dir.path("helper.flatppl"),
        "config = external(anything)\nf(x) = record(value = x, config = config)\n",
    )
    .unwrap();
    fs::write(
        dir.path("wrapper.flatppl"),
        "config = external(anything)\nM = load_module(\"helper.flatppl\", config = config)\nf = M.f\nalias = f\n",
    )
    .unwrap();
    let input = dir.path("main.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(&input, "A = load_module(\"wrapper.flatppl\", config = 1.0)\nB = load_module(\"wrapper.flatppl\", config = 2)\np = elementof(posreals)\nq = elementof(nonnegreals)\na = A.f(p)\nb = B.alias([1.0,2.0])\nc = A.alias(q)\nd = B.f([1,2,3])\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["infer", "--level=shape"])
        .arg(input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let out = fs::read_to_string(output).unwrap();
    let line = |name: &str| {
        out.lines()
            .find(|line| line.contains(&format!("(%bind {name} ")))
            .unwrap()
    };
    for (name, domain) in [("a", "posreals"), ("c", "nonnegreals")] {
        let line = line(name);
        assert!(line.contains(&format!("(%field value {domain})")), "{line}");
        assert!(line.contains("(config (%scalar real))"), "{line}");
    }
    for (name, size, scalar) in [("b", 2, "real"), ("d", 3, "integer")] {
        let line = line(name);
        assert!(
            line.contains(&format!("(value (%array 1 ({size}) (%scalar {scalar})))")),
            "{line}"
        );
        assert!(line.contains("(config (%scalar integer))"), "{line}");
    }
}

#[test]
fn imported_calls_still_report_invalid_arity() {
    let dir = Scratch::new("infer-imported-arity");
    fs::write(dir.path("helper.flatppl"), "f(a, b) = a + b\n").unwrap();
    let input = dir.path("main.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(&input, "L = load_module(\"helper.flatppl\")\ny = L.f(1)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(input)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!output.exists());
}

#[test]
fn imported_one_input_kernel_binds_a_record_whole() {
    let dir = Scratch::new("infer-imported-kernel-record");
    fs::write(
        dir.path("helper.flatppl"),
        "x = elementof(reals)\nk = functionof(Normal(mu = x, sigma = 1.0), x = x)\n",
    )
    .unwrap();
    let input = dir.path("main.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(
        &input,
        "L = load_module(\"helper.flatppl\")\ny = L.k(record(x = 1.0))\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(input)
        .arg(output)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("bound whole"), "{stderr}");
}
