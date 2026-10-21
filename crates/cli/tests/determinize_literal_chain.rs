//! A chain of literal bindings folds in one canonicalization sweep, so
//! determinizing it takes time linear in the chain length.

use std::fs;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

mod common;
use common::Scratch;

#[test]
fn a_long_literal_chain_determinizes_within_the_deadline() {
    let dir = Scratch::new("determinize-literal-chain");
    let count = 20_000;
    let mut bindings = vec!["x0 = 1.0".to_string()];
    bindings.extend((1..=count).map(|i| format!("x{i} = x{} + 1.0", i - 1)));
    for order in ["forward", "reverse"] {
        let input = dir.path(&format!("chain-{order}.flatppl"));
        let query = format!("lp = logdensityof(Normal(mu = x{count}, sigma = 1.0), 0.5)");
        let mut ordered = bindings.clone();
        if order == "reverse" {
            ordered.reverse();
        }
        fs::write(&input, format!("{}\n{query}\n", ordered.join("\n"))).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("determinize")
            .arg(&input)
            .arg("--keep")
            .arg("lp")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("{order}: determinizing {count} literal bindings exceeded 20 s");
            }
            thread::sleep(Duration::from_millis(10));
        };
        let stderr = std::io::read_to_string(child.stderr.take().unwrap()).unwrap();
        let stdout = std::io::read_to_string(child.stdout.take().unwrap()).unwrap();
        assert!(status.success(), "{order}: {stderr}");
        // x0 = 1.0 plus 20 000 increments of 1.0, exact in f64.
        assert_eq!(
            stdout.trim(),
            "lp = builtin_logdensityof(Normal, record(mu = 20001.0, sigma = 1.0), 0.5)",
            "{order}"
        );
    }
}
