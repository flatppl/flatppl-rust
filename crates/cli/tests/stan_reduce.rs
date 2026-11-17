#![cfg(feature = "stan")]
//! `flatppl stan --reduce`: opt-in reduction families, off by default.

use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tests run in parallel threads: each call gets its own directory.
static CALLS: AtomicUsize = AtomicUsize::new(0);

const MODEL: &str = "flatppl_compat = \"0.1\"\n\
    samples = [0.25, -1.5, 2.0]\n\
    mu = elementof(reals)\n\
    sigma = elementof(posreals)\n\
    y ~ iid(Normal(mu, sigma), 3)\n\
    lik = likelihoodof(kernelof(record(y = y), mu = mu, sigma = sigma), record(y = samples))\n";

fn stan(extra: &[&str]) -> Output {
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("flatppl-stan-reduce-{}-{call}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("model.flatppl");
    std::fs::write(&source, MODEL).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["stan", "--target", "lik"])
        .args(extra)
        .arg(&source)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    out
}

fn code(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

#[test]
fn default_emits_no_summary() {
    let text = code(&stan(&[]));
    assert!(text.contains("samples ~ normal(mu, sigma);"), "{text}");
    assert!(!text.contains("_summarized"), "{text}");
    assert!(!text.contains("target +="), "{text}");
}

#[test]
fn sufficient_family_prepares_summaries_in_either_form() {
    let repeated = code(&stan(&["--reduce", "sufficient"]));
    assert!(repeated.contains("samples_summarized"), "{repeated}");
    let delimited = code(&stan(&["--reduce=sufficient,sufficient"]));
    assert_eq!(repeated, delimited);
}

#[test]
fn unknown_family_is_a_usage_error() {
    let out = stan(&["--reduce", "conjugate"]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("invalid value 'conjugate'"), "{err}");
    assert!(err.contains("sufficient"), "{err}");
}

#[test]
fn the_old_switch_is_gone() {
    let out = stan(&["--sufficient-statistics=false"]);
    assert_eq!(out.status.code(), Some(2));
}
