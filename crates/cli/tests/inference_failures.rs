//! A `%failed` projection is an inference error, never a successful artifact.

use std::fs;
use std::process::Command;

mod common;
use common::Scratch;

struct FixedModule;

impl flatppl_fileaccess::Fetcher for FixedModule {
    fn fetch(&self, url: &str) -> Result<flatppl_fileaccess::Fetched, String> {
        Ok(flatppl_fileaccess::Fetched {
            bytes: b"present = 1\n".to_vec(),
            resolved_url: url.to_string(),
            content_type: Some("text/plain".to_string()),
            etag: None,
            last_modified: None,
        })
    }
}

#[test]
fn static_projection_failures_exit_with_an_error() {
    let dir = Scratch::new("infer-projection-failure");
    for (name, source, message) in [
        ("tuple", "x = (1, 2)[3]\n", "tuple index out of range"),
        ("record", "x = record(a = 1).b\n", "record has no field `b`"),
        (
            "table",
            "x = table(a = [1, 2]).b\n",
            "table has no column `b`",
        ),
    ] {
        let input = dir.path(&format!("{name}.flatppl"));
        let output = dir.path(&format!("{name}.flatpir"));
        fs::write(&input, source).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("infer")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert_eq!(result.status.code(), Some(1), "{name}: {stderr}");
        assert!(stderr.contains(message), "{name}: {stderr}");
        assert!(!output.exists(), "{name}: invalid output was published");
    }
}

#[test]
fn addaxes_refuses_rank_beyond_the_resource_limit() {
    let dir = Scratch::new("infer-addaxes-rank");
    for (count, succeeds) in [(63u32, true), (64u32, false), (u32::MAX, false)] {
        let input = dir.path(&format!("rank-{count}.flatppl"));
        let output = dir.path(&format!("rank-{count}.flatpir"));
        fs::write(&input, format!("x = addaxes([1], {count}, 0)\n")).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("infer")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        if succeeds {
            assert!(result.status.success(), "{stderr}");
        } else {
            assert_eq!(result.status.code(), Some(1), "{stderr}");
            assert!(stderr.contains("structural rank"), "{stderr}");
            assert!(stderr.contains("resource guard"), "{stderr}");
            assert!(!output.exists(), "oversized rank output was published");
        }
    }
}

#[test]
fn load_data_without_a_valueset_is_not_published() {
    let dir = Scratch::new("infer-load-data-arity");
    let input = dir.path("bad.flatppl");
    let output = dir.path("bad.flatpir");
    fs::write(&input, "d = load_data(\"x.csv\")\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("takes 2 arguments"), "{stderr}");
    assert!(!output.exists(), "invalid output was published");
}

#[test]
fn remote_module_inference_errors_redact_url_userinfo() {
    let dir = Scratch::new("infer-remote-userinfo");
    let cache_dir = dir.path("cache");
    let input = dir.path("root.flatppl");
    let output = dir.path("root.flatpir");
    let url = "https://fixture-user:p%40ss@host.test/dep@name.flatppl?email=a@b";

    flatppl_fileaccess::Cache::new(cache_dir.clone(), false, true)
        .get(url, &FixedModule, &flatppl_fileaccess::DenyAll)
        .unwrap();
    fs::write(
        &input,
        format!("dep = load_module(\"{url}\")\nx = dep.absent\n"),
    )
    .unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .env("FLATPPL_CACHEDIR", cache_dir)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);

    assert_eq!(result.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains(
            "module `https://<redacted>@host.test/dep@name.flatppl?email=a@b` has no binding `absent`"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains("fixture-user"), "{stderr}");
    assert!(!stderr.contains("p%40ss"), "{stderr}");
    assert!(!output.exists(), "invalid output was published");
}

#[test]
fn a_directory_is_refused_as_an_inference_input() {
    let dir = Scratch::new("infer-directory-input");
    let input = dir.path("source.flatppl");
    let output = dir.path("out.flatpir");
    fs::create_dir(&input).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);

    assert_eq!(result.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("not a regular file"), "{stderr}");
    assert!(!output.exists(), "invalid output was published");
}

#[cfg(unix)]
#[test]
fn special_inputs_are_refused_but_regular_symlinks_are_allowed() {
    use std::os::unix::fs::symlink;
    use std::process::Stdio;
    use std::thread;
    use std::time::{Duration, Instant};

    let dir = Scratch::new("infer-special-input");
    let target = dir.path("target.flatppl");
    let link = dir.path("linked.flatppl");
    let linked_output = dir.path("linked.flatpir");
    fs::write(&target, "x = 1\n").unwrap();
    symlink(&target, &link).unwrap();
    let linked = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&link)
        .arg(&linked_output)
        .output()
        .unwrap();
    assert!(
        linked.status.success(),
        "{}",
        String::from_utf8_lossy(&linked.stderr)
    );

    let fifo = dir.path("blocking.flatppl");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let fifo_output = dir.path("fifo.flatpir");
    let mut child = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&fifo)
        .arg(&fifo_output)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("infer blocked while opening a FIFO source");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stderr = child.stderr.take().unwrap();
    let stderr = std::io::read_to_string(stderr).unwrap();

    assert_eq!(status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("not a regular file"), "{stderr}");
    assert!(!fifo_output.exists(), "invalid output was published");
}
