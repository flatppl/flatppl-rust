//! Shared test helpers for the CLI integration tests.
//!
//! Lives under `tests/common/` so cargo does not compile it as its own test
//! binary; each test file pulls it in with `mod common;`.

use std::fs;
use std::path::PathBuf;

/// A scratch dir unique to this test process, cleaned up on drop.
pub struct Scratch(PathBuf);

impl Scratch {
    /// Create a scratch dir named for `label` and this process id (so the three
    /// CLI test binaries never collide, even with shared labels).
    pub fn new(label: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("flatppl-cli-test-{label}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}
