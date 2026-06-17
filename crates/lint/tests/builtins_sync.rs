//! Guard `flatppl_lint::builtins::BUILTINS` against drift from the authoritative
//! `flatppl-grammars/keyword-lists.json`. Skipped (passes) when the sibling repo
//! is not checked out, so CI without it still builds.

use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn builtins_match_keyword_lists() {
    // Resolve relative to the crate manifest dir → workspace `../flatppl-grammars`.
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../flatppl-grammars/keyword-lists.json");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let expected: BTreeSet<String> = json["categories"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c["words"].as_array().unwrap())
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    let actual: BTreeSet<String> = flatppl_lint::test_builtins()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(actual, expected, "BUILTINS drifted from keyword-lists.json");
}

/// `shadows-builtin` looks up names with `binary_search`, which requires the
/// `BUILTINS` slice to be sorted and duplicate-free. This pins that invariant
/// independently of the sibling repo (always runs).
#[test]
fn builtins_sorted_and_unique() {
    let b = flatppl_lint::test_builtins();
    assert!(
        b.windows(2).all(|w| w[0] < w[1]),
        "BUILTINS must be strictly sorted for binary_search"
    );
}
