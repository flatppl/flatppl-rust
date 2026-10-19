use std::io::{Cursor, Read, Write};
use std::sync::Arc;

use flatppl_bundle::{Bundle, Compression, Limits, MemberKind, pack};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

const PORTABLE: &[u8] = include_bytes!("fixtures/portable.zip");

fn raw_zip(files: &[(&str, &[u8])], links: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for (name, bytes) in files {
        writer.start_file(*name, options).unwrap();
        writer.write_all(bytes).unwrap();
    }
    for (name, target) in links {
        writer.add_symlink(*name, *target, options).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn open(bytes: Vec<u8>) -> flatppl_bundle::Result<Bundle> {
    Bundle::open("test.zip", bytes.into(), Limits::default())
}

#[test]
fn reads_independent_archive_bytes_links_and_nested_members() {
    let bundle = Bundle::open(
        "https://example.test/model.zip?x=1",
        PORTABLE.into(),
        Limits::default(),
    )
    .unwrap();
    assert_eq!(&*bundle.read("data/raw.bin").unwrap(), &[0, 1, 255, 128]);
    assert_eq!(
        &*bundle.read("data/observed-link.json").unwrap(),
        b"[1, 2]\n"
    );
    assert_eq!(
        bundle
            .symlink_target("data/observed-link.json")
            .unwrap()
            .as_deref(),
        Some("observed.json")
    );
    assert_eq!(
        bundle
            .resolve("modules/prior.flatppl", "../data/observed.json")
            .unwrap(),
        "data/observed.json"
    );
    assert!(
        bundle
            .members()
            .any(|member| member.path == "data/raw.bin" && member.kind == MemberKind::File)
    );
    let child = bundle.nested("child.zip").unwrap();
    assert_eq!(
        &*child.read(child.entrypoint()).unwrap(),
        b"offset = external(reals)\nvalue = offset + 2\n"
    );
    assert_ne!(
        bundle.identity_for("main.flatppl"),
        child.identity_for("main.flatppl")
    );
    assert!(Arc::ptr_eq(
        &bundle.read("data/raw.bin").unwrap(),
        &bundle.read("data/raw.bin").unwrap()
    ));
}

#[test]
fn wrapped_archives_and_directory_imports_keep_distinct_roots() {
    let bundle = open(raw_zip(
        &[
            ("wrapper/main.flatppl", b"m = load_module(\"sub\")"),
            ("wrapper/sub/main.flatppl", b"x = 3"),
            ("wrapper/data.json", b"[1]"),
        ],
        &[],
    ))
    .unwrap();
    assert_eq!(
        &*bundle.read("main.flatppl").unwrap(),
        b"m = load_module(\"sub\")"
    );
    assert_eq!(
        bundle.identity_for("main.flatppl"),
        "flatppl-zip:test.zip!/wrapper/main.flatppl"
    );
    let sub = bundle.subdirectory("sub").unwrap();
    assert_eq!(&*sub.read("main.flatppl").unwrap(), b"x = 3");
    assert!(sub.resolve("main.flatppl", "../data.json").is_err());
    assert_ne!(
        sub.identity_for("main.flatppl"),
        bundle.identity_for("sub/main.flatppl")
    );
    assert_eq!(
        bundle
            .subdirectory(".")
            .unwrap()
            .identity_for("main.flatppl"),
        bundle.identity_for("main.flatppl")
    );
}

#[test]
fn root_entry_takes_precedence_and_wrapper_selection_is_unambiguous() {
    let both = open(raw_zip(
        &[
            ("main.flatppl", b"root"),
            ("wrapped/main.flatppl", b"wrapped"),
        ],
        &[],
    ))
    .unwrap();
    assert_eq!(&*both.read(both.entrypoint()).unwrap(), b"root");
    assert!(
        open(raw_zip(
            &[("a/main.flatppl", b"a"), ("b/main.flatppl", b"b")],
            &[]
        ))
        .is_err()
    );
    assert!(
        open(raw_zip(
            &[("a/main.flatppl", b"a"), ("README", b"extra")],
            &[]
        ))
        .is_err()
    );
    assert!(open(raw_zip(&[("other.flatppl", b"no main")], &[])).is_err());
}

#[test]
fn encrypted_or_unsupported_compression_is_not_exposed() {
    let original = raw_zip(&[("main.flatppl", b"x=1")], &[]);
    let central = original
        .windows(4)
        .position(|window| window == b"PK\x01\x02")
        .unwrap();
    let mut encrypted = original.clone();
    encrypted[6] |= 1;
    encrypted[central + 8] |= 1;
    assert!(open(encrypted).is_err());
    let mut unknown = original;
    unknown[8..10].copy_from_slice(&199u16.to_le_bytes());
    unknown[central + 10..central + 12].copy_from_slice(&199u16.to_le_bytes());
    assert!(open(unknown).is_err());
}

#[test]
fn symlinks_resolve_before_parent_traversal_and_stay_contained() {
    let files: &[(&str, &[u8])] = &[
        ("main.flatppl", b"x=1"),
        ("dir/deep/file", b"ok"),
        ("dir/data", b"data"),
    ];
    let bundle = open(raw_zip(files, &[("link", "dir/deep")])).unwrap();
    assert_eq!(&*bundle.read("link/../data").unwrap(), b"data");
    for links in [
        vec![("bad", "../outside")],
        vec![("bad", "/outside")],
        vec![("bad", "loop"), ("loop", "bad")],
    ] {
        assert!(open(raw_zip(files, &links)).is_err());
    }
}

#[test]
fn long_paths_and_link_targets_have_bounded_linear_resolution() {
    let path = format!("{}x", "a/".repeat(32_000));
    let bytes = raw_zip(
        &[("main.flatppl", b"x=1"), (&path, b"deep")],
        &[("link", &path)],
    );
    let bundle = open(bytes).unwrap();
    assert_eq!(&*bundle.read("link").unwrap(), b"deep");
    assert_eq!(&*bundle.read("link/../x").unwrap(), b"deep");
    assert!(bundle.is_dir(&path[..path.len() - 2]).unwrap());

    let limits = Limits {
        path_bytes: 32,
        ..Limits::default()
    };
    let files = [("main.flatppl", b"x=1".as_slice())];
    let target = format!("{}main.flatppl", "./".repeat(10));
    let bytes = raw_zip(&files, &[("link", &target)]);
    let bundle = Bundle::open("bounded", bytes.into(), limits).unwrap();
    assert_eq!(&*bundle.read("link").unwrap(), b"x=1");
    let oversized = raw_zip(&files, &[("link", &format!("./{target}"))]);
    assert!(Bundle::open("bounded", oversized.into(), limits).is_err());
}

#[test]
fn rejects_ambiguous_or_escaping_archive_names_before_loading() {
    for name in [
        "../escape",
        "a/../escape",
        "/absolute",
        "C:/absolute",
        "a\\b",
    ] {
        assert!(open(raw_zip(&[("main.flatppl", b"x=1"), (name, b"bad")], &[])).is_err());
    }
    for files in [
        vec![
            ("main.flatppl", b"x=1".as_slice()),
            ("a", b"file"),
            ("a/b", b"bad"),
        ],
        vec![
            ("main.flatppl", b"x=1".as_slice()),
            ("a/./b", b"one"),
            ("a/b", b"two"),
        ],
    ] {
        assert!(open(raw_zip(&files, &[])).is_err());
    }
    // Two equal-length names let us forge an actual central-directory duplicate.
    let mut bytes = raw_zip(
        &[("main.flatppl", b"x=1"), ("aaaa", b"one"), ("bbbb", b"two")],
        &[],
    );
    for index in 0..bytes.len() - 3 {
        if &bytes[index..index + 4] == b"bbbb" {
            bytes[index..index + 4].copy_from_slice(b"aaaa");
        }
    }
    assert!(open(bytes).is_err());
}

#[test]
fn crc_and_decoded_size_checks_reject_modified_content() {
    let mut bytes = raw_zip(&[("main.flatppl", b"x=123456789")], &[]);
    let offset = bytes
        .windows(11)
        .position(|window| window == b"x=123456789")
        .unwrap();
    bytes[offset + 2] = b'0';
    let bundle = open(bytes).unwrap();
    assert!(bundle.read("main.flatppl").is_err());

    let mut bytes = raw_zip(&[("main.flatppl", b"x=123456789")], &[]);
    let central = bytes
        .windows(4)
        .position(|window| window == b"PK\x01\x02")
        .unwrap();
    // Forge matching local and central sizes, so the actual stream must detect it.
    bytes[22..26].copy_from_slice(&2u32.to_le_bytes());
    bytes[central + 24..central + 28].copy_from_slice(&2u32.to_le_bytes());
    let bundle = open(bytes).unwrap();
    assert!(bundle.read("main.flatppl").is_err());
}

#[test]
fn nested_archives_share_byte_member_and_depth_limits() {
    let child = raw_zip(&[("main.flatppl", b"x=123456789")], &[]);
    let outer = raw_zip(&[("main.flatppl", b"x=1"), ("child.zip", &child)], &[]);
    let limits = Limits {
        total_bytes: child.len() as u64 + 3,
        ..Limits::default()
    };
    let bundle = Bundle::open("outer", outer.clone().into(), limits).unwrap();
    bundle.read("main.flatppl").unwrap();
    let nested = bundle.nested("child.zip").unwrap();
    assert!(nested.read("main.flatppl").is_err());
    let no_depth = Limits {
        depth: 0,
        ..Limits::default()
    };
    assert!(
        Bundle::open("outer", outer.clone().into(), no_depth)
            .unwrap()
            .nested("child.zip")
            .is_err()
    );
    let count = Limits {
        members: 2,
        ..Limits::default()
    };
    assert!(
        Bundle::open("outer", outer.into(), count)
            .unwrap()
            .nested("child.zip")
            .is_err()
    );
    let small = Limits {
        member_bytes: 2,
        ..Limits::default()
    };
    assert!(Bundle::open("child", child.into(), small).is_err());
}

#[test]
fn zip64_counts_obey_the_same_member_boundary() {
    let original = raw_zip(&[("main.flatppl", b"x=1"), ("data", b"value")], &[]);
    let footer = original.len() - 22;
    let mut end = original[footer..].to_vec();
    let size = u32::from_le_bytes(end[12..16].try_into().unwrap());
    let offset = u32::from_le_bytes(end[16..20].try_into().unwrap());
    let mut bytes = original[..footer].to_vec();
    bytes.extend_from_slice(b"PK\x06\x06");
    bytes.extend_from_slice(&44u64.to_le_bytes());
    bytes.extend_from_slice(&45u16.to_le_bytes());
    bytes.extend_from_slice(&45u16.to_le_bytes());
    bytes.extend_from_slice(&[0; 8]);
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&u64::from(size).to_le_bytes());
    bytes.extend_from_slice(&u64::from(offset).to_le_bytes());
    bytes.extend_from_slice(b"PK\x06\x07");
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(footer as u64).to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    end[8..20].fill(255);
    bytes.extend(end);
    let limits = Limits {
        members: 2,
        ..Limits::default()
    };
    let bundle = Bundle::open("zip64", bytes.clone().into(), limits).unwrap();
    assert_eq!(&*bundle.read("data").unwrap(), b"value");
    let limits = Limits {
        members: 1,
        ..limits
    };
    assert!(Bundle::open("zip64", bytes.into(), limits).is_err());
}

#[test]
fn writer_keeps_exact_bytes_and_ignores_input_order() {
    let first = vec![
        ("main.flatppl".into(), b"x=1".to_vec()),
        ("data.bin".into(), vec![0, 255, 128]),
    ];
    let mut reversed = first.clone();
    reversed.reverse();
    let bytes = pack(first, Compression::Deflated).unwrap();
    assert_eq!(bytes, pack(reversed, Compression::Deflated).unwrap());
    let mut zip = ZipArchive::new(Cursor::new(bytes)).unwrap();
    let mut data = Vec::new();
    zip.by_name("data.bin")
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();
    assert_eq!(data, vec![0, 255, 128]);
}
