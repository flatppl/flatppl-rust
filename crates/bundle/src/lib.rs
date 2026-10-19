//! Portable FlatPPL ZIP bundles (§04, Module composition — Bundles).
//!
//! Hosts provide bytes and an opaque identity. Loading never extracts files or
//! performs I/O. Archive members decompress on demand and share a bounded cache,
//! including nested archives. [`pack`] writes portable, deterministic ZIP files.

mod path;
#[cfg(feature = "source")]
pub mod source;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::io::{Cursor, Read, Write};
use std::sync::{Arc, Mutex};

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use path::{PathTree, bounded, entry_path, join, percent};

/// Private host keys remain valid `file` source strings during IR reinference.
/// They identify mapped archive bytes, never files to read from the host OS.
pub const DATA_SOURCE_PREFIX: &str = "file:///__flatppl_bundle__/";

#[derive(Debug)]
pub struct Error(String);

impl Error {
    fn new(message: impl ToString) -> Self {
        Self(message.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Host policy, not language limits. Archive bytes are bounded before indexing;
/// decoded sizes before member allocation. Total bytes and counts include nested ZIPs.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub archive_bytes: u64,
    pub member_bytes: u64,
    pub total_bytes: u64,
    /// Checked after zip indexes central metadata; `archive_bytes` bounds its input first.
    pub members: usize,
    pub depth: usize,
    /// Maximum UTF-8 bytes in a member path, reference, or expanded link target.
    pub path_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            archive_bytes: 64 * 1024 * 1024,
            member_bytes: 128 * 1024 * 1024,
            total_bytes: 256 * 1024 * 1024,
            members: 10_000,
            depth: 8,
            path_bytes: u16::MAX.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    File,
    Directory,
    Symlink,
}

/// Metadata relative to the selected bundle root. A symlink's size is its target text.
#[derive(Clone, Debug)]
pub struct Member {
    pub path: String,
    pub size: u64,
    pub kind: MemberKind,
}

struct Entry {
    index: usize,
    member: Member,
}

#[derive(Default)]
struct Budget {
    bytes: u64,
    members: usize,
}

struct Reader {
    zip: ZipArchive<Cursor<Arc<[u8]>>>,
    decoded: HashMap<String, Arc<[u8]>>,
}

struct Archive {
    entries: BTreeMap<String, Entry>,
    paths: PathTree,
    reader: Mutex<Reader>,
    children: Mutex<HashMap<String, Bundle>>,
    budget: Arc<Mutex<Budget>>,
    limits: Limits,
    depth: usize,
}

struct View {
    archive: Arc<Archive>,
    identity: String,
    root: String,
    root_node: usize,
    identity_prefix: String,
    members: Vec<Member>,
}

/// An immutable archive view. Clones share decompressed bytes and resource limits.
#[derive(Clone)]
pub struct Bundle {
    inner: Arc<View>,
}

impl Bundle {
    pub fn open(identity: impl Into<String>, bytes: Arc<[u8]>, limits: Limits) -> Result<Self> {
        Self::open_with_budget(identity.into(), bytes, limits, 0, Arc::default())
    }

    fn open_with_budget(
        identity: String,
        bytes: Arc<[u8]>,
        limits: Limits,
        depth: usize,
        budget: Arc<Mutex<Budget>>,
    ) -> Result<Self> {
        if bytes.len() as u64 > limits.archive_bytes || depth > limits.depth {
            return Err(Error::new("ZIP archive exceeds its byte or nesting limit"));
        }
        let mut zip = ZipArchive::new(Cursor::new(bytes)).map_err(Error::new)?;
        {
            let mut budget = budget.lock().map_err(Error::new)?;
            if zip.len() > limits.members.saturating_sub(budget.members) {
                return Err(Error::new("ZIP archive exceeds its member-count limit"));
            }
            budget.members += zip.len();
        }
        // zip checks duplicate raw names, matching local/central headers, and overlaps.
        zip.validate_header_consistency().map_err(Error::new)?;
        let mut entries = BTreeMap::new();
        let mut paths = PathTree::default();
        let mut declared_bytes = 0u64;
        for index in 0..zip.len() {
            let file = zip.by_index_data(index).map_err(Error::new)?;
            // Check the raw path too: a Unicode extra field must not hide traversal.
            bounded(file.name_raw().len(), limits.path_bytes)?;
            entry_path(&String::from_utf8_lossy(file.name_raw()))?;
            let name = file.name().map_err(Error::new)?;
            bounded(name.len(), limits.path_bytes)?;
            let name = entry_path(&name)?;
            if file.encrypted()
                || !matches!(
                    file.compression(),
                    CompressionMethod::Stored | CompressionMethod::Deflated
                )
            {
                return Err(Error::new(format!(
                    "unsupported ZIP encryption or compression: {name}"
                )));
            }
            let mode = file.unix_mode().unwrap_or(0) & 0o170000;
            let kind = match mode {
                0 | 0o100000 if !file.is_dir() => MemberKind::File,
                0 | 0o040000 if file.is_dir() => MemberKind::Directory,
                0o120000 if !file.is_dir() => MemberKind::Symlink,
                _ => return Err(Error::new(format!("unsupported ZIP member type: {name}"))),
            };
            if kind == MemberKind::Symlink && file.size() > limits.path_bytes as u64 {
                return Err(Error::new(
                    "symbolic link target exceeds the path-byte limit",
                ));
            }
            declared_bytes = declared_bytes
                .checked_add(file.size())
                .ok_or_else(|| Error::new("ZIP member sizes overflow"))?;
            if file.size() > limits.member_bytes || declared_bytes > limits.total_bytes {
                return Err(Error::new("ZIP archive exceeds its decoded-byte limit"));
            }
            if kind == MemberKind::Directory && file.size() != 0 {
                return Err(Error::new(format!(
                    "ZIP directory has file content: {name}"
                )));
            }
            paths.insert(&name, kind)?;
            let member = Member {
                path: name.clone(),
                size: file.size(),
                kind,
            };
            if entries
                .insert(name.clone(), Entry { index, member })
                .is_some()
            {
                return Err(Error::new(format!("duplicate normalized ZIP path: {name}")));
            }
        }
        let root = bundle_root(entries.keys().map(String::as_str))?;
        let archive = Arc::new(Archive {
            entries,
            paths,
            reader: Mutex::new(Reader {
                zip,
                decoded: HashMap::new(),
            }),
            children: Mutex::default(),
            budget,
            limits,
            depth,
        });
        Self::view(archive, identity, root.clone(), root)
    }

    fn view(
        archive: Arc<Archive>,
        identity: String,
        root: String,
        identity_prefix: String,
    ) -> Result<Self> {
        let prefix = if root.is_empty() {
            String::new()
        } else {
            format!("{root}/")
        };
        let members = archive
            .entries
            .values()
            .filter_map(|entry| {
                let path = entry.member.path.strip_prefix(&prefix)?;
                if path.is_empty() {
                    return None;
                }
                Some(Member {
                    path: path.to_owned(),
                    ..entry.member.clone()
                })
            })
            .collect();
        let root_node = archive
            .paths
            .find(&root)
            .ok_or_else(|| Error::new("missing bundle root"))?;
        let bundle = Self {
            inner: Arc::new(View {
                archive,
                identity,
                root,
                root_node,
                identity_prefix,
                members,
            }),
        };
        for member in bundle
            .members()
            .filter(|member| member.kind == MemberKind::Symlink)
        {
            bundle.canonical(&member.path)?;
        }
        let entry = bundle.canonical(bundle.entrypoint())?;
        let raw = join(&bundle.inner.root, &entry);
        if !bundle
            .inner
            .archive
            .entries
            .get(&raw)
            .is_some_and(|e| e.member.kind == MemberKind::File)
        {
            return Err(Error::new("bundle main.flatppl does not resolve to a file"));
        }
        Ok(bundle)
    }

    pub fn entrypoint(&self) -> &str {
        "main.flatppl"
    }

    pub fn members(&self) -> impl Iterator<Item = &Member> {
        self.inner.members.iter()
    }

    /// A stable opaque ID for a resolved member path. Explicit import edges,
    /// not reparsing this ID, resolve modules.
    pub fn identity_for(&self, path: &str) -> String {
        format!(
            "flatppl-zip:{}!/{}",
            percent(&self.inner.identity, false),
            percent(&join(&self.inner.identity_prefix, path), true)
        )
    }

    /// A private data key for lowering. Hosts must map it to bytes and refuse
    /// unmapped keys. This avoids adding an archive scheme to the language.
    pub fn data_identity_for(&self, path: &str) -> String {
        format!(
            "{DATA_SOURCE_PREFIX}{}",
            percent(&self.identity_for(path), false)
        )
    }

    pub fn is_dir(&self, path: &str) -> Result<bool> {
        let canonical = self.canonical(path)?;
        Ok(self
            .inner
            .archive
            .paths
            .find(&join(&self.inner.root, &canonical))
            .is_some_and(|node| self.inner.archive.paths.is_dir(node)))
    }

    pub fn read(&self, path: &str) -> Result<Arc<[u8]>> {
        let canonical = self.canonical(path)?;
        self.inner.archive.read(&join(&self.inner.root, &canonical))
    }

    /// The literal target for explicit extraction. Links remain relative to their parent.
    pub fn symlink_target(&self, path: &str) -> Result<Option<String>> {
        bounded(path.len(), self.inner.archive.limits.path_bytes)?;
        let raw = join(&self.inner.root, &entry_path(path)?);
        if !self.inner.archive.is_symlink(&raw) {
            return Ok(None);
        }
        let bytes = self.inner.archive.read(&raw)?;
        std::str::from_utf8(&bytes)
            .map(|value| Some(value.to_owned()))
            .map_err(Error::new)
    }

    /// Select a directory bundle with its own containment boundary and identity.
    pub fn subdirectory(&self, path: &str) -> Result<Self> {
        let canonical = self.canonical(path)?;
        if canonical.is_empty() {
            return Ok(self.clone());
        }
        if !self.is_dir(&canonical)? {
            return Err(Error::new(format!("not a bundle directory: {path}")));
        }
        Self::view(
            self.inner.archive.clone(),
            self.identity_for(&canonical),
            join(&self.inner.root, &canonical),
            String::new(),
        )
    }

    /// Open a nested archive without resetting its ancestor's resource budgets.
    pub fn nested(&self, path: &str) -> Result<Self> {
        let canonical = self.canonical(path)?;
        let identity = self.identity_for(&canonical);
        let mut children = self.inner.archive.children.lock().map_err(Error::new)?;
        if let Some(bundle) = children.get(&identity) {
            return Ok(bundle.clone());
        }
        let bundle = Self::open_with_budget(
            identity.clone(),
            self.read(&canonical)?,
            self.inner.archive.limits,
            self.inner.archive.depth + 1,
            self.inner.archive.budget.clone(),
        )?;
        children.insert(identity, bundle.clone());
        Ok(bundle)
    }
}

impl Archive {
    fn is_symlink(&self, path: &str) -> bool {
        self.entries
            .get(path)
            .is_some_and(|entry| entry.member.kind == MemberKind::Symlink)
    }

    fn read(&self, path: &str) -> Result<Arc<[u8]>> {
        let entry = self
            .entries
            .get(path)
            .ok_or_else(|| Error::new(format!("missing bundle member: {path}")))?;
        if entry.member.kind == MemberKind::Directory {
            return Err(Error::new(format!("bundle member is a directory: {path}")));
        }
        let mut reader = self.reader.lock().map_err(Error::new)?;
        if let Some(bytes) = reader.decoded.get(path) {
            return Ok(bytes.clone());
        }
        let mut budget = self.budget.lock().map_err(Error::new)?;
        let remaining = self.limits.total_bytes.saturating_sub(budget.bytes);
        if entry.member.size > remaining {
            return Err(Error::new(
                "nested ZIPs exceed the total decoded-byte limit",
            ));
        }
        let limit = entry
            .member
            .size
            .min(self.limits.member_bytes)
            .min(remaining);
        let mut bytes = Vec::new();
        let capacity = usize::try_from(limit.saturating_add(1))
            .map_err(|_| Error::new("ZIP member size exceeds this host's address space"))?;
        // The size has passed both limits. Reserve once to avoid geometric
        // growth and a doubled buffer when checking the byte after the payload.
        bytes.try_reserve_exact(capacity).map_err(Error::new)?;
        // Read one extra byte: neither forged sizes nor a short stream can pass.
        reader
            .zip
            .by_index(entry.index)
            .map_err(Error::new)?
            .take(limit.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(Error::new)?;
        if bytes.len() as u64 != entry.member.size {
            return Err(Error::new(format!(
                "ZIP decoded size disagrees with metadata: {path}"
            )));
        }
        budget.bytes += bytes.len() as u64;
        let bytes: Arc<[u8]> = bytes.into();
        reader.decoded.insert(path.to_owned(), bytes.clone());
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Compression {
    Stored,
    Deflated,
}

/// Write sorted regular files with fixed metadata. Data and source bytes are unchanged.
pub fn pack(
    entries: impl IntoIterator<Item = (String, Vec<u8>)>,
    compression: Compression,
) -> Result<Vec<u8>> {
    let mut files = BTreeMap::new();
    for (name, bytes) in entries {
        let path = entry_path(&name)?;
        if name.ends_with('/') || files.insert(path.clone(), bytes).is_some() {
            return Err(Error::new(format!(
                "duplicate or directory pack member: {name}"
            )));
        }
    }
    bundle_root(files.keys().map(String::as_str))?;
    let mut paths = PathTree::default();
    for path in files.keys() {
        paths.insert(path, MemberKind::File)?;
    }
    let method = match compression {
        Compression::Stored => CompressionMethod::Stored,
        Compression::Deflated => CompressionMethod::Deflated,
    };
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default()
        .compression_method(method)
        .unix_permissions(0o644);
    for (name, bytes) in files {
        zip.start_file(
            name,
            options.large_file(bytes.len() as u64 >= u64::from(u32::MAX)),
        )
        .map_err(Error::new)?;
        zip.write_all(&bytes).map_err(Error::new)?;
    }
    Ok(zip.finish().map_err(Error::new)?.into_inner())
}

fn bundle_root<'a>(paths: impl Iterator<Item = &'a str>) -> Result<String> {
    let mut top = HashSet::new();
    let mut root_main = false;
    let mut wrapped_main = false;
    for path in paths {
        root_main |= path == "main.flatppl";
        let root = path.split('/').next().unwrap();
        top.insert(root);
        wrapped_main |= path.strip_prefix(root) == Some("/main.flatppl");
    }
    if root_main {
        Ok(String::new())
    } else if top.len() == 1 && wrapped_main {
        Ok((*top.iter().next().unwrap()).to_owned())
    } else {
        Err(Error::new(
            "ZIP bundle has no root main.flatppl or sole enclosing directory",
        ))
    }
}
