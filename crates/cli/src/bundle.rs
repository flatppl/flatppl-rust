//! Portable ZIP export and explicit extraction (spec §04, Module composition).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use flatppl_bundle::source::{Source, SourceKind, pack_sources};
use flatppl_bundle::{Bundle, Limits, MemberKind};
use flatppl_fileaccess::Location;

use crate::resolve::CliResolver;
use crate::{Failure, Format};

/// Capture every parsed module/data dependency into a portable archive.
/// Only source arguments change. Missing or non-fixed dependencies refuse export.
pub fn pack(input: &Path, output: &Path) -> Result<(), Failure> {
    let resolver = CliResolver::cache_only();
    let root = resolver.model_location(&Location::Local(input.to_path_buf()))?;
    let mut locations = HashMap::from([(root.display(), root.clone())]);
    let mut snapshots = HashMap::new();
    let read = |location: &Location, kind| -> Result<Source, Failure> {
        if kind == SourceKind::Module && resolver.model_format(location)? != Format::FlatPpl {
            return Err(Failure::Plain(
                "packing requires FlatPPL source modules".into(),
            ));
        }
        Ok(Source {
            identity: location.display(),
            name: resolver.file_name(location),
            bytes: resolver.read_bytes(location)?.into(),
        })
    };
    let root = read(&root, SourceKind::Module)?;
    snapshots.insert(root.identity.clone(), root.clone());
    let bytes = pack_sources(root, |importer, path, kind| {
        let mut resolve = || -> Result<Source, Failure> {
            let location = resolver.relative(&locations[importer], path)?;
            let location = if kind == SourceKind::Module {
                resolver.model_location(&location)?
            } else {
                location
            };
            let identity = location.display();
            if let Some(snapshot) = snapshots.get(&identity) {
                return Ok(snapshot.clone());
            }
            let snapshot = read(&location, kind)?;
            locations.insert(identity.clone(), location);
            snapshots.insert(identity, snapshot.clone());
            Ok(snapshot)
        };
        resolve().map_err(|error| match error {
            Failure::Plain(message)
            | Failure::Diagnostic { message, .. }
            | Failure::Refuse(message)
            | Failure::Usage(message) => message,
        })
    })
    .map_err(|error| Failure::Plain(error.to_string()))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| Failure::Plain(format!("{}: {e}", crate::terminal_path(output))))?;
    file.write_all(&bytes)
        .map_err(|e| Failure::Plain(e.to_string()))
}

/// Extract into a newly created destination. Create validated links last.
/// No member can overwrite a pre-existing path or escape through a link.
pub fn unpack(input: &Path, destination: &Path) -> Result<(), Failure> {
    crate::require_regular_file(input).map_err(|e| Failure::Plain(e.to_string()))?;
    let bytes = std::fs::read(input).map_err(|e| Failure::Plain(e.to_string()))?;
    let bundle = Bundle::open(input.display().to_string(), bytes.into(), Limits::default())
        .map_err(|e| Failure::Plain(e.to_string()))?;
    // Validate every payload before creating the destination.
    let mut files = Vec::new();
    let mut directories = Vec::new();
    let mut links = Vec::new();
    for member in bundle.members() {
        match member.kind {
            MemberKind::File => files.push((
                member.path.clone(),
                bundle
                    .read(&member.path)
                    .map_err(|e| Failure::Plain(e.to_string()))?,
            )),
            MemberKind::Directory => directories.push(member.path.clone()),
            MemberKind::Symlink => links.push((
                member.path.clone(),
                bundle
                    .symlink_target(&member.path)
                    .map_err(|e| Failure::Plain(e.to_string()))?
                    .expect("symlink metadata"),
                bundle
                    .is_dir(&member.path)
                    .map_err(|e| Failure::Plain(e.to_string()))?,
            )),
        }
    }
    std::fs::create_dir(destination)
        .map_err(|e| Failure::Plain(format!("{}: {e}", crate::terminal_path(destination))))?;
    for name in directories {
        std::fs::create_dir_all(destination.join(name))
            .map_err(|e| Failure::Plain(e.to_string()))?;
    }
    for (name, bytes) in files {
        let path = destination.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Failure::Plain(e.to_string()))?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| Failure::Plain(format!("{}: {e}", crate::terminal_path(&path))))?;
        file.write_all(&bytes)
            .map_err(|e| Failure::Plain(e.to_string()))?;
    }
    for (name, target, directory) in links {
        let path = destination.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Failure::Plain(e.to_string()))?;
        }
        create_symlink(&target, &path, directory).map_err(|e| Failure::Plain(e.to_string()))?;
    }
    Ok(())
}

#[cfg(unix)]
fn create_symlink(target: &str, path: &Path, _directory: bool) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, path)
}

#[cfg(windows)]
fn create_symlink(target: &str, path: &Path, directory: bool) -> std::io::Result<()> {
    if directory {
        std::os::windows::fs::symlink_dir(target, path)
    } else {
        std::os::windows::fs::symlink_file(target, path)
    }
}

#[cfg(not(any(unix, windows)))]
fn create_symlink(_target: &str, _path: &Path, _directory: bool) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "this host cannot create symlinks",
    ))
}
