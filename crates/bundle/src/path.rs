use std::collections::{HashMap, VecDeque};

use crate::{Bundle, Error, MemberKind, Result};

#[derive(Default)]
struct Node {
    children: HashMap<String, usize>,
    member: Option<MemberKind>,
}

/// A flat arena keeps long paths linear in time and memory, including on drop.
/// Storing every full directory prefix instead would take quadratic space.
pub(crate) struct PathTree(Vec<Node>);

impl Default for PathTree {
    fn default() -> Self {
        Self(vec![Node::default()])
    }
}

impl PathTree {
    pub(crate) fn insert(&mut self, path: &str, kind: MemberKind) -> Result<()> {
        let mut node = 0;
        for part in path.split('/') {
            if self.0[node]
                .member
                .is_some_and(|kind| kind != MemberKind::Directory)
            {
                return Err(Error::new(format!("ZIP file/directory collision: {path}")));
            }
            node = if let Some(child) = self.child(node, part) {
                child
            } else {
                let child = self.0.len();
                self.0.push(Node::default());
                self.0[node].children.insert(part.to_owned(), child);
                child
            };
        }
        if kind != MemberKind::Directory && !self.0[node].children.is_empty() {
            return Err(Error::new(format!("ZIP file/directory collision: {path}")));
        }
        self.0[node].member = Some(kind);
        Ok(())
    }

    pub(crate) fn find(&self, path: &str) -> Option<usize> {
        path.split('/')
            .filter(|part| !part.is_empty())
            .try_fold(0, |node, part| self.child(node, part))
    }

    fn child(&self, node: usize, part: &str) -> Option<usize> {
        self.0[node].children.get(part).copied()
    }

    pub(crate) fn is_dir(&self, node: usize) -> bool {
        node == 0
            || self.0[node].member == Some(MemberKind::Directory)
            || !self.0[node].children.is_empty()
    }
}

pub(crate) fn bounded(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        return Err(Error::new("bundle path exceeds the path-byte limit"));
    }
    Ok(())
}

/// ZIP names use `/` on every host and may not contain parent segments (§04).
pub(crate) fn entry_path(name: &str) -> Result<String> {
    relative(name)?;
    let mut parts = Vec::new();
    for part in name.split('/') {
        match part {
            ".." => return Err(Error::new(format!("ZIP entry contains `..`: {name}"))),
            "" | "." => {}
            part => parts.push(part),
        }
    }
    if parts.is_empty() {
        return Err(Error::new("ZIP entry has an empty path"));
    }
    Ok(parts.join("/"))
}

pub(crate) fn relative(path: &str) -> Result<()> {
    let scheme = path
        .split('/')
        .next()
        .is_some_and(|part| part.contains(':'));
    if path.is_empty() || path.starts_with('/') || path.contains(['\\', '\0']) || scheme {
        return Err(Error::new(format!(
            "bundle paths must be relative and use `/`: {path}"
        )));
    }
    Ok(())
}

pub(crate) fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

pub(crate) fn join(root: &str, path: &str) -> String {
    if root.is_empty() {
        path.to_owned()
    } else if path.is_empty() {
        root.to_owned()
    } else {
        format!("{root}/{path}")
    }
}

pub(crate) fn percent(value: &str, slash: bool) -> String {
    let mut output = String::new();
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) || (slash && byte == b'/') {
            output.push(char::from(byte));
        } else {
            output.push('%');
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 15)]));
        }
    }
    output
}

impl Bundle {
    /// Resolve a source relative to its declaring member, following contained links.
    ///
    /// Missing members remain paths until read, so an unused `load_data` stays lazy.
    /// Parent traversal is evaluated after links, as for directory bundles (§04).
    pub fn resolve(&self, importer: &str, source: &str) -> Result<String> {
        bounded(source.len(), self.inner.archive.limits.path_bytes)?;
        relative(source)?;
        let base = if importer.is_empty() {
            String::new()
        } else {
            let importer = self.canonical(importer)?;
            parent(&importer).to_owned()
        };
        self.canonical(&join(&base, source))
    }

    pub(crate) fn canonical(&self, path: &str) -> Result<String> {
        if path.is_empty() {
            return Ok(String::new());
        }
        let limit = self.inner.archive.limits.path_bytes;
        bounded(path.len(), limit)?;
        relative(path)?;
        let mut pending: VecDeque<String> = path.split('/').map(String::from).collect();
        let mut resolved = String::new();
        let mut ancestors = Vec::new();
        let paths = &self.inner.archive.paths;
        let mut node = Some(self.inner.root_node);
        let mut links = 0;
        while let Some(part) = pending.pop_front() {
            match part.as_str() {
                "" | "." => continue,
                ".." => {
                    let (length, parent) = ancestors.pop().ok_or_else(|| {
                        Error::new(format!("path escapes its bundle root: {path}"))
                    })?;
                    resolved.truncate(length);
                    node = parent;
                    continue;
                }
                _ => {}
            }
            ancestors.push((resolved.len(), node));
            if !resolved.is_empty() {
                resolved.push('/');
            }
            resolved.push_str(&part);
            bounded(resolved.len(), limit)?;
            node = node.and_then(|node| paths.child(node, &part));
            if node.is_some_and(|node| paths.0[node].member == Some(MemberKind::Symlink)) {
                links += 1;
                if links > 40 {
                    return Err(Error::new(format!(
                        "cyclic or excessive symbolic links: {path}"
                    )));
                }
                let raw = join(&self.inner.root, &resolved);
                let bytes = self.inner.archive.read(&raw)?;
                let target = std::str::from_utf8(&bytes)
                    .map_err(|_| Error::new(format!("symbolic link target is not UTF-8: {raw}")))?;
                relative(target)?;
                let (length, parent) = ancestors.pop().unwrap();
                resolved.truncate(length);
                node = parent;
                for component in target.split('/').rev() {
                    pending.push_front(component.to_owned());
                }
            }
        }
        Ok(resolved)
    }
}
