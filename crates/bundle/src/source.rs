//! Portable source export. Hosts resolve identities and supply immutable bytes.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use flatppl_core::{CallHead, Idx, Module, NamedKind, Node, NodeId, RefNs, Scalar};
use sha2::{Digest, Sha256};

use crate::{Compression, Error, Result};

/// A source or data snapshot supplied by the host's existing resolver.
#[derive(Clone)]
pub struct Source {
    pub identity: String,
    pub name: String,
    pub bytes: Arc<[u8]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    Module,
    Data,
}

/// Pack parsed dependencies without inference or evaluation. Only path argument
/// spans change; all other source text and all data bytes are preserved.
pub fn pack_sources(
    root: Source,
    resolve: impl FnMut(&str, &str, SourceKind) -> std::result::Result<Source, String>,
) -> Result<Vec<u8>> {
    crate::pack(
        source_files(root, [], resolve)?.files,
        Compression::Deflated,
    )
}

/// Portable files plus the archive member assigned to each module identity.
pub struct SourceFiles {
    pub files: BTreeMap<String, Vec<u8>>,
    pub modules: HashMap<String, String>,
}

/// Collect a root and optional extra modules using one dependency/data namespace.
pub fn source_files(
    root: Source,
    additional: impl IntoIterator<Item = Source>,
    mut resolve: impl FnMut(&str, &str, SourceKind) -> std::result::Result<Source, String>,
) -> Result<SourceFiles> {
    let mut module_names = HashMap::from([(root.identity.clone(), "main.flatppl".to_string())]);
    let mut names = HashSet::from(["main.flatppl".to_string()]);
    let mut data_names: HashMap<String, String> = HashMap::new();
    let mut queue = VecDeque::from([root]);
    for source in additional {
        if !module_names.contains_key(&source.identity) {
            let name = member_name("modules", &source.name, &mut names);
            module_names.insert(source.identity.clone(), name);
            queue.push_back(source);
        }
    }
    let mut files = BTreeMap::new();
    while let Some(snapshot) = queue.pop_front() {
        let fail = |error| Error::new(format!("{}: {error}", snapshot.identity));
        let mut source = std::str::from_utf8(&snapshot.bytes)
            .map_err(|error| fail(error.to_string()))?
            .to_owned();
        let module = flatppl_syntax::parse(&source).map_err(|error| fail(error.to_string()))?;
        let member = &module_names[&snapshot.identity];
        let parent_prefix = if member == "main.flatppl" { "" } else { "../" };
        let mut replacements = Vec::new();
        for (head, kind) in [
            ("load_module", SourceKind::Module),
            ("load_data", SourceKind::Data),
        ] {
            for argument in source_arguments(&module, head) {
                let path = fixed_source(&module, argument).ok_or_else(|| {
                    fail(format!(
                        "{head} needs a fixed source to export a complete bundle"
                    ))
                })?;
                let dependency = resolve(&snapshot.identity, path, kind).map_err(fail)?;
                let target = if kind == SourceKind::Module {
                    if let Some(name) = module_names.get(&dependency.identity) {
                        name.clone()
                    } else {
                        let name = member_name("modules", &dependency.name, &mut names);
                        module_names.insert(dependency.identity.clone(), name.clone());
                        queue.push_back(dependency);
                        name
                    }
                } else if let Some(name) = data_names.get(&dependency.identity) {
                    name.clone()
                } else {
                    let name = member_name("data", &dependency.name, &mut names);
                    files.insert(name.clone(), dependency.bytes.to_vec());
                    data_names.insert(dependency.identity, name.clone());
                    name
                };
                let target = format!("{parent_prefix}{target}");
                if target == path {
                    continue;
                }
                let span = module
                    .span_of(argument)
                    .ok_or_else(|| fail("source argument has no text span".into()))?;
                let replacement =
                    serde_json::to_string(&target).map_err(|error| fail(error.to_string()))?;
                replacements.push((span.start as usize, span.end as usize, replacement));
            }
        }
        replacements.sort_unstable_by_key(|(start, _, _)| std::cmp::Reverse(*start));
        for (start, end, replacement) in replacements {
            source.replace_range(start..end, &replacement);
        }
        files.insert(
            module_names[&snapshot.identity].clone(),
            source.into_bytes(),
        );
    }
    Ok(SourceFiles {
        files,
        modules: module_names,
    })
}

/// Hash source and recursive dependencies, replacing only dependency path arguments
/// with their content digests. File locations and ZIP encoding do not affect identity.
pub fn fingerprint_sources(
    root: Source,
    mut resolve: impl FnMut(&str, &str, SourceKind) -> std::result::Result<Source, String>,
) -> Result<[u8; 32]> {
    fn visit(
        source: Source,
        kind: SourceKind,
        resolve: &mut impl FnMut(&str, &str, SourceKind) -> std::result::Result<Source, String>,
        hashes: &mut HashMap<(String, SourceKind), [u8; 32]>,
        active: &mut HashSet<(String, SourceKind)>,
    ) -> Result<[u8; 32]> {
        let key = (source.identity.clone(), kind);
        if let Some(hash) = hashes.get(&key) {
            return Ok(*hash);
        }
        if !active.insert(key.clone()) {
            return Err(Error::new("cyclic module dependencies"));
        }
        let mut hash = Sha256::new();
        if kind == SourceKind::Data {
            hash.update(b"FlatPPL data\0");
            hash.update(&source.bytes);
        } else {
            let mut text = std::str::from_utf8(&source.bytes)
                .map_err(Error::new)?
                .to_owned();
            let module = flatppl_syntax::parse(&text).map_err(Error::new)?;
            let mut replacements = Vec::new();
            for (head, child_kind) in [
                ("load_module", SourceKind::Module),
                ("load_data", SourceKind::Data),
            ] {
                for argument in source_arguments(&module, head) {
                    let path = fixed_source(&module, argument).ok_or_else(|| {
                        Error::new(format!("{head} needs a fixed source to compare content"))
                    })?;
                    let child = resolve(&source.identity, path, child_kind).map_err(Error::new)?;
                    let digest = visit(child, child_kind, resolve, hashes, active)?;
                    let digest = digest
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>();
                    let span = module
                        .span_of(argument)
                        .ok_or_else(|| Error::new("source argument has no text span"))?;
                    replacements.push((
                        span.start as usize,
                        span.end as usize,
                        format!("\"{digest}\""),
                    ));
                }
            }
            replacements.sort_unstable_by_key(|(start, _, _)| std::cmp::Reverse(*start));
            for (start, end, replacement) in replacements {
                text.replace_range(start..end, &replacement);
            }
            hash.update(b"FlatPPL module\0");
            hash.update(text.as_bytes());
        }
        let hash = hash.finalize().into();
        active.remove(&key);
        hashes.insert(key, hash);
        Ok(hash)
    }
    visit(
        root,
        SourceKind::Module,
        &mut resolve,
        &mut HashMap::new(),
        &mut HashSet::new(),
    )
}

fn member_name(directory: &str, basename: &str, used: &mut HashSet<String>) -> String {
    let basename: String = basename
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\\|?*/\"".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let basename = if basename.is_empty() || matches!(basename.as_str(), "." | "..") {
        "file"
    } else {
        &basename
    };
    let path = Path::new(basename);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(basename);
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    let mut name = format!("{directory}/{basename}");
    let mut suffix = 2;
    while !used.insert(name.clone()) {
        name = format!("{directory}/{stem}-{suffix}{extension}");
        suffix += 1;
    }
    name
}

/// Follow fixed local aliases of a source argument.
pub fn fixed_source(module: &Module, mut id: NodeId) -> Option<&str> {
    for _ in 0..module.node_count() {
        match module.node(id) {
            Node::Lit(Scalar::Str(s)) => return Some(s),
            Node::Ref(reference) if reference.ns == RefNs::SelfMod => {
                id = module.binding(module.binding_by_name(reference.name)?).rhs;
            }
            _ => return None,
        }
    }
    None
}

/// Source argument nodes retain their spans for portable rewriting.
pub fn source_arguments(module: &Module, head: &str) -> Vec<NodeId> {
    (0..module.node_count())
        .filter_map(|i| {
            let Node::Call(call) = module.node(NodeId::from_usize(i)) else {
                return None;
            };
            let CallHead::Builtin(h) = call.head else {
                return None;
            };
            if module.resolve(h) != head {
                return None;
            }
            call.args.first().copied().or_else(|| {
                if head != "load_data" {
                    return None;
                }
                call.named
                    .iter()
                    .find(|named| {
                        named.kind == NamedKind::Kwarg && module.resolve(named.name) == "source"
                    })
                    .map(|named| named.value)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_data_sources_accept_named_source_arguments() {
        let mut module =
            flatppl_syntax::parse("m = load_module(source = \"dep.flatppl\")").unwrap();
        let binding = module.bindings().next().unwrap().1.rhs;
        let Node::Call(mut call) = module.node(binding).clone() else {
            panic!("expected module call");
        };
        // Raw IR can carry a kwarg even though surface module calls use Assign.
        call.named[0].kind = NamedKind::Kwarg;
        module.alloc(Node::Call(call));
        assert!(source_arguments(&module, "load_module").is_empty());

        let data =
            flatppl_syntax::parse("x = load_data(source = \"values.json\", valueset = reals)")
                .unwrap();
        let arguments = source_arguments(&data, "load_data");
        assert_eq!(arguments.len(), 1);
        assert_eq!(fixed_source(&data, arguments[0]), Some("values.json"));
    }
}
