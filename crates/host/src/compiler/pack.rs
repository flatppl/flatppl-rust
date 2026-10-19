//! Export the loaded definition closure through the shared source packer.

use std::collections::HashMap;

use flatppl_bundle::source::{Source, SourceKind, pack_sources};
use flatppl_fileaccess::Location;

use super::{DataSource, Definition, Diagnostic, LoadedModule, Origin, join_source};

impl LoadedModule {
    /// Export this immutable source/dependency snapshot as a portable Deflate ZIP.
    /// Missing data and non-fixed dependency paths refuse the export.
    pub fn pack(&self) -> Result<Vec<u8>, Diagnostic> {
        let mut definitions = HashMap::new();
        let mut pending = vec![self.definition.clone()];
        while let Some(definition) = pending.pop() {
            if definitions.contains_key(&definition.identity) {
                continue;
            }
            pending.extend(definition.imports.iter().map(|(_, child)| child.clone()));
            definitions.insert(definition.identity.clone(), definition);
        }
        let mut snapshots = HashMap::new();
        let root = snapshot(&self.definition);
        snapshots.insert(root.identity.clone(), root.clone());
        pack_sources(root, |importer, path, kind| {
            let definition = &definitions[importer];
            if kind == SourceKind::Module {
                let child = definition
                    .imports
                    .iter()
                    .find(|(name, _)| name == path)
                    .map(|(_, child)| child)
                    .ok_or_else(|| format!("module `{path}` is absent from the loaded snapshot"))?;
                return Ok(snapshots
                    .entry(child.identity.clone())
                    .or_insert_with(|| {
                        let mut source = snapshot(child);
                        if matches!(child.origin, Origin::Inline { .. }) {
                            source.name = module_name(path);
                        }
                        source
                    })
                    .clone());
            }
            let identity = data_source(definition, path)?;
            if let Some(source) = snapshots.get(&identity) {
                return Ok(source.clone());
            }
            let bytes = self.read_data(&identity)?;
            let data = self.data.lock().map_err(|error| error.to_string())?;
            let name = match data.get(&identity) {
                Some(DataSource::Archive { member, .. }) => basename(member),
                _ => Location::parse(&identity).name(),
            };
            let source = Source {
                identity: identity.clone(),
                name,
                bytes: bytes.into(),
            };
            snapshots.insert(identity, source.clone());
            Ok(source)
        })
        .map_err(|error| Diagnostic::new("pack", self.source_name(), error))
    }
}

fn snapshot(definition: &Definition) -> Source {
    // Binding replacements from Context::set have no original text span.
    // Print that effective definition instead of exporting stale external declarations.
    let changed = definition
        .parsed
        .bindings()
        .any(|(_, binding)| !binding.synthetic && definition.parsed.span_of(binding.rhs).is_none());
    let source = if changed {
        flatppl_syntax::print(&definition.parsed)
    } else {
        definition.source.clone()
    };
    let name = match &definition.origin {
        Origin::Archive { member, .. } => basename(member),
        Origin::File { location, .. } => location.name(),
        Origin::Inline { name, .. } => basename(name),
    };
    Source {
        identity: definition.identity.clone(),
        name: module_name(&name),
        bytes: source.into_bytes().into(),
    }
}

fn data_source(definition: &Definition, source: &str) -> Result<String, String> {
    // Bound/sliced definitions already contain the mapped private data key.
    if source.starts_with(flatppl_bundle::DATA_SOURCE_PREFIX) {
        return Ok(source.into());
    }
    Ok(match &definition.origin {
        Origin::Archive { bundle, member } => {
            let path = bundle
                .resolve(member, source)
                .map_err(|error| error.to_string())?;
            bundle.data_identity_for(&path)
        }
        Origin::File {
            location,
            bundle_root,
        } => join_source(location, source, bundle_root.is_some()).display(),
        Origin::Inline { file, .. } => file
            .as_ref()
            .map_or_else(
                || Location::parse(source).normalized(),
                |base| base.join(source).normalized(),
            )
            .display(),
    })
}

fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_owned()
}

fn module_name(path: &str) -> String {
    let mut name = basename(path);
    if !name.ends_with(".flatppl") {
        name.push_str(".flatppl");
    }
    name
}
