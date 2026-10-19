//! Export the loaded definition closure through the shared source packer.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use flatppl_bundle::source::{Source, SourceKind, fingerprint_sources, pack_sources, source_files};
use flatppl_fileaccess::Location;
use sha2::{Digest, Sha256};

use super::{
    Context, DataSource, Definition, Diagnostic, LoadedModule, Origin, join_source, registry_name,
};

const CONTEXT_FILE: &str = ".flatppl-context.json";

impl LoadedModule {
    /// Export this immutable source/dependency snapshot as a portable Deflate ZIP.
    /// Missing data and non-fixed dependency paths refuse the export.
    pub fn pack(&self) -> Result<Vec<u8>, Diagnostic> {
        let mut sources = Sources::new([self.definition.clone()], &self.data);
        pack_sources(snapshot(&self.definition), |importer, path, kind| {
            sources.resolve(importer, path, kind)
        })
        .map_err(|error| Diagnostic::new("pack", self.source_name(), error))
    }

    /// Content identity of source, imports and data, independent of file locations.
    /// The first call snapshots lazy data; subsequent calls reuse the digest.
    pub fn fingerprint(&self) -> Result<[u8; 32], Diagnostic> {
        fingerprint(&self.definition, &self.data)
    }
}

impl Context {
    /// Export a selected module plus every registered module in this context.
    /// The selected module remains main.flatppl for ordinary bundle consumers.
    pub fn pack(&self, module: &LoadedModule) -> Result<Vec<u8>, Diagnostic> {
        let fail = |error| Diagnostic::new("pack", module.source_name(), error);
        if !self.owns(module) {
            return Err(fail("the module belongs to a different context".into()));
        }
        let registered: BTreeMap<_, _> = self.registry.iter().collect();
        let roots = std::iter::once(module.definition.clone())
            .chain(registered.values().map(|definition| (*definition).clone()));
        let mut sources = Sources::new(roots, &self.data);
        let additional = registered.iter().map(|(name, definition)| {
            let mut source = snapshot(definition);
            source.name = module_name(name);
            source
        });
        let mut archive = source_files(
            snapshot(&module.definition),
            additional,
            |importer, path, kind| sources.resolve(importer, path, kind),
        )
        .map_err(|error| fail(error.to_string()))?;
        let names: BTreeMap<_, _> = registered
            .iter()
            .map(|(name, definition)| (*name, &archive.modules[&definition.identity]))
            .collect();
        archive.files.insert(
            CONTEXT_FILE.into(),
            serde_json::to_vec(&names).map_err(|error| fail(error.to_string()))?,
        );
        flatppl_bundle::pack(archive.files, flatppl_bundle::Compression::Deflated)
            .map_err(|error| fail(error.to_string()))
    }

    /// Content identity of registered names and their source/dependency snapshots.
    /// Anonymous modules, file caches and compiled artifacts do not enter identity.
    pub fn fingerprint(&self) -> Result<[u8; 32], Diagnostic> {
        if let Some(hash) = self.fingerprint.get() {
            return Ok(*hash);
        }
        let mut hash = Sha256::new();
        hash.update(b"FlatPPL context\0");
        for (name, definition) in self.registry.iter().collect::<BTreeMap<_, _>>() {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(fingerprint(definition, &self.data)?);
        }
        let hash = hash.finalize().into();
        let _ = self.fingerprint.set(hash);
        Ok(hash)
    }

    pub(super) fn restore_registry(&mut self, root: &Arc<Definition>) -> Result<(), Diagnostic> {
        let Origin::Archive { bundle, member } = &root.origin else {
            return Ok(());
        };
        if !bundle.members().any(|entry| entry.path == CONTEXT_FILE) {
            return Ok(());
        }
        let fail = |error| Diagnostic::new("context", &root.label, error);
        let entrypoint = bundle
            .resolve("", bundle.entrypoint())
            .map_err(|error| fail(error.to_string()))?;
        if member != &entrypoint {
            return Ok(());
        }
        let bytes = bundle
            .read(CONTEXT_FILE)
            .map_err(|error| fail(error.to_string()))?;
        let names: BTreeMap<String, String> =
            serde_json::from_slice(&bytes).map_err(|error| fail(error.to_string()))?;
        let mut pending = Vec::new();
        for (name, member) in names {
            if registry_name(&name)? != name {
                return Err(fail(format!("registered name `{name}` is not canonical")));
            }
            let definition = self.archive_member(bundle.clone(), &member, &mut Vec::new())?;
            self.instantiate(definition.clone())?;
            if let Some(existing) = self.registry.get(&name) {
                if !Arc::ptr_eq(existing, &definition)
                    && fingerprint(existing, &self.data)? != fingerprint(&definition, &self.data)?
                {
                    return Err(fail(format!(
                        "registered name `{name}` has different content; use a new context"
                    )));
                }
            } else {
                pending.push((name, definition));
            }
        }
        // Validate every member and collision before exposing any restored names.
        if !pending.is_empty() {
            self.registry.extend(pending);
            self.fingerprint.take();
        }
        Ok(())
    }
}

fn fingerprint(
    definition: &Arc<Definition>,
    data: &Mutex<HashMap<String, DataSource>>,
) -> Result<[u8; 32], Diagnostic> {
    if let Some(hash) = definition.fingerprint.get() {
        return Ok(*hash);
    }
    let mut sources = Sources::new([definition.clone()], data);
    let hash = fingerprint_sources(snapshot(definition), |importer, path, kind| {
        sources.resolve(importer, path, kind)
    })
    .map_err(|error| Diagnostic::new("fingerprint", &definition.label, error))?;
    let _ = definition.fingerprint.set(hash);
    Ok(hash)
}

struct Sources<'a> {
    definitions: HashMap<String, Arc<Definition>>,
    snapshots: HashMap<String, Source>,
    data: &'a Mutex<HashMap<String, DataSource>>,
}

impl<'a> Sources<'a> {
    fn new(
        roots: impl IntoIterator<Item = Arc<Definition>>,
        data: &'a Mutex<HashMap<String, DataSource>>,
    ) -> Self {
        let mut definitions = HashMap::new();
        let mut pending: Vec<_> = roots.into_iter().collect();
        while let Some(definition) = pending.pop() {
            if definitions.contains_key(&definition.identity) {
                continue;
            }
            pending.extend(definition.imports.iter().map(|(_, child)| child.clone()));
            definitions.insert(definition.identity.clone(), definition);
        }
        Self {
            definitions,
            snapshots: HashMap::new(),
            data,
        }
    }

    fn resolve(&mut self, importer: &str, path: &str, kind: SourceKind) -> Result<Source, String> {
        let definition = &self.definitions[importer];
        if kind == SourceKind::Module {
            let child = definition
                .imports
                .iter()
                .find(|(name, _)| name == path)
                .map(|(_, child)| child)
                .ok_or_else(|| format!("module `{path}` is absent from the loaded snapshot"))?;
            return Ok(self
                .snapshots
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
        if let Some(source) = self.snapshots.get(&identity) {
            return Ok(source.clone());
        }
        let bytes = definition.read_data(self.data, &identity)?;
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
        self.snapshots.insert(identity, source.clone());
        Ok(source)
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
