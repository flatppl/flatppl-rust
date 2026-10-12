//! Host-owned source snapshots. Imported definitions never cache inferred instances.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use flatppl_core::{Binding, CallHead, Idx, Module, Node, NodeId, RefNs, Scalar};
use flatppl_fileaccess::{Cache, DenyAll, Location, OfflineFetcher, Resolver};
use flatppl_hs3::PyhfOptions;
use flatppl_infer::{Level, ModuleBundle, Severity};
use flatppl_stablehlo::{BatchSpec, EmitOptions};
use serde::Serialize;

use crate::Constant;
use crate::export::{Export, emit_query};
use crate::inspect::{self, BindingKind, Domain};

/// A compiler error. `span` holds byte offsets into the text of `source`.
#[derive(Debug, Serialize)]
pub struct Diagnostic {
    pub stage: &'static str,
    pub source: Box<str>,
    pub message: String,
    pub span: Option<(u32, u32)>,
    /// Written as the keys `line`, `column` and `excerpt` when the span is known.
    #[serde(flatten)]
    pub position: Option<Position>,
    pub import_chain: Vec<String>,
}

/// Where a span starts: the line and column count from 1, the column in
/// Unicode characters, and `excerpt` is the text of that line.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Position {
    pub line: u32,
    pub column: u32,
    pub excerpt: Box<str>,
}

impl Diagnostic {
    pub(crate) fn new(stage: &'static str, source: &str, message: impl ToString) -> Self {
        Self {
            stage,
            source: source.into(),
            message: message.to_string(),
            span: None,
            position: None,
            import_chain: Vec::new(),
        }
    }

    /// Locate the error at `node`, whose span indexes `text`.
    fn at(self, module: &Module, node: Option<NodeId>, text: &str) -> Self {
        let span = node.and_then(|id| module.span_of(id));
        self.spanned(span.map(|s| (s.start, s.end)), text)
    }

    fn spanned(mut self, span: Option<(u32, u32)>, text: &str) -> Self {
        self.span = span;
        self.position = span
            .and_then(|(start, _)| inspect::locate(text, start))
            .map(|(line, column, excerpt)| Position {
                line,
                column,
                excerpt: excerpt.into(),
            });
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): {}", self.source, self.stage, self.message)
    }
}

impl std::error::Error for Diagnostic {}

#[derive(Clone)]
enum Origin {
    File {
        location: Location,
        bundle_root: Option<PathBuf>,
    },
    Inline {
        name: String,
        file: Option<Location>,
    },
}

struct Definition {
    identity: String,
    label: String,
    source: String,
    origin: Origin,
    parsed: Arc<Module>,
    imports: Vec<(String, Arc<Definition>)>,
    data_roots: Vec<(String, PathBuf)>,
}

struct DataSource {
    canonical: PathBuf,
    bytes: Vec<u8>,
}

/// An immutable root instance and the parsed definitions required to compile it.
pub struct LoadedModule {
    owner: Arc<()>,
    definition: Arc<Definition>,
    typed: Module,
    bundle: ModuleBundle,
    data: Arc<Mutex<HashMap<String, DataSource>>>,
}

/// One public binding as hosts see it. Positions refer to the right-hand side
/// in the module's source text. A value bound by [`Context::set`] has none.
#[derive(Serialize)]
pub struct BindingInfo {
    pub name: String,
    pub kind: BindingKind,
    pub value_type: Option<String>,
    pub phase: Option<String>,
    pub domain: Option<Domain>,
    /// Array extents, `None` per dynamic extent. Absent for records, tuples,
    /// tables and non-values.
    pub shape: Option<Vec<Option<u32>>>,
    /// The number of scalar elements, when every extent is known.
    pub size: Option<u64>,
    pub span: Option<(u32, u32)>,
    pub source: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    /// The public bindings this one references directly.
    pub dependencies: Vec<String>,
}

/// One binding rendered as mathematics.
#[derive(Debug, Serialize)]
pub struct MathRow {
    pub name: String,
    /// Every binding the row binds: one, or a decomposition's targets.
    pub names: Vec<String>,
    /// `draw`, `value`, `measure`, `callable`, `likelihood` or `module`.
    pub kind: &'static str,
    pub text: String,
}

/// The notation [`LoadedModule::math`] writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFormat {
    Tex,
    MathMl,
    Typst,
}

impl LoadedModule {
    pub(crate) fn parsed(&self) -> &Module {
        &self.definition.parsed
    }

    pub fn source(&self) -> &str {
        &self.definition.source
    }

    pub fn source_name(&self) -> &str {
        &self.definition.label
    }

    pub fn bindings(&self) -> Vec<BindingInfo> {
        let module = &self.typed;
        let text = self.source();
        module
            .public_bindings()
            .map(|(id, b)| {
                let span = module.span_of(b.rhs).map(|s| (s.start, s.end));
                let position = span.and_then(|(start, _)| inspect::locate(text, start));
                let ty = module.type_of(b.rhs);
                BindingInfo {
                    name: module.resolve(b.name).into(),
                    kind: inspect::kind(module, b.rhs),
                    value_type: ty.map(|t| module.display_type(t)),
                    phase: module
                        .phase_of(b.rhs)
                        .map(|p| format!("{p:?}").to_lowercase()),
                    domain: module.valueset_of(b.rhs).map(|s| Domain::of(module, s)),
                    shape: ty.and_then(inspect::shape),
                    size: ty.and_then(inspect::size),
                    span,
                    source: span
                        .and_then(|(start, end)| text.get(start as usize..end as usize))
                        .map(String::from),
                    line: position.as_ref().map(|p| p.0),
                    column: position.as_ref().map(|p| p.1),
                    dependencies: inspect::dependencies(module, id),
                }
            })
            .collect()
    }

    /// Read a public binding's right-hand side back as a value.
    ///
    /// The read is structural. See [`Constant`] for what reads back. Any other
    /// expression is refused, because the host evaluates nothing.
    pub fn constant(&self, name: &str) -> Result<Constant, Diagnostic> {
        let module = &self.typed;
        let fail = |message: String| Diagnostic::new("constant", self.source_name(), message);
        let (_, binding) = module
            .public_bindings()
            .find(|(_, b)| module.resolve(b.name) == name)
            .ok_or_else(|| fail(format!("no public binding named `{name}`")))?;
        Constant::read(module, binding.rhs).map_err(|message| {
            fail(format!("`{name}` does not read back as a value: {message}")).at(
                module,
                Some(binding.rhs),
                self.source(),
            )
        })
    }

    /// Render every binding as mathematics, in source order.
    pub fn math(&self, format: MathFormat) -> Vec<MathRow> {
        flatppl_mathdoc::render_with_source(&self.typed, Some(self.source()))
            .bindings
            .into_iter()
            .map(|row| MathRow {
                text: match format {
                    MathFormat::Tex => flatppl_mathdoc::tex::statement(&row.statement),
                    MathFormat::MathMl => row.mathml,
                    MathFormat::Typst => flatppl_mathdoc::typst::statement(&row.statement),
                },
                name: row.name,
                names: row.names,
                kind: row.kind.as_str(),
            })
            .collect()
    }

    pub fn compile(&self, options: &EmitOptions) -> Result<Export, Diagnostic> {
        self.compile_export(options, None)
    }

    /// Compile independent calls over explicit host batch axes.
    pub fn compile_batched(
        &self,
        options: &EmitOptions,
        batch: &BatchSpec,
    ) -> Result<Export, Diagnostic> {
        self.compile_export(options, Some(batch))
    }

    fn compile_export(
        &self,
        options: &EmitOptions,
        batch: Option<&BatchSpec>,
    ) -> Result<Export, Diagnostic> {
        let roots: Vec<_> = self
            .typed
            .public_bindings()
            .filter(|(_, b)| matches!(self.typed.resolve(b.name), "inputs" | "outputs"))
            .map(|(_, b)| b.name)
            .collect();
        if roots.is_empty() {
            return Err(Diagnostic::new(
                "signature",
                self.source_name(),
                "declare inputs and outputs before compiling a module",
            ));
        }
        let mut lowered = flatppl_determinizer::determinize_with_options(
            &self.typed,
            &self.bundle,
            Some(&roots),
            &flatppl_determinizer::LoweringOptions {
                preserve_rand_tuple: true,
                ..options.lowering_options()
            },
        )
        .map_err(|e| {
            Diagnostic::new("determinize", self.source_name(), e.reason).at(
                &self.typed,
                Some(e.node),
                self.source(),
            )
        })?;
        crate::load_fixed_data(&mut lowered, |source| self.read_data(source))
            .map_err(|error| Diagnostic::new("load_data", self.source_name(), error))?;
        emit_query(lowered, &self.typed, options, batch).map_err(|e| {
            // Lowering appends nodes and never renumbers, so only nodes of the
            // typed root module index its source.
            let node = e.node.filter(|id| id.index() < self.typed.node_count());
            Diagnostic::new("stablehlo", self.source_name(), e.msg).at(
                &self.typed,
                node,
                self.source(),
            )
        })
    }

    fn read_data(&self, source: &str) -> Result<Vec<u8>, String> {
        let mut data = self.data.lock().map_err(|error| error.to_string())?;
        if let Some(loaded) = data.get(source) {
            self.check_data_path(source, &loaded.canonical)?;
            return Ok(loaded.bytes.clone());
        }
        let location = Location::parse(source);
        if matches!(&location, Location::Local(path) if path.is_relative()) {
            return Err("provide source_path for relative data sources in inline code".into());
        }
        let mut cache = Cache::from_env();
        cache.set_offline(true);
        let resolver = Resolver::new(cache, &OfflineFetcher, &DenyAll);
        let path = resolver
            .local_path(&location)
            .map_err(|error| error.to_string())?;
        let canonical = std::fs::canonicalize(&path).map_err(|error| error.to_string())?;
        self.check_data_path(source, &canonical)?;
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        data.insert(
            source.into(),
            DataSource {
                canonical,
                bytes: bytes.clone(),
            },
        );
        Ok(bytes)
    }

    fn check_data_path(&self, source: &str, canonical: &Path) -> Result<(), String> {
        let mut pending = vec![&self.definition];
        let mut seen = HashSet::new();
        while let Some(definition) = pending.pop() {
            if !seen.insert(&definition.identity) {
                continue;
            }
            for (name, root) in &definition.data_roots {
                if name == source {
                    check_bundle_path(canonical, Some(root), source)
                        .map_err(|error| error.to_string())?;
                }
            }
            pending.extend(definition.imports.iter().map(|(_, child)| child));
        }
        Ok(())
    }
}

/// Append-only definitions and explicit names. Existing modules own their import closure.
#[derive(Default)]
pub struct Context {
    owner: Arc<()>,
    registry: HashMap<String, Arc<Definition>>,
    files: HashMap<(String, Option<PathBuf>), Arc<Definition>>,
    sources: HashMap<String, String>,
    next_inline: usize,
    data: Arc<Mutex<HashMap<String, DataSource>>>,
}

impl Context {
    pub(crate) fn owns(&self, module: &LoadedModule) -> bool {
        Arc::ptr_eq(&self.owner, &module.owner)
    }

    /// Bind fixed values to declared externals in a new immutable module.
    pub fn set(
        &mut self,
        module: &LoadedModule,
        constants: &[(String, Constant)],
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let fail = |message: String| Diagnostic::new("constants", module.source_name(), message);
        if !Arc::ptr_eq(&self.owner, &module.owner) {
            return Err(fail("the module belongs to a different context".into()));
        }
        let mut parsed = (*module.definition.parsed).clone();
        let mut declarations = Vec::new();
        let mut seen = HashSet::new();
        for (name, value) in constants {
            if !seen.insert(name.as_str()) {
                return Err(fail(format!("constant `{name}` was supplied twice")));
            }
            let bid = parsed
                .public_bindings()
                .find(|(_, b)| parsed.resolve(b.name) == name)
                .map(|(bid, _)| bid)
                .ok_or_else(|| fail(format!("no external binding named `{name}`")))?;
            if !matches!(parsed.node(parsed.binding(bid).rhs), Node::Call(c)
                if matches!(c.head, CallHead::Builtin(s) if parsed.resolve(s) == "external"))
            {
                return Err(fail(format!("`{name}` is not an unbound external")));
            }
            declarations.push((bid, parsed.binding(bid).rhs));
            let ty = module
                .typed
                .type_of(module.typed.binding(bid).rhs)
                .ok_or_else(|| fail(format!("unresolved type for `{name}`")))?;
            let value = value
                .normalized(&module.typed, ty)
                .map_err(|error| fail(format!("constant `{name}`: {error}")))?;
            let node = value.alloc(&mut parsed).map_err(fail)?;
            parsed.set_binding_rhs(bid, node);
        }
        // Resolve every declaration after simultaneous substitution, including
        // dimensions supplied in this call. Keep these roots out of the snapshot.
        let mut validation = parsed.clone();
        for (i, &(_, rhs)) in declarations.iter().enumerate() {
            let name = validation.intern(&format!("<constant-domain-{i}>"));
            validation.add_binding(Binding {
                name,
                rhs,
                synthetic: true,
                public: false,
                doc: None,
            });
        }
        if let Some(d) = flatppl_infer::infer_module(&mut validation, &module.bundle, Level::Shape)
            .into_iter()
            .find(|d| d.severity == Severity::Error)
        {
            return Err(fail(d.message));
        }
        for ((name, value), (bid, declaration)) in constants.iter().zip(declarations) {
            let value = value
                .validated(&validation, declaration)
                .map_err(|error| fail(format!("constant `{name}`: {error}")))?;
            let node = value.alloc(&mut parsed).map_err(fail)?;
            parsed.set_binding_rhs(bid, node);
        }
        let inputs = parsed
            .public_bindings()
            .find(|(_, b)| parsed.resolve(b.name) == "inputs")
            .map(|(bid, b)| (bid, b.clone()));
        if let Some((bid, binding)) = inputs {
            let entries = match parsed.node(binding.rhs) {
                Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if parsed.resolve(s) == "tuple") => {
                    c.args.to_vec()
                }
                _ => vec![binding.rhs],
            };
            let remaining: Vec<_> = entries.into_iter().filter(|&id| {
                !matches!(parsed.node(id), Node::Ref(r) if r.ns == RefNs::SelfMod && seen.contains(parsed.resolve(r.name)))
            }).collect();
            match remaining.as_slice() {
                [] => {
                    let keep = parsed
                        .bindings()
                        .filter(|(id, _)| *id != bid)
                        .map(|(id, _)| id)
                        .collect();
                    parsed.retain_bindings(&keep);
                }
                [only] => parsed.set_binding_rhs(bid, *only),
                _ => {
                    let rhs = crate::constants::call(&mut parsed, "tuple", remaining, vec![]);
                    parsed.set_binding_rhs(bid, rhs);
                }
            }
        }
        let identity = format!("bound:{}", self.next_inline);
        self.next_inline += 1;
        self.instantiate(Arc::new(Definition {
            identity,
            label: module.definition.label.clone(),
            source: module.definition.source.clone(),
            origin: module.definition.origin.clone(),
            parsed: Arc::new(parsed),
            imports: module.definition.imports.clone(),
            data_roots: module.definition.data_roots.clone(),
        }))
    }

    /// Keep the bindings that `outputs` reach, transitively, in a new module.
    ///
    /// The result's source is the printed slice, so its positions index that
    /// text. Imports resolve as they did for `module`, and values bound by
    /// [`Self::set`] carry over.
    pub fn slice(
        &mut self,
        module: &LoadedModule,
        outputs: &[&str],
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let fail = |message: String| Diagnostic::new("slice", module.source_name(), message);
        if !Arc::ptr_eq(&self.owner, &module.owner) {
            return Err(fail("the module belongs to a different context".into()));
        }
        let mut parsed = (*module.definition.parsed).clone();
        let roots = outputs
            .iter()
            .map(|name| {
                parsed
                    .public_bindings()
                    .find(|(_, b)| parsed.resolve(b.name) == *name)
                    .map(|(_, b)| b.name)
                    .ok_or_else(|| fail(format!("no public binding named `{name}`")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let keep = inspect::reachable(&parsed, &roots);
        parsed.retain_bindings(&keep);
        let source = flatppl_syntax::print(&parsed);
        let identity = format!("slice:{}", self.next_inline);
        self.next_inline += 1;
        let definition = self.definition(
            identity,
            module.definition.label.clone(),
            source,
            module.definition.origin.clone(),
            &mut Vec::new(),
        )?;
        self.instantiate(definition)
    }

    pub fn parse(
        &mut self,
        source: &str,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let logical = match name {
            Some(name) => registry_name(name)?,
            None => format!("<inline-{}>", self.next_inline),
        };
        let identity = format!("inline:{}", self.next_inline);
        self.next_inline += 1;
        let file = source_path.map(absolute_location).transpose()?;
        let label = file
            .as_ref()
            .map_or_else(|| logical.clone(), Location::display);
        let origin = Origin::Inline {
            name: logical,
            file,
        };
        let definition =
            self.definition(identity, label, source.into(), origin, &mut Vec::new())?;
        self.instantiate(definition)
    }

    /// Import a pyhf model or workspace into this context.
    ///
    /// `options` selects the measurement and whether observations are bound.
    /// The returned source is generated FlatPPL. Names, source origins, and
    /// registration follow the same rules as [`Self::parse`].
    pub fn import_pyhf(
        &mut self,
        json: &str,
        options: &PyhfOptions,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let module = flatppl_hs3::read_pyhf_with(json, options).map_err(|error| {
            Diagnostic::new("pyhf", source_path.or(name).unwrap_or("<pyhf>"), error)
        })?;
        let source = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Minimal);
        self.parse(&source, name, source_path)
    }

    pub fn load(&mut self, path: &str) -> Result<Arc<LoadedModule>, Diagnostic> {
        let location = absolute_location(path)?;
        let definition = self.file(location, None, &mut Vec::new())?;
        self.instantiate(definition)
    }

    pub fn register(&mut self, name: &str, module: &LoadedModule) -> Result<(), Diagnostic> {
        let name = registry_name(name)?;
        if !Arc::ptr_eq(&self.owner, &module.owner) {
            return Err(Diagnostic::new(
                "context",
                &name,
                "the module belongs to a different context",
            ));
        }
        if self.registry.contains_key(&name) {
            return Err(Diagnostic::new(
                "context",
                &name,
                "the name is already registered; use a new context to change a definition",
            ));
        }
        self.registry.insert(name, module.definition.clone());
        Ok(())
    }

    fn instantiate(&self, definition: Arc<Definition>) -> Result<Arc<LoadedModule>, Diagnostic> {
        let mut bundle = ModuleBundle::new();
        bundle.set_root(&definition.identity);
        let mut pending = vec![definition.clone()];
        let mut seen = HashSet::new();
        while let Some(parent) = pending.pop() {
            if !seen.insert(parent.identity.clone()) {
                continue;
            }
            for (literal, child) in &parent.imports {
                bundle.insert_resolved(
                    &parent.identity,
                    literal,
                    &child.identity,
                    child.parsed.clone(),
                );
                pending.push(child.clone());
            }
        }
        let mut typed = (*definition.parsed).clone();
        let (diagnostics, bundle) =
            flatppl_infer::infer_module_with_inputs(&mut typed, bundle, Level::Shape);
        if let Some(d) = diagnostics
            .into_iter()
            .find(|d| d.severity == Severity::Error)
        {
            return Err(Diagnostic::new("infer", &definition.label, d.message).at(
                &typed,
                d.node,
                &definition.source,
            ));
        }
        Ok(Arc::new(LoadedModule {
            owner: self.owner.clone(),
            definition,
            typed,
            bundle,
            data: self.data.clone(),
        }))
    }

    fn definition(
        &mut self,
        identity: String,
        label: String,
        source: String,
        origin: Origin,
        stack: &mut Vec<String>,
    ) -> Result<Arc<Definition>, Diagnostic> {
        if stack.contains(&identity) {
            return Err(Diagnostic::new("resolve", &label, "cyclic module imports"));
        }
        let mut parsed = flatppl_syntax::parse(&source)
            .map_err(|e| Diagnostic::new("parse", &label, e.message).spanned(e.span, &source))?;
        let mut data_roots = Vec::new();
        crate::resolve_data_sources(&mut parsed, |literal| {
            let (base, root) = match &origin {
                Origin::File {
                    location,
                    bundle_root,
                } => (Some(location), bundle_root.as_deref()),
                Origin::Inline { file, .. } => (file.as_ref(), None),
            };
            if root.is_some()
                && (Path::new(literal).is_absolute()
                    || flatppl_core::text::uri_scheme(literal).is_some())
            {
                return Err("data sources inside a bundle must be relative paths".into());
            }
            let location = base
                .map_or_else(|| Location::parse(literal), |base| base.join(literal))
                .normalized();
            let name = location.display();
            if let (Some(root), Location::Local(path)) = (root, &location) {
                check_bundle_path(path, Some(root), &name).map_err(|error| error.to_string())?;
                data_roots.push((name.clone(), root.to_path_buf()));
            }
            Ok(name)
        })
        .map_err(|error| Diagnostic::new("load_data", &label, error))?;
        let mut literals = Vec::new();
        for i in 0..parsed.node_count() {
            if let Node::Call(c) = parsed.node(NodeId::from_usize(i))
                && matches!(c.head, CallHead::Builtin(s) if parsed.resolve(s) == "load_module")
                && let Some(&arg) = c.args.first()
            {
                let id = resolve_ref(&parsed, arg);
                if let Node::Lit(Scalar::Str(s)) = parsed.node(id)
                    && !literals.contains(&s.to_string())
                {
                    literals.push(s.to_string());
                }
            }
        }
        stack.push(identity.clone());
        let imports = literals
            .into_iter()
            .map(|literal| {
                self.resolve(&origin, &literal, stack)
                    .map(|d| (literal, d))
                    .map_err(|mut e| {
                        e.import_chain.insert(0, label.clone());
                        e
                    })
            })
            .collect::<Result<Vec<_>, _>>();
        stack.pop();
        Ok(Arc::new(Definition {
            identity,
            label,
            source,
            origin,
            parsed: Arc::new(parsed),
            imports: imports?,
            data_roots,
        }))
    }

    fn resolve(
        &mut self,
        origin: &Origin,
        literal: &str,
        stack: &mut Vec<String>,
    ) -> Result<Arc<Definition>, Diagnostic> {
        let Origin::Inline { name, file } = origin else {
            let Origin::File {
                location,
                bundle_root,
            } = origin
            else {
                unreachable!()
            };
            let root = if Path::new(literal).is_absolute()
                || flatppl_core::text::uri_scheme(literal).is_some()
            {
                None
            } else {
                bundle_root.clone()
            };
            return self.file(location.join(literal), root, stack);
        };
        let explicit_location =
            flatppl_core::text::uri_scheme(literal).is_some() || Path::new(literal).is_absolute();
        let location = if explicit_location {
            Some(Location::parse(literal))
        } else {
            file.as_ref().map(|f| f.join(literal))
        };
        let registered = if explicit_location {
            None
        } else {
            let relative = Path::new(name)
                .parent()
                .unwrap_or(Path::new(""))
                .join(literal);
            registry_name(&relative.to_string_lossy())
                .ok()
                .and_then(|key| self.registry.get(&key))
                .cloned()
        };
        if let Some(definition) = registered {
            if let Some(Location::Local(path)) = &location
                && path.exists()
            {
                let root = if path.is_dir() {
                    Some(
                        std::fs::canonicalize(path)
                            .map_err(|e| Diagnostic::new("resolve", literal, e))?,
                    )
                } else {
                    None
                };
                let target = if root.is_some() {
                    path.join("main.flatppl")
                } else {
                    path.clone()
                };
                let canonical = std::fs::canonicalize(target)
                    .map_err(|e| Diagnostic::new("resolve", literal, e))?;
                let key = (canonical.to_string_lossy().into_owned(), root);
                if file_identity(&key) != definition.identity {
                    return Err(Diagnostic::new(
                        "resolve",
                        literal,
                        "ambiguous registered and file source",
                    ));
                }
            }
            return Ok(definition);
        }
        match location {
            Some(location) => self.file(location, None, stack),
            None => Err(Diagnostic::new(
                "resolve",
                literal,
                "no registered module; provide source_path for relative file imports",
            )),
        }
    }

    fn file(
        &mut self,
        mut location: Location,
        mut bundle_root: Option<PathBuf>,
        stack: &mut Vec<String>,
    ) -> Result<Arc<Definition>, Diagnostic> {
        location = location.normalized();
        let requested = location.display();
        let request_key = (requested.clone(), bundle_root.clone());
        if let Some(definition) = self.files.get(&request_key) {
            return Ok(definition.clone());
        }
        if let Location::Local(path) = &location {
            let path = if path.is_dir() {
                let directory = std::fs::canonicalize(path)
                    .map_err(|e| Diagnostic::new("resolve", &requested, e))?;
                check_bundle_path(&directory, bundle_root.as_deref(), &requested)?;
                bundle_root = Some(directory);
                path.join("main.flatppl")
            } else {
                path.clone()
            };
            location = Location::Local(
                std::fs::canonicalize(path)
                    .map_err(|e| Diagnostic::new("resolve", &requested, e))?,
            );
            if let Location::Local(path) = &location {
                check_bundle_path(path, bundle_root.as_deref(), &requested)?;
            }
        }
        let label = location.display();
        let key = (label.clone(), bundle_root.clone());
        if let Some(definition) = self.files.get(&key).cloned() {
            self.files.insert(request_key, definition.clone());
            return Ok(definition);
        }
        let identity = file_identity(&key);
        let source = if let Some(source) = self.sources.get(&label) {
            source.clone()
        } else {
            let mut cache = Cache::from_env();
            cache.set_offline(true);
            let resolver = Resolver::new(cache, &OfflineFetcher, &DenyAll);
            let bytes = resolver
                .read(&location)
                .map_err(|e| Diagnostic::new("resolve", &label, e))?;
            let source =
                String::from_utf8(bytes).map_err(|e| Diagnostic::new("parse", &label, e))?;
            self.sources.insert(label.clone(), source.clone());
            source
        };
        let definition = self.definition(
            identity,
            label,
            source,
            Origin::File {
                location,
                bundle_root,
            },
            stack,
        )?;
        self.files.insert(key, definition.clone());
        self.files.insert(request_key, definition.clone());
        Ok(definition)
    }
}

fn file_identity(key: &(String, Option<PathBuf>)) -> String {
    format!("file:{key:?}")
}

fn check_bundle_path(path: &Path, root: Option<&Path>, label: &str) -> Result<(), Diagnostic> {
    if root.is_some_and(|root| !path.starts_with(root)) {
        return Err(Diagnostic::new(
            "resolve",
            label,
            "relative import escapes its bundle root",
        ));
    }
    Ok(())
}

fn absolute_location(source: &str) -> Result<Location, Diagnostic> {
    let location = Location::parse(source);
    match location {
        Location::Local(path) if path.is_relative() => Ok(Location::Local(
            std::env::current_dir()
                .map_err(|e| Diagnostic::new("resolve", source, e))?
                .join(path),
        )
        .normalized()),
        location => Ok(location.normalized()),
    }
}

fn registry_name(name: &str) -> Result<String, Diagnostic> {
    let mut parts = Vec::new();
    for part in Path::new(name).components() {
        match part {
            Component::Normal(p) => parts.push(p.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir if !parts.is_empty() => {
                parts.pop();
            }
            _ => {
                return Err(Diagnostic::new(
                    "context",
                    name,
                    "registry names must stay within the context's relative namespace",
                ));
            }
        }
    }
    if parts.is_empty() || flatppl_core::text::uri_scheme(name).is_some() {
        return Err(Diagnostic::new(
            "context",
            name,
            "expected a nonempty relative registry name",
        ));
    }
    Ok(parts.join("/"))
}

pub(crate) fn resolve_ref(module: &Module, mut id: NodeId) -> NodeId {
    let mut seen = HashSet::new();
    while seen.insert(id) {
        if let Node::Ref(r) = module.node(id)
            && r.ns == RefNs::SelfMod
            && let Some(b) = module.binding_by_name(r.name)
        {
            id = module.binding(b).rhs;
        } else {
            break;
        }
    }
    id
}
