//! Source resolution for the CLI: read the main input and (for `infer`) the
//! transitive `load_module` dependency graph through the `flatppl-fileaccess`
//! abstraction — local paths pass through, `http`/`https` URLs are fetched and
//! cached (spec §sec:url-cache).
//!
//! This is the host-side counterpart to `load_module`: the engine stays
//! I/O-free and consumes a pre-assembled [`ModuleBundle`]; this module builds
//! that bundle. Trust is batched per discovery wave — each level of the
//! dependency tree is approved together (interactive prompt; non-interactive
//! refuses untrusted URLs, per spec §sec:url-cache).

use std::cell::RefCell;
use std::collections::HashSet;
use std::io::IsTerminal;

use flatppl_core::{CallHead, Idx, Module, Node, NodeId, Scalar};
use flatppl_fileaccess::{Cache, Fetcher, Location, OfflineFetcher};

use crate::Failure;

/// The CLI's file-access layer: a [`Cache`] + a fetcher plus the interactive
/// trust policy. `source` reads (deps) go through here so local paths and URLs
/// are handled uniformly. A cache-only resolver (`convert`/`infer`) carries an
/// [`OfflineFetcher`] and never touches the network; the fetching resolver
/// (`flatppl fetch`) carries the HTTP client.
pub struct CliResolver {
    cache: Cache,
    fetcher: Box<dyn Fetcher>,
    /// Whether we may prompt for trust (stdin + stderr are both TTYs).
    interactive: bool,
    /// Re-fetch URLs even when cached (`flatppl fetch --update`).
    update: bool,
    /// URLs approved this session (so a batch-prompted wave is not re-prompted
    /// when each member is then read).
    approved: RefCell<HashSet<String>>,
}

impl CliResolver {
    /// A cache-only resolver: local files + the existing cache, never the
    /// network (no HTTP client linked). Forces offline regardless of
    /// `FLATPPL_CACHE_OFFLINE`. Used by `convert`/`infer` — a remote dependency
    /// that is not cached is an error pointing at `flatppl fetch`.
    pub fn cache_only() -> Self {
        let mut cache = Cache::from_env();
        cache.set_offline(true);
        CliResolver {
            cache,
            fetcher: Box::new(OfflineFetcher),
            interactive: false,
            update: false,
            approved: RefCell::new(HashSet::new()),
        }
    }

    /// A fetching resolver for `flatppl fetch`: online (HTTP), trust-prompting
    /// when interactive, honoring `FLATPPL_CACHEDIR` / `_OFFLINE` / `FLATPPL_TRUST`
    /// (so `FLATPPL_CACHE_OFFLINE` makes a fetch fail loudly). `update` re-fetches
    /// URLs even when already cached.
    #[cfg(feature = "fetch")]
    pub fn fetching(update: bool) -> Self {
        CliResolver {
            cache: Cache::from_env(),
            fetcher: Box::new(flatppl_fileaccess::HttpFetcher),
            interactive: std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
            update,
            approved: RefCell::new(HashSet::new()),
        }
    }

    /// Batch-approve every not-yet-trusted URL among `locs` (one prompt for the
    /// whole wave). Non-interactive tooling errors instead of prompting. Local
    /// locations and already-trusted/cached URLs need nothing.
    pub fn ensure_trusted(&self, locs: &[&Location]) -> Result<(), Failure> {
        let mut need: Vec<String> = Vec::new();
        for loc in locs {
            if let Location::Remote(url) = loc {
                if !self.approved.borrow().contains(url) && self.cache.needs_approval(url) {
                    need.push(url.clone());
                }
            }
        }
        need.sort();
        need.dedup();
        if need.is_empty() {
            return Ok(());
        }
        if !self.interactive {
            return Err(Failure::Plain(format!(
                "refusing to fetch untrusted URL(s) — non-interactive; set FLATPPL_TRUST to \
                 allow, or pre-trust them:\n  {}",
                need.join("\n  ")
            )));
        }
        if prompt_trust(&need)? {
            let mut approved = self.approved.borrow_mut();
            for url in need {
                approved.insert(url);
            }
            Ok(())
        } else {
            Err(Failure::Plain(
                "aborted: declined to trust the listed URL(s)".to_string(),
            ))
        }
    }

    /// Resolve `loc` to a local file path: a local path is returned if it
    /// exists; a URL is fetched + cached (gated on trust, which a prior
    /// `ensure_trusted` may already have granted for the wave) and its cached
    /// object path returned. Reads no content — the data reader / parser does.
    pub fn resolve_path(&self, loc: &Location) -> Result<std::path::PathBuf, Failure> {
        self.ensure_trusted(&[loc])?;
        match loc {
            Location::Local(p) => {
                if p.exists() {
                    Ok(p.clone())
                } else {
                    Err(Failure::Plain(format!("file not found: {}", p.display())))
                }
            }
            Location::Remote(url) => {
                let oracle = |u: &str| self.approved.borrow().contains(u);
                let fetched = if self.update {
                    self.cache.refetch(url, &*self.fetcher, &oracle)
                } else {
                    self.cache.get(url, &*self.fetcher, &oracle)
                };
                fetched.map_err(|e| match e {
                    flatppl_fileaccess::Error::Offline(u) => Failure::Plain(format!(
                        "`{u}` is not in the local cache — run `flatppl fetch <model>` to fetch \
                         its dependencies"
                    )),
                    other => Failure::Plain(other.to_string()),
                })
            }
        }
    }

    /// Resolve `loc` and read it as UTF-8 text (for FlatPPL/FlatPIR sources).
    pub fn read_string(&self, loc: &Location) -> Result<String, Failure> {
        let path = self.resolve_path(loc)?;
        std::fs::read_to_string(&path)
            .map_err(|e| Failure::Plain(format!("reading `{}`: {e}", loc.display())))
    }

    /// Resolve a batch of locations to local files (fetching + caching URLs,
    /// validating local paths) with a single batched trust prompt. Used to make
    /// a model's `load_data` sources locally available; the bytes are read later
    /// by the data reader, not here.
    pub fn resolve_all(&self, locs: &[Location]) -> Result<(), Failure> {
        let refs: Vec<&Location> = locs.iter().collect();
        self.ensure_trusted(&refs)?;
        for loc in locs {
            self.resolve_path(loc)?;
        }
        Ok(())
    }
}

/// Prompt (on stderr) to trust + fetch a wave of URLs; read the answer from
/// stdin. Returns `true` on `y`/`yes`.
fn prompt_trust(urls: &[String]) -> Result<bool, Failure> {
    use std::io::Write;
    eprintln!("flatppl: the following URL source(s) are not yet trusted:");
    for url in urls {
        eprintln!("  {url}");
    }
    eprint!("Fetch and trust them? [y/N]: ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| Failure::Plain(format!("reading approval: {e}")))?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// The string literal `source` of a `load_module`/`load_data` call: the first
/// positional argument, or a `source =` keyword argument (the form `load_data`
/// commonly uses). `None` for a non-literal source — it cannot be resolved
/// statically by the host.
fn source_of<'m>(module: &'m Module, call: &flatppl_core::Call) -> Option<&'m str> {
    if let Some(&arg0) = call.args.first() {
        if let Node::Lit(Scalar::Str(s)) = module.node(arg0) {
            return Some(s);
        }
    }
    for named in call.named.iter() {
        if module.resolve(named.name) == "source" {
            if let Node::Lit(Scalar::Str(s)) = module.node(named.value) {
                return Some(s);
            }
        }
    }
    None
}

/// Every `source` of the builtin `head` in `module`, as `(source-string,
/// resolved Location)` with the source resolved relative to `base`.
/// `standard_module` is a catalogue reference (not a file) and never matched
/// here; non-literal sources are skipped.
fn sources_with_head(module: &Module, base: &Location, head: &str) -> Vec<(String, Location)> {
    let mut out = Vec::new();
    for i in 0..module.node_count() {
        let id = NodeId::from_usize(i);
        let Node::Call(call) = module.node(id) else {
            continue;
        };
        let CallHead::Builtin(h) = call.head else {
            continue;
        };
        if module.resolve(h) != head {
            continue;
        }
        if let Some(source) = source_of(module, call) {
            out.push((source.to_string(), base.join(source)));
        }
    }
    out
}

/// The `load_module` directives in `module`, each resolved relative to `base`.
pub fn directives_of(module: &Module, base: &Location) -> Vec<(String, Location)> {
    sources_with_head(module, base, "load_module")
}

/// The `load_data` source locations in `module`, resolved relative to `base`.
pub fn data_sources_of(module: &Module, base: &Location) -> Vec<Location> {
    sources_with_head(module, base, "load_data")
        .into_iter()
        .map(|(_, loc)| loc)
        .collect()
}

/// Discover and resolve a model's references for inference. Walks the
/// `load_module` graph breadth-first, resolving + parsing each dependency
/// through `resolver` into a [`ModuleBundle`] keyed by the directive string the
/// engine looks up (trust batched per BFS level), and collects every
/// `load_data` source location found across the graph (root + dependencies),
/// deduplicated. The data locations are returned, not fetched here — the caller
/// decides whether/when to resolve them (see [`CliResolver::resolve_all`]).
///
/// The bundle is keyed by directive string (the engine's lookup key), so a
/// given string must denote one file across the whole graph — a conflict is a
/// hard error rather than a silent last-writer-wins.
#[cfg(feature = "infer")]
pub fn build_bundle(
    root: &Module,
    root_loc: &Location,
    resolver: &CliResolver,
) -> Result<(flatppl_infer::ModuleBundle, Vec<Location>), Failure> {
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut bundle = flatppl_infer::ModuleBundle::new();
    // Parse each distinct file once, keyed by its resolved location.
    let mut parsed: HashMap<String, Arc<Module>> = HashMap::new();
    // Locations whose own dependencies have already been queued (cycle guard).
    let mut walked: HashSet<String> = HashSet::new();
    // directive string → resolved location, to catch a string used for two files.
    let mut key_loc: HashMap<String, String> = HashMap::new();
    // `load_data` source locations across the graph, deduped by display.
    let mut data: Vec<Location> = Vec::new();
    let mut data_seen: HashSet<String> = HashSet::new();
    let mut collect_data = |module: &Module, base: &Location| {
        for loc in data_sources_of(module, base) {
            if data_seen.insert(loc.display()) {
                data.push(loc);
            }
        }
    };

    collect_data(root, root_loc);
    walked.insert(root_loc.display());
    let mut level = directives_of(root, root_loc);

    while !level.is_empty() {
        let locs: Vec<&Location> = level.iter().map(|(_, l)| l).collect();
        resolver.ensure_trusted(&locs)?;

        let mut next: Vec<(String, Location)> = Vec::new();
        for (directive, loc) in &level {
            let ld = loc.display();
            match key_loc.get(directive) {
                Some(prev) if prev != &ld => {
                    return Err(Failure::Plain(format!(
                        "load_module(\"{directive}\") refers to two different files \
                         ({prev} and {ld}); the inference bundle is keyed by the directive \
                         string, so a name must denote one file across the module graph"
                    )));
                }
                _ => {
                    key_loc.insert(directive.clone(), ld.clone());
                }
            }

            let module = match parsed.get(&ld) {
                Some(m) => m.clone(),
                None => {
                    let source = resolver.read_string(loc)?;
                    let format = crate::Format::from_location(loc).map_err(Failure::Plain)?;
                    let module =
                        crate::read_module(format, &source).map_err(|(message, line, span)| {
                            Failure::Diagnostic {
                                path: std::path::PathBuf::from(loc.display()),
                                source: source.clone(),
                                message,
                                line,
                                span,
                            }
                        })?;
                    let module = Arc::new(module);
                    parsed.insert(ld.clone(), module.clone());
                    module
                }
            };
            bundle.insert(directive.clone(), module.clone());

            // Queue this file's own dependencies (and collect its data
            // sources) exactly once.
            if walked.insert(ld.clone()) {
                collect_data(&module, loc);
                next.extend(directives_of(&module, loc));
            }
        }
        level = next;
    }
    Ok((bundle, data))
}

/// Fetch a model's transitive dependencies into the cache (the `flatppl fetch`
/// command). BFS over each input file's `load_module` graph — reading + parsing
/// each module to discover its deps, fetching remote ones — then resolve every
/// `load_data` source. Local files and local deps need no fetch. Trust is
/// batched per BFS level. The `resolver`'s `update` flag controls whether
/// already-cached URLs are re-fetched.
#[cfg(feature = "fetch")]
pub fn fetch_graph(files: &[Location], resolver: &CliResolver) -> Result<(), Failure> {
    let mut walked: HashSet<String> = HashSet::new();
    let mut data: Vec<Location> = Vec::new();
    let mut data_seen: HashSet<String> = HashSet::new();
    let mut level: Vec<Location> = files.to_vec();

    while !level.is_empty() {
        let refs: Vec<&Location> = level.iter().collect();
        resolver.ensure_trusted(&refs)?;
        let mut next: Vec<Location> = Vec::new();
        for loc in &level {
            if !walked.insert(loc.display()) {
                continue;
            }
            let source = resolver.read_string(loc)?;
            let format = crate::Format::from_location(loc).map_err(Failure::Plain)?;
            let module = crate::read_module(format, &source).map_err(|(message, line, span)| {
                Failure::Diagnostic {
                    path: std::path::PathBuf::from(loc.display()),
                    source: source.clone(),
                    message,
                    line,
                    span,
                }
            })?;
            for d in data_sources_of(&module, loc) {
                if data_seen.insert(d.display()) {
                    data.push(d);
                }
            }
            for (_, dep) in directives_of(&module, loc) {
                next.push(dep);
            }
        }
        level = next;
    }
    resolver.resolve_all(&data)
}

#[cfg(all(test, feature = "infer"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The `bayesian_inference_3` fixture loads `common`, which loads `priors`
    /// (two-level nesting). Building the bundle from local fixtures and inferring
    /// must resolve every cross-module reference (no "not found"/"deferred"),
    /// proving the CLI walker assembles the same graph the engine expects.
    #[test]
    fn build_bundle_resolves_local_two_level_nesting() {
        let dir: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "../../fixtures/flatppl/bayesian_inference",
        ]
        .iter()
        .collect();
        let root_path = dir.join("bayesian_inference_3.flatppl");
        let root_loc = Location::Local(root_path.clone());

        let resolver = CliResolver::cache_only(); // local-only: no env/net/trust needed
        let source = std::fs::read_to_string(&root_path).unwrap();
        let mut root = flatppl_syntax::parse(&source).expect("root parses");

        let (bundle, _data) = build_bundle(&root, &root_loc, &resolver).expect("bundle builds");
        let diags = flatppl_infer::infer_module(&mut root, &bundle, flatppl_infer::Level::Shape);

        let resolution_errors: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.severity == flatppl_infer::Severity::Error
                    && (d.message.contains("not found")
                        || d.message.contains("has no binding")
                        || d.message.contains("deferred"))
            })
            .collect();
        assert!(
            resolution_errors.is_empty(),
            "nested local deps must resolve via the CLI walker; got {resolution_errors:?}"
        );
    }

    /// `load_data` sources are discovered (kwarg `source =` form included) and
    /// routed through the same resolver: a present local file resolves, a
    /// missing one errors — proving load_data file resolution goes through
    /// `fileaccess` like `load_module`.
    #[test]
    fn load_data_sources_discovered_and_resolved() {
        let dir = std::env::temp_dir().join(format!("flatppl-ld-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("events.csv"), "a\n1\n").unwrap();
        let model = "obs = load_data(source = \"events.csv\", valueset = reals)\n\
                     w = load_data(\"weights.csv\", reals)\n";
        let root_loc = Location::Local(dir.join("model.flatppl"));
        let root = flatppl_syntax::parse(model).expect("parses");
        let resolver = CliResolver::cache_only();

        let (_bundle, data) = build_bundle(&root, &root_loc, &resolver).expect("walks");
        let names: Vec<String> = data.iter().map(|l| l.name()).collect();
        assert!(
            names.contains(&"events.csv".to_string()) && names.contains(&"weights.csv".to_string()),
            "both load_data sources (kwarg + positional) must be discovered; got {names:?}"
        );

        // Batched resolution: the present file resolves; a batch including the
        // missing `weights.csv` errors.
        assert!(
            resolver
                .resolve_all(&[Location::Local(dir.join("events.csv"))])
                .is_ok()
        );
        assert!(resolver.resolve_all(&data).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(all(test, feature = "fetch"))]
mod fetch_tests {
    use super::*;

    /// `fetch_graph` walks an all-local model graph (module dep + data source)
    /// with nothing to fetch, and errors when a dependency is missing. (Uses a
    /// cache-only resolver — local files need no network — so it exercises the
    /// walk itself without hitting the wire.)
    #[test]
    fn fetch_graph_walks_local_graph_and_reports_missing() {
        let dir = std::env::temp_dir().join(format!("flatppl-fg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("helper.flatppl"), "h = 1.0\n").unwrap();
        std::fs::write(dir.join("data.csv"), "a\n1\n").unwrap();
        std::fs::write(
            dir.join("model.flatppl"),
            "m = load_module(\"helper.flatppl\")\n\
             d = load_data(source = \"data.csv\", valueset = reals)\n\
             x = m.h\n",
        )
        .unwrap();
        let resolver = CliResolver::cache_only();

        let ok = vec![Location::Local(dir.join("model.flatppl"))];
        assert!(
            fetch_graph(&ok, &resolver).is_ok(),
            "all-local graph resolves"
        );

        std::fs::write(
            dir.join("broken.flatppl"),
            "m = load_module(\"absent.flatppl\")\n",
        )
        .unwrap();
        let broken = vec![Location::Local(dir.join("broken.flatppl"))];
        assert!(
            fetch_graph(&broken, &resolver).is_err(),
            "missing dep errors"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
