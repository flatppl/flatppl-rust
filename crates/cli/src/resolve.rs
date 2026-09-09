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
#[cfg(feature = "prepare")]
use std::io::IsTerminal;

use flatppl_core::{CallHead, Idx, Module, Node, NodeId, Scalar};
use flatppl_fileaccess::{Cache, Fetcher, Location, OfflineFetcher};

use crate::Failure;

/// The CLI's file-access layer: a [`Cache`] + a fetcher plus the interactive
/// trust policy. `source` reads (deps) go through here so local paths and URLs
/// are handled uniformly. A cache-only resolver (`convert`/`infer`) carries an
/// [`OfflineFetcher`] and never touches the network; the fetching resolver
/// (`flatppl prepare`) carries the HTTP client.
pub struct CliResolver {
    cache: Cache,
    fetcher: Box<dyn Fetcher>,
    /// Whether we may prompt for trust (stdin + stderr are both TTYs).
    interactive: bool,
    /// Re-fetch URLs even when cached (`flatppl prepare --update`).
    update: bool,
    /// URLs approved this session (so a batch-prompted wave is not re-prompted
    /// when each member is then read).
    approved: RefCell<HashSet<String>>,
}

impl CliResolver {
    /// A cache-only resolver: local files + the existing cache, never the
    /// network (no HTTP client linked). Forces offline regardless of
    /// `FLATPPL_CACHE_OFFLINE`. Used by `convert`/`infer` — a remote dependency
    /// that is not cached is an error pointing at `flatppl prepare`.
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

    /// A fetching resolver for `flatppl prepare`: online (HTTP), trust-prompting
    /// when interactive, honoring `FLATPPL_CACHEDIR` / `_OFFLINE` / `FLATPPL_TRUST`
    /// (so `FLATPPL_CACHE_OFFLINE` makes a fetch fail loudly). `update` re-fetches
    /// URLs even when already cached.
    #[cfg(feature = "prepare")]
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
            if let Location::Remote(url) = loc
                && !self.approved.borrow().contains(url)
                && self.cache.needs_approval(url)
            {
                need.push(url.clone());
            }
        }
        need.sort();
        need.dedup();
        if need.is_empty() {
            return Ok(());
        }
        if !self.interactive {
            return Err(Failure::Plain(noninteractive_trust_message(&need)));
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
            Location::Local(p) => match crate::require_regular_file(p) {
                Ok(()) => Ok(p.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(Failure::Plain(
                    format!("file not found: {}", crate::terminal_path(p)),
                )),
                Err(error) => Err(Failure::Plain(format!(
                    "invalid source `{}`: {error}",
                    crate::terminal_path(p)
                ))),
            },
            Location::Remote(url) => {
                let oracle = |u: &str| self.approved.borrow().contains(u);
                let fetched = if self.update {
                    self.cache.refetch(url, &*self.fetcher, &oracle)
                } else {
                    self.cache.get(url, &*self.fetcher, &oracle)
                };
                fetched.map_err(|e| match e {
                    flatppl_fileaccess::Error::Offline(u) => Failure::Plain(format!(
                        "`{}` is not in the local cache — run `flatppl prepare <model>` to fetch \
                         its dependencies",
                        terminal_url(&u)
                    )),
                    other => Failure::Plain(terminal_fileaccess_error(other)),
                })
            }
        }
    }

    /// Resolve `loc` and read it as UTF-8 text (for FlatPPL/FlatPIR sources).
    pub fn read_string(&self, loc: &Location) -> Result<String, Failure> {
        let path = self.resolve_path(loc)?;
        std::fs::read_to_string(&path)
            .map_err(|e| Failure::Plain(format!("reading `{}`: {e}", terminal_location(loc))))
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
    eprint!("{}", interactive_trust_message(urls));
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

/// Render untrusted URL text without allowing it to control terminal layout.
/// URL identity and fetch semantics continue to use the original string.
fn terminal_url(url: &str) -> String {
    terminal_message(url)
}

/// Render a diagnostic message without allowing embedded source URLs or text
/// to disclose authority userinfo or control terminal layout. The inference
/// engine and module bundle retain their original strings.
pub fn terminal_message(message: &str) -> String {
    let display = redact_url_userinfo(message);
    // `escape_debug` also exposes bidi, zero-width, and other Unicode format
    // characters while preserving ordinary printable Unicode.
    crate::terminal_text(&display)
}

fn terminal_location(location: &Location) -> String {
    match location {
        Location::Local(path) => crate::terminal_path(path),
        Location::Remote(url) => terminal_url(url),
    }
}

/// A diagnostic label derived from a source location. Bundle and cache
/// identities continue to use the original location; only this displayed copy
/// drops remote authority userinfo. Control escaping remains centralized in
/// the diagnostic renderer's `terminal_path` call.
fn diagnostic_path(location: &Location) -> std::path::PathBuf {
    match location {
        Location::Local(path) => path.clone(),
        Location::Remote(url) => std::path::PathBuf::from(redact_url_userinfo(url).as_ref()),
    }
}

/// Redact authority userinfo in a display-only copy. The last literal `@`
/// before the path/query/fragment is the delimiter, so encoded delimiters stay
/// inside the redacted span and `@` elsewhere in the URL is not hidden.
fn redact_url_userinfo(text: &str) -> std::borrow::Cow<'_, str> {
    let mut result: Option<String> = None;
    let mut copied_through = 0;
    let mut search_from = 0;
    while let Some(relative_scheme_end) = text[search_from..].find("://") {
        let scheme_end = search_from + relative_scheme_end;
        let authority_start = scheme_end + 3;
        let authority_end = text[authority_start..]
            .find(|c: char| matches!(c, '/' | '?' | '#') || c.is_whitespace())
            .map_or(text.len(), |i| authority_start + i);
        let authority = &text[authority_start..authority_end];
        if let Some(at) = authority.rfind('@') {
            let host_start = authority_start + at + 1;
            let output = result.get_or_insert_with(|| String::with_capacity(text.len()));
            output.push_str(&text[copied_through..authority_start]);
            output.push_str("<redacted>@");
            copied_through = host_start;
        }
        if authority_end == text.len() {
            break;
        }
        search_from = authority_end;
    }
    let Some(mut result) = result else {
        return text.into();
    };
    result.push_str(&text[copied_through..]);
    result.into()
}

fn terminal_url_list(urls: &[String]) -> String {
    urls.iter()
        .map(|url| terminal_url(url))
        .collect::<Vec<_>>()
        .join("\n  ")
}

fn noninteractive_trust_message(urls: &[String]) -> String {
    format!(
        "refusing to fetch untrusted URL(s) — non-interactive; set FLATPPL_TRUST to \
         allow, or pre-trust them:\n  {}",
        terminal_url_list(urls)
    )
}

fn interactive_trust_message(urls: &[String]) -> String {
    format!(
        "flatppl: the following URL source(s) are not yet trusted:\n  {}\n\
         Fetch and trust them? [y/N]: ",
        terminal_url_list(urls)
    )
}

fn terminal_fileaccess_error(error: flatppl_fileaccess::Error) -> String {
    match error {
        flatppl_fileaccess::Error::NotFound(path) => {
            format!("file not found: {}", crate::terminal_path(&path))
        }
        flatppl_fileaccess::Error::Untrusted(url) => format!(
            "`{}` is not trusted — approve it (or set FLATPPL_TRUST) to fetch it",
            terminal_url(&url)
        ),
        flatppl_fileaccess::Error::Fetch { url, reason } => format!(
            "failed to fetch `{}`: {}",
            terminal_url(&url),
            terminal_url(&reason)
        ),
        other => other.to_string(),
    }
}

/// The string literal `source` of a `load_module`/`load_data` call: the first
/// positional argument, or a `source =` keyword argument (the form `load_data`
/// commonly uses). `None` for a non-literal source — it cannot be resolved
/// statically by the host.
fn source_of<'m>(module: &'m Module, call: &flatppl_core::Call) -> Option<&'m str> {
    if let Some(&arg0) = call.args.first()
        && let Node::Lit(Scalar::Str(s)) = module.node(arg0)
    {
        return Some(s);
    }
    for named in call.named.iter() {
        if module.resolve(named.name) == "source"
            && let Node::Lit(Scalar::Str(s)) = module.node(named.value)
        {
            return Some(s);
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
/// through `resolver` into a [`ModuleBundle`] keyed by resolved location (trust
/// batched per BFS level), and collects every `load_data` source location found
/// across the graph (root + dependencies), deduplicated. The data locations are
/// returned, not fetched here — the caller decides whether/when to resolve them
/// (see [`CliResolver::resolve_all`]).
///
/// Each dependency is recorded under its resolved location together with the
/// `(declaring file, directive string)` pair that reached it, so the engine
/// resolves a literal per importer. Two importer-local `common.flatppl` files
/// are therefore two distinct entries — spec §04 "Path resolution" resolves each
/// against its own declaring file — rather than a conflict to refuse.
#[cfg(feature = "infer")]
pub fn build_bundle(
    root: &Module,
    root_loc: &Location,
    resolver: &CliResolver,
) -> Result<(flatppl_infer::ModuleBundle, Vec<Location>), Failure> {
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut bundle = flatppl_infer::ModuleBundle::new();
    bundle.set_root(root_loc.display());
    // Parse each distinct file once, keyed by its resolved location.
    let mut parsed: HashMap<String, Arc<Module>> = HashMap::new();
    // Locations whose own dependencies have already been queued (cycle guard).
    let mut walked: HashSet<String> = HashSet::new();
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
    // Each entry carries the identity of the file that DECLARED the directive,
    // because that is what the directive resolves against.
    let mut level: Vec<(String, String, Location)> = directives_of(root, root_loc)
        .into_iter()
        .map(|(directive, loc)| (root_loc.display(), directive, loc))
        .collect();

    while !level.is_empty() {
        let locs: Vec<&Location> = level.iter().map(|(_, _, l)| l).collect();
        resolver.ensure_trusted(&locs)?;

        let mut next: Vec<(String, String, Location)> = Vec::new();
        for (importer, directive, loc) in &level {
            let ld = loc.display();
            let module = match parsed.get(&ld) {
                Some(m) => m.clone(),
                None => {
                    let source = resolver.read_string(loc)?;
                    let format = crate::Format::from_location(loc).map_err(Failure::Plain)?;
                    let module =
                        crate::read_module(format, &source).map_err(|(message, line, span)| {
                            Failure::Diagnostic {
                                path: diagnostic_path(loc),
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
            bundle.insert_resolved(
                importer.clone(),
                directive.clone(),
                ld.clone(),
                module.clone(),
            );

            // Queue this file's own dependencies (and collect its data
            // sources) exactly once.
            if walked.insert(ld.clone()) {
                collect_data(&module, loc);
                next.extend(
                    directives_of(&module, loc)
                        .into_iter()
                        .map(|(d, l)| (ld.clone(), d, l)),
                );
            }
        }
        level = next;
    }
    Ok((bundle, data))
}

/// Fetch a model's transitive dependencies into the cache (the `flatppl prepare`
/// command). BFS over each input file's `load_module` graph — reading + parsing
/// each module to discover its deps, fetching remote ones — then resolve every
/// `load_data` source. Local files and local deps need no fetch. Trust is
/// batched per BFS level. The `resolver`'s `update` flag controls whether
/// already-cached URLs are re-fetched.
#[cfg(feature = "prepare")]
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
                    path: diagnostic_path(loc),
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

#[cfg(test)]
mod terminal_output_tests {
    use super::*;

    fn control_urls() -> Vec<String> {
        vec![
            "https://user:secret@example.test/\n\r\t\0\u{1b}\u{202e}\u{200b}\u{ad}/café"
                .to_string(),
            "https://user%3Aname:p%40ss@example.test/data?email=a%40b".to_string(),
        ]
    }

    const ESCAPED_URL: &str =
        "https://<redacted>@example.test/\\n\\r\\t\\0\\u{1b}\\u{202e}\\u{200b}\\u{ad}/café";
    const ENCODED_URL: &str = "https://<redacted>@example.test/data?email=a%40b";

    #[test]
    fn interactive_trust_output_escapes_url_controls() {
        assert_eq!(
            interactive_trust_message(&control_urls()),
            format!(
                "flatppl: the following URL source(s) are not yet trusted:\n  \
                 {ESCAPED_URL}\n  {ENCODED_URL}\nFetch and trust them? [y/N]: "
            )
        );
    }

    #[test]
    fn noninteractive_trust_output_escapes_url_controls() {
        assert_eq!(
            noninteractive_trust_message(&control_urls()),
            format!(
                "refusing to fetch untrusted URL(s) — non-interactive; set FLATPPL_TRUST to \
                 allow, or pre-trust them:\n  {ESCAPED_URL}\n  {ENCODED_URL}"
            )
        );
    }

    #[test]
    fn fileaccess_error_output_escapes_url_controls() {
        let error = flatppl_fileaccess::Error::Fetch {
            url: "https://user%3Aname:p%40ss@example.test/\n\u{1b}/café".to_string(),
            reason: "invalid\ruri".to_string(),
        };
        assert_eq!(
            terminal_fileaccess_error(error),
            "failed to fetch `https://<redacted>@example.test/\\n\\u{1b}/café`: invalid\\ruri"
        );
    }

    #[test]
    fn at_signs_outside_authority_userinfo_remain_visible() {
        let url = "https://example.test/path@name?email=user@example.test";
        assert_eq!(terminal_url(url), url);
    }

    #[test]
    fn diagnostic_messages_redact_each_url_and_escape_controls_once() {
        let message = "from https://safe.test/a to \
                       https://first:p%40ss@one.test/path@name and \
                       http://second:secret@two.test?q=a@b\nfailed";
        assert_eq!(
            terminal_message(message),
            "from https://safe.test/a to https://<redacted>@one.test/path@name and \
             http://<redacted>@two.test?q=a@b\\nfailed"
        );
    }
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

#[cfg(all(test, feature = "prepare"))]
mod fetch_tests {
    use super::*;

    struct InvalidRemote;

    impl Fetcher for InvalidRemote {
        fn fetch(&self, url: &str) -> Result<flatppl_fileaccess::Fetched, String> {
            Ok(flatppl_fileaccess::Fetched {
                bytes: b"@(".to_vec(),
                resolved_url: url.to_string(),
                content_type: Some("text/plain".to_string()),
                etag: None,
                last_modified: None,
            })
        }
    }

    #[test]
    fn remote_parse_diagnostic_redacts_url_userinfo() {
        let dir =
            std::env::temp_dir().join(format!("flatppl-remote-diagnostic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let url = "http://user:p%40ss@host.test/bad@name.flatppl?email=a@b";
        let model = dir.join("model.flatppl");
        std::fs::write(&model, format!("bad = load_module(\"{url}\")\n")).unwrap();
        let resolver = CliResolver {
            cache: Cache::new(dir.join("cache"), false, true),
            fetcher: Box::new(InvalidRemote),
            interactive: false,
            update: false,
            approved: RefCell::new(HashSet::new()),
        };

        let error = fetch_graph(&[Location::Local(model)], &resolver)
            .expect_err("the fetched invalid source must produce a parse diagnostic");
        let Failure::Diagnostic { path, .. } = error else {
            panic!("expected a parse diagnostic, got {error:?}");
        };
        assert_eq!(
            crate::terminal_path(&path),
            "http://<redacted>@host.test/bad@name.flatppl?email=a@b"
        );
        assert_eq!(
            Location::Remote(url.to_string()).display(),
            url,
            "semantic location identity remains unchanged"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

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
