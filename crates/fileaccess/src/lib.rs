//! `flatppl-fileaccess` — resolve a FlatPPL `source` (a local path or an
//! `http`/`https` URL) to a local file, the host-side counterpart of
//! `load_module` / `load_data`.
//!
//! It is a thin **file-access abstraction**: a [`Location`] (path or URL) with
//! relative-`join` resolution (spec §04 path resolution + the URL analogue), and
//! a [`Resolver`] that hands back a local file path — passing local paths through
//! and fetching+caching remote URLs via the shared on-disk cache (spec
//! §sec:url-cache). It does **no file-format decoding** (CSV/JSON/Arrow parsing
//! for `load_data` is a separate concern) and does not parse FlatPPL.
//!
//! The cache is network-free at its core: fetching goes through a [`Fetcher`]
//! and the trust decision through a [`TrustOracle`], so the whole mechanism is
//! unit-testable without a network. The real HTTP client lives behind the `net`
//! feature ([`HttpFetcher`]). This is a **native** host-layer library (fs +
//! optional network), not one of the wasm-targeted core libraries.

mod cache;
mod fetch;
mod location;
mod trust;

use std::path::PathBuf;

pub use cache::{Cache, Meta};
pub use fetch::{Fetched, Fetcher, OfflineFetcher};
pub use location::Location;
pub use trust::{ApproveAll, DenyAll, TrustOracle};

#[cfg(feature = "net")]
pub use fetch::HttpFetcher;

/// A source-resolution failure.
#[derive(Debug)]
pub enum Error {
    /// A local source does not exist.
    NotFound(PathBuf),
    /// A filesystem / cache I/O error.
    Io(std::io::Error),
    /// Offline mode (`FLATPPL_CACHE_OFFLINE`) and the URL is not cached.
    Offline(String),
    /// The URL has no trust marker and approval was refused (or non-interactive).
    Untrusted(String),
    /// The fetch failed — network error, non-`2xx` status, or bad redirect.
    Fetch { url: String, reason: String },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotFound(p) => write!(f, "file not found: {}", p.display()),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Offline(url) => write!(
                f,
                "offline (FLATPPL_CACHE_OFFLINE) and `{url}` is not in the cache"
            ),
            Error::Untrusted(url) => write!(
                f,
                "`{url}` is not trusted — approve it (or set FLATPPL_TRUST) to fetch it"
            ),
            Error::Fetch { url, reason } => write!(f, "failed to fetch `{url}`: {reason}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Resolves [`Location`]s to local files, bundling the [`Cache`] with the
/// [`Fetcher`] and [`TrustOracle`] a resolve needs.
pub struct Resolver<'a> {
    cache: Cache,
    fetcher: &'a dyn Fetcher,
    trust: &'a dyn TrustOracle,
}

impl<'a> Resolver<'a> {
    /// Bundle a cache with the fetch + trust policy to use for remote sources.
    pub fn new(cache: Cache, fetcher: &'a dyn Fetcher, trust: &'a dyn TrustOracle) -> Self {
        Resolver {
            cache,
            fetcher,
            trust,
        }
    }

    /// The underlying cache (for configuration / introspection).
    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    /// Resolve a location to a readable local file path: a local path is
    /// returned if it exists; a remote URL is fetched + cached (subject to
    /// offline / trust gating) and its cached object path returned.
    pub fn local_path(&self, loc: &Location) -> Result<PathBuf, Error> {
        match loc {
            Location::Local(p) => {
                if p.exists() {
                    Ok(p.clone())
                } else {
                    Err(Error::NotFound(p.clone()))
                }
            }
            Location::Remote(url) => self.cache.get(url, self.fetcher, self.trust),
        }
    }

    /// Resolve a location and read its bytes.
    pub fn read(&self, loc: &Location) -> Result<Vec<u8>, Error> {
        let path = self.local_path(loc)?;
        Ok(std::fs::read(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A network-free [`Fetcher`] that returns canned bytes and records the URLs
    /// it was asked to fetch (so tests can assert a cache hit did *not* refetch).
    struct FakeFetcher {
        body: Vec<u8>,
        calls: RefCell<Vec<String>>,
    }
    impl FakeFetcher {
        fn new(body: &str) -> Self {
            FakeFetcher {
                body: body.as_bytes().to_vec(),
                calls: RefCell::new(Vec::new()),
            }
        }
        fn call_count(&self) -> usize {
            self.calls.borrow().len()
        }
    }
    impl Fetcher for FakeFetcher {
        fn fetch(&self, url: &str) -> Result<Fetched, String> {
            self.calls.borrow_mut().push(url.to_string());
            Ok(Fetched {
                bytes: self.body.clone(),
                resolved_url: url.to_string(),
                content_type: Some("text/plain".to_string()),
                etag: Some("\"abc\"".to_string()),
                last_modified: None,
            })
        }
    }

    /// A fetcher that always fails (network error / non-2xx).
    struct FailFetcher;
    impl Fetcher for FailFetcher {
        fn fetch(&self, _url: &str) -> Result<Fetched, String> {
            Err("503 Service Unavailable".to_string())
        }
    }

    /// A unique temp cache root per test (no env, no network).
    fn temp_root() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "flatppl-fa-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn read(p: &Path) -> String {
        String::from_utf8(std::fs::read(p).unwrap()).unwrap()
    }

    const URL: &str = "https://example.com/models/m.flatppl";

    #[test]
    fn miss_fetches_stores_and_writes_metadata() {
        let root = temp_root();
        let cache = Cache::new(root.clone(), false, true); // trust_all
        let fetcher = FakeFetcher::new("x = 1");
        let path = cache.get(URL, &fetcher, &DenyAll).expect("resolves");

        assert_eq!(read(&path), "x = 1");
        assert_eq!(fetcher.call_count(), 1, "first resolve fetches once");
        assert!(path.starts_with(root.join("v1/objects")));

        // Metadata sits beside the object and is valid JSON with the URL.
        let meta_path = path.with_file_name(format!(
            "{}_meta.json",
            path.file_stem().unwrap().to_str().unwrap()
        ));
        let meta: Meta = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        assert_eq!(meta.url, URL);
        assert_eq!(meta.content_type.as_deref(), Some("text/plain"));
        assert!(
            meta.retrieved.ends_with('Z'),
            "ISO-8601 UTC: {}",
            meta.retrieved
        );
    }

    #[test]
    fn hit_does_not_refetch() {
        let root = temp_root();
        let cache = Cache::new(root, false, true);
        let fetcher = FakeFetcher::new("x = 1");
        let p1 = cache.get(URL, &fetcher, &DenyAll).unwrap();
        let p2 = cache.get(URL, &fetcher, &DenyAll).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(fetcher.call_count(), 1, "second resolve is a cache hit");
    }

    #[test]
    fn offline_miss_is_an_error_and_does_not_fetch() {
        let root = temp_root();
        let cache = Cache::new(root, true, true); // offline
        let fetcher = FakeFetcher::new("x = 1");
        let err = cache.get(URL, &fetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Offline(_)), "got {err:?}");
        assert_eq!(fetcher.call_count(), 0);
    }

    #[test]
    fn untrusted_is_refused_without_marker_and_does_not_fetch() {
        let root = temp_root();
        let cache = Cache::new(root, false, false); // not trust_all
        let fetcher = FakeFetcher::new("x = 1");
        let err = cache.get(URL, &fetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Untrusted(_)), "got {err:?}");
        assert_eq!(fetcher.call_count(), 0, "refused before any fetch");
    }

    #[test]
    fn approval_writes_marker_so_next_run_is_trusted() {
        let root = temp_root();
        let cache = Cache::new(root, false, false);
        let fetcher = FakeFetcher::new("x = 1");
        // First: approve once → fetch + marker written.
        cache.get(URL, &fetcher, &ApproveAll).unwrap();
        // Remove the object to force another fetch, but the trust marker remains.
        let k = super::cache::cache_key(URL);
        std::fs::remove_file(cache.object_path(&k)).unwrap();
        // Now DenyAll would refuse — but the marker makes it trusted.
        let p = cache
            .get(URL, &fetcher, &DenyAll)
            .expect("marker trusts it");
        assert_eq!(read(&p), "x = 1");
        assert_eq!(fetcher.call_count(), 2);
    }

    #[test]
    fn failed_fetch_writes_nothing() {
        let root = temp_root();
        let cache = Cache::new(root, false, true);
        let err = cache.get(URL, &FailFetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Fetch { .. }), "got {err:?}");
        let k = super::cache::cache_key(URL);
        assert!(!cache.object_path(&k).exists(), "no partial object");
        assert!(!cache.meta_path(&k).exists(), "no orphan metadata");
    }

    #[test]
    fn resolver_local_passthrough_and_notfound() {
        let root = temp_root();
        let file = root.join("model.flatppl");
        std::fs::write(&file, "y = 2").unwrap();
        let fetcher = FakeFetcher::new("");
        let r = Resolver::new(Cache::new(root.clone(), false, true), &fetcher, &DenyAll);

        let here = Location::Local(file.clone());
        assert_eq!(r.local_path(&here).unwrap(), file);
        assert_eq!(r.read(&here).unwrap(), b"y = 2");

        let missing = Location::Local(root.join("nope.flatppl"));
        assert!(matches!(r.local_path(&missing), Err(Error::NotFound(_))));
        assert_eq!(fetcher.call_count(), 0, "local paths never fetch");
    }
}
