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

mod location;

#[cfg(feature = "cache")]
mod cache;
#[cfg(feature = "cache")]
mod fetch;
#[cfg(feature = "cache")]
mod trust;

// `Location` (the spec §04 path/URL join) is the dependency-free core, always
// available. Everything below is the host-side pull cache, gated behind `cache`.
pub use location::Location;

#[cfg(feature = "cache")]
use std::path::PathBuf;

#[cfg(feature = "cache")]
pub use cache::{Cache, Meta};
#[cfg(feature = "cache")]
pub use fetch::{Fetched, Fetcher, OfflineFetcher};
#[cfg(feature = "cache")]
pub use trust::{ApproveAll, DenyAll, TrustOracle};

#[cfg(feature = "net")]
pub use fetch::HttpFetcher;

/// A source-resolution failure.
#[cfg(feature = "cache")]
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

#[cfg(feature = "cache")]
fn display_text(text: &str) -> String {
    text.chars().flat_map(char::escape_debug).collect()
}

#[cfg(feature = "cache")]
fn display_path(path: &std::path::Path) -> String {
    display_text(&path.to_string_lossy())
}

#[cfg(feature = "cache")]
fn redact_url_userinfo(url: &str) -> std::borrow::Cow<'_, str> {
    let mut result: Option<String> = None;
    let mut copied_through = 0;
    let mut search_from = 0;
    while let Some(relative_scheme_end) = url[search_from..].find("://") {
        let scheme_end = search_from + relative_scheme_end;
        let authority_start = scheme_end + 3;
        let authority_end = url[authority_start..]
            .find(|c: char| matches!(c, '/' | '?' | '#') || c.is_whitespace())
            .map_or(url.len(), |i| authority_start + i);
        let authority = &url[authority_start..authority_end];
        if let Some(at) = authority.rfind('@') {
            let host_start = authority_start + at + 1;
            let output = result.get_or_insert_with(|| String::with_capacity(url.len()));
            output.push_str(&url[copied_through..authority_start]);
            output.push_str("<redacted>@");
            copied_through = host_start;
        }
        if authority_end == url.len() {
            break;
        }
        search_from = authority_end;
    }
    let Some(mut result) = result else {
        return url.into();
    };
    result.push_str(&url[copied_through..]);
    result.into()
}

/// Redact authority userinfo in a display-only copy. The original URL remains
/// in the structured error for programmatic handling.
#[cfg(feature = "cache")]
fn display_url(url: &str) -> String {
    display_text(&redact_url_userinfo(url))
}

#[cfg(feature = "cache")]
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotFound(path) => f
                .debug_tuple("NotFound")
                .field(&path.to_string_lossy())
                .finish(),
            Error::Io(error) => {
                let message = error.to_string();
                f.debug_struct("Io")
                    .field("kind", &error.kind())
                    .field("message", &redact_url_userinfo(&message))
                    .finish()
            }
            Error::Offline(url) => f
                .debug_tuple("Offline")
                .field(&redact_url_userinfo(url))
                .finish(),
            Error::Untrusted(url) => f
                .debug_tuple("Untrusted")
                .field(&redact_url_userinfo(url))
                .finish(),
            Error::Fetch { url, reason } => f
                .debug_struct("Fetch")
                .field("url", &redact_url_userinfo(url))
                .field("reason", &redact_url_userinfo(reason))
                .finish(),
        }
    }
}

#[cfg(feature = "cache")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotFound(p) => write!(f, "file not found: {}", display_path(p)),
            Error::Io(e) => write!(f, "I/O error: {}", display_url(&e.to_string())),
            Error::Offline(url) => write!(
                f,
                "offline (FLATPPL_CACHE_OFFLINE) and `{}` is not in the cache",
                display_url(url)
            ),
            Error::Untrusted(url) => write!(
                f,
                "`{}` is not trusted — approve it (or set FLATPPL_TRUST) to fetch it",
                display_url(url)
            ),
            Error::Fetch { url, reason } => write!(
                f,
                "failed to fetch `{}`: {}",
                display_url(url),
                display_url(reason)
            ),
        }
    }
}

#[cfg(feature = "cache")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "cache")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Resolves [`Location`]s to local files, bundling the [`Cache`] with the
/// [`Fetcher`] and [`TrustOracle`] a resolve needs.
#[cfg(feature = "cache")]
pub struct Resolver<'a> {
    cache: Cache,
    fetcher: &'a dyn Fetcher,
    trust: &'a dyn TrustOracle,
}

#[cfg(feature = "cache")]
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
            Location::Local(p) => match std::fs::metadata(p) {
                Ok(metadata) if metadata.is_file() => Ok(p.clone()),
                Ok(_) => Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "not a regular file",
                ))),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    Err(Error::NotFound(p.clone()))
                }
                Err(error) => Err(Error::Io(error)),
            },
            Location::Remote(url) => self.cache.get(url, self.fetcher, self.trust),
        }
    }

    /// Resolve a location and read its bytes.
    pub fn read(&self, loc: &Location) -> Result<Vec<u8>, Error> {
        let path = self.local_path(loc)?;
        Ok(std::fs::read(path)?)
    }
}

#[cfg(all(test, feature = "cache"))]
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

    /// A private temp cache root per test (no env, no network), deleted when the
    /// test ends.
    ///
    /// The directory is claimed with `create_dir`, which fails when the name is
    /// already taken. The name embeds the pid, and the OS recycles pids, so
    /// without that exclusive claim a run could inherit the populated cache a
    /// previous run left at the same path — a hit where the test expects a miss.
    /// The `Drop` removal keeps such leftovers from accumulating at all.
    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> TempRoot {
            static N: AtomicU64 = AtomicU64::new(0);
            let base = std::env::temp_dir();
            loop {
                let dir = base.join(format!(
                    "flatppl-fa-test-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::Relaxed)
                ));
                match std::fs::create_dir(&dir) {
                    Ok(()) => return TempRoot(dir),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("temp cache root {}: {e}", dir.display()),
                }
            }
        }

        fn path(&self) -> PathBuf {
            self.0.clone()
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn read(p: &Path) -> String {
        String::from_utf8(std::fs::read(p).unwrap()).unwrap()
    }

    const URL: &str = "https://example.com/models/m.flatppl";

    #[test]
    fn error_display_escapes_local_paths_without_changing_the_path() {
        let path = PathBuf::from("missing\n\u{202e}\u{200b} café.flatppl");
        let error = Error::NotFound(path.clone());

        assert_eq!(
            error.to_string(),
            "file not found: missing\\n\\u{202e}\\u{200b} café.flatppl"
        );
        assert!(matches!(error, Error::NotFound(actual) if actual == path));

        let error = Error::Io(std::io::Error::other("failed\n\u{202e} café"));
        assert_eq!(error.to_string(), "I/O error: failed\\n\\u{202e} café");
        assert!(matches!(error, Error::Io(inner) if inner.to_string() == "failed\n\u{202e} café"));

        let error = Error::Io(std::io::Error::other(
            "failed at http://io-user:io-secret@host.test/a@b\n",
        ));
        assert_eq!(
            error.to_string(),
            "I/O error: failed at http://<redacted>@host.test/a@b\\n"
        );
        assert!(matches!(
            error,
            Error::Io(inner)
                if inner.to_string() == "failed at http://io-user:io-secret@host.test/a@b\n"
        ));
    }

    #[test]
    fn error_display_redacts_and_escapes_urls_without_changing_them() {
        let url = "http://user:p%40ss@host.test/a\n\u{202e} café.flatppl?keep=@value".to_string();
        let displayed = "http://<redacted>@host.test/a\\n\\u{202e} café.flatppl?keep=@value";
        let errors = [
            Error::Offline(url.clone()),
            Error::Untrusted(url.clone()),
            Error::Fetch {
                url: url.clone(),
                reason: "bad\nresponse at http://reason:secret@other.test/x".to_string(),
            },
        ];

        assert_eq!(
            errors[0].to_string(),
            format!("offline (FLATPPL_CACHE_OFFLINE) and `{displayed}` is not in the cache")
        );
        assert_eq!(
            errors[1].to_string(),
            format!("`{displayed}` is not trusted — approve it (or set FLATPPL_TRUST) to fetch it")
        );
        assert_eq!(
            errors[2].to_string(),
            format!(
                "failed to fetch `{displayed}`: bad\\nresponse at http://<redacted>@other.test/x"
            )
        );
        let no_userinfo = Error::Offline("https://host.test/a@b?keep=@value".to_string());
        assert_eq!(
            no_userinfo.to_string(),
            "offline (FLATPPL_CACHE_OFFLINE) and `https://host.test/a@b?keep=@value` is not in the cache"
        );
        for error in errors {
            match error {
                Error::Offline(actual) | Error::Untrusted(actual) => assert_eq!(actual, url),
                Error::Fetch {
                    url: actual,
                    reason,
                } => {
                    assert_eq!(actual, url);
                    assert_eq!(reason, "bad\nresponse at http://reason:secret@other.test/x");
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn error_debug_redacts_urls_without_changing_structured_fields() {
        let url = "http://user:p%40ss@host.test/a\n\u{202e} café.flatppl?keep=@value".to_string();
        let reason = "bad\nresponse at https://safe.test/x then http://reason:secret@other.test/x"
            .to_string();
        let errors = [
            Error::Offline(url.clone()),
            Error::Untrusted(url.clone()),
            Error::Fetch {
                url: url.clone(),
                reason: reason.clone(),
            },
        ];

        assert_eq!(
            format!("{:?}", errors[0]),
            "Offline(\"http://<redacted>@host.test/a\\n\\u{202e} café.flatppl?keep=@value\")"
        );
        assert_eq!(
            format!("{:?}", errors[1]),
            "Untrusted(\"http://<redacted>@host.test/a\\n\\u{202e} café.flatppl?keep=@value\")"
        );
        assert_eq!(
            format!("{:?}", errors[2]),
            "Fetch { url: \"http://<redacted>@host.test/a\\n\\u{202e} café.flatppl?keep=@value\", reason: \"bad\\nresponse at https://safe.test/x then http://<redacted>@other.test/x\" }"
        );
        for error in errors {
            match error {
                Error::Offline(actual) | Error::Untrusted(actual) => assert_eq!(actual, url),
                Error::Fetch {
                    url: actual,
                    reason: actual_reason,
                } => {
                    assert_eq!(actual, url);
                    assert_eq!(actual_reason, reason);
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }

        let io = Error::Io(std::io::Error::other(
            "failed\nhttp://io-user:io-secret@host.test/x",
        ));
        assert_eq!(
            format!("{io:?}"),
            "Io { kind: Other, message: \"failed\\nhttp://<redacted>@host.test/x\" }"
        );
        assert!(matches!(
            io,
            Error::Io(inner)
                if inner.to_string() == "failed\nhttp://io-user:io-secret@host.test/x"
        ));
    }

    #[test]
    fn miss_fetches_stores_and_writes_metadata() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true); // trust_all
        let fetcher = FakeFetcher::new("x = 1");
        let path = cache.get(URL, &fetcher, &DenyAll).expect("resolves");

        assert_eq!(read(&path), "x = 1");
        assert_eq!(fetcher.call_count(), 1, "first resolve fetches once");
        assert!(path.starts_with(root.path().join("v1/objects")));

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
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true);
        let fetcher = FakeFetcher::new("x = 1");
        let p1 = cache.get(URL, &fetcher, &DenyAll).unwrap();
        let p2 = cache.get(URL, &fetcher, &DenyAll).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(fetcher.call_count(), 1, "second resolve is a cache hit");
    }

    #[test]
    fn offline_miss_is_an_error_and_does_not_fetch() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), true, true); // offline
        let fetcher = FakeFetcher::new("x = 1");
        let err = cache.get(URL, &fetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Offline(_)), "got {err:?}");
        assert_eq!(fetcher.call_count(), 0);
    }

    #[test]
    fn untrusted_is_refused_without_marker_and_does_not_fetch() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, false); // not trust_all
        let fetcher = FakeFetcher::new("x = 1");
        let err = cache.get(URL, &fetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Untrusted(_)), "got {err:?}");
        assert_eq!(fetcher.call_count(), 0, "refused before any fetch");
    }

    #[test]
    fn approval_writes_marker_so_next_run_is_trusted() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, false);
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

    #[cfg(unix)]
    #[test]
    fn new_cache_layout_is_private_without_chmodding_an_existing_root() {
        use std::os::unix::fs::PermissionsExt;

        fn mode(path: &Path) -> u32 {
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777
        }

        fn assert_private_layout(root: &Path, cache: &Cache, object: &Path) {
            let k = super::cache::cache_key(URL);
            for dir in [
                root.to_path_buf(),
                root.join("v1"),
                root.join("v1/tmp"),
                root.join("v1/objects"),
                object.parent().unwrap().to_path_buf(),
                root.join("v1/trust"),
                cache.trust_path(&k).parent().unwrap().to_path_buf(),
            ] {
                assert_eq!(mode(&dir), 0o700, "private cache directory {dir:?}");
            }
            for file in [
                object.to_path_buf(),
                cache.meta_path(&k),
                cache.trust_path(&k),
            ] {
                assert_eq!(mode(&file), 0o600, "private cache file {file:?}");
            }
        }

        let parent = TempRoot::new();
        let new_root = parent.path().join("new-cache");
        let new_cache = Cache::new(new_root.clone(), false, false);
        let new_object = new_cache
            .get(URL, &FakeFetcher::new("private"), &ApproveAll)
            .unwrap();
        assert_private_layout(&new_root, &new_cache, &new_object);

        let existing_root = parent.path().join("existing-cache");
        std::fs::create_dir(&existing_root).unwrap();
        std::fs::set_permissions(&existing_root, std::fs::Permissions::from_mode(0o755)).unwrap();
        let existing_cache = Cache::new(existing_root.clone(), false, false);
        let existing_object = existing_cache
            .get(URL, &FakeFetcher::new("private"), &ApproveAll)
            .unwrap();
        assert_eq!(
            mode(&existing_root),
            0o755,
            "a pre-existing configured root is not silently chmodded"
        );
        for dir in [
            existing_root.join("v1"),
            existing_root.join("v1/tmp"),
            existing_root.join("v1/objects"),
            existing_object.parent().unwrap().to_path_buf(),
            existing_root.join("v1/trust"),
            existing_cache
                .trust_path(&super::cache::cache_key(URL))
                .parent()
                .unwrap()
                .to_path_buf(),
        ] {
            assert_eq!(mode(&dir), 0o700, "new descendant is private {dir:?}");
        }
        for file in [
            existing_object,
            existing_cache.meta_path(&super::cache::cache_key(URL)),
            existing_cache.trust_path(&super::cache::cache_key(URL)),
        ] {
            assert_eq!(mode(&file), 0o600, "new cache file is private {file:?}");
        }
    }

    #[test]
    fn failed_fetch_writes_nothing() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true);
        let err = cache.get(URL, &FailFetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Fetch { .. }), "got {err:?}");
        let k = super::cache::cache_key(URL);
        assert!(!cache.object_path(&k).exists(), "no partial object");
        assert!(!cache.meta_path(&k).exists(), "no orphan metadata");
    }

    #[test]
    fn temp_name_does_not_reduce_the_valid_object_filename_limit() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true);
        let suffix = "a".repeat(180);
        let url = format!("https://example.test/x.{suffix}");

        let object = cache
            .get(&url, &FakeFetcher::new("body"), &DenyAll)
            .expect("a valid final object component must be publishable");

        assert!(object.is_file());
        assert!(
            object
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(&format!(".{suffix}")),
            "the normative suffix remains complete: {object:?}"
        );
    }

    #[cfg(feature = "net")]
    fn one_http_response(response: &'static [u8]) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            stream.write_all(response).unwrap();
        });
        (format!("http://{addr}/model.flatppl"), server)
    }

    #[cfg(feature = "net")]
    #[test]
    fn final_304_is_not_published_as_an_empty_cache_object() {
        let (url, server) = one_http_response(
            b"HTTP/1.1 304 Not Modified\r\nETag: fixture\r\nConnection: close\r\n\r\n",
        );
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true);

        let err = cache
            .get(&url, &HttpFetcher, &DenyAll)
            .expect_err("a final 304 is not fetched content");
        server.join().unwrap();

        assert!(matches!(err, Error::Fetch { .. }), "got {err:?}");
        let k = super::cache::cache_key(&url);
        assert!(!cache.object_path(&k).exists(), "no empty cache object");
        assert!(!cache.meta_path(&k).exists(), "no metadata for a 304");
    }

    #[cfg(feature = "net")]
    #[test]
    fn final_101_is_not_fetch_success() {
        let (url, server) =
            one_http_response(b"HTTP/1.1 101 Switching Protocols\r\nConnection: close\r\n\r\n");

        let err = match HttpFetcher.fetch(&url) {
            Ok(_) => panic!("a protocol upgrade is not fetched content"),
            Err(err) => err,
        };
        server.join().unwrap();

        assert!(err.contains("101"), "got {err}");
    }

    #[test]
    fn refetch_reports_an_object_publish_failure() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true);
        let object = cache
            .get(URL, &FakeFetcher::new("old"), &DenyAll)
            .expect("initial publish");
        std::fs::remove_file(&object).unwrap();
        std::fs::create_dir(&object).unwrap();

        let err = cache
            .refetch(URL, &FakeFetcher::new("new"), &DenyAll)
            .expect_err("a directory cannot become a successful cache object");
        assert!(matches!(err, Error::Io(_)), "got {err:?}");
        assert!(
            object.is_dir(),
            "the failed publish must not report this usable"
        );
        assert_eq!(
            std::fs::read_dir(root.path().join("v1/tmp"))
                .unwrap()
                .count(),
            0,
            "a failed publish must remove its temporary content"
        );
        cache
            .get(URL, &FakeFetcher::new("newer"), &DenyAll)
            .expect_err("a directory must not become a cache hit");
    }

    #[cfg(unix)]
    #[test]
    fn cache_object_symlink_does_not_bypass_trust() {
        use std::os::unix::fs::symlink;

        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, false);
        let k = super::cache::cache_key(URL);
        let object = cache.object_path(&k);
        std::fs::create_dir_all(object.parent().unwrap()).unwrap();
        let planted = root.path().join("planted.flatppl");
        std::fs::write(&planted, "attacker bytes").unwrap();
        symlink(&planted, &object).unwrap();

        assert!(cache.needs_approval(URL), "a symlink is not a cache hit");
        let fetcher = FakeFetcher::new("network bytes");
        let err = cache.get(URL, &fetcher, &DenyAll).unwrap_err();
        assert!(matches!(err, Error::Untrusted(_)), "got {err:?}");
        assert_eq!(fetcher.call_count(), 0, "trust refusal precedes fetch");
        assert_eq!(read(&planted), "attacker bytes");
    }

    #[test]
    fn refetch_replaces_a_regular_cached_object() {
        let root = TempRoot::new();
        let cache = Cache::new(root.path(), false, true);
        let object = cache
            .get(URL, &FakeFetcher::new("old"), &DenyAll)
            .expect("initial publish");
        cache
            .refetch(URL, &FakeFetcher::new("new"), &DenyAll)
            .expect("regular refresh");
        assert_eq!(read(&object), "new");
    }

    #[test]
    fn resolver_local_passthrough_and_notfound() {
        let root = TempRoot::new();
        let file = root.path().join("model.flatppl");
        std::fs::write(&file, "y = 2").unwrap();
        let fetcher = FakeFetcher::new("");
        let r = Resolver::new(Cache::new(root.path(), false, true), &fetcher, &DenyAll);

        let here = Location::Local(file.clone());
        assert_eq!(r.local_path(&here).unwrap(), file);
        assert_eq!(r.read(&here).unwrap(), b"y = 2");

        let missing = Location::Local(root.path().join("nope.flatppl"));
        assert!(matches!(r.local_path(&missing), Err(Error::NotFound(_))));

        let directory = Location::Local(root.path());
        let error = r
            .local_path(&directory)
            .expect_err("a directory is not a source file");
        assert_eq!(error.to_string(), "I/O error: not a regular file");

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let link = root.path().join("linked.flatppl");
            symlink(&file, &link).unwrap();
            let linked = Location::Local(link.clone());
            assert_eq!(r.local_path(&linked).unwrap(), link);
            assert_eq!(r.read(&linked).unwrap(), b"y = 2");
        }
        assert_eq!(fetcher.call_count(), 0, "local paths never fetch");
    }
}
