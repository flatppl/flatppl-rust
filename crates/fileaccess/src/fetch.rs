//! The URL-fetch seam. The cache calls a [`Fetcher`] on a miss; the trait keeps
//! the cache logic network-free (and unit-testable with a fake), and isolates
//! the one place an HTTP client + TLS are pulled in (the `net`-gated
//! [`HttpFetcher`]).

/// The result of fetching a URL: the body plus the response metadata the cache
/// records in `_meta.json` (spec §sec:url-cache "Layout and keys").
pub struct Fetched {
    /// The fetched bytes (the final response body, after any redirects).
    pub bytes: Vec<u8>,
    /// The URL actually fetched after following redirects (may equal the input).
    pub resolved_url: String,
    /// `Content-Type` header, if present.
    pub content_type: Option<String>,
    /// `ETag` validator, if present.
    pub etag: Option<String>,
    /// `Last-Modified` validator, if present.
    pub last_modified: Option<String>,
}

/// Fetches a URL's bytes, following redirects. Implementations must report a
/// failed fetch — a network error, a final non-`2xx` status, or an unresolvable
/// redirect — as `Err` (the cache then writes nothing; spec §sec:url-cache
/// "Resolve and fetch").
pub trait Fetcher {
    fn fetch(&self, url: &str) -> Result<Fetched, String>;
}

/// A fetcher that never reaches the network — for cache-only / offline
/// resolution. The cache invokes a fetcher only on a miss it intends to fetch,
/// so in offline mode this is never actually called; it satisfies the type
/// without linking an HTTP client (so a cache-only tool stays TLS-free).
pub struct OfflineFetcher;

impl Fetcher for OfflineFetcher {
    fn fetch(&self, url: &str) -> Result<Fetched, String> {
        Err(format!("offline: refusing to fetch `{url}`"))
    }
}

/// A blocking HTTP(S) fetcher backed by `ureq` (rustls TLS). Follows redirects
/// and treats a non-`2xx` final status as an error, per the spec.
#[cfg(feature = "net")]
pub struct HttpFetcher;

#[cfg(feature = "net")]
impl Fetcher for HttpFetcher {
    fn fetch(&self, url: &str) -> Result<Fetched, String> {
        use std::io::Read;
        // `ureq` follows redirects by default and returns `Err` for non-2xx.
        let resp = ureq::get(url).call().map_err(|e| e.to_string())?;
        let resolved_url = resp.get_url().to_string();
        let header = |name: &str| resp.header(name).map(str::to_string);
        let content_type = header("Content-Type");
        let etag = header("ETag");
        let last_modified = header("Last-Modified");
        let mut bytes = Vec::new();
        resp.into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Ok(Fetched {
            bytes,
            resolved_url,
            content_type,
            etag,
            last_modified,
        })
    }
}
