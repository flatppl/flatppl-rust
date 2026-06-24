//! The shared on-disk cache for remote `source` content (spec §sec:url-cache):
//! content-addressed by URL hash, per-URL trust markers, atomic publishes, no
//! index. Network access is delegated to a [`Fetcher`]; the trust *policy* to a
//! [`TrustOracle`]. This module owns the layout, keys, and the resolve
//! algorithm.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Fetcher, TrustOracle};

/// The mandatory `<key>_meta.json` sidecar (spec §sec:url-cache "Layout and
/// keys"). Unknown fields are ignored on read; absent validators are `null`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Meta {
    /// The original request URL.
    pub url: String,
    /// The URL actually fetched after redirects.
    pub resolved_url: String,
    /// ISO 8601 UTC time the content was retrieved.
    pub retrieved: String,
    /// `Content-Type`, if the server provided one.
    pub content_type: Option<String>,
    /// `ETag` validator, if any (informational; the cache never revalidates).
    pub etag: Option<String>,
    /// `Last-Modified` validator, if any (informational).
    pub last_modified: Option<String>,
}

/// The cache key for a URL: `key` (hex SHA-256 of the URL sans `#`-fragment),
/// `kk` (its first two chars, the shard dir), and the optional trailing `ext`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub key: String,
    pub kk: String,
    pub ext: Option<String>,
}

/// Derive the cache key from a URL: SHA-256 of the URL with any `#`-fragment
/// removed (no other normalization), plus the full trailing extension of the
/// final path segment (everything after its first `.`; `None` if it has none).
pub(crate) fn cache_key(url: &str) -> Key {
    let no_frag = url.split('#').next().unwrap_or(url);
    let digest = Sha256::digest(no_frag.as_bytes());
    let key = hex(&digest);
    let kk = key[..2].to_string();
    Key {
        key,
        kk,
        ext: url_ext(no_frag),
    }
}

/// Lowercase-hex encode bytes.
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

/// The full trailing extension of a URL's final path segment — everything after
/// the first `.` (so `data.tar.gz` → `tar.gz`, `.flatppl` → `flatppl`); `None`
/// for a final segment with no `.` (or an empty extension). Query is excluded.
fn url_ext(url: &str) -> Option<String> {
    // Drop scheme://authority, then the query.
    let after_authority = match url.find("://") {
        Some(i) => {
            let rest = &url[i + 3..];
            match rest.find('/') {
                Some(slash) => &rest[slash..],
                None => "",
            }
        }
        None => url,
    };
    let path = after_authority
        .find('?')
        .map_or(after_authority, |i| &after_authority[..i]);
    let seg = path.rsplit('/').next().unwrap_or("");
    match seg.split_once('.') {
        Some((_, rest)) if !rest.is_empty() => Some(rest.to_string()),
        _ => None,
    }
}

/// The shared remote-content cache rooted at `<flatppl-cachedir>` (the `/v1/`
/// layout lives under it).
pub struct Cache {
    root: PathBuf,
    offline: bool,
    trust_all: bool,
}

impl Cache {
    /// Build from the environment (spec §sec:url-cache "Environment variables"):
    /// `FLATPPL_CACHEDIR` (verbatim) or the platform default cache dir +
    /// `flatppl`; `FLATPPL_CACHE_OFFLINE` and `FLATPPL_TRUST` enabled by presence.
    pub fn from_env() -> Cache {
        let root = match std::env::var_os("FLATPPL_CACHEDIR") {
            Some(d) => PathBuf::from(d),
            None => dirs::cache_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("flatppl"),
        };
        Cache {
            root,
            offline: std::env::var_os("FLATPPL_CACHE_OFFLINE").is_some(),
            trust_all: std::env::var_os("FLATPPL_TRUST").is_some(),
        }
    }

    /// Explicit construction (tests, embedders): `root` is `<flatppl-cachedir>`.
    pub fn new(root: PathBuf, offline: bool, trust_all: bool) -> Cache {
        Cache {
            root,
            offline,
            trust_all,
        }
    }

    /// The configured cache root (`<flatppl-cachedir>`).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Force the offline flag (e.g. a tool that is cache-only regardless of
    /// `FLATPPL_CACHE_OFFLINE`). When offline, a cache miss is an error and no
    /// fetch is attempted.
    pub fn set_offline(&mut self, offline: bool) {
        self.offline = offline;
    }

    /// Would resolving `url` prompt for trust? `true` only when a fetch would
    /// actually happen *and* needs approval: not offline, not already cached,
    /// not blanket-trusted (`FLATPPL_TRUST`), and no per-URL marker. Lets a
    /// caller batch the trust prompt for a set of URLs before resolving them.
    pub fn needs_approval(&self, url: &str) -> bool {
        if self.offline || self.trust_all {
            return false;
        }
        let k = cache_key(url);
        // A cached object is returned without fetching → no prompt.
        if self.object_path(&k).exists() {
            return false;
        }
        !self.trust_path(&k).exists()
    }

    /// Resolve `url` to a local cached object path, fetching it on a miss.
    ///
    /// A present object is authoritative and returned as-is — the cache never
    /// revalidates (URLs are treated as immutable). On a miss: error if offline;
    /// else gate on trust (marker present, `FLATPPL_TRUST`, or oracle approval,
    /// which writes the marker), fetch, and atomically publish `_meta.json` then
    /// the object. A failed fetch writes nothing.
    pub fn get(
        &self,
        url: &str,
        fetcher: &dyn Fetcher,
        trust: &dyn TrustOracle,
    ) -> Result<PathBuf, Error> {
        let k = cache_key(url);
        let object = self.object_path(&k);
        if object.exists() {
            return Ok(object);
        }
        self.fetch_and_store(url, &k, object, fetcher, trust)
    }

    /// Re-fetch `url` and overwrite any cached object (the `--update` path):
    /// ignores an existing object, but is otherwise identical to [`get`](Self::get)
    /// (offline → error; trust-gated; atomic publish).
    pub fn refetch(
        &self,
        url: &str,
        fetcher: &dyn Fetcher,
        trust: &dyn TrustOracle,
    ) -> Result<PathBuf, Error> {
        let k = cache_key(url);
        let object = self.object_path(&k);
        self.fetch_and_store(url, &k, object, fetcher, trust)
    }

    /// Trust-gate, fetch, and atomically publish `_meta.json` then the object.
    fn fetch_and_store(
        &self,
        url: &str,
        k: &Key,
        object: PathBuf,
        fetcher: &dyn Fetcher,
        trust: &dyn TrustOracle,
    ) -> Result<PathBuf, Error> {
        if self.offline {
            return Err(Error::Offline(url.to_string()));
        }

        // Trust gate (only reached on a miss/refresh that needs a fetch).
        let trusted = self.trust_all || self.trust_path(k).exists();
        if !trusted {
            if trust.approve(url) {
                self.create_trust_marker(k)?;
            } else {
                return Err(Error::Untrusted(url.to_string()));
            }
        }

        let fetched = fetcher.fetch(url).map_err(|reason| Error::Fetch {
            url: url.to_string(),
            reason,
        })?;

        let meta = Meta {
            url: url.to_string(),
            resolved_url: fetched.resolved_url,
            retrieved: now_iso8601(),
            content_type: fetched.content_type,
            etag: fetched.etag,
            last_modified: fetched.last_modified,
        };
        // Metadata first, then the object — so a present object always has meta.
        let meta_bytes = serde_json::to_vec_pretty(&meta).map_err(std::io::Error::other)?;
        self.write_atomic(&self.meta_path(k), &meta_bytes)?;
        self.write_atomic(&object, &fetched.bytes)?;
        Ok(object)
    }

    // ── paths ────────────────────────────────────────────────────────────────

    fn v1(&self) -> PathBuf {
        self.root.join("v1")
    }

    pub(crate) fn object_path(&self, k: &Key) -> PathBuf {
        let name = match &k.ext {
            Some(ext) => format!("{}.{}", k.key, ext),
            None => k.key.clone(),
        };
        self.v1().join("objects").join(&k.kk).join(name)
    }

    pub(crate) fn meta_path(&self, k: &Key) -> PathBuf {
        self.v1()
            .join("objects")
            .join(&k.kk)
            .join(format!("{}_meta.json", k.key))
    }

    pub(crate) fn trust_path(&self, k: &Key) -> PathBuf {
        self.v1().join("trust").join(&k.kk).join(&k.key)
    }

    fn tmp_dir(&self) -> PathBuf {
        self.v1().join("tmp")
    }

    // ── write helpers ─────────────────────────────────────────────────────────

    /// Write `bytes` to `dest` atomically: download to `tmp/`, fsync, rename.
    /// A rename that fails because `dest` already exists (Windows; or a
    /// concurrent tool that won the race) is treated as success — the cache is
    /// lock-free and content is addressed by URL hash.
    fn write_atomic(&self, dest: &Path, bytes: &[u8]) -> Result<(), Error> {
        let tmp_dir = self.tmp_dir();
        fs::create_dir_all(&tmp_dir)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = tmp_dir.join(tmp_name(dest));
        {
            let mut f = File::create(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        match fs::rename(&tmp, dest) {
            Ok(()) => Ok(()),
            Err(e) => {
                if dest.exists() {
                    let _ = fs::remove_file(&tmp);
                    Ok(())
                } else {
                    Err(Error::Io(e))
                }
            }
        }
    }

    /// Create the trust marker with an exclusive create. An existing marker
    /// (concurrent approval) is fine.
    fn create_trust_marker(&self, k: &Key) -> Result<(), Error> {
        let path = self.trust_path(k);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(Error::Io(e)),
        }
    }
}

/// A per-process-unique temp filename, so concurrent writers never collide in
/// `tmp/` before the atomic rename.
fn tmp_name(dest: &Path) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let base = dest
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("object");
    format!("{base}.{}.{n}.tmp", std::process::id())
}

/// Format the current time as ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`).
fn now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    iso8601_utc(secs)
}

/// Civil ISO 8601 UTC for a Unix timestamp (seconds), via Howard Hinnant's
/// days→civil algorithm. Valid for the proleptic Gregorian calendar.
fn iso8601_utc(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let (h, m, s) = (
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    );

    // days→(year, month, day), epoch shifted to 0000-03-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = y + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_sha256_of_url_without_fragment() {
        // Oracles computed independently (`printf … | sha256sum`).
        let k = cache_key("https://example.com/models/m.flatppl");
        assert_eq!(
            k.key,
            "85112c1cdb4c8595f8766b87aa90f366d7e660612d551f16b301f44c05fedf30"
        );
        assert_eq!(k.kk, "85");
        assert_eq!(k.ext.as_deref(), Some("flatppl"));

        // The fragment is stripped before hashing → same key as without it.
        let frag = cache_key("https://example.com/models/m.flatppl#section");
        assert_eq!(frag.key, k.key);
        assert_eq!(frag.ext.as_deref(), Some("flatppl"));
    }

    #[test]
    fn key_strips_query_for_ext_but_not_for_hash() {
        let k = cache_key("https://example.com/data/events.csv?v=2");
        assert_eq!(
            k.key,
            "e9a91ab1e90a8379fed116e3e458e48d7192e01575fd6fb3086eca2f6d0ada9a"
        );
        assert_eq!(k.ext.as_deref(), Some("csv"));
    }

    #[test]
    fn key_for_extensionless_final_segment_has_no_ext() {
        let k = cache_key("https://example.com/x");
        assert_eq!(
            k.key,
            "54cef8f42f3f31ad349075022cd36ce1a378d039c1df7af45d61d693d9a35c6a"
        );
        assert_eq!(k.ext, None);
    }

    #[test]
    fn multi_part_extension_is_kept_whole() {
        assert_eq!(
            cache_key("https://h/data.tar.gz").ext.as_deref(),
            Some("tar.gz")
        );
        // Leading-dot segment: everything after the first '.'.
        assert_eq!(
            cache_key("https://h/dir/.flatppl").ext.as_deref(),
            Some("flatppl")
        );
    }

    #[test]
    fn iso8601_matches_independent_oracle() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(iso8601_utc(1_719_263_400), "2024-06-24T21:10:00Z");
    }

    #[test]
    fn needs_approval_tracks_offline_trust_and_cache_state() {
        let url = "https://example.com/models/m.flatppl";
        // Fresh, online, not trusted → would prompt.
        let dir = std::env::temp_dir().join(format!("flatppl-na-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = Cache::new(dir.clone(), false, false);
        assert!(c.needs_approval(url));
        // Blanket trust or offline → never prompts.
        assert!(!Cache::new(dir.clone(), false, true).needs_approval(url));
        assert!(!Cache::new(dir.clone(), true, false).needs_approval(url));
        // A trust marker suppresses the prompt.
        let k = cache_key(url);
        std::fs::create_dir_all(c.trust_path(&k).parent().unwrap()).unwrap();
        std::fs::write(c.trust_path(&k), b"").unwrap();
        assert!(!c.needs_approval(url));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn object_path_layout_matches_spec() {
        let c = Cache::new(PathBuf::from("/cache"), false, false);
        let k = cache_key("https://example.com/models/m.flatppl");
        assert_eq!(
            c.object_path(&k),
            PathBuf::from(
                "/cache/v1/objects/85/85112c1cdb4c8595f8766b87aa90f366d7e660612d551f16b301f44c05fedf30.flatppl"
            )
        );
        assert_eq!(
            c.meta_path(&k),
            PathBuf::from(
                "/cache/v1/objects/85/85112c1cdb4c8595f8766b87aa90f366d7e660612d551f16b301f44c05fedf30_meta.json"
            )
        );
        assert_eq!(
            c.trust_path(&k),
            PathBuf::from(
                "/cache/v1/trust/85/85112c1cdb4c8595f8766b87aa90f366d7e660612d551f16b301f44c05fedf30"
            )
        );
    }
}
