//! A resolved FlatPPL `source` location — a local path or an `http`/`https` URL
//! — and relative-`join` resolution (spec §04 "Path resolution" for paths;
//! the analogous origin-relative join for URLs).

use std::path::{Path, PathBuf};

/// Where a `load_module` / `load_data` `source` points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Location {
    /// A local filesystem path.
    Local(PathBuf),
    /// An `http`/`https` URL (verbatim, including any query; the cache strips
    /// the `#`-fragment when keying).
    Remote(String),
}

/// Does `s` start with an `http://` / `https://` scheme (case-insensitive)?
fn is_http_url(s: &str) -> bool {
    let b = s.as_bytes();
    (b.len() >= 7 && s[..7].eq_ignore_ascii_case("http://"))
        || (b.len() >= 8 && s[..8].eq_ignore_ascii_case("https://"))
}

impl Location {
    /// Interpret a top-level `source` string: an `http`/`https` URL becomes
    /// [`Location::Remote`], anything else a [`Location::Local`] path.
    pub fn parse(source: &str) -> Location {
        if is_http_url(source) {
            Location::Remote(source.to_string())
        } else {
            Location::Local(PathBuf::from(source))
        }
    }

    /// Resolve `source` relative to `self` — the location of the file that
    /// *contains* the `load_module`/`load_data` call (spec §04: relative paths
    /// resolve against the directory of that file; `/` is the path separator;
    /// `..` is allowed; absolute paths are permitted). An absolute `http`/`https`
    /// URL in `source` is taken as-is regardless of the base.
    pub fn join(&self, source: &str) -> Location {
        if is_http_url(source) {
            return Location::Remote(source.to_string());
        }
        match self {
            Location::Local(base_file) => Location::Local(join_local(base_file, source)),
            Location::Remote(base_url) => Location::Remote(join_url(base_url, source)),
        }
    }

    /// A human-readable rendering for diagnostics.
    pub fn display(&self) -> String {
        match self {
            Location::Local(p) => p.display().to_string(),
            Location::Remote(u) => u.clone(),
        }
    }

    /// The final filename component — a local path's file name, or a URL's last
    /// path segment with any query/fragment stripped. Empty when there is none
    /// (e.g. a URL ending in `/`). Used to detect a format from the extension.
    pub fn name(&self) -> String {
        match self {
            Location::Local(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string(),
            Location::Remote(url) => {
                let no_qf = url.split(['?', '#']).next().unwrap_or(url);
                no_qf.rsplit('/').next().unwrap_or("").to_string()
            }
        }
    }

    /// Canonicalize this location for equality comparison. A local path has its
    /// `.`/`..` components resolved lexically (so two spellings of the same path
    /// compare equal); a remote URL is returned verbatim — URLs are already
    /// canonical as keyed (spec §sec:url-cache keys the request URL with no
    /// normalization, and a query string is significant). Pairs with [`join`],
    /// whose resolved result is already in this form.
    ///
    /// [`join`]: Location::join
    pub fn normalized(&self) -> Location {
        match self {
            Location::Local(p) => Location::Local(lexical_normalize(p)),
            Location::Remote(_) => self.clone(),
        }
    }
}

/// Join a relative (or absolute) path `source` against the directory of the
/// base *file* path, then lexically normalise `.`/`..`.
fn join_local(base_file: &Path, source: &str) -> PathBuf {
    let src = Path::new(source);
    let joined = if src.is_absolute() {
        src.to_path_buf()
    } else {
        match base_file.parent() {
            Some(dir) => dir.join(src),
            None => src.to_path_buf(),
        }
    };
    lexical_normalize(&joined)
}

/// Resolve `.`/`..`/`.` components without touching the filesystem. Leading
/// `..` on a relative path are preserved (can't pop above an unknown cwd).
fn lexical_normalize(p: &Path) -> PathBuf {
    use std::path::Component::*;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Prefix(_) | RootDir => out.push(c.as_os_str()),
            CurDir => {}
            ParentDir => {
                let pop = matches!(out.components().next_back(), Some(Normal(_)));
                if pop {
                    out.pop();
                } else {
                    out.push("..");
                }
            }
            Normal(s) => out.push(s),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// Join a relative URL reference against an absolute `http`/`https` base URL:
/// split the base into origin + path, resolve `source` against the base path's
/// directory (or as an absolute path when it starts with `/`), normalise
/// `.`/`..`, and reattach the origin. The base's query/fragment are dropped.
fn join_url(base_url: &str, source: &str) -> String {
    let (origin, base_path) = split_url(base_url);
    let combined = if source.starts_with('/') {
        source.to_string()
    } else {
        let dir = match base_path.rfind('/') {
            Some(i) => &base_path[..=i],
            None => "/",
        };
        format!("{dir}{source}")
    };
    format!("{origin}{}", normalize_url_path(&combined))
}

/// Split `scheme://authority/path?query#frag` into `("scheme://authority",
/// "/path")`. The query and fragment are discarded; a base with no path yields
/// path `"/"`.
fn split_url(url: &str) -> (String, String) {
    let Some(after_scheme_idx) = url.find("://") else {
        // Not a well-formed absolute URL — treat the whole thing as origin.
        return (url.to_string(), "/".to_string());
    };
    let authority_start = after_scheme_idx + 3;
    // Path begins at the first '/' after the authority.
    let rest = &url[authority_start..];
    match rest.find('/') {
        Some(slash) => {
            let origin = &url[..authority_start + slash];
            let path_qf = &url[authority_start + slash..];
            // Strip query / fragment.
            let path_end = path_qf.find(['?', '#']).unwrap_or(path_qf.len());
            (origin.to_string(), path_qf[..path_end].to_string())
        }
        None => {
            // No path: strip any query/fragment riding directly on the authority.
            let origin_end = rest
                .find(['?', '#'])
                .map_or(url.len(), |i| authority_start + i);
            (url[..origin_end].to_string(), "/".to_string())
        }
    }
}

/// Normalise an absolute URL path: drop empty (`//`) and `.` segments, pop on
/// `..`, and re-emit a leading-slash path.
fn normalize_url_path(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    format!("/{}", out.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_distinguishes_url_from_path() {
        assert_eq!(
            Location::parse("https://h.example/m.flatppl"),
            Location::Remote("https://h.example/m.flatppl".to_string())
        );
        assert_eq!(
            Location::parse("HTTP://h.example/m"),
            Location::Remote("HTTP://h.example/m".to_string())
        );
        assert_eq!(
            Location::parse("models/m.flatppl"),
            Location::Local(PathBuf::from("models/m.flatppl"))
        );
        assert_eq!(
            Location::parse("/abs/m.flatppl"),
            Location::Local(PathBuf::from("/abs/m.flatppl"))
        );
    }

    #[test]
    fn join_local_resolves_against_base_dir() {
        let base = Location::Local(PathBuf::from("/proj/dir/model.flatppl"));
        assert_eq!(
            base.join("helper.flatppl"),
            Location::Local(PathBuf::from("/proj/dir/helper.flatppl"))
        );
        assert_eq!(
            base.join("../lib/x.flatppl"),
            Location::Local(PathBuf::from("/proj/lib/x.flatppl"))
        );
        assert_eq!(
            base.join("/elsewhere/y.flatppl"),
            Location::Local(PathBuf::from("/elsewhere/y.flatppl"))
        );
    }

    #[test]
    fn join_url_resolves_against_base_url() {
        let base = Location::Remote("https://h.example/dir/model.flatppl".to_string());
        assert_eq!(
            base.join("helper.flatppl"),
            Location::Remote("https://h.example/dir/helper.flatppl".to_string())
        );
        assert_eq!(
            base.join("../other/x.flatppl"),
            Location::Remote("https://h.example/other/x.flatppl".to_string())
        );
        assert_eq!(
            base.join("/abs/y.flatppl"),
            Location::Remote("https://h.example/abs/y.flatppl".to_string())
        );
    }

    #[test]
    fn join_absolute_url_overrides_any_base() {
        let local = Location::Local(PathBuf::from("/proj/model.flatppl"));
        let remote = Location::Remote("https://h.example/dir/model.flatppl".to_string());
        let abs = "https://other.example/z.flatppl";
        assert_eq!(local.join(abs), Location::Remote(abs.to_string()));
        assert_eq!(remote.join(abs), Location::Remote(abs.to_string()));
    }

    #[test]
    fn name_is_final_segment_sans_query() {
        assert_eq!(
            Location::Local(PathBuf::from("/a/b/model.flatppl")).name(),
            "model.flatppl"
        );
        assert_eq!(
            Location::Remote("https://h/dir/events.csv?v=2".to_string()).name(),
            "events.csv"
        );
        assert_eq!(
            Location::Remote("https://h/m.flatpir#frag".to_string()).name(),
            "m.flatpir"
        );
    }

    #[test]
    fn join_url_with_port_keeps_origin() {
        let base = Location::Remote("https://h.example:8443/a/b/model.flatppl".to_string());
        assert_eq!(
            base.join("c.flatppl"),
            Location::Remote("https://h.example:8443/a/b/c.flatppl".to_string())
        );
    }

    #[test]
    fn normalized_canonicalizes_local_paths_for_comparison() {
        // A stored path carrying redundant `.`/`..` compares equal to its
        // canonical form (what `join` already produces on the resolved side).
        assert_eq!(
            Location::parse("/proj/sub/../sub/./x.flatppl").normalized(),
            Location::parse("/proj/sub/x.flatppl")
        );
        // Already-canonical paths are unchanged.
        assert_eq!(
            Location::parse("helpers.flatppl").normalized(),
            Location::Local(PathBuf::from("helpers.flatppl"))
        );
    }

    #[test]
    fn normalized_leaves_remote_urls_verbatim() {
        // URLs are canonical as keyed (spec §sec:url-cache keys verbatim, no
        // normalization), and a query string is significant — never stripped.
        let u = "https://h.example/a/b.flatppl?v=2";
        assert_eq!(
            Location::Remote(u.to_string()).normalized(),
            Location::Remote(u.to_string())
        );
    }
}
