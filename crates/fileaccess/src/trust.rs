//! The trust decision for fetching an untrusted URL (spec §sec:url-cache
//! "Trust"). The cache owns the *mechanism* (per-URL `trust/<kk>/<key>` markers,
//! the `FLATPPL_TRUST` blanket override); the oracle supplies the *policy* —
//! interactive tooling prompts and approves, non-interactive tooling denies.

/// Decides whether an untrusted URL may be fetched. Called only on a cache miss
/// for a URL that has no trust marker and when `FLATPPL_TRUST` is unset.
pub trait TrustOracle {
    /// `true` to approve fetching `url` (the cache then writes its trust marker),
    /// `false` to refuse (the resolve fails with an "untrusted" error).
    fn approve(&self, url: &str) -> bool;
}

/// Any `Fn(&str) -> bool` is a trust oracle — lets callers pass a closure
/// (e.g. a TTY prompt) without a newtype.
impl<F: Fn(&str) -> bool> TrustOracle for F {
    fn approve(&self, url: &str) -> bool {
        self(url)
    }
}

/// Refuse every untrusted URL — the correct default for non-interactive tooling
/// (CI, the language server): a not-yet-trusted URL is an error, never a silent
/// fetch.
pub struct DenyAll;

impl TrustOracle for DenyAll {
    fn approve(&self, _url: &str) -> bool {
        false
    }
}

/// Approve every URL — for tests and for tools that have already obtained
/// blanket consent through their own channel.
pub struct ApproveAll;

impl TrustOracle for ApproveAll {
    fn approve(&self, _url: &str) -> bool {
        true
    }
}
