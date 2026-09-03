//! Char-boundary-safe string helpers.
//!
//! A byte count derived from `str::len`, `str::bytes` or a byte-wise scan is not
//! a valid `str` index: slicing with one panics (`byte index N is not a char
//! boundary`) whenever it lands inside a multi-byte character. FlatPPL names,
//! `source` strings and HS3 parameter names are all arbitrary UTF-8, so every
//! such site needs the boundary check. These helpers do the check once.

/// Does `s` begin with `prefix`, comparing ASCII case-insensitively?
///
/// `prefix` must be ASCII. The comparison runs on bytes, so it never slices `s`
/// and never panics on multi-byte input — the case that made
/// `is_http_url("αβγδ")` abort during scheme detection.
pub fn starts_with_ascii_ignore_case(s: &str, prefix: &str) -> bool {
    debug_assert!(prefix.is_ascii(), "prefix must be ASCII");
    let s = s.as_bytes();
    let prefix = prefix.as_bytes();
    s.len() >= prefix.len()
        && s[..prefix.len()]
            .iter()
            .zip(prefix)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Byte length of the longest common prefix of `a` and `b` that ends on a
/// character boundary — a valid `str` index into either.
///
/// Comparing whole characters is what makes the result sliceable: two names
/// differing inside one character (`γ_0` / `δ_1`, which share only the lead
/// byte `CE`) have a common *character* prefix of length 0, not 1.
pub fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        len += ca.len_utf8();
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_prefix_matches_either_case() {
        assert!(starts_with_ascii_ignore_case("HTTP://x", "http://"));
        assert!(starts_with_ascii_ignore_case("https://x", "HTTPS://"));
        assert!(!starts_with_ascii_ignore_case("ftp://x", "http://"));
    }

    #[test]
    fn ascii_prefix_survives_multibyte_input() {
        // Both are 8 bytes, so a byte-slicing check indexed byte 7 — inside the
        // last character — and panicked.
        assert!(!starts_with_ascii_ignore_case("αβγδ", "http://"));
        assert!(!starts_with_ascii_ignore_case("αβγδ", "https://"));
        assert!(!starts_with_ascii_ignore_case("αβγ¤", "http://"));
        // A short multi-byte string must not read past its end either.
        assert!(!starts_with_ascii_ignore_case("α", "http://"));
        assert!(!starts_with_ascii_ignore_case("", "http://"));
    }

    #[test]
    fn common_prefix_is_a_valid_index() {
        for (a, b) in [
            ("gamma_stat_0", "gamma_stat_1"),
            ("γ_0", "δ_1"),
            ("γ_0", "γ_1"),
            ("μ_signal_0", "μ_signal_12"),
            ("", "x"),
            ("αβγ", "αβγ"),
        ] {
            let n = common_prefix_len(a, b);
            // The panic this replaces: slicing at a non-boundary byte index.
            let _ = &a[..n];
            let _ = &b[..n];
            assert_eq!(&a[..n], &b[..n]);
        }
    }

    #[test]
    fn common_prefix_stops_before_a_split_character() {
        // `γ` is CE B3 and `δ` is CE B4: one shared byte, no shared character.
        assert_eq!(common_prefix_len("γ_0", "δ_1"), 0);
        assert_eq!(common_prefix_len("γ_0", "γ_1"), 3);
        assert_eq!(common_prefix_len("gamma_stat_0", "gamma_stat_1"), 11);
        assert_eq!(common_prefix_len("αβγ", "αβγ"), 6);
    }
}
