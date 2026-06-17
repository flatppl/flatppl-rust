//! Generated-file banner.
//!
//! Files written by the `flatppl` driver carry a single leading comment line
//! marking them as machine-generated, so a hand-editor is warned off.
//!
//! It is deliberately minimal — just "do not edit". Earlier revisions embedded a
//! timestamp, the invoking user and host, the platform, and the full command
//! line; that is dropped on purpose. Such fields **leak personal and system
//! information** (usernames, hostnames, local file paths in the command) while
//! being only pseudo-provenance: not reliable, and gratuitously non-reproducible
//! (a regenerated file would differ for no semantic reason). The one fact worth
//! recording — the targeted FlatPPL language version — travels *in the model* as
//! the leading `flatppl_compat` binding ([`flatppl_core::FLATPPL_COMPAT`]), not in
//! this comment. `--no-header` omits the line entirely.

use crate::CommentStyle;

/// The banner text (no comment prefix, no trailing newline). Contains no `;`, so
/// it is safe as a single `#` line comment in FlatPPL (spec §05: a `#` comment
/// ends at the first `;`).
const BANNER: &str = "AUTOMATICALLY GENERATED — do not edit";

/// The leading generated-file banner for `style`, terminated by a blank line and
/// ready to prepend to output. Empty for [`CommentStyle::None`] (JSON has no
/// comment syntax).
pub fn banner(style: CommentStyle) -> String {
    match style {
        CommentStyle::None => String::new(),
        CommentStyle::Line(prefix) => format!("{prefix} {BANNER}\n\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_is_a_single_prefixed_line() {
        // FlatPPL: a `#` line comment (safe — the text carries no `;`).
        let h = banner(CommentStyle::Line("#"));
        assert_eq!(h, "# AUTOMATICALLY GENERATED — do not edit\n\n");

        // FlatPIR: per-line `;`.
        let h = banner(CommentStyle::Line(";"));
        assert!(h.starts_with("; AUTOMATICALLY GENERATED — do not edit"));

        // No personal/system fields ever leak into the banner.
        for leaked in ["generated:", "by:", "platform:", "command:", "from:"] {
            assert!(!h.contains(leaked), "banner must not contain `{leaked}`");
        }

        // JSON: no comment syntax → no banner.
        assert!(banner(CommentStyle::None).is_empty());
    }
}
