//! Provenance header for generator output files.
//!
//! Every file the `flatppl` driver writes carries a leading comment block
//! recording how it was produced — when, from what source, by whom, on which
//! platform — so a generated artifact stays traceable to its origin. The block
//! uses the target format's line comment (`%` for FlatPPL, `;` for FlatPIR).
//!
//! The header embeds a wall-clock timestamp and the invoking user, so output is
//! not byte-reproducible by default. Pass `--no-header` to omit the block
//! entirely, or set `SOURCE_DATE_EPOCH` to pin the timestamp, when reproducible
//! output (golden diffs, deterministic builds) is required.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// One generation event, rendered into the leading comment block.
pub struct Provenance<'a> {
    /// Human description of the source format, e.g. `"FlatPPL"`, `"FlatPIR"`.
    pub converted_from: &'a str,
    /// The original input file (its name is recorded; the full path is the fallback).
    pub source: &'a Path,
    /// The subcommand that produced the output, e.g. `"convert"`, `"infer"`.
    pub generator: &'a str,
}

impl Provenance<'_> {
    /// Render the header as a `comment`-prefixed (`%` or `;`) block terminated by
    /// a blank line, suitable for prepending to the generated source.
    pub fn header(&self, comment: &str) -> String {
        let user = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "unknown".into());
        let host = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .ok();
        let source = self
            .source
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.source.display().to_string());

        let who = match host {
            Some(h) => format!("{user}@{h}"),
            None => user,
        };
        let lines = [
            "AUTOMATICALLY GENERATED — do not edit; regenerate from the source below.".to_string(),
            format!(
                "generator:  flatppl {} ({})",
                env!("CARGO_PKG_VERSION"),
                self.generator
            ),
            format!("generated:  {}", generated_at()),
            format!("from:       {} file `{source}`", self.converted_from),
            format!("by:         {who}"),
            format!(
                "platform:   {}/{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
            format!("command:    {}", invocation()),
        ];
        let mut out = String::new();
        for l in lines {
            out.push_str(comment);
            out.push(' ');
            out.push_str(&l);
            out.push('\n');
        }
        out.push('\n');
        out
    }
}

/// The full command that produced the output: the program name (basename of
/// `argv[0]`) followed by every argument verbatim — flags, values, and paths —
/// so the exact invocation is recorded. Arguments containing whitespace are
/// quoted so the line is a faithful, re-runnable record.
fn invocation() -> String {
    let mut args = std::env::args();
    let prog = args
        .next()
        .map(|a0| {
            Path::new(&a0)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or(a0)
        })
        .unwrap_or_else(|| "flatppl".into());
    let mut parts = vec![prog];
    for a in args {
        if a.is_empty() || a.contains(|c: char| c.is_whitespace() || c == '"') {
            parts.push(format!(
                "\"{}\"",
                a.replace('\\', "\\\\").replace('"', "\\\"")
            ));
        } else {
            parts.push(a);
        }
    }
    parts.join(" ")
}

/// ISO-8601 UTC timestamp (`YYYY-MM-DDThh:mm:ssZ`). Honors `SOURCE_DATE_EPOCH`
/// (UNIX seconds) for reproducible output; otherwise uses the wall clock.
fn generated_at() -> String {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });
    iso8601_utc(secs)
}

/// Format UNIX seconds as `YYYY-MM-DDThh:mm:ssZ` with no external dependency,
/// using Howard Hinnant's civil-from-days algorithm.
fn iso8601_utc(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_known_epochs() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        // 2021-01-01T00:00:00Z
        assert_eq!(iso8601_utc(1_609_459_200), "2021-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1_781_700_896), "2026-06-17T12:54:56Z");
    }

    #[test]
    fn header_uses_comment_prefix_and_fields() {
        // SAFETY: single-threaded test process; set a fixed epoch for determinism.
        unsafe {
            std::env::set_var("SOURCE_DATE_EPOCH", "0");
        }
        let p = Provenance {
            converted_from: "HS3 JSON",
            source: Path::new("/tmp/model.json"),
            generator: "convert --from hs3",
        };
        let h = p.header("%");
        assert!(h.lines().all(|l| l.is_empty() || l.starts_with('%')));
        assert!(h.contains("from:       HS3 JSON file `model.json`"));
        assert!(h.contains("generated:  1970-01-01T00:00:00Z"));
        assert!(h.contains("convert --from hs3"));
        // The full invocation is recorded (under the test harness, argv[0] is the
        // test binary — we only assert the line is present).
        assert!(h.contains("command:    "), "missing command line in:\n{h}");
        assert!(h.ends_with("\n\n"));
        unsafe {
            std::env::remove_var("SOURCE_DATE_EPOCH");
        }
    }
}
